"""Can SpikingJelly's fused CUDA kernel actually run here, and does it agree?

This is a GATE, not an experiment. It writes no results and trains nothing. It
answers the four questions that decide whether `src/multistep/` is worth building:

  1. is `cupy` importable and functional on this machine?
     SpikingJelly swallows a failed `import cupy` into `logging.info` and sets
     `neuron.cupy = None` (neuron.py:12-20), so a missing cupy does NOT raise at
     import -- it surfaces much later as an AttributeError on None, mid-training.
     Checking it up front is the whole reason this file exists.

  2. does the BPTT kernel COMPILE for our exact parameter combination?
     The kernel source is generated per (decay_input, hard_reset, detach_reset,
     surrogate, dtype). Ours is an unusual corner: detach_reset=True with ATan.
     Compilation happens lazily on the first training forward, so nothing but a
     real forward+backward can confirm it.

  3. do the three execution modes produce the SAME numbers?
        A  looped step_mode='s' + torch   <- what the pipeline does today
        B  step_mode='m' + torch          <- SpikingJelly's own Python loop
        C  step_mode='m' + cupy           <- the fused kernel
     Spikes must match EXACTLY. Gradients must match to float32 tolerance. If
     they do not, the fused kernel is not a drop-in and the port is off.

  4. is folding T into the batch dimension identical to looping, for the
     stateless layers? The whole design rests on this: Conv2d, MaxPool2d,
     Flatten and Linear are batch-independent, so `conv(x.flatten(0,1))` should
     equal `stack([conv(x[t]) for t in T])`. Verified against real cuDNN rather
     than assumed, because cuDNN picks its algorithm partly from the batch size.

and then reports what the speedup actually is, at the real training shape.

    python probe_spikingjelly_multistep.py --config config/config_ex2.yaml \
        --experiment ex2

Runs on CPU too, where the cupy checks report SKIP rather than FAIL -- useful for
checking this script's own logic before spending a Colab session on it.

Design and execution flow: local_docs/spikingjelly_multistep_intro.md
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Callable

import torch
import torch.nn as nn

from spikingjelly.activation_based import functional, neuron

from src.adapters.spikingjelly_lif import SURROGATES
from src.config import (
    load_config,
    require_bool,
    require_float,
    require_int,
    require_str,
    run_banner,
)

# Spikes are 0/1 and the recurrence is the same arithmetic in all three modes, so
# an exact match is the honest expectation -- not a tolerance.
SPIKE_TOLERANCE = 0.0

# Gradients go through different reduction orders (a Python loop of small kernels
# versus one fused kernel), so bit-equality is NOT expected. Agreement is accepted
# on EITHER bound, because the two kinds of gradient here live on wildly different
# scales and a single test cannot serve both:
#
#   input gradients   are O(1) per element      -> the absolute bound is the tight one
#   weight gradients  are SUMS over T*N*H*W     -> measured magnitude ~7.2e4 on the
#                                                  conv1 probe, so an absolute bound
#                                                  of 2e-5 is physically impossible
#                                                  and the relative one is the real test
#
# Measured for the conv1 weight gradient: 0.19 absolute, 2.6e-6 relative (~22 float32
# eps, the expected cost of reordering a ~144,000-term sum). Cross-checked against a
# float64 reference: BOTH orderings are correct to ~1e-6 relative, so neither is wrong.
GRAD_TOLERANCE = 2e-5
GRAD_TOLERANCE_REL = 1e-4

# Small shape for the numerical checks: big enough for a meaningful spike rate,
# small enough that a failure is inspectable.
CHECK_SHAPE = (8, 12, 10, 10)      # batch, channels, height, width

# The real conv1 output shape at T=20, batch=128 on N-MNIST -- 12 channels of
# 30x30. Timing at any smaller shape would understate the kernel's advantage,
# because the win comes from replacing T small launches with one large one.
TIMING_SHAPE = (128, 12, 30, 30)

# Input is scaled so that neurons genuinely fire. An all-silent tensor would make
# every comparison below pass trivially, which is the one way this probe could
# lie, so the spike rate is asserted to be inside a sane band.
INPUT_SCALE = 2.0
MIN_SPIKE_RATE_PCT = 1.0
MAX_SPIKE_RATE_PCT = 99.0


# ---------------------------------------------------------------------------
# Working around a stale numpy alias inside SpikingJelly
# ---------------------------------------------------------------------------


def patch_check_ctypes() -> str:
    """Replace `CKernel.check_ctypes`, which crashes on any numpy >= 1.24.

    `auto_cuda/base.py:249` reads `np.int`, an alias numpy REMOVED in 1.24. Since
    Colab runs Python 3.12 and the earliest numpy with 3.12 wheels is 1.26, there
    is no installable numpy on which that line works -- pinning numpy cannot fix
    this, which is why it is patched instead.

    The line sits in `check_ctypes`, a pure VALIDATION method: it asserts that an
    integer cupy array was declared as an `int` ctype and computes nothing. It is
    nevertheless called on every single kernel launch (`base.py:326`) with no flag
    to disable it, so it has to work.

    `np.int` was literally an alias for the builtin `int` -- numpy's own
    deprecation message states that substituting `int` "will not modify any
    behavior and is safe". So this reproduces the original method exactly, with
    that one substitution, and asserts no less than the original did.

    Returns a human-readable description of what was patched.
    """
    import numpy as np

    from spikingjelly.activation_based.auto_cuda import base as ac_base

    def check_ctypes(self, py_dict: dict) -> None:
        for key, value in py_dict.items():
            ctype: str = self.cparams[key]
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.float:
                    assert ac_base.startswiths(ctype, ("const float", "float"))
                elif value.dtype == torch.half:
                    assert ac_base.startswiths(ctype, ("const half2", "half2"))

            if ac_base.cupy is not None and isinstance(value, ac_base.cupy.ndarray):
                if value.dtype == np.float32:
                    assert ac_base.startswiths(ctype, ("const float", "float"))
                elif value.dtype == np.float16:
                    assert ac_base.startswiths(ctype, ("const half2", "half2"))
                elif value.dtype == int:  # was `np.int`, removed in numpy 1.24
                    assert ac_base.startswiths(ctype, ("const int", "int"))

    ac_base.CKernel.check_ctypes = check_ctypes
    return (
        "patched spikingjelly.activation_based.auto_cuda.base.CKernel.check_ctypes "
        "(np.int -> int; the alias was removed in numpy 1.24)"
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class Report:
    """Collects PASS/FAIL/SKIP so the exit code reflects the whole run."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.skips: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}" + (f"   {detail}" if detail else ""))
        if not ok:
            self.failures.append(label)
        return ok

    def check_agrees(self, a: torch.Tensor, b: torch.Tensor, label: str) -> bool:
        """Check two tensors agree to float32 precision. `a` is the reference."""
        ok, detail = agrees(a, b)
        return self.check(ok, label, detail)

    def skip(self, label: str, reason: str) -> None:
        print(f"  [SKIP] {label}   {reason}")
        self.skips.append(f"{label}: {reason}")

    def note(self, text: str) -> None:
        print(f"         {text}")


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


# ---------------------------------------------------------------------------
# Building the neuron under test
#
# Deliberately NOT via src.adapters.spikingjelly_lif.build_lif_node: that one
# refuses step_mode 'm' on purpose, and this probe's job is to test 'm'. The
# surrogate table IS imported from there, so there is only one mapping from a
# config string to a surrogate class.
# ---------------------------------------------------------------------------


def read_neuron_params(neuron_cfg: dict[str, Any]) -> dict[str, Any]:
    """Every SpikingJelly value the probe uses, read from the config once.

    Reading them from the config rather than hardcoding them is the point: this
    probe validates the neuron you are ACTUALLY going to train, so a later config
    edit cannot leave the probe verifying something else.

    `step_mode` and `backend` are deliberately absent -- the probe overrides both,
    since testing all three combinations is what it is for.
    """
    return {
        "tau": require_float(neuron_cfg, "spikingjelly.tau"),
        "decay_input": require_bool(neuron_cfg, "spikingjelly.decay_input"),
        "v_threshold": require_float(neuron_cfg, "spikingjelly.v_threshold"),
        "v_reset": require_float(neuron_cfg, "spikingjelly.v_reset"),
        "detach_reset": require_bool(neuron_cfg, "spikingjelly.detach_reset"),
        "surrogate_type": require_str(neuron_cfg, "spikingjelly.surrogate.type"),
        "surrogate_alpha": require_float(neuron_cfg, "spikingjelly.surrogate.alpha"),
    }


def make_node(params: dict[str, Any], step_mode: str, backend: str) -> neuron.LIFNode:
    return neuron.LIFNode(
        tau=params["tau"],
        decay_input=params["decay_input"],
        v_threshold=params["v_threshold"],
        v_reset=params["v_reset"],
        surrogate_function=SURROGATES[params["surrogate_type"]](
            alpha=params["surrogate_alpha"]
        ),
        detach_reset=params["detach_reset"],
        step_mode=step_mode,
        backend=backend,
    )


# ---------------------------------------------------------------------------
# The three execution modes, as interchangeable callables
# ---------------------------------------------------------------------------


def runner(
    params: dict[str, Any], step_mode: str, backend: str, device: torch.device
) -> Callable[[torch.Tensor], torch.Tensor]:
    """A fresh-state callable mapping [T, N, C, H, W] -> [T, N, C, H, W].

    `train()` is not cosmetic: LIFNode.multi_step_forward branches on
    `self.training` (neuron.py:930), and the cupy kernel is reached ONLY in
    training mode. In eval it uses a torch.jit path instead. Since this project
    measures training speed, training mode is the mode that matters -- but it
    also means a probe left in eval would silently never test cupy at all.
    """
    node = make_node(params, step_mode, backend).to(device)
    node.train()

    def run(x_seq: torch.Tensor) -> torch.Tensor:
        functional.reset_net(node)
        if step_mode == "m":
            return node(x_seq)
        # The current pipeline: one frame at a time, stacked back up afterwards
        # so the return shape is comparable.
        return torch.stack([node(x_seq[t]) for t in range(x_seq.shape[0])])

    run.node = node  # type: ignore[attr-defined]  # so callers can inspect kernels
    return run


def forward_backward(
    run: Callable[[torch.Tensor], torch.Tensor], base_x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spikes and d(sum of spikes)/dx, from a FRESH leaf tensor.

    A fresh leaf per call is required: reusing one would accumulate gradients
    across modes and make every comparison meaningless.
    """
    x = base_x.clone().requires_grad_(True)
    spikes = run(x)
    spikes.sum().backward()
    assert x.grad is not None
    return spikes.detach(), x.grad.detach().clone()


def time_forward_backward(
    run: Callable[[torch.Tensor], torch.Tensor],
    base_x: torch.Tensor,
    device: torch.device,
    reps: int,
    warmup: int,
) -> float:
    """Milliseconds per forward+backward, synchronised on both sides.

    The warmup is not padding: the first cupy call COMPILES the kernel, and the
    first cuDNN call picks an algorithm. Timing either would measure one-off
    setup rather than the steady state the training loop actually sees.
    """
    for _ in range(warmup):
        forward_backward(run, base_x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(reps):
        forward_backward(run, base_x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1000.0 * (time.perf_counter() - start) / reps


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max().item())


def max_rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest elementwise difference as a fraction of the reference value.

    `a` is the reference. The clamp only guards a division by zero; wherever the
    reference is genuinely ~0 the absolute bound is the meaningful one anyway, and
    `agrees()` accepts either.
    """
    reference = a.double().abs().clamp(min=1e-12)
    return float(((a.double() - b.double()).abs() / reference).max().item())


def agrees(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, str]:
    """Do two tensors agree to float32 precision, on either scale?

    Returns (verdict, printable detail). Accepting either bound is deliberate --
    see the tolerance constants for why one bound cannot serve both the O(1) input
    gradients and the ~1e5-magnitude weight-gradient accumulators.
    """
    absolute = max_abs_diff(a, b)
    relative = max_rel_diff(a, b)
    ok = absolute <= GRAD_TOLERANCE or relative <= GRAD_TOLERANCE_REL
    return ok, f"abs {absolute:.3g}, rel {relative:.3g} (tol {GRAD_TOLERANCE:g} / {GRAD_TOLERANCE_REL:g})"


# ---------------------------------------------------------------------------
# Check 1 -- environment
# ---------------------------------------------------------------------------


def check_environment(report: Report, device: torch.device) -> bool:
    """Is there a GPU, and is cupy genuinely usable on it?

    Returns whether the cupy checks can proceed at all.
    """
    section("1. Environment")

    print(f"         torch {torch.__version__}")
    try:
        from importlib.metadata import version

        print(f"         spikingjelly {version('spikingjelly')}")
    except Exception as error:  # noqa: BLE001 - metadata is a nicety, not a gate
        report.note(f"spikingjelly version unavailable ({error})")

    if device.type != "cuda":
        report.skip("CUDA available", "running on CPU; every cupy check below skips")
        return False

    report.check(True, "CUDA available", torch.cuda.get_device_name(device))

    # This is the check that matters. `neuron.cupy` is None whenever SpikingJelly's
    # own guarded import failed -- for a missing package, a CUDA/cupy version
    # mismatch, or a broken driver. All three look identical from the outside and
    # all three fail LATE, so they are caught here.
    if neuron.cupy is None:
        report.check(
            False,
            "spikingjelly sees cupy",
            "neuron.cupy is None -- its guarded import failed silently",
        )
        report.note("SpikingJelly hides the reason. Reproduce it directly with:")
        report.note("    python -c 'import cupy'")
        report.note("On Colab, `pip install cupy-cuda12x` matching the CUDA build.")
        return False

    report.check(True, "spikingjelly sees cupy", "neuron.cupy is not None")

    try:
        import cupy

        # Import alone is not proof: cupy can import and still fail to reach the
        # GPU. Force a real device allocation and a real kernel.
        value = int(cupy.arange(1024, dtype=cupy.float32).sum().get())
        report.check(
            value == 523776,
            "cupy runs a kernel on the GPU",
            f"cupy {cupy.__version__}, sum check = {value}",
        )
        report.check(
            cupy.cuda.runtime.getDeviceCount() > 0,
            "cupy sees at least one device",
            f"{cupy.cuda.runtime.getDeviceCount()} device(s)",
        )
    except Exception as error:  # noqa: BLE001 - this is exactly what we are testing
        report.check(False, "cupy runs a kernel on the GPU", f"{type(error).__name__}: {error}")
        return False

    return True


# ---------------------------------------------------------------------------
# Check 2 -- what this installed version claims to support
# ---------------------------------------------------------------------------


def check_supported_backends(report: Report, params: dict[str, Any]) -> None:
    """Confirm the 's' -> torch-only, 'm' -> torch+cupy rule on THIS version.

    The adapter's error messages assert this. Asserting it back against the
    installed package means those messages cannot quietly go stale.
    """
    section("2. Declared backend support (this installed version)")

    single = make_node(params, "s", "torch")
    multi = make_node(params, "m", "torch")

    report.check(
        single.supported_backends == ("torch",),
        "step_mode 's' supports only torch",
        f"{single.supported_backends}",
    )
    report.check(
        multi.supported_backends == ("torch", "cupy"),
        "step_mode 'm' supports torch and cupy",
        f"{multi.supported_backends}",
    )
    report.note("So the fused kernel is unreachable from 's'. Confirms the adapter guard.")


# ---------------------------------------------------------------------------
# Check 3 -- does the kernel compile for OUR parameters?
# ---------------------------------------------------------------------------


def check_kernel_compiles(
    report: Report, params: dict[str, Any], device: torch.device, time_steps: int
) -> bool:
    """Force a training forward+backward so the lazy kernel build actually runs.

    Our combination is a corner worth testing rather than assuming: detach_reset
    is True (which changes the generated backward source), the reset is hard, the
    input decays, and the surrogate is ATan. Each of those four is a separate
    branch in the generated CUDA.
    """
    section("3. Kernel compilation for the configured parameters")

    report.note(
        f"decay_input={params['decay_input']}  "
        f"hard_reset={params['v_reset'] is not None}  "
        f"detach_reset={params['detach_reset']}  "
        f"surrogate={params['surrogate_type']}(alpha={params['surrogate_alpha']})"
    )

    surrogate_instance = SURROGATES[params["surrogate_type"]](alpha=params["surrogate_alpha"])
    if not report.check(
        hasattr(surrogate_instance, "cuda_codes"),
        "surrogate exposes cuda_codes",
        f"{params['surrogate_type']} -- required to generate the backward kernel",
    ):
        return False

    run = runner(params, "m", "cupy", device)
    x = INPUT_SCALE * torch.rand(time_steps, *CHECK_SHAPE, device=device)

    try:
        spikes, grad = forward_backward(run, x)
    except Exception as error:  # noqa: BLE001 - the failure mode being probed
        report.check(False, "forward+backward with cupy", f"{type(error).__name__}: {error}")
        return False

    node = run.node  # type: ignore[attr-defined]
    report.check(node.forward_kernel is not None, "LIFNodeFPTTKernel built")
    report.check(node.backward_kernel is not None, "LIFNodeBPTTKernel built")
    report.check(
        torch.isfinite(grad).all().item() and grad.abs().sum().item() > 0,
        "gradient is finite and non-zero",
        f"|grad| max {grad.abs().max().item():.4g}",
    )

    rate = 100.0 * spikes.mean().item()
    report.check(
        MIN_SPIKE_RATE_PCT < rate < MAX_SPIKE_RATE_PCT,
        "neurons genuinely fire under the probe input",
        f"spike rate {rate:.1f}%",
    )
    report.note("If they did not, every comparison in check 4 would pass trivially.")
    return True


# ---------------------------------------------------------------------------
# Check 4 -- the correctness gate
# ---------------------------------------------------------------------------


def check_numerical_agreement(
    report: Report,
    params: dict[str, Any],
    device: torch.device,
    time_steps: int,
    cupy_ok: bool,
) -> None:
    """Compare looped 's', 'm'+torch and 'm'+cupy on identical input.

    A is the reference because it is what every result in the project so far was
    produced with. Anything that disagrees with A is a change in the science, not
    an optimisation.
    """
    section("4. Numerical agreement: looped 's' vs 'm'+torch vs 'm'+cupy")

    torch.manual_seed(0)
    x = INPUT_SCALE * torch.rand(time_steps, *CHECK_SHAPE, device=device)

    spikes_a, grad_a = forward_backward(runner(params, "s", "torch", device), x)
    report.note(
        f"reference A (looped 's'): {int(spikes_a.sum().item())} spikes, "
        f"rate {100.0 * spikes_a.mean().item():.1f}%"
    )

    spikes_b, grad_b = forward_backward(runner(params, "m", "torch", device), x)
    report.check(
        max_abs_diff(spikes_a, spikes_b) <= SPIKE_TOLERANCE,
        "B 'm'+torch    spikes identical to A",
        f"max diff {max_abs_diff(spikes_a, spikes_b):.3g}",
    )
    report.check_agrees(grad_a, grad_b, "B 'm'+torch    gradients match A")

    if not cupy_ok:
        report.skip("C 'm'+cupy vs A", "cupy unavailable")
        return

    spikes_c, grad_c = forward_backward(runner(params, "m", "cupy", device), x)
    spike_diff = max_abs_diff(spikes_a, spikes_c)

    report.check(
        spike_diff <= SPIKE_TOLERANCE,
        "C 'm'+cupy     spikes identical to A",
        f"max diff {spike_diff:.3g}",
    )
    report.check_agrees(grad_a, grad_c, "C 'm'+cupy     gradients match A")
    if spike_diff > SPIKE_TOLERANCE:
        report.note("Spikes MUST match exactly. A difference here means the fused")
        report.note("kernel is not computing our neuron -- do not port until resolved.")


# ---------------------------------------------------------------------------
# Check 5 -- the design assumption for every stateless layer
# ---------------------------------------------------------------------------


def check_fold_identity(
    report: Report, device: torch.device, time_steps: int, channels: int = 2
) -> None:
    """Is `conv(x.flatten(0,1))` the same as looping conv over T?

    The whole multi-step network design rests on yes. Conv2d is the hard case:
    it is the only layer here whose backend picks an algorithm based on the batch
    size, so folding T into the batch can legitimately select different cuDNN
    code. Worth measuring rather than reasoning about.
    """
    section("5. Folding T into the batch dimension, for stateless layers")

    torch.manual_seed(0)
    conv = nn.Conv2d(channels, 12, kernel_size=5).to(device)
    x = torch.rand(time_steps, 8, channels, 34, 34, device=device)

    looped_x = x.clone().requires_grad_(True)
    looped = torch.stack([conv(looped_x[t]) for t in range(time_steps)])
    conv.zero_grad(set_to_none=True)
    looped.sum().backward()
    looped_weight_grad = conv.weight.grad.detach().clone()

    folded_x = x.clone().requires_grad_(True)
    flat = conv(folded_x.flatten(0, 1))
    folded = flat.view(time_steps, x.shape[1], *flat.shape[1:])
    conv.zero_grad(set_to_none=True)
    folded.sum().backward()
    folded_weight_grad = conv.weight.grad.detach().clone()

    report.check(
        looped.shape == folded.shape, "output shapes match", f"{tuple(folded.shape)}"
    )
    report.check_agrees(looped.detach(), folded.detach(), "Conv2d output agrees")
    report.check_agrees(
        looped_weight_grad, folded_weight_grad, "Conv2d weight gradient agrees"
    )
    report.check_agrees(looped_x.grad, folded_x.grad, "Conv2d input gradient agrees")
    report.note(
        "The weight gradient is a sum over T*N*H*W, so it agrees RELATIVELY, not"
    )
    report.note(
        "absolutely -- reordering a ~1e5 accumulation costs a few float32 eps."
    )


# ---------------------------------------------------------------------------
# Check 6 -- is it worth it?
# ---------------------------------------------------------------------------


def check_speed(
    report: Report,
    params: dict[str, Any],
    device: torch.device,
    time_steps: int,
    cupy_ok: bool,
    reps: int,
    warmup: int,
) -> None:
    """Time all three modes at the real conv1 shape.

    Only the NEURON is timed, not a whole network, so the number here is an upper
    bound on the end-to-end gain: the conv and pooling layers are unchanged and
    will dilute it. It answers 'is the kernel meaningfully faster', which is the
    only question this probe needs to settle.
    """
    section("6. Speed at the real training shape")

    if device.type != "cuda":
        report.skip("timing", "meaningless on CPU")
        return

    batch, channels, height, width = TIMING_SHAPE
    elements = time_steps * batch * channels * height * width
    report.note(
        f"shape [T={time_steps}, N={batch}, {channels}, {height}, {width}]  "
        f"= {elements / 1e6:.1f}M elements, {reps} reps after {warmup} warmup"
    )

    torch.manual_seed(0)
    try:
        x = INPUT_SCALE * torch.rand(time_steps, *TIMING_SHAPE, device=device)
    except torch.cuda.OutOfMemoryError:
        report.skip("timing", "not enough GPU memory for the real shape")
        return

    modes = [("A  looped 's' + torch", "s", "torch"), ("B  'm' + torch", "m", "torch")]
    if cupy_ok:
        modes.append(("C  'm' + cupy", "m", "cupy"))

    timings: dict[str, float] = {}
    for label, step_mode, backend in modes:
        try:
            timings[label] = time_forward_backward(
                runner(params, step_mode, backend, device), x, device, reps, warmup
            )
        except torch.cuda.OutOfMemoryError:
            report.skip(f"timing {label}", "out of GPU memory")
            torch.cuda.empty_cache()

    if not timings:
        return

    baseline = timings.get("A  looped 's' + torch")
    print()
    for label, milliseconds in timings.items():
        speedup = f"{baseline / milliseconds:5.2f}x" if baseline else "    -"
        print(f"         {label:<24s} {milliseconds:8.2f} ms/iter   {speedup}")

    if cupy_ok and "C  'm' + cupy" in timings and baseline:
        gain = baseline / timings["C  'm' + cupy"]
        report.check(
            gain > 1.0,
            "cupy is faster than the current looped 's'",
            f"{gain:.2f}x on the neuron alone",
        )
        report.note("End-to-end gain will be lower: conv and pooling are unchanged.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate: can SpikingJelly's fused CUDA kernel run here, and does it agree?"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="required, no default: the neuron whose parameters get validated",
    )
    parser.add_argument("--experiment", help="label only; this probe writes nothing")
    parser.add_argument(
        "--patch-spikingjelly",
        action="store_true",
        help="work around the np.int crash in auto_cuda/base.py (see patch_check_ctypes)",
    )
    parser.add_argument("--reps", type=int, default=10, help="timed iterations (default 10)")
    parser.add_argument("--warmup", type=int, default=3, help="untimed iterations (default 3)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    neuron_cfg = config["neuron"]
    params = read_neuron_params(neuron_cfg)
    time_steps = require_int(config, "dataset.framing.n_time_bins")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        run_banner(
            "probe_spikingjelly_multistep.py -- can the fused CUDA kernel run, and does it agree?",
            experiment=args.experiment,
            config_path=args.config,
            config=config,
            framework="spikingjelly",
            extra={
                "device": device,
                "T": time_steps,
                "configured": (
                    f"step_mode={require_str(neuron_cfg, 'spikingjelly.step_mode')} "
                    f"backend={require_str(neuron_cfg, 'spikingjelly.backend')} "
                    "(OVERRIDDEN below -- all three modes are tested)"
                ),
            },
            writes_results=False,
        )
    )

    report = Report()

    if args.patch_spikingjelly:
        print()
        print(f"NOTE: {patch_check_ctypes()}")
        print("      Results below are therefore for a PATCHED spikingjelly.")

    cupy_ok = check_environment(report, device)
    check_supported_backends(report, params)
    if cupy_ok:
        cupy_ok = check_kernel_compiles(report, params, device, time_steps)
    else:
        section("3. Kernel compilation for the configured parameters")
        report.skip("kernel compilation", "cupy unavailable")
    check_numerical_agreement(report, params, device, time_steps, cupy_ok)
    check_fold_identity(report, device, time_steps)
    check_speed(report, params, device, time_steps, cupy_ok, args.reps, args.warmup)

    section("Verdict")
    if report.failures:
        print(f"  {len(report.failures)} FAILED:")
        for failure in report.failures:
            print(f"    - {failure}")
        print()
        print("  Do NOT build src/multistep/ until these are resolved: the port")
        print("  would change the numbers rather than only the speed.")
        return 1

    if report.skips:
        print(f"  no failures, but {len(report.skips)} check(s) skipped:")
        for skipped in report.skips:
            print(f"    - {skipped}")
        print()
        if device.type != "cuda":
            print("  This was a CPU run, so the cupy gate is still OPEN. Rerun on Colab.")
            return 0

    print("  All checks passed. The fused kernel runs here, agrees with the current")
    print("  pipeline, and folding T into the batch is exact. Safe to build")
    print("  src/multistep/ -- see local_docs/spikingjelly_multistep_intro.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
