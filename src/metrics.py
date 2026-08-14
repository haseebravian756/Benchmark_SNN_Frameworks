"""Measurement primitives: time, memory, spiking activity, GPU energy.

Framework-agnostic -- nothing here knows which SNN library it is measuring. The
definitions and their justification are in docs/metrics_reference.md and
docs/metrics_plan.md; this module only implements them.

Three rules baked in, each of which would silently corrupt results if broken:

  * every timed region synchronises the GPU on both sides, or you time the CUDA
    queue instead of the work
  * spike counting is switched on only inside `spike_counting()`, because the
    counter costs a GPU reduction per layer per timestep
  * energy is measured for the whole training run only -- NVML's ~10 Hz sensor
    cannot resolve anything shorter
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import torch

from src.adapters.base import BaseLIF

# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def synchronize(device: torch.device) -> None:
    """Wait for all queued GPU work to actually finish. No-op on CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed(device: torch.device) -> Iterator[dict[str, float]]:
    """Time a region honestly.

        with timed(device) as t:
            ...work...
        t["seconds"]

    The synchronise before `t0` excludes work queued earlier; the one before
    `t1` waits for this region's work to complete. Without them, a GPU timing is
    meaningless -- CUDA calls return as soon as the work is *queued*.
    """
    result: dict[str, float] = {}
    synchronize(device)
    start = time.perf_counter()
    try:
        yield result
    finally:
        synchronize(device)
        result["seconds"] = time.perf_counter() - start


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> dict[str, float | None]:
    """Peak GPU memory since the last reset.

    `allocated` is what tensors were given -- the fairest measure of what the
    model needs. `reserved` is what PyTorch's caching allocator took from the
    driver, which is closer to what nvidia-smi shows; the gap between them
    reflects allocation behaviour, which is itself a framework property.
    """
    if device.type != "cuda":
        return {"allocated_mb": None, "reserved_mb": None}
    return {
        "allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


# ---------------------------------------------------------------------------
# Spiking activity
# ---------------------------------------------------------------------------


@contextmanager
def spike_counting(net) -> Iterator[None]:
    """Switch spike counting on for this block only, and clear it first.

    Counting adds a GPU reduction per layer per timestep. Leaving it on during a
    timed region would inflate exactly the numbers being measured, so it is
    scoped and always switched back off -- including if the block raises.
    """
    layers: list[BaseLIF] = net.lif_layers()
    for layer in layers:
        layer.reset_spike_stats()
        layer.count_spikes = True
    try:
        yield
    finally:
        for layer in layers:
            layer.count_spikes = False


def spike_rates(net) -> dict[str, Any]:
    """Read the accumulated counts, in BOTH standard units.

    `spike_rate_pct` -- share of (neuron x timestep) chances taken. Intuitive.
    `spikes_per_neuron_per_inference` -- the same thing x T. This is the unit the
        SNN literature usually reports (e.g. "0.4 spikes per neuron" for
        convolutional SNNs), so quoting it makes the result directly comparable
        to published figures without the reader converting anything.

    The overall figure is spike-WEIGHTED (total spikes / total opportunities),
    not the mean of the per-layer percentages -- averaging the percentages would
    give a 10-neuron output layer the same weight as a 10,800-neuron conv layer.
    """
    time_steps = getattr(net, "time_steps", None)
    per_layer = []
    total_spikes = 0.0
    total_slots = 0

    for index, layer in enumerate(net.lif_layers()):
        spikes = layer.spike_total
        if isinstance(spikes, torch.Tensor):
            spikes = spikes.item()
        rate = layer.spike_rate()  # fraction
        per_layer.append(
            {
                "layer_index": index,
                "layer_type": type(layer).__name__,
                "neurons": layer.neurons(),
                "total_spikes": float(spikes),
                "opportunities": layer.spike_slots,
                "spike_rate_pct": 100.0 * rate,
                # One inference = one forward pass over all T timesteps for one
                # sample, so a neuron gets T chances to fire per inference.
                "spikes_per_neuron_per_inference": (
                    rate * time_steps if time_steps else None
                ),
            }
        )
        total_spikes += float(spikes)
        total_slots += layer.spike_slots

    overall_fraction = total_spikes / total_slots if total_slots else 0.0
    return {
        "layers": per_layer,
        "spike_rate_pct": 100.0 * overall_fraction,
        "spikes_per_neuron_per_inference": (
            overall_fraction * time_steps if time_steps else None
        ),
        "total_spikes": total_spikes,
        "total_opportunities": total_slots,
    }


# ---------------------------------------------------------------------------
# Gradient magnitude
#
# Why this is measured at all: with ONE shared learning rate, a framework whose
# surrogate gradient is systematically larger takes systematically larger steps,
# so it is not being compared at the same effective learning rate as the others.
# Experiment 1 hit exactly this -- Norse's released SuperSpike ignores its alpha
# and produced a 6.03x conv1 gradient norm. That was measured by a throwaway
# script; this makes it a standing column of every run instead.
# ---------------------------------------------------------------------------


def gradient_norms(net) -> dict[str, Any]:
    """Read whatever gradients are currently sitting on the trainable parameters.

    Reads only. Nothing is zeroed here, so the caller stays in charge of when
    the gradients cease to exist.

    Three numbers per parameter tensor, because one is not enough:

      grad_norm     L2 norm ||g||. Grows with the element count, so it compares
                    honestly across runs for the SAME tensor and dishonestly
                    between tensors of different sizes.
      grad_rms      ||g|| / sqrt(n) -- per-element magnitude. THIS is the one to
                    compare conv1 against fc, or one framework against another.
      grad_max_abs  the largest single element, which is what actually explodes.

    `grad_norm_global` is sqrt(sum of squares over all trainable parameters) --
    exactly the quantity `clip_grad_norm_` would clip on, and deliberately not
    the mean of the per-tensor norms.
    """
    per_param: list[dict[str, Any]] = []
    total_square = 0.0

    for name, tensor in net.named_parameters():
        if not tensor.requires_grad or tensor.grad is None:
            continue
        grad = tensor.grad.detach()
        count = grad.numel()
        norm = grad.norm(2).item()
        total_square += norm * norm
        per_param.append(
            {
                "param_name": name,
                "param_count": count,
                "grad_norm": norm,
                "grad_rms": norm / math.sqrt(count) if count else 0.0,
                "grad_max_abs": grad.abs().max().item(),
            }
        )

    return {
        "params": per_param,
        "grad_norm_global": math.sqrt(total_square) if per_param else None,
    }


def gradient_probe(net, frames, labels, device, loss_fn) -> dict[str, Any]:
    """Gradient magnitudes on ONE fixed batch, leaving the run untouched.

    Forward and backward, and NO optimizer step -- so the weights, and with them
    the training trajectory, are exactly what they would have been without this
    call. Gradients are cleared before and after, so nothing leaks into the next
    training batch either.

    `frames`/`labels` are meant to be the SAME batch on every call. Then a change
    in these numbers between epochs is a change in the WEIGHTS, which is the
    behaviour being tracked, rather than a change in which digits turned up.

    The batch is expected on the CPU and copied here: holding a batch on the GPU
    for the whole run would raise `peak_memory_train_mb` by its own size and
    quietly change a reported metric.

    Cost is one batch's forward+backward per call. Against ~469 training batches
    per epoch that is ~0.2%, which is why it can sit inside the epoch loop -- but
    it is timed and reported (`probe_seconds`) rather than assumed negligible.
    """
    was_training = net.training
    net.train()
    net.zero_grad(set_to_none=True)

    synchronize(device)
    start = time.perf_counter()
    with torch.enable_grad():  # correct even if a caller wrapped us in no_grad
        loss = loss_fn(net(frames.to(device)), labels.to(device))
        loss.backward()
    result = gradient_norms(net)
    synchronize(device)
    result["probe_seconds"] = time.perf_counter() - start
    result["probe_loss"] = loss.item()

    net.zero_grad(set_to_none=True)
    if not was_training:
        net.eval()
    return result


# ---------------------------------------------------------------------------
# Latency -- one sample at a time (MLPerf "Single-Stream")
# ---------------------------------------------------------------------------


def measure_latency(
    net,
    samples: Sequence[torch.Tensor],
    device: torch.device,
    warmup: int = 5,
) -> dict[str, float | int]:
    """Time single-sample forward passes.

    `samples` must already be on `device`, each shaped [T, 1, C, H, W]. The
    host-to-device copy is deliberately excluded: this measures the network, and
    the copy is identical for all three frameworks anyway.

    Per-sample synchronisation IS the point here, unlike in bulk timing --
    latency is defined as when the output is genuinely ready.
    """
    if not samples:
        raise ValueError("measure_latency needs at least one sample")

    net.eval()
    with torch.no_grad():
        for _ in range(warmup):
            net(samples[0])
        synchronize(device)

        durations_ms: list[float] = []
        for sample in samples:
            synchronize(device)
            start = time.perf_counter()
            net(sample)
            synchronize(device)
            durations_ms.append(1000.0 * (time.perf_counter() - start))

    ordered = sorted(durations_ms)
    p90_index = min(len(ordered) - 1, int(round(0.90 * (len(ordered) - 1))))
    return {
        "latency_ms": statistics.median(durations_ms),      # headline
        "latency_mean_ms": statistics.fmean(durations_ms),
        "latency_p90_ms": ordered[p90_index],               # MLPerf convention
        "latency_min_ms": ordered[0],
        "latency_max_ms": ordered[-1],
        "latency_samples": len(durations_ms),
    }


# ---------------------------------------------------------------------------
# GPU power and energy (NVML)
#
# Training run only. NVML's power sensor refreshes at roughly 10 Hz and reports
# a time-AVERAGED value, so anything shorter than a few seconds cannot be
# measured honestly. Published comparisons against physical power meters show
# errors up to 73%, which is why these numbers support "framework A used more
# than framework B, measured identically here" and not "this costs X joules".
# ---------------------------------------------------------------------------


def nvml_handle(device_index: int = 0):
    """An NVML handle, or None if NVML or the power sensor is unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        pynvml.nvmlDeviceGetPowerUsage(handle)  # probe: not all GPUs support it
        return handle
    except Exception:  # noqa: BLE001 - absence is a normal outcome, not an error
        return None


def read_power_w(handle) -> float:
    """Instantaneous whole-GPU power in watts. NVML returns milliwatts."""
    import pynvml

    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0


def detect_update_interval_ms(handle, probe_seconds: float = 2.0) -> float | None:
    """How often does the power sensor actually produce a NEW value?

    Poll flat out and record when the reading CHANGES. The median gap between
    changes is the sensor's true resolution. Polling faster than this just
    re-reads the same number.

    Returns None if the reading never changed during the probe -- which happens
    on a perfectly idle GPU and means "could not determine", not "infinitely
    fast".
    """
    if handle is None:
        return None

    change_times: list[float] = []
    previous = read_power_w(handle)
    start = time.perf_counter()
    while time.perf_counter() - start < probe_seconds:
        current = read_power_w(handle)
        if current != previous:
            change_times.append(time.perf_counter())
            previous = current

    if len(change_times) < 2:
        return None
    gaps = [1000.0 * (b - a) for a, b in zip(change_times, change_times[1:])]
    return statistics.median(gaps)


class PowerSampler:
    """Polls GPU power in a background thread while the main thread works."""

    def __init__(self, handle, interval_s: float = 0.05) -> None:
        self.handle = handle
        self.interval_s = interval_s
        self.samples: list[tuple[float, float]] = []  # (perf_counter, watts)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return self.handle is not None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append((time.perf_counter(), read_power_w(self.handle)))
            except Exception:  # noqa: BLE001 - a failed read must not kill training
                break
            # Event.wait rather than sleep, so stopping is immediate.
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        if not self.available:
            return
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[tuple[float, float]]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5.0)
            self._thread = None
        return self.samples


def integrate_energy_j(samples: Sequence[tuple[float, float]]) -> float | None:
    """Trapezoidal integration of power over time -> joules.

    Uses the real elapsed time between consecutive samples rather than the
    nominal poll interval, so a delayed poll does not distort the total.
    """
    if len(samples) < 2:
        return None
    total = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        total += 0.5 * (p0 + p1) * (t1 - t0)
    return total


def summarise_power(samples: Sequence[tuple[float, float]]) -> dict[str, float | int | None]:
    """Mean/min/max of a power window. The spread is reported so that the
    uncertainty in an idle baseline is visible instead of hidden in one figure."""
    if not samples:
        return {"mean_w": None, "min_w": None, "max_w": None, "n": 0}
    watts = [w for _, w in samples]
    return {
        "mean_w": statistics.fmean(watts),
        "min_w": min(watts),
        "max_w": max(watts),
        "n": len(watts),
    }


def measure_idle_power(
    handle, seconds: float, interval_s: float = 0.05
) -> dict[str, float | int | None]:
    """Sample power with nothing running. Blocks for `seconds`."""
    if handle is None:
        return {"mean_w": None, "min_w": None, "max_w": None, "n": 0}
    sampler = PowerSampler(handle, interval_s)
    sampler.start()
    time.sleep(seconds)
    return summarise_power(sampler.stop())


def dynamic_energy_j(
    total_j: float | None, idle_w: float | None, duration_s: float
) -> float | None:
    """Energy attributable to the work, rather than to the machine being on.

    The idle subtraction is the ONLY assumption in the energy numbers, which is
    why the unsubtracted total is always reported alongside this. Can come out
    NEGATIVE if the baseline is wrong -- see energy_warnings().
    """
    if total_j is None or idle_w is None:
        return None
    return total_j - idle_w * duration_s


def energy_warnings(
    total_j: float | None,
    dynamic_j: float | None,
    idle: dict[str, float | int | None],
    load: dict[str, float | int | None],
    duration_s: float,
    update_interval_ms: float | None,
) -> list[str]:
    """Reasons not to trust an energy measurement.

    Every one of these has been seen in practice. A negative dynamic energy in
    particular is not a rounding artefact -- it means the GPU drew LESS power
    while "working" than while idle, i.e. the GPU was not what was working, the
    baseline was taken while something else used the GPU, or another process
    was competing. Silently writing such a number to a results file is worse
    than writing nothing.
    """
    problems: list[str] = []
    if total_j is None:
        problems.append("no power samples collected -- energy unmeasurable")
        return problems

    if dynamic_j is not None and dynamic_j < 0:
        problems.append(
            f"dynamic energy is NEGATIVE ({dynamic_j:.2f} J): the idle baseline "
            f"exceeds the power drawn under load. The GPU was probably not doing "
            f"the work, or the baseline was polluted by another process."
        )

    idle_mean, load_mean = idle.get("mean_w"), load.get("mean_w")
    if idle_mean is not None and load_mean is not None and load_mean <= idle_mean:
        problems.append(
            f"mean power under load ({load_mean:.2f} W) is not above idle "
            f"({idle_mean:.2f} W) -- the measured region did not load the GPU."
        )

    if idle_mean and idle.get("max_w") and idle.get("min_w") is not None:
        spread = idle["max_w"] - idle["min_w"]
        if spread > 0.5 * idle_mean:
            problems.append(
                f"idle baseline is unstable (range {idle['min_w']:.2f}-"
                f"{idle['max_w']:.2f} W around a mean of {idle_mean:.2f} W); "
                f"the subtraction carries that much uncertainty."
            )

    # NVML updates roughly every 100 ms, so a short region yields too few
    # genuinely distinct readings to integrate meaningfully.
    if update_interval_ms:
        expected = duration_s * 1000.0 / update_interval_ms
        if expected < 20:
            problems.append(
                f"region lasted {duration_s:.1f} s, which is only ~{expected:.0f} "
                f"sensor updates at {update_interval_ms:.0f} ms. Too short for a "
                f"trustworthy energy figure -- measure a longer region."
            )
    return problems
