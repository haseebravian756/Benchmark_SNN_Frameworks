"""SpikingJelly's LIF layer in MULTI-STEP mode -- the fused-kernel variant.

The difference from `src/adapters/spikingjelly_lif.py` is not the neuron, it is
*how many timesteps arrive per call*:

    single-step ('s')   forward(x)          x is [batch, C, H, W]      -- one frame
    multi-step  ('m')   forward_sequence(x) x is [T, batch, C, H, W]   -- all frames

Same arithmetic either way. Verified on a T4: `'m'`+cupy produces spikes
BIT-IDENTICAL to looped `'s'`, with gradients agreeing to 1.19e-07.

Why this file is separate rather than a flag on the existing adapter: the existing
one is the code that produced every ex1 and ex2 result. Leaving it untouched means
those results stay reproducible from the same commit, and no edit here can reach
them. See local_docs/spikingjelly_multistep_intro.md section 3.

Two backends are reachable from 'm', and BOTH are useful:

    backend='cupy'    one fused CUDA kernel over all T. ~5x on the neuron. The point.
    backend='torch'   SpikingJelly's own Python loop over T. Measured at 1.00x, i.e.
                      no faster than our own loop -- which makes it the CONTROL that
                      separates "sequence execution" from "CUDA fusion". It also
                      runs on CPU, so the network can be tested without a GPU.
"""

from __future__ import annotations

from typing import Any

import torch

from spikingjelly.activation_based import functional, neuron

from src.adapters.base import BaseLIF
from src.adapters.spikingjelly_lif import SURROGATES
from src.config import (
    ConfigError,
    require_bool,
    require_choice,
    require_float,
    require_str,
)
from src.multistep import sj_numpy_compat


def build_multistep_lif_node(neuron_cfg: dict[str, Any]) -> neuron.LIFNode:
    """Construct one multi-step LIFNode from the config's `spikingjelly` block.

    Deliberately NOT a call into `spikingjelly_lif.build_lif_node`: that function
    refuses step_mode 'm' on purpose, and refusing is the correct behaviour for the
    single-step pipeline. Duplicating the parameter reads is the price of leaving it
    untouched, and `describe()` below reports what was actually built so the two
    cannot silently diverge in a results file.
    """
    step_mode = require_choice(neuron_cfg, "spikingjelly.step_mode", ["s", "m"])
    if step_mode != "m":
        raise ConfigError(
            "build_multistep_lif_node requires neuron.spikingjelly.step_mode 'm'. "
            f"Got '{step_mode}'. For 's', the normal pipeline in "
            "src/adapters/spikingjelly_lif.py handles it -- nothing here is needed."
        )

    backend = require_choice(neuron_cfg, "spikingjelly.backend", ["torch", "cupy"])

    if backend == "cupy":
        # Fail HERE, at build time, rather than on the first backward pass. That
        # matters because SpikingJelly swallows a failed `import cupy` into
        # logging.info and sets neuron.cupy = None (neuron.py:12-20), so without
        # this check a broken cupy surfaces as an AttributeError on None partway
        # into training -- after the data has loaded and the clock has started.
        if not torch.cuda.is_available():
            raise ConfigError(
                "neuron.spikingjelly.backend 'cupy' needs a CUDA device, and "
                "torch.cuda.is_available() is False. On CPU use backend 'torch' "
                "with step_mode 'm': same numbers, no fused kernel."
            )
        if neuron.cupy is None:
            raise ConfigError(
                "neuron.spikingjelly.backend 'cupy' was requested, but SpikingJelly "
                "could not import cupy (neuron.cupy is None). It hides the reason, "
                "so reproduce it directly with:  python -c 'import cupy'\n"
                "Known cause: cupy 14+ requires numpy>=2, while tonic 1.6.0 "
                "requires numpy<2. Verified working combination: "
                "cupy-cuda12x 13.6.0 + numpy 1.26.4 + tonic 1.6.0."
            )
        # Without this, every kernel launch raises AttributeError on np.int.
        sj_numpy_compat.apply()

    surrogate_type = require_str(neuron_cfg, "spikingjelly.surrogate.type")
    if surrogate_type not in SURROGATES:
        raise ConfigError(
            f"neuron.spikingjelly.surrogate.type '{surrogate_type}' is not supported. "
            f"Supported: {sorted(SURROGATES)}"
        )
    surrogate_alpha = require_float(neuron_cfg, "spikingjelly.surrogate.alpha")

    node = neuron.LIFNode(
        tau=require_float(neuron_cfg, "spikingjelly.tau"),
        decay_input=require_bool(neuron_cfg, "spikingjelly.decay_input"),
        v_threshold=require_float(neuron_cfg, "spikingjelly.v_threshold"),
        v_reset=require_float(neuron_cfg, "spikingjelly.v_reset"),
        surrogate_function=SURROGATES[surrogate_type](alpha=surrogate_alpha),
        detach_reset=require_bool(neuron_cfg, "spikingjelly.detach_reset"),
        step_mode=step_mode,
        backend=backend,
        # Leave False: True retains a full [T, ...] voltage tensor per layer, for
        # nothing we measure.
        store_v_seq=False,
    )

    # Belt and braces. `backend` is a plain attribute with no validation against
    # step_mode, so an inconsistent pair would otherwise be discovered only by the
    # numbers coming out wrong.
    if backend not in node.supported_backends:
        raise ConfigError(
            f"backend '{backend}' is not in supported_backends "
            f"{node.supported_backends} for step_mode '{step_mode}'."
        )
    return node


class SpikingJellyMultiStepLIF(BaseLIF):
    """One LIF layer that consumes the whole [T, batch, ...] sequence at once.

    State handling is unchanged from the single-step adapter: SpikingJelly keeps
    the membrane on the module as `node.v`, so `reset()` is a library call and
    `has_state()` asks whether `v` has become a tensor yet.
    """

    def __init__(self, neuron_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.neuron_cfg = neuron_cfg
        self.node = build_multistep_lif_node(neuron_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refuses, on purpose. Use `forward_sequence`.

        A multi-step node handed a single [batch, C, H, W] frame reads the BATCH
        dimension as time and returns a correctly-shaped, meaningless answer --
        verified: a [1, 12, 30, 30] tensor comes back as [1, 12, 30, 30]. Nothing
        would crash and no number would look wrong. That silent failure is the
        single biggest hazard in this port, so the door is nailed shut instead.
        """
        raise RuntimeError(
            "SpikingJellyMultiStepLIF consumes the whole [T, batch, ...] sequence; "
            "it must be driven by SequenceSpikingNet via forward_sequence(), not "
            "called with one timestep. A multi-step node given a single frame reads "
            "the batch dimension as time and silently returns nonsense of the right "
            "shape, which is why this raises instead of guessing."
        )

    def forward_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Charge, fire and reset for ALL T timesteps. Returns spikes shaped like x."""
        spikes = self.node(x_seq)
        self._record_sequence(spikes)
        return spikes

    def _record_sequence(self, spikes: torch.Tensor) -> None:
        """Spike statistics for a [T, batch, ...] tensor.

        The base `_record` takes `shape[1:]` as one sample's shape, which is right
        when dim 0 is the batch. Here dim 0 is T and dim 1 is the batch, so the
        neuron count is `shape[2:]` -- otherwise `neurons()` comes out inflated by
        the batch size and corrupts layers.csv.

        `spike_slots` needs no adjustment and that is worth knowing: this adds
        T*N*neurons in ONE call where the stepwise path adds N*neurons T times.
        Same total, so `spike_rate_pct` and `spikes_per_neuron_per_inference` come
        out identical in both modes -- a free cross-check on the port.

        As in the base class, the sum stays a device tensor: calling .item() here
        would force a host/device sync per layer and wreck the timing measurements.
        """
        if not self.count_spikes:
            return
        if self.spike_shape is None:
            self.spike_shape = tuple(spikes.shape[2:])  # drop T and batch
        self.spike_total = self.spike_total + spikes.detach().sum()
        self.spike_slots += spikes.numel()

    def reset(self) -> None:
        """SpikingJelly's own reset, scoped to this node and nothing else."""
        functional.reset_net(self.node)

    def has_state(self) -> bool:
        # v starts life as the float 0.0 and becomes a tensor on first input.
        return isinstance(self.node.v, torch.Tensor)

    def describe(self) -> dict[str, Any]:
        return {
            "class": "spikingjelly.LIFNode",
            "tau": self.node.tau,
            "decay_input": self.node.decay_input,
            "v_threshold": self.node.v_threshold,
            "v_reset": self.node.v_reset,
            "detach_reset": self.node.detach_reset,
            "step_mode": self.node.step_mode,
            "backend": self.node.backend,
            "surrogate": (
                f"{require_str(self.neuron_cfg, 'spikingjelly.surrogate.type')}"
                f"(alpha={require_float(self.neuron_cfg, 'spikingjelly.surrogate.alpha')})"
            ),
            # Recorded so a results file states whether the library was patched,
            # rather than leaving it to the report text to remember.
            "sj_numpy_patch": sj_numpy_compat.is_applied(),
        }
