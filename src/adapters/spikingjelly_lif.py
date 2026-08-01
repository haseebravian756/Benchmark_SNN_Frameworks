"""SpikingJelly implementation of the LIF layer.

SpikingJelly's neuron is `neuron.LIFNode`. With `decay_input=False` its update is

    v = v - v/tau + input      i.e.    v = (1 - 1/tau) * v + input

so the decay is `1 - 1/tau`, and `tau = 1 / (1 - decay)`. For decay 0.9, tau = 10.

The structural difference from snnTorch and Norse: **SpikingJelly keeps the
membrane on the module itself** (`node.v`) rather than handing it back to the
caller. So `forward` returns spikes alone, and resetting is a library call.

Verified against spikingjelly 0.0.0.0.14.
"""

from __future__ import annotations

from typing import Any

import torch

from spikingjelly.activation_based import functional, neuron
from spikingjelly.activation_based import surrogate as sj_surrogate

from src.adapters.base import BaseLIF
from src.config import (
    ConfigError,
    require_bool,
    require_choice,
    require_float,
    require_str,
)

# SpikingJelly spells its surrogate gradients as classes to instantiate.
SURROGATES = {
    "atan": sj_surrogate.ATan,
    "sigmoid": sj_surrogate.Sigmoid,
}


def build_lif_node(
    neuron_cfg: dict[str, Any], v_threshold: float | None = None
) -> neuron.LIFNode:
    """Construct one LIFNode from the config's `spikingjelly` block.

    Every SpikingJelly parameter is read here and nowhere else.

    `v_threshold` overrides the configured one. Only the equivalence check uses
    that, to build a neuron that can never fire.
    """
    surrogate_type = require_str(neuron_cfg, "spikingjelly.surrogate.type")
    if surrogate_type not in SURROGATES:
        raise ConfigError(
            f"neuron.spikingjelly.surrogate.type '{surrogate_type}' is not supported. "
            f"Supported: {sorted(SURROGATES)}"
        )
    surrogate_alpha = require_float(neuron_cfg, "spikingjelly.surrogate.alpha")

    # The shared network feeds one timestep at a time, so only 's' works here.
    # Fail loudly rather than silently producing nonsense shapes.
    step_mode = require_choice(neuron_cfg, "spikingjelly.step_mode", ["s", "m"])
    if step_mode != "s":
        raise ConfigError(
            "neuron.spikingjelly.step_mode must be 's' for this pipeline. The "
            "shared network in src/network.py loops over timesteps and hands "
            "each layer ONE timestep, which is what 's' means. Multi-step mode "
            "('m', required for the cupy backend) needs the whole [T, ...] "
            "tensor and a different network design."
        )

    backend = require_choice(neuron_cfg, "spikingjelly.backend", ["torch", "cupy"])
    if backend != "torch":
        raise ConfigError(
            f"neuron.spikingjelly.backend '{backend}' is not available with "
            f"step_mode 's' (verified: single-step supports only ('torch',)). "
            f"The cupy backend requires step_mode 'm'."
        )

    return neuron.LIFNode(
        tau=require_float(neuron_cfg, "spikingjelly.tau"),
        # Default is True, which divides the input by tau and makes the input
        # gain 1/tau instead of 1.
        decay_input=require_bool(neuron_cfg, "spikingjelly.decay_input"),
        v_threshold=(
            require_float(neuron_cfg, "spikingjelly.v_threshold")
            if v_threshold is None else v_threshold
        ),
        v_reset=require_float(neuron_cfg, "spikingjelly.v_reset"),
        # Default is Sigmoid(alpha=4.0), not ATan.
        surrogate_function=SURROGATES[surrogate_type](alpha=surrogate_alpha),
        # Whether gradient flows through the reset. Measurably NOT cosmetic:
        # on a 6-step test, flipping it changed d(spikes)/d(weight) from 2.64
        # to 4.34. False lets gradient through, matching snnTorch.
        detach_reset=require_bool(neuron_cfg, "spikingjelly.detach_reset"),
        step_mode=step_mode,
        backend=backend,
    )


class SpikingJellyLIF(BaseLIF):
    """SpikingJelly state handling: the membrane lives on the module as node.v."""

    def __init__(self, neuron_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.neuron_cfg = neuron_cfg
        self.node = build_lif_node(neuron_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spikes = self.node(x)
        self._record(spikes)
        return spikes

    def reset(self) -> None:
        """SpikingJelly's own reset call.

        Unlike snnTorch's `utils.reset`, this one is genuinely scoped: it walks
        the modules of whatever you hand it, so passing our single node resets
        that node and nothing else.
        """
        functional.reset_net(self.node)

    def has_state(self) -> bool:
        # v starts life as the float 0.0 and only becomes a tensor once the
        # first input arrives, so "is it a tensor yet" is the honest test.
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
        }
