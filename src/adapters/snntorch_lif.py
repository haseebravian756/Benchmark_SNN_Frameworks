"""snnTorch implementation of the LIF layer.

snnTorch's neuron is `snn.Leaky`. Its update rule is the plainest of the three:

    mem = beta * mem + input          then fire if mem > threshold, then reset

`beta` IS the per-step decay factor, and the input gain is fixed at 1, so this
framework needs no translation arithmetic at all -- which is exactly why it is
the reference the other two are matched against.

Verified against snntorch 1.0.0.
"""

from __future__ import annotations

from typing import Any

import torch

import snntorch as snn
from snntorch import surrogate as snn_surrogate

from src.adapters.base import BaseLIF
from src.config import ConfigError, require_bool, require_float, require_str

# snnTorch spells its surrogate gradients as factory functions.
SURROGATES = {
    "atan": snn_surrogate.atan,
    "sigmoid": snn_surrogate.sigmoid,
    "fast_sigmoid": snn_surrogate.fast_sigmoid,
}


def build_leaky(neuron_cfg: dict[str, Any], threshold: float | None = None) -> snn.Leaky:
    """Construct one snn.Leaky from the config's `snntorch` block.

    Every snnTorch parameter is read here and nowhere else, so there is a single
    place to look when a value needs checking or changing.

    `threshold` overrides the configured one. Only the equivalence check uses
    that, to build a neuron that can never fire.
    """
    surrogate_type = require_str(neuron_cfg, "snntorch.surrogate.type")
    if surrogate_type not in SURROGATES:
        raise ConfigError(
            f"neuron.snntorch.surrogate.type '{surrogate_type}' is not supported. "
            f"Supported: {sorted(SURROGATES)}"
        )
    surrogate_alpha = require_float(neuron_cfg, "snntorch.surrogate.alpha")

    return snn.Leaky(
        beta=require_float(neuron_cfg, "snntorch.beta"),
        threshold=(
            require_float(neuron_cfg, "snntorch.threshold") if threshold is None
            else threshold
        ),
        # Default is "subtract" (soft reset), which would not match the other two.
        reset_mechanism=require_str(neuron_cfg, "snntorch.reset_mechanism"),
        # Default is True, which defers a spike's reset to the NEXT timestep.
        # SpikingJelly and Norse both reset immediately.
        reset_delay=require_bool(neuron_cfg, "snntorch.reset_delay"),
        # Backward pass only -- the forward dynamics are identical regardless.
        spike_grad=SURROGATES[surrogate_type](alpha=surrogate_alpha),
        # We manage the membrane ourselves rather than letting snnTorch hide it.
        # See the note on reset() below for why.
        init_hidden=False,
    )


class SnnTorchLIF(BaseLIF):
    """snnTorch state handling: the membrane is passed in and returned."""

    def __init__(self, neuron_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.neuron_cfg = neuron_cfg
        self.lif = build_leaky(neuron_cfg)
        # None means "fresh neuron". The real tensor is created on the first
        # step, because only then do we know the batch shape.
        self.membrane: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.membrane is None:
            self.membrane = torch.zeros_like(x)
        spikes, self.membrane = self.lif(x, self.membrane)
        self._record(spikes)
        return spikes

    def reset(self) -> None:
        """Drop the membrane.

        snnTorch's own documented approach is `utils.reset(net)`. We do not use
        it, and the reason is worth knowing: `utils.reset` works through CLASS-level
        methods (`snn.Leaky.reset_hidden()`), so it resets every snn.Leaky
        instance alive in the process, not just the ones in your network. Setting
        our own state to None is local, explicit, and cannot reach into anything
        else. It also detaches the computation graph implicitly, which is the
        other thing utils.reset does.
        """
        self.membrane = None

    def has_state(self) -> bool:
        return self.membrane is not None

    def describe(self) -> dict[str, Any]:
        return {
            "class": "snntorch.Leaky",
            "beta": self.lif.beta.item() if torch.is_tensor(self.lif.beta) else self.lif.beta,
            "threshold": (
                self.lif.threshold.item()
                if torch.is_tensor(self.lif.threshold) else self.lif.threshold
            ),
            "reset_mechanism": require_str(self.neuron_cfg, "snntorch.reset_mechanism"),
            "reset_delay": require_bool(self.neuron_cfg, "snntorch.reset_delay"),
            "surrogate": (
                f"{require_str(self.neuron_cfg, 'snntorch.surrogate.type')}"
                f"(alpha={require_float(self.neuron_cfg, 'snntorch.surrogate.alpha')})"
            ),
        }
