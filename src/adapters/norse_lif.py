"""Norse implementation of the LIF layer.

Norse's neuron is `LIFBoxCell`. Read from the installed source, its update is

    dv = dt * tau_mem_inv * (input + v_leak - v)
    v  = v + dv  =  (1 - dt*tau_mem_inv) * v + (dt*tau_mem_inv) * input

so decay and input gain come from the SAME number, `dt * tau_mem_inv`. They
cannot be set independently: choosing decay 0.9 forces input gain 0.1. The
config's `input_scale` multiplies the input by 1/0.1 = 10 to bring the effective
gain back to 1, matching the other two frameworks.

Two things to know about Norse specifically:

  * Use `LIFBoxCell`, NOT `LIFCell`. The latter is second-order -- it carries an
    extra synaptic-current state -- and is a different neuron entirely.
  * Norse has no ATan surrogate (its options are heaviside, super, triangle,
    tanh, circ, heavi_erfc). SuperSpike is the closest available. This is a
    documented framework limitation, not a configuration mistake, and it is the
    one place where Norse genuinely cannot be matched to the other two.

Verified against norse 1.1.0.
"""

from __future__ import annotations

from typing import Any

import torch

from norse.torch.functional.lif_box import LIFBoxFeedForwardState, LIFBoxParameters
from norse.torch.functional.reset import reset_subtract, reset_value
from norse.torch.module.lif_box import LIFBoxCell

from src.adapters.base import BaseLIF
from src.config import ConfigError, require_choice, require_float, require_str

# Norse names its surrogate by a method string rather than an object or factory.
# Deliberately NOT including "heaviside": it has no usable gradient, so a network
# using it silently cannot learn.
SURROGATES = ["super", "triangle", "tanh", "circ", "heavi_erfc"]

# Hard reset ("value") jumps the membrane to v_reset; soft reset ("subtract")
# subtracts the threshold instead. The equivalent knob is `reset_mechanism` in
# snnTorch and `v_reset` in SpikingJelly.
RESET_METHODS = {"value": reset_value, "subtract": reset_subtract}


def build_lif_box_cell(
    neuron_cfg: dict[str, Any], v_th: float | None = None
) -> LIFBoxCell:
    """Construct one LIFBoxCell from the config's `norse` block.

    Every Norse parameter is read here and nowhere else.

    `v_th` overrides the configured threshold. Only the equivalence check uses
    that, to build a neuron that can never fire.
    """
    surrogate_type = require_str(neuron_cfg, "norse.surrogate.type")
    if surrogate_type not in SURROGATES:
        raise ConfigError(
            f"neuron.norse.surrogate.type '{surrogate_type}' is not supported. "
            f"Supported: {SURROGATES}. Note Norse has no ATan surrogate -- that "
            f"is a known framework limitation, not a config error."
        )

    reset_method = require_choice(neuron_cfg, "norse.reset_method", sorted(RESET_METHODS))

    parameters = LIFBoxParameters(
        tau_mem_inv=torch.as_tensor(require_float(neuron_cfg, "norse.tau_mem_inv")),
        v_leak=torch.as_tensor(require_float(neuron_cfg, "norse.v_leak")),
        v_th=torch.as_tensor(
            require_float(neuron_cfg, "norse.v_th") if v_th is None else v_th
        ),
        v_reset=torch.as_tensor(require_float(neuron_cfg, "norse.v_reset")),
        method=surrogate_type,
        alpha=torch.as_tensor(require_float(neuron_cfg, "norse.surrogate.alpha")),
        reset_method=RESET_METHODS[reset_method],
    )
    return LIFBoxCell(p=parameters, dt=require_float(neuron_cfg, "norse.dt"))


def input_scale(neuron_cfg: dict[str, Any]) -> float:
    """Factor applied to the input before it reaches the neuron.

    Compensates for Norse's gain being locked to dt*tau_mem_inv. Read here so
    that the training adapter and the equivalence check cannot disagree.
    """
    return require_float(neuron_cfg, "norse.input_scale")


class NorseLIF(BaseLIF):
    """Norse state handling: state goes in and out; None means a fresh neuron."""

    def __init__(self, neuron_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.neuron_cfg = neuron_cfg
        self.cell = build_lif_box_cell(neuron_cfg)
        self.input_scale = input_scale(neuron_cfg)
        self.state: LIFBoxFeedForwardState | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scaling here rather than in the preceding layer's weights keeps the
        # compensation visible and keeps the network framework-agnostic. The
        # chain rule carries the same factor into the backward pass, so the
        # effective gradient w.r.t. the input matches the other two frameworks.
        spikes, self.state = self.cell(x * self.input_scale, self.state)
        self._record(spikes)
        return spikes

    def reset(self) -> None:
        self.state = None

    def has_state(self) -> bool:
        return self.state is not None

    def describe(self) -> dict[str, Any]:
        parameters = self.cell.p
        step = self.cell.dt * float(parameters.tau_mem_inv)
        return {
            "class": "norse.LIFBoxCell",
            "dt": self.cell.dt,
            "tau_mem_inv": float(parameters.tau_mem_inv),
            "v_th": float(parameters.v_th),
            "v_reset": float(parameters.v_reset),
            "v_leak": float(parameters.v_leak),
            "input_scale": self.input_scale,
            "reset_method": require_str(self.neuron_cfg, "norse.reset_method"),
            "surrogate": f"{parameters.method}(alpha={float(parameters.alpha)})",
            # Derived, printed for reading only -- nothing consumes it.
            "effective_decay_gain": f"{1 - step:.3f}/{step * self.input_scale:.3f}",
        }
