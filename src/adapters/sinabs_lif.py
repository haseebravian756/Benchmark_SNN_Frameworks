"""sinabs implementation of the LIF layer.

sinabs' neuron is `sinabs.layers.LIF`. Its update, quoted from the class docstring:

    norm_input=True  :  v = alpha * v + (1 - alpha) * input
    norm_input=False :  v = alpha * v +               input      <- what ex1 uses

with `alpha = exp(-1/tau_mem)`. Two consequences worth stating up front:

  * `tau_mem` is a time constant measured in TIMESTEPS. sinabs has no `dt`, so
    tau_mem IS tau -- no unit conversion, unlike Norse's dt * tau_mem_inv.
  * `norm_input` is the input-GAIN switch:
        false -> gain 1.0,       decoupled from the decay   (like snnTorch)
        true  -> gain 1 - decay, locked to it               (like Norse)

That switch is why sinabs needs no `input_scale` compensation. Norse's gain is
structurally locked to its decay, which is what forced ex1's input_scale=10.0;
sinabs hands you the decoupled gain directly. For decay 0.9:

    tau_mem = -1 / ln(0.9) = 9.4912

THREE SINABS BEHAVIOURS THAT WOULD BITE IF UNHANDLED. All three are handled
below, and all three are silent rather than loud, which is why they are listed
here rather than left to be discovered:

  1. DIMENSION 1 IS TIME. `LIF.forward` opens with

         batch_size, time_steps, *trailing_dim = input_data.shape

     so handing it one timestep shaped (batch, C, H, W) makes it read the
     CHANNEL axis as time and (H, W) as the neuron shape. No error is raised --
     the numbers are simply wrong. `forward` below adds a length-1 time axis and
     removes it again.

  2. ITS DEFAULTS TARGET ANN->SNN CONVERSION, NOT THIS COMPARISON.
     `spike_fn=MultiSpike` lets a single neuron emit SEVERAL spikes in one
     timestep, and `reset_fn=MembraneSubtract()` is a soft reset. Neither is a
     mistake: no-leak + multi-spike + subtract-reset are exactly the three
     conditions under which a spiking neuron's firing rate equals ReLU, which is
     the ANN-conversion guarantee sinabs is built to provide. But none of the
     other three frameworks behaves that way, so ex1 overrides both. See
     local_docs/SNNs_Introduction_BaseConcepts.md section 4.4.

  3. NO ATan SURROGATE. Options are single_exponential, periodic_exponential,
     gaussian, multi_gaussian and heaviside. This is the same limitation already
     documented for Norse, with the same consequence: the FORWARD dynamics can be
     matched exactly, the backward pass cannot.

Written against the sinabs `develop` source, READ AND NOT EXECUTED -- sinabs is
not yet installed in this project. Every quoted line above is reproduced with its
source file in local_docs/Intro-to-Sinabs.md. Treat the first run as a
verification run, not a measurement run.
"""

from __future__ import annotations

from typing import Any

import torch

from sinabs.activation import (
    Gaussian,
    Heaviside,
    MembraneReset,
    MembraneSubtract,
    MultiGaussian,
    MultiSpike,
    PeriodicExponential,
    SingleExponential,
    SingleSpike,
)
from sinabs.layers import LIF

from src.adapters.base import BaseLIF
from src.config import (
    ConfigError,
    require_bool,
    require_choice,
    require_float,
    require_optional_float,
    require_str,
)

# How many spikes one neuron may emit in one timestep.
#
# These are passed as the CLASS, not an instance: sinabs calls `spike_fn.apply(...)`,
# so it wants the torch.autograd.Function itself. (MaxSpike is the exception -- it
# must be instantiated -- and is deliberately not wired up here, being a
# hardware-quantisation feature with no counterpart in the other frameworks.)
SPIKE_FNS = {
    "single": SingleSpike,  # (v - threshold >= 0)          -> 0 or 1
    "multi": MultiSpike,    # (v > 0) * trunc(v / threshold) -> 0, 1, 2, 3, ...
}

# sinabs' equivalent of snnTorch's `reset_mechanism`. Named to match it, since
# they mean exactly the same thing.
#   zero     - hard reset: membrane jumps to v_reset
#   subtract - soft reset: threshold is subtracted, remainder kept
RESET_MECHANISMS = ["zero", "subtract"]

# Surrogate gradients, with the config keys each one needs. sinabs does NOT use
# the name `alpha` for any of them, so this project's usual `surrogate.alpha` key
# does not apply here -- each type declares its own real parameter names instead.
# Pretending they were all `alpha` would misreport what was actually configured.
SURROGATE_PARAMS = {
    "single_exponential": ("grad_width", "grad_scale"),
    "periodic_exponential": ("grad_width", "grad_scale"),
    "gaussian": ("mu", "sigma", "grad_scale"),
    "multi_gaussian": ("mu", "sigma", "h", "s", "grad_scale"),
    "heaviside": ("window",),
}


def build_surrogate(neuron_cfg: dict[str, Any]) -> Any:
    """Construct the surrogate gradient object from the config's sinabs block.

    Only the keys belonging to the CHOSEN type are read, so a config carries the
    parameters of one surrogate rather than the union of all five.
    """
    kind = require_choice(
        neuron_cfg, "sinabs.surrogate.type", sorted(SURROGATE_PARAMS)
    )

    def param(name: str) -> float:
        return require_float(neuron_cfg, f"sinabs.surrogate.{name}")

    if kind == "single_exponential":
        return SingleExponential(
            grad_width=param("grad_width"), grad_scale=param("grad_scale")
        )
    if kind == "periodic_exponential":
        return PeriodicExponential(
            grad_width=param("grad_width"), grad_scale=param("grad_scale")
        )
    if kind == "gaussian":
        return Gaussian(
            mu=param("mu"), sigma=param("sigma"), grad_scale=param("grad_scale")
        )
    if kind == "multi_gaussian":
        return MultiGaussian(
            mu=param("mu"),
            sigma=param("sigma"),
            h=param("h"),
            s=param("s"),
            grad_scale=param("grad_scale"),
        )
    if kind == "heaviside":
        # A box gradient. Unlike Norse's "heaviside" -- which has NO gradient and
        # cannot learn -- sinabs' Heaviside returns a rectangular window, so it is
        # trainable, just crude.
        return Heaviside(window=param("window"))

    raise ConfigError(f"no surrogate wired up for '{kind}'")  # pragma: no cover


def build_lif(
    neuron_cfg: dict[str, Any], spike_threshold: float | None = None
) -> LIF:
    """Construct one sinabs LIF from the config's `sinabs` block.

    Every sinabs parameter is read here and nowhere else, so there is a single
    place to look when a value needs checking or changing.

    `spike_threshold` overrides the configured one. Only the equivalence check
    uses that, to build a neuron that can never fire.
    """
    reset_mechanism = require_choice(
        neuron_cfg, "sinabs.reset_mechanism", RESET_MECHANISMS
    )
    if reset_mechanism == "zero":
        reset_fn: Any = MembraneReset(
            reset_value=require_float(neuron_cfg, "sinabs.v_reset")
        )
    else:
        # No argument: MembraneSubtract's subtract_value defaults to None, which
        # its own code reads as "subtract the threshold".
        reset_fn = MembraneSubtract()

    spike_fn = SPIKE_FNS[
        require_choice(neuron_cfg, "sinabs.spike_fn", sorted(SPIKE_FNS))
    ]

    threshold = (
        require_float(neuron_cfg, "sinabs.spike_threshold")
        if spike_threshold is None
        else spike_threshold
    )

    return LIF(
        # In timesteps. alpha = exp(-1/tau_mem) is computed by sinabs.
        tau_mem=require_float(neuron_cfg, "sinabs.tau_mem"),
        # null = first-order neuron, one state, matching the other three.
        # A value would add a second synaptic state and a different neuron.
        tau_syn=require_optional_float(neuron_cfg, "sinabs.tau_syn"),
        spike_threshold=torch.as_tensor(threshold),
        # Default is MultiSpike, which is not a binary spike. See note 2 above.
        spike_fn=spike_fn,
        # Default is MembraneSubtract (soft). See note 2 above.
        reset_fn=reset_fn,
        surrogate_grad_fn=build_surrogate(neuron_cfg),
        # null = no lower clamp on the membrane, matching the other three.
        min_v_mem=require_optional_float(neuron_cfg, "sinabs.min_v_mem"),
        # Whether the LEARNABLE parameter is tau_mem or exp(-1/tau_mem).
        # Must stay false for this project: none of the other three frameworks
        # learns its time constant, so letting sinabs learn its own would compare
        # training recipes instead of frameworks.
        train_alphas=require_bool(neuron_cfg, "sinabs.train_alphas"),
        # Default is True, which multiplies the input by (1 - alpha) and changes
        # the input gain from 1.0 to 0.1 at decay 0.9. See the module docstring.
        norm_input=require_bool(neuron_cfg, "sinabs.norm_input"),
        # Debug recording of v_mem/i_syn per timestep. Costs memory and would
        # distort the wall-clock measurement, so it stays false.
        record_states=require_bool(neuron_cfg, "sinabs.record_states"),
    )


class SinabsLIF(BaseLIF):
    """sinabs state handling: the membrane lives on the module, in a buffer.

    Structurally this is the SpikingJelly case rather than the snnTorch/Norse
    case -- `forward` returns spikes alone and the state is held on the layer --
    but the reset differs, for the reason documented on `reset()`.
    """

    def __init__(self, neuron_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.neuron_cfg = neuron_cfg
        self.lif = build_lif(neuron_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Charge, fire, reset for ONE timestep.

        The unsqueeze/squeeze is not cosmetic. sinabs is written for whole
        sequences and unpacks its input as

            batch_size, time_steps, *trailing_dim = input_data.shape

        so a single timestep must arrive as (batch, 1, C, H, W). Passing
        (batch, C, H, W) instead would be accepted without complaint and would
        treat the channel axis as time -- wrong answers, no error.

        Because the membrane persists in a buffer between calls, feeding one
        timestep per call is equivalent to feeding the whole sequence at once,
        the same way SpikingJelly's step_mode='s' relates to 'm'.
        """
        spikes = self.lif(x.unsqueeze(1)).squeeze(1)
        self._record(spikes)
        return spikes

    def reset(self) -> None:
        """Drop the state by restoring its buffers to zero-size.

        sinabs' own call is `reset_states()`, and we deliberately do NOT use it.
        Its body is `buffer.zero_()` -- it zeroes the values but KEEPS the
        buffer's shape, which leaves the layer "initialised" for a specific batch
        size. Two problems follow from that, both avoided here:

          1. A short final batch then takes sinabs'
             `handle_state_batch_size_mismatch` path, which resamples the state
             with `torch.randint` -- i.e. it fills each neuron's membrane by
             copying a RANDOM other sample's. Harmless only as long as the
             buffer happens to be all zeros at that moment; not something to
             depend on.
          2. `is_state_initialised()` would keep returning True after a reset, so
             `has_state()` could not honestly report whether the reset worked --
             which is the whole point of check_network.py verifying it.

        Restoring the zero-size tensor that `register_buffer` originally held
        makes `is_state_initialised()` False again, so the next forward pass
        re-infers the shape from its actual input. That is exactly the
        "None means fresh neuron" semantics of the snnTorch and Norse adapters.
        `zero_()`'s companion `detach_()` is not needed either: a brand-new
        tensor carries no graph.
        """
        for name, buffer in list(self.lif.named_buffers()):
            self.lif.register_buffer(name, torch.zeros((0), device=buffer.device))

    def has_state(self) -> bool:
        """sinabs' own test: are all buffers still zero-size?

        Cheap and exact -- it reads shapes only, so unlike a "is the membrane
        non-zero" test it forces no host/device synchronisation.
        """
        return bool(self.lif.is_state_initialised())

    def describe(self) -> dict[str, Any]:
        tau_mem = require_float(self.neuron_cfg, "sinabs.tau_mem")
        norm_input = require_bool(self.neuron_cfg, "sinabs.norm_input")
        decay = float(torch.exp(torch.tensor(-1.0 / tau_mem)))
        gain = (1.0 - decay) if norm_input else 1.0
        surrogate_type = require_str(self.neuron_cfg, "sinabs.surrogate.type")
        surrogate_args = ", ".join(
            f"{name}={require_float(self.neuron_cfg, f'sinabs.surrogate.{name}')}"
            for name in SURROGATE_PARAMS[surrogate_type]
        )
        return {
            "class": "sinabs.layers.LIF",
            "tau_mem": tau_mem,
            "tau_syn": require_optional_float(self.neuron_cfg, "sinabs.tau_syn"),
            "spike_threshold": require_float(
                self.neuron_cfg, "sinabs.spike_threshold"
            ),
            "spike_fn": require_str(self.neuron_cfg, "sinabs.spike_fn"),
            "reset_mechanism": require_str(self.neuron_cfg, "sinabs.reset_mechanism"),
            "v_reset": require_float(self.neuron_cfg, "sinabs.v_reset"),
            "min_v_mem": require_optional_float(self.neuron_cfg, "sinabs.min_v_mem"),
            "norm_input": norm_input,
            "train_alphas": require_bool(self.neuron_cfg, "sinabs.train_alphas"),
            "surrogate": f"{surrogate_type}({surrogate_args})",
            # Derived, printed for reading only -- nothing consumes it. Put on the
            # same decay/gain scale as the other three adapters so the four can be
            # compared at a glance.
            "effective_decay_gain": f"{decay:.4f}/{gain:.4f}",
        }
