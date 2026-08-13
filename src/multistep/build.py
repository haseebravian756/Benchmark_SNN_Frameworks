"""The one decision point: normal pipeline, or multi-step?

Every script that builds a network calls `build_net` instead of `build_network`.
For all six existing configurations it returns exactly what `build_network` returns,
by calling it -- so this file is inert for every result produced so far.

    step_mode 's'  (or any non-SpikingJelly framework)  ->  build_network, unchanged
    step_mode 'm'                                       ->  SequenceSpikingNet

The interesting part is HOW the multi-step network is built, in `build_net` below:
it reuses `build_network` wholesale and then swaps the LIF layers out. That is
possible because SpikingJelly's LIFNode carries no learnable parameters at all --

    LIFNode(...).parameters()  ->  []
    LIFNode(...).state_dict()  ->  {}

`tau` is a plain float and the surrogate's `alpha` a plain attribute. So swapping
LIF layers after construction moves no weights, and the weight fingerprint from
check_network.py is unchanged.

The alternative would have been to re-declare the architecture here, which is
exactly what src/network.py's docstring exists to prevent: a second copy that can
drift. There is still only one place that says what the network is.
"""

from __future__ import annotations

from typing import Any

from src.adapters import lif_factory
from src.adapters.base import BaseLIF
from src.config import require_choice
from src.data import DataInfo
from src.network import SpikingNet, build_network


def wants_multistep(framework: str, neuron_cfg: dict[str, Any]) -> bool:
    """Does this configuration ask for the multi-step path?

    Reading `step_mode` through `require_choice` means an invalid value fails here
    with the project's normal config error, rather than being read as "not 'm'" and
    quietly taking the default path.
    """
    if framework != "spikingjelly":
        return False
    return require_choice(neuron_cfg, "spikingjelly.step_mode", ["s", "m"]) == "m"


def single_step_config(neuron_cfg: dict[str, Any]) -> dict[str, Any]:
    """The same config with SpikingJelly forced to single-step.

    Used to build the scaffold network in `build_net`. Copies rather than mutates:
    the caller's dict is shared with the results file and the config hash, so
    editing it in place would misreport what was run.
    """
    return {
        **neuron_cfg,
        "spikingjelly": {
            **neuron_cfg["spikingjelly"],
            "step_mode": "s",
            "backend": "torch",
        },
    }


def build_net(
    framework: str, neuron_cfg: dict[str, Any], info: DataInfo, seed: int
) -> SpikingNet:
    """Build the network this config asks for. Drop-in for `build_network`."""
    if not wants_multistep(framework, neuron_cfg):
        return build_network(lif_factory(framework, neuron_cfg), info, seed=seed)

    # Imported here rather than at module scope so that the normal path above never
    # imports the multi-step adapter -- consistent with src/adapters/__init__.py,
    # which keeps one framework per process on purpose.
    from src.multistep.sequence_net import SequenceSpikingNet
    from src.multistep.sj_lif_multistep import SpikingJellyMultiStepLIF

    # Step 1: build a completely normal single-step network. This is what gives the
    # multi-step run byte-identical starting weights: torch.manual_seed(seed) then
    # conv1, conv2, linear draw from the RNG in that order, and infer_flat_features
    # runs its shape probe exactly as it does for every other run.
    #
    # It must be single-step for that probe to work: the probe pushes one
    # [1, C, H, W] frame through the layers, and a multi-step node would read that
    # leading 1 as T. (Verified: it returns the same shape, so nothing would crash
    # and the error would be silent.)
    scaffold = build_network(
        lif_factory("spikingjelly", single_step_config(neuron_cfg)), info, seed=seed
    )

    # Step 2: swap each LIF for its multi-step twin. No parameters move -- see the
    # module docstring. Conv2d, MaxPool2d, Flatten and Linear are carried over as
    # the very same objects, weights included.
    layers = [
        SpikingJellyMultiStepLIF(neuron_cfg) if isinstance(layer, BaseLIF) else layer
        for layer in scaffold.layers
    ]

    return SequenceSpikingNet(layers, scaffold.time_steps)
