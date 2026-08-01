"""Pick a framework's LIF layer by name.

Imports are deliberately LAZY -- done inside the function rather than at the top
of this file. Reason: this project runs one framework per process on purpose, and
importing all three would pull three libraries into memory, any of which might
initialise CUDA or set global torch state. Importing only the one being measured
keeps the runs independent.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from src.adapters.base import BaseLIF
from src.config import ConfigError

FRAMEWORKS = ["snntorch", "spikingjelly", "norse"]

# Which of the above are actually implemented so far.
IMPLEMENTED = ["snntorch", "spikingjelly", "norse"]


def lif_factory(framework: str, neuron_cfg: dict[str, Any]) -> Callable[[], BaseLIF]:
    """Return a zero-argument callable that builds one LIF layer.

    The network calls this once per LIF position, so every LIF in the network
    gets identical settings from the framework's own config block.
    """
    if framework not in FRAMEWORKS:
        raise ConfigError(
            f"unknown framework '{framework}'. Must be one of {FRAMEWORKS}"
        )
    if framework not in IMPLEMENTED:
        raise ConfigError(
            f"framework '{framework}' is not implemented yet. "
            f"Currently available: {IMPLEMENTED}"
        )

    if framework == "snntorch":
        from src.adapters.snntorch_lif import SnnTorchLIF

        return partial(SnnTorchLIF, neuron_cfg)

    if framework == "spikingjelly":
        from src.adapters.spikingjelly_lif import SpikingJellyLIF

        return partial(SpikingJellyLIF, neuron_cfg)

    if framework == "norse":
        from src.adapters.norse_lif import NorseLIF

        return partial(NorseLIF, neuron_cfg)

    raise ConfigError(f"no factory wired up for '{framework}'")  # pragma: no cover
