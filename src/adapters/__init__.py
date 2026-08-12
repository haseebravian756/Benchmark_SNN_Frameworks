"""Pick a framework's LIF layer by name.

Imports are deliberately LAZY -- done inside the function rather than at the top
of this file. Reason: this project runs one framework per process on purpose, and
importing all of them would pull every library into memory, any of which might
initialise CUDA or set global torch state. Importing only the one being measured
keeps the runs independent.

The laziness also means a framework that is NOT INSTALLED costs nothing until it
is asked for -- which is currently the case for sinabs.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from src.adapters.base import BaseLIF
from src.config import ConfigError

FRAMEWORKS = ["snntorch", "spikingjelly", "norse", "sinabs"]

# Which of the above are actually implemented so far.
#
# sinabs is listed: its adapter exists and its config block exists. But the
# PACKAGE is not installed yet, so asking for it raises ImportError at the lazy
# import below rather than a ConfigError here. That is the honest failure -- the
# wiring is done, the dependency is not -- and the message below says so.
IMPLEMENTED = ["snntorch", "spikingjelly", "norse", "sinabs"]


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

    if framework == "sinabs":
        try:
            from src.adapters.sinabs_lif import SinabsLIF
        except ImportError as error:  # sinabs is not installed yet
            raise ConfigError(
                "framework 'sinabs' is wired up but the package is not installed "
                f"in this environment ({error}).\n"
                "  Install it with:  pip install sinabs\n"
                "  CHECK AFTERWARDS that pip did not move torch -- this project's "
                "speed numbers assume one fixed torch version, and a silent "
                "upgrade would invalidate the runs already recorded."
            ) from error

        return partial(SinabsLIF, neuron_cfg)

    raise ConfigError(f"no factory wired up for '{framework}'")  # pragma: no cover
