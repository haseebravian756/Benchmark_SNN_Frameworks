"""The spiking convolutional network. ONE class, shared by all three frameworks.

The architecture lives here in Python, not in the config, on purpose: it is read
top to bottom as one line per layer.

Only `make_lif` changes between frameworks. Every Conv2d, MaxPool2d and Linear is
plain torch.nn and is therefore literally the same code in all three runs. That
makes "identical architecture, only the framework changed" true by construction
rather than by careful copying -- there is no second copy that could drift.

Architecture: 12C5 - MP2 - 32C5 - MP2 - FC10
From the snnTorch/Tonic N-MNIST tutorial, so it is a cited, proven design rather
than one invented for this project.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from src.adapters.base import BaseLIF
from src.data import DataInfo

# ---------------------------------------------------------------------------
# Architecture knobs. Change these to change the network.
#
# The flatten size is NOT here: it is measured with a dummy forward pass, so
# editing anything below cannot leave a stale hardcoded number behind.
# ---------------------------------------------------------------------------
CONV1_CHANNELS = 12
CONV1_KERNEL = 5
CONV2_CHANNELS = 32
CONV2_KERNEL = 5
POOL_SIZE = 2


def build_feature_layers(make_lif: Callable[[], BaseLIF], info: DataInfo) -> list[nn.Module]:
    """Everything up to and including the flatten. One line per layer, in order.

    Shapes on N-MNIST (34x34, 2 channels), shown per timestep:

        input                     2 x 34 x 34
        Conv2d(2->12, k5)        12 x 30 x 30    34 - 5 + 1 = 30
        LIF                      12 x 30 x 30
        MaxPool2d(2)             12 x 15 x 15    30 / 2 = 15
        Conv2d(12->32, k5)       32 x 11 x 11    15 - 5 + 1 = 11
        LIF                      32 x 11 x 11
        MaxPool2d(2)             32 x  5 x  5    11 / 2 = 5, rounded down
        Flatten                       800        32 * 5 * 5

    Note the pooling comes AFTER the LIF, so it pools on spikes rather than on
    raw membrane voltages.
    """
    return [
        nn.Conv2d(info.channels, CONV1_CHANNELS, kernel_size=CONV1_KERNEL),
        make_lif(),
        nn.MaxPool2d(POOL_SIZE),
        nn.Conv2d(CONV1_CHANNELS, CONV2_CHANNELS, kernel_size=CONV2_KERNEL),
        make_lif(),
        nn.MaxPool2d(POOL_SIZE),
        nn.Flatten(),
    ]


def infer_flat_features(layers: list[nn.Module], info: DataInfo) -> int:
    """Push one empty frame through the layers and see how wide the output is.

    This is what replaces hardcoding 800. Change a kernel size, a channel count
    or the dataset, and the number simply follows.

    A batch of 1 zeros is enough: only shapes matter, and the LIF layers preserve
    shape whatever the values. Their state is reset afterwards so this probe
    leaves nothing behind.
    """
    probe = torch.zeros(1, info.channels, info.height, info.width)
    with torch.no_grad():
        for layer in layers:
            probe = layer(probe)
    for layer in layers:
        if isinstance(layer, BaseLIF):
            layer.reset()
    return probe.shape[1]


class SpikingNet(nn.Module):
    """Runs the layer stack over T timesteps and returns output spike counts.

    Input  [T, batch, channels, height, width]
    Output [batch, num_classes] -- how many times each output neuron fired
            across the T timesteps. Highest count is the prediction.
    """

    def __init__(self, layers: list[nn.Module], time_steps: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.time_steps = time_steps

    def lif_layers(self) -> list[BaseLIF]:
        return [layer for layer in self.layers if isinstance(layer, BaseLIF)]

    def reset(self) -> None:
        """Clear every neuron's state, whatever framework it comes from."""
        for layer in self.lif_layers():
            layer.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != self.time_steps:
            raise ValueError(
                f"expected {self.time_steps} timesteps on dim 0, got shape "
                f"{tuple(x.shape)}. Input must be [T, batch, C, H, W]."
            )

        # Reset here, not in the training loop. Stateful neurons leaking across
        # samples is the classic silent bug in SNN code; doing it here means it
        # cannot be forgotten at a call site.
        self.reset()

        total = None
        for step in range(self.time_steps):
            out = x[step]
            for layer in self.layers:
                out = layer(out)
            total = out if total is None else total + out
        return total


def build_network(
    make_lif: Callable[[], BaseLIF], info: DataInfo, seed: int
) -> SpikingNet:
    """Build the network with reproducible weights.

    The seed is set immediately before construction, so the random weight init
    depends only on the seed. The LIF wrappers draw no random numbers, which is
    what makes all three frameworks start from byte-identical weights: the only
    consumers of the RNG are the two Conv2d layers and the Linear, in that order.
    """
    torch.manual_seed(seed)

    features = build_feature_layers(make_lif, info)
    flat_features = infer_flat_features(features, info)

    classifier = [
        nn.Linear(flat_features, info.num_classes),
        make_lif(),
    ]
    return SpikingNet(features + classifier, info.time_steps)
