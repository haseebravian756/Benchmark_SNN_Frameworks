"""The network that hands each layer the whole [T, ...] sequence at once.

A subclass of `SpikingNet` that overrides **only** `forward`. It inherits the layer
list, `time_steps`, `lif_layers()` and `reset()` unchanged, so it is the same network
with a different execution strategy -- not a second network that could drift.

The idea in one line: only the LIF layers care about time, so time can be folded
into the batch dimension for everything else.

    Conv2d, MaxPool2d, Flatten, Linear   are BATCH-INDEPENDENT, so
        [T, N, ...] -> flatten to [T*N, ...] -> layer -> reshape back
    is exactly equal to looping the layer over T.

That is the same trick SpikingJelly's own `functional.seq_to_ann_forward` uses, but
written in plain torch so no framework-specific code enters the shared network. The
frameworks stay swappable.

Verified on a T4 at the real conv1 shape: outputs bit-identical, input gradients
agreeing to 8.3e-07, weight gradients to 5.4e-07 relative (they are sums over
T*N*H*W with magnitude ~7e4, so reordering them costs a few float32 eps -- and
checked against a float64 reference, BOTH orderings are correct).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.adapters.base import BaseLIF
from src.network import SpikingNet


def apply_stateless(layer: nn.Module, x_seq: torch.Tensor) -> torch.Tensor:
    """Run a time-independent layer over all T timesteps in ONE call.

    `reshape` rather than `view` on the way back: `Flatten` can return a
    non-contiguous result, on which `view` would raise. Where the tensor is already
    contiguous -- the usual case -- reshape costs nothing.
    """
    time_steps, batch = x_seq.shape[0], x_seq.shape[1]
    folded = layer(x_seq.flatten(0, 1))
    return folded.reshape(time_steps, batch, *folded.shape[1:])


class SequenceSpikingNet(SpikingNet):
    """Runs the layer stack over all T timesteps at once; returns spike counts.

    Input  [T, batch, channels, height, width]
    Output [batch, num_classes] -- identical meaning to SpikingNet's output, so
            every caller (train.py, the metrics, the check scripts) is unaffected.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != self.time_steps:
            raise ValueError(
                f"expected {self.time_steps} timesteps on dim 0, got shape "
                f"{tuple(x.shape)}. Input must be [T, batch, C, H, W]."
            )

        # Same contract as SpikingNet: reset here, not at the call site, so stateful
        # neurons cannot leak across samples. Inherited unchanged -- it walks
        # lif_layers() and calls each reset().
        self.reset()

        out = x
        for layer in self.layers:
            out = self.apply_layer(layer, out)

        # [T, batch, num_classes] -> [batch, num_classes]. SpikingNet accumulates
        # the same sum inside its timestep loop; here the whole sequence exists, so
        # it is one reduction.
        return out.sum(0)

    def apply_layer(self, layer: nn.Module, x_seq: torch.Tensor) -> torch.Tensor:
        """Run ONE layer over a [T, batch, ...] tensor, whatever kind it is.

        Public so that diagnostics -- check_network.py's layer-by-layer shape walk --
        can step through the stack without reimplementing the LIF-versus-stateless
        dispatch. One copy of that decision, so the diagnostic cannot disagree with
        what `forward` actually does.
        """
        if not isinstance(layer, BaseLIF):
            return apply_stateless(layer, x_seq)

        sequence_forward = getattr(layer, "forward_sequence", None)
        if sequence_forward is None:
            raise TypeError(
                f"{type(layer).__name__} is in a SequenceSpikingNet but has no "
                "forward_sequence(). Every LIF layer here must consume the whole "
                "[T, batch, ...] tensor. Build the network through "
                "src.multistep.build.build_net, which swaps the layers for you, "
                "rather than constructing this class directly."
            )
        return sequence_forward(x_seq)
