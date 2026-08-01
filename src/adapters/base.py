"""The contract every framework's LIF layer must satisfy.

This is the whole adapter layer. Everything else in the pipeline -- the network,
the training loop, the metrics -- talks only to this interface and never learns
which framework it got.

Only the neuron differs between snnTorch, SpikingJelly and Norse. Conv2d,
MaxPool2d and Linear are plain torch.nn in all three cases. So instead of three
network classes that could drift apart, there is ONE network class and three
implementations of the single layer below.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseLIF(nn.Module):
    """One layer of leaky integrate-and-fire neurons.

    Subclasses must provide exactly two behaviours:

      forward(x) -> spikes, the same shape as x
      reset()    -> forget all neuron state

    The state handling is the interesting part, and it differs per framework:
    snnTorch and Norse hand state back to the caller, SpikingJelly keeps it on
    the module. Hiding that here is what makes the training loop
    framework-agnostic -- and it makes the "forgot to reset between batches"
    bug (the #1 silent bug in the reference doc) structurally impossible, because
    the network resets itself at the start of every forward pass.
    """

    def __init__(self) -> None:
        super().__init__()
        # Spike counting for the spike-rate metric, filled in later by metrics.py.
        #
        # Off by default and deliberately so: summing spikes costs an extra GPU
        # kernel per layer per timestep, and this project measures wall-clock
        # time. Enable it only for the dedicated measurement passes.
        self.count_spikes: bool = False
        self.spike_total: torch.Tensor | float = 0.0  # kept on-device, never .item()ed here
        self.spike_slots: int = 0                    # neurons x timesteps x batch seen
        # Shape of one sample's output, e.g. (12, 30, 30). Learned from the first
        # counted step so layers.csv can report neurons per layer without the
        # network having to describe itself.
        self.spike_shape: tuple[int, ...] | None = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Charge, fire, reset for one timestep. Returns spikes shaped like x."""

    @abstractmethod
    def reset(self) -> None:
        """Drop all neuron state. Called at the start of every batch."""

    def has_state(self) -> bool:
        """Is any neuron state currently being held?

        Not part of the computational contract -- this exists so that a reset
        can be *verified* rather than assumed, which is what check_network.py
        does. Each framework stores its state somewhere different, so each
        adapter answers for itself.
        """
        return False

    def _record(self, spikes: torch.Tensor) -> None:
        """Accumulate spike statistics without stalling the GPU.

        .sum() stays a device tensor -- calling .item() here would force a
        host/device synchronisation on every layer of every timestep and wreck
        the timing measurements. It is converted to a number only when read.
        """
        if not self.count_spikes:
            return
        if self.spike_shape is None:
            self.spike_shape = tuple(spikes.shape[1:])  # drop the batch dimension
        self.spike_total = self.spike_total + spikes.detach().sum()
        self.spike_slots += spikes.numel()

    def reset_spike_stats(self) -> None:
        self.spike_total = 0.0
        self.spike_slots = 0
        self.spike_shape = None

    def neurons(self) -> int:
        """Neurons in this layer, per sample. 0 until something has been counted."""
        if self.spike_shape is None:
            return 0
        count = 1
        for dimension in self.spike_shape:
            count *= dimension
        return count

    def spike_rate(self) -> float:
        """Mean spikes per neuron per timestep, as a FRACTION (x100 for %).

        Reads the accumulated device tensor exactly once -- doing it inside the
        forward loop would force a host/device sync on every layer of every step.
        """
        if self.spike_slots == 0:
            return 0.0
        total = self.spike_total
        if isinstance(total, torch.Tensor):
            total = total.item()
        return float(total) / self.spike_slots

    def describe(self) -> dict[str, Any]:
        """Framework-specific settings actually in force, for the results file."""
        return {}
