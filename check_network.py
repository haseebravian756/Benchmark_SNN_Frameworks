"""Build the network for one framework and report what it actually is.

No dataset needed -- it feeds a dummy tensor of the right shape. Use it after
changing the architecture, or when adding a framework, to confirm that:

  * every layer's output shape is what you expect
  * the flatten size was measured correctly (not hardcoded)
  * the starting weights are byte-identical to the other frameworks' for the
    same seed, which is what makes the benchmark a fair comparison

    python check_network.py --config config/default.yaml --framework snntorch
    python check_network.py --config config/default.yaml --framework snntorch --seed 1
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

from src.adapters import IMPLEMENTED, lif_factory
from src.config import (
    ConfigError,
    load_config,
    require,
    require_bool,
    require_optional_int,
    require_str,
)
from src.data import DATASETS, DataInfo, build_framing
from src.network import SpikingNet, build_network


def data_info_without_download(dataset_cfg: dict) -> DataInfo:
    """Describe the data shape from the dataset class alone.

    tonic exposes sensor_size as a class attribute, so the network shape can be
    derived without downloading anything. prepare_data.py already confirmed
    these numbers against the real files.
    """
    name = require_str(dataset_cfg, "name")
    if name not in DATASETS:
        raise ConfigError(f"unknown dataset.name '{name}'. Supported: {sorted(DATASETS)}")
    entry = DATASETS[name]
    width, height, channels = entry["class"].sensor_size
    _, time_steps, framing = build_framing(dataset_cfg, entry["class"].sensor_size)

    return DataInfo(
        name=name,
        time_steps=time_steps,
        channels=channels,
        height=height,
        width=width,
        num_classes=entry["num_classes"],
        cache_root=Path("(not used by this check)"),
        framing=framing,
        denoise_filter_time_us=require_optional_int(dataset_cfg, "denoise_filter_time_us"),
        binarize=require_bool(dataset_cfg, "binarize"),
    )


def weight_fingerprint(net: SpikingNet) -> str:
    """Short hash of every TRAINABLE weight, in a fixed order.

    Two frameworks built with the same seed must produce the same string. If they
    do not, they are not starting from the same place and no accuracy comparison
    between them means anything.

    Trainable parameters only -- deliberately NOT state_dict(). Each framework
    registers its own non-trainable buffers on its neuron (snnTorch adds
    threshold, beta, graded_spikes_factor and reset_mechanism_val per layer;
    the other two register different ones). Including those would make the
    fingerprints differ for reasons that have nothing to do with the weights,
    which is exactly what this check is supposed to rule out.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(
        (n, t) for n, t in net.named_parameters() if t.requires_grad
    ):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--framework", default="snntorch",
                        help=f"implemented so far: {IMPLEMENTED}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4,
                        help="dummy batch size (small keeps this fast on CPU)")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = require(config, "dataset")
    neuron_cfg = require(config, "neuron")

    info = data_info_without_download(dataset_cfg)
    print(f"framework : {args.framework}")
    print(f"seed      : {args.seed}")
    print(f"data      : T={info.time_steps}  {info.channels}x{info.height}x{info.width}"
          f"  -> {info.num_classes} classes")
    print()

    make_lif = lif_factory(args.framework, neuron_cfg)
    net = build_network(make_lif, info, seed=args.seed)

    # Walk one timestep through the stack, printing what each layer produces.
    print("=" * 72)
    print("layer by layer, shapes for ONE timestep")
    print("=" * 72)
    activation = torch.zeros(args.batch, info.channels, info.height, info.width)
    print(f"  {'input':<28} {tuple(activation.shape)}")
    with torch.no_grad():
        for layer in net.layers:
            activation = layer(activation)
            label = type(layer).__name__
            extra = ""
            if isinstance(layer, torch.nn.Conv2d):
                extra = f"({layer.in_channels}->{layer.out_channels}, k{layer.kernel_size[0]})"
            elif isinstance(layer, torch.nn.MaxPool2d):
                extra = f"({layer.kernel_size})"
            elif isinstance(layer, torch.nn.Linear):
                extra = f"({layer.in_features}->{layer.out_features})"
            print(f"  {label + ' ' + extra:<28} {tuple(activation.shape)}")
    net.reset()

    linear = [m for m in net.layers if isinstance(m, torch.nn.Linear)][0]
    print()
    print(f"  flatten size measured by dummy forward: {linear.in_features}")

    print()
    print("=" * 72)
    print("full forward pass over all timesteps")
    print("=" * 72)
    batch = torch.zeros(info.time_steps, args.batch, info.channels, info.height, info.width)
    with torch.no_grad():
        output = net(batch)
    print(f"  input  {tuple(batch.shape)}   [T, batch, C, H, W]")
    print(f"  output {tuple(output.shape)}   [batch, num_classes] = spike counts over T")

    print()
    print("=" * 72)
    print("parameters and neuron settings")
    print("=" * 72)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  trainable parameters: {trainable:,}")
    for index, layer in enumerate(net.lif_layers()):
        print(f"  LIF {index}: {layer.describe()}")

    print()
    print(f"  weight fingerprint (seed {args.seed}): {weight_fingerprint(net)}")
    print("  ^ must be IDENTICAL across frameworks for the same seed")

    # Reset must genuinely clear state, or neurons leak between batches.
    with torch.no_grad():
        net(batch)
    leftover = [
        type(layer).__name__ for layer in net.lif_layers()
        if getattr(layer, "membrane", None) is not None
    ]
    net.reset()
    still_set = [
        type(layer).__name__ for layer in net.lif_layers()
        if getattr(layer, "membrane", None) is not None
    ]
    print()
    print(f"  state present after a forward pass: {leftover or 'none'}")
    print(f"  state present after reset():        {still_set or 'none'}  (should be none)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as error:
        print(f"\nCONFIG ERROR: {error}", file=sys.stderr)
        sys.exit(2)
