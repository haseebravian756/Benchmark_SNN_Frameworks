"""Download the dataset, build the Tonic disk cache, and report what came out.

Run this ONCE per machine before training. Building the cache here rather than
during training matters for fairness: Tonic converts raw event lists into frames
on first access, and whichever framework happened to run first would otherwise
absorb that entire one-time cost and look slow for no reason.

    # both splits (this is what you run on Colab)
    python prepare_data.py --config config/default.yaml

    # test split only (enough to verify shapes locally without 4 GB of cache)
    python prepare_data.py --config config/default.yaml --splits test

    # check shapes without writing the whole cache
    python prepare_data.py --config config/default.yaml --splits test --no-prebuild

There is no option to use a subset of a split. Splits are always complete.
"""

from __future__ import annotations

import argparse
import sys

import torch

from src.config import ConfigError, load_config, require
from src.data import build_loader, build_split, prebuild_cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--splits", nargs="+", choices=["train", "test"], default=["train", "test"],
        help="which splits to prepare (each is always the COMPLETE split)",
    )
    parser.add_argument(
        "--no-prebuild", action="store_true",
        help="skip writing the full cache; just fetch one batch and report shapes",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = require(config, "dataset")

    print(f"tonic dataset: {dataset_cfg['name']}")
    print(f"framing mode:  {dataset_cfg['framing']['mode']}")
    print(f"denoise:       {dataset_cfg['denoise_filter_time_us']} us")
    print(f"binarize:      {dataset_cfg['binarize']}")
    print()

    info = None
    for split in args.splits:
        is_train = split == "train"
        print(f"[{split}]")
        dataset, info = build_split(dataset_cfg, train=is_train)
        print(f"  samples: {len(dataset)}")

        if not args.no_prebuild:
            prebuild_cache(dataset, split)

        # One real batch, to confirm the shape rather than assume it. The doc
        # asks for exactly this check before anything gets built on top.
        loader = build_loader(
            dataset, dataset_cfg,
            shuffle=require(config, f"dataset.shuffle_{'train' if is_train else 'test'}"),
            seed=0,
        )
        frames, labels = next(iter(loader))
        print(f"  batch frames : {tuple(frames.shape)}  dtype {frames.dtype}")
        print(f"  batch labels : {tuple(labels.shape)}  dtype {labels.dtype}")
        print(f"  value range  : min {frames.min().item():.1f}  max {frames.max().item():.1f}")
        print(f"  mean events per frame per sample: {frames.sum().item() / (frames.shape[0] * frames.shape[1]):.1f}")
        print(f"  memory for this batch: {frames.element_size() * frames.nelement() / 1024**2:.1f} MB")
        print()

    if info is None:
        print("no splits requested")
        return 1

    print("=" * 60)
    print("shape the network will be built from")
    print("=" * 60)
    print(f"  T (timesteps)  {info.time_steps}")
    print(f"  channels       {info.channels}")
    print(f"  height x width {info.height} x {info.width}")
    print(f"  num_classes    {info.num_classes}")
    print(f"  sample shape   {info.sample_shape}  (batch dimension added on top)")
    print(f"  cache root     {info.cache_root}")
    print()
    print("Expected batch shape is [T, batch, channels, height, width].")
    print(f"With batch_size {dataset_cfg['batch_size']}: "
          f"({info.time_steps}, {dataset_cfg['batch_size']}, {info.channels}, "
          f"{info.height}, {info.width})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as error:
        print(f"\nCONFIG ERROR: {error}", file=sys.stderr)
        sys.exit(2)
