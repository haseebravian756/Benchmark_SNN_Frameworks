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

ON COLAB, add --cache-archive. Colab wipes the local disk on every disconnect, so
without it the cache is rebuilt from scratch each session. With it, the built
cache is packed onto Drive once and unpacked in every later session:

    # first session: builds the cache, then parks it on Drive
    python prepare_data.py --config config/config_ex2.yaml \
        --cache-archive "/content/drive/MyDrive/snn_cache"

    # every session after that: same command, unpacks instead of rebuilding
    python prepare_data.py --config config/config_ex2.yaml \
        --cache-archive "/content/drive/MyDrive/snn_cache"

The same command both times -- it stores when the cache is new and restores when
a matching archive is already there. "Matching" means the archive's manifest is
equal to the one this config asks for, the same test the local cache already
uses, so a changed dataset setting can never restore the wrong data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src import cache_archive
from src.config import ConfigError, load_config, require, run_banner
from src.data import build_loader, build_split, prebuild_cache, split_identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="path to the experiment's YAML config. REQUIRED and with no default, so a run can never silently use another experiment's neuron: config/default.yaml is ex1 (forced-equivalent neuron), config/config_ex2.yaml is ex2 (each framework out of the box).",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="LABEL ONLY, e.g. ex2. The cache is shared by dataset settings, not "
        "by experiment; the label just records what you were preparing for.",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=["train", "test"], default=["train", "test"],
        help="which splits to prepare (each is always the COMPLETE split)",
    )
    parser.add_argument(
        "--no-prebuild", action="store_true",
        help="skip writing the full cache; just fetch one batch and report shapes",
    )
    parser.add_argument(
        "--cache-archive", default=None, metavar="DIR",
        help="a folder that OUTLIVES this machine, normally a mounted Google "
        "Drive. A matching archive there is unpacked instead of rebuilding the "
        "cache; a newly built cache is packed into it. Omit for the original "
        "behaviour -- nothing is read or written.",
    )
    parser.add_argument(
        "--cache-archive-compress", action="store_true",
        help="gzip the archive. Smaller on Drive and quicker to upload, at the "
        "cost of CPU on both ends. Colab typically has ~2 cores, so this is a "
        "real trade rather than a free win. Either kind restores.",
    )
    args = parser.parse_args()

    archive_root = Path(args.cache_archive) if args.cache_archive else None

    config = load_config(args.config)
    dataset_cfg = require(config, "dataset")

    print(run_banner(
        "prepare_data.py -- build the dataset cache",
        experiment=args.experiment,
        config_path=args.config,
        config=config,
        writes_results=False,
    ))
    print("The cache is keyed by the DATASET settings only, so any two experiments")
    print("with identical dataset blocks share it -- exactly what you want, since")
    print("they are meant to see byte-identical data. Changing a dataset setting")
    print("builds a separate cache rather than overwriting the old one.")
    print()
    print(f"tonic dataset: {dataset_cfg['name']}")
    print(f"framing mode:  {dataset_cfg['framing']['mode']}")
    print(f"denoise:       {dataset_cfg['denoise_filter_time_us']} us")
    print(f"binarize:      {dataset_cfg['binarize']}")
    if archive_root is not None:
        print(f"cache archive: {archive_root}"
              f"{'  (gzip)' if args.cache_archive_compress else ''}")
        if args.no_prebuild:
            print("               NOTE: --no-prebuild never completes the cache, so")
            print("               nothing will be stored. Restoring still works.")
    print()

    info = None
    for split in args.splits:
        is_train = split == "train"
        print(f"[{split}]")
        dataset, info = build_split(
            dataset_cfg, train=is_train, cache_archive=archive_root
        )
        print(f"  samples: {len(dataset)}")

        if not args.no_prebuild:
            prebuild_cache(dataset, split)

            # After prebuild and not before: `store` refuses a partial cache, so
            # archiving here is the only point where the split is known complete.
            if archive_root is not None:
                identity = split_identity(dataset_cfg, train=is_train)
                cache_archive.store(
                    archive_root,
                    identity.archive_stem,
                    identity.cache_dir,
                    identity.manifest,
                    expected_samples=len(dataset),
                    compress=args.cache_archive_compress,
                )

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
