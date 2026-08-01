"""Event dataset loading via Tonic. Completely framework-agnostic.

This module exists to guarantee one thing: **all three frameworks receive
byte-identical batches from the same disk cache.** If data preparation differed
between frameworks, the speed comparison would be measuring Tonic, not the
frameworks.

The pipeline per sample:

    raw events -> Denoise -> ToFrame -> fixed frame count -> CACHE -> binarize -> batch
    (x,y,t,p)     drop      bin them    pad or crop         written    optional    stacked
                  isolated                                  once      0/1 clamp

Note where the cache sits: denoising and framing are expensive and are cached;
the optional binarize clamp is a cheap elementwise op applied AFTER reading from
the cache. That way flipping `binarize` costs nothing, instead of forcing a
multi-gigabyte cache rebuild.

Batches come out shaped [T, batch, channels, height, width] -- time first,
because the training loop walks the timesteps and feeds one frame of the whole
batch to the network at a time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np
import torch
import tonic
from torch.utils.data import DataLoader

from src.config import (
    ConfigError,
    require_bool,
    require_choice,
    require_int,
    require_optional_int,
    require_str,
)

# Datasets this module can load. num_classes is a fact about the dataset, not an
# experiment setting, so it lives here rather than in the config.
DATASETS: dict[str, dict[str, Any]] = {
    "nmnist": {"class": tonic.datasets.NMNIST, "num_classes": 10},
}

# Which framing modes are supported, and which config key supplies each one's
# parameter. Tonic also offers event_count and n_event_bins; add them here if
# you ever want them.
FRAMING_MODES = {
    "n_time_bins": "n_time_bins",
    "time_window": "time_window_us",
}


class DataInfo(NamedTuple):
    """Everything the rest of the pipeline needs to know about the data.

    The network is built from these numbers, so changing the dataset or the
    framing automatically changes the network shape -- nothing is hardcoded.
    """

    name: str
    time_steps: int          # T -- frames per sample
    channels: int            # 2 for event polarity (on/off)
    height: int
    width: int
    num_classes: int
    cache_root: Path         # the framing-specific cache directory
    framing: dict[str, Any]  # exactly what produced this cache
    denoise_filter_time_us: int | None
    binarize: bool           # applied after the cache, so not part of its identity

    @property
    def sample_shape(self) -> tuple[int, int, int, int]:
        """One sample, without the batch dimension: [T, C, H, W]."""
        return (self.time_steps, self.channels, self.height, self.width)


class PadOrCropFrames:
    """Force every sample to exactly `frames` frames.

    Needed for time_window framing: recordings are not all the same length, so
    a fixed time window produces a different frame count per sample -- and a
    batch is one rectangular tensor, which cannot hold rows of different
    lengths. Short samples get empty (all-zero) frames appended; long ones are
    truncated.

    A module-level class rather than a lambda because DataLoader workers have to
    pickle the transform.
    """

    def __init__(self, frames: int) -> None:
        self.frames = frames

    def __call__(self, frame_array: np.ndarray) -> np.ndarray:
        count = frame_array.shape[0]
        if count == self.frames:
            return frame_array
        if count > self.frames:
            return frame_array[: self.frames]
        padding = np.zeros(
            (self.frames - count, *frame_array.shape[1:]), dtype=frame_array.dtype
        )
        return np.concatenate([frame_array, padding], axis=0)


class ClampToBinary:
    """Turn event counts into 0/1 spikes.

    ToFrame ADDS UP every event that lands in the same pixel, polarity and time
    bin, so raw frame values go above 1 (measured max on N-MNIST at T=20: 8).
    Clamping makes the network's input actual spikes rather than counts.

    Applied after the disk cache, so switching it on or off does not invalidate
    the cache. Module-level class so DataLoader workers can pickle it.
    """

    def __call__(self, frame_array: np.ndarray) -> np.ndarray:
        return np.minimum(frame_array, 1)


def build_framing(dataset_cfg: dict[str, Any], sensor_size: tuple[int, int, int]):
    """Return (to_frame_transform, effective_T, framing_description).

    framing_description is the exact set of values that determine the frames.
    It names the cache directory and is written into the cache manifest, so a
    cache built with different settings can never be silently reused.
    """
    mode = require_choice(dataset_cfg, "framing.mode", sorted(FRAMING_MODES))

    if mode == "n_time_bins":
        bins = require_int(dataset_cfg, "framing.n_time_bins")
        if bins < 1:
            raise ConfigError(f"dataset.framing.n_time_bins must be >= 1, got {bins}")
        to_frame = tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=bins)
        # n_time_bins already guarantees exactly this many frames. The pad/crop
        # is applied anyway so that T is enforced in one single place.
        time_steps = bins
        description = {"mode": mode, "n_time_bins": bins, "frames": time_steps}

    else:  # time_window
        window = require_int(dataset_cfg, "framing.time_window_us")
        frames = require_int(dataset_cfg, "framing.frames")
        if window < 1:
            raise ConfigError(f"dataset.framing.time_window_us must be >= 1, got {window}")
        if frames < 1:
            raise ConfigError(f"dataset.framing.frames must be >= 1, got {frames}")
        # Timestamps in Tonic event data are in microseconds, so time_window is too.
        to_frame = tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=window)
        time_steps = frames
        description = {"mode": mode, "time_window_us": window, "frames": frames}

    return to_frame, time_steps, description


def cache_key(framing: dict[str, Any], dataset_options: dict[str, Any]) -> str:
    """Directory name encoding everything that changes the cached frames.

    Different framing settings therefore land in different directories
    automatically: both stay on disk and both stay reusable, and there is no way
    to accidentally read 20-bin frames when you asked for 30.
    """
    parts = [f"{key}-{value}" for key, value in sorted(framing.items())]
    parts += [f"{key}-{value}" for key, value in sorted(dataset_options.items())]
    return "_".join(parts)


def check_manifest(cache_dir: Path, expected: dict[str, Any]) -> bool:
    """Verify a cache directory was built with `expected` settings.

    Returns True if the cache already exists and matches. Raises if it exists
    but does not match, or exists without a manifest -- both mean the data on
    disk is not what the config asked for, and silently using it would corrupt
    the experiment.
    """
    manifest_path = cache_dir / "manifest.json"

    if not manifest_path.is_file():
        # An empty or absent directory is fine -- Tonic will fill it. A directory
        # with cached samples but no manifest is not: we cannot tell what it is.
        has_cached_files = cache_dir.is_dir() and any(cache_dir.glob("*.hdf5"))
        if has_cached_files:
            raise ConfigError(
                f"cache directory {cache_dir} contains cached samples but no "
                f"manifest.json, so the framing settings that produced it are "
                f"unknown. Delete the directory to rebuild it."
            )
        return False

    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if recorded != expected:
        raise ConfigError(
            f"cache at {cache_dir} was built with different settings.\n"
            f"  on disk: {recorded}\n"
            f"  config:  {expected}\n"
            f"Delete the directory to rebuild it, or point cache_dir elsewhere."
        )
    return True


def write_manifest(cache_dir: Path, settings: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "manifest.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )


def collate_time_first(batch: list[tuple[np.ndarray, int]]):
    """Stack samples into [T, batch, C, H, W] plus a label vector.

    Cast to float32 happens here rather than in the cached transform: the cache
    stores the compact integer frames Tonic produces, which keeps it small.
    """
    frames = torch.stack([torch.as_tensor(np.asarray(sample)) for sample, _ in batch])
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    # [batch, T, C, H, W] -> [T, batch, C, H, W]
    return frames.movedim(1, 0).float(), labels


def build_split(
    dataset_cfg: dict[str, Any],
    train: bool,
) -> tuple[Any, DataInfo]:
    """Build one cached split (train or test) and describe it.

    Returns the dataset ready for a DataLoader, plus the DataInfo describing it.
    """
    name = require_str(dataset_cfg, "name")
    if name not in DATASETS:
        raise ConfigError(
            f"unknown dataset.name '{name}'. Supported: {sorted(DATASETS)}"
        )
    entry = DATASETS[name]
    dataset_class = entry["class"]
    sensor_size = dataset_class.sensor_size  # (x, y, polarity)

    # Options that change the raw events themselves, so they belong in the cache key.
    dataset_options = {
        "first_saccade_only": require_bool(dataset_cfg, "first_saccade_only"),
        "stabilize": require_bool(dataset_cfg, "stabilize"),
    }

    to_frame, time_steps, framing = build_framing(dataset_cfg, sensor_size)

    # Denoise works on raw events, so it must run BEFORE framing. It is
    # expensive and it changes which events exist, so it is cached and it is
    # part of the cache identity.
    denoise_us = require_optional_int(dataset_cfg, "denoise_filter_time_us")
    stages: list[Callable] = []
    if denoise_us is not None:
        stages.append(tonic.transforms.Denoise(filter_time=denoise_us))
    stages.append(to_frame)
    stages.append(PadOrCropFrames(time_steps))
    cached_transform = tonic.transforms.Compose(stages)

    # Binarize works on frames and is a single elementwise clamp, so it runs
    # after the cache read. Flipping it therefore does not invalidate the cache
    # and is deliberately NOT part of the cache key.
    binarize = require_bool(dataset_cfg, "binarize")
    post_cache_transform = ClampToBinary() if binarize else None

    cache_identity = {**dataset_options, "denoise_filter_time_us": denoise_us}
    split_name = "train" if train else "test"
    cache_root = Path(require_str(dataset_cfg, "cache_dir")) / name / cache_key(
        framing, cache_identity
    )
    cache_dir = cache_root / split_name

    manifest = {"dataset": name, "split": split_name, "framing": framing,
                "dataset_options": dataset_options,
                "denoise_filter_time_us": denoise_us,
                "tonic": tonic.__version__}
    existed = check_manifest(cache_dir, manifest)
    print(
        f"  {split_name:<5} cache {cache_dir}  "
        f"{'REUSING (already built)' if existed else 'will be built on first pass'}"
    )

    raw = dataset_class(
        save_to=require_str(dataset_cfg, "root"),
        train=train,
        transform=cached_transform,
        **dataset_options,
    )
    cached = tonic.DiskCachedDataset(
        raw,
        cache_path=str(cache_dir),
        # Applied on the way OUT of the cache, not on the way in.
        transform=post_cache_transform,
        # Compression trades CPU for disk. Colab has plenty of disk and few CPU
        # cores, so false is usually faster there. Identical for all three
        # frameworks either way, so it cannot bias the comparison.
        compress=require_bool(dataset_cfg, "cache_compress"),
    )
    write_manifest(cache_dir, manifest)

    # x,y,p from Tonic; ToFrame emits [T, polarity, y, x], hence C,H,W below.
    width, height, channels = sensor_size
    info = DataInfo(
        name=name,
        time_steps=time_steps,
        channels=channels,
        height=height,
        width=width,
        num_classes=entry["num_classes"],
        cache_root=cache_root,
        framing=framing,
        denoise_filter_time_us=denoise_us,
        binarize=binarize,
    )
    return cached, info


def build_loader(
    dataset: Any,
    dataset_cfg: dict[str, Any],
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    """Wrap a cached split in a DataLoader.

    batch_size and num_workers come from the config and MUST be identical across
    frameworks -- num_workers changes wall-clock time, so a different value per
    framework makes the speed comparison meaningless.
    """
    generator = None
    if shuffle and seed is not None:
        # Own generator, so the batch order depends only on this seed and not on
        # whatever else in the process has drawn random numbers.
        generator = torch.Generator().manual_seed(seed)

    num_workers = require_int(dataset_cfg, "num_workers")
    return DataLoader(
        dataset,
        batch_size=require_int(dataset_cfg, "batch_size"),
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_time_first,
        generator=generator,
        # Only valid with worker processes; pinning speeds up the host->GPU copy.
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def prebuild_cache(dataset: Any, label: str, report_every: int = 2000) -> None:
    """Walk a cached dataset once so every sample is written to disk.

    Tonic caches lazily, one sample at a time, so without this the first epoch
    pays the whole conversion cost. Your doc calls this out: if framework #1
    builds the cache, it looks slow for a reason that has nothing to do with the
    framework.
    """
    total = len(dataset)
    print(f"  building {label} cache: {total} samples (one-time, slow)")
    for index in range(total):
        dataset[index]
        if (index + 1) % report_every == 0 or index + 1 == total:
            print(f"    {index + 1}/{total}")
