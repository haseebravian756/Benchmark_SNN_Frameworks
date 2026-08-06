"""Park the built Tonic cache somewhere durable (Google Drive) and get it back.

THE PROBLEM. Building the cache is a one-time cost per machine -- Tonic converts
raw event lists into frames on first access, which is why `prepare_data.py`
exists at all. On a laptop that cost is genuinely paid once. On Colab it is paid
*every session*, because the local disk is wiped when the runtime disconnects,
and a disconnect is routine on the free tier. Drive survives the disconnect.

THE RULE THIS MODULE FOLLOWS. Identity is decided by exactly the same thing the
local cache already uses: the manifest dict built in `src.data.build_split`. An
archive is reused only when its recorded manifest is EQUAL to the one the current
config asks for, and the manifest that comes out of the archive is then handed to
`src.data.check_manifest` like any other. So this module cannot weaken the
existing guarantee. It can only skip work that would have produced the same
bytes -- or decline to, and let the normal path rebuild.

WHY ONE TAR PER SPLIT, NOT A FOLDER COPY. Google Drive's FUSE mount is fine with
a few large files and pathological with many small ones, and one N-MNIST split is
60,000 separate `.hdf5` files. Copying those one at a time over Drive can take
longer than rebuilding the cache from scratch, which would defeat the whole
point. One sequential tar does not have that problem.

LAYOUT inside the archive folder:

    <archive_root>/
        nmnist__t20_dn10000_9f3a2b__train.tar
        nmnist__t20_dn10000_9f3a2b__train.json     <- sidecar
        nmnist__t20_dn10000_9f3a2b__test.tar
        nmnist__t20_dn10000_9f3a2b__test.json

The stem is `<dataset>__<cache_key>__<split>`, and `<cache_key>` is the very same
string that names the local cache directory, so a local cache folder and its
archive are recognisably the same thing at a glance.

THE SIDECAR is the reason this is fast to check. It holds the manifest plus the
sample count, so "do I already have this?" is a few hundred bytes read off Drive
rather than a multi-gigabyte tar. It is written AFTER the tar lands, never
before: that ordering means a half-written tar has no sidecar, and a tar with no
sidecar is invisible to `restore`. An interrupted upload therefore fails safe.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

# Uncompressed first: it is what `store` writes by default, so it is the common
# case. `.tar.gz` is still restorable if you chose to compress.
SUFFIXES = [".tar", ".tar.gz"]

MODES = {".tar": "w", ".tar.gz": "w:gz"}


def archive_stem(dataset_name: str, cache_key: str, split: str) -> str:
    """The shared name of one split's archive and its sidecar."""
    return f"{dataset_name}__{cache_key}__{split}"


def _sidecar_path(root: Path, stem: str) -> Path:
    return root / f"{stem}.json"


def _count_samples(directory: Path) -> int:
    """How many cached samples are actually on disk.

    Tonic's DiskCachedDataset writes one `.hdf5` per sample, so this is the count
    that tells a complete cache from a half-built one.
    """
    return sum(1 for _ in directory.glob("*.hdf5"))


def read_sidecar(root: Path, stem: str) -> dict[str, Any] | None:
    """The sidecar's contents, or None if it is missing or unreadable.

    Unreadable is treated as missing on purpose: a corrupt sidecar should send us
    down the rebuild path, not raise. The cost of being wrong here is time, and
    the alternative -- crashing a Colab session over a truncated JSON file -- is
    worse.
    """
    path = _sidecar_path(root, stem)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def find_archive(
    root: Path, stem: str, manifest: dict[str, Any]
) -> tuple[Path, dict[str, Any]] | None:
    """Locate a usable archive for `manifest`, or None.

    "Usable" means: the sidecar exists, its manifest is EQUAL to the one asked
    for, and the tar it describes is actually present. A manifest that differs in
    any field is not a near-miss to be tolerated -- it is a different cache, and
    the caller must rebuild.
    """
    sidecar = read_sidecar(root, stem)
    if sidecar is None:
        return None
    if sidecar.get("manifest") != manifest:
        return None

    for suffix in SUFFIXES:
        candidate = root / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate, sidecar
    return None


def restore(
    root: Path,
    stem: str,
    cache_dir: Path,
    manifest: dict[str, Any],
) -> bool:
    """Unpack an archived split into `cache_dir`. True if it now holds that cache.

    Returns False -- never raises -- whenever anything is missing, mismatched or
    broken, because every False here simply means "build it the normal way". The
    only outcome this must never produce is a `cache_dir` that looks valid but
    is not, which is why the extracted manifest and sample count are both checked
    before the directory is moved into place.
    """
    if not root.is_dir():
        print(f"  archive  {root} not reachable -- skipping restore")
        return False

    # Never unpack on top of existing samples. If something is already there,
    # the normal path's own checks are the right ones to run against it.
    if cache_dir.is_dir() and _count_samples(cache_dir) > 0:
        return False

    found = find_archive(root, stem, manifest)
    if found is None:
        print(f"  archive  no match for {stem} in {root}")
        return False
    tar_path, sidecar = found

    size_mb = tar_path.stat().st_size / 1024**2
    expected = sidecar.get("sample_count")
    print(f"  archive  found {tar_path.name} ({size_mb:.0f} MB, "
          f"{expected} samples) -- restoring")

    # Unpack beside the target, then move into place, so an interrupted restore
    # can never leave a partial cache under the real name.
    staging = cache_dir.parent / f"{cache_dir.name}.restoring"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    try:
        with tarfile.open(tar_path, "r:*") as archive:
            # `filter="data"` refuses absolute paths, `..` and special files.
            # We wrote these archives ourselves, so this is belt and braces --
            # but the belt is one keyword long.
            if hasattr(tarfile, "data_filter"):
                archive.extractall(staging, filter="data")
            else:  # pragma: no cover - only on Python < 3.11.4
                archive.extractall(staging)
    except (tarfile.TarError, OSError) as error:
        print(f"  archive  RESTORE FAILED ({error}) -- will build normally")
        shutil.rmtree(staging, ignore_errors=True)
        return False

    # Two checks before this is allowed to become the real cache. Both are the
    # same questions the local path asks; neither is new policy.
    restored_manifest_path = staging / "manifest.json"
    if not restored_manifest_path.is_file():
        print("  archive  RESTORE REJECTED: no manifest.json inside the archive")
        shutil.rmtree(staging, ignore_errors=True)
        return False
    try:
        restored_manifest = json.loads(restored_manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print("  archive  RESTORE REJECTED: manifest.json inside the archive is unreadable")
        shutil.rmtree(staging, ignore_errors=True)
        return False
    if restored_manifest != manifest:
        print("  archive  RESTORE REJECTED: the manifest inside the archive "
              "disagrees with its own sidecar")
        shutil.rmtree(staging, ignore_errors=True)
        return False

    actual = _count_samples(staging)
    if expected is not None and actual != expected:
        print(f"  archive  RESTORE REJECTED: {actual} samples unpacked, "
              f"sidecar says {expected}")
        shutil.rmtree(staging, ignore_errors=True)
        return False

    # An empty cache_dir may already exist (a previous run that got as far as
    # writing a manifest). Clear it so the move lands cleanly.
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging), str(cache_dir))

    elapsed = time.perf_counter() - started
    print(f"  archive  RESTORED {actual} samples in {elapsed:.0f}s "
          f"-- no rebuild needed")
    return True


def store(
    root: Path,
    stem: str,
    cache_dir: Path,
    manifest: dict[str, Any],
    expected_samples: int,
    compress: bool = False,
    overwrite: bool = False,
) -> Path | None:
    """Archive a COMPLETE split into `root`. Returns the archive path, or None.

    Refuses to archive a partial cache. That refusal is the important part of
    this function: an incomplete archive would be restored happily by every later
    session and would silently train on a subset of the data, which is the kind
    of bug that never announces itself.
    """
    if not cache_dir.is_dir():
        print(f"  archive  nothing to store: {cache_dir} does not exist")
        return None

    actual = _count_samples(cache_dir)
    if actual != expected_samples:
        print(f"  archive  NOT STORING: cache holds {actual} samples but the split "
              f"has {expected_samples}. Run without --no-prebuild first.")
        return None
    if not (cache_dir / "manifest.json").is_file():
        print(f"  archive  NOT STORING: {cache_dir} has no manifest.json")
        return None

    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"  archive  NOT STORING: cannot create {root} ({error})")
        return None

    if not overwrite and find_archive(root, stem, manifest) is not None:
        print(f"  archive  already stored: {stem} -- nothing to do")
        return None

    suffix = ".tar.gz" if compress else ".tar"
    final = root / f"{stem}{suffix}"

    # Stage the tar locally, not on Drive. Writing a multi-gigabyte file straight
    # into a FUSE mount and then having it fail leaves rubbish on Drive that the
    # next session has to notice; building it on local disk and copying the
    # finished file across is both faster and easier to clean up after.
    staging_dir = Path(tempfile.mkdtemp(prefix="snn_cache_archive_"))
    staging_tar = staging_dir / f"{stem}{suffix}"

    started = time.perf_counter()
    try:
        print(f"  archive  packing {actual} samples -> {final.name}"
              f"{' (gzip)' if compress else ''}")
        with tarfile.open(staging_tar, MODES[suffix]) as archive:
            # arcname="." keeps the archive flat: its members are the split's own
            # files, so it can be unpacked into any directory.
            archive.add(cache_dir, arcname=".")

        size_mb = staging_tar.stat().st_size / 1024**2
        print(f"  archive  copying {size_mb:.0f} MB to {root}")
        shutil.copy2(staging_tar, final)
    except OSError as error:
        print(f"  archive  STORE FAILED ({error}) -- the local cache is unaffected")
        _remove_quietly(final)
        shutil.rmtree(staging_dir, ignore_errors=True)
        return None
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    # Sidecar LAST. Until this line runs the tar is invisible to `restore`, so an
    # upload that dies half way through cannot be mistaken for a good archive.
    sidecar = {
        "stem": stem,
        "manifest": manifest,
        "sample_count": actual,
        "archive": final.name,
        "bytes": final.stat().st_size,
    }
    try:
        _sidecar_path(root, stem).write_text(
            json.dumps(sidecar, indent=2), encoding="utf-8"
        )
    except OSError as error:
        print(f"  archive  STORE FAILED writing the sidecar ({error}) -- removing "
              f"the tar so it is not half-registered")
        _remove_quietly(final)
        return None

    elapsed = time.perf_counter() - started
    print(f"  archive  STORED {final} in {elapsed:.0f}s")
    return final


def _remove_quietly(path: Path) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
