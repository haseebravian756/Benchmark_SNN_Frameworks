"""Pull Colab results into an experiment folder, merging instead of overwriting.

The round trip is: Colab writes to Google Drive -> Drive syncs (or you download
the folder once) -> this script files everything into experiments/exN/.

Merging matters. If a Colab session restarts, the new session begins a fresh
`runs.csv` containing only its own rows. Copying that over the local file would
silently delete earlier runs. This script concatenates instead, keyed on
`run_id`, so re-running it is always safe and never duplicates a row.

    python collect_results.py --from "G:/My Drive/snn_results/ex1" --experiment ex1
    python collect_results.py --from temp_colab --experiment ex1     # manual download
    python collect_results.py --from temp_colab --experiment ex1 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

CSV_FILES = ["runs.csv", "epochs.csv", "layers.csv"]


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file() or path.stat().st_size == 0:
        return [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def row_key(row: dict[str, str], filename: str) -> tuple:
    """What makes a row unique, per file."""
    if filename == "runs.csv":
        return (row.get("run_id", ""),)
    if filename == "epochs.csv":
        return (row.get("run_id", ""), row.get("epoch", ""))
    return (row.get("run_id", ""), row.get("layer_index", ""))


def merge_csv(source: Path, target: Path, dry_run: bool) -> str:
    src_header, src_rows = read_rows(source)
    if not src_rows:
        return "nothing to merge"

    dst_header, dst_rows = read_rows(target)

    if dst_header and dst_header != src_header:
        raise SystemExit(
            f"\nERROR: {target.name} has a different schema on each side, so "
            f"merging would misalign rows.\n"
            f"  local  : {len(dst_header)} columns\n"
            f"  incoming: {len(src_header)} columns\n"
            f"They were probably written by different code versions. Rename the "
            f"local file to keep it, then re-run."
        )

    seen = {row_key(row, source.name) for row in dst_rows}
    added = [row for row in src_rows if row_key(row, source.name) not in seen]

    if not dry_run and added:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_header = not dst_rows
        with target.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=src_header)
            if write_header:
                writer.writeheader()
            writer.writerows(added)

    skipped = len(src_rows) - len(added)
    return (f"+{len(added)} new row(s)"
            + (f", {skipped} already present" if skipped else ""))


def copy_files(sources: list[Path], target_dir: Path, dry_run: bool) -> int:
    copied = 0
    for path in sources:
        destination = target_dir / path.name
        if destination.exists() and destination.stat().st_size == path.stat().st_size:
            continue
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        copied += 1
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="source", required=True,
                        help="folder holding the Colab output (Drive folder, or a download)")
    parser.add_argument("--experiment", required=True,
                        help="target experiment folder name, e.g. ex1")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would happen, change nothing")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"source folder not found: {source.resolve()}")

    target = Path("experiments") / args.experiment
    print(f"from : {source.resolve()}")
    print(f"to   : {target.resolve()}")
    if args.dry_run:
        print("DRY RUN - nothing will be written")
    print()

    # The Colab side may be laid out as <src>/results/... or flat, depending on
    # whether it came from Drive or a manual download. Accept both.
    results_src = source / "results" if (source / "results").is_dir() else source
    equiv_src = source / "equivalence" if (source / "equivalence").is_dir() else source

    print("CSV rows")
    for name in CSV_FILES:
        candidate = results_src / name
        if not candidate.is_file():
            print(f"  {name:<12} not found in source")
            continue
        print(f"  {name:<12} {merge_csv(candidate, target / 'results' / name, args.dry_run)}")

    runs_src = results_src / "runs"
    per_run = sorted(runs_src.glob("*.json")) if runs_src.is_dir() else \
        sorted(p for p in results_src.glob("*_seed*.json"))
    print(f"\nper-run JSON  {copy_files(per_run, target / 'results' / 'runs', args.dry_run)} "
          f"copied of {len(per_run)} found")

    equiv = sorted(p for p in equiv_src.glob("equivalence_*"))
    print(f"equivalence   {copy_files(equiv, target / 'equivalence', args.dry_run)} "
          f"copied of {len(equiv)} found")

    if not args.dry_run:
        runs_csv = target / "results" / "runs.csv"
        _, rows = read_rows(runs_csv)
        print(f"\n{runs_csv} now holds {len(rows)} run(s):")
        for row in rows:
            print(f"  {row.get('framework','?'):<14} seed {row.get('seed','?'):<3} "
                  f"acc {row.get('test_accuracy_pct','?'):<8} {row.get('run_id','')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
