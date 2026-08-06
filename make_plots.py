"""Draw every figure for one experiment, straight from its result CSVs.

Figures are never edited by hand. Everything here is regenerated from
`runs.csv` / `epochs.csv` / `layers.csv`, so a figure can always be traced back to
the rows that produced it, and re-running after a new seed arrives is a one-liner.

    python make_plots.py --experiment ex1
    python make_plots.py --experiment ex1 --formats png
    python make_plots.py --results-dir some/other/results --out-dir /tmp/figs

Design, sources and conventions: `local_docs/plotting_schema.md`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import output_dirs
from src.plots import apply_rcparams, load
from src.plots import figures as F
from src.plots.style import set_formats

# (family, function). Order is the order they are written and reported.
FAMILIES = [
    ("F0  fairness evidence", F.f0_start_state),
    ("F1  per-metric distribution", F.f1_overview),
    ("F1  per-metric distribution", F.f1_speed),
    ("F1  per-metric distribution", F.f1_energy),
    ("F2  paired within-seed views", F.f2_slopegraph),
    ("F2  paired within-seed views", F.f2_differences),
    ("F3  profile and trade-offs", F.f3_profile),
    ("F3  profile and trade-offs", F.f3_tradeoff),
    ("F4  training dynamics", F.f4_learning_curves),
    ("F4  training dynamics", F.f4_epoch_time),
    ("F4  training dynamics", F.f4_sparsification),
    ("F4  training dynamics", F.f4_loss),
    ("F5  structure", F.f5_layer_activity),
    ("F5  structure", F.f5_spike_budget),
    ("F6  measurement quality", F.f6_effect_vs_noise),
    ("F6  measurement quality", F.f6_idle_baseline),
    ("F6  measurement quality", F.f6_sensor),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the figure set for one experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        help="which experiment folder, e.g. ex1. Omit only with --results-dir.",
    )
    parser.add_argument(
        "--results-root",
        default="experiments",
        help="where experiment folders live (default: experiments)",
    )
    parser.add_argument(
        "--results-dir",
        help="read CSVs from here instead of deriving the path from --experiment",
    )
    parser.add_argument(
        "--out-dir",
        help="write figures here instead of <experiment>/figures",
    )
    parser.add_argument(
        "--condition",
        default="framework",
        help="the column being compared (default: framework). A later experiment "
        "comparing variants or datasets points this elsewhere.",
    )
    parser.add_argument(
        "--formats",
        default="png",
        help="comma-separated output formats (default: png). Add pdf for vector "
        "output, e.g. --formats png,pdf.",
    )
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Where to read from and where to write to.

    Paths come from the command line, never from a config file: a config describes
    the experiment's science, and the folder is chosen per run.
    """
    if args.results_dir:
        results_dir = Path(args.results_dir)
        out_dir = Path(args.out_dir) if args.out_dir else results_dir.parent / "figures"
        return results_dir, out_dir

    if not args.experiment:
        raise SystemExit("need --experiment (e.g. --experiment ex1) or --results-dir")

    results_dir, _equivalence_dir, _flat = output_dirs(args.experiment, args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir.parent / "figures"
    return results_dir, out_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir, out_dir = resolve_paths(args)
    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())

    try:
        results = load(results_dir, condition=args.condition)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        print(
            "hint: run collect_results.py first, or point --results-dir at the CSVs.",
            file=sys.stderr,
        )
        return 1

    apply_rcparams()
    set_formats(formats)

    print(f"reading  {results_dir}")
    print(f"writing  {out_dir}")
    print(
        f"{len(results.runs)} runs · {len(results.conditions)} conditions "
        f"({', '.join(results.conditions)}) · seeds {results.blocks}"
    )
    if len(results.blocks) < 2:
        print(
            "note: only one seed present — the paired figures (F2) and every ±SD "
            "interval need at least two, and will be skipped or empty."
        )
    print()

    written: list[Path] = []
    skipped: list[str] = []
    last_family = None

    for family, function in FAMILIES:
        if family != last_family:
            print(family)
            last_family = family
        try:
            paths = function(results, out_dir)
        except Exception as error:  # one bad figure must not lose the rest
            print(f"   FAILED  {function.__name__}: {error}")
            skipped.append(f"{function.__name__} ({error})")
            continue
        if not paths:
            print(f"   skipped {function.__name__} — required columns absent")
            skipped.append(f"{function.__name__} (no data)")
            continue
        written.extend(paths)
        print(f"   {paths[0].stem}")

    print()
    print(f"{len(written)} files written to {out_dir}")
    if skipped:
        print(f"{len(skipped)} figure(s) not produced:")
        for item in skipped:
            print(f"   - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
