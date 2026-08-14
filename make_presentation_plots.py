"""Cross-experiment figures, for the presentation.

Every figure in `make_plots.py` describes ONE experiment, because it loads one
results folder. This script is for the figures that only exist by putting two
experiments side by side -- which is the whole point of the forced-equivalent vs
out-of-the-box design, and therefore the argument the talk is built on.

    python make_presentation_plots.py --from ex1 --to ex2

Both arguments are required and have no defaults, for the same reason `--config`
has none anywhere else in this project: a figure comparing two experiments must
never be able to silently pick up the wrong pair.

Style, identities and provenance come from `src/plots/style.py` unchanged, so a
reader who learned that green-dotted-triangle means Norse in experiment 1 reads
these figures without relearning anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plots.data import METRICS, load, summary
from src.plots.style import (
    FAINT_GREY,
    TEXT_GREY,
    apply_rcparams,
    better_label,
    identity,
    provenance,
    save,
    set_formats,
    title_block,
    wrap,
)

# The two panels of FX.2. Accuracy is what a user notices; spike rate is what it
# would cost them on a chip. They move in opposite directions, which is the point.
PANELS = ("accuracy", "spike_rate")


def _endpoint_labels(ax, entries: list[tuple[float, str, str]]) -> None:
    """Write one label per line at its right-hand endpoint, without overlaps.

    Frameworks can finish within a hundredth of each other -- snnTorch and sinabs
    differ by 0.01 pp in ex2 -- so labels placed at the raw data value collide and
    become unreadable. Positions are therefore nudged apart to a minimum spacing
    expressed as a fraction of the axis range, and a thin leader line keeps each
    label attached to the point it belongs to.
    """
    low, high = ax.get_ylim()
    # Each label is two lines tall, so the gap has to clear a whole label rather
    # than a line: snnTorch and sinabs finish 0.01 pp apart in ex2 and would
    # otherwise print on top of each other.
    min_gap = 0.105 * (high - low)

    entries = sorted(entries, key=lambda item: item[0])
    positions = [value for value, _, _ in entries]
    for index in range(1, len(positions)):
        if positions[index] - positions[index - 1] < min_gap:
            positions[index] = positions[index - 1] + min_gap

    # Pushing labels up can drive the top one off the axes; shift the block back
    # down if that happened, so the spacing is preserved but stays inside.
    overshoot = positions[-1] - (high - 0.02 * (high - low))
    if overshoot > 0:
        positions = [p - overshoot for p in positions]

    for (value, text, colour), y in zip(entries, positions):
        ax.annotate(
            text,
            xy=(1.0, value),
            xytext=(1.09, y),
            textcoords=("axes fraction", "data"),
            va="center",
            ha="left",
            fontsize=8.5,
            color=colour,
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color=colour, lw=0.7, alpha=0.55),
        )


def fx2_cost_of_defaults(
    before, after, before_label: str, after_label: str, out_dir: Path
) -> list[Path]:
    """FX.2 -- what each framework's own defaults cost, per framework.

    A slopegraph across EXPERIMENTS rather than across seeds. Each framework is
    one line joining its forced-equivalent result to its out-of-the-box result,
    so the figure answers "what do I lose by not configuring this?" -- which is
    the usability question -- instead of "how do the frameworks differ?", which
    the neurons already guarantee.

    Both panels are drawn on the same grammar deliberately: the accuracy panel is
    nearly flat except for one collapse, and the spike-rate panel is the opposite
    shape. Reading them together is the finding.
    """
    conditions = [c for c in before.conditions if c in after.conditions]
    if not conditions:
        raise SystemExit("the two experiments share no frameworks")

    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.1 * len(PANELS), 4.9))
    axes = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes, PANELS):
        metric = METRICS[key]
        stats_before = summary(before, key).set_index("condition")
        stats_after = summary(after, key).set_index("condition")

        labels: list[tuple[float, str, str]] = []
        for condition in conditions:
            ident = identity(condition)
            start = float(stats_before.loc[condition, "mean"])
            end = float(stats_after.loc[condition, "mean"])
            spread = [
                float(stats_before.loc[condition, "std"]),
                float(stats_after.loc[condition, "std"]),
            ]
            ax.errorbar(
                [0.0, 1.0],
                [start, end],
                yerr=spread,
                color=ident.colour,
                marker=ident.marker,
                linestyle=ident.linestyle,
                markersize=7,
                markerfacecolor="white",
                markeredgewidth=1.6,
                linewidth=2.0,
                capsize=3,
                elinewidth=1.0,
                zorder=3,
            )
            delta = end - start
            labels.append(
                (end, f"{ident.label}\n{metric.fmt(end)}  ({delta:+.2f})", ident.colour)
            )

        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels([wrap(before_label, 18), wrap(after_label, 18)], fontsize=9)
        ax.set_xlim(-0.18, 1.05)
        ax.set_ylabel(
            better_label(metric.label, metric.unit, metric.better, compact=True),
            fontsize=9,
        )
        ax.set_title(
            wrap(metric.plain, 52),
            fontsize=8.5,
            fontweight="normal",
            color=TEXT_GREY,
            loc="left",
        )
        ax.grid(axis="x", visible=False)
        _endpoint_labels(ax, labels)

    title_block(
        fig,
        "FX.2  What each framework's own defaults cost you",
        "One line per framework, joining the SAME framework's two results: left, forced to compute one "
        "identical neuron; right, left at its own out-of-the-box settings. Only the neuron changed -- "
        "network, data, optimiser, seeds and hardware are identical. Accuracy is flat except for one "
        "collapse; spiking is the opposite shape.",
    )
    provenance(
        fig,
        f"{before_label}: {before.note()}   |   {after_label}: {after.note()}"
        "   ·   bars = ±1 SD across seeds   ·   ↑ higher is better, ↓ lower is better",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.86), w_pad=5.0)
    return save(fig, out_dir, "FX.2_cost_of_defaults")


# ---------------------------------------------------------------------------
# FX.3 -- the network diagram
#
# Read off `src/network.py`, not off snnTorch's tutorial: the two differ in one
# place that matters. Our LIF sits BEFORE the MaxPool, so pooling runs on spikes
# rather than on membrane voltages, and that is what makes the LIF layer sizes
# 10,800 and 3,872 rather than 2,700 and 800.
#
# (stage, detail, output shape, kind)
STAGES = [
    ("Input", "event frames", "2 x 34 x 34", "io"),
    ("Conv2d", "2 -> 12, 5x5", "12 x 30 x 30", "torch"),
    ("LIF", "10,800 neurons", "12 x 30 x 30", "lif"),
    ("MaxPool2d", "2x2, on spikes", "12 x 15 x 15", "torch"),
    ("Conv2d", "12 -> 32, 5x5", "32 x 11 x 11", "torch"),
    ("LIF", "3,872 neurons", "32 x 11 x 11", "lif"),
    ("MaxPool2d", "2x2, on spikes", "32 x 5 x 5", "torch"),
    ("Flatten", "", "800", "torch"),
    ("Linear", "800 -> 10", "10", "torch"),
    ("LIF", "10 neurons", "10", "lif"),
    ("Output", "spike counts", "10 classes", "io"),
]

FILL = {"io": "#F2F2F2", "torch": "#FFFFFF", "lif": "#D9EEF9"}
EDGE = {"io": FAINT_GREY, "torch": TEXT_GREY, "lif": "#56B4E9"}


def fx3_network(out_dir: Path, time_steps: int = 20, params: int = 18254) -> list[Path]:
    """FX.3 -- the shared network, drawn once.

    One diagram serves every experiment, because the architecture is the control:
    the same `SpikingNet` class is used by all four frameworks and only the LIF
    layer is swapped. That is the fact the figure has to make obvious, so the
    three swappable layers are the only coloured boxes on it.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    box_w, box_h, gap = 1.0, 0.62, 0.34
    pitch = box_w + gap
    fig, ax = plt.subplots(figsize=(0.92 * len(STAGES) * pitch, 3.5))

    for index, (name, detail, shape, kind) in enumerate(STAGES):
        x = index * pitch
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.0),
                box_w,
                box_h,
                boxstyle="round,pad=0.012,rounding_size=0.06",
                linewidth=1.5 if kind == "lif" else 1.0,
                edgecolor=EDGE[kind],
                facecolor=FILL[kind],
                zorder=2,
            )
        )
        ax.text(
            x + box_w / 2,
            box_h * 0.62,
            name,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold" if kind == "lif" else "normal",
            color=EDGE["lif"] if kind == "lif" else TEXT_GREY,
            zorder=3,
        )
        if detail:
            ax.text(
                x + box_w / 2, box_h * 0.26, detail, ha="center", va="center",
                fontsize=7.2, color=TEXT_GREY, zorder=3,
            )
        # Output shape below each box: the reader can follow the tensor shrinking.
        ax.text(
            x + box_w / 2, -0.20, shape, ha="center", va="center",
            fontsize=7.4, color=FAINT_GREY, family="monospace",
        )
        if index < len(STAGES) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + box_w, box_h / 2),
                    (x + box_w + gap, box_h / 2),
                    arrowstyle="-|>", mutation_scale=9,
                    color=FAINT_GREY, linewidth=0.9, zorder=1,
                )
            )

    span_end = (len(STAGES) - 1) * pitch + box_w
    ax.annotate(
        "",
        xy=(0, box_h + 0.30),
        xytext=(span_end, box_h + 0.30),
        arrowprops=dict(arrowstyle="<->", color=TEXT_GREY, lw=1.0),
    )
    ax.text(
        span_end / 2,
        box_h + 0.40,
        f"the whole chain runs once per timestep — {time_steps} times per sample",
        ha="center", va="bottom", fontsize=8.5, color=TEXT_GREY, style="italic",
    )

    ax.set_xlim(-0.35, span_end + 0.35)
    ax.set_ylim(-0.45, box_h + 0.75)
    ax.axis("off")

    title_block(
        fig,
        "FX.3  The network — identical in all four frameworks",
        "Only the shaded LIF layers are swapped between frameworks. Every convolution, pooling and linear "
        "layer is plain PyTorch and byte-identical, so nothing outside those three boxes can explain a "
        "difference between frameworks. Note the LIF sits BEFORE the pooling, so pooling runs on spikes.",
        subtitle_y=0.90,
    )
    provenance(
        fig,
        f"read from src/network.py · {params:,} trainable parameters "
        "(612 + 9,632 + 8,010) · none of them in the LIF layers · N-MNIST, 34x34, 2 polarities",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.80))
    return save(fig, out_dir, "FX.3_network")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the cross-experiment figures used in the presentation."
    )
    parser.add_argument(
        "--from",
        dest="before",
        required=True,
        help="baseline experiment folder, e.g. ex1 (forced-equivalent neuron)",
    )
    parser.add_argument(
        "--to",
        dest="after",
        required=True,
        help="comparison experiment folder, e.g. ex2 (out-of-the-box neuron)",
    )
    parser.add_argument(
        "--results-root",
        default="experiments",
        help="where experiment folders live (default: experiments)",
    )
    parser.add_argument(
        "--out-dir",
        help="where to write (default: the --to experiment's figures/ folder)",
    )
    parser.add_argument(
        "--before-label", default="Forced identical neuron", help="left-hand axis label"
    )
    parser.add_argument(
        "--after-label", default="Out of the box", help="right-hand axis label"
    )
    parser.add_argument("--formats", default="png", help="comma-separated, e.g. png,pdf")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.results_root)

    before = load(root / args.before / "results")
    after = load(root / args.after / "results")

    out_dir = Path(args.out_dir) if args.out_dir else root / args.after / "figures"

    apply_rcparams()
    set_formats(tuple(f.strip() for f in args.formats.split(",") if f.strip()))

    written = fx2_cost_of_defaults(
        before, after, args.before_label, args.after_label, out_dir
    )
    # The architecture is a control, not a result, so it is read from the source
    # rather than from either CSV -- but it is stamped with the run's own
    # timesteps and parameter count so a mismatch with the data is visible.
    time_steps = int(before.runs["time_steps"].iloc[0])
    params = int(before.runs["trainable_params"].iloc[0])
    written += fx3_network(out_dir, time_steps=time_steps, params=params)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
