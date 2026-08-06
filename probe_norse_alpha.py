"""Evidence figure: Norse 1.1.0's SuperSpike surrogate ignores its `alpha`.

    python probe_norse_alpha.py --experiment ex2

SINGLE-PURPOSE MODULE, deliberately standalone. It imports nothing from the
training pipeline except `src.plots.style`, and only to keep its colours and
fonts consistent with the other figures. It reads no CSV, touches no config, and
nothing in `train.py` / `src/network.py` / `src/adapters/` imports it. Deleting
this file cannot break anything else.

WHY IT EXISTS SEPARATELY: Experiment 2 runs Norse on its default surrogate
(`method='super'`, `alpha=100`), and that default carries a released bug. The
experiment's Norse numbers therefore need a piece of evidence sitting next to
them showing exactly what the bug is. That evidence is a property of the library,
not of any run, so it does not belong in the results pipeline.

SAFE TO DELETE once Norse ships a release containing commit `1d2671a`. Until
then the figure it produces is expected in the final report.

WHAT IT MEASURES, not asserts: every curve here is obtained by running a tensor
through the real surrogate and calling `.backward()`. No gradient formula is
transcribed by hand, so the figure cannot disagree with the installed code.

Background: `local_docs/norse_superspike_alpha_finding.md`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.plots.style import (
    FAINT_GREY,
    GRID_GREY,
    OKABE_ITO,
    TEXT_GREY,
    apply_rcparams,
    provenance,
    save,
    title_block,
)

ALPHAS = [1.0, 10.0, 100.0]
X = np.linspace(-2.0, 2.0, 801)


# --- measurement --------------------------------------------------------------


def measured_gradient(surrogate_fn, alpha: float) -> np.ndarray:
    """Run x through a surrogate and read the gradient autograd actually produces.

    Measuring rather than re-deriving is the whole point: a hand-copied formula
    could be wrong in the same direction as the claim being tested.
    """
    x = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    out = surrogate_fn(x, alpha)
    out.sum().backward()
    return x.grad.detach().numpy().copy()


def norse_super(x: torch.Tensor, alpha: float) -> torch.Tensor:
    from norse.torch.functional.superspike import super_fn

    return super_fn(x, torch.tensor(alpha))


def norse_circ(x: torch.Tensor, alpha: float) -> torch.Tensor:
    from norse.torch.functional import threshold as norse_threshold

    return norse_threshold.threshold(x, "circ", torch.tensor(alpha))


def snntorch_atan(x: torch.Tensor, alpha: float) -> torch.Tensor:
    from snntorch import surrogate as snn_surrogate

    return snn_surrogate.atan(alpha=alpha)(x)


def documented_superspike(alpha: float) -> np.ndarray:
    """The SuperSpike gradient as PUBLISHED and as fixed upstream.

        1 / (alpha * |x| + 1)^2

    Source, both agreeing: Zenke & Ganguli (2018) section 3.3.2,
    doi:10.1162/neco_a_01086 -- and the current upstream Norse implementation,
    whose backward reads `grad_output / (alpha * torch.abs(inp) + 1.0).pow(2)`.
    The installed 1.1.0 backward is the same expression with `alpha` absent,
    which is why this curve has to be computed here rather than measured.
    """
    return 1.0 / np.power(alpha * np.abs(X) + 1.0, 2)


PROBE_DISTANCES = [0.0, 0.5, 1.0, 2.0]


def gradient_at_distances() -> dict[str, list[float]]:
    """Gradient value at fixed distances from threshold, for the three surrogates.

    Deliberately NOT a gradient norm through a synthetic layer. A made-up layer's
    weight scale decides how far its pre-activations sit from threshold, and the
    surrogates differ only in their TAILS -- so that number would say more about
    the arbitrary layer than about the surrogates. Reading the curves at stated
    distances is exact, reproducible, and cannot be tuned.

    The practical consequence on the REAL network is a separate measurement, made
    in Experiment 1 over 8 batches of conv1: a 6.03x gradient-norm ratio. That is
    the number to quote for real effect; these are the number to quote for why.
    """
    x = torch.tensor(PROBE_DISTANCES, dtype=torch.float32)

    def read(surrogate_fn, alpha: float) -> list[float]:
        xx = x.clone().requires_grad_(True)
        surrogate_fn(xx, alpha).sum().backward()
        return [round(float(v), 4) for v in xx.grad]

    return {
        "super(100) as shipped": read(norse_super, 100.0),
        "atan(2.0)": read(snntorch_atan, 2.0),
        "circ(0.5)": read(norse_circ, 0.5),
    }


# --- figure -------------------------------------------------------------------


def build_figure(out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    shipped = {a: measured_gradient(norse_super, a) for a in ALPHAS}
    intended = {a: documented_superspike(a) for a in ALPHAS}
    atan2 = measured_gradient(snntorch_atan, 2.0)
    circ05 = measured_gradient(norse_circ, 0.5)

    # The claim, as a number: largest disagreement between any two alphas.
    stack = np.stack(list(shipped.values()))
    worst_shipped = float(np.abs(stack.max(axis=0) - stack.min(axis=0)).max())
    intended_stack = np.stack(list(intended.values()))
    worst_intended = float(
        np.abs(intended_stack.max(axis=0) - intended_stack.min(axis=0)).max()
    )
    at_distance = gradient_at_distances()

    # Two rows: plots on top, their explanations underneath in their own axes.
    # Putting the text inside the plots buried the very curves it describes -- and
    # panel C's table is large enough that no in-axes placement avoids the data.
    fig = plt.figure(figsize=(15.0, 6.9))
    grid = fig.add_gridspec(2, 3, height_ratios=[2.5, 1.0], hspace=0.42, wspace=0.24)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    notes = [fig.add_subplot(grid[1, i]) for i in range(3)]
    for note_ax in notes:
        note_ax.axis("off")

    styles = [
        (OKABE_ITO["blue"], "-", 3.4),
        (OKABE_ITO["vermillion"], "--", 2.2),
        (OKABE_ITO["bluish_green"], ":", 1.6),
    ]

    # --- Panel A: what norse 1.1.0 actually gives you ---
    ax = axes[0]
    for (alpha, curve), (colour, dash, width) in zip(shipped.items(), styles):
        ax.plot(X, curve, color=colour, linestyle=dash, linewidth=width,
                label=f"alpha = {alpha:g}")
    ax.set_title("A.  What norse 1.1.0 gives you", fontsize=10.5, loc="left")
    ax.set_xlabel("v − threshold")
    ax.set_ylabel("surrogate gradient")
    ax.legend(loc="upper right", fontsize=9)
    notes[0].text(
        0.0, 1.0,
        "Three curves are plotted here.\nYou can see one.\n\n"
        f"Largest difference between any two alphas:  {worst_shipped:.2e}\n"
        "Not merely small — exactly zero. alpha is\nstored and then never read.",
        fontsize=8.8, va="top", ha="left", color=TEXT_GREY, linespacing=1.5,
        bbox=dict(facecolor="#FFF6E5", edgecolor=OKABE_ITO["orange"], linewidth=1.1, pad=6),
    )

    # --- Panel B: what it is supposed to give you ---
    ax = axes[1]
    for (alpha, curve), (colour, dash, width) in zip(intended.items(), styles):
        ax.plot(X, curve, color=colour, linestyle=dash, linewidth=width,
                label=f"alpha = {alpha:g}")
    ax.set_title("B.  What it is supposed to give you", fontsize=10.5, loc="left")
    ax.set_xlabel("v − threshold")
    ax.set_ylabel("surrogate gradient")
    ax.legend(loc="upper right", fontsize=9)
    notes[1].text(
        0.0, 1.0,
        "The same three alphas, using the published\nformula   1/(alpha·|x| + 1)²\n\n"
        f"Largest difference here:  {worst_intended:.2f}\n"
        "alpha is meant to control sharpness across\nalmost the whole range.",
        fontsize=8.8, va="top", ha="left", color=TEXT_GREY, linespacing=1.5,
        bbox=dict(facecolor="white", edgecolor=GRID_GREY, linewidth=0.9, pad=6),
    )

    # --- Panel C: so what? ---
    ax = axes[2]
    ax.plot(X, shipped[100.0], color=OKABE_ITO["bluish_green"], linestyle="-",
            linewidth=2.6, label="Norse default: super(100)\n→ behaves as alpha 1")
    ax.plot(X, atan2, color=OKABE_ITO["blue"], linestyle="--", linewidth=2.2,
            label="snnTorch & SpikingJelly:\natan(2.0)")
    ax.plot(X, circ05, color=OKABE_ITO["reddish_purple"], linestyle=":", linewidth=2.4,
            label="what ex1 used instead:\ncirc(0.5)")
    ax.set_title("C.  Why it matters for the comparison", fontsize=10.5, loc="left")
    ax.set_xlabel("v − threshold")
    ax.set_ylabel("surrogate gradient")
    ax.legend(loc="upper right", fontsize=8.2)
    shipped_vals = at_distance["super(100) as shipped"]
    atan_vals = at_distance["atan(2.0)"]
    circ_vals = at_distance["circ(0.5)"]
    rows = "\n".join(
        f"  {d:>4.1f}   {s:6.4f}  {a:6.4f}  {c:6.4f}   {s / a:4.2f}x"
        for d, s, a, c in zip(PROBE_DISTANCES, shipped_vals, atan_vals, circ_vals)
    )
    notes[2].text(
        0.0, 1.0,
        "All three agree AT threshold; they differ in the TAILS.\n"
        " |x|    super    atan    circ   ratio\n" + rows + "\n"
        "Most neurons sit far from threshold, so Norse's default\n"
        "takes much larger steps at the same learning rate.\n"
        "circ(0.5) tracks atan(2.0) closely — why ex1 chose it.",
        fontsize=7.8, va="top", ha="left", color=TEXT_GREY, linespacing=1.5,
        family="monospace",
        bbox=dict(facecolor="white", edgecolor=GRID_GREY, linewidth=0.9, pad=6),
    )

    for ax in axes:
        ax.axvline(0.0, color=FAINT_GREY, linewidth=0.9, alpha=0.7)
        ax.set_xlim(X[0], X[-1])
        ax.set_ylim(-0.03, 1.06)

    title_block(
        fig,
        "Norse 1.1.0: the SuperSpike surrogate ignores its alpha",
        "The surrogate gradient is what replaces the un-differentiable spike during backpropagation — it decides "
        "how strongly a neuron's weights get corrected. `alpha` is meant to control its sharpness. In the released "
        "version it does nothing, so Norse's default surrogate is permanently the widest, strongest-gradient one.",
    )
    provenance(
        fig,
        "Panels A and C measured by autograd on the installed norse 1.1.0 / snntorch 1.0.0 · "
        "Panel B is the published form (Zenke & Ganguli 2018, doi:10.1162/neco_a_01086), "
        "identical to the unreleased upstream fix (commit 1d2671a)",
    )
    fig.subplots_adjust(top=0.80, bottom=0.10, left=0.05, right=0.985)
    return save(fig, out_dir, "FX.1_norse_superspike_alpha")


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evidence figure: Norse 1.1.0's SuperSpike ignores alpha."
    )
    parser.add_argument(
        "--experiment",
        help="write into experiments/<exN>/figures/ (e.g. ex2)",
    )
    parser.add_argument(
        "--results-root", default="experiments", help="default: experiments"
    )
    parser.add_argument("--out-dir", help="write here instead")
    args = parser.parse_args(argv)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.experiment:
        out_dir = Path(args.results_root) / args.experiment / "figures"
    else:
        out_dir = Path("local_runs") / "figures"
        print("no --experiment given, writing to local_runs/figures/")

    apply_rcparams()
    written = build_figure(out_dir)

    print()
    print("The claim, verified against the installed package:")
    shipped = {a: measured_gradient(norse_super, a) for a in ALPHAS}
    reference = shipped[ALPHAS[0]]
    for alpha in ALPHAS:
        delta = float(np.abs(shipped[alpha] - reference).max())
        print(f"   super(alpha={alpha:>5g})   max |difference| vs alpha=1: {delta:.2e}")
    print()
    print("Gradient at fixed distances from threshold:")
    at_distance = gradient_at_distances()
    header = "   |x|      " + "".join(f"{d:>8.1f}" for d in PROBE_DISTANCES)
    print(header)
    for name, values in at_distance.items():
        print(f"   {name:<22s}" + "".join(f"{v:8.4f}" for v in values))
    print()
    print("   All three agree at threshold; they differ in the tails. Real-network")
    print("   effect was measured in ex1: 6.03x gradient norm over 8 batches of conv1.")
    print()
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
