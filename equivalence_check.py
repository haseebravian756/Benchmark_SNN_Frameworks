"""Prove that the three frameworks' LIF neurons behave identically.

Builds ONE leaky integrate-and-fire neuron in snnTorch, SpikingJelly and Norse
using the per-framework parameters in the config, feeds all three the exact
same input current, and compares what comes out.

There is no network, no dataset and no training here -- just one neuron each,
so that if the traces disagree the cause can only be the parameter translation.

    python equivalence_check.py --config config/default.yaml

Exit code is 0 if every test passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib

# Write PNGs without needing a display -- required on Colab/Kaggle.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.config import (  # noqa: E402
    ConfigError,
    load_config,
    require,
    require_bool,
    require_choice,
    require_float,
    require_int,
    require_str,
)

# The order these appear in every table, plot legend and comparison.
FRAMEWORKS = ["snntorch", "spikingjelly", "norse"]

# A threshold no membrane in this experiment can reach. Used for the "shadow"
# neurons described in the runners below: a neuron that can never fire also
# never resets, so it reports the raw charge value.
NEVER_FIRES_THRESHOLD = 1.0e9


class Trace(NamedTuple):
    """What one framework produced, per timestep.

    v_pre  -- membrane after charging but BEFORE any reset. This is the value
              that gets compared against the threshold, so it is the one that
              visibly crosses the line on a plot.
    v_post -- membrane after the reset, i.e. the state actually carried into
              the next timestep. Right after a spike this is 0.
    spikes -- 1.0 where the neuron fired, else 0.0.
    """

    v_pre: torch.Tensor
    v_post: torch.Tensor
    spikes: torch.Tensor


# ---------------------------------------------------------------------------
# Versions -- recorded in the results file because framework benchmarks are
# version-sensitive and these numbers belong in the write-up.
# ---------------------------------------------------------------------------


def framework_versions() -> dict[str, str]:
    import norse
    import snntorch

    return {
        "torch": torch.__version__,
        "snntorch": snntorch.__version__,
        "norse": norse.__version__,
        # spikingjelly exposes no __version__ attribute; ask pip instead.
        "spikingjelly": package_version("spikingjelly"),
    }


# ---------------------------------------------------------------------------
# Input currents. Every framework receives the identical tensor, so any
# difference in the output is the neuron's doing, not the input's.
# ---------------------------------------------------------------------------


def make_input(spec: dict[str, Any], steps: int) -> torch.Tensor:
    """Build a 1-D current of length `steps` from one entry of equivalence.inputs."""
    if "name" not in spec:
        raise ConfigError(f"each entry of equivalence.inputs needs a 'name': {spec!r}")
    name = spec["name"]

    if name == "constant_step":
        onset = int(spec["onset"])
        amplitude = float(spec["amplitude"])
        current = torch.zeros(steps)
        current[onset:] = amplitude
        return current

    if name == "poisson":
        rate = float(spec["rate"])
        amplitude = float(spec["amplitude"])
        # A dedicated generator, so this input is identical no matter what else
        # in the process has consumed random numbers.
        generator = torch.Generator().manual_seed(int(spec["seed"]))
        spikes = (torch.rand(steps, generator=generator) < rate).float()
        return spikes * amplitude

    raise ConfigError(
        f"unknown equivalence input name '{name}'. "
        f"Supported: constant_step, poisson"
    )


# ---------------------------------------------------------------------------
# One runner per framework.
#
# Each takes the identical `current` and returns a Trace of length T.
#
# Getting v_pre (the value before the reset) needs a trick, because all three
# libraries only hand back the post-reset state. Each runner therefore builds a
# SHADOW neuron: same parameters, but a threshold it can never reach. A neuron
# that never fires never resets, so its charge step is untouched. Re-seeding the
# shadow from the real neuron's state at every timestep and stepping it once
# yields the pre-reset value -- computed by that framework's own code, which is
# the whole point. Re-deriving the equation here would prove nothing.
# ---------------------------------------------------------------------------


def run_snntorch(neuron_cfg: dict[str, Any], current: torch.Tensor) -> Trace:
    # Same builder the training adapter uses, so snnTorch's parameters are read
    # in exactly one place and the two cannot drift apart.
    from src.adapters.snntorch_lif import build_leaky

    real = build_leaky(neuron_cfg)
    shadow = build_leaky(neuron_cfg, threshold=NEVER_FIRES_THRESHOLD)

    membrane = torch.zeros(1)
    v_pre, v_post, spikes = [], [], []
    for value in current:
        step_input = value.reshape(1)

        _, charged = shadow(step_input, membrane.clone())
        spike, membrane = real(step_input, membrane)

        v_pre.append(charged.reshape(()))
        v_post.append(membrane.reshape(()))
        spikes.append(spike.reshape(()))

    return Trace(torch.stack(v_pre), torch.stack(v_post), torch.stack(spikes))


def run_spikingjelly(neuron_cfg: dict[str, Any], current: torch.Tensor) -> Trace:
    from spikingjelly.activation_based import functional

    # Same builder the training adapter uses, so SpikingJelly's parameters are
    # read in exactly one place and the two cannot drift apart.
    from src.adapters.spikingjelly_lif import build_lif_node

    real = build_lif_node(neuron_cfg)
    shadow = build_lif_node(neuron_cfg, v_threshold=NEVER_FIRES_THRESHOLD)
    functional.reset_net(real)
    functional.reset_net(shadow)

    membrane = torch.zeros(1)
    v_pre, v_post, spikes = [], [], []
    for value in current:
        step_input = value.reshape(1)

        # SpikingJelly keeps state on the module, so seed it by assignment.
        shadow.v = membrane.clone()
        shadow(step_input)
        charged = shadow.v.clone()

        real.v = membrane.clone()
        spike = real(step_input)
        membrane = real.v.clone()

        v_pre.append(charged.reshape(()))
        v_post.append(membrane.reshape(()))
        spikes.append(spike.reshape(()))

    return Trace(torch.stack(v_pre), torch.stack(v_post), torch.stack(spikes))


def run_norse(neuron_cfg: dict[str, Any], current: torch.Tensor) -> Trace:
    from norse.torch.functional.lif_box import LIFBoxFeedForwardState

    # Same builder the training adapter uses, so Norse's parameters are read in
    # exactly one place and the two cannot drift apart.
    from src.adapters.norse_lif import build_lif_box_cell
    from src.adapters.norse_lif import input_scale as norse_input_scale

    real = build_lif_box_cell(neuron_cfg)
    shadow = build_lif_box_cell(neuron_cfg, v_th=NEVER_FIRES_THRESHOLD)

    # Norse's update is v = (1 - dt*tau_mem_inv)*v + (dt*tau_mem_inv)*input, so
    # its input gain is locked to dt*tau_mem_inv (0.1 here). Scaling the input
    # up by 1/0.1 = 10 restores an effective gain of 1, matching the other two.
    input_scale = norse_input_scale(neuron_cfg)

    membrane = torch.zeros(1)
    v_pre, v_post, spikes = [], [], []
    for value in current:
        step_input = (value * input_scale).reshape(1)

        _, charged_state = shadow(step_input, LIFBoxFeedForwardState(v=membrane.clone()))
        spike, state = real(step_input, LIFBoxFeedForwardState(v=membrane))
        membrane = state.v

        v_pre.append(charged_state.v.reshape(()))
        v_post.append(membrane.reshape(()))
        spikes.append(spike.reshape(()))

    return Trace(torch.stack(v_pre), torch.stack(v_post), torch.stack(spikes))


RUNNERS = {
    "snntorch": run_snntorch,
    "spikingjelly": run_spikingjelly,
    "norse": run_norse,
}


# ---------------------------------------------------------------------------
# Threshold boundary probe
# ---------------------------------------------------------------------------


def probe_threshold_boundary(
    neuron_cfg: dict[str, Any], thresholds: dict[str, float]
) -> dict[str, bool]:
    """Does each framework fire when the membrane lands EXACTLY on threshold?

    One timestep, one input equal to the threshold, onto a fresh membrane --
    so v ends up exactly at the threshold and nothing else can interfere.

    This is INFORMATION, not a pass/fail criterion: the comparison operator is
    hard-coded in each library and no config value can change it.
    """
    fires: dict[str, bool] = {}
    for name in FRAMEWORKS:
        single_step = torch.tensor([thresholds[name]])
        fires[name] = bool(RUNNERS[name](neuron_cfg, single_step).spikes[0].item() > 0)
    return fires


def print_threshold_boundary(fires: dict[str, bool], thresholds: dict[str, float]) -> None:
    print()
    print("-" * 68)
    print("threshold boundary behaviour  (information only, not pass/fail)")
    print("-" * 68)
    print("  with the membrane landing exactly ON the threshold:")
    for name in FRAMEWORKS:
        verdict = (
            "FIRES     -> rule is  v >= threshold" if fires[name]
            else "no spike  -> rule is  v >  threshold"
        )
        print(f"    {name:<14} v = {thresholds[name]:<6} {verdict}")

    if len(set(fires.values())) > 1:
        disagreeing = [n for n in FRAMEWORKS if fires[n]]
        print()
        print(f"  NOTE: {', '.join(disagreeing)} use >= while the others use >.")
        print("  This is hard-coded in each library and cannot be configured away.")
        print("  It only bites when the membrane lands exactly on the threshold, which")
        print("  is why equivalence input amplitudes should not equal the threshold.")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare(
    traces: dict[str, Trace],
    max_v_deviation: float,
    min_spike_time_match: float,
) -> dict[str, Any]:
    """Compare every pair of frameworks and decide PASS/FAIL.

    Both membrane traces are checked. v_post alone would miss nothing in
    practice, but reporting both makes it obvious whether a disagreement came
    from the charging arithmetic (v_pre) or from the reset (v_post).
    """
    pairs: list[dict[str, Any]] = []
    worst_deviation = 0.0
    worst_spike_match = 1.0

    for index, first in enumerate(FRAMEWORKS):
        for second in FRAMEWORKS[index + 1:]:
            a, b = traces[first], traces[second]

            deviation_pre = (a.v_pre - b.v_pre).abs().max().item()
            deviation_post = (a.v_post - b.v_post).abs().max().item()
            spike_match = (a.spikes == b.spikes).float().mean().item()

            worst_deviation = max(worst_deviation, deviation_pre, deviation_post)
            worst_spike_match = min(worst_spike_match, spike_match)
            pairs.append(
                {
                    "a": first,
                    "b": second,
                    "max_v_pre_deviation": deviation_pre,
                    "max_v_post_deviation": deviation_post,
                    "spike_time_match": spike_match,
                }
            )

    per_framework = {}
    for name, trace in traces.items():
        firing_steps = torch.nonzero(trace.spikes).flatten()
        per_framework[name] = {
            "total_spikes": int(trace.spikes.sum().item()),
            "first_spike_step": int(firing_steps[0].item()) if len(firing_steps) else None,
            "peak_v_pre": trace.v_pre.max().item(),
        }

    passed = worst_deviation <= max_v_deviation and worst_spike_match >= min_spike_time_match
    return {
        "per_framework": per_framework,
        "pairs": pairs,
        "worst_max_v_deviation": worst_deviation,
        "worst_spike_time_match": worst_spike_match,
        "passed": passed,
    }


def print_report(test_name: str, result: dict[str, Any], tolerance: dict[str, float]) -> None:
    print()
    print("=" * 68)
    print(f"TEST: {test_name}")
    print("=" * 68)

    print(f"{'framework':<14}{'total spikes':>14}{'first spike step':>18}{'peak v (pre-reset)':>22}")
    for name in FRAMEWORKS:
        stats = result["per_framework"][name]
        first = "never" if stats["first_spike_step"] is None else stats["first_spike_step"]
        print(
            f"{name:<14}{stats['total_spikes']:>14}{str(first):>18}"
            f"{stats['peak_v_pre']:>22.6f}"
        )

    print()
    print("pairwise comparison:")
    for pair in result["pairs"]:
        print(
            f"  {pair['a']:<13} vs {pair['b']:<13} "
            f"max|dv| pre = {pair['max_v_pre_deviation']:.3e}   "
            f"post = {pair['max_v_post_deviation']:.3e}   "
            f"spikes match = {pair['spike_time_match'] * 100:.1f}%"
        )

    print()
    print(
        f"  worst max|dv|            {result['worst_max_v_deviation']:.3e}   "
        f"(tolerance {tolerance['max_v_deviation']:.1e})"
    )
    print(
        f"  worst spike time match   {result['worst_spike_time_match'] * 100:.1f}%   "
        f"(required {tolerance['min_spike_time_match'] * 100:.1f}%)"
    )
    print()
    print(f"  RESULT: {'PASS' if result['passed'] else 'FAIL'}")


def print_divergence(
    current: torch.Tensor,
    traces: dict[str, Trace],
    max_v_deviation: float,
    context_steps: int = 6,
) -> None:
    """Show the first timestep where the frameworks stop agreeing.

    A bare FAIL tells you something is wrong; this tells you *where*, which is
    normally enough to identify which of the four usual causes it is: input
    gain, reset type, decay factor, or an off-by-one in when the reset lands.
    """
    stacked_pre = torch.stack([traces[name].v_pre for name in FRAMEWORKS])
    stacked_post = torch.stack([traces[name].v_post for name in FRAMEWORKS])
    stacked_spikes = torch.stack([traces[name].spikes for name in FRAMEWORKS])

    # At each timestep: how far apart are the most-distant two frameworks?
    def spread(stack: torch.Tensor) -> torch.Tensor:
        return stack.max(dim=0).values - stack.min(dim=0).values

    spikes_differ = stacked_spikes.max(dim=0).values != stacked_spikes.min(dim=0).values
    bad_steps = torch.nonzero(
        (spread(stacked_pre) > max_v_deviation)
        | (spread(stacked_post) > max_v_deviation)
        | spikes_differ
    ).flatten()
    if len(bad_steps) == 0:
        return

    first = int(bad_steps[0].item())
    start = max(0, first - 2)
    stop = min(len(current), first + context_steps)

    print()
    print(f"  first divergence at timestep {first}. Steps {start}-{stop - 1}:")
    header = f"    {'t':>3} {'input':>8}"
    for name in FRAMEWORKS:
        header += f" | {name[:10] + ' pre':>15} {'post':>10} {'spk':>4}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for t in range(start, stop):
        row = f"    {t:>3} {current[t].item():>8.4f}"
        for index in range(len(FRAMEWORKS)):
            row += (
                f" | {stacked_pre[index, t].item():>15.6f}"
                f" {stacked_post[index, t].item():>10.6f}"
                f" {stacked_spikes[index, t].item():>4.0f}"
            )
        print(row + ("   <-- here" if t == first else ""))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

# Deliberately different line widths and dash patterns: when the three traces
# agree they sit exactly on top of each other, and the only way to see that all
# three are present is for the thinner ones to draw over the thicker ones.
PLOT_STYLE = {
    "snntorch": {"color": "#1f77b4", "linewidth": 4.5, "linestyle": "-", "alpha": 0.45},
    "spikingjelly": {"color": "#d62728", "linewidth": 2.2, "linestyle": "--"},
    "norse": {"color": "#2ca02c", "linewidth": 1.2, "linestyle": ":"},
}

MEMBRANE_LABEL = {
    "pre_reset": "membrane potential\n(before reset)",
    "post_reset": "membrane potential\n(after reset)",
}


def metadata_lines(
    neuron_cfg: dict[str, Any], spec: dict[str, Any], steps: int, current: torch.Tensor
) -> list[str]:
    """The neuron settings that produced this figure, for the caption block.

    A saved plot is worthless six weeks later if you cannot tell which
    parameters produced it, so every value that shapes the run is printed on
    the figure itself.
    """
    snn_cfg = neuron_cfg["snntorch"]
    sj_cfg = neuron_cfg["spikingjelly"]
    norse_cfg = neuron_cfg["norse"]

    # Derived, and shown for reading convenience only -- nothing in the code
    # consumes these. Each framework is still driven purely by its own block.
    snn_decay, snn_gain = snn_cfg["beta"], 1.0
    sj_decay, sj_gain = 1.0 - 1.0 / sj_cfg["tau"], 1.0
    norse_step = norse_cfg["dt"] * norse_cfg["tau_mem_inv"]
    norse_decay, norse_gain = 1.0 - norse_step, norse_step * norse_cfg["input_scale"]

    input_bits = " ".join(f"{k}={v}" for k, v in spec.items() if k != "name")
    events = int((current != 0).sum().item())

    return [
        f"snntorch      beta={snn_cfg['beta']}  threshold={snn_cfg['threshold']}  "
        f"reset_mechanism={snn_cfg['reset_mechanism']}  reset_delay={snn_cfg['reset_delay']}  "
        f"surrogate={snn_cfg['surrogate']['type']}(alpha={snn_cfg['surrogate']['alpha']})",

        f"spikingjelly  tau={sj_cfg['tau']}  decay_input={sj_cfg['decay_input']}  "
        f"v_threshold={sj_cfg['v_threshold']}  v_reset={sj_cfg['v_reset']}  "
        f"surrogate={sj_cfg['surrogate']['type']}(alpha={sj_cfg['surrogate']['alpha']})",

        f"norse         dt={norse_cfg['dt']}  tau_mem_inv={norse_cfg['tau_mem_inv']}  "
        f"v_th={norse_cfg['v_th']}  v_reset={norse_cfg['v_reset']}  v_leak={norse_cfg['v_leak']}  "
        f"input_scale={norse_cfg['input_scale']}  "
        f"surrogate={norse_cfg['surrogate']['type']}(alpha={norse_cfg['surrogate']['alpha']})",

        f"effective     v = decay*v + gain*I  ->  snntorch {snn_decay:.3f}/{snn_gain:.3f}   "
        f"spikingjelly {sj_decay:.3f}/{sj_gain:.3f}   norse {norse_decay:.3f}/{norse_gain:.3f}",

        f"input         {spec['name']}  steps={steps}  {input_bits}  "
        f"({events} non-zero timesteps)",
    ]


def save_plot(
    test_name: str,
    current: torch.Tensor,
    traces: dict[str, Trace],
    threshold: float,
    result: dict[str, Any],
    output_path: Path,
    caption: list[str],
    run_timestamp: str,
    membrane_view: str,
) -> None:
    steps = torch.arange(len(current))
    figure, (ax_input, ax_membrane, ax_raster) = plt.subplots(
        3, 1, figsize=(12, 9.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 3, 1]},
    )

    # Drawn as a step, not a line: the signal is discrete -- one value per
    # timestep, instantly on and instantly off. A plain line plot connects the
    # samples with slopes and makes a single one-step event look like a
    # triangle that "rises and falls", which it does not.
    ax_input.step(steps, current, where="mid", color="#555555", linewidth=1.4)
    nonzero = torch.nonzero(current).flatten()
    ax_input.scatter(nonzero, current[nonzero], s=12, color="#555555", zorder=3)
    ax_input.set_ylabel("input\ncurrent I")
    ax_input.grid(alpha=0.3)

    for name in FRAMEWORKS:
        trace = traces[name]
        series = trace.v_pre if membrane_view == "pre_reset" else trace.v_post
        ax_membrane.plot(steps, series, label=name, **PLOT_STYLE[name])
    ax_membrane.axhline(threshold, color="black", linestyle="-.", linewidth=1,
                        label=f"threshold = {threshold}")
    ax_membrane.set_ylabel(MEMBRANE_LABEL[membrane_view])
    ax_membrane.legend(loc="upper right")
    ax_membrane.grid(alpha=0.3)

    for row, name in enumerate(FRAMEWORKS):
        firing_steps = torch.nonzero(traces[name].spikes).flatten()
        ax_raster.scatter(firing_steps, torch.full_like(firing_steps, row),
                          marker="|", s=180, color=PLOT_STYLE[name]["color"])
    ax_raster.set_yticks(range(len(FRAMEWORKS)))
    ax_raster.set_yticklabels(FRAMEWORKS)
    ax_raster.set_ylim(-0.6, len(FRAMEWORKS) - 0.4)
    ax_raster.set_xlabel("timestep")
    ax_raster.set_ylabel("spikes")
    ax_raster.grid(alpha=0.3, axis="x")

    verdict = "PASS" if result["passed"] else "FAIL"
    spike_counts = "  ".join(
        f"{name}={result['per_framework'][name]['total_spikes']}" for name in FRAMEWORKS
    )
    figure.suptitle(
        f"LIF equivalence  |  {test_name}  |  {verdict}  |  {run_timestamp}\n"
        f"worst max|dv| = {result['worst_max_v_deviation']:.2e}    "
        f"spike times match = {result['worst_spike_time_match'] * 100:.1f}%    "
        f"total spikes: {spike_counts}",
        fontsize=12,
    )

    # Leave room at the bottom for the parameter caption.
    figure.tight_layout(rect=(0, 0.155, 1, 0.955))
    figure.text(
        0.012, 0.012, "\n".join(caption),
        family="monospace", fontsize=7.5, va="bottom", color="#333333",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml",
                        help="path to the YAML config (default: config/default.yaml)")
    args = parser.parse_args()

    config = load_config(args.config)
    neuron_cfg = require(config, "neuron")

    versions = framework_versions()
    print("versions:", ", ".join(f"{k} {v}" for k, v in versions.items()))
    print(f"config:   {Path(args.config).resolve()}")

    device = require_str(config, "equivalence.device")
    torch.set_default_device(device)

    steps = require_int(config, "equivalence.steps")
    membrane_view = require_choice(
        config, "equivalence.plot_membrane", ["pre_reset", "post_reset"]
    )
    tolerance = {
        "max_v_deviation": require_float(config, "equivalence.tolerance.max_v_deviation"),
        "min_spike_time_match": require_float(config, "equivalence.tolerance.min_spike_time_match"),
    }
    output_dir = Path(require_str(config, "equivalence.output_dir"))

    input_specs = require(config, "equivalence.inputs")
    if not isinstance(input_specs, list) or not input_specs:
        raise ConfigError("equivalence.inputs must be a non-empty list")

    # Threshold is per-framework in the config; the plot needs one number to
    # draw a line at. They are supposed to be equal, so take snnTorch's and say
    # so loudly if the three disagree.
    thresholds = {
        "snntorch": require_float(neuron_cfg, "snntorch.threshold"),
        "spikingjelly": require_float(neuron_cfg, "spikingjelly.v_threshold"),
        "norse": require_float(neuron_cfg, "norse.v_th"),
    }
    if len(set(thresholds.values())) != 1:
        print(f"\nWARNING: thresholds differ between frameworks: {thresholds}")

    # Every run gets its own timestamp, and it goes into every filename. Runs
    # never overwrite each other, so you can change a parameter and compare the
    # new plot against the old one side by side.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"run:      {run_timestamp}")

    summary: dict[str, Any] = {
        "run_timestamp": run_timestamp,
        "versions": versions,
        "config_path": str(args.config),
        "neuron": neuron_cfg,
        "plot_membrane": membrane_view,
        "tests": {},
    }
    all_passed = True

    # No gradients needed: this compares forward dynamics only.
    with torch.no_grad():
        boundary = probe_threshold_boundary(neuron_cfg, thresholds)
        print_threshold_boundary(boundary, thresholds)
        summary["threshold_boundary_fires_at_exactly_threshold"] = boundary

        for spec in input_specs:
            test_name = spec["name"]
            current = make_input(spec, steps)

            traces = {name: RUNNERS[name](neuron_cfg, current) for name in FRAMEWORKS}
            result = compare(traces, **tolerance)

            print_report(test_name, result, tolerance)
            if not result["passed"]:
                print_divergence(current, traces, tolerance["max_v_deviation"])

            plot_path = output_dir / f"equivalence_{test_name}_{run_timestamp}.png"
            save_plot(
                test_name, current, traces, thresholds["snntorch"], result, plot_path,
                caption=metadata_lines(neuron_cfg, spec, steps, current),
                run_timestamp=run_timestamp,
                membrane_view=membrane_view,
            )
            print(f"  plot:   {plot_path}")

            summary["tests"][test_name] = {"input": spec, **result}
            all_passed = all_passed and result["passed"]

    summary["passed"] = all_passed
    summary_path = output_dir / f"equivalence_summary_{run_timestamp}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 68)
    print(f"OVERALL: {'PASS' if all_passed else 'FAIL'}   (summary written to {summary_path})")
    print("=" * 68)
    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as error:
        # Config problems are your problem to fix, not a bug -- show the message
        # plainly instead of burying it in a traceback.
        print(f"\nCONFIG ERROR: {error}", file=sys.stderr)
        sys.exit(2)
