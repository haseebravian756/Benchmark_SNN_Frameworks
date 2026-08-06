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
    run_banner,
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

    What this check is for: the three FRAMEWORKS, same seed, SAME MACHINE, must
    produce the same string. If they do not, they are not starting from the same
    weights and no accuracy comparison between them means anything.

    What it is NOT for: comparing machines. PyTorch does not promise identical
    random numbers across torch versions, platforms or CPU/CUDA builds, so the
    fingerprint legitimately differs between (say) a Windows CPU laptop and a
    Colab T4. That does not matter here -- all runs of one experiment happen on
    one machine. Record the value per machine.

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


def summarise_framework(
    framework: str, neuron_cfg: dict, info: DataInfo, seed: int, batch: int
) -> dict:
    """Build one framework's network and report the things that must match."""
    net = build_network(lif_factory(framework, neuron_cfg), info, seed=seed)

    dummy = torch.zeros(info.time_steps, batch, info.channels, info.height, info.width)
    with torch.no_grad():
        output = net(dummy)
    state_before = [layer.has_state() for layer in net.lif_layers()]
    net.reset()
    state_after = [layer.has_state() for layer in net.lif_layers()]

    linear = [m for m in net.layers if isinstance(m, torch.nn.Linear)][0]
    return {
        "framework": framework,
        "fingerprint": weight_fingerprint(net),
        "params": sum(p.numel() for p in net.parameters() if p.requires_grad),
        "flatten": linear.in_features,
        "output_shape": tuple(output.shape),
        "neuron": type(net.lif_layers()[0]).__name__,
        "surrogate": net.lif_layers()[0].describe().get("surrogate", "?"),
        # The FULL neuron settings, so the reader can confirm the experiment really
        # got the neuron it intended. All LIF layers in a network share one config,
        # so layer 0 speaks for all of them -- verified below before trusting that.
        "describe": net.lif_layers()[0].describe(),
        "layers_agree": len({str(l.describe()) for l in net.lif_layers()}) == 1,
        "state_after_forward": any(state_before),
        "state_after_reset": any(state_after),
    }


def compare_all(neuron_cfg: dict, info: DataInfo, seed: int, batch: int) -> int:
    """Build every implemented framework and check they agree where they must.

    Safe to load all three in one process here: this script measures nothing, so
    there is no timing or CUDA state to contaminate. train.py deliberately does
    NOT do this -- there, one framework per process is the whole point.
    """
    rows = [summarise_framework(f, neuron_cfg, info, seed, batch) for f in IMPLEMENTED]

    print("=" * 78)
    print(f"ALL FRAMEWORKS, seed {seed}")
    print("=" * 78)
    print(f"{'framework':<14}{'fingerprint':>18}{'params':>10}{'flatten':>9}"
          f"{'output':>12}{'reset ok':>10}")
    print("-" * 78)
    for row in rows:
        reset_ok = "yes" if (row["state_after_forward"] and not row["state_after_reset"]) else "NO"
        print(f"{row['framework']:<14}{row['fingerprint']:>18}{row['params']:>10,}"
              f"{row['flatten']:>9}{str(row['output_shape']):>12}{reset_ok:>10}")

    print()
    print("=" * 78)
    print("NEURON ACTUALLY BUILT, per framework")
    print("=" * 78)
    print("These are SUPPOSED to differ between frameworks -- but they must be the")
    print("values THIS experiment's config asked for. Read them against the config.")
    print()
    for row in rows:
        print(f"  {row['framework']}  ({row['neuron']})")
        for key, value in row["describe"].items():
            print(f"      {key:<22}{value}")
        if not row["layers_agree"]:
            print("      !! this framework's LIF layers do NOT all share one config")
        print()

    print("=" * 78)
    checks = [
        ("weight fingerprints identical", len({r["fingerprint"] for r in rows}) == 1),
        ("trainable parameters identical", len({r["params"] for r in rows}) == 1),
        ("flatten size identical", len({r["flatten"] for r in rows}) == 1),
        ("output shapes identical", len({r["output_shape"] for r in rows}) == 1),
        ("reset clears state everywhere",
         all(r["state_after_forward"] and not r["state_after_reset"] for r in rows)),
        # Every LIF layer within one framework must carry the same settings. A
        # mismatch would mean the config was applied per-layer inconsistently, which
        # no amount of cross-framework agreement would reveal.
        ("each framework's LIF layers all share one config",
         all(r["layers_agree"] for r in rows)),
    ]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    passed = all(ok for _, ok in checks)
    print("=" * 78)
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("\nThe frameworks do NOT start from the same place. Any accuracy")
        print("comparison between them would be meaningless until this is fixed.")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="path to the experiment's YAML config. REQUIRED and with no default, so a run can never silently use another experiment's neuron: config/default.yaml is ex1 (forced-equivalent neuron), config/config_ex2.yaml is ex2 (each framework out of the box).",
    )
    parser.add_argument("--framework", default="snntorch",
                        help=f"implemented so far: {IMPLEMENTED}")
    parser.add_argument(
        "--experiment",
        default=None,
        help="LABEL ONLY, e.g. ex2. This script writes nothing; the label just "
        "records in the output which experiment you were checking for.",
    )
    parser.add_argument("--all", action="store_true",
                        help="build every framework and compare them side by side")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4,
                        help="dummy batch size (small keeps this fast on CPU)")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = require(config, "dataset")
    neuron_cfg = require(config, "neuron")

    info = data_info_without_download(dataset_cfg)

    print(run_banner(
        "check_network.py -- is the network identical across frameworks?",
        experiment=args.experiment,
        config_path=args.config,
        config=config,
        framework="all three" if args.all else args.framework,
        extra={"seed": args.seed},
        writes_results=False,
    ))
    print()

    if args.all:
        print(f"data      : T={info.time_steps}  "
              f"{info.channels}x{info.height}x{info.width}  "
              f"-> {info.num_classes} classes\n")
        return compare_all(neuron_cfg, info, args.seed, args.batch)

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
    print("  ^ must match the OTHER FRAMEWORKS on THIS machine, same seed.")
    print("    A different value on a different machine is expected and fine:")
    print("    torch does not promise identical RNG across versions/platforms.")

    # Reset must genuinely clear state, or neurons leak between batches.
    # Each adapter answers has_state() for itself, since every framework keeps
    # its membrane somewhere different.
    with torch.no_grad():
        net(batch)
    leftover = [type(layer).__name__ for layer in net.lif_layers() if layer.has_state()]
    net.reset()
    still_set = [type(layer).__name__ for layer in net.lif_layers() if layer.has_state()]
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
