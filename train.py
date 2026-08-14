"""Train one framework's spiking network and measure it.

Follows the execution order in docs/metrics_plan.md exactly -- the ordering is
not incidental. Spike counting is switched on only in its own untimed pass, the
warm-up never touches the weights, and the GPU is synchronised at region
boundaries rather than per batch.

    # full run, everything measured
    python train.py --config config/default.yaml --framework snntorch

    # quick smoke run
    python train.py --config config/default.yaml --framework snntorch \
        --max-batches 20 --max-eval-batches 10 --no-energy

    # force CPU (the laptop has no CUDA build of torch)
    python train.py --config config/default.yaml --device cpu --max-batches 3

Writes results/runs.csv, results/epochs.csv, results/layers.csv,
results/gradients.csv and results/runs/<run_id>.json. All are append-only with a
fixed schema.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.adapters import IMPLEMENTED
from src.config import (
    ConfigError,
    ephemeral_storage_warning,
    load_config,
    output_dirs,
    require,
    require_bool,
    require_choice,
    require_float,
    require_int,
    require_str,
)
from src.data import build_loader, build_split
from src.metrics import (
    PowerSampler,
    detect_update_interval_ms,
    dynamic_energy_j,
    energy_warnings,
    gradient_probe,
    integrate_energy_j,
    measure_idle_power,
    measure_latency,
    nvml_handle,
    peak_memory_mb,
    reset_peak_memory,
    spike_counting,
    spike_rates,
    summarise_power,
    timed,
)
# build_net is build_network for every config with step_mode 's' -- it delegates
# straight to it. Only neuron.spikingjelly.step_mode 'm' takes a different path.
from src.multistep.build import build_net
from src.results import (
    EPOCH_COLUMNS,
    LAYER_COLUMNS,
    RUN_COLUMNS,
    ResultsError,
    config_hash,
    environment_info,
    make_run_id,
    write_results,
)

# Plain torch optimizers and losses -- none of the three frameworks supplies
# these, so all three get identical treatment.
OPTIMIZERS = {
    "nadam": torch.optim.NAdam,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}
LOSSES = {"cross_entropy": nn.CrossEntropyLoss}


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ConfigError(
                "training.device is 'cuda' but torch reports no CUDA available. "
                "Either this is a CPU-only torch build (check with check_env.py) "
                "or there is no GPU. Use --device cpu, or set training.device: cpu."
            )
        return torch.device("cuda")
    raise ConfigError(f"training.device must be 'cuda' or 'cpu', got '{requested}'")


def build_optimizer(training_cfg: dict, parameters) -> torch.optim.Optimizer:
    name = require_str(training_cfg, "optimizer.type")
    if name not in OPTIMIZERS:
        raise ConfigError(
            f"training.optimizer.type '{name}' is not supported. "
            f"Supported: {sorted(OPTIMIZERS)}"
        )
    return OPTIMIZERS[name](parameters, lr=require_float(training_cfg, "optimizer.lr"))


def build_loss(training_cfg: dict) -> nn.Module:
    return LOSSES[require_choice(training_cfg, "loss", sorted(LOSSES))]()


def weight_fingerprint(net) -> str:
    """Trainable parameters only -- see check_network.py for why not state_dict."""
    digest = hashlib.sha256()
    for name, tensor in sorted(
        (n, t) for n, t in net.named_parameters() if t.requires_grad
    ):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def warm_up(net, loader, device, loss_fn, iterations: int):
    """Compile the CUDA kernels and start the DataLoader workers.

    Forward AND backward, because backward kernels need compiling too -- but no
    `optimizer.step()`, so the weights are untouched and the timed epochs begin
    exactly where they would have without this. Gradients are cleared after.

    The same first batch is reused rather than consuming the epoch's batches.

    Returns that batch, still on the CPU, for the gradient probe to reuse --
    None when there is no warm-up. REUSE, not a fresh batch, and not for tidiness:
    `loader` shuffles from its own seeded generator, so every extra `iter(loader)`
    would draw another permutation and shift the batch order of every later epoch.
    That would change the run, which is exactly what a measurement must not do.
    """
    if iterations <= 0:
        return None
    net.train()
    frames, labels = next(iter(loader))
    device_frames, device_labels = frames.to(device), labels.to(device)
    for _ in range(iterations):
        loss_fn(net(device_frames), device_labels).backward()
    net.zero_grad(set_to_none=True)
    # The CPU copies, deliberately: keeping a batch resident on the GPU for the
    # whole run would raise peak_memory_train_mb by its own size (~24 MB here).
    return frames, labels


def probe_gradients(
    net, probe_batch, device, loss_fn, epoch: int, rows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Measure gradient magnitude at this point in training and record it.

    Returns None when there is no fixed batch to probe with, so every caller
    degrades to "no gradient numbers" rather than to a crash.

    `epoch` 0 means before any training: at that point every framework holds
    byte-identical weights, so a difference in the numbers is the neuron and its
    surrogate and nothing else -- the cleanest gradient-scale comparison there is.
    """
    if probe_batch is None:
        return None
    frames, labels = probe_batch
    result = gradient_probe(net, frames, labels, device, loss_fn)
    rows.extend(
        {
            "epoch": epoch,
            "probe_seconds": result["probe_seconds"],
            "probe_loss": result["probe_loss"],
            "grad_norm_global": result["grad_norm_global"],
            **parameter,
        }
        for parameter in result["params"]
    )
    return result


def gradient_line(result: dict[str, Any]) -> str:
    return (f"||g|| {result['grad_norm_global']:.4g}   "
            f"probe loss {result['probe_loss']:.4f}   "
            f"({result['probe_seconds']:.2f}s, untimed)")


def print_gradient_table(rows: list[dict[str, Any]]) -> None:
    """Parameters down the side, epochs across the top.

    That way round because a run has more epochs than parameter tensors, and a
    table wider than the terminal is a table nobody reads.
    """
    if not rows:
        return
    epochs = sorted({row["epoch"] for row in rows})
    # dict.fromkeys keeps the order the parameters appear in the network.
    names = list(dict.fromkeys(row["param_name"] for row in rows))
    norm = {(row["epoch"], row["param_name"]): row["grad_norm"] for row in rows}
    overall = {row["epoch"]: row["grad_norm_global"] for row in rows}
    losses = {row["epoch"]: row["probe_loss"] for row in rows}

    label_width = max(len(name) for name in names + ["global ||g||"]) + 2
    print(f"  {'parameter':<{label_width}}"
          + "".join(f"{'e' + str(epoch):>11}" for epoch in epochs))
    for name in names:
        print(f"  {name:<{label_width}}" + "".join(
            f"{norm.get((epoch, name), float('nan')):>11.4g}" for epoch in epochs))
    print(f"  {'global ||g||':<{label_width}}"
          + "".join(f"{overall.get(epoch, float('nan')):>11.4g}" for epoch in epochs))
    print(f"  {'probe loss':<{label_width}}"
          + "".join(f"{losses.get(epoch, float('nan')):>11.4f}" for epoch in epochs))
    print("  e0 is before any training, so all frameworks are at identical weights")
    print("  there; a difference at e0 is the surrogate gradient, nothing else.")


def run_pass(
    net,
    loader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    label: str,
    report_every: int = 25,
) -> dict[str, float]:
    """One pass over a loader. optimizer=None means evaluate, don't learn."""
    training = optimizer is not None
    net.train() if training else net.eval()

    total_loss, correct, seen, batches = 0.0, 0, 0, 0
    for frames, labels in loader:
        if max_batches is not None and batches >= max_batches:
            break
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            spike_counts = net(frames)
            loss = loss_fn(spike_counts, labels)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.numel()
        correct += (spike_counts.argmax(dim=1) == labels).sum().item()
        seen += labels.numel()
        batches += 1
        if batches % report_every == 0:
            print(f"    {label} batch {batches:>4}  loss {total_loss / seen:.4f}  "
                  f"acc {100 * correct / seen:5.1f}%")

    if seen == 0:
        return {"loss": 0.0, "accuracy_pct": 0.0, "samples": 0}
    return {"loss": total_loss / seen, "accuracy_pct": 100 * correct / seen,
            "samples": seen}


def collect_single_samples(loader, device, count: int) -> list[torch.Tensor]:
    """`count` individual samples, on-device, each shaped [T, 1, C, H, W].

    Pre-loaded so the latency measurement times the network, not the data
    pipeline -- and the host-to-device copy is identical for all three
    frameworks anyway.
    """
    samples: list[torch.Tensor] = []
    for frames, _ in loader:
        frames = frames.to(device)
        for index in range(frames.shape[1]):
            samples.append(frames[:, index: index + 1].contiguous())
            if len(samples) >= count:
                return samples
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="path to the experiment's YAML config. REQUIRED and with no default, so a run can never silently use another experiment's neuron: config/default.yaml is ex1 (forced-equivalent neuron), config/config_ex2.yaml is ex2 (each framework out of the box).",
    )
    parser.add_argument("--framework", default="snntorch",
                        help=f"implemented: {IMPLEMENTED}")
    parser.add_argument("--device", default=None, help="override training.device")
    parser.add_argument("--seed", type=int, default=None, help="override training.seed")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="stop each training epoch after this many batches")
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--no-energy", action="store_true",
                        help="skip NVML entirely (also skipped if unavailable)")
    parser.add_argument("--experiment", default=None,
                        help="experiment folder name, e.g. ex2. Omit for scratch "
                             "runs, which go flat into local_runs/")
    parser.add_argument("--results-root", default="experiments",
                        help="where experiment folders live; on Colab use "
                             "/content/drive/MyDrive/snn_results")
    parser.add_argument("--allow-ephemeral", action="store_true",
                        help="permit writing results to Colab's temporary disk")
    parser.add_argument("--cache-archive", default=None, metavar="DIR",
                        help="folder holding archived dataset caches, normally on "
                             "Drive. RESTORE ONLY here: if no local cache exists a "
                             "matching archive is unpacked, but training never "
                             "writes one -- prepare_data.py does that. Omit for the "
                             "original behaviour.")
    parser.add_argument("--notes", default="", help="free text stored with the run")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = require(config, "dataset")
    neuron_cfg = require(config, "neuron")
    training_cfg = require(config, "training")
    metrics_cfg = require(config, "metrics")

    device = resolve_device(args.device or require_str(training_cfg, "device"))
    seed = args.seed if args.seed is not None else require_int(training_cfg, "seed")
    epochs = args.epochs if args.epochs is not None else require_int(training_cfg, "epochs")

    results_dir, _, flat = output_dirs(args.experiment, args.results_root)
    warning = ephemeral_storage_warning(results_dir)
    if warning and not args.allow_ephemeral:
        raise ConfigError(warning)
    if warning:
        print(f"WARNING: {warning}\n")

    started = datetime.now()
    run_id = make_run_id(args.framework, seed, started)

    print("=" * 70)
    print(f"run_id      {run_id}")
    print(f"experiment  {args.experiment or 'none (scratch)'}   -> {results_dir}")
    print(f"framework   {args.framework}")
    print(f"device      {device}" + (f"  ({torch.cuda.get_device_name(0)})"
                                     if device.type == "cuda" else ""))
    print(f"seed        {seed}   epochs {epochs}")
    print(f"config      {args.config}   hash {config_hash(config)}")
    print("=" * 70)

    # ---- region 0: data, model, NVML probe --------------------------------
    print("\n[0] setup")
    cache_archive_root = Path(args.cache_archive) if args.cache_archive else None
    train_set, info = build_split(dataset_cfg, train=True,
                                  cache_archive=cache_archive_root)
    test_set, _ = build_split(dataset_cfg, train=False,
                              cache_archive=cache_archive_root)
    train_loader = build_loader(train_set, dataset_cfg,
                                shuffle=require_bool(dataset_cfg, "shuffle_train"),
                                seed=seed)
    test_loader = build_loader(test_set, dataset_cfg,
                               shuffle=require_bool(dataset_cfg, "shuffle_test"),
                               seed=seed)
    print(f"  train {len(train_set)}   test {len(test_set)}   "
          f"T={info.time_steps}  batch {require_int(dataset_cfg, 'batch_size')}")

    net = build_net(args.framework, neuron_cfg, info, seed=seed).to(device)
    optimizer = build_optimizer(training_cfg, net.parameters())
    loss_fn = build_loss(training_cfg)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    fingerprint = weight_fingerprint(net)
    print(f"  params {trainable:,}   weight fingerprint {fingerprint}")

    measure_energy = require_bool(metrics_cfg, "measure_energy") and not args.no_energy
    handle = nvml_handle() if measure_energy else None
    update_interval_ms = detect_update_interval_ms(handle) if handle else None
    poll_s = max(update_interval_ms / 2000.0, 0.005) if update_interval_ms else 0.05
    if handle is None:
        print("  energy: NVML unavailable or disabled -- columns will be empty")
    else:
        print(f"  energy: NVML sensor updates every "
              f"{update_interval_ms:.0f} ms" if update_interval_ms
              else "  energy: NVML available (update rate undetermined)")

    # ---- region 1: cold idle baseline -------------------------------------
    idle_seconds = require_int(metrics_cfg, "idle_seconds")
    idle_cold = {"mean_w": None, "min_w": None, "max_w": None, "n": 0}
    if handle is not None:
        print(f"\n[1] cold idle baseline ({idle_seconds}s)")
        idle_cold = measure_idle_power(handle, idle_seconds, poll_s)
        print(f"  {idle_cold['mean_w']:.2f} W  "
              f"(range {idle_cold['min_w']:.2f}-{idle_cold['max_w']:.2f})")

    # ---- region 2: warm-up (untimed, weights untouched) --------------------
    print(f"\n[2] warm-up x{require_int(metrics_cfg, 'warmup_iterations')} "
          f"(no optimizer step)")
    before = weight_fingerprint(net)
    probe_batch = warm_up(net, train_loader, device, loss_fn,
                          require_int(metrics_cfg, "warmup_iterations"))
    print(f"  weights unchanged: {before == weight_fingerprint(net)}")

    # ---- region 2b: gradient probe at epoch 0 ------------------------------
    # Untimed, no optimizer step, and BEFORE the power sampler starts, so this
    # first probe costs the measured run nothing whatsoever.
    gradient_rows: list[dict[str, Any]] = []
    probe_seconds_total = 0.0
    # Tracked apart from the total because the epoch-0 probe just below happens
    # outside the energy span, and only what is INSIDE it can distort energy.
    probe_seconds_in_span = 0.0
    if probe_batch is None:
        print("  warmup_iterations is 0, so there is no fixed batch -- "
              "gradient probe disabled for this run")
    else:
        print("\n[2b] gradient probe, before training")
        before = weight_fingerprint(net)
        initial_grads = probe_gradients(net, probe_batch, device, loss_fn, 0,
                                        gradient_rows)
        probe_seconds_total += initial_grads["probe_seconds"]
        print(f"  {gradient_line(initial_grads)}")
        print(f"  weights unchanged: {before == weight_fingerprint(net)}")

    # ---- region 3: training ------------------------------------------------
    print("\n[3] training")
    reset_peak_memory(device)
    sampler = PowerSampler(handle, poll_s) if handle is not None else None
    if sampler:
        sampler.start()

    energy_span_start = time.perf_counter()
    epoch_rows: list[dict[str, Any]] = []
    train_time_s = 0.0
    last_train, last_test = {}, {}

    for epoch in range(1, epochs + 1):
        print(f"  epoch {epoch}/{epochs}")
        with timed(device) as epoch_timer:
            last_train = run_pass(net, train_loader, device, loss_fn, optimizer,
                                  args.max_batches, "train")
        train_time_s += epoch_timer["seconds"]

        # Per-epoch evaluation is UNTIMED, so spike counting rides along for free
        # rather than needing its own extra pass.
        with spike_counting(net), torch.no_grad():
            last_test = run_pass(net, test_loader, device, loss_fn, None,
                                 args.max_eval_batches, "eval", report_every=10**9)
            epoch_activity = spike_rates(net)

        # Deliberately OUTSIDE the block above: the probe's own spikes must not
        # land in this epoch's spike rate, and it needs gradients, which
        # `torch.no_grad()` would have switched off.
        #
        # One extra batch against an epoch's ~469 is ~0.2% -- small enough to sit
        # inside the energy span rather than distort the trace by sitting outside
        # it, and its cost is totalled and printed below instead of assumed.
        epoch_grads = probe_gradients(net, probe_batch, device, loss_fn, epoch,
                                      gradient_rows)
        if epoch_grads is not None:
            probe_seconds_total += epoch_grads["probe_seconds"]
            probe_seconds_in_span += epoch_grads["probe_seconds"]

        print(f"    train loss {last_train['loss']:.4f} acc {last_train['accuracy_pct']:.1f}%"
              f"   test loss {last_test['loss']:.4f} acc {last_test['accuracy_pct']:.1f}%"
              f"   spikes {epoch_activity['spike_rate_pct']:.2f}%"
              f"   ({epoch_timer['seconds']:.1f}s train)")
        if epoch_grads is not None:
            print(f"    gradient   {gradient_line(epoch_grads)}")

        epoch_rows.append({
            "epoch": epoch,
            "train_loss": last_train["loss"],
            "train_accuracy_pct": last_train["accuracy_pct"],
            "test_loss": last_test["loss"],
            "test_accuracy_pct": last_test["accuracy_pct"],
            "epoch_train_time_s": epoch_timer["seconds"],
            "spike_rate_pct": epoch_activity["spike_rate_pct"],
        })

    energy_span_s = time.perf_counter() - energy_span_start
    power_samples = sampler.stop() if sampler else []
    train_memory = peak_memory_mb(device)

    # ---- region 3b: hot idle baseline --------------------------------------
    idle_hot = {"mean_w": None, "min_w": None, "max_w": None, "n": 0}
    if handle is not None:
        print(f"\n[3b] hot idle baseline ({idle_seconds}s, GPU idle but warm)")
        idle_hot = measure_idle_power(handle, idle_seconds, poll_s)
        print(f"  {idle_hot['mean_w']:.2f} W  "
              f"(range {idle_hot['min_w']:.2f}-{idle_hot['max_w']:.2f})")

    total_energy_j = integrate_energy_j(power_samples)
    dynamic_j = dynamic_energy_j(total_energy_j, idle_hot["mean_w"], energy_span_s)
    warnings = energy_warnings(
        total_energy_j, dynamic_j, idle_hot, summarise_power(power_samples),
        energy_span_s, update_interval_ms,
    ) if handle is not None else []

    # ---- region 4: batched evaluation (timed, counting OFF) ----------------
    print("\n[4] evaluation (batched, timed)")
    reset_peak_memory(device)
    with timed(device) as eval_timer:
        final_test = run_pass(net, test_loader, device, loss_fn, None,
                              args.max_eval_batches, "eval", report_every=10**9)
    infer_memory = peak_memory_mb(device)
    throughput = (final_test["samples"] / eval_timer["seconds"]
                  if eval_timer["seconds"] > 0 else None)
    print(f"  accuracy {final_test['accuracy_pct']:.2f}%   "
          f"throughput {throughput:.1f} samples/s")

    # ---- region 5: latency (batch_size = 1) --------------------------------
    print("\n[5] latency (batch_size=1)")
    singles = collect_single_samples(test_loader, device,
                                     require_int(metrics_cfg, "latency_samples"))
    latency = measure_latency(net, singles, device)
    print(f"  median {latency['latency_ms']:.2f} ms   "
          f"p90 {latency['latency_p90_ms']:.2f} ms   "
          f"({latency['latency_samples']} samples)")

    # ---- region 6: spike counting (untimed) --------------------------------
    print("\n[6] spike activity (untimed pass)")
    with spike_counting(net), torch.no_grad():
        run_pass(net, test_loader, device, loss_fn, None,
                 require_int(metrics_cfg, "spike_batches"), "spikes",
                 report_every=10**9)
        activity = spike_rates(net)
    print(f"  {'layer':<7}{'type':<18}{'neurons':>9}{'rate %':>9}"
          f"{'spikes/neuron/inference':>26}")
    for row in activity["layers"]:
        print(f"  {row['layer_index']:<7}{row['layer_type']:<18}{row['neurons']:>9}"
              f"{row['spike_rate_pct']:>9.3f}"
              f"{row['spikes_per_neuron_per_inference']:>26.4f}")
    print(f"  {'overall':<7}{'':<18}{'':>9}{activity['spike_rate_pct']:>9.3f}"
          f"{activity['spikes_per_neuron_per_inference']:>26.4f}")
    print("  ^ spikes/neuron/inference = rate x T; the unit the SNN literature")
    print("    reports, so it is directly comparable to published figures")

    # ---- region 7: write ---------------------------------------------------
    environment = environment_info(args.framework)
    norse_surrogate = neuron_cfg[args.framework].get("surrogate", {})
    run_row: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": started.isoformat(timespec="seconds"),
        "framework": args.framework,
        "seed": seed,
        "config_path": args.config,
        "config_hash": config_hash(config),
        "dataset": info.name,
        "time_steps": info.time_steps,
        "batch_size": require_int(dataset_cfg, "batch_size"),
        "num_workers": require_int(dataset_cfg, "num_workers"),
        "binarize": info.binarize,
        "denoise_us": info.denoise_filter_time_us,
        "epochs": epochs,
        "optimizer": require_str(training_cfg, "optimizer.type"),
        "lr": require_float(training_cfg, "optimizer.lr"),
        "surrogate": f"{norse_surrogate.get('type')}({norse_surrogate.get('alpha')})",
        "trainable_params": trainable,
        "weight_fingerprint": fingerprint,
        "test_accuracy_pct": final_test["accuracy_pct"],
        "train_loss_final": last_train.get("loss"),
        "test_loss_final": final_test["loss"],
        "train_time_s": train_time_s,
        "train_time_per_epoch_s": train_time_s / epochs if epochs else None,
        "inference_throughput_samples_per_s": throughput,
        "inference_latency_bs1_ms": latency["latency_ms"],
        "inference_latency_bs1_mean_ms": latency["latency_mean_ms"],
        "inference_latency_bs1_p90_ms": latency["latency_p90_ms"],
        "spike_rate_pct": activity["spike_rate_pct"],
        "peak_memory_train_mb": train_memory["allocated_mb"],
        "peak_memory_infer_mb": infer_memory["allocated_mb"],
        "peak_reserved_train_mb": train_memory["reserved_mb"],
        "peak_reserved_infer_mb": infer_memory["reserved_mb"],
        "nvml_update_interval_ms": update_interval_ms,
        "idle_power_cold_w": idle_cold["mean_w"],
        "idle_power_after_train_w": idle_hot["mean_w"],
        "train_energy_duration_s": energy_span_s if handle is not None else None,
        "train_energy_j": total_energy_j,
        "train_energy_dynamic_j": dynamic_j,
        "energy_warnings": " | ".join(warnings) if warnings else None,
        "notes": args.notes or None,
        **environment,
    }

    layer_rows = [
        {
            "layer_index": row["layer_index"],
            "layer_type": row["layer_type"],
            "neurons": row["neurons"],
            "total_spikes": row["total_spikes"],
            "opportunities": row["opportunities"],
            "spike_rate_pct": row["spike_rate_pct"],
        }
        for row in activity["layers"]
    ]

    paths = write_results(
        results_dir, run_row, epoch_rows, layer_rows,
        json_payload={
            "run_id": run_id,
            "config": config,
            "environment": environment,
            "metrics": run_row,
            "epochs": epoch_rows,
            "layers": layer_rows,
            "idle_power_cold": idle_cold,
            "idle_power_after_train": idle_hot,
            "energy_warnings": warnings,
            "gradients": gradient_rows,
            "gradient_probe_seconds_total": probe_seconds_total,
            "gradient_probe_seconds_in_energy_span": probe_seconds_in_span,
        },
        flat=flat,
        gradient_rows=gradient_rows,
    )

    print("\n[7] results written")
    for name, path in paths.items():
        print(f"  {name:<7} {path}")

    if gradient_rows:
        # Printed after the writing, on purpose: nothing about presenting a table
        # may be able to lose a finished run's results.
        print("\ngradient behaviour -- ||g|| on ONE fixed batch, untimed, "
              "no optimizer step")
        print_gradient_table(gradient_rows)
        share = (100.0 * probe_seconds_in_span / energy_span_s
                 if energy_span_s else 0.0)
        print(f"  probe cost {probe_seconds_total:.1f}s in total, of which "
              f"{probe_seconds_in_span:.1f}s fell inside the "
              f"{energy_span_s:.0f}s energy span ({share:.2f}% of it)")

    if warnings:
        print(f"\n  {len(warnings)} ENERGY WARNING(S) -- recorded in the CSV:")
        for warning in warnings:
            print(f"    - {warning}")

    print("\n" + "=" * 70)
    print(f"accuracy {final_test['accuracy_pct']:.2f}%   "
          f"train {train_time_s:.1f}s   "
          f"throughput {throughput:.1f}/s   "
          f"latency {latency['latency_ms']:.2f}ms   "
          f"spikes {activity['spike_rate_pct']:.2f}% "
          f"({activity['spikes_per_neuron_per_inference']:.3f}/neuron/inference)")
    if total_energy_j is not None:
        print(f"energy   {total_energy_j:.1f} J total   "
              f"{dynamic_j:.1f} J dynamic   "
              f"(idle {idle_hot['mean_w']:.1f} W over {energy_span_s:.0f}s)")
    if gradient_rows:
        print(f"gradient ||g|| {gradient_rows[0]['grad_norm_global']:.4g} "
              f"before training -> "
              f"{gradient_rows[-1]['grad_norm_global']:.4g} after")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ConfigError, ResultsError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(2)
