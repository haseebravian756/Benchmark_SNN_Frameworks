"""Train one framework's spiking network and print how it went.

This is deliberately minimal: plain print statements, no NVML, no CSV, no
timing methodology. Its job is to prove the pipeline runs end to end and that
the network actually learns. Metrics and results capture come later.

    # quick smoke run - a few batches, seconds not hours
    python train.py --config config/default.yaml --framework snntorch \
        --max-batches 20 --max-eval-batches 10

    # a real epoch
    python train.py --config config/default.yaml --framework snntorch

    # force CPU (the laptop has no CUDA build of torch)
    python train.py --config config/default.yaml --device cpu --max-batches 3
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn as nn

from src.adapters import IMPLEMENTED, lif_factory
from src.config import (
    ConfigError,
    load_config,
    require,
    require_bool,
    require_choice,
    require_float,
    require_int,
    require_str,
)
from src.data import build_loader, build_split
from src.network import build_network

# Plain torch optimizers -- none of the three frameworks supplies these, so all
# three get identical treatment.
OPTIMIZERS = {
    "nadam": torch.optim.NAdam,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}

# Plain torch losses, for the same reason.
LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
}


def resolve_device(requested: str) -> torch.device:
    """Turn the configured device into a real one, failing loudly if impossible."""
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
    name = require_choice(training_cfg, "loss", sorted(LOSSES))
    return LOSSES[name]()


def run_epoch(
    net,
    loader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    label: str,
    report_every: int = 10,
) -> tuple[float, float, int]:
    """One pass over a loader. optimizer=None means evaluate, don't learn.

    Returns (mean loss, accuracy, samples seen).
    """
    training = optimizer is not None
    net.train() if training else net.eval()

    total_loss = 0.0
    correct = 0
    seen = 0
    batches = 0

    for frames, labels in loader:
        if max_batches is not None and batches >= max_batches:
            break

        # frames arrive as [T, batch, C, H, W] -- time first, from collate.
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # torch.no_grad for evaluation: no graph, much less memory.
        with torch.set_grad_enabled(training):
            spike_counts = net(frames)          # [batch, num_classes]
            loss = loss_fn(spike_counts, labels)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_correct = (spike_counts.argmax(dim=1) == labels).sum().item()
        total_loss += loss.item() * labels.numel()
        correct += batch_correct
        seen += labels.numel()
        batches += 1

        if batches % report_every == 0:
            print(
                f"    {label} batch {batches:>4}  "
                f"loss {total_loss / seen:.4f}  "
                f"running acc {100 * correct / seen:5.1f}%"
            )

    if seen == 0:
        return 0.0, 0.0, 0
    return total_loss / seen, 100 * correct / seen, seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--framework", default="snntorch",
                        help=f"implemented so far: {IMPLEMENTED}")
    parser.add_argument("--device", default=None,
                        help="override training.device (cuda | cpu)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override training.epochs")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="stop each training epoch after this many batches")
    parser.add_argument("--max-eval-batches", type=int, default=None,
                        help="stop evaluation after this many batches")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = require(config, "dataset")
    neuron_cfg = require(config, "neuron")
    training_cfg = require(config, "training")

    device = resolve_device(args.device or require_str(training_cfg, "device"))
    seed = require_int(training_cfg, "seed")
    epochs = args.epochs if args.epochs is not None else require_int(training_cfg, "epochs")

    print("=" * 66)
    print(f"framework   {args.framework}")
    print(f"device      {device}" + (f"  ({torch.cuda.get_device_name(0)})"
                                     if device.type == "cuda" else ""))
    print(f"seed        {seed}")
    print(f"epochs      {epochs}")
    print(f"optimizer   {require_str(training_cfg, 'optimizer.type')} "
          f"lr={require_float(training_cfg, 'optimizer.lr')}")
    print(f"loss        {require_str(training_cfg, 'loss')}")
    print("=" * 66)

    # ---- data -------------------------------------------------------------
    print("\ndata")
    train_set, info = build_split(dataset_cfg, train=True)
    test_set, _ = build_split(dataset_cfg, train=False)
    print(f"  train samples {len(train_set)}   test samples {len(test_set)}")
    print(f"  T={info.time_steps}  {info.channels}x{info.height}x{info.width}"
          f"  -> {info.num_classes} classes")
    print(f"  binarize {info.binarize}   denoise {info.denoise_filter_time_us} us")

    train_loader = build_loader(
        train_set, dataset_cfg,
        shuffle=require_bool(dataset_cfg, "shuffle_train"), seed=seed,
    )
    test_loader = build_loader(
        test_set, dataset_cfg,
        shuffle=require_bool(dataset_cfg, "shuffle_test"), seed=seed,
    )
    print(f"  batch_size {require_int(dataset_cfg, 'batch_size')}   "
          f"num_workers {require_int(dataset_cfg, 'num_workers')}")

    # ---- model ------------------------------------------------------------
    print("\nmodel")
    make_lif = lif_factory(args.framework, neuron_cfg)
    net = build_network(make_lif, info, seed=seed).to(device)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  trainable parameters {trainable:,}")
    print(f"  LIF layers {len(net.lif_layers())}")
    for index, layer in enumerate(net.lif_layers()):
        print(f"    LIF {index}: {layer.describe()}")

    optimizer = build_optimizer(training_cfg, net.parameters())
    loss_fn = build_loss(training_cfg)

    # ---- train ------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        print(f"\nepoch {epoch}/{epochs}  (train)")
        start = time.perf_counter()
        loss, accuracy, seen = run_epoch(
            net, train_loader, device, loss_fn, optimizer, args.max_batches, "train"
        )
        elapsed = time.perf_counter() - start
        print(f"  train: loss {loss:.4f}  acc {accuracy:.1f}%  "
              f"({seen} samples in {elapsed:.1f}s)")
        print("  ^ wall clock only, NOT a benchmark measurement "
              "(no warm-up, no cuda.synchronize)")

        print(f"epoch {epoch}/{epochs}  (test)")
        loss, accuracy, seen = run_epoch(
            net, test_loader, device, loss_fn, None, args.max_eval_batches, "test"
        )
        print(f"  test:  loss {loss:.4f}  acc {accuracy:.1f}%  ({seen} samples)")

    if device.type == "cuda":
        print(f"\npeak GPU memory: "
              f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    print("\ndone")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as error:
        print(f"\nCONFIG ERROR: {error}", file=sys.stderr)
        sys.exit(2)
