"""Exercise every measurement in src/metrics.py in isolation, on dummy data.

No dataset, no training -- the point is to see each measurement working and
producing a sane number before any of it is wired into train.py.

    python check_metrics.py --framework snntorch
    python check_metrics.py --framework norse --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from check_network import data_info_without_download
from src.adapters import IMPLEMENTED, lif_factory
from src.config import ConfigError, load_config, require
from src.metrics import (
    PowerSampler,
    detect_update_interval_ms,
    dynamic_energy_j,
    energy_warnings,
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
from src.network import build_network


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--framework", default="snntorch",
                        help=f"implemented: {IMPLEMENTED}")
    parser.add_argument("--device", default=None, help="cuda | cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--latency-samples", type=int, default=20)
    parser.add_argument("--idle-seconds", type=float, default=3.0,
                        help="short here; a real run uses longer")
    args = parser.parse_args()

    config = load_config(args.config)
    info = data_info_without_download(require(config, "dataset"))
    neuron_cfg = require(config, "neuron")

    requested = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    print(f"framework {args.framework}   device {device}   "
          f"T={info.time_steps}  batch={args.batch}")

    net = build_network(lif_factory(args.framework, neuron_cfg), info, seed=0).to(device)
    batch = torch.rand(
        info.time_steps, args.batch, info.channels, info.height, info.width,
        device=device,
    )

    # ---- timing -----------------------------------------------------------
    print("\n" + "=" * 68)
    print("TIMING  (synchronised on both sides)")
    print("=" * 68)
    with torch.no_grad():
        for _ in range(3):                      # warm-up: compile the kernels
            net(batch)
        with timed(device) as forward_timer:
            for _ in range(5):
                net(batch)
    per_pass = forward_timer["seconds"] / 5
    print(f"  5 forward passes      {forward_timer['seconds']:.4f} s")
    print(f"  per pass              {per_pass * 1000:.2f} ms")
    print(f"  throughput            {args.batch / per_pass:.1f} samples/s")
    print("  ^ samples/s, NOT ms/sample -- throughput is a rate")

    # ---- memory -----------------------------------------------------------
    print("\n" + "=" * 68)
    print("MEMORY")
    print("=" * 68)
    reset_peak_memory(device)
    with torch.no_grad():
        net(batch)
    memory = peak_memory_mb(device)
    if memory["allocated_mb"] is None:
        print("  CPU device -- no GPU memory stats (both fields will be empty)")
    else:
        print(f"  peak allocated        {memory['allocated_mb']:.1f} MB")
        print(f"  peak reserved         {memory['reserved_mb']:.1f} MB")
        print(f"  reserved/allocated    {memory['reserved_mb'] / memory['allocated_mb']:.2f}x")

    # ---- spike counting ---------------------------------------------------
    print("\n" + "=" * 68)
    print("SPIKE ACTIVITY")
    print("=" * 68)
    with spike_counting(net):
        with torch.no_grad():
            net(batch)
        activity = spike_rates(net)

    print(f"  {'layer':<6}{'type':<18}{'neurons':>10}{'spikes':>14}"
          f"{'opportunities':>16}{'rate %':>10}{'/neuron/inf':>14}")
    for row in activity["layers"]:
        print(f"  {row['layer_index']:<6}{row['layer_type']:<18}"
              f"{row['neurons']:>10}{row['total_spikes']:>14.0f}"
              f"{row['opportunities']:>16}{row['spike_rate_pct']:>10.3f}"
              f"{row['spikes_per_neuron_per_inference']:>14.4f}")
    print(f"  overall (spike-weighted): {activity['spike_rate_pct']:.3f}%  "
          f"= {activity['spikes_per_neuron_per_inference']:.4f} spikes/neuron/inference")

    # Counting must be OFF again, or every later timing is polluted.
    still_counting = [type(l).__name__ for l in net.lif_layers() if l.count_spikes]
    print(f"  counting still enabled after the block: {still_counting or 'no'}"
          f"   (must be 'no')")

    # Sanity: hand-check the arithmetic on the first layer.
    first = activity["layers"][0]
    expected = first["neurons"] * info.time_steps * args.batch
    print(f"  arithmetic check: neurons x T x batch = {first['neurons']} x "
          f"{info.time_steps} x {args.batch} = {expected}, "
          f"reported {first['opportunities']}  "
          f"{'OK' if expected == first['opportunities'] else 'MISMATCH'}")

    # ---- latency ----------------------------------------------------------
    print("\n" + "=" * 68)
    print("LATENCY  (batch_size = 1, one sample at a time)")
    print("=" * 68)
    singles = [
        torch.rand(info.time_steps, 1, info.channels, info.height, info.width,
                   device=device)
        for _ in range(args.latency_samples)
    ]
    latency = measure_latency(net, singles, device)
    print(f"  median                {latency['latency_ms']:.3f} ms   <- headline")
    print(f"  mean                  {latency['latency_mean_ms']:.3f} ms")
    print(f"  p90                   {latency['latency_p90_ms']:.3f} ms   (MLPerf convention)")
    print(f"  min / max             {latency['latency_min_ms']:.3f} / "
          f"{latency['latency_max_ms']:.3f} ms")
    print(f"  samples               {latency['latency_samples']}")

    # ---- energy -----------------------------------------------------------
    print("\n" + "=" * 68)
    print("ENERGY  (NVML)")
    print("=" * 68)
    handle = nvml_handle()
    if handle is None:
        print("  NVML unavailable or no power sensor on this device.")
        print("  Energy columns will be written EMPTY -- runs do not fail.")
    else:
        interval_ms = detect_update_interval_ms(handle)
        if interval_ms is None:
            print("  sensor update interval: could not determine "
                  "(reading never changed -- GPU too idle)")
            poll_s = 0.05
        else:
            print(f"  sensor update interval: {interval_ms:.1f} ms "
                  f"({1000 / interval_ms:.1f} Hz)")
            print("  ^ literature reports ~10 Hz on most NVIDIA GPUs; the doc's "
                  "10-20 ms is far faster than the sensor")
            poll_s = max(interval_ms / 2000.0, 0.005)

        print(f"\n  idle baseline ({args.idle_seconds:.0f} s, nothing running)...")
        idle = measure_idle_power(handle, args.idle_seconds, poll_s)
        print(f"    mean {idle['mean_w']:.2f} W   range "
              f"{idle['min_w']:.2f}-{idle['max_w']:.2f} W   ({idle['n']} samples)")

        print("\n  under load (repeated forward passes)...")
        sampler = PowerSampler(handle, poll_s)
        sampler.start()
        with timed(device) as load_timer:
            with torch.no_grad():
                deadline = time.perf_counter() + 5.0
                while time.perf_counter() < deadline:
                    net(batch)
        samples = sampler.stop()
        loaded = summarise_power(samples)
        total_j = integrate_energy_j(samples)
        dynamic_j = dynamic_energy_j(total_j, idle["mean_w"], load_timer["seconds"])

        print(f"    mean {loaded['mean_w']:.2f} W   range "
              f"{loaded['min_w']:.2f}-{loaded['max_w']:.2f} W   ({loaded['n']} samples)")
        print(f"    duration              {load_timer['seconds']:.2f} s")
        print(f"    total energy          {total_j:.2f} J")
        print(f"    dynamic energy        {dynamic_j:.2f} J  "
              f"(total - {idle['mean_w']:.2f} W x {load_timer['seconds']:.2f} s)")

        problems = energy_warnings(
            total_j, dynamic_j, idle, loaded, load_timer["seconds"], interval_ms
        )
        if problems:
            print(f"\n  {len(problems)} REASON(S) NOT TO TRUST THIS MEASUREMENT:")
            for problem in problems:
                print(f"    - {problem}")
        else:
            print("\n  no validity warnings")
        print("\n  ^ even when clean: a relative comparison between frameworks on")
        print("    THIS machine. NVML error vs a physical meter can reach 73%.")

    print("\ndone")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as error:
        print(f"\nCONFIG ERROR: {error}", file=sys.stderr)
        sys.exit(2)
