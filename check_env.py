"""Print the versions of everything this project depends on.

Run this first on any new machine (your laptop, Colab, Kaggle) to confirm the
environment is complete, and paste its output into your thesis appendix --
framework benchmarks are version-sensitive, so these numbers are a result too.

    python check_env.py
"""

from __future__ import annotations

import importlib
import platform
import sys

# (import name, human-readable name). Import name != pip name for some of these.
PACKAGES: list[tuple[str, str]] = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("snntorch", "snntorch"),
    ("norse", "norse"),
    ("spikingjelly", "spikingjelly"),
    ("tonic", "tonic"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("yaml", "pyyaml"),
    ("matplotlib", "matplotlib"),
    ("pynvml", "nvidia-ml-py"),
]


def package_versions() -> dict[str, str]:
    """Version string per package, or an explicit failure marker. Never raises."""
    found: dict[str, str] = {}
    for import_name, display_name in PACKAGES:
        try:
            module = importlib.import_module(import_name)
        except Exception as exc:  # noqa: BLE001 - we want to report any failure
            found[display_name] = f"NOT INSTALLED ({type(exc).__name__})"
            continue
        version = getattr(module, "__version__", None)
        if version is None:
            # spikingjelly has no __version__ attribute; fall back to pip metadata.
            try:
                from importlib.metadata import version as pkg_version

                version = pkg_version(display_name)
            except Exception:  # noqa: BLE001
                version = "installed (version unknown)"
        found[display_name] = str(version)
    return found


def gpu_info() -> dict[str, str]:
    """GPU name, memory and driver. Empty dict if there is no usable GPU."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return {}
    # NVML is probed regardless of whether TORCH can use the GPU: a CPU-only
    # torch build on a machine that has an NVIDIA card still reports power
    # perfectly well, and that is worth knowing before assuming energy cannot be
    # measured here.
    if not torch.cuda.is_available():
        info = {"cuda": "not available to torch (CPU-only build or no GPU)"}
    else:
        info = {
            "cuda": torch.version.cuda or "unknown",
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": str(torch.cuda.device_count()),
            "gpu_memory_gb": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}",
        }

    # Driver version and power-reading support both come from NVML, which is what
    # the energy metric uses. If power reads fail here, energy cannot be measured.
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info["nvml_driver"] = pynvml.nvmlSystemGetDriverVersion()
        try:
            milliwatts = pynvml.nvmlDeviceGetPowerUsage(handle)
            info["nvml_power_readable"] = f"yes ({milliwatts / 1000:.1f} W right now)"

            # How often does the sensor actually produce a NEW value? Published
            # figures for NVIDIA GPUs are around 10 Hz, i.e. ~100 ms -- far
            # slower than the 10-20 ms polling that is commonly assumed, and it
            # caps the resolution of every energy measurement. Detected, not
            # assumed; also run at the start of every training run.
            from src.metrics import detect_update_interval_ms

            interval = detect_update_interval_ms(handle)
            if interval is None:
                info["nvml_update_interval"] = (
                    "undetermined (reading never changed - GPU too idle to tell)"
                )
            else:
                info["nvml_update_interval"] = (
                    f"{interval:.0f} ms ({1000 / interval:.1f} Hz)"
                )
        except Exception as exc:  # noqa: BLE001
            info["nvml_power_readable"] = f"NO - energy metric unavailable ({exc})"
        pynvml.nvmlShutdown()
    except Exception as exc:  # noqa: BLE001
        info["nvml_driver"] = f"NVML unavailable ({exc})"

    return info


def norse_build() -> dict[str, str]:
    """Is Norse running compiled C++ here, or pure Python?

    v1.1.0's release notes say it "transformed Norse into a Python-only module by
    eliminating C++ code", so pure Python is EXPECTED on every machine rather
    than being a failed build. Recorded anyway, because it belongs in the
    write-up: this Norse is a pure-Python implementation being compared against
    libraries that may use compiled or fused kernels.
    """
    try:
        import norse
    except Exception as exc:  # noqa: BLE001
        return {"norse_build": f"not importable ({type(exc).__name__})"}

    from pathlib import Path

    root = Path(norse.__file__).parent
    compiled = [p.name for p in root.rglob("*.so")] + [p.name for p in root.rglob("*.pyd")]
    return {
        "norse_version": getattr(norse, "__version__", "unknown"),
        "norse_build": (
            f"compiled extensions present: {compiled}" if compiled
            else "pure Python (no compiled extension) - expected for v1.1.0"
        ),
    }


def main() -> int:
    print("=" * 62)
    print("ENVIRONMENT")
    print("=" * 62)
    print(f"{'python':<22} {platform.python_version()}")
    print(f"{'platform':<22} {platform.system()} {platform.release()}")
    print()

    versions = package_versions()
    for name, version in versions.items():
        print(f"{name:<22} {version}")

    print()
    print("=" * 62)
    print("GPU")
    print("=" * 62)
    info = gpu_info()
    if not info:
        print("torch missing - cannot query GPU")
    for key, value in info.items():
        print(f"{key:<22} {value}")

    print()
    print("=" * 62)
    print("NORSE BUILD")
    print("=" * 62)
    for key, value in norse_build().items():
        print(f"{key:<22} {value}")

    missing = [n for n, v in versions.items() if v.startswith("NOT INSTALLED")]
    print()
    if missing:
        print(f"FAIL: missing packages -> {', '.join(missing)}")
        return 1
    print("OK: all packages import successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
