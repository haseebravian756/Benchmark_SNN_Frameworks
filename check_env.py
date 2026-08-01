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
    if not torch.cuda.is_available():
        return {"cuda": "not available (CPU-only build or no GPU)"}

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
        except Exception as exc:  # noqa: BLE001
            info["nvml_power_readable"] = f"NO - energy metric unavailable ({exc})"
        pynvml.nvmlShutdown()
    except Exception as exc:  # noqa: BLE001
        info["nvml_driver"] = f"NVML unavailable ({exc})"

    return info


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

    missing = [n for n, v in versions.items() if v.startswith("NOT INSTALLED")]
    print()
    if missing:
        print(f"FAIL: missing packages -> {', '.join(missing)}")
        return 1
    print("OK: all packages import successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
