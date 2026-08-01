"""Writing results: three CSVs plus one JSON per run.

The schema is FIXED and defined here, once. That is the whole point of this
module: runs come from different configs (`default.yaml`, `norse_super.yaml`,
future variants), on different machines, with metrics that may or may not be
measurable -- and none of that may be allowed to change the shape of the files.

Four guarantees:

  1. Fixed column set. A metric that could not be measured writes an EMPTY CELL,
     never a missing column.
  2. `schema_version` in every row.
  3. A header mismatch is a loud error, not a silent append that misaligns every
     subsequent row.
  4. Append-only. Rows are never rewritten, reordered or deduplicated.

Runs are told apart by `run_id` (the join key across all four files) and
`config_hash` (a hash of the FULLY RESOLVED config, after `extends:` merging),
so two runs stay distinguishable even if the config file is later edited.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1


class ResultsError(Exception):
    """Raised when writing results would corrupt an existing file."""


# ---------------------------------------------------------------------------
# The schema. Change these and you must bump SCHEMA_VERSION.
# ---------------------------------------------------------------------------

RUN_COLUMNS: list[str] = [
    # identity
    "schema_version", "run_id", "timestamp", "framework", "seed",
    "config_path", "config_hash",
    # setup
    "dataset", "time_steps", "batch_size", "num_workers", "binarize",
    "denoise_us", "epochs", "optimizer", "lr", "surrogate",
    # integrity -- the fairness evidence travels WITH the numbers
    "trainable_params", "weight_fingerprint",
    # accuracy
    "test_accuracy_pct", "train_loss_final", "test_loss_final",
    # speed
    "train_time_s", "train_time_per_epoch_s",
    "inference_throughput_samples_per_s",
    "inference_latency_bs1_ms", "inference_latency_bs1_mean_ms",
    "inference_latency_bs1_p90_ms",
    # activity
    "spike_rate_pct",
    # memory
    "peak_memory_train_mb", "peak_memory_infer_mb",
    "peak_reserved_train_mb", "peak_reserved_infer_mb",
    # energy (training run only)
    "nvml_update_interval_ms", "idle_power_cold_w", "idle_power_after_train_w",
    "train_energy_duration_s", "train_energy_j", "train_energy_dynamic_j",
    "energy_warnings",
    # environment
    "gpu_name", "driver", "cuda", "torch_version", "framework_version",
    "python_version", "platform",
    # free text -- e.g. "smoke run, --max-batches 20"
    "notes",
]

EPOCH_COLUMNS: list[str] = [
    "schema_version", "run_id", "epoch",
    "train_loss", "train_accuracy_pct",
    "test_loss", "test_accuracy_pct",
    "epoch_train_time_s", "spike_rate_pct",
]

LAYER_COLUMNS: list[str] = [
    "schema_version", "run_id", "layer_index", "layer_type",
    "neurons", "total_spikes", "opportunities", "spike_rate_pct",
]


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def make_run_id(framework: str, seed: int, when: datetime | None = None) -> str:
    """`<timestamp>_<framework>_seed<n>` -- unique, sortable, self-describing."""
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{framework}_seed{seed}"


def config_hash(config: dict[str, Any]) -> str:
    """Hash of the fully resolved config.

    Two runs differing in ANY config value get different hashes, which is what
    keeps a `norse_super.yaml` run distinguishable from a `default.yaml` one even
    though both write identical columns.
    """
    text = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def environment_info(framework: str) -> dict[str, Any]:
    """Machine and library versions, for the results row."""
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "gpu_name": "",
        "driver": "",
        "cuda": "",
        "framework_version": "",
    }

    try:
        if framework == "snntorch":
            import snntorch

            info["framework_version"] = snntorch.__version__
        elif framework == "norse":
            import norse

            info["framework_version"] = norse.__version__
        elif framework == "spikingjelly":
            from importlib.metadata import version

            info["framework_version"] = version("spikingjelly")
    except Exception:  # noqa: BLE001 - a missing version must not fail a run
        pass

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda or ""
        try:
            import pynvml

            pynvml.nvmlInit()
            info["driver"] = pynvml.nvmlSystemGetDriverVersion()
        except Exception:  # noqa: BLE001
            pass

    return info


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _format(value: Any) -> str:
    """None becomes an empty cell. Everything else becomes its plain text."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def append_row(path: Path, columns: list[str], row: dict[str, Any]) -> None:
    """Append one row, refusing to corrupt an existing file.

    Unknown keys are an error rather than being dropped: a typo in a column name
    would otherwise silently lose a measurement.
    """
    unknown = sorted(set(row) - set(columns))
    if unknown:
        raise ResultsError(
            f"unknown column(s) for {path.name}: {unknown}. "
            f"Add them to the schema in src/results.py and bump SCHEMA_VERSION."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    existing_header: list[str] | None = None
    if path.is_file() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle), None)

    if existing_header is not None and existing_header != columns:
        missing = sorted(set(columns) - set(existing_header))
        extra = sorted(set(existing_header) - set(columns))
        raise ResultsError(
            f"{path} was written with a different schema, so appending would "
            f"misalign every row.\n"
            f"  columns now expected but absent on disk: {missing or 'none'}\n"
            f"  columns on disk that no longer exist:    {extra or 'none'}\n"
            f"Rename or move the old file to keep it, then re-run."
        )

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if existing_header is None:
            writer.writerow(columns)
        writer.writerow([_format(row.get(column)) for column in columns])


def write_run_json(results_dir: Path, run_id: str, payload: dict[str, Any]) -> Path:
    """The complete record for one run: full config, versions, every metric.

    Exists because the reference doc wants the whole config snapshot, and a YAML
    stuffed into a CSV cell makes the CSV unreadable.
    """
    path = results_dir / "runs" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def write_results(
    results_dir: Path,
    run_row: dict[str, Any],
    epoch_rows: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
    json_payload: dict[str, Any],
) -> dict[str, Path]:
    """Write all four artefacts for one run."""
    results_dir = Path(results_dir)
    run_id = run_row["run_id"]

    for row in epoch_rows:
        row.setdefault("run_id", run_id)
        row.setdefault("schema_version", SCHEMA_VERSION)
    for row in layer_rows:
        row.setdefault("run_id", run_id)
        row.setdefault("schema_version", SCHEMA_VERSION)
    run_row.setdefault("schema_version", SCHEMA_VERSION)

    paths = {
        "runs": results_dir / "runs.csv",
        "epochs": results_dir / "epochs.csv",
        "layers": results_dir / "layers.csv",
    }
    append_row(paths["runs"], RUN_COLUMNS, run_row)
    for row in epoch_rows:
        append_row(paths["epochs"], EPOCH_COLUMNS, row)
    for row in layer_rows:
        append_row(paths["layers"], LAYER_COLUMNS, row)

    paths["json"] = write_run_json(results_dir, run_id, json_payload)
    return paths


def main() -> int:  # pragma: no cover - tiny self-test
    """`python -m src.results` writes two dummy rows and checks the guarantees."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        row = {name: None for name in RUN_COLUMNS}
        row.update({"run_id": "t1", "framework": "snntorch", "seed": 0,
                    "test_accuracy_pct": 95.0})
        write_results(root, dict(row), [], [], {"hello": "world"})
        row["run_id"] = "t2"
        write_results(root, dict(row), [], [], {"hello": "world"})

        lines = (root / "runs.csv").read_text(encoding="utf-8").strip().splitlines()
        print(f"rows after two writes: {len(lines) - 1} (expect 2)")

        try:
            append_row(root / "runs.csv", RUN_COLUMNS[:-1], {"run_id": "t3"})
        except ResultsError as error:
            print("header mismatch correctly refused:")
            print("  " + str(error).splitlines()[0])
        try:
            append_row(root / "runs.csv", RUN_COLUMNS, {"nonsense": 1})
        except ResultsError as error:
            print(f"unknown column correctly refused: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
