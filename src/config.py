"""Load the YAML config and read values out of it strictly.

The whole point of this module is the word *strictly*: if a value is missing or
the wrong type, you get an immediate error naming the exact key, instead of a
silent default that quietly turns your benchmark into a comparison of three
different neurons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when the config is missing a value or has the wrong type."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay `override` on `base`. Lists are replaced, not merged."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, _chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a YAML config file into a dict.

    A config may start with `extends: other.yaml` (resolved relative to its own
    directory) and then override only the keys it cares about. Everything else
    is inherited.

    This exists to stop config duplication. In a study whose whole claim is
    "everything held equal", two near-identical config files are a real hazard:
    change batch_size in one and forget the other, and the comparison is
    silently invalid. With `extends`, shared values exist in exactly one place.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path.resolve()}")

    resolved = path.resolve()
    if resolved in _chain:
        loop = " -> ".join(p.name for p in (*_chain, resolved))
        raise ConfigError(f"circular 'extends' in config files: {loop}")

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ConfigError(f"config file {path} must contain a YAML mapping at the top level")

    parent_name = loaded.pop("extends", None)
    if parent_name is None:
        return loaded
    if not isinstance(parent_name, str):
        raise ConfigError(
            f"'extends' in {path} must be a filename, got {parent_name!r}"
        )

    base = load_config(path.parent / parent_name, _chain=(*_chain, resolved))
    return _deep_merge(base, loaded)


def require(config: dict[str, Any], dotted_key: str) -> Any:
    """Fetch a nested value, e.g. require(cfg, "neuron.norse.dt").

    Raises ConfigError naming the full path if any part of it is missing, and
    lists the keys that *do* exist at the point of failure -- which is usually
    enough to spot a typo without opening the YAML.
    """
    node: Any = config
    walked: list[str] = []

    for part in dotted_key.split("."):
        if not isinstance(node, dict):
            location = ".".join(walked) or "<top level>"
            raise ConfigError(
                f"cannot read '{dotted_key}': '{location}' is a "
                f"{type(node).__name__}, not a mapping"
            )
        if part not in node:
            location = ".".join(walked) or "<top level>"
            available = ", ".join(sorted(map(str, node.keys()))) or "(nothing)"
            raise ConfigError(
                f"missing config key '{dotted_key}'. "
                f"'{location}' contains: {available}"
            )
        node = node[part]
        walked.append(part)

    return node


def require_float(config: dict[str, Any], dotted_key: str) -> float:
    """Same as require(), but insists the value is a number and returns a float."""
    value = require(config, dotted_key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"config key '{dotted_key}' must be a number, got {value!r} "
            f"({type(value).__name__})"
        )
    return float(value)


def require_int(config: dict[str, Any], dotted_key: str) -> int:
    """Same as require(), but insists the value is a whole number."""
    value = require(config, dotted_key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"config key '{dotted_key}' must be a whole number, got {value!r} "
            f"({type(value).__name__})"
        )
    return value


def require_optional_int(config: dict[str, Any], dotted_key: str) -> int | None:
    """Same as require_int(), but the value may also be null to mean "off".

    The key must still be PRESENT -- an explicit `null` is a decision, a missing
    key is an oversight, and they should not look the same.
    """
    value = require(config, dotted_key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"config key '{dotted_key}' must be a whole number or null, got "
            f"{value!r} ({type(value).__name__})"
        )
    return value


def require_bool(config: dict[str, Any], dotted_key: str) -> bool:
    """Same as require(), but insists the value is true/false.

    Worth being strict here: YAML turns the unquoted word `no` into the string
    "no" in some styles, and a truthy string would silently invert a setting
    like decay_input.
    """
    value = require(config, dotted_key)
    if not isinstance(value, bool):
        raise ConfigError(
            f"config key '{dotted_key}' must be true or false, got {value!r} "
            f"({type(value).__name__})"
        )
    return value


def require_str(config: dict[str, Any], dotted_key: str) -> str:
    """Same as require(), but insists the value is text."""
    value = require(config, dotted_key)
    if not isinstance(value, str):
        raise ConfigError(
            f"config key '{dotted_key}' must be text, got {value!r} "
            f"({type(value).__name__})"
        )
    return value


LOCAL_RUNS = Path("local_runs")


def output_dirs(
    experiment: str | None, results_root: str = "experiments"
) -> tuple[Path, Path, bool]:
    """Where output goes: (results_dir, equivalence_dir, flat).

    Config files know NOTHING about this. A config describes the experiment's
    science; the command line decides the folder and the machine:

        --experiment ex2        which folder, chosen per run
        --results-root <path>   where folders live, chosen per machine

    With no --experiment, everything lands flat in `local_runs/`. Filenames carry
    timestamps, so scratch work needs neither a hierarchy nor a naming decision.
    """
    if experiment is None:
        return LOCAL_RUNS, LOCAL_RUNS, True
    base = Path(results_root) / experiment
    return base / "results", base / "equivalence", False


def run_banner(
    script: str,
    *,
    experiment: str | None = None,
    config_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    framework: str | None = None,
    output_dir: str | Path | None = None,
    extra: dict[str, Any] | None = None,
    writes_results: bool = True,
) -> str:
    """The identity block every script prints before doing anything.

    One shared formatter so all scripts announce the same facts the same way. The
    point is that a scrolled-back terminal, or a pasted snippet in a lab notebook,
    still says WHICH experiment and WHICH config produced what follows -- the two
    things that decide whether a number means anything.

    `experiment` is printed even when it is None, because "I forgot --experiment"
    and "I meant scratch" look identical afterwards otherwise.

    `writes_results=False` for the check scripts, which produce no files. For those
    `--experiment` is a LABEL: it says which experiment you are checking for, and
    changes nothing. Saying so in the banner stops the label being mistaken for an
    output path.
    """
    lines = ["=" * 74, script]

    def row(label: str, value: Any) -> None:
        lines.append(f"  {label:<12}{value}")

    if experiment:
        row("experiment", experiment + ("" if writes_results else "   (label only)"))
    elif writes_results:
        row("experiment", "none (scratch -> local_runs/)")
    else:
        row("experiment", "not stated -- pass --experiment to label this check")
    if config_path is not None:
        digest = ""
        if config is not None:
            from src.results import config_hash  # local: avoids a circular import

            digest = f"   hash {config_hash(config)}"
        row("config", f"{config_path}{digest}")
    if framework is not None:
        row("framework", framework)
    if output_dir is not None:
        row("writing to", output_dir)
    for label, value in (extra or {}).items():
        row(label, value)

    lines.append("=" * 74)
    return "\n".join(lines)


def ephemeral_storage_warning(results_root: Path) -> str | None:
    """On Colab, is this path about to be deleted when the session ends?

    /content is wiped on disconnect. Writing results there has already cost one
    run, so this is checked before anything expensive starts rather than
    discovered afterwards.
    """
    on_colab = Path("/content").is_dir()
    if not on_colab:
        return None
    absolute = results_root.resolve()
    if str(absolute).startswith("/content/drive"):
        return None  # Google Drive, persistent
    return (
        f"results would be written to {absolute}, which is on Colab's temporary "
        f"disk and is DELETED when the session ends.\n"
        f"  Fix: --results-root /content/drive/MyDrive/snn_results  "
        f"(after mounting Drive)\n"
        f"  Or pass --allow-ephemeral if you really mean it."
    )


def require_choice(config: dict[str, Any], dotted_key: str, allowed: list[str]) -> str:
    """Same as require_str(), but the value must be one of `allowed`."""
    value = require_str(config, dotted_key)
    if value not in allowed:
        raise ConfigError(
            f"config key '{dotted_key}' must be one of {allowed}, got '{value}'"
        )
    return value
