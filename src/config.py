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


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file into a dict."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ConfigError(f"config file {path} must contain a YAML mapping at the top level")
    return loaded


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


def require_choice(config: dict[str, Any], dotted_key: str, allowed: list[str]) -> str:
    """Same as require_str(), but the value must be one of `allowed`."""
    value = require_str(config, dotted_key)
    if value not in allowed:
        raise ConfigError(
            f"config key '{dotted_key}' must be one of {allowed}, got '{value}'"
        )
    return value
