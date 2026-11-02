"""YAML presets with dotted ``KEY=VALUE`` overrides."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


def merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``update`` into a copy of ``base``."""
    result = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_overrides(
    config: Mapping[str, Any], overrides: Iterable[str]
) -> dict[str, Any]:
    """Apply overrides such as ``model.hidden_dim=3``; values are parsed as YAML."""
    result = copy.deepcopy(dict(config))
    for item in overrides:
        key, separator, raw = item.partition("=")
        if not separator or not key:
            raise ValueError(f"Override must be KEY=VALUE, received {item!r}.")
        *parents, leaf = key.split(".")
        node = result
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return result


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return apply_overrides(config, overrides)


def write_config(config: Mapping[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
