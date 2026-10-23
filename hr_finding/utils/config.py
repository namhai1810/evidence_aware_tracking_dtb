"""Small YAML configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def save_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    result = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must use key=value: {override}")
        dotted_key, raw_value = override.split("=", 1)
        value: Any = yaml.safe_load(raw_value)
        target = result
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise KeyError(f"Unknown configuration path: {dotted_key}")
            target = target[key]
        if keys[-1] not in target:
            raise KeyError(f"Unknown configuration key: {dotted_key}")
        target[keys[-1]] = value
    return result


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()
