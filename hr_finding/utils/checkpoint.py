"""Atomic checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=False)
