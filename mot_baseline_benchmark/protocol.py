"""Frozen protocol helpers shared by the MOT benchmark commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    sequences: tuple[str, ...]
    physical_lr_size: tuple[int, int]
    imgsz: int
    expected_botsort: dict[str, float | int]


DATASETS = {
    "quadtrack": DatasetSpec(
        name="quadtrack_512x120",
        sequences=("0016", "0017", "0025", "0031"),
        physical_lr_size=(512, 120),
        imgsz=512,
        expected_botsort={
            "id_switches": 92,
            "idf1": 0.4476,
            "hota": 0.3618,
            "assa": 0.4362,
        },
    ),
    "dtb": DatasetSpec(
        name="dtb_640x320",
        sequences=("normal_walking", "crossing", "partial_occ", "seam", "full_occ"),
        physical_lr_size=(640, 320),
        imgsz=640,
        expected_botsort={
            "id_switches": 94,
            "idf1": 0.3520,
            "hota": 0.3422,
            "assa": 0.2273,
        },
    ),
}


def parse_frame_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)[xX](\d+)", value.strip())
    if match is None:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT")
    width, height = map(int, match.groups())
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("WIDTH and HEIGHT must be positive")
    return width, height


def resolve_sequence_dir(root: Path, sequence: str) -> Path:
    candidates = [root / sequence, *(root / split / sequence for split in ("train", "val", "test"))]
    matches = [path for path in candidates if (path / "img1").is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one directory for {sequence!r} below {root}; "
            f"found {[str(path) for path in matches]}"
        )
    return matches[0]


def list_frames(sequence_dir: Path, max_frames: int | None = None) -> list[Path]:
    frames = sorted(
        path for path in (sequence_dir / "img1").iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise RuntimeError(f"No frames in {sequence_dir / 'img1'}")
    return frames


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def frame_manifest(frames: list[Path]) -> list[dict[str, int | str]]:
    return [
        {"name": path.name, "bytes": path.stat().st_size}
        for path in frames
    ]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

