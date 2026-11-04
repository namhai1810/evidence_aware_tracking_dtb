"""JPEG encoded-byte accounting for A0--A3."""

from __future__ import annotations

from pathlib import Path

import cv2

from bandwidth_accuracy_benchmark.bandwidth import clipped_crop, encode_jpeg, resolve_frame
from bandwidth_accuracy_benchmark.hive_accounting import raw_box_index
from mot_baseline_benchmark.protocol import list_frames, resolve_sequence_dir

from .protocol import logged_events


def lr_sequence(*, sequence: str, dataset_root: Path, lr_size: tuple[int, int], quality: int) -> dict:
    frames = list_frames(resolve_sequence_dir(dataset_root, sequence))
    total = 0
    for path in frames:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.resize(image, lr_size, interpolation=cv2.INTER_AREA)
        encoded, _ = encode_jpeg(image, quality)
        total += encoded
    return {"frames": len(frames), "lr_bytes": total}


def _selected(row: dict) -> tuple[int, int] | None:
    decision = row.get("decision", {})
    hr = decision.get("hr", {})
    hive = decision.get("hive", {})
    if not bool(hr.get("requested")) or int(hr.get("crop_count", 0)) <= 0:
        return None
    if "selected_frame" not in hive or "selected_raw_id" not in hive:
        return None
    return int(hive["selected_frame"]), int(hive["selected_raw_id"])


def hr_sequence(*, sequence: str, dataset_root: Path, run: Path, quality: int) -> dict:
    events = [row for row in logged_events(run, [sequence])]
    requests = sum(bool(row.get("decision", {}).get("hr", {}).get("requested")) for row in events)
    transmissions = [item for row in events if (item := _selected(row)) is not None]
    boxes = raw_box_index(run / "raw_mot" / f"{sequence}.txt")
    sequence_dir = resolve_sequence_dir(dataset_root, sequence)
    hr_bytes = 0
    for frame, raw_id in transmissions:
        key = (frame, raw_id)
        if key not in boxes:
            raise RuntimeError(f"Missing HIVE-selected raw box {sequence}/{key}")
        image = cv2.imread(str(resolve_frame(sequence_dir, frame)), cv2.IMREAD_COLOR)
        crop = clipped_crop(image, boxes[key], pad_fraction=0.05)
        encoded, _ = encode_jpeg(crop, quality)
        hr_bytes += encoded
    return {"hr_requests": int(requests), "hr_crops": len(transmissions), "hr_bytes": hr_bytes}


def dataset_bandwidth(*, dataset_root: Path, sequences: list[str], lr_size: tuple[int, int], quality: int, fps: float, online_runs: dict[str, Path]) -> dict:
    lr_rows = {sequence: lr_sequence(sequence=sequence, dataset_root=dataset_root, lr_size=lr_size, quality=quality) for sequence in sequences}
    frames = sum(row["frames"] for row in lr_rows.values())
    lr_bytes = sum(row["lr_bytes"] for row in lr_rows.values())
    duration = frames / fps
    result = {}
    for variant in ("a0", "a1", "a2", "a3"):
        hr_rows = (
            {sequence: hr_sequence(sequence=sequence, dataset_root=dataset_root, run=online_runs[variant], quality=quality) for sequence in sequences}
            if variant != "a0" else {}
        )
        hr_bytes = sum(row["hr_bytes"] for row in hr_rows.values())
        total = lr_bytes + hr_bytes
        result[variant] = {
            "jpeg_quality": quality,
            "lr_bytes": lr_bytes,
            "hr_bytes": hr_bytes,
            "total_bytes": total,
            "total_mb": total / 1_000_000.0,
            "mbps": 8.0 * total / 1_000_000.0 / duration if duration else 0.0,
            "hr_requests": sum(row["hr_requests"] for row in hr_rows.values()),
            "hr_crops": sum(row["hr_crops"] for row in hr_rows.values()),
            "measurement": "in-memory JPEG encoded bytes; not network throughput",
        }
    return result

