"""Protocol, artifact and paired-run helpers for the ablation study."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from evaluate_mot import load_mot


VARIANTS = {
    "a0": "YOLO11l + BoT-SORT (LR-only)",
    "a1": "A0 + ECIA Base-only (no HR)",
    "a2": "A0 + HIVE + ECIA Full (rule trigger)",
    "a3": "A2 + Base-Guided gating + Safety",
}

EXPECTED_MODES = {
    "a1": "base_guided_hr_disabled",
    "a2": "online",
    "a3": "base_guided_safety_online",
}

EXPECTED_METRICS = {
    "quadtrack": {
        "a0": {"id_switches": 92, "idf1": 0.4476, "hota": 0.3618, "assa": 0.4362},
        "a3": {"id_switches": 78, "idf1": 0.4663, "hota": 0.3725, "assa": 0.4628},
    },
    "dtb": {
        "a0": {"id_switches": 94, "idf1": 0.3520, "hota": 0.3422, "assa": 0.2273},
        "a3": {"id_switches": 89, "idf1": 0.3722, "hota": 0.3525, "assa": 0.2408},
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_protocol(variant: str, run: Path) -> dict:
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    if variant == "a0":
        row = manifest["protocol"]
        return {
            "dataset_root": str(Path(row["dataset_root"]).resolve()),
            "sequences": list(row["sequences"]),
            "physical_lr_size": list(row["physical_lr_size"]),
            "imgsz": int(row["imgsz"]),
            "conf": float(row["conf"]),
            "iou": float(row["nms_iou"]),
            "detector": str(Path(row["weights"]).resolve()),
            "tracker": str(Path(row["botsort_config"]).resolve()),
        }
    return {
        "dataset_root": str(Path(manifest["dataset_root"]).resolve()),
        "sequences": list(manifest["sequences"]),
        "physical_lr_size": list(manifest["physical_lr_size"]),
        "imgsz": int(manifest["imgsz"]),
        "conf": float(manifest["conf"]),
        "iou": float(manifest["iou"]),
        "detector": str(Path(manifest["detector"]).resolve()),
        "tracker": str(Path(manifest["tracker_config"]).resolve()),
    }


def mot_path(variant: str, run: Path, sequence: str, *, raw: bool = False) -> Path:
    if variant == "a0":
        if raw:
            raise ValueError("A0 has no separate raw_mot directory")
        return run / "mot" / "botsort" / f"{sequence}.txt"
    return run / ("raw_mot" if raw else "mot") / f"{sequence}.txt"


def mot_equivalent(left: Path, right: Path, box_atol: float = 0.011) -> tuple[bool, list[dict]]:
    """Compare MOT rows while tolerating only serialization-level box rounding."""

    left_rows = load_mot(left, ground_truth=False)
    right_rows = load_mot(right, ground_truth=False)
    differences = []
    for frame in sorted(set(left_rows) | set(right_rows)):
        lhs = left_rows.get(frame, [])
        rhs = right_rows.get(frame, [])
        if len(lhs) != len(rhs):
            differences.append({"frame": frame, "reason": "row_count", "left": len(lhs), "right": len(rhs)})
            continue
        lhs = sorted(lhs, key=lambda row: row[0])
        rhs = sorted(rhs, key=lambda row: row[0])
        for a, b in itertools.zip_longest(lhs, rhs):
            if a is None or b is None or int(a[0]) != int(b[0]):
                differences.append({"frame": frame, "reason": "track_id", "left": None if a is None else int(a[0]), "right": None if b is None else int(b[0])})
                continue
            if not np.allclose(np.asarray(a[1]), np.asarray(b[1]), atol=box_atol, rtol=0.0):
                differences.append({"frame": frame, "track_id": int(a[0]), "reason": "box", "max_abs": float(np.max(np.abs(np.asarray(a[1]) - np.asarray(b[1]))))})
            elif not np.isclose(float(a[2]), float(b[2]), atol=1e-6, rtol=0.0):
                differences.append({"frame": frame, "track_id": int(a[0]), "reason": "confidence", "left": float(a[2]), "right": float(b[2])})
        if len(differences) >= 100:
            break
    return not differences, differences[:100]


def online_timing(run: Path, sequences: list[str]) -> dict:
    rows = [json.loads((run / "timing" / f"{sequence}.json").read_text(encoding="utf-8")) for sequence in sequences]
    frames = sum(int(row["frames"]) for row in rows)
    wall = sum(float(row["wall_seconds"]) for row in rows)
    hr = {
        key: sum(int(row["engine"]["hr_cost"].get(key, 0)) for row in rows)
        for key in ("requests", "crops", "bytes_read", "crop_pixels")
    }
    return {
        "frames": frames,
        "seconds": wall,
        "fps": frames / wall if wall else 0.0,
        "latency_ms": 1000.0 * wall / frames if frames else 0.0,
        "hr_runtime": hr,
        "scope": "detector + BoT-SORT + online controller; decode/resize/write excluded",
    }


def baseline_timing(run: Path, sequences: list[str]) -> dict:
    rows = [json.loads((run / "timing" / "botsort" / f"{sequence}.json").read_text(encoding="utf-8")) for sequence in sequences]
    frames = sum(int(row["frames"]) for row in rows)
    elapsed_ms = sum(float(row["end_to_end_ms_total"]) for row in rows)
    return {
        "frames": frames,
        "seconds": elapsed_ms / 1000.0,
        "fps": 1000.0 * frames / elapsed_ms if elapsed_ms else 0.0,
        "latency_ms": elapsed_ms / frames if frames else 0.0,
        "hr_runtime": {"requests": 0, "crops": 0, "bytes_read": 0, "crop_pixels": 0},
        "scope": "shared detector + cached BoT-SORT replay; decode/resize/write excluded",
    }


def logged_events(run: Path, sequences: list[str]) -> list[dict]:
    rows = []
    for sequence in sequences:
        for line in (run / "diagnostics" / f"{sequence}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("type") not in {"recovery_event", "base_guided_event"}:
                continue
            row["sequence"] = sequence
            rows.append(row)
    return rows

