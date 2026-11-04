"""Recompute JPEG bandwidth for frozen Existing Base-Guided Safety artifacts."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from evaluate_mot import load_mot
from mot_baseline_benchmark.protocol import list_frames, resolve_sequence_dir

from .bandwidth import (
    bandwidth_summary,
    clipped_crop,
    encode_jpeg,
    resolve_frame,
)


def raw_box_index(path: Path) -> dict[tuple[int, int], np.ndarray]:
    frames = load_mot(path, ground_truth=False)
    return {
        (int(frame), int(track_id)): np.asarray(box, dtype=np.float32)
        for frame, rows in frames.items()
        for track_id, box, _ in rows
    }


def requested_events(path: Path) -> dict:
    """Return all controller requests and only physically retrievable crops.

    A Base-guided event may request HR before HIVE discovers that its causal
    history has no candidate. Such an event is a request, but it transmits no
    crop and therefore contributes zero HR bytes.
    """

    rows = []
    request_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        hr = event.get("decision", {}).get("hr", {})
        hive = event.get("decision", {}).get("hive", {})
        if event.get("type") == "base_guided_event" and bool(hr.get("requested")):
            request_count += 1
            if (
                int(hr.get("crop_count", 0)) <= 0
                or "selected_frame" not in hive
                or "selected_raw_id" not in hive
            ):
                continue
            rows.append(
                {
                    "request_frame": int(event["frame"]),
                    "selected_frame": int(hive["selected_frame"]),
                    "selected_raw_id": int(hive["selected_raw_id"]),
                }
            )
    return {"requests": request_count, "transmissions": rows}


def account_hive_sequence(
    *, sequence: str, dataset_root: Path, hive_run: Path,
    lr_size: tuple[int, int], quality: int, fps: float,
) -> dict:
    sequence_dir = resolve_sequence_dir(dataset_root, sequence)
    frames = list_frames(sequence_dir)
    lr_bytes = hr_bytes = 0
    encode_ms = 0.0
    for path in frames:
        hr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if hr is None:
            raise FileNotFoundError(path)
        lr = cv2.resize(hr, lr_size, interpolation=cv2.INTER_AREA)
        encoded_bytes, elapsed = encode_jpeg(lr, quality)
        lr_bytes += encoded_bytes
        encode_ms += elapsed

    event_audit = requested_events(hive_run / "diagnostics" / f"{sequence}.jsonl")
    events = event_audit["transmissions"]
    boxes = raw_box_index(hive_run / "raw_mot" / f"{sequence}.txt")
    transmission_keys = []
    crop_pixels = 0
    for event in events:
        key = (event["selected_frame"], event["selected_raw_id"])
        if key not in boxes:
            raise RuntimeError(f"Missing selected HIVE box for {sequence}/{key}")
        source_path = resolve_frame(sequence_dir, event["selected_frame"])
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(source_path)
        # Match the frozen online recovery crop exactly: five percent context
        # is added around the HIVE-selected person box before ReID.
        crop = clipped_crop(image, boxes[key], pad_fraction=0.05)
        if not crop.size:
            raise RuntimeError(f"Empty selected HIVE crop for {sequence}/{key}")
        encoded_bytes, elapsed = encode_jpeg(crop, quality)
        hr_bytes += encoded_bytes
        encode_ms += elapsed
        crop_pixels += int(crop.shape[0] * crop.shape[1])
        transmission_keys.append(f"{key[0]}:{key[1]}")

    bandwidth = bandwidth_summary(
        lr_bytes=lr_bytes, hr_bytes=hr_bytes, frames=len(frames), fps=fps,
        lr_frames=len(frames), hr_frames=0, hr_crops=len(events),
    )
    bandwidth.update(
        {
            "jpeg_quality": quality,
            "jpeg_encoding_ms_total": encode_ms,
            "jpeg_encoding_ms_per_frame": encode_ms / len(frames) if frames else 0.0,
            "hr_crop_pixels": crop_pixels,
            "hr_requests": int(event_audit["requests"]),
            "physical_transmissions": len(events),
            "unique_crop_keys": len(set(transmission_keys)),
            "cache_hits": 0,
            "cache_policy": "disabled; repeated requests are counted again",
        }
    )
    return bandwidth


def aggregate(rows: dict[str, dict], fps: float) -> dict:
    result = bandwidth_summary(
        lr_bytes=sum(row["lr_bytes"] for row in rows.values()),
        hr_bytes=sum(row["hr_bytes"] for row in rows.values()),
        frames=sum(row["lr_frames"] + row["hr_frames"] for row in rows.values()),
        fps=fps,
        lr_frames=sum(row["lr_frames"] for row in rows.values()),
        hr_frames=sum(row["hr_frames"] for row in rows.values()),
        hr_crops=sum(row["hr_crops"] for row in rows.values()),
    )
    result.update(
        {
            "jpeg_encoding_ms_total": sum(row["jpeg_encoding_ms_total"] for row in rows.values()),
            "physical_transmissions": sum(row["physical_transmissions"] for row in rows.values()),
            "hr_requests": sum(row["hr_requests"] for row in rows.values()),
            "cache_hits": 0,
        }
    )
    result["jpeg_encoding_ms_per_frame"] = (
        result["jpeg_encoding_ms_total"] / result["lr_frames"]
        if result["lr_frames"] else 0.0
    )
    return result

