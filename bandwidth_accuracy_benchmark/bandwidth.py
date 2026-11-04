"""Encoded-byte accounting and coordinate helpers."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


def encode_jpeg(image: np.ndarray, quality: int) -> tuple[int, float]:
    if image is None or not image.size:
        raise ValueError("Cannot JPEG-encode an empty image")
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in [1,100]")
    started = time.perf_counter()
    success, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not success:
        raise RuntimeError("OpenCV JPEG encoding failed")
    return int(encoded.nbytes), elapsed_ms


def scale_boxes(
    boxes_xyxy: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Scale Nx4 boxes between physical image coordinate systems."""

    result = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
    if not len(result) or source_size == target_size:
        return result
    source_width, source_height = source_size
    target_width, target_height = target_size
    result[:, [0, 2]] *= target_width / source_width
    result[:, [1, 3]] *= target_height / source_height
    result[:, [0, 2]] = result[:, [0, 2]].clip(0, target_width)
    result[:, [1, 3]] = result[:, [1, 3]].clip(0, target_height)
    return result


def clipped_crop(
    image: np.ndarray, box_xyxy: np.ndarray, pad_fraction: float = 0.0
) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(float, box_xyxy)
    box_width, box_height = x2 - x1, y2 - y1
    x1 -= box_width * pad_fraction
    x2 += box_width * pad_fraction
    y1 -= box_height * pad_fraction
    y2 += box_height * pad_fraction
    left = max(0, min(width, int(np.floor(x1))))
    top = max(0, min(height, int(np.floor(y1))))
    right = max(0, min(width, int(np.ceil(x2))))
    bottom = max(0, min(height, int(np.ceil(y2))))
    return image[top:bottom, left:right]


def periodic_is_hr(frame_index: int, interval: int) -> bool:
    if interval <= 0:
        raise ValueError("Periodic interval must be positive")
    return frame_index % interval == 0


def bandwidth_summary(
    *, lr_bytes: int, hr_bytes: int, frames: int, fps: float,
    lr_frames: int, hr_frames: int, hr_crops: int,
) -> dict:
    total = int(lr_bytes) + int(hr_bytes)
    duration = frames / fps if fps > 0 else 0.0
    return {
        "lr_bytes": int(lr_bytes),
        "hr_bytes": int(hr_bytes),
        "total_bytes": total,
        "total_mb": total / 1_000_000.0,
        "duration_seconds": duration,
        "mb_per_second": total / 1_000_000.0 / duration if duration else 0.0,
        "megabits_per_second": 8.0 * total / 1_000_000.0 / duration if duration else 0.0,
        "bytes_per_frame": total / frames if frames else 0.0,
        "lr_frames": int(lr_frames),
        "hr_frames": int(hr_frames),
        "hr_crops": int(hr_crops),
        "measurement": "in-memory JPEG encoded bytes; not measured network throughput",
    }


def resolve_frame(sequence_dir: Path, frame_id: int) -> Path:
    candidates = [
        sequence_dir / "img1" / f"{frame_id:06d}{suffix}"
        for suffix in (".jpg", ".png", ".jpeg", ".bmp")
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one image for frame {frame_id} in {sequence_dir / 'img1'}"
        )
    return matches[0]

