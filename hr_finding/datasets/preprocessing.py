"""LR-only image formation, crop extraction, and tracking metadata helpers."""

from __future__ import annotations

import csv
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


class FrameCache:
    """Small per-worker LRU cache for original panorama frames."""

    def __init__(self, max_items: int = 32) -> None:
        self.max_items = max(1, max_items)
        self._items: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self._items:
            image = self._items.pop(key)
            self._items[key] = image
            return image
        image = cv2.imread(key, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        self._items[key] = image
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)
        return image


def clamp_xyxy(box: list[float] | np.ndarray, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = (float(value) for value in box)
    left = max(0, min(width - 1, int(np.floor(x1))))
    top = max(0, min(height - 1, int(np.floor(y1))))
    right = max(left + 1, min(width, int(np.ceil(x2))))
    bottom = max(top + 1, min(height, int(np.ceil(y2))))
    return [left, top, right, bottom]


def crop_xyxy(image: np.ndarray, box: list[float] | np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clamp_xyxy(box, width, height)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Empty crop after clamping: {box}")
    return crop


def downsample_full_frame(image: np.ndarray, lowres_hw: tuple[int, int]) -> np.ndarray:
    """Reproduce the real LR panorama before cropping; never resize an HR crop."""
    height, width = lowres_hw
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def scale_box_to_lr(
    box: list[float] | np.ndarray,
    source_hw: tuple[int, int],
    lowres_hw: tuple[int, int],
) -> np.ndarray:
    source_h, source_w = source_hw
    lowres_h, lowres_w = lowres_hw
    scaled = np.asarray(box, dtype=np.float32).copy()
    scaled[[0, 2]] *= lowres_w / source_w
    scaled[[1, 3]] *= lowres_h / source_h
    return scaled


def reconstruct_lr_crop(
    source_image: Path,
    box_in_source_coordinates: list[float] | np.ndarray,
    lowres_hw: tuple[int, int],
    cache: FrameCache,
) -> np.ndarray:
    """Form the full LR panorama first and then crop the requested object."""
    original = cache.get(source_image)
    lr_frame = downsample_full_frame(original, lowres_hw)
    lr_box = scale_box_to_lr(box_in_source_coordinates, original.shape[:2], lowres_hw)
    return crop_xyxy(lr_frame, lr_box)


def letterbox(image: np.ndarray, output_hw: tuple[int, int], fill: int = 114) -> np.ndarray:
    output_h, output_w = output_hw
    height, width = image.shape[:2]
    scale = min(output_w / max(width, 1), output_h / max(height, 1))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((output_h, output_w, 3), fill, dtype=np.uint8)
    left = (output_w - resized_w) // 2
    top = (output_h - resized_h) // 2
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas


def image_to_tensor(
    image_bgr: np.ndarray,
    output_hw: tuple[int, int],
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    image = cv2.cvtColor(letterbox(image_bgr, output_hw), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)
    mean_tensor = tensor.new_tensor(mean).view(3, 1, 1)
    std_tensor = tensor.new_tensor(std).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def load_materialized_crop(path: Path, output_hw: tuple[int, int]) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image_to_tensor(image, output_hw)


def load_mot_tracks(path: Path) -> dict[int, list[tuple[int, np.ndarray, float]]]:
    frames: dict[int, list[tuple[int, np.ndarray, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 6:
                continue
            frame = int(float(row[0]))
            track_id = int(float(row[1]))
            x, y, width, height = map(float, row[2:6])
            confidence = float(row[6]) if len(row) > 6 else 1.0
            if width > 0 and height > 0 and confidence > 0:
                frames[frame].append(
                    (
                        track_id,
                        np.asarray([x, y, x + width, y + height], dtype=np.float32),
                        confidence,
                    )
                )
    return frames


def box_iou(box_a: list[float] | np.ndarray, box_b: list[float] | np.ndarray) -> float:
    a = np.asarray(box_a, dtype=np.float32)
    b = np.asarray(box_b, dtype=np.float32)
    top_left = np.maximum(a[:2], b[:2])
    bottom_right = np.minimum(a[2:], b[2:])
    wh = np.maximum(0.0, bottom_right - top_left)
    intersection = float(wh[0] * wh[1])
    area_a = float(max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]))
    area_b = float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_lr_detection(
    rows: list[tuple[int, np.ndarray, float]],
    reference_box: list[float] | np.ndarray,
    minimum_iou: float,
) -> tuple[tuple[int, np.ndarray, float] | None, float]:
    if not rows:
        return None, 0.0
    scores = [box_iou(reference_box, row[1]) for row in rows]
    index = int(np.argmax(scores))
    return (rows[index], scores[index]) if scores[index] >= minimum_iou else (None, scores[index])


def select_track_history(
    frames: dict[int, list[tuple[int, np.ndarray, float]]],
    track_id: int,
    target_frame: int,
    history_frames: int,
    max_samples: int,
) -> list[tuple[int, np.ndarray, float]]:
    observations: list[tuple[int, np.ndarray, float]] = []
    start = max(1, target_frame - history_frames)
    for frame in range(start, target_frame):
        for row_track_id, box, confidence in frames.get(frame, []):
            if row_track_id == track_id:
                observations.append((frame, box, confidence))
    observations.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[int, np.ndarray, float]] = []
    for item in observations:
        if all(abs(item[0] - other[0]) >= 2 for other in selected):
            selected.append(item)
            if len(selected) == max_samples:
                break
    return sorted(selected, key=lambda item: item[0])


def temporal_metadata(seconds_before: float, max_seconds: float = 5.0) -> torch.Tensor:
    age = min(1.0, max(0.0, seconds_before / max(max_seconds, 1e-6)))
    log_age = np.log1p(seconds_before) / np.log1p(max_seconds)
    return torch.tensor([age, age * age, float(log_age)], dtype=torch.float32)
