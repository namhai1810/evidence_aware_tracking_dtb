"""Versioned, sequence-local cache for YOLO detections and native features."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CACHE_VERSION = 1


@dataclass
class DetectionCache:
    frame_ids: np.ndarray
    offsets: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    features: np.ndarray
    original_shapes: np.ndarray
    metadata: dict

    def frame(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return (
            self.boxes[start:end],
            self.scores[start:end],
            self.classes[start:end],
            self.features[start:end],
        )


def array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_cache(path: Path, cache: DetectionCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = array_digest(
        cache.frame_ids, cache.offsets, cache.boxes, cache.scores,
        cache.classes, cache.features, cache.original_shapes,
    )
    metadata = {**cache.metadata, "cache_version": CACHE_VERSION, "array_sha256": digest}
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            frame_ids=cache.frame_ids,
            offsets=cache.offsets,
            boxes=cache.boxes,
            scores=cache.scores,
            classes=cache.classes,
            features=cache.features,
            original_shapes=cache.original_shapes,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )


def load_cache(path: Path, expected_protocol_sha256: str | None = None) -> DetectionCache:
    with np.load(path, allow_pickle=False) as payload:
        cache = DetectionCache(
            frame_ids=payload["frame_ids"],
            offsets=payload["offsets"],
            boxes=payload["boxes"],
            scores=payload["scores"],
            classes=payload["classes"],
            features=payload["features"],
            original_shapes=payload["original_shapes"],
            metadata=json.loads(str(payload["metadata"].item())),
        )
    if int(cache.metadata.get("cache_version", -1)) != CACHE_VERSION:
        raise RuntimeError(f"Unsupported detection cache version in {path}")
    if expected_protocol_sha256 is not None and (
        cache.metadata.get("protocol_sha256") != expected_protocol_sha256
    ):
        raise RuntimeError(f"Detection cache protocol mismatch: {path}")
    actual = array_digest(
        cache.frame_ids, cache.offsets, cache.boxes, cache.scores,
        cache.classes, cache.features, cache.original_shapes,
    )
    if actual != cache.metadata.get("array_sha256"):
        raise RuntimeError(f"Detection cache checksum mismatch: {path}")
    if len(cache.offsets) != len(cache.frame_ids) + 1:
        raise RuntimeError(f"Invalid cache offsets: {path}")
    if int(cache.offsets[-1]) != len(cache.scores):
        raise RuntimeError(f"Invalid cache detection count: {path}")
    return cache

