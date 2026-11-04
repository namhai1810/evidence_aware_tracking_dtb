"""Adapters that consume one shared YOLO detection/feature cache."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML

from track_people import Sort
from online_hive_ecia.types import TrackerSignal


def _external_ocsort_class():
    """Load the repository's vendored reference OC-SORT without MMDet imports."""

    tracker_root = (
        Path(__file__).resolve().parents[1]
        / "external"
        / "OmniTrack"
        / "projects"
        / "mmdet3d_plugin"
        / "models"
        / "trackers"
    )
    value = str(tracker_root)
    if value not in sys.path:
        sys.path.insert(0, value)
    from ocsort_tracker.ocsort import OCSort

    return OCSort


def _tracker_args(config_path: Path, device: str) -> IterableSimpleNamespace:
    config = YAML.load(str(config_path.resolve()))
    config["device"] = torch.device(f"cuda:{device}" if device.isdigit() else device)
    return IterableSimpleNamespace(**config)


def clip_tracks_to_frame(tracks: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    """Mirror ``Results.update`` clipping applied by ``model.track``.

    Direct tracker APIs return unconstrained Kalman boxes, while Ultralytics'
    public tracking path clips x/y coordinates to the physical input image
    before exposing ``result.boxes``.  Apply the same operation so cached
    replay is byte-comparable with the original pipeline.
    """

    if len(tracks) == 0:
        return tracks
    height, width = frame_shape
    clipped = np.asarray(tracks, dtype=float).copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0.0, float(width))
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0.0, float(height))
    return clipped


class CachedTrackerAdapter:
    """Uniform wrapper for standard SORT, OC-SORT, ByteTrack and BoT-SORT."""

    def __init__(
        self,
        method: str,
        *,
        device: str,
        bytetrack_config: Path,
        botsort_config: Path,
        sort_max_age: int,
        sort_min_hits: int,
        sort_iou_threshold: float,
        ocsort_det_thresh: float = 0.25,
        ocsort_max_age: int = 30,
        ocsort_min_hits: int = 3,
        ocsort_iou_threshold: float = 0.30,
        ocsort_delta_t: int = 3,
        ocsort_inertia: float = 0.20,
        ocsort_use_byte: bool = False,
    ) -> None:
        self.method = method
        self._last_output_scores: dict[int, float] = {}
        if method == "sort":
            self.tracker = Sort(sort_max_age, sort_min_hits, sort_iou_threshold)
        elif method == "ocsort":
            self.tracker = _external_ocsort_class()(
                det_thresh=ocsort_det_thresh,
                max_age=ocsort_max_age,
                min_hits=ocsort_min_hits,
                iou_threshold=ocsort_iou_threshold,
                delta_t=ocsort_delta_t,
                inertia=ocsort_inertia,
                use_byte=ocsort_use_byte,
            )
        elif method == "bytetrack":
            self.tracker = BYTETracker(_tracker_args(bytetrack_config, device))
        elif method == "botsort":
            self.tracker = BOTSORT(_tracker_args(botsort_config, device))
        else:
            raise ValueError(f"Unknown tracker method: {method}")

    def update(
        self,
        *,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        features: np.ndarray,
        lr_frame: np.ndarray,
    ) -> np.ndarray:
        """Return Nx6 ``xyxy,id,score`` rows in physical-LR coordinates."""

        if self.method == "sort":
            detections = np.concatenate(
                (boxes_xyxy.astype(float), scores.reshape(-1, 1).astype(float)), axis=1
            ) if len(scores) else np.empty((0, 5), dtype=float)
            return clip_tracks_to_frame(self.tracker.update(detections), lr_frame.shape[:2])

        if self.method == "ocsort":
            detections = np.concatenate(
                (boxes_xyxy.astype(float), scores.reshape(-1, 1).astype(float)), axis=1
            ) if len(scores) else np.empty((0, 5), dtype=float)
            result = self.tracker.update(detections)
            rows = result[0] if isinstance(result, tuple) else result
            rows = np.asarray(rows, dtype=float).reshape(-1, 5)
            if len(rows) == 0:
                return np.empty((0, 6), dtype=float)
            score_by_id = {
                int(track.id) + 1: (
                    float(track.last_observation[4])
                    if getattr(track, "last_observation", np.empty(0)).size >= 5
                    and float(track.last_observation[4]) >= 0.0
                    else 1.0
                )
                for track in self.tracker.trackers
            }
            # A newly initialized reference OC-SORT track has no populated
            # ``last_observation`` yet. Recover its detector score from the
            # current geometry instead of emitting the placeholder 1.0.
            for row in rows:
                raw_id = int(row[4])
                track = next((value for value in self.tracker.trackers if int(value.id) + 1 == raw_id), None)
                observation = getattr(track, "last_observation", np.empty(0)) if track is not None else np.empty(0)
                if observation.size >= 5 and float(observation[0]) >= 0.0:
                    continue
                if len(boxes_xyxy):
                    xx1 = np.maximum(row[0], boxes_xyxy[:, 0])
                    yy1 = np.maximum(row[1], boxes_xyxy[:, 1])
                    xx2 = np.minimum(row[2], boxes_xyxy[:, 2])
                    yy2 = np.minimum(row[3], boxes_xyxy[:, 3])
                    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
                    row_area = max(0.0, row[2] - row[0]) * max(0.0, row[3] - row[1])
                    det_area = np.maximum(0.0, boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * np.maximum(0.0, boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
                    iou = inter / np.maximum(row_area + det_area - inter, 1e-6)
                    score_by_id[raw_id] = float(scores[int(np.argmax(iou))])
            self._last_output_scores = dict(score_by_id)
            output = np.concatenate(
                (
                    rows[:, :5],
                    np.asarray([score_by_id.get(int(row[4]), 1.0) for row in rows], dtype=float).reshape(-1, 1),
                ),
                axis=1,
            )
            return clip_tracks_to_frame(output, lr_frame.shape[:2])

        data = np.concatenate(
            (
                boxes_xyxy.astype(np.float32),
                scores.reshape(-1, 1).astype(np.float32),
                classes.reshape(-1, 1).astype(np.float32),
            ),
            axis=1,
        ) if len(scores) else np.empty((0, 6), dtype=np.float32)
        detections = Boxes(data, orig_shape=lr_frame.shape[:2]).cpu().numpy()
        native_features = features if self.method == "botsort" else None
        rows = self.tracker.update(
            detections,
            img=lr_frame,
            feats=native_features,
        )
        if len(rows) == 0:
            return np.empty((0, 6), dtype=float)
        # Ultralytics returns [x1,y1,x2,y2,id,score,class,detection_index].
        return clip_tracks_to_frame(np.asarray(rows[:, :6], dtype=float), lr_frame.shape[:2])

    def signals(self, frame_id: int) -> tuple[dict[int, TrackerSignal], dict[int, TrackerSignal]]:
        """Expose causal active/lost lifecycle state to the recovery controller."""

        if self.method not in {"sort", "ocsort"}:
            raise RuntimeError(f"signals() is intended for SORT-family adapters, got {self.method}")
        active: dict[int, TrackerSignal] = {}
        lost: dict[int, TrackerSignal] = {}
        for track in self.tracker.trackers:
            raw_id = int(track.id) + (1 if self.method == "ocsort" else 0)
            if self.method == "ocsort" and getattr(track, "last_observation", np.empty(0)).size >= 4 and float(track.last_observation[0]) >= 0.0:
                bbox = np.asarray(track.last_observation[:4], dtype=np.float32)
                score = float(track.last_observation[4]) if track.last_observation.size >= 5 else None
            else:
                state_fn = getattr(track, "get_state", None) or getattr(track, "state", None)
                bbox = np.asarray(state_fn(), dtype=np.float32).reshape(-1)[:4]
                score = float(getattr(track, "score", 1.0))
            if raw_id in self._last_output_scores:
                score = self._last_output_scores[raw_id]
            time_since_update = int(getattr(track, "time_since_update", 0))
            signal = TrackerSignal(
                raw_id=raw_id,
                bbox_lr=bbox,
                state="tracked" if time_since_update == 0 else "lost",
                frame_id=frame_id - time_since_update,
                start_frame=None,
                score=score,
                smooth_feature=None,
                current_feature=None,
            )
            (active if time_since_update == 0 else lost)[raw_id] = signal
        return active, lost

