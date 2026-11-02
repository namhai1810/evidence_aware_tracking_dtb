"""Version-tolerant read-only adapter for Ultralytics BoT-SORT state."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .types import TrackerSignal


def _array(value) -> np.ndarray | None:
    if value is None:
        return None
    if callable(value):
        value = value()
    try:
        result = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    return result.copy() if result.size else None


def _bbox(track) -> np.ndarray | None:
    for name in ("tlbr", "xyxy"):
        result = _array(getattr(track, name, None))
        if result is not None and result.size >= 4:
            return result[:4]
    tlwh = _array(getattr(track, "tlwh", None))
    if tlwh is not None and tlwh.size >= 4:
        x, y, width, height = tlwh[:4]
        return np.asarray([x, y, x + width, y + height], dtype=np.float32)
    return None


def _track_id(track) -> int | None:
    value = getattr(track, "track_id", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _signals(tracks: Iterable, state: str) -> dict[int, TrackerSignal]:
    result = {}
    for track in tracks or ():
        raw_id = _track_id(track)
        bbox = _bbox(track)
        if raw_id is None or bbox is None:
            continue
        score = getattr(track, "score", None)
        result[raw_id] = TrackerSignal(
            raw_id=raw_id,
            bbox_lr=bbox,
            state=state,
            frame_id=(
                int(getattr(track, "frame_id"))
                if getattr(track, "frame_id", None) is not None
                else None
            ),
            start_frame=(
                int(getattr(track, "start_frame"))
                if getattr(track, "start_frame", None) is not None
                else None
            ),
            score=float(score) if score is not None else None,
            smooth_feature=_array(getattr(track, "smooth_feat", None)),
            current_feature=_array(getattr(track, "curr_feat", None)),
        )
    return result


def inspect_botsort(model) -> tuple[dict[int, TrackerSignal], dict[int, TrackerSignal]]:
    """Read active/lost states without changing tracker association internals."""

    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", ())
    if not trackers:
        return {}, {}
    tracker = trackers[0]
    active = _signals(getattr(tracker, "tracked_stracks", ()), "tracked")
    lost = _signals(getattr(tracker, "lost_stracks", ()), "lost")
    return active, lost

