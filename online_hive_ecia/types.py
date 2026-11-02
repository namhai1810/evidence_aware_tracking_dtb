"""Small data contracts shared by the online tracker integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True, order=True)
class TrackToken:
    """A raw BoT-SORT ID plus a generation protecting against ID reuse."""

    raw_id: int
    generation: int = 0

    def key(self) -> str:
        return f"{self.raw_id}:{self.generation}"


@dataclass
class TrackerSignal:
    """Online state exposed by an Ultralytics BoT-SORT track object."""

    raw_id: int
    bbox_lr: np.ndarray
    state: str
    frame_id: int | None = None
    start_frame: int | None = None
    score: float | None = None
    smooth_feature: np.ndarray | None = None
    current_feature: np.ndarray | None = None


@dataclass
class Observation:
    """One causal observation retained in the bounded online history."""

    frame_id: int
    token: TrackToken
    canonical_id: int
    bbox_lr: np.ndarray
    bbox_hr: np.ndarray
    confidence: float
    lr_crop: np.ndarray
    source_path: Path
    native_feature: np.ndarray | None = None
    descriptor: np.ndarray | None = None


@dataclass
class LostTrack:
    """Last known state for a track that left the active output set."""

    token: TrackToken
    canonical_id: int
    lost_frame: int
    last_observation: Observation
    predicted_bbox_lr: np.ndarray | None = None
    native_feature: np.ndarray | None = None


@dataclass
class TriggerProposal:
    """A GT-free Lost->New recovery hypothesis and its diagnostic signals."""

    frame_id: int
    old: LostTrack
    new_token: TrackToken
    new_bbox_lr: np.ndarray
    gap_frames: int
    motion_iou: float
    reid_similarity: float | None
    association_score: float
    association_margin: float
    ambiguous: bool
    reasons: list[str] = field(default_factory=list)
    pair_kind: str = "lost_to_new"

