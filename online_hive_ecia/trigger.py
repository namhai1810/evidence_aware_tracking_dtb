"""Rule-based, online-only ambiguity trigger for BoT-SORT Lost->New events."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from hr_finding.datasets.preprocessing import box_iou

from .types import LostTrack, TrackToken, TriggerProposal


def cosine(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    """Return a stable cosine or ``None`` when BoT-SORT exposes no feature."""

    if left is None or right is None:
        return None
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        return None
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else None


class AmbiguityTrigger:
    """Propose at most one plausible Lost identity for each fresh raw track.

    The trigger never sees GT, evaluator matches, future frames, or oracle switch
    events. It combines the lost/new lifecycle transition with current motion,
    native BoT-SORT appearance similarity, and the margin between competing
    lost identities. If the installed BoT-SORT does not expose an appearance
    vector, the trigger remains conservative and relies on motion only.
    """

    def __init__(self, config: dict) -> None:
        self.max_lost_age = int(config["max_lost_age"])
        self.max_new_age = int(config["max_new_age"])
        self.min_motion_iou = float(config["min_motion_iou"])
        self.min_reid_similarity = float(config["min_reid_similarity"])
        self.strong_reid_similarity = float(config["strong_reid_similarity"])
        self.min_association_score = float(config["min_association_score"])
        self.max_ambiguity_margin = float(config["max_ambiguity_margin"])
        self.motion_weight = float(config["motion_weight"])
        self.reid_weight = float(config["reid_weight"])
        self.require_native_reid_or_motion = bool(
            config.get("require_native_reid_or_motion", True)
        )
        if self.max_lost_age <= 0 or self.max_new_age <= 0:
            raise ValueError("trigger ages must be positive")
        if abs(self.motion_weight + self.reid_weight - 1.0) > 1e-6:
            raise ValueError("trigger motion_weight + reid_weight must equal 1")

    def _score(
        self,
        lost: LostTrack,
        new_bbox_lr: np.ndarray,
        new_feature: np.ndarray | None,
    ) -> tuple[float, float, float | None]:
        motion_box = (
            lost.predicted_bbox_lr
            if lost.predicted_bbox_lr is not None
            else lost.last_observation.bbox_lr
        )
        motion = box_iou(motion_box, new_bbox_lr)
        reid = cosine(lost.native_feature, new_feature)
        # Cosine is mapped from [-1,1] to [0,1]. A missing native feature is
        # deliberately neutral rather than silently replaced by GT or HR.
        reid_unit = 0.5 if reid is None else 0.5 * (reid + 1.0)
        score = self.motion_weight * motion + self.reid_weight * reid_unit
        return float(score), float(motion), reid

    def propose(
        self,
        *,
        frame_id: int,
        new_token: TrackToken,
        new_bbox_lr: np.ndarray,
        new_feature: np.ndarray | None,
        new_age: int,
        lost_tracks: list[LostTrack],
    ) -> TriggerProposal | None:
        if new_age > self.max_new_age:
            return None
        candidates = []
        for lost in lost_tracks:
            gap = int(frame_id - lost.lost_frame)
            if gap < 0 or gap > self.max_lost_age or lost.token == new_token:
                continue
            score, motion, reid = self._score(lost, new_bbox_lr, new_feature)
            has_pair_signal = motion >= self.min_motion_iou or (
                reid is not None and reid >= self.min_reid_similarity
            )
            if self.require_native_reid_or_motion and not has_pair_signal:
                continue
            candidates.append((score, motion, reid, gap, lost))
        if not candidates:
            return None

        candidates.sort(key=lambda row: (row[0], -row[3]), reverse=True)
        score, motion, reid, gap, lost = candidates[0]
        second = candidates[1][0] if len(candidates) > 1 else 0.0
        margin = float(score - second)
        ambiguous = len(candidates) > 1 and margin <= self.max_ambiguity_margin
        strong_reid = reid is not None and reid >= self.strong_reid_similarity
        if score < self.min_association_score and not strong_reid:
            return None

        reasons = ["lost_to_new"]
        if motion >= self.min_motion_iou:
            reasons.append("motion_plausible")
        if reid is not None and reid >= self.min_reid_similarity:
            reasons.append("reid_plausible")
        if ambiguous:
            reasons.append("association_ambiguous")
        if strong_reid:
            reasons.append("strong_reid")
        return TriggerProposal(
            frame_id=frame_id,
            old=lost,
            new_token=new_token,
            new_bbox_lr=np.asarray(new_bbox_lr, dtype=np.float32),
            gap_frames=gap,
            motion_iou=motion,
            reid_similarity=reid,
            association_score=score,
            association_margin=margin,
            ambiguous=ambiguous,
            reasons=reasons,
        )

    @staticmethod
    def diagnostics(proposal: TriggerProposal) -> dict:
        row = asdict(proposal)
        row["old_token"] = proposal.old.token.key()
        row["new_token"] = proposal.new_token.key()
        row.pop("old", None)
        row["new_bbox_lr"] = proposal.new_bbox_lr.tolist()
        return row

