"""GT-free candidate generation and frozen-ECIA-Base HR trigger policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .trigger import AmbiguityTrigger
from .types import LostTrack, Observation, TrackToken, TriggerProposal


@dataclass(frozen=True)
class BaseTriggerDecision:
    """Action derived only from frozen Base logits and validation thresholds."""

    action: str
    request_hr: bool
    direct_merge: bool
    old_probability: float
    old_margin: float
    old_competitor_margin: float
    top_probability: float
    normalized_entropy: float
    predicted_index: int
    identity_predicted_index: int
    abstain_probability: float
    reasons: tuple[str, ...]


class BaseGuidedCandidateProposer(AmbiguityTrigger):
    """Generate plausible pairs; ECIA Base, not rules, decides HR usage.

    BoT-SORT lifecycle, motion and native appearance are used only to bound and
    rank the candidate set.  The old rule trigger's association-score decision
    is deliberately not applied here.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config["candidate"])
        self.include_active_conflicts = bool(config["candidate"].get("include_active_conflicts", True))
        self.active_min_age = int(config["candidate"].get("active_min_age", 4))
        self.active_min_motion_iou = float(config["candidate"].get("active_min_motion_iou", 0.05))
        self.active_min_reid_similarity = float(
            config["candidate"].get("active_min_reid_similarity", 0.65)
        )

    def _active_as_lost(
        self,
        *,
        frame_id: int,
        observation: Observation,
    ) -> LostTrack:
        return LostTrack(
            token=observation.token,
            canonical_id=observation.canonical_id,
            lost_frame=frame_id,
            last_observation=observation,
            predicted_bbox_lr=observation.bbox_lr,
            native_feature=observation.native_feature,
        )

    def propose_candidates(
        self,
        *,
        frame_id: int,
        new_observation: Observation,
        new_age: int,
        lost_tracks: list[LostTrack],
        active_observations: list[Observation],
        observation_counts: dict[TrackToken, int],
    ) -> TriggerProposal | None:
        if new_age > self.max_new_age:
            return None
        candidates: list[tuple[float, float, float | None, int, LostTrack, str]] = []

        for old in lost_tracks:
            gap = int(frame_id - old.lost_frame)
            if gap < 0 or gap > self.max_lost_age or old.token == new_observation.token:
                continue
            score, motion, reid = self._score(
                old, new_observation.bbox_lr, new_observation.native_feature
            )
            if self.require_native_reid_or_motion and not (
                motion >= self.min_motion_iou
                or (reid is not None and reid >= self.min_reid_similarity)
            ):
                continue
            candidates.append((score, motion, reid, gap, old, "lost_to_new"))

        if self.include_active_conflicts:
            for active in active_observations:
                if active.token == new_observation.token:
                    continue
                if int(observation_counts.get(active.token, 0)) < self.active_min_age:
                    continue
                old = self._active_as_lost(frame_id=frame_id, observation=active)
                score, motion, reid = self._score(
                    old, new_observation.bbox_lr, new_observation.native_feature
                )
                if not (
                    motion >= self.active_min_motion_iou
                    or (reid is not None and reid >= self.active_min_reid_similarity)
                ):
                    continue
                candidates.append((score, motion, reid, 0, old, "active_to_active"))

        if not candidates:
            return None
        candidates.sort(key=lambda row: (row[0], -row[3]), reverse=True)
        score, motion, reid, gap, old, pair_kind = candidates[0]
        second = candidates[1][0] if len(candidates) > 1 else 0.0
        margin = float(score - second)
        reasons = [pair_kind, "botsort_candidate"]
        if motion >= self.min_motion_iou:
            reasons.append("motion_plausible")
        if reid is not None and reid >= self.min_reid_similarity:
            reasons.append("native_reid_plausible")
        if len(candidates) > 1 and margin <= self.max_ambiguity_margin:
            reasons.append("association_ambiguous")
        return TriggerProposal(
            frame_id=frame_id,
            old=old,
            new_token=new_observation.token,
            new_bbox_lr=np.asarray(new_observation.bbox_lr, dtype=np.float32),
            gap_frames=gap,
            motion_iou=motion,
            reid_similarity=reid,
            association_score=score,
            association_margin=margin,
            ambiguous="association_ambiguous" in reasons,
            reasons=reasons,
            pair_kind=pair_kind,
        )


class BaseGuidedHRPolicy:
    """Map frozen Base confidence to direct merge, HR request, or abstain."""

    def __init__(self, config: dict) -> None:
        self.direct_probability = float(config["direct_merge_min_old_probability"])
        self.direct_margin = float(config["direct_merge_min_old_margin"])
        self.request_min_probability = float(config["request_min_old_probability"])
        self.request_max_probability = float(config["request_max_old_probability"])
        self.request_max_margin = float(config["request_max_competitor_margin"])
        self.request_min_entropy = float(config["request_min_normalized_entropy"])
        self.require_old_identity_top1 = bool(config.get("require_old_identity_top1", True))

    def decide(self, diagnostics: dict) -> BaseTriggerDecision:
        old_probability = float(diagnostics["old_probability"])
        old_margin = float(diagnostics["old_margin"])
        competitor_margin = float(diagnostics["old_competitor_margin"])
        top_probability = float(diagnostics["top_probability"])
        entropy = float(diagnostics["normalized_entropy"])
        predicted_index = int(diagnostics["predicted_index"])
        identity_predicted_index = int(diagnostics["identity_predicted_index"])
        abstain_probability = float(diagnostics["abstain_probability"])
        old_is_decision = predicted_index == 0
        old_is_top = identity_predicted_index == 0

        if old_is_decision and old_probability >= self.direct_probability and old_margin >= self.direct_margin:
            return BaseTriggerDecision(
                "direct_merge", False, True, old_probability, old_margin, competitor_margin,
                top_probability, entropy, predicted_index, identity_predicted_index, abstain_probability,
                ("base_confident_old",),
            )

        plausible = old_probability >= self.request_min_probability
        uncertain = (
            old_probability <= self.request_max_probability
            or competitor_margin <= self.request_max_margin
            or entropy >= self.request_min_entropy
        )
        allowed_top = old_is_top or not self.require_old_identity_top1
        if plausible and uncertain and allowed_top:
            reasons = ["base_plausible_old", "base_uncertain"]
            if old_probability <= self.request_max_probability:
                reasons.append("low_probability")
            if competitor_margin <= self.request_max_margin:
                reasons.append("low_competitor_margin")
            if entropy >= self.request_min_entropy:
                reasons.append("high_entropy")
            return BaseTriggerDecision(
                "request_hr", True, False, old_probability, old_margin, competitor_margin,
                top_probability, entropy, predicted_index, identity_predicted_index, abstain_probability,
                tuple(reasons),
            )

        reasons = ["base_abstain"]
        if not plausible:
            reasons.append("old_probability_too_low")
        if not allowed_top:
            reasons.append("old_not_top_identity")
        return BaseTriggerDecision(
            "abstain", False, False, old_probability, old_margin, competitor_margin,
            top_probability, entropy, predicted_index, identity_predicted_index, abstain_probability,
            tuple(reasons),
        )

    @staticmethod
    def diagnostics(decision: BaseTriggerDecision) -> dict:
        return asdict(decision)
