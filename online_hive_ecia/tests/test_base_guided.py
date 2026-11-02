from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from hive_ecia.recovery_aware import RecoveryAwareECIA
from online_hive_ecia.base_trigger import BaseGuidedCandidateProposer, BaseGuidedHRPolicy
from online_hive_ecia.types import LostTrack, Observation, TrackToken


def candidate_config() -> dict:
    return {
        "candidate": {
            "max_lost_age": 30,
            "max_new_age": 3,
            "min_motion_iou": 0.02,
            "min_reid_similarity": 0.30,
            "strong_reid_similarity": 0.75,
            "min_association_score": 0.0,
            "max_ambiguity_margin": 0.15,
            "motion_weight": 0.4,
            "reid_weight": 0.6,
            "require_native_reid_or_motion": True,
            "include_active_conflicts": True,
            "active_min_age": 4,
            "active_min_motion_iou": 0.05,
            "active_min_reid_similarity": 0.65,
        }
    }


def observation(frame: int, raw_id: int, box=(0, 0, 10, 20)) -> Observation:
    token = TrackToken(raw_id)
    return Observation(
        frame_id=frame,
        token=token,
        canonical_id=raw_id,
        bbox_lr=np.asarray(box, dtype=np.float32),
        bbox_hr=np.asarray(box, dtype=np.float32) * 4,
        confidence=0.9,
        lr_crop=np.ones((20, 10, 3), dtype=np.uint8),
        source_path=Path("/tmp/not_read.jpg"),
        native_feature=np.asarray([1.0, 0.0], dtype=np.float32),
    )


class BaseSplitTest(unittest.TestCase):
    def test_split_forward_is_numerically_identical(self) -> None:
        torch.manual_seed(7)
        model = RecoveryAwareECIA(
            embedding_dim=16,
            projection_dim=8,
            relation_hidden_dim=6,
            event_hidden_dim=5,
            parameter_limit=50_000,
        ).eval()
        old = torch.randn(2, 16)
        target = torch.randn(2, 16)
        hr = torch.randn(2, 16)
        competitors = torch.randn(2, 3, 16)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        available = torch.tensor([True, False])
        with torch.inference_mode():
            legacy = model(old, target, hr, competitors, mask, available)
            base = model.forward_base(old, target, competitors, mask)
            split = model.forward_from_base(base, hr, available)
        for field in (
            "base_scores", "delta_scores", "final_scores", "abstain_logit",
            "base_decision_logits", "decision_logits", "gate", "gate_features",
        ):
            self.assertTrue(torch.equal(getattr(legacy, field), getattr(split, field)), field)


class BasePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = BaseGuidedHRPolicy(
            {
                "direct_merge_min_old_probability": 0.80,
                "direct_merge_min_old_margin": 0.40,
                "request_min_old_probability": 0.20,
                "request_max_old_probability": 0.80,
                "request_max_competitor_margin": 0.40,
                "request_min_normalized_entropy": 0.35,
                "require_old_identity_top1": True,
            }
        )

    @staticmethod
    def row(**overrides) -> dict:
        value = {
            "old_probability": 0.5,
            "old_margin": 0.1,
            "old_competitor_margin": 0.1,
            "top_probability": 0.5,
            "normalized_entropy": 0.5,
            "predicted_index": 0,
            "identity_predicted_index": 0,
            "abstain_probability": 0.2,
        }
        value.update(overrides)
        return value

    def test_confident_old_merges_without_hr(self) -> None:
        result = self.policy.decide(self.row(old_probability=0.9, old_margin=0.6))
        self.assertTrue(result.direct_merge)
        self.assertFalse(result.request_hr)

    def test_uncertain_plausible_old_requests_hr(self) -> None:
        result = self.policy.decide(self.row())
        self.assertEqual(result.action, "request_hr")

    def test_competitor_winner_abstains(self) -> None:
        result = self.policy.decide(
            self.row(predicted_index=1, identity_predicted_index=1)
        )
        self.assertEqual(result.action, "abstain")


class CandidateProposerTest(unittest.TestCase):
    def test_lost_to_new_and_active_conflict_are_generated(self) -> None:
        proposer = BaseGuidedCandidateProposer(candidate_config())
        old_observation = observation(8, 1)
        lost = LostTrack(
            token=old_observation.token,
            canonical_id=1,
            lost_frame=9,
            last_observation=old_observation,
            predicted_bbox_lr=old_observation.bbox_lr,
            native_feature=old_observation.native_feature,
        )
        new = observation(10, 2)
        proposal = proposer.propose_candidates(
            frame_id=10,
            new_observation=new,
            new_age=1,
            lost_tracks=[lost],
            active_observations=[new],
            observation_counts={},
        )
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.pair_kind, "lost_to_new")

        active_old = observation(10, 3)
        proposal = proposer.propose_candidates(
            frame_id=10,
            new_observation=new,
            new_age=1,
            lost_tracks=[],
            active_observations=[active_old, new],
            observation_counts={active_old.token: 10},
        )
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.pair_kind, "active_to_active")


if __name__ == "__main__":
    unittest.main()
