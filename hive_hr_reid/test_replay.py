"""Small deterministic regression tests; no model weights are loaded."""

from __future__ import annotations

import numpy as np

from hive_hr_reid.hota import combine_hota_raw, evaluate_hota_raw
from hive_hr_reid.replay import replay_identity_merges
from hive_hr_reid.run_experiment import decide_merge


def row(track_id: int, x: float = 0.0) -> tuple:
    return track_id, np.asarray([x, 0.0, x + 10.0, 20.0]), 0.9


def test_replay_fixes_non_overlapping_tracklets() -> None:
    frames = {1: [row(1)], 2: [row(1)], 3: [row(2)], 4: [row(2)]}
    events = [
        {
            "event_id": "e1",
            "target_frame": 3,
            "lr_last_match_frame": 2,
            "lr_old_pred_id": 1,
            "lr_new_pred_id": 2,
            "merge": True,
        }
    ]
    replayed, operations = replay_identity_merges(frames, events)
    assert operations[0]["applied"]
    assert [replayed[frame][0][0] for frame in sorted(replayed)] == [1, 1, 1, 1]


def test_replay_rejects_same_frame_collision() -> None:
    frames = {1: [row(1)], 2: [row(1)], 3: [row(1, 20), row(2)], 4: [row(2)]}
    events = [
        {
            "event_id": "e1",
            "target_frame": 3,
            "lr_last_match_frame": 2,
            "lr_old_pred_id": 1,
            "lr_new_pred_id": 2,
            "merge": True,
        }
    ]
    _, operations = replay_identity_merges(frames, events)
    assert not operations[0]["applied"]
    assert operations[0]["reason"] == "same_frame_identity_collision"


def test_perfect_hota_is_one() -> None:
    gt = {1: [row(7)], 2: [row(7)]}
    pred = {1: [row(3)], 2: [row(3)]}
    metrics = combine_hota_raw([evaluate_hota_raw(gt, pred)])
    assert abs(metrics["hota"] - 1.0) < 1e-9
    assert abs(metrics["deta"] - 1.0) < 1e-9


def test_policy_ignores_evaluation_fields() -> None:
    scores = {
        "embedding_available": True,
        "evidence_bridge": 0.8,
        "similarity_old_target": 0.8,
        "evidence_margin": 0.2,
        "target_margin": 0.2,
        "evaluation_selected_hr_correct": False,
        "gt_id": 999,
    }
    thresholds = {
        "bridge_min": 0.65,
        "old_target_min": 0.7,
        "evidence_margin_min": 0.1,
        "target_margin_min": 0.1,
    }
    assert decide_merge(scores, thresholds) == (True, "reid_merge")


if __name__ == "__main__":
    test_replay_fixes_non_overlapping_tracklets()
    test_replay_rejects_same_frame_collision()
    test_perfect_hota_is_one()
    test_policy_ignores_evaluation_fields()
    print("All HIVE HR-ReID replay tests passed")
