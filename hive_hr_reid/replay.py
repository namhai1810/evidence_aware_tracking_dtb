"""Tracker-result replay with conservative, frame-local identity stitching."""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluate_mot import iou_matrix


def clone_frames(frames: dict[int, list[tuple]]) -> dict[int, list[tuple]]:
    return {
        int(frame): [(int(row[0]), np.asarray(row[1], dtype=float).copy(), float(row[2])) for row in rows]
        for frame, rows in frames.items()
    }


def _find_row(
    frames: dict[int, list[tuple]], frame_id: int, raw_id: int, *, at_or_before: bool = False
) -> tuple[int, int] | None:
    candidates = [frame_id]
    if at_or_before:
        candidates = sorted((frame for frame in frames if frame <= frame_id), reverse=True)
    for frame in candidates:
        for index, row in enumerate(frames.get(frame, [])):
            if int(row[0]) == raw_id:
                return int(frame), index
    return None


def _continuous_segment(
    raw_frames: dict[int, list[tuple]], raw_id: int, start_frame: int, max_gap: int
) -> list[tuple[int, int]]:
    occurrences = []
    for frame in sorted(frame for frame in raw_frames if frame >= start_frame):
        for index, row in enumerate(raw_frames[frame]):
            if int(row[0]) == raw_id:
                occurrences.append((int(frame), index))
    if not occurrences:
        return []
    start_slot = next((i for i, (frame, _) in enumerate(occurrences) if frame >= start_frame), None)
    if start_slot is None:
        return []
    selected = [occurrences[start_slot]]
    for item in occurrences[start_slot + 1 :]:
        if item[0] - selected[-1][0] > max_gap:
            break
        selected.append(item)
    return selected


def replay_identity_merges(
    raw_frames: dict[int, list[tuple]],
    event_decisions: list[dict],
    max_segment_gap: int = 30,
) -> tuple[dict[int, list[tuple]], list[dict]]:
    """Apply accepted old/new joins without ever creating duplicate IDs in a frame.

    A join relabels the continuous future segment of ``lr_new_pred_id`` to the
    output identity carried by ``lr_old_pred_id`` at its last matched frame.
    Spatial boxes and confidences are never changed. A merge is rejected when it
    would create two detections with the same output ID in any frame.
    """
    output = clone_frames(raw_frames)
    raw = clone_frames(raw_frames)
    operations: list[dict] = []
    for event in sorted(event_decisions, key=lambda row: (int(row["target_frame"]), row["event_id"])):
        operation = {
            "event_id": event["event_id"],
            "target_frame": int(event["target_frame"]),
            "requested_merge": bool(event.get("merge", False)),
            "applied": False,
            "reason": "policy_keep",
            "affected_keys": [],
            "output_id": None,
        }
        if not operation["requested_merge"]:
            operations.append(operation)
            continue
        old_key = _find_row(
            raw,
            int(event["lr_last_match_frame"]),
            int(event["lr_old_pred_id"]),
            at_or_before=True,
        )
        segment = _continuous_segment(
            raw, int(event["lr_new_pred_id"]), int(event["target_frame"]), max_segment_gap
        )
        if old_key is None:
            operation["reason"] = "old_track_missing"
            operations.append(operation)
            continue
        if not segment or segment[0][0] != int(event["target_frame"]):
            operation["reason"] = "new_track_missing_at_target"
            operations.append(operation)
            continue
        old_output_id = int(output[old_key[0]][old_key[1]][0])
        segment_set = set(segment)
        collision = False
        for frame, _ in segment:
            if any(
                int(row[0]) == old_output_id and (frame, index) not in segment_set
                for index, row in enumerate(output.get(frame, []))
            ):
                collision = True
                break
        if collision:
            operation["reason"] = "same_frame_identity_collision"
            operations.append(operation)
            continue
        for frame, index in segment:
            _, box, confidence = output[frame][index]
            output[frame][index] = (old_output_id, box, confidence)
        operation.update(
            {
                "applied": True,
                "reason": "merged",
                "affected_keys": [[frame, index] for frame, index in segment],
                "output_id": old_output_id,
            }
        )
        operations.append(operation)
    return output, operations


def matched_identity_by_prediction(
    gt_frames: dict[int, list[tuple]], pred_frames: dict[int, list[tuple]], iou_threshold: float
) -> dict[tuple[int, int], int]:
    """Map (frame, prediction-row-index) to GT identity for post-hoc evaluation."""
    mapping: dict[tuple[int, int], int] = {}
    for frame in sorted(set(gt_frames) | set(pred_frames)):
        gt_rows = gt_frames.get(frame, [])
        pred_rows = pred_frames.get(frame, [])
        if not gt_rows or not pred_rows:
            continue
        overlaps = iou_matrix(
            np.asarray([row[1] for row in gt_rows], dtype=float).reshape(-1, 4),
            np.asarray([row[1] for row in pred_rows], dtype=float).reshape(-1, 4),
        )
        gt_slots, pred_slots = linear_sum_assignment(-overlaps)
        for gt_slot, pred_slot in zip(gt_slots, pred_slots):
            if overlaps[gt_slot, pred_slot] >= iou_threshold:
                mapping[(frame, int(pred_slot))] = int(gt_rows[int(gt_slot)][0])
    return mapping


def association_at(
    gt_frames: dict[int, list[tuple]],
    pred_frames: dict[int, list[tuple]],
    frame_id: int,
    gt_id: int,
    iou_threshold: float,
) -> int | None:
    gt_rows = gt_frames.get(frame_id, [])
    pred_rows = pred_frames.get(frame_id, [])
    if not gt_rows or not pred_rows:
        return None
    overlaps = iou_matrix(
        np.asarray([row[1] for row in gt_rows], dtype=float).reshape(-1, 4),
        np.asarray([row[1] for row in pred_rows], dtype=float).reshape(-1, 4),
    )
    gt_slots, pred_slots = linear_sum_assignment(-overlaps)
    for gt_slot, pred_slot in zip(gt_slots, pred_slots):
        if int(gt_rows[int(gt_slot)][0]) == gt_id and overlaps[gt_slot, pred_slot] >= iou_threshold:
            return int(pred_rows[int(pred_slot)][0])
    return None


def annotate_operation_outcomes(
    events: list[dict],
    operations: list[dict],
    gt_frames: dict[int, list[tuple]],
    before_frames: dict[int, list[tuple]],
    after_frames: dict[int, list[tuple]],
    iou_threshold: float,
) -> list[dict]:
    """Measure true event repair and affected-segment purity using GT post hoc."""
    event_map = {event["event_id"]: event for event in events}
    matched_after = matched_identity_by_prediction(gt_frames, after_frames, iou_threshold)
    outcomes = []
    for operation in operations:
        event = event_map[operation["event_id"]]
        gt_id = int(event["gt_id"])
        last_frame = int(event["lr_last_match_frame"])
        target_frame = int(event["target_frame"])
        before_old = association_at(gt_frames, before_frames, last_frame, gt_id, iou_threshold)
        before_new = association_at(gt_frames, before_frames, target_frame, gt_id, iou_threshold)
        after_old = association_at(gt_frames, after_frames, last_frame, gt_id, iou_threshold)
        after_new = association_at(gt_frames, after_frames, target_frame, gt_id, iou_threshold)
        was_switch = before_old is not None and before_new is not None and before_old != before_new
        actually_fixed = bool(
            operation["applied"]
            and was_switch
            and after_old is not None
            and after_old == after_new
        )
        affected_counts = Counter(
            matched_after[key]
            for key in (tuple(value) for value in operation.get("affected_keys", []))
            if key in matched_after
        )
        dominant_gt = affected_counts.most_common(1)[0][0] if affected_counts else None
        wrong_correction = bool(
            operation["applied"] and (not actually_fixed or dominant_gt not in {None, gt_id})
        )
        outcomes.append(
            {
                **operation,
                "gt_id": gt_id,
                "was_switch": was_switch,
                "before_old_output_id": before_old,
                "before_target_output_id": before_new,
                "after_old_output_id": after_old,
                "after_target_output_id": after_new,
                "actually_fixed": actually_fixed,
                "affected_segment_gt_counts": {str(key): value for key, value in affected_counts.items()},
                "affected_segment_dominant_gt": dominant_gt,
                "wrong_correction": wrong_correction,
            }
        )
    return outcomes

