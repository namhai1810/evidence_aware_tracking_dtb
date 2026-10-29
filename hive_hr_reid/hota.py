"""Dependency-free HOTA evaluation for MOT-style frame dictionaries.

The implementation follows the TrackEval formulation: global alignment scores
guide per-frame Hungarian matching, then DetA and AssA are integrated over IoU
thresholds 0.05..0.95.  It intentionally exposes raw per-alpha counts so that
video-disjoint sequences can be micro-aggregated without mixing identities.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluate_mot import iou_matrix


ALPHAS = np.arange(0.05, 0.99, 0.05, dtype=float)


def _identity_counts(frames: dict[int, list[tuple]]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for rows in frames.values():
        for track_id, *_ in rows:
            counts[int(track_id)] += 1
    return dict(counts)


def _potential_matches(
    gt_frames: dict[int, list[tuple]],
    pred_frames: dict[int, list[tuple]],
    gt_index: dict[int, int],
    pred_index: dict[int, int],
) -> np.ndarray:
    """Accumulate soft potential matches used by HOTA global alignment."""
    counts = np.zeros((len(gt_index), len(pred_index)), dtype=float)
    for frame_id in sorted(set(gt_frames) | set(pred_frames)):
        gt_rows = gt_frames.get(frame_id, [])
        pred_rows = pred_frames.get(frame_id, [])
        if not gt_rows or not pred_rows:
            continue
        gt_boxes = np.asarray([row[1] for row in gt_rows], dtype=float).reshape(-1, 4)
        pred_boxes = np.asarray([row[1] for row in pred_rows], dtype=float).reshape(-1, 4)
        similarities = iou_matrix(gt_boxes, pred_boxes)
        # TrackEval normalizes each potential match by the total similarity mass
        # touching the GT and tracker detection in this frame.
        row_sum = similarities.sum(axis=1, keepdims=True)
        col_sum = similarities.sum(axis=0, keepdims=True)
        normalized = similarities / np.maximum(row_sum + col_sum - similarities, 1e-10)
        for gi, gt_row in enumerate(gt_rows):
            for pi, pred_row in enumerate(pred_rows):
                counts[gt_index[int(gt_row[0])], pred_index[int(pred_row[0])]] += normalized[gi, pi]
    return counts


def evaluate_hota_raw(
    gt_frames: dict[int, list[tuple]],
    pred_frames: dict[int, list[tuple]],
    alphas: Iterable[float] = ALPHAS,
) -> dict:
    """Return per-alpha HOTA components for one sequence."""
    alpha_values = np.asarray(list(alphas), dtype=float)
    gt_counts = _identity_counts(gt_frames)
    pred_counts = _identity_counts(pred_frames)
    gt_ids = sorted(gt_counts)
    pred_ids = sorted(pred_counts)
    gt_index = {track_id: index for index, track_id in enumerate(gt_ids)}
    pred_index = {track_id: index for index, track_id in enumerate(pred_ids)}
    potential = _potential_matches(gt_frames, pred_frames, gt_index, pred_index)
    gt_totals = np.asarray([gt_counts[value] for value in gt_ids], dtype=float)[:, None]
    pred_totals = np.asarray([pred_counts[value] for value in pred_ids], dtype=float)[None, :]
    global_alignment = potential / np.maximum(gt_totals + pred_totals - potential, 1e-10)

    result = {
        "alphas": alpha_values.tolist(),
        "tp": np.zeros(len(alpha_values), dtype=float),
        "fn": np.zeros(len(alpha_values), dtype=float),
        "fp": np.zeros(len(alpha_values), dtype=float),
        "assa_numerator": np.zeros(len(alpha_values), dtype=float),
    }
    total_gt = float(sum(gt_counts.values()))
    total_pred = float(sum(pred_counts.values()))

    for alpha_index, alpha in enumerate(alpha_values):
        matches_by_pair: dict[tuple[int, int], int] = defaultdict(int)
        matched_pairs: list[tuple[int, int]] = []
        for frame_id in sorted(set(gt_frames) | set(pred_frames)):
            gt_rows = gt_frames.get(frame_id, [])
            pred_rows = pred_frames.get(frame_id, [])
            if not gt_rows or not pred_rows:
                continue
            gt_boxes = np.asarray([row[1] for row in gt_rows], dtype=float).reshape(-1, 4)
            pred_boxes = np.asarray([row[1] for row in pred_rows], dtype=float).reshape(-1, 4)
            similarities = iou_matrix(gt_boxes, pred_boxes)
            alignment = np.asarray(
                [
                    [global_alignment[gt_index[int(gt[0])], pred_index[int(pred[0])]] for pred in pred_rows]
                    for gt in gt_rows
                ],
                dtype=float,
            )
            scores = alignment * similarities
            scores[similarities < alpha - np.finfo(float).eps] = 0.0
            gt_slots, pred_slots = linear_sum_assignment(-scores)
            for gt_slot, pred_slot in zip(gt_slots, pred_slots):
                if similarities[gt_slot, pred_slot] < alpha - np.finfo(float).eps:
                    continue
                gt_id = int(gt_rows[int(gt_slot)][0])
                pred_id = int(pred_rows[int(pred_slot)][0])
                matches_by_pair[(gt_id, pred_id)] += 1
                matched_pairs.append((gt_id, pred_id))

        tp = float(len(matched_pairs))
        result["tp"][alpha_index] = tp
        result["fn"][alpha_index] = total_gt - tp
        result["fp"][alpha_index] = total_pred - tp
        association_sum = 0.0
        for gt_id, pred_id in matched_pairs:
            tpa = float(matches_by_pair[(gt_id, pred_id)])
            denominator = gt_counts[gt_id] + pred_counts[pred_id] - tpa
            association_sum += tpa / denominator if denominator else 0.0
        result["assa_numerator"][alpha_index] = association_sum
    return result


def combine_hota_raw(sequence_results: Iterable[dict]) -> dict[str, float | list[float]]:
    """Micro-combine disjoint sequences and finalize HOTA/DetA/AssA."""
    rows = list(sequence_results)
    if not rows:
        return {"hota": 0.0, "deta": 0.0, "assa": 0.0, "loca": 0.0}
    tp = np.sum([row["tp"] for row in rows], axis=0)
    fn = np.sum([row["fn"] for row in rows], axis=0)
    fp = np.sum([row["fp"] for row in rows], axis=0)
    assa_num = np.sum([row["assa_numerator"] for row in rows], axis=0)
    deta = tp / np.maximum(tp + fn + fp, 1e-10)
    assa = assa_num / np.maximum(tp, 1e-10)
    hota = np.sqrt(np.maximum(0.0, deta * assa))
    return {
        "hota": float(np.mean(hota)),
        "deta": float(np.mean(deta)),
        "assa": float(np.mean(assa)),
        "hota_by_alpha": hota.tolist(),
        "deta_by_alpha": deta.tolist(),
        "assa_by_alpha": assa.tolist(),
        "alphas": rows[0]["alphas"],
    }

