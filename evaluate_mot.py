#!/usr/bin/env python3
"""Compare MOT-format tracking results with QuadTrack ground truth."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_GT = CODE_DIR.parent / "QuadTrack" / "train" / "0015" / "gt" / "gt.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more MOT result files against QuadTrack GT at a fixed IoU."
    )
    parser.add_argument("results", nargs="+", type=Path, help="MOT-format prediction TXT files.")
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--json", type=Path, default=None, help="Optional path for machine-readable metrics.")
    return parser.parse_args()


def load_mot(
    path: Path, *, ground_truth: bool
) -> dict[int, list[tuple[int, np.ndarray, float]]]:
    frames: dict[int, list[tuple[int, np.ndarray, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) < 6:
                raise ValueError(f"Malformed MOT row {line_number} in {path}")
            frame_id = int(float(row[0]))
            track_id = int(float(row[1]))
            x, y, width, height = map(float, row[2:6])
            confidence = float(row[6]) if len(row) > 6 else 1.0
            class_id = int(float(row[7])) if len(row) > 7 else 1
            if width <= 0 or height <= 0 or confidence <= 0:
                continue
            if ground_truth and class_id != 1:
                continue
            frames[frame_id].append(
                (track_id, np.array([x, y, x + width, y + height], dtype=float), confidence)
            )
    return frames


def iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.empty((len(gt_boxes), len(pred_boxes)), dtype=float)
    gt = gt_boxes[:, None, :]
    pred = pred_boxes[None, :, :]
    x1 = np.maximum(gt[..., 0], pred[..., 0])
    y1 = np.maximum(gt[..., 1], pred[..., 1])
    x2 = np.minimum(gt[..., 2], pred[..., 2])
    y2 = np.minimum(gt[..., 3], pred[..., 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    gt_area = np.maximum(0.0, gt[..., 2] - gt[..., 0]) * np.maximum(0.0, gt[..., 3] - gt[..., 1])
    pred_area = np.maximum(0.0, pred[..., 2] - pred[..., 0]) * np.maximum(0.0, pred[..., 3] - pred[..., 1])
    return intersection / np.maximum(gt_area + pred_area - intersection, 1e-9)


def evaluate(
    gt_frames: dict[int, list[tuple[int, np.ndarray, float]]],
    pred_frames: dict[int, list[tuple[int, np.ndarray, float]]],
    threshold: float,
) -> dict[str, float | int]:
    tp = fp = fn = id_switches = 0
    matched_iou_sum = 0.0
    gt_total = sum(len(rows) for rows in gt_frames.values())
    pred_total = sum(len(rows) for rows in pred_frames.values())
    last_prediction_for_gt: dict[int, int] = {}
    identity_counts: dict[tuple[int, int], int] = defaultdict(int)
    gt_ids: set[int] = set()
    pred_ids: set[int] = set()

    for frame_id in sorted(set(gt_frames) | set(pred_frames)):
        gt_rows = gt_frames.get(frame_id, [])
        pred_rows = pred_frames.get(frame_id, [])
        gt_ids.update(row[0] for row in gt_rows)
        pred_ids.update(row[0] for row in pred_rows)
        gt_boxes = np.asarray([row[1] for row in gt_rows], dtype=float).reshape(-1, 4)
        pred_boxes = np.asarray([row[1] for row in pred_rows], dtype=float).reshape(-1, 4)
        overlaps = iou_matrix(gt_boxes, pred_boxes)

        matches: list[tuple[int, int]] = []
        if len(gt_rows) and len(pred_rows):
            gt_indices, pred_indices = linear_sum_assignment(-overlaps)
            matches = [
                (int(gt_index), int(pred_index))
                for gt_index, pred_index in zip(gt_indices, pred_indices)
                if overlaps[gt_index, pred_index] >= threshold
            ]

        tp += len(matches)
        fn += len(gt_rows) - len(matches)
        fp += len(pred_rows) - len(matches)
        for gt_index, pred_index in matches:
            gt_id = gt_rows[gt_index][0]
            pred_id = pred_rows[pred_index][0]
            matched_iou_sum += float(overlaps[gt_index, pred_index])
            identity_counts[(gt_id, pred_id)] += 1
            previous_pred_id = last_prediction_for_gt.get(gt_id)
            if previous_pred_id is not None and previous_pred_id != pred_id:
                id_switches += 1
            last_prediction_for_gt[gt_id] = pred_id

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mota = 1.0 - (fn + fp + id_switches) / gt_total if gt_total else 0.0

    gt_id_list = sorted(gt_ids)
    pred_id_list = sorted(pred_ids)
    identity_matrix = np.zeros((len(gt_id_list), len(pred_id_list)), dtype=float)
    gt_index = {track_id: index for index, track_id in enumerate(gt_id_list)}
    pred_index = {track_id: index for index, track_id in enumerate(pred_id_list)}
    for (gt_id, pred_id), count in identity_counts.items():
        identity_matrix[gt_index[gt_id], pred_index[pred_id]] = count
    idtp = 0
    if identity_matrix.size:
        rows, columns = linear_sum_assignment(-identity_matrix)
        idtp = int(identity_matrix[rows, columns].sum())
    idfn = gt_total - idtp
    idfp = pred_total - idtp
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if 2 * idtp + idfp + idfn else 0.0

    return {
        "gt": gt_total,
        "pred": pred_total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": matched_iou_sum / tp if tp else 0.0,
        "id_switches": id_switches,
        "mota": mota,
        "idf1": idf1,
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "gt_tracks": len(gt_ids),
        "pred_tracks": len(pred_ids),
    }


def print_table(results: list[tuple[str, dict[str, float | int]]], threshold: float) -> None:
    headers = ("result", "P", "R", "F1", "mIoU", "MOTA", "IDF1", "IDSW", "TP", "FP", "FN")
    rows = []
    for name, metrics in results:
        rows.append(
            (
                name,
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1']:.3f}",
                f"{metrics['mean_iou']:.3f}",
                f"{metrics['mota']:.3f}",
                f"{metrics['idf1']:.3f}",
                str(metrics["id_switches"]),
                str(metrics["tp"]),
                str(metrics["fp"]),
                str(metrics["fn"]),
            )
        )
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
    print(f"Evaluation at IoU >= {threshold:.2f}")
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> None:
    args = parse_args()
    if not 0.0 < args.iou <= 1.0:
        raise ValueError("--iou must be in (0, 1]")
    gt_frames = load_mot(args.gt.resolve(), ground_truth=True)
    evaluations = []
    json_results = {}
    for result_path in args.results:
        resolved = result_path.resolve()
        metrics = evaluate(gt_frames, load_mot(resolved, ground_truth=False), args.iou)
        evaluations.append((resolved.stem, metrics))
        json_results[resolved.stem] = metrics
    print_table(evaluations, args.iou)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(json_results, indent=2) + "\n", encoding="utf-8")
        print(f"Saved JSON: {args.json.resolve()}")


if __name__ == "__main__":
    main()
