#!/usr/bin/env python3
"""Compare missed GT instances and ID switches across MOT prediction files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluate_mot import iou_matrix, load_mot
from extract_id_switch_cases import find_switches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named MOT result. Repeat once per model.",
    )
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--json", type=Path, required=True)
    return parser.parse_args()


def parse_predictions(values: list[str]) -> dict[str, Path]:
    predictions: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, received: {value}")
        name, raw_path = value.split("=", 1)
        if not name or name in predictions:
            raise ValueError(f"Prediction name must be non-empty and unique: {name!r}")
        predictions[name] = Path(raw_path).resolve()
    if len(predictions) < 2:
        raise ValueError("At least two --prediction arguments are required")
    return predictions


def matched_gt_instances(gt_frames, pred_frames, threshold: float) -> set[tuple[int, int]]:
    matched: set[tuple[int, int]] = set()
    for frame_id, gt_rows in gt_frames.items():
        pred_rows = pred_frames.get(frame_id, [])
        if not gt_rows or not pred_rows:
            continue
        gt_boxes = np.asarray([row[1] for row in gt_rows], dtype=float).reshape(-1, 4)
        pred_boxes = np.asarray([row[1] for row in pred_rows], dtype=float).reshape(-1, 4)
        overlaps = iou_matrix(gt_boxes, pred_boxes)
        gt_indices, pred_indices = linear_sum_assignment(-overlaps)
        for gt_index, pred_index in zip(gt_indices, pred_indices):
            if overlaps[gt_index, pred_index] >= threshold:
                matched.add((frame_id, gt_rows[int(gt_index)][0]))
    return matched


def serialize_instances(instances: set[tuple[int, int]], limit: int | None = None) -> list[dict]:
    ordered = sorted(instances)
    if limit is not None:
        ordered = ordered[:limit]
    return [{"frame": frame, "gt_id": gt_id} for frame, gt_id in ordered]


def main() -> None:
    args = parse_args()
    if not 0.0 < args.iou <= 1.0:
        raise ValueError("--iou must be in (0, 1]")
    prediction_paths = parse_predictions(args.prediction)
    gt_frames = load_mot(args.gt.resolve(), ground_truth=True)
    all_gt = {
        (frame_id, gt_id)
        for frame_id, rows in gt_frames.items()
        for gt_id, _, _ in rows
    }

    misses: dict[str, set[tuple[int, int]]] = {}
    switches: dict[str, list[dict]] = {}
    for name, path in prediction_paths.items():
        pred_frames = load_mot(path, ground_truth=False)
        misses[name] = all_gt - matched_gt_instances(gt_frames, pred_frames, args.iou)
        switches[name] = find_switches(gt_frames, pred_frames, args.iou)

    names = list(prediction_paths)
    common_misses = set.intersection(*(misses[name] for name in names))
    union_misses = set.union(*(misses[name] for name in names))
    per_model = {}
    for name in names:
        other_union = set.union(*(misses[other] for other in names if other != name))
        gt_counts = Counter(gt_id for _, gt_id in misses[name])
        per_model[name] = {
            "missed_gt_instances": len(misses[name]),
            "unique_misses": len(misses[name] - other_union),
            "shared_by_all": len(misses[name] & common_misses),
            "shared_by_all_fraction": len(common_misses) / len(misses[name]) if misses[name] else 0.0,
            "id_switches": len(switches[name]),
            "switch_events": switches[name],
            "most_missed_gt_ids": [
                {"gt_id": gt_id, "misses": count}
                for gt_id, count in gt_counts.most_common(10)
            ],
        }

    pairwise = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            pairwise[f"{left}|{right}"] = len(misses[left] & misses[right])

    exact_switch_events = {
        name: {(event["gt_id"], event["switch_frame"]) for event in events}
        for name, events in switches.items()
    }
    common_switches = set.intersection(*(exact_switch_events[name] for name in names))
    switch_pairwise = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = exact_switch_events[left] & exact_switch_events[right]
            switch_pairwise[f"{left}|{right}"] = {
                "count": len(shared),
                "events": serialize_instances({(frame, gt_id) for gt_id, frame in shared}),
            }

    output = {
        "iou_threshold": args.iou,
        "gt_instances": len(all_gt),
        "models": per_model,
        "miss_overlap": {
            "union": len(union_misses),
            "common_to_all": len(common_misses),
            "common_to_all_fraction_of_gt": len(common_misses) / len(all_gt) if all_gt else 0.0,
            "pairwise": pairwise,
            "common_examples": serialize_instances(common_misses, limit=30),
        },
        "exact_id_switch_overlap": {
            "common_to_all": serialize_instances(
                {(frame, gt_id) for gt_id, frame in common_switches}
            ),
            "pairwise": switch_pairwise,
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"GT instances: {len(all_gt)}")
    for name in names:
        values = per_model[name]
        print(
            f"{name}: misses={values['missed_gt_instances']}, "
            f"shared_all={values['shared_by_all']} "
            f"({values['shared_by_all_fraction']:.1%}), "
            f"unique={values['unique_misses']}, IDSW={values['id_switches']}"
        )
    print(f"Common misses across all models: {len(common_misses)}")
    print(f"Saved JSON: {args.json.resolve()}")


if __name__ == "__main__":
    main()
