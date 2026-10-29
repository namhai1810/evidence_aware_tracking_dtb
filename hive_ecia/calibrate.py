"""Select ECIA merge thresholds on a calibration split disjoint from model-val."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from hive_hr_reid import run_experiment as baseline
from hive_hr_reid.replay import annotate_operation_outcomes, replay_identity_merges


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--validation-decisions", type=Path, required=True)
    parser.add_argument("--validation-events", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=baseline.DEFAULT_DATASET)
    parser.add_argument("--lr-pred-dir", type=Path, default=baseline.DEFAULT_LR_PREDICTIONS)
    parser.add_argument(
        "--lr-pred-pattern", default="{sequence}_yolo11l_botsort_reid_pre512x120.txt"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-segment-gap", type=int, default=30)
    return parser.parse_args()


def apply_thresholds(rows: list[dict], probability: float, margin: float) -> list[dict]:
    frozen = []
    for row in rows:
        predicted_old = int(row.get("predicted_class", -1)) == 0
        merge = (
            predicted_old
            and float(row.get("old_probability", 0.0)) >= probability
            and float(row.get("old_margin", -1.0)) >= margin
        )
        frozen.append(
            {
                **row,
                "merge": merge,
                "reason": "ecia_merge" if merge else "validation_threshold_reject",
            }
        )
    return frozen


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    calibration = config["calibration"]
    decisions = load_jsonl(args.validation_decisions)
    if any("gt_id" in row for row in decisions):
        raise ValueError("Validation inference artifact must be GT-free")
    event_lookup = {row["event_id"]: row for row in load_jsonl(args.validation_events)}
    evaluation_events = [
        {**row, "gt_id": int(event_lookup[row["event_id"]]["gt_id"])}
        for row in decisions
    ]
    sequences = sorted({row["sequence"] for row in decisions})
    predictions = {
        sequence: baseline.load_mot(
            args.lr_pred_dir / args.lr_pred_pattern.format(sequence=sequence),
            ground_truth=False,
        )
        for sequence in sequences
    }
    ground_truth = {
        sequence: baseline.load_mot(
            args.dataset_root / sequence / "gt" / "gt.txt", ground_truth=True
        )
        for sequence in sequences
    }
    metric_args = argparse.Namespace(iou=args.iou)
    baseline_metrics = baseline.basic_metrics(
        metric_args, sequences, predictions, ground_truth, include_hota=False
    )["aggregate"]
    events_by_sequence: dict[str, list[dict]] = defaultdict(list)
    for event in evaluation_events:
        events_by_sequence[event["sequence"]].append(event)

    rows = []
    for probability in calibration["old_probability_grid"]:
        for margin in calibration["old_margin_grid"]:
            thresholded = apply_thresholds(decisions, float(probability), float(margin))
            by_sequence: dict[str, list[dict]] = defaultdict(list)
            for decision in thresholded:
                by_sequence[decision["sequence"]].append(decision)
            replayed = {}
            operations_by_sequence = {}
            for sequence in sequences:
                replayed[sequence], operations_by_sequence[sequence] = replay_identity_merges(
                    predictions[sequence],
                    by_sequence[sequence],
                    args.max_segment_gap,
                )
            metrics = baseline.basic_metrics(
                metric_args, sequences, replayed, ground_truth, include_hota=False
            )["aggregate"]
            outcomes = []
            for sequence in sequences:
                outcomes.extend(
                    annotate_operation_outcomes(
                        events_by_sequence[sequence],
                        operations_by_sequence[sequence],
                        ground_truth[sequence],
                        predictions[sequence],
                        replayed[sequence],
                        args.iou,
                    )
                )
            applied = sum(row["applied"] for row in outcomes)
            wrong = sum(row["wrong_correction"] for row in outcomes)
            wrong_rate = wrong / max(1, applied)
            idsw_reduction = (
                baseline_metrics["id_switches"] - metrics["id_switches"]
            ) / max(1, baseline_metrics["id_switches"])
            objective = (
                metrics["idf1"]
                + float(calibration["idsw_reduction_weight"]) * idsw_reduction
                - float(calibration["wrong_correction_penalty"]) * wrong_rate
            )
            rows.append(
                {
                    "thresholds": {
                        "min_old_probability": float(probability),
                        "min_old_margin": float(margin),
                    },
                    "objective": objective,
                    "idf1": metrics["idf1"],
                    "idsw": metrics["id_switches"],
                    "applied_merges": applied,
                    "actually_fixed": sum(row["actually_fixed"] for row in outcomes),
                    "wrong_corrections": wrong,
                    "wrong_rate": wrong_rate,
                }
            )

    rows.sort(
        key=lambda row: (
            row["objective"],
            -row["wrong_corrections"],
            row["idf1"],
            -row["idsw"],
            row["thresholds"]["min_old_probability"],
            row["thresholds"]["min_old_margin"],
        ),
        reverse=True,
    )
    payload = {
        "selection_split": "calibration_only",
        "calibration_sequences": sequences,
        "selected_thresholds": rows[0]["thresholds"],
        "selected_validation_result": rows[0],
        "baseline_validation": baseline_metrics,
        "grid": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["selected_thresholds"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
