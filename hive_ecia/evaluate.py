"""Replay ECIA decisions and compare IDSW, IDF1, HOTA, and wrong corrections.

GT is loaded only in this post-hoc evaluator.  The input decision JSONL must be
the GT-free artifact emitted by ``hive_ecia.inference``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from hive_hr_reid import run_experiment as baseline
from hive_hr_reid.replay import annotate_operation_outcomes, replay_identity_merges


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument(
        "--evaluation-events",
        type=Path,
        required=True,
        help="HIVE event JSONL containing GT labels; read only by this evaluator.",
    )
    parser.add_argument("--dataset-root", type=Path, default=baseline.DEFAULT_DATASET)
    parser.add_argument("--lr-pred-dir", type=Path, default=baseline.DEFAULT_LR_PREDICTIONS)
    parser.add_argument(
        "--lr-pred-pattern", default="{sequence}_yolo11l_botsort_reid_pre512x120.txt"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-segment-gap", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    decisions = load_jsonl(args.decisions)
    if not decisions:
        raise ValueError("Decision file is empty")
    if any("gt_id" in row for row in decisions):
        raise ValueError("Inference decision artifact unexpectedly contains GT identity")
    if any(int(row.get("hr_crop_count", 0)) > 1 for row in decisions):
        raise ValueError("Decision artifact violates the one-HR-crop/event budget")

    # Evaluation-only join: GT IDs are attached after all ECIA decisions have
    # already been frozen on disk.
    evaluation_events = {
        row["event_id"]: row for row in load_jsonl(args.evaluation_events)
    }
    missing = {row["event_id"] for row in decisions} - set(evaluation_events)
    if missing:
        raise KeyError(f"Missing evaluation metadata for {len(missing)} events")
    events = [
        {
            **row,
            "gt_id": int(evaluation_events[row["event_id"]]["gt_id"]),
        }
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

    decisions_by_sequence: dict[str, list[dict]] = defaultdict(list)
    events_by_sequence: dict[str, list[dict]] = defaultdict(list)
    for decision, event in zip(decisions, events):
        decisions_by_sequence[decision["sequence"]].append(decision)
        events_by_sequence[event["sequence"]].append(event)

    replayed = {}
    operations_by_sequence = {}
    all_operations = []
    for sequence in sequences:
        sequence_frames, operations = replay_identity_merges(
            predictions[sequence],
            decisions_by_sequence[sequence],
            args.max_segment_gap,
        )
        replayed[sequence] = sequence_frames
        operations_by_sequence[sequence] = operations
        all_operations.extend(operations)

    metric_args = argparse.Namespace(iou=args.iou)
    baseline_metrics = baseline.basic_metrics(
        metric_args, sequences, predictions, ground_truth, include_hota=True
    )
    ecia_metrics = baseline.basic_metrics(
        metric_args, sequences, replayed, ground_truth, include_hota=True
    )

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

    correction = {
        "events": len(decisions),
        "merge_requests": sum(row.get("merge", False) for row in decisions),
        "applied_merges": sum(row["applied"] for row in all_operations),
        "actually_fixed": sum(row["actually_fixed"] for row in outcomes),
        "wrong_corrections": sum(row["wrong_correction"] for row in outcomes),
        "collision_or_missing_rejections": sum(
            row["requested_merge"] and not row["applied"] for row in all_operations
        ),
    }
    payload = {
        "protocol": {
            "decisions_frozen_before_gt_join": True,
            "evaluation_iou": args.iou,
            "sequences": sequences,
            "hr_requests": sum(
                row.get("selected_candidate_id") is not None for row in decisions
            ),
            "encoded_hr_crops": sum(int(row.get("hr_crop_count", 0)) for row in decisions),
            "max_hr_crops_per_event": 1,
        },
        "baseline": baseline_metrics,
        "ecia": ecia_metrics,
        "correction": correction,
        "outcomes": outcomes,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "outcomes.jsonl").open("w", encoding="utf-8") as handle:
        for row in outcomes:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    before = baseline_metrics["aggregate"]
    after = ecia_metrics["aggregate"]
    report = [
        "# HIVE + Lightweight ECIA evaluation",
        "",
        "| Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | MOTA ↑ |",
        "|---|---:|---:|---:|---:|",
        f"| LR-only | {before['id_switches']} | {before['idf1']:.4f} | {before['hota']:.4f} | {before['mota']:.4f} |",
        f"| HIVE + ECIA | {after['id_switches']} | {after['idf1']:.4f} | {after['hota']:.4f} | {after['mota']:.4f} |",
        "",
        "## Correction outcomes",
        "",
        f"- Merge requests: {correction['merge_requests']}",
        f"- Applied merges: {correction['applied_merges']}",
        f"- Actually fixed: {correction['actually_fixed']}",
        f"- Wrong corrections: {correction['wrong_corrections']}",
        f"- Collision/missing rejections: {correction['collision_or_missing_rejections']}",
        "",
        "GT was joined only after the decision JSONL had been frozen.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Report: {output_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
