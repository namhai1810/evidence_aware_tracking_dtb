#!/usr/bin/env python3
"""Evaluate a matching MOT result for every QuadTrack sequence and log failed frames."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluate_mot import evaluate, iou_matrix, load_mot


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = CODE_DIR.parent / "QuadTrack" / "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument(
        "--pred-pattern",
        default="{sequence}_yolo11l_bytetrack_pre512x120.txt",
        help="Prediction filename pattern; it must contain {sequence}.",
    )
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--failures-json", type=Path, required=True)
    parser.add_argument("--failures-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--physical-input", default="512x120")
    parser.add_argument("--raw-pixel-fraction", default="6.25%")
    parser.add_argument("--detector", default="YOLO11l")
    parser.add_argument("--tracker", default="ByteTrack")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    return parser.parse_args()


def discover_sequences(dataset_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "img1").is_dir() and (path / "gt" / "gt.txt").is_file()
    )


def rounded_box(box: np.ndarray) -> list[float]:
    return [round(float(value), 2) for value in box]


def collect_failure_frames(
    sequence: str,
    gt_frames: dict,
    pred_frames: dict,
    threshold: float,
) -> list[dict]:
    """Return frame-level FN/FP/ID-switch evidence using evaluator-identical matching."""
    failures: list[dict] = []
    last_prediction_for_gt: dict[int, dict] = {}

    for frame_id in sorted(set(gt_frames) | set(pred_frames)):
        gt_rows = gt_frames.get(frame_id, [])
        pred_rows = pred_frames.get(frame_id, [])
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

        matched_gt = {gt_index for gt_index, _ in matches}
        matched_pred = {pred_index for _, pred_index in matches}
        false_negatives = [
            {"gt_id": int(gt_rows[index][0]), "bbox_xyxy": rounded_box(gt_rows[index][1])}
            for index in range(len(gt_rows))
            if index not in matched_gt
        ]
        false_positives = [
            {
                "pred_id": int(pred_rows[index][0]),
                "bbox_xyxy": rounded_box(pred_rows[index][1]),
                "confidence": round(float(pred_rows[index][2]), 5),
            }
            for index in range(len(pred_rows))
            if index not in matched_pred
        ]

        switches = []
        matched_context = []
        for gt_index, pred_index in matches:
            gt_id = int(gt_rows[gt_index][0])
            pred_id = int(pred_rows[pred_index][0])
            overlap = float(overlaps[gt_index, pred_index])
            previous = last_prediction_for_gt.get(gt_id)
            if previous is not None and previous["pred_id"] != pred_id:
                switches.append(
                    {
                        "gt_id": gt_id,
                        "old_pred_id": previous["pred_id"],
                        "new_pred_id": pred_id,
                        "previous_match_frame": previous["frame_id"],
                        "gap_frames": frame_id - previous["frame_id"] - 1,
                        "match_iou": round(overlap, 6),
                    }
                )
            last_prediction_for_gt[gt_id] = {
                "pred_id": pred_id,
                "frame_id": frame_id,
            }
            matched_context.append(
                {
                    "gt_id": gt_id,
                    "pred_id": pred_id,
                    "iou": round(overlap, 6),
                }
            )

        if false_negatives or false_positives or switches:
            failures.append(
                {
                    "sequence": sequence,
                    "frame_id": frame_id,
                    "image": f"{frame_id:06d}.jpg",
                    "gt_count": len(gt_rows),
                    "pred_count": len(pred_rows),
                    "tp": len(matches),
                    "fn": len(false_negatives),
                    "fp": len(false_positives),
                    "id_switches": len(switches),
                    "false_negatives": false_negatives,
                    "false_positives": false_positives,
                    "switches": switches,
                    "matches": matched_context,
                }
            )
    return failures


def aggregate_metrics(sequence_metrics: dict[str, dict]) -> dict:
    summed_keys = (
        "gt",
        "pred",
        "tp",
        "fp",
        "fn",
        "id_switches",
        "idtp",
        "idfp",
        "idfn",
        "gt_tracks",
        "pred_tracks",
    )
    aggregate = {key: sum(int(metrics[key]) for metrics in sequence_metrics.values()) for key in summed_keys}
    aggregate["sequences"] = len(sequence_metrics)
    tp, fp, fn = aggregate["tp"], aggregate["fp"], aggregate["fn"]
    aggregate["precision"] = tp / (tp + fp) if tp + fp else 0.0
    aggregate["recall"] = tp / (tp + fn) if tp + fn else 0.0
    precision, recall = aggregate["precision"], aggregate["recall"]
    aggregate["f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    aggregate["mean_iou"] = (
        sum(float(metrics["mean_iou"]) * int(metrics["tp"]) for metrics in sequence_metrics.values()) / tp
        if tp
        else 0.0
    )
    aggregate["mota"] = (
        1.0 - (fn + fp + aggregate["id_switches"]) / aggregate["gt"] if aggregate["gt"] else 0.0
    )
    denominator = 2 * aggregate["idtp"] + aggregate["idfp"] + aggregate["idfn"]
    aggregate["idf1"] = 2 * aggregate["idtp"] / denominator if denominator else 0.0
    return aggregate


def write_failures_csv(path: Path, failures: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "sequence",
        "frame_id",
        "image",
        "gt_count",
        "pred_count",
        "tp",
        "fp",
        "fn",
        "id_switches",
        "fn_gt_ids",
        "fp_pred_ids",
        "switch_gt_ids",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in failures:
            writer.writerow(
                {
                    **{key: frame[key] for key in fields[:9]},
                    "fn_gt_ids": ";".join(str(item["gt_id"]) for item in frame["false_negatives"]),
                    "fp_pred_ids": ";".join(str(item["pred_id"]) for item in frame["false_positives"]),
                    "switch_gt_ids": ";".join(str(item["gt_id"]) for item in frame["switches"]),
                }
            )


def metric_row(name: str, metrics: dict) -> str:
    return (
        f"| {name} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | "
        f"{metrics['f1']:.3f} | {metrics['mean_iou']:.3f} | {metrics['mota']:.3f} | "
        f"{metrics['idf1']:.3f} | {metrics['id_switches']} | {metrics['tp']} | "
        f"{metrics['fp']} | {metrics['fn']} |"
    )


def write_report(
    path: Path,
    args: argparse.Namespace,
    sequences: list[str],
    sequence_metrics: dict[str, dict],
    aggregate: dict,
    failures: list[dict],
) -> None:
    failure_sequences = {sequence: [] for sequence in sequences}
    for frame in failures:
        failure_sequences[frame["sequence"]].append(frame)
    frames_with_fn = sum(1 for frame in failures if frame["fn"])
    frames_with_fp = sum(1 for frame in failures if frame["fp"])
    frames_with_switch = sum(1 for frame in failures if frame["id_switches"])

    artifact_root = args.metrics_json.resolve().parent
    try:
        artifact_label = artifact_root.relative_to(path.resolve().parent).as_posix()
    except ValueError:
        artifact_label = artifact_root.as_posix()
    lines = [
        f"# QuadTrack all-sequence evaluation: {args.physical_input} + {args.tracker}",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Configuration",
        "",
        f"- Sequences: `{sequences[0]}`–`{sequences[-1]}` ({len(sequences)} total)",
        f"- Detector/tracker: `{args.detector}` + `{args.tracker}`",
        f"- Physical input: `{args.physical_input}` ({args.raw_pixel_fraction} of 2048x480 raw pixels)",
        f"- Inference size: `imgsz={args.imgsz}`",
        f"- Detector confidence / NMS IoU: `{args.conf:.2f}` / `{args.nms_iou:.2f}`",
        f"- Evaluation IoU: `>= {args.iou:.2f}`",
        "- Class: person only (COCO class 0; QuadTrack GT class 1)",
        "- Aggregate policy: micro-average; identities from different sequences are treated as disjoint.",
        "",
        "## Aggregate result",
        "",
        "| Physical input | Raw pixels vs. original | Tracker | Precision | Recall | F1 | mIoU | MOTA | IDF1 | IDSW | TP | FP | FN |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {args.physical_input} | {args.raw_pixel_fraction} | **{args.tracker}** | {aggregate['precision']:.3f} | {aggregate['recall']:.3f} | {aggregate['f1']:.3f} | {aggregate['mean_iou']:.3f} | {aggregate['mota']:.3f} | {aggregate['idf1']:.3f} | {aggregate['id_switches']} | {aggregate['tp']} | {aggregate['fp']} | {aggregate['fn']} |",
        "",
        "## Per-sequence results",
        "",
        "| Sequence | Precision | Recall | F1 | mIoU | MOTA | IDF1 | IDSW | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(metric_row(sequence, sequence_metrics[sequence]) for sequence in sequences)
    lines.extend(
        [
            "",
            "## Incorrect-frame log",
            "",
            "A frame is logged when evaluator matching at the configured IoU contains at least one false negative, false positive, or identity switch. A single frame can belong to multiple categories.",
            "",
            f"- Unique incorrect frames: **{len(failures)}**",
            f"- Frames with at least one FN: **{frames_with_fn}**",
            f"- Frames with at least one FP: **{frames_with_fp}**",
            f"- Frames with at least one ID switch: **{frames_with_switch}**",
            "",
            "| Sequence | Incorrect frames | With FN | With FP | With ID switch |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for sequence in sequences:
        rows = failure_sequences[sequence]
        lines.append(
            f"| {sequence} | {len(rows)} | {sum(1 for row in rows if row['fn'])} | "
            f"{sum(1 for row in rows if row['fp'])} | {sum(1 for row in rows if row['id_switches'])} |"
        )
    lines.extend(
        [
            "",
            "The JSON log retains bounding boxes, confidence, matched GT/pred IDs and IoUs, plus the previous matched frame and gap length for every ID switch. The CSV is a compact frame index for filtering.",
            "",
            "## Artifacts",
            "",
            f"- [Machine-readable metrics]({artifact_label}/{args.metrics_json.name})",
            f"- [Detailed incorrect-frame log]({artifact_label}/{args.failures_json.name})",
            f"- [Compact incorrect-frame CSV]({artifact_label}/{args.failures_csv.name})",
            f"- MOT result files and annotated videos: `{artifact_label}/predictions/`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if "{sequence}" not in args.pred_pattern:
        raise ValueError("--pred-pattern must contain {sequence}")
    if not 0.0 < args.iou <= 1.0:
        raise ValueError("--iou must be in (0, 1]")

    dataset_root = args.dataset_root.resolve()
    pred_dir = args.pred_dir.resolve()
    sequences = args.sequences or discover_sequences(dataset_root)
    if not sequences:
        raise RuntimeError(f"No QuadTrack sequences found under {dataset_root}")

    sequence_metrics: dict[str, dict] = {}
    failures: list[dict] = []
    for sequence in sequences:
        gt_path = dataset_root / sequence / "gt" / "gt.txt"
        pred_path = pred_dir / args.pred_pattern.format(sequence=sequence)
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)
        if not pred_path.is_file():
            raise FileNotFoundError(pred_path)
        gt_frames = load_mot(gt_path, ground_truth=True)
        pred_frames = load_mot(pred_path, ground_truth=False)
        sequence_metrics[sequence] = evaluate(gt_frames, pred_frames, args.iou)
        failures.extend(collect_failure_frames(sequence, gt_frames, pred_frames, args.iou))

    aggregate = aggregate_metrics(sequence_metrics)
    metrics_payload = {
        "configuration": {
            "dataset_root": str(dataset_root),
            "prediction_dir": str(pred_dir),
            "prediction_pattern": args.pred_pattern,
            "sequences": sequences,
            "physical_input": args.physical_input,
            "raw_pixel_fraction": args.raw_pixel_fraction,
            "detector": args.detector,
            "tracker": args.tracker,
            "imgsz": args.imgsz,
            "confidence": args.conf,
            "nms_iou": args.nms_iou,
            "evaluation_iou": args.iou,
        },
        "aggregate": aggregate,
        "sequences": sequence_metrics,
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
    args.failures_json.parent.mkdir(parents=True, exist_ok=True)
    args.failures_json.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    write_failures_csv(args.failures_csv, failures)
    write_report(args.report, args, sequences, sequence_metrics, aggregate, failures)

    print(f"Evaluated {len(sequences)} sequences at IoU >= {args.iou:.2f}")
    print(metric_row("ALL", aggregate))
    print(f"Logged {len(failures)} incorrect frames")
    print(f"Report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
