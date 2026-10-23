#!/usr/bin/env python3
"""Rank high-resolution history frames that can provide evidence for low-resolution ID switches."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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
    parser.add_argument("--lowres-dir", type=Path, required=True)
    parser.add_argument("--highres-dir", type=Path, required=True)
    parser.add_argument(
        "--lowres-pattern",
        default="{sequence}_yolo11l_bytetrack_pre512x120.txt",
    )
    parser.add_argument(
        "--highres-pattern",
        default="{sequence}_yolo11l_bytetrack_imgsz960.txt",
    )
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-seconds-before", type=float, default=5.0)
    parser.add_argument("--max-seconds-before", type=float, default=10.0)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--min-spacing-frames",
        type=int,
        default=15,
        help="Prefer diverse selected frames separated by this many frames before filling remaining slots.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--tracker-label",
        default="ByteTrack",
        help="Tracker name used in the generated report.",
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=None,
        help="Optional retrieval summary JSON to include as a direct comparison.",
    )
    parser.add_argument("--baseline-label", default="Baseline")
    return parser.parse_args()


def discover_sequences(dataset_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "gt" / "gt.txt").is_file() and (path / "img1").is_dir()
    )


def rounded_box(box: np.ndarray) -> list[float]:
    return [round(float(value), 2) for value in box]


def build_match_index(gt_frames: dict, pred_frames: dict, threshold: float) -> tuple[dict, dict]:
    """Build evaluator-identical GT-to-prediction matches indexed by frame and GT identity."""
    by_frame: dict[int, dict[int, dict]] = {}
    history: dict[int, list[dict]] = defaultdict(list)
    for frame_id in sorted(set(gt_frames) | set(pred_frames)):
        gt_rows = gt_frames.get(frame_id, [])
        pred_rows = pred_frames.get(frame_id, [])
        gt_boxes = np.asarray([row[1] for row in gt_rows], dtype=float).reshape(-1, 4)
        pred_boxes = np.asarray([row[1] for row in pred_rows], dtype=float).reshape(-1, 4)
        overlaps = iou_matrix(gt_boxes, pred_boxes)
        frame_matches: dict[int, dict] = {}
        if len(gt_rows) and len(pred_rows):
            gt_indices, pred_indices = linear_sum_assignment(-overlaps)
            for gt_index, pred_index in zip(gt_indices, pred_indices):
                overlap = float(overlaps[gt_index, pred_index])
                if overlap < threshold:
                    continue
                gt_id, gt_box, _ = gt_rows[int(gt_index)]
                pred_id, pred_box, confidence = pred_rows[int(pred_index)]
                record = {
                    "frame_id": frame_id,
                    "gt_id": int(gt_id),
                    "pred_id": int(pred_id),
                    "iou": overlap,
                    "confidence": float(confidence),
                    "gt_bbox_xyxy": rounded_box(gt_box),
                    "pred_bbox_xyxy": rounded_box(pred_box),
                }
                frame_matches[int(gt_id)] = record
                history[int(gt_id)].append(record)
        by_frame[frame_id] = frame_matches
    return by_frame, history


def find_switch_events(match_by_frame: dict[int, dict[int, dict]]) -> list[dict]:
    """Use the same last-matched-ID switch rule as evaluate_mot.py."""
    last_for_gt: dict[int, dict] = {}
    events = []
    for frame_id in sorted(match_by_frame):
        for gt_id, match in sorted(match_by_frame[frame_id].items()):
            previous = last_for_gt.get(gt_id)
            if previous is not None and previous["pred_id"] != match["pred_id"]:
                events.append(
                    {
                        "target_frame": frame_id,
                        "gt_id": gt_id,
                        "lr_old_pred_id": previous["pred_id"],
                        "lr_new_pred_id": match["pred_id"],
                        "lr_last_match_frame": previous["frame_id"],
                        "lr_gap_frames": frame_id - previous["frame_id"] - 1,
                        "lr_target_iou": match["iou"],
                        "lr_target_confidence": match["confidence"],
                        "lr_target_gt_bbox_xyxy": match["gt_bbox_xyxy"],
                        "lr_target_pred_bbox_xyxy": match["pred_bbox_xyxy"],
                    }
                )
            last_for_gt[gt_id] = match
    return events


def area_fraction(box: list[float], image_width: int = 2048, image_height: int = 480) -> float:
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    return width * height / (image_width * image_height)


def rank_score(candidate: dict, min_frames: int, max_frames: int) -> float:
    """Quality prior used only after the oracle helpful/not-helpful label."""
    span = max(1, max_frames - min_frames)
    temporal_closeness = 1.0 - min(1.0, max(0.0, (candidate["frames_before"] - min_frames) / span))
    local_support = min(1.0, candidate["local_same_track_matches_15"] / 15.0)
    area_score = min(1.0, candidate["bbox_area_fraction"] / 0.01)
    return (
        0.35 * candidate["hr_iou"]
        + 0.30 * candidate["hr_confidence"]
        + 0.20 * local_support
        + 0.10 * area_score
        + 0.05 * temporal_closeness
    )


def select_diverse_top_k(candidates: list[dict], top_k: int, min_spacing: int) -> list[dict]:
    selected = []
    for helpful in (True, False):
        group = sorted(
            (candidate for candidate in candidates if candidate["oracle_helpful"] is helpful),
            key=lambda item: item["rank_score"],
            reverse=True,
        )
        for candidate in group:
            if all(
                abs(candidate["candidate_frame"] - other["candidate_frame"]) >= min_spacing
                for other in selected
            ):
                selected.append(candidate)
                if len(selected) == top_k:
                    return selected
        for candidate in group:
            if candidate not in selected:
                selected.append(candidate)
                if len(selected) == top_k:
                    return selected
    return selected


def analyze_sequence(
    sequence: str,
    dataset_root: Path,
    lowres_path: Path,
    highres_path: Path,
    threshold: float,
    fps: float,
    min_seconds: float,
    max_seconds: float,
    top_k: int,
    min_spacing: int,
) -> tuple[list[dict], list[dict], dict]:
    gt_frames = load_mot(dataset_root / sequence / "gt" / "gt.txt", ground_truth=True)
    lowres_frames = load_mot(lowres_path, ground_truth=False)
    highres_frames = load_mot(highres_path, ground_truth=False)
    lowres_index, lowres_history = build_match_index(gt_frames, lowres_frames, threshold)
    highres_index, highres_history = build_match_index(gt_frames, highres_frames, threshold)
    events = find_switch_events(lowres_index)

    min_frames = int(round(min_seconds * fps))
    max_frames = int(round(max_seconds * fps))
    all_candidates = []
    event_results = []

    for event_number, event in enumerate(events, start=1):
        target_frame = event["target_frame"]
        gt_id = event["gt_id"]
        window_start = max(0, target_frame - max_frames)
        window_end = target_frame - min_frames
        hr_track_history = highres_history.get(gt_id, [])
        target_hr = highres_index.get(target_frame, {}).get(gt_id)
        target_position = next(
            (index for index, record in enumerate(hr_track_history) if record["frame_id"] == target_frame),
            None,
        )
        previous_hr = (
            hr_track_history[target_position - 1]
            if target_position is not None and target_position > 0
            else None
        )
        hr_target_stable = bool(
            target_hr is not None
            and previous_hr is not None
            and previous_hr["pred_id"] == target_hr["pred_id"]
        )
        candidate_rows = []
        if window_end >= window_start:
            for candidate in hr_track_history:
                candidate_frame = candidate["frame_id"]
                if candidate_frame < window_start or candidate_frame > window_end:
                    continue
                target_pred_id = target_hr["pred_id"] if target_hr is not None else None
                same_hr_track = target_pred_id is not None and candidate["pred_id"] == target_pred_id
                segment = [
                    record
                    for record in hr_track_history
                    if candidate_frame <= record["frame_id"] <= target_frame
                ]
                continuity_ratio = (
                    sum(record["pred_id"] == target_pred_id for record in segment) / len(segment)
                    if segment and target_pred_id is not None
                    else 0.0
                )
                local_same = sum(
                    record["pred_id"] == candidate["pred_id"]
                    and abs(record["frame_id"] - candidate_frame) <= 15
                    for record in hr_track_history
                )
                lowres_candidate = lowres_index.get(candidate_frame, {}).get(gt_id)
                row = {
                    "sequence": sequence,
                    "event_id": f"{sequence}_{event_number:04d}",
                    "target_frame": target_frame,
                    "gt_id": gt_id,
                    "lr_old_pred_id": event["lr_old_pred_id"],
                    "lr_new_pred_id": event["lr_new_pred_id"],
                    "lr_last_match_frame": event["lr_last_match_frame"],
                    "lr_gap_frames": event["lr_gap_frames"],
                    "window_start": window_start,
                    "window_end": window_end,
                    "candidate_frame": candidate_frame,
                    "frames_before": target_frame - candidate_frame,
                    "seconds_before": round((target_frame - candidate_frame) / fps, 3),
                    "image_path": str(dataset_root / sequence / "img1" / f"{candidate_frame:06d}.jpg"),
                    "hr_pred_id": candidate["pred_id"],
                    "hr_gt_bbox_xyxy": candidate["gt_bbox_xyxy"],
                    "hr_pred_bbox_xyxy": candidate["pred_bbox_xyxy"],
                    "hr_iou": round(candidate["iou"], 6),
                    "hr_confidence": round(candidate["confidence"], 6),
                    "bbox_area_fraction": round(area_fraction(candidate["pred_bbox_xyxy"]), 8),
                    "local_same_track_matches_15": local_same,
                    "continuity_ratio_to_target": round(continuity_ratio, 6),
                    "lr_pred_id_at_candidate": lowres_candidate["pred_id"] if lowres_candidate else None,
                    "lr_iou_at_candidate": round(lowres_candidate["iou"], 6) if lowres_candidate else None,
                    "lr_confidence_at_candidate": (
                        round(lowres_candidate["confidence"], 6) if lowres_candidate else None
                    ),
                    "lr_pred_bbox_xyxy_at_candidate": (
                        lowres_candidate["pred_bbox_xyxy"] if lowres_candidate else None
                    ),
                    "hr_target_pred_id": target_pred_id,
                    "hr_target_iou": round(target_hr["iou"], 6) if target_hr else None,
                    "hr_target_confidence": round(target_hr["confidence"], 6) if target_hr else None,
                    "hr_target_pred_bbox_xyxy": target_hr["pred_bbox_xyxy"] if target_hr else None,
                    "hr_target_stable": hr_target_stable,
                    "same_hr_track_at_target": same_hr_track,
                    "oracle_helpful": bool(hr_target_stable and same_hr_track),
                    "selected_top5": False,
                }
                row["rank_score"] = round(rank_score(row, min_frames, max_frames), 6)
                candidate_rows.append(row)

        selected = select_diverse_top_k(candidate_rows, top_k, min_spacing)
        selected_keys = {(row["candidate_frame"], row["hr_pred_id"]) for row in selected}
        for row in candidate_rows:
            row["selected_top5"] = (row["candidate_frame"], row["hr_pred_id"]) in selected_keys
        helpful_count = sum(row["oracle_helpful"] for row in candidate_rows)
        recoverable = bool(hr_target_stable and helpful_count)
        selected_helpful = sum(row["oracle_helpful"] for row in selected)
        event_result = {
            **event,
            "sequence": sequence,
            "event_id": f"{sequence}_{event_number:04d}",
            "target_image_path": str(dataset_root / sequence / "img1" / f"{target_frame:06d}.jpg"),
            "window_start": window_start,
            "window_end": window_end,
            "candidate_count": len(candidate_rows),
            "helpful_candidate_count": helpful_count,
            "hr_target_matched": target_hr is not None,
            "hr_target_stable": hr_target_stable,
            "hr_target_pred_id": target_hr["pred_id"] if target_hr else None,
            "hr_target_iou": round(target_hr["iou"], 6) if target_hr else None,
            "recoverable_with_hr_history": recoverable,
            "five_helpful_frames_available": helpful_count >= top_k,
            "selected_helpful_count": selected_helpful,
            "top_frames": [
                {
                    key: row[key]
                    for key in (
                        "candidate_frame",
                        "seconds_before",
                        "image_path",
                        "hr_pred_id",
                        "hr_pred_bbox_xyxy",
                        "hr_iou",
                        "hr_confidence",
                        "bbox_area_fraction",
                        "local_same_track_matches_15",
                        "continuity_ratio_to_target",
                        "oracle_helpful",
                        "rank_score",
                    )
                }
                for row in selected
            ],
        }
        event_results.append(event_result)
        all_candidates.extend(candidate_rows)

    metrics = evaluate(gt_frames, lowres_frames, threshold)
    return event_results, all_candidates, metrics


def write_candidates_csv(path: Path, candidates: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not candidates:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)


def write_top_frames_csv(path: Path, events: list[dict]) -> None:
    fields = (
        "sequence",
        "event_id",
        "target_frame",
        "gt_id",
        "lr_old_pred_id",
        "lr_new_pred_id",
        "rank",
        "candidate_frame",
        "seconds_before",
        "image_path",
        "hr_pred_id",
        "hr_pred_bbox_xyxy",
        "hr_iou",
        "hr_confidence",
        "oracle_helpful",
        "rank_score",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            for rank, frame in enumerate(event["top_frames"], start=1):
                writer.writerow(
                    {
                        "sequence": event["sequence"],
                        "event_id": event["event_id"],
                        "target_frame": event["target_frame"],
                        "gt_id": event["gt_id"],
                        "lr_old_pred_id": event["lr_old_pred_id"],
                        "lr_new_pred_id": event["lr_new_pred_id"],
                        "rank": rank,
                        **{key: frame[key] for key in fields[7:]},
                    }
                )


def write_report(path: Path, args: argparse.Namespace, summary: dict, events: list[dict]) -> None:
    examples = [event for event in events if event["recoverable_with_hr_history"]][:10]
    output_dir = args.output_dir.resolve()
    try:
        artifact_root = output_dir.relative_to(path.resolve().parent).as_posix()
    except ValueError:
        artifact_root = output_dir.as_posix()
    lines = [
        f"# High-resolution retrieval using GT/Hungarian oracle labels: 512x120 {args.tracker_label}",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Experiment",
        "",
        f"For every ID-switch event produced by the 512x120 YOLO11l + {args.tracker_label} pipeline, this experiment searches the original-resolution {args.tracker_label} tracking result between 5 and 10 seconds before the switch. Predictions are assigned to GT identities with Hungarian matching at the configured evaluation IoU; those GT-linked identities create the oracle labels. This experiment does not perform cross-resolution embedding matching.",
        "",
        f"- Evaluation IoU: `{args.iou:.2f}`",
        f"- Frame rate: `{args.fps:.1f}` FPS",
        f"- Retrieval interval: `{args.min_seconds_before:.1f}`–`{args.max_seconds_before:.1f}` seconds before the switch",
        f"- Candidate interval: `{summary['min_frames_before']}`–`{summary['max_frames_before']}` frames before the switch",
        "- Retrieval succeeds when at least `1` helpful HR frame is found.",
        f"- Up to `{args.top_k}` frames are retained per switch as ranked alternatives; {args.top_k} helpful frames are not required.",
        f"- When multiple frames are available, ranking prefers at least `{args.min_spacing_frames}` frames of spacing to reduce near-duplicates.",
        "",
        "## Oracle retrieval potential",
        "",
        "| Measure | Count | Fraction |",
        "|---|---:|---:|",
        f"| Low-resolution ID-switch events | {summary['total_lr_switch_events']} | 100.0% |",
        f"| Unique low-resolution switch frames | {summary['unique_lr_switch_frames']} | 100.0% |",
        f"| Events with at least one HR candidate in the window | {summary['events_with_candidates']} | {summary['events_with_candidates_fraction']:.1%} |",
        f"| Events where HR is matched and stable at target | {summary['events_hr_stable_at_target']} | {summary['events_hr_stable_fraction']:.1%} |",
        f"| Recoverable events with at least one helpful HR frame | **{summary['recoverable_events']}** | **{summary['recoverable_event_fraction']:.1%}** |",
        f"| Recoverable unique target frames with at least one helpful HR frame | **{summary['recoverable_unique_frames']}** | **{summary['recoverable_unique_frame_fraction']:.1%}** |",
        f"| Events with five or more helpful HR alternatives | {summary['events_with_five_helpful_frames']} | {summary['five_helpful_fraction']:.1%} |",
        "",
        "Overlapping diagnostics for non-recoverable cases (these counts are not mutually exclusive):",
        "",
        f"- `{summary['events_without_valid_time_window']}` switches occur too early to have a complete candidate interval.",
        f"- `{summary['events_hr_unmatched_at_target']}` switches have no original-resolution match at the target frame.",
        f"- `{summary['events_hr_matched_but_unstable']}` switches are also identity-unstable in the original-resolution stream.",
        f"- `{summary['events_hr_stable_without_support']}` switches have a stable HR target but no same-track HR candidate in the requested interval.",
        "",
        "If every oracle-recoverable switch were removed while keeping FN and FP unchanged, IDSW would fall from "
        f"`{summary['total_lr_switch_events']}` to `{summary['oracle_remaining_idsw']}`, and MOTA would rise from "
        f"`{summary['lr_mota']:.3f}` to at most `{summary['oracle_mota_upper_bound']:.3f}`. IDF1 is not estimated because it requires actually merging complete predicted track segments, not editing isolated switch counts.",
        "",
    ]
    if args.baseline_summary is not None:
        with args.baseline_summary.resolve().open(encoding="utf-8") as handle:
            baseline = json.load(handle)
        combined_drop = baseline["total_lr_switch_events"] - summary["oracle_remaining_idsw"]
        combined_fraction = combined_drop / baseline["total_lr_switch_events"]
        lines.extend(
            [
                f"## Comparison with {args.baseline_label}",
                "",
                "| Pipeline | Initial IDSW | Oracle-recoverable | Recovery rate | Oracle remaining IDSW | MOTA | Oracle MOTA upper bound |",
                "|---|---:|---:|---:|---:|---:|---:|",
                f"| {args.baseline_label} | {baseline['total_lr_switch_events']} | {baseline['recoverable_events']} | {baseline['recoverable_event_fraction']:.1%} | {baseline['oracle_remaining_idsw']} | {baseline['lr_mota']:.3f} | {baseline['oracle_mota_upper_bound']:.3f} |",
                f"| {args.tracker_label} | {summary['total_lr_switch_events']} | {summary['recoverable_events']} | {summary['recoverable_event_fraction']:.1%} | {summary['oracle_remaining_idsw']} | {summary['lr_mota']:.3f} | {summary['oracle_mota_upper_bound']:.3f} |",
                "",
                f"Relative to the original {args.baseline_label} starting point, using {args.tracker_label} and then resolving every oracle-recoverable event would reduce IDSW by **{combined_drop} ({combined_fraction:.1%})**, from `{baseline['total_lr_switch_events']}` to `{summary['oracle_remaining_idsw']}`. This is an oracle upper bound across two tracker configurations, not a measured online merge result.",
                "",
            ]
        )
    lines.extend(
        [
            "## Per-sequence retrieval yield",
            "",
            "| Sequence | LR IDSW events | Recoverable with ≥1 frame | Has ≥5 alternatives | Candidate rows |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for sequence, values in summary["per_sequence"].items():
        lines.append(
            f"| {sequence} | {values['switch_events']} | {values['recoverable_events']} | "
            f"{values['events_with_five_helpful_frames']} | {values['candidate_rows']} |"
        )
    lines.extend(
        [
        "",
        "## Example ranked retrievals",
        "",
        "| Event | Target | GT ID | LR ID change | Ranked HR frames (seconds before target) |",
        "|---|---:|---:|---|---|",
        ]
    )
    for event in examples:
        frames = ", ".join(
            f"{frame['candidate_frame']} ({frame['seconds_before']:.1f}s)"
            for frame in event["top_frames"]
        )
        lines.append(
            f"| {event['event_id']} | {event['target_frame']} | {event['gt_id']} | "
            f"{event['lr_old_pred_id']} → {event['lr_new_pred_id']} | {frames} |"
        )
    lines.extend(
        [
            "",
            "## Ranking data",
            "",
            "Candidates are labeled with `oracle_helpful`. A switch is counted as recoverable as soon as at least one candidate has this label. The ranking prior then uses HR match IoU, detector confidence, local same-track support, box area, and temporal proximity. IoU and GT identity are supervision signals and must not be used as online inputs to the future ranking model. Deployable inputs can include confidence, crop size, sharpness, motion/occlusion state, and an appearance embedding.",
            "",
            "The output retains up to five candidates as alternatives, but the runtime system may stop after retrieving the first sufficiently strong frame. Multiple candidates are useful only when the first frame is blurred, occluded, ambiguous, or fails the ReID threshold.",
            "",
            "## Deployment implication",
            "",
            f"The current {args.tracker_label} pipeline does not automatically consume an isolated frame retrieved from 5–10 seconds ago as long-term identity evidence. A deployable version needs a retrieval-aware association layer: trigger on tracking uncertainty, request ranked HR crops one by one, extract an appearance embedding, and stop as soon as one frame gives a sufficiently confident identity match. Additional ranked frames are fallback evidence before merging or preserving the identity.",
            "",
            f"This report is therefore an oracle upper-bound and ranking-dataset experiment, not a claim that inserting isolated HR frames into the existing {args.tracker_label} loop already produces these gains.",
            "",
            "## Artifacts",
            "",
            f"- [Summary]({artifact_root}/summary.json)",
            f"- [All switch events with top-five frames]({artifact_root}/switch_events_top5.json)",
            f"- [Top-five frame table]({artifact_root}/top5_frames.csv)",
            f"- [All ranking candidates and labels]({artifact_root}/ranking_candidates.csv)",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.top_k <= 0 or args.min_spacing_frames < 0:
        raise ValueError("--fps and --top-k must be positive; spacing must be non-negative")
    if not 0 < args.min_seconds_before < args.max_seconds_before:
        raise ValueError("expected 0 < --min-seconds-before < --max-seconds-before")
    if not 0 < args.iou <= 1:
        raise ValueError("--iou must be in (0, 1]")

    dataset_root = args.dataset_root.resolve()
    lowres_dir = args.lowres_dir.resolve()
    highres_dir = args.highres_dir.resolve()
    output_dir = args.output_dir.resolve()
    sequences = args.sequences or discover_sequences(dataset_root)
    all_events = []
    all_candidates = []
    lowres_metrics = []
    per_sequence = {}

    for sequence in sequences:
        lowres_path = lowres_dir / args.lowres_pattern.format(sequence=sequence)
        highres_path = highres_dir / args.highres_pattern.format(sequence=sequence)
        if not lowres_path.is_file():
            raise FileNotFoundError(lowres_path)
        if not highres_path.is_file():
            raise FileNotFoundError(highres_path)
        events, candidates, metrics = analyze_sequence(
            sequence,
            dataset_root,
            lowres_path,
            highres_path,
            args.iou,
            args.fps,
            args.min_seconds_before,
            args.max_seconds_before,
            args.top_k,
            args.min_spacing_frames,
        )
        all_events.extend(events)
        all_candidates.extend(candidates)
        lowres_metrics.append(metrics)
        per_sequence[sequence] = {
            "switch_events": len(events),
            "recoverable_events": sum(event["recoverable_with_hr_history"] for event in events),
            "events_with_five_helpful_frames": sum(event["five_helpful_frames_available"] for event in events),
            "candidate_rows": len(candidates),
        }

    total_events = len(all_events)
    unique_frames = {(event["sequence"], event["target_frame"]) for event in all_events}
    recoverable_events = [event for event in all_events if event["recoverable_with_hr_history"]]
    recoverable_frames = {(event["sequence"], event["target_frame"]) for event in recoverable_events}
    events_with_candidates = sum(bool(event["candidate_count"]) for event in all_events)
    stable_events = sum(event["hr_target_stable"] for event in all_events)
    five_helpful = sum(event["five_helpful_frames_available"] for event in all_events)
    events_without_window = sum(event["window_end"] < event["window_start"] for event in all_events)
    events_hr_unmatched = sum(not event["hr_target_matched"] for event in all_events)
    events_hr_unstable = sum(
        event["hr_target_matched"] and not event["hr_target_stable"] for event in all_events
    )
    events_hr_stable_without_support = sum(
        event["hr_target_stable"] and not event["recoverable_with_hr_history"] for event in all_events
    )
    gt_total = sum(int(metrics["gt"]) for metrics in lowres_metrics)
    fp_total = sum(int(metrics["fp"]) for metrics in lowres_metrics)
    fn_total = sum(int(metrics["fn"]) for metrics in lowres_metrics)
    idsw_total = sum(int(metrics["id_switches"]) for metrics in lowres_metrics)
    lr_mota = 1.0 - (fn_total + fp_total + idsw_total) / gt_total
    oracle_remaining_idsw = idsw_total - len(recoverable_events)
    oracle_mota = 1.0 - (fn_total + fp_total + oracle_remaining_idsw) / gt_total
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sequences": sequences,
        "fps": args.fps,
        "min_seconds_before": args.min_seconds_before,
        "max_seconds_before": args.max_seconds_before,
        "min_frames_before": int(round(args.min_seconds_before * args.fps)),
        "max_frames_before": int(round(args.max_seconds_before * args.fps)),
        "top_k": args.top_k,
        "evaluation_iou": args.iou,
        "tracker_label": args.tracker_label,
        "total_lr_switch_events": total_events,
        "unique_lr_switch_frames": len(unique_frames),
        "events_with_candidates": events_with_candidates,
        "events_with_candidates_fraction": events_with_candidates / total_events if total_events else 0.0,
        "events_hr_stable_at_target": stable_events,
        "events_hr_stable_fraction": stable_events / total_events if total_events else 0.0,
        "recoverable_events": len(recoverable_events),
        "recoverable_event_fraction": len(recoverable_events) / total_events if total_events else 0.0,
        "recoverable_unique_frames": len(recoverable_frames),
        "recoverable_unique_frame_fraction": len(recoverable_frames) / len(unique_frames) if unique_frames else 0.0,
        "events_with_five_helpful_frames": five_helpful,
        "five_helpful_fraction": five_helpful / total_events if total_events else 0.0,
        "events_without_valid_time_window": events_without_window,
        "events_hr_unmatched_at_target": events_hr_unmatched,
        "events_hr_matched_but_unstable": events_hr_unstable,
        "events_hr_stable_without_support": events_hr_stable_without_support,
        "ranking_candidate_rows": len(all_candidates),
        "gt_instances": gt_total,
        "lr_fp": fp_total,
        "lr_fn": fn_total,
        "lr_mota": lr_mota,
        "oracle_remaining_idsw": oracle_remaining_idsw,
        "oracle_mota_upper_bound": oracle_mota,
        "per_sequence": per_sequence,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "switch_events_top5.json").write_text(
        json.dumps(all_events, indent=2) + "\n", encoding="utf-8"
    )
    write_candidates_csv(output_dir / "ranking_candidates.csv", all_candidates)
    write_top_frames_csv(output_dir / "top5_frames.csv", all_events)
    write_report(args.report, args, summary, all_events)
    print(json.dumps({key: value for key, value in summary.items() if key != "per_sequence"}, indent=2))
    print(f"Report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
