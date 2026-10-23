#!/usr/bin/env python3
"""Rank historical full-resolution detections using cross-resolution ReID embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.trackers.utils.reid import ReID

from evaluate_mot import load_mot
from rank_highres_retrieval_frames import build_match_index, find_switch_events


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = CODE_DIR.parent / "QuadTrack" / "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--lowres-dir", type=Path, required=True)
    parser.add_argument("--highres-dir", type=Path, required=True)
    parser.add_argument("--lowres-pattern", required=True)
    parser.add_argument("--highres-pattern", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-seconds-before", type=float, default=0.0)
    parser.add_argument("--max-seconds-before", type=float, default=5.0)
    parser.add_argument("--evaluation-iou", type=float, default=0.5)
    parser.add_argument("--similarity-threshold", type=float, default=0.30)
    parser.add_argument("--query-history-frames", type=int, default=30)
    parser.add_argument("--query-samples", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-spacing-frames", type=int, default=15)
    parser.add_argument("--lowres-width", type=int, default=512)
    parser.add_argument("--lowres-height", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def discover_sequences(dataset_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "gt" / "gt.txt").is_file() and (path / "img1").is_dir()
    )


def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = (float(value) for value in box)
    return np.asarray([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float32)


def normalized(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else None


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    top_left = np.maximum(box1[:2], box2[:2])
    bottom_right = np.minimum(box1[2:], box2[2:])
    width_height = np.maximum(0.0, bottom_right - top_left)
    intersection = float(width_height[0] * width_height[1])
    area1 = float(max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1]))
    area2 = float(max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1]))
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def quality(row: tuple, image_area: float) -> float:
    _, box, confidence = row
    area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    return float(confidence) * min(1.0, np.sqrt(area / max(image_area * 0.01, 1.0)))


def select_query_rows(
    observations: list[tuple[int, tuple]],
    end_frame: int,
    history_frames: int,
    samples: int,
    image_area: float,
) -> list[tuple[int, tuple]]:
    start_frame = max(1, end_frame - history_frames + 1)
    eligible = [item for item in observations if start_frame <= item[0] <= end_frame]
    eligible.sort(key=lambda item: (quality(item[1], image_area), item[0]), reverse=True)
    selected: list[tuple[int, tuple]] = []
    for item in eligible:
        if all(abs(item[0] - previous[0]) >= 2 for previous in selected):
            selected.append(item)
            if len(selected) == samples:
                return sorted(selected)
    for item in eligible:
        if item not in selected:
            selected.append(item)
            if len(selected) == samples:
                break
    return sorted(selected)


def run_crop_batches(encoder: ReID, crops: list[np.ndarray]) -> list[np.ndarray | None]:
    """Run the Ultralytics ReID backend in efficient batches while preserving empty crops."""
    valid = [bool(crop.size) for crop in crops]
    valid_crops = [crop for crop, keep in zip(crops, valid) if keep]
    if not valid_crops:
        return [None] * len(crops)
    valid_features: list[np.ndarray] = []
    external_batch = 256
    with torch.no_grad():
        for start in range(0, len(valid_crops), external_batch):
            group = valid_crops[start : start + external_batch]
            batch = encoder._crops_to_tensor(group)
            fixed_batch = encoder.batch_size
            if fixed_batch is None:
                output = encoder.model(batch)
                if isinstance(output, (list, tuple)):
                    output = output[0]
            else:
                parts = []
                for offset in range(0, batch.shape[0], fixed_batch):
                    chunk = batch[offset : offset + fixed_batch]
                    actual = chunk.shape[0]
                    if actual < fixed_batch:
                        chunk = torch.cat(
                            [chunk, chunk[-1:].expand(fixed_batch - actual, *chunk.shape[1:])], dim=0
                        )
                    out = encoder.model(chunk)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    parts.append(out[:actual].clone())
                output = torch.cat(parts, dim=0)
            for feature in output:
                vector = normalized(feature.detach().cpu().numpy())
                valid_features.append(vector)
    iterator = iter(valid_features)
    return [next(iterator) if keep else None for keep in valid]


def extract_embeddings(
    encoder: ReID,
    dataset_root: Path,
    sequence: str,
    lowres_rows: dict[tuple[int, int], tuple],
    highres_frames: dict[int, list[tuple]],
    required_highres_frames: set[int],
    lowres_size: tuple[int, int],
) -> tuple[dict, dict]:
    lr_by_frame: dict[int, list[tuple[tuple[int, int], tuple]]] = defaultdict(list)
    for key, row in lowres_rows.items():
        lr_by_frame[key[0]].append((key, row))

    embeddings_lr: dict[tuple[int, int], np.ndarray | None] = {}
    embeddings_hr: dict[tuple[int, int], np.ndarray | None] = {}
    pending_crops: list[np.ndarray] = []
    pending_keys: list[tuple[str, tuple[int, int]]] = []

    def flush() -> None:
        if not pending_crops:
            return
        features = run_crop_batches(encoder, pending_crops)
        for (stream, key), feature in zip(pending_keys, features):
            (embeddings_lr if stream == "lr" else embeddings_hr)[key] = feature
        pending_crops.clear()
        pending_keys.clear()

    needed_frames = sorted(set(lr_by_frame) | required_highres_frames)
    image_dir = dataset_root / sequence / "img1"
    for frame_id in needed_frames:
        path = image_dir / f"{frame_id:06d}.jpg"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)

        hr_rows = highres_frames.get(frame_id, []) if frame_id in required_highres_frames else []
        if hr_rows:
            detections = np.asarray([xyxy_to_xywh(row[1]) for row in hr_rows], dtype=np.float32)
            crops = encoder._crop_detections(image, detections)
            for index, crop in enumerate(crops):
                pending_crops.append(crop)
                pending_keys.append(("hr", (frame_id, index)))

        if frame_id in lr_by_frame:
            resized = cv2.resize(image, lowres_size, interpolation=cv2.INTER_AREA)
            scale_x = lowres_size[0] / image.shape[1]
            scale_y = lowres_size[1] / image.shape[0]
            entries = lr_by_frame[frame_id]
            scaled_boxes = []
            for _, row in entries:
                box = np.asarray(row[1], dtype=np.float32).copy()
                box[[0, 2]] *= scale_x
                box[[1, 3]] *= scale_y
                scaled_boxes.append(xyxy_to_xywh(box))
            crops = encoder._crop_detections(resized, np.asarray(scaled_boxes, dtype=np.float32))
            for (key, _), crop in zip(entries, crops):
                pending_crops.append(crop)
                pending_keys.append(("lr", key))

        if len(pending_crops) >= 256:
            flush()
    flush()
    return embeddings_lr, embeddings_hr


def select_spaced(candidates: list[dict], top_k: int, spacing: int) -> list[dict]:
    # Similarity is the ranking signal. Prefer the frame nearest to the switch only
    # when two candidates have the same rounded similarity.
    ordered = sorted(
        candidates,
        key=lambda item: (item["bridge_similarity"], item["candidate_frame"]),
        reverse=True,
    )
    selected = []
    for candidate in ordered:
        if all(abs(candidate["candidate_frame"] - item["candidate_frame"]) >= spacing for item in selected):
            selected.append(candidate)
            if len(selected) == top_k:
                return selected
    for candidate in ordered:
        if candidate not in selected:
            selected.append(candidate)
            if len(selected) == top_k:
                break
    return sorted(selected, key=lambda item: item["bridge_similarity"], reverse=True)


def analyze_sequence(args: argparse.Namespace, encoder: ReID, sequence: str) -> tuple[list[dict], dict]:
    dataset_root = args.dataset_root.resolve()
    gt_frames = load_mot(dataset_root / sequence / "gt" / "gt.txt", ground_truth=True)
    lowres_frames = load_mot(
        args.lowres_dir.resolve() / args.lowres_pattern.format(sequence=sequence), ground_truth=False
    )
    highres_frames = load_mot(
        args.highres_dir.resolve() / args.highres_pattern.format(sequence=sequence), ground_truth=False
    )
    lowres_index, _ = build_match_index(gt_frames, lowres_frames, args.evaluation_iou)
    switches = find_switch_events(lowres_index)

    observations_by_track: dict[int, list[tuple[int, tuple]]] = defaultdict(list)
    for frame_id, rows in lowres_frames.items():
        for row in rows:
            observations_by_track[int(row[0])].append((frame_id, row))

    min_frames = int(round(args.min_seconds_before * args.fps))
    max_frames = int(round(args.max_seconds_before * args.fps))
    lowres_rows_needed: dict[tuple[int, int], tuple] = {}
    required_highres_frames: set[int] = set()
    prepared = []
    # MOT boxes are stored in original-frame coordinates; area fraction is resolution invariant.
    image_area = float(2048 * 480)

    for event_number, event in enumerate(switches, start=1):
        target_frame = int(event["target_frame"])
        old_id = int(event["lr_old_pred_id"])
        new_id = int(event["lr_new_pred_id"])
        query_rows = select_query_rows(
            observations_by_track.get(old_id, []),
            int(event["lr_last_match_frame"]),
            args.query_history_frames,
            args.query_samples,
            image_area,
        )
        for frame_id, row in query_rows:
            lowres_rows_needed[(frame_id, int(row[0]))] = row
        target_row = next(
            (row for row in lowres_frames.get(target_frame, []) if int(row[0]) == new_id), None
        )
        if target_row is not None:
            lowres_rows_needed[(target_frame, new_id)] = target_row
        start = max(1, target_frame - max_frames)
        # Never include the switch frame itself. With the default 0–5 second
        # lookback, the exact candidate interval is T-150 ... T-1 at 30 FPS.
        end = target_frame - max(1, min_frames)
        if end >= start:
            required_highres_frames.update(range(start, end + 1))
        prepared.append(
            {
                **event,
                "event_id": f"{sequence}_{event_number:04d}",
                "sequence": sequence,
                "window_start": start,
                "window_end": end,
                "query_keys": [(frame_id, int(row[0])) for frame_id, row in query_rows],
                "target_key": (target_frame, new_id) if target_row is not None else None,
            }
        )

    embeddings_lr, embeddings_hr = extract_embeddings(
        encoder,
        dataset_root,
        sequence,
        lowres_rows_needed,
        highres_frames,
        required_highres_frames,
        (args.lowres_width, args.lowres_height),
    )

    results = []
    for event in prepared:
        query_vectors = [embeddings_lr.get(tuple(key)) for key in event["query_keys"]]
        query_vectors = [vector for vector in query_vectors if vector is not None]
        target_vector = embeddings_lr.get(tuple(event["target_key"])) if event["target_key"] else None
        query_prototype = normalized(np.mean(query_vectors, axis=0)) if query_vectors else None
        frame_best: list[dict] = []
        correct_available = False
        candidate_detections = 0

        if query_prototype is not None and target_vector is not None and event["window_end"] >= event["window_start"]:
            # Visit the most recent frame first, then move backward in time.
            for frame_id in range(event["window_end"], event["window_start"] - 1, -1):
                gt_row = next(
                    (row for row in gt_frames.get(frame_id, []) if int(row[0]) == int(event["gt_id"])), None
                )
                candidates_this_frame = []
                for index, row in enumerate(highres_frames.get(frame_id, [])):
                    feature = embeddings_hr.get((frame_id, index))
                    if feature is None:
                        continue
                    candidate_detections += 1
                    sim_old = float(np.dot(query_prototype, feature))
                    sim_target = float(np.dot(target_vector, feature))
                    bridge = min(sim_old, sim_target)
                    correct = bool(gt_row is not None and iou(np.asarray(row[1]), np.asarray(gt_row[1])) >= args.evaluation_iou)
                    correct_available = correct_available or correct
                    box = [round(float(value), 2) for value in row[1]]
                    candidates_this_frame.append(
                        {
                            "candidate_frame": frame_id,
                            "seconds_before": round((event["target_frame"] - frame_id) / args.fps, 3),
                            "hr_pred_id": int(row[0]),
                            "hr_bbox_xyxy": box,
                            "hr_confidence": round(float(row[2]), 6),
                            "similarity_to_old_track": round(sim_old, 6),
                            "similarity_to_target": round(sim_target, 6),
                            "bridge_similarity": round(bridge, 6),
                            "correct_gt_identity": correct,
                            "image_path": str(dataset_root / sequence / "img1" / f"{frame_id:06d}.jpg"),
                        }
                    )
                if candidates_this_frame:
                    frame_best.append(max(candidates_this_frame, key=lambda item: item["bridge_similarity"]))

        selected = select_spaced(frame_best, args.top_k, args.min_spacing_frames)
        top1 = selected[0] if selected else None
        top1_accepted = bool(top1 and top1["bridge_similarity"] >= args.similarity_threshold)
        top1_correct = bool(top1 and top1["correct_gt_identity"])
        top5_correct = any(item["correct_gt_identity"] for item in selected)
        result = {
            "sequence": sequence,
            "event_id": event["event_id"],
            "target_frame": int(event["target_frame"]),
            "gt_id": int(event["gt_id"]),
            "lr_old_pred_id": int(event["lr_old_pred_id"]),
            "lr_new_pred_id": int(event["lr_new_pred_id"]),
            "lr_last_match_frame": int(event["lr_last_match_frame"]),
            "lr_gap_frames": int(event["lr_gap_frames"]),
            "window_start": int(event["window_start"]),
            "window_end": int(event["window_end"]),
            "query_frames": [int(key[0]) for key in event["query_keys"]],
            "query_embedding_count": len(query_vectors),
            "target_embedding_available": target_vector is not None,
            "candidate_detection_count": candidate_detections,
            "correct_identity_available_in_window": correct_available,
            "top1_accepted": top1_accepted,
            "top1_correct": top1_correct,
            "top1_accepted_correct": bool(top1_accepted and top1_correct),
            "top1_improvable": top1_correct,
            "top5_improvable": top5_correct,
            "top_frames": selected,
        }
        results.append(result)

    diagnostics = {
        "switch_events": len(results),
        "events_with_query_embedding": sum(bool(row["query_embedding_count"]) for row in results),
        "events_with_target_embedding": sum(row["target_embedding_available"] for row in results),
        "events_with_candidates": sum(bool(row["candidate_detection_count"]) for row in results),
        "events_with_correct_identity_available": sum(row["correct_identity_available_in_window"] for row in results),
        "top1_accepted": sum(row["top1_accepted"] for row in results),
        "top1_improvable": sum(row["top1_improvable"] for row in results),
        "top5_improvable": sum(row["top5_improvable"] for row in results),
    }
    return results, diagnostics


def write_top5_csv(path: Path, events: list[dict]) -> None:
    fields = (
        "sequence", "event_id", "target_frame", "gt_id", "lr_old_pred_id", "lr_new_pred_id",
        "rank", "candidate_frame", "seconds_before", "hr_pred_id", "hr_bbox_xyxy", "hr_confidence",
        "similarity_to_old_track", "similarity_to_target", "bridge_similarity", "correct_gt_identity",
        "image_path",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            for rank, candidate in enumerate(event["top_frames"], start=1):
                writer.writerow(
                    {
                        "sequence": event["sequence"],
                        "event_id": event["event_id"],
                        "target_frame": event["target_frame"],
                        "gt_id": event["gt_id"],
                        "lr_old_pred_id": event["lr_old_pred_id"],
                        "lr_new_pred_id": event["lr_new_pred_id"],
                        "rank": rank,
                        **candidate,
                    }
                )


def build_summary(args: argparse.Namespace, events: list[dict], per_sequence: dict) -> dict:
    total = len(events)
    available = sum(event["correct_identity_available_in_window"] for event in events)
    unique_frames = {(event["sequence"], event["target_frame"]) for event in events}
    top1 = [event for event in events if event["top1_improvable"]]
    top5 = [event for event in events if event["top5_improvable"]]
    sweep = {}
    for threshold in (0.20, 0.30, 0.40, 0.50, 0.60):
        accepted = [
            event for event in events
            if event["top_frames"] and event["top_frames"][0]["bridge_similarity"] >= threshold
        ]
        correct = [event for event in accepted if event["top_frames"][0]["correct_gt_identity"]]
        sweep[f"{threshold:.2f}"] = {
            "accepted": len(accepted),
            "correct": len(correct),
            "wrong": len(accepted) - len(correct),
            "precision": len(correct) / len(accepted) if accepted else 0.0,
            "recoverable_fraction_all_switches": len(correct) / total if total else 0.0,
        }
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": "cross_resolution_reid_embedding",
        "model": str(args.model.resolve()),
        "sequences": sorted(per_sequence),
        "fps": args.fps,
        "retrieval_seconds": [args.min_seconds_before, args.max_seconds_before],
        "retrieval_frames": [
            int(round(args.min_seconds_before * args.fps)),
            int(round(args.max_seconds_before * args.fps)),
        ],
        "query_history_frames": args.query_history_frames,
        "query_samples": args.query_samples,
        "top_k": args.top_k,
        "similarity_threshold": args.similarity_threshold,
        "evaluation_iou": args.evaluation_iou,
        "total_lr_switch_events": total,
        "unique_lr_switch_frames": len(unique_frames),
        "events_with_correct_identity_available": available,
        "correct_identity_availability_fraction": available / total if total else 0.0,
        "top1_improvable_events": len(top1),
        "top1_improvable_fraction": len(top1) / total if total else 0.0,
        "top1_recall_when_available": len(top1) / available if available else 0.0,
        "top1_improvable_unique_frames": len({(e['sequence'], e['target_frame']) for e in top1}),
        "top5_improvable_events": len(top5),
        "top5_improvable_fraction": len(top5) / total if total else 0.0,
        "top5_recall_when_available": len(top5) / available if available else 0.0,
        "estimated_remaining_idsw_top1": total - len(top1),
        "estimated_remaining_idsw_top5": total - len(top5),
        "threshold_sweep": sweep,
        "per_sequence": per_sequence,
    }


def write_report(path: Path, args: argparse.Namespace, summary: dict, events: list[dict]) -> None:
    examples = [event for event in events if event["top1_improvable"]][:10]
    root = args.output_dir.resolve()
    try:
        artifact = root.relative_to(path.resolve().parent).as_posix()
    except ValueError:
        artifact = root.as_posix()
    lines = [
        "# Cross-resolution ReID embedding retrieval for 512x120 ID switches",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## What this experiment does",
        "",
        "This experiment performs actual appearance-embedding retrieval. For each GT-identified ID-switch event in the 512x120 stream, it builds an embedding prototype from up to five observations of the old low-resolution track, embeds the switched target detection, searches full-resolution tracking detections in the immediately preceding lookback window, and ranks candidates by the minimum cosine similarity to the old-track prototype and target embedding.",
        "",
        "Ground truth is not used to select or rank candidates. It is used only to identify the evaluation switch events and to label whether the selected full-resolution box is the correct person.",
        "",
        f"- ReID extractor: `{Path(summary['model']).name}` (512-dimensional L2-normalized crop embeddings)",
        f"- Low-resolution query input: physical `{args.lowres_width}x{args.lowres_height}` crops",
        f"- Old-track gallery: up to `{args.query_samples}` observations from the preceding `{args.query_history_frames}` frames",
        f"- Historical search window: the previous `{args.max_seconds_before:.1f}` seconds, excluding the switch frame itself; at `{args.fps:g}` FPS this is exactly `T-{summary['retrieval_frames'][1]}` through `T-1`",
        "- Search order: nearest historical frame first, then progressively older frames; ReID bridge similarity remains the ranking score, with recency used only to break equal-score ties",
        f"- Exploratory acceptance threshold: cosine bridge similarity `>= {args.similarity_threshold:.2f}`; threshold-free top-1/top-5 ranking accuracy is also reported",
        f"- Correct-person evaluation: selected HR box IoU `>= {args.evaluation_iou:.2f}` with the switch identity's GT box",
        "",
        "## Main result",
        "",
        "| Measure | Count | Fraction |",
        "|---|---:|---:|",
        f"| Low-resolution ID-switch events | {summary['total_lr_switch_events']} | 100.0% |",
        f"| Events where the correct HR identity exists in the search window | {summary['events_with_correct_identity_available']} | {summary['correct_identity_availability_fraction']:.1%} |",
        f"| Top-1 embedding ranking selects the correct HR identity | **{summary['top1_improvable_events']}** | **{summary['top1_improvable_fraction']:.1%}** |",
        f"| Unique target frames supported by top-1 retrieval | **{summary['top1_improvable_unique_frames']}** | — |",
        f"| Correct identity appears in top-5 ranking | **{summary['top5_improvable_events']}** | **{summary['top5_improvable_fraction']:.1%}** |",
        "",
        f"Among events where a correct HR detection is available, top-1 retrieval recall is `{summary['top1_recall_when_available']:.1%}` and top-5 recall is `{summary['top5_recall_when_available']:.1%}`.",
        "",
        f"If every correct top-1 bridge led to a successful identity merge, IDSW would be estimated to fall from `{summary['total_lr_switch_events']}` to `{summary['estimated_remaining_idsw_top1']}`. With an oracle choice among the top five, the estimate is `{summary['estimated_remaining_idsw_top5']}`. These are retrieval-supported estimates, not metrics from rerunning the tracker after track merging.",
        "",
        "## Threshold sweep for the top-ranked frame",
        "",
        "| Cosine threshold | Accepted | Correct | Wrong | Precision | Correct / all switches |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold, values in summary["threshold_sweep"].items():
        lines.append(
            f"| {threshold} | {values['accepted']} | {values['correct']} | {values['wrong']} | "
            f"{values['precision']:.1%} | {values['recoverable_fraction_all_switches']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Per-sequence retrieval result",
            "",
            "| Sequence | IDSW events | Correct HR available | Top-1 improvable | Top-5 improvable |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for sequence, values in summary["per_sequence"].items():
        lines.append(
            f"| {sequence} | {values['switch_events']} | {values['events_with_correct_identity_available']} | "
            f"{values['top1_improvable']} | {values['top5_improvable']} |"
        )
    lines.extend(
        [
            "",
            "## Example successful top-1 retrievals",
            "",
            "| Event | Target | LR ID change | Query frames | HR frame | Similarity old / target / bridge |",
            "|---|---:|---|---|---:|---|",
        ]
    )
    for event in examples:
        top = event["top_frames"][0]
        queries = ", ".join(str(frame) for frame in event["query_frames"])
        lines.append(
            f"| {event['event_id']} | {event['target_frame']} | {event['lr_old_pred_id']} → {event['lr_new_pred_id']} | "
            f"{queries} | {top['candidate_frame']} | {top['similarity_to_old_track']:.3f} / "
            f"{top['similarity_to_target']:.3f} / {top['bridge_similarity']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "- Ranking is based only on appearance embeddings; GT identity and IoU do not enter the ranking score.",
            "- Switch events are still supplied by the evaluator using GT. A deployable system needs its own uncertainty trigger.",
            "- The experiment does not yet inject retrieved embeddings into BoT-SORT or rewrite/merge complete track segments, so post-merge IDF1 and MOTA are not claimed.",
            "- The standalone crop model is an official Ultralytics ReID extractor. It differs from the native YOLO11 feature-map embeddings used by the earlier BoT-SORT runs; this distinction is intentional and recorded for reproducibility.",
            "",
            "## Artifacts",
            "",
            f"- [Summary]({artifact}/summary.json)",
            f"- [All switch events and ranked frames]({artifact}/switch_events_top5.json)",
            f"- [Top-five retrieval table]({artifact}/top5_frames.csv)",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0 <= args.min_seconds_before < args.max_seconds_before:
        raise ValueError("expected 0 <= min seconds < max seconds")
    if not -1 <= args.similarity_threshold <= 1:
        raise ValueError("similarity threshold must be in [-1, 1]")
    sequences = args.sequences or discover_sequences(args.dataset_root.resolve())
    encoder = ReID(str(args.model.resolve()), device=args.device)
    all_events = []
    per_sequence = {}
    for sequence in sequences:
        events, diagnostics = analyze_sequence(args, encoder, sequence)
        all_events.extend(events)
        per_sequence[sequence] = diagnostics
        print(
            f"{sequence}: switches={diagnostics['switch_events']} "
            f"top1={diagnostics['top1_improvable']} top5={diagnostics['top5_improvable']}"
        )

    summary = build_summary(args, all_events, per_sequence)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "switch_events_top5.json").write_text(
        json.dumps(all_events, indent=2) + "\n", encoding="utf-8"
    )
    write_top5_csv(output_dir / "top5_frames.csv", all_events)
    write_report(args.report.resolve(), args, summary, all_events)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"per_sequence", "threshold_sweep"}}, indent=2))
    print(f"Report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
