#!/usr/bin/env python3
"""Extract annotated before/after evidence for MOT identity switches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluate_mot import iou_matrix, load_mot


CODE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Sequence img1 directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional number of representative switches to render; longest gaps are selected.",
    )
    return parser.parse_args()


def frame_matches(gt_rows, pred_rows, threshold: float) -> list[tuple[int, int, float]]:
    gt_boxes = np.asarray([row[1] for row in gt_rows], dtype=float).reshape(-1, 4)
    pred_boxes = np.asarray([row[1] for row in pred_rows], dtype=float).reshape(-1, 4)
    if not len(gt_rows) or not len(pred_rows):
        return []
    overlaps = iou_matrix(gt_boxes, pred_boxes)
    gt_indices, pred_indices = linear_sum_assignment(-overlaps)
    return [
        (int(gt_index), int(pred_index), float(overlaps[gt_index, pred_index]))
        for gt_index, pred_index in zip(gt_indices, pred_indices)
        if overlaps[gt_index, pred_index] >= threshold
    ]


def find_switches(gt_frames, pred_frames, threshold: float) -> list[dict]:
    last_for_gt: dict[int, dict] = {}
    switches = []
    for frame_id in sorted(set(gt_frames) | set(pred_frames)):
        gt_rows = gt_frames.get(frame_id, [])
        pred_rows = pred_frames.get(frame_id, [])
        for gt_index, pred_index, overlap in frame_matches(gt_rows, pred_rows, threshold):
            gt_id = gt_rows[gt_index][0]
            pred_id = pred_rows[pred_index][0]
            previous = last_for_gt.get(gt_id)
            if previous is not None and previous["pred_id"] != pred_id:
                switches.append(
                    {
                        "gt_id": gt_id,
                        "old_pred_id": previous["pred_id"],
                        "new_pred_id": pred_id,
                        "last_matched_frame": previous["frame_id"],
                        "switch_frame": frame_id,
                        "gap_frames": frame_id - previous["frame_id"] - 1,
                        "old_iou": previous["iou"],
                        "new_iou": overlap,
                        "old_confidence": previous["confidence"],
                        "new_confidence": pred_rows[pred_index][2],
                    }
                )
            last_for_gt[gt_id] = {
                "pred_id": pred_id,
                "frame_id": frame_id,
                "iou": overlap,
                "confidence": pred_rows[pred_index][2],
            }
    return switches


def image_path(image_dir: Path, frame_id: int) -> Path:
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        path = image_dir / f"{frame_id:06d}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No image found for frame {frame_id} under {image_dir}")


def draw_label(frame, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    top = max(0, y - height - baseline - 5)
    cv2.rectangle(frame, (x, top), (x + width + 6, y), color, -1)
    cv2.putText(
        frame,
        text,
        (x + 3, max(height, y - baseline - 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def annotate_frame(
    image_dir: Path,
    frame_id: int,
    gt_rows,
    pred_rows,
    threshold: float,
    target_gt_id: int,
    title: str,
) -> np.ndarray:
    frame = cv2.imread(str(image_path(image_dir, frame_id)))
    if frame is None:
        raise RuntimeError(f"Could not decode frame {frame_id}")
    height, width = frame.shape[:2]
    matches = frame_matches(gt_rows, pred_rows, threshold)
    matched_prediction = {gt_index: (pred_index, overlap) for gt_index, pred_index, overlap in matches}

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 48), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.putText(frame, title, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    for gt_index, (gt_id, box, _) in enumerate(gt_rows):
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        target = gt_id == target_gt_id
        color = (0, 255, 0) if target else (70, 150, 70)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4 if target else 1)
        label = f"GT {gt_id}"
        if gt_index in matched_prediction:
            pred_index, overlap = matched_prediction[gt_index]
            label += f" -> PRED {pred_rows[pred_index][0]} IoU {overlap:.2f}"
        elif target:
            label += " -> NO MATCH"
        draw_label(frame, label, (max(0, x1), max(50, y1)), color)

    for pred_index, (pred_id, box, confidence) in enumerate(pred_rows):
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        is_target_match = any(
            gt_rows[gt_index][0] == target_gt_id and matched_pred_index == pred_index
            for gt_index, matched_pred_index, _ in matches
        )
        color = (0, 165, 255) if is_target_match else (255, 180, 40)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_target_match else 1)
        draw_label(frame, f"PRED {pred_id} conf {confidence:.2f}", (max(0, x1), min(height - 1, y2)), color)
    return frame


def main() -> None:
    args = parse_args()
    gt_frames = load_mot(args.gt.resolve(), ground_truth=True)
    pred_frames = load_mot(args.pred.resolve(), ground_truth=False)
    switches = find_switches(gt_frames, pred_frames, args.iou)
    total_switches = len(switches)
    if args.max_cases is not None:
        if args.max_cases <= 0:
            raise ValueError("--max-cases must be positive")
        switches = sorted(switches, key=lambda event: event["gap_frames"], reverse=True)[: args.max_cases]
        switches.sort(key=lambda event: event["switch_frame"])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for case_number, event in enumerate(switches, start=1):
        current = event["switch_frame"]
        frame_specs = [
            ("last_match", event["last_matched_frame"], f"Last match: GT {event['gt_id']} -> PRED {event['old_pred_id']}"),
            ("previous", current - 1, f"Immediately before switch: GT {event['gt_id']}"),
            ("switch", current, f"ID switch: GT {event['gt_id']} -> PRED {event['new_pred_id']}"),
        ]
        rendered = []
        for label, frame_id, title in frame_specs:
            annotated = annotate_frame(
                args.images.resolve(),
                frame_id,
                gt_frames.get(frame_id, []),
                pred_frames.get(frame_id, []),
                args.iou,
                event["gt_id"],
                f"Frame {frame_id} | {title}",
            )
            path = output_dir / f"case_{case_number:02d}_{label}_f{frame_id:06d}.jpg"
            if not cv2.imwrite(str(path), annotated):
                raise RuntimeError(f"Could not write {path}")
            rendered.append(annotated)
        triptych = np.vstack(rendered)
        triptych_path = output_dir / f"case_{case_number:02d}_timeline.jpg"
        if not cv2.imwrite(str(triptych_path), triptych):
            raise RuntimeError(f"Could not write {triptych_path}")
        event["images"] = {
            "last_match": frame_specs[0][1],
            "previous": frame_specs[1][1],
            "switch": frame_specs[2][1],
            "timeline": triptych_path.name,
        }

    summary_path = output_dir / "id_switch_cases.json"
    summary_path.write_text(json.dumps(switches, indent=2) + "\n", encoding="utf-8")
    print(
        f"Extracted {len(switches)} representative ID-switch cases "
        f"from {total_switches} total switches to {output_dir}"
    )
    for index, event in enumerate(switches, start=1):
        print(
            f"Case {index}: GT {event['gt_id']}, PRED {event['old_pred_id']} -> "
            f"{event['new_pred_id']}, frames {event['last_matched_frame']} -> "
            f"{event['switch_frame']}, gap={event['gap_frames']}"
        )


if __name__ == "__main__":
    main()
