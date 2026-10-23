#!/usr/bin/env python3
"""Render QuadTrack MOT-format ground truth as MP4 videos."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = CODE_DIR.parent / "QuadTrack" / "train"
DEFAULT_OUTPUT_DIR = CODE_DIR / "outputs" / "ground_truth"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create videos with QuadTrack ground-truth person boxes and IDs."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--sequences",
        nargs="*",
        help="Sequence names, e.g. 0015 0016. Defaults to the first sequences found.",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=3,
        help="Number of videos when --sequences is omitted (default: 3).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit per sequence; useful for a quick smoke test.",
    )
    parser.add_argument("--line-thickness", type=int, default=2)
    return parser.parse_args()


def id_color(track_id: int) -> tuple[int, int, int]:
    """Return a repeatable bright BGR color for a track ID."""
    return (
        64 + (37 * track_id) % 192,
        64 + (17 * track_id) % 192,
        64 + (29 * track_id) % 192,
    )


def discover_sequences(root: Path, requested: list[str] | None, limit: int) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    if requested:
        sequences = [root / name for name in requested]
    else:
        sequences = sorted(
            path for path in root.iterdir() if (path / "img1").is_dir() and (path / "gt" / "gt.txt").is_file()
        )[:limit]

    invalid = [str(path) for path in sequences if not (path / "img1").is_dir() or not (path / "gt" / "gt.txt").is_file()]
    if invalid:
        raise FileNotFoundError("Invalid QuadTrack sequence(s): " + ", ".join(invalid))
    if not sequences:
        raise RuntimeError(f"No valid sequences found under {root}")
    return sequences


def image_paths(sequence_dir: Path, max_frames: int | None) -> list[Path]:
    paths = sorted(
        path for path in (sequence_dir / "img1").iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    return paths if max_frames is None else paths[:max_frames]


def load_ground_truth(gt_path: Path) -> dict[int, list[tuple[int, float, float, float, float, float]]]:
    """Load frame,id,x,y,w,h,confidence,... rows, retaining valid person boxes."""
    annotations: dict[int, list[tuple[int, float, float, float, float, float]]] = defaultdict(list)
    with gt_path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) < 6:
                raise ValueError(f"Malformed row {line_number} in {gt_path}: expected at least 6 columns")
            frame_id = int(float(row[0]))
            track_id = int(float(row[1]))
            x, y, width, height = map(float, row[2:6])
            confidence = float(row[6]) if len(row) > 6 else 1.0
            class_id = int(float(row[7])) if len(row) > 7 else 1
            if confidence > 0 and class_id == 1 and width > 0 and height > 0:
                annotations[frame_id].append((track_id, x, y, width, height, confidence))
    return annotations


def open_writer(output_path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {output_path}")
    return writer


def render_sequence(
    sequence_dir: Path,
    output_dir: Path,
    fps: float,
    max_frames: int | None,
    line_thickness: int,
) -> Path:
    frames = image_paths(sequence_dir, max_frames)
    if not frames:
        raise RuntimeError(f"No images found in {sequence_dir / 'img1'}")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Could not read {frames[0]}")
    height, width = first.shape[:2]
    output_path = output_dir / f"{sequence_dir.name}_ground_truth.mp4"
    writer = open_writer(output_path, fps, (width, height))
    annotations = load_ground_truth(sequence_dir / "gt" / "gt.txt")

    try:
        progress = tqdm(frames, desc=f"GT {sequence_dir.name}", unit="frame", dynamic_ncols=True)
        for path in progress:
            frame = cv2.imread(str(path))
            if frame is None:
                progress.write(f"Warning: skipping unreadable image: {path}")
                continue
            frame_id = int(path.stem)
            boxes = annotations.get(frame_id, [])
            for track_id, x, y, box_width, box_height, _ in boxes:
                x1, y1 = max(0, round(x)), max(0, round(y))
                x2, y2 = min(width - 1, round(x + box_width)), min(height - 1, round(y + box_height))
                color = id_color(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)
                label = f"GT ID {track_id}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                )
                label_top = max(0, y1 - text_height - baseline - 5)
                cv2.rectangle(frame, (x1, label_top), (x1 + text_width + 6, y1), color, -1)
                cv2.putText(
                    frame,
                    label,
                    (x1 + 3, max(text_height, y1 - baseline - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                frame,
                f"Sequence {sequence_dir.name} | Frame {frame_id} | People {len(boxes)}",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()
    return output_path


def main() -> None:
    args = parse_args()
    if args.num_videos < 1:
        raise ValueError("--num-videos must be at least 1")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    sequences = discover_sequences(args.dataset_root.resolve(), args.sequences, args.num_videos)
    print("Sequences:", ", ".join(path.name for path in sequences))
    for sequence in sequences:
        output_path = render_sequence(
            sequence,
            args.output_dir.resolve(),
            args.fps,
            args.max_frames,
            args.line_thickness,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

