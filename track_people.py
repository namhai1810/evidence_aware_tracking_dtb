#!/usr/bin/env python3
"""Run YOLO person detection with ByteTrack, BoT-SORT+ReID, and/or SORT on QuadTrack."""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(CODE_DIR / ".ultralytics"))

import cv2
import numpy as np
import torch
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from ultralytics import YOLO


DEFAULT_DATASET_ROOT = CODE_DIR.parent / "QuadTrack" / "train"
DEFAULT_OUTPUT_DIR = CODE_DIR / "outputs" / "tracking"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track people in QuadTrack with YOLO + ByteTrack, BoT-SORT+ReID, or SORT."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--sequences", nargs="+", default=["0015"], help="Sequence names to process.")
    parser.add_argument(
        "--tracker",
        choices=("bytetrack", "botsort_reid", "sort", "both"),
        default="both",
        help="Tracker to run; 'both' preserves the original ByteTrack+SORT pair.",
    )
    parser.add_argument("--weights", default="yolo11n.pt", help="Ultralytics weights path or model name.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional suffix for outputs, e.g. imgsz1280, so controlled runs do not overwrite each other.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 1, ...")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--pre-resize",
        type=parse_frame_size,
        default=None,
        metavar="WIDTHxHEIGHT",
        help=(
            "Physically resize each decoded frame before inference to simulate a lower-bandwidth "
            "source. MOT boxes are scaled back to the original frame coordinates."
        ),
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=3,
        help=(
            "Run this many samples before timing, then reset tracker state. CUDA is synchronized "
            "before and after the measured loop (default: 3)."
        ),
    )
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--sort-max-age", type=int, default=30)
    parser.add_argument("--sort-min-hits", type=int, default=3)
    parser.add_argument("--sort-iou-threshold", type=float, default=0.3)
    return parser.parse_args()


def parse_frame_size(value: str) -> tuple[int, int]:
    """Parse a WIDTHxHEIGHT frame size for physical pre-inference resizing."""
    match = re.fullmatch(r"(\d+)[xX](\d+)", value.strip())
    if match is None:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, for example 1024x240")
    width, height = (int(part) for part in match.groups())
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("WIDTH and HEIGHT must be positive")
    return width, height


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def list_frames(sequence_dir: Path, max_frames: int | None) -> list[Path]:
    image_dir = sequence_dir / "img1"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    frames = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise RuntimeError(f"No frames found in {image_dir}")
    return frames


def id_color(track_id: int) -> tuple[int, int, int]:
    return (
        64 + (37 * track_id) % 192,
        64 + (17 * track_id) % 192,
        64 + (29 * track_id) % 192,
    )


def draw_tracks(
    frame: np.ndarray,
    tracks: np.ndarray,
    tracker_name: str,
    frame_id: int,
    line_thickness: int,
) -> np.ndarray:
    for row in tracks:
        x1, y1, x2, y2, track_id, score = row
        track_id = int(track_id)
        color = id_color(track_id)
        point1, point2 = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(frame, point1, point2, color, line_thickness)
        label = f"ID {track_id} {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_top = max(0, point1[1] - text_height - baseline - 5)
        cv2.rectangle(frame, (point1[0], label_top), (point1[0] + text_width + 6, point1[1]), color, -1)
        cv2.putText(
            frame,
            label,
            (point1[0] + 3, max(text_height, point1[1] - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        f"YOLO + {tracker_name.upper()} | Frame {frame_id} | Tracks {len(tracks)}",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    center_x = bbox[0] + width / 2.0
    center_y = bbox[1] + height / 2.0
    scale = width * height
    ratio = width / max(height, 1e-6)
    return np.array([center_x, center_y, scale, ratio]).reshape((4, 1))


def convert_x_to_bbox(state: np.ndarray) -> np.ndarray:
    width = np.sqrt(max(0.0, state[2, 0] * state[3, 0]))
    height = state[2, 0] / max(width, 1e-6)
    return np.array(
        [state[0, 0] - width / 2.0, state[1, 0] - height / 2.0,
         state[0, 0] + width / 2.0, state[1, 0] + height / 2.0]
    )


class KalmanBoxTracker:
    """Internal Kalman-filter track used by the original SORT formulation."""

    next_id = 1

    def __init__(self, bbox: np.ndarray, score: float):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1]], dtype=float
        )
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]], dtype=float
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.next_id
        KalmanBoxTracker.next_id += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.score = float(score)

    def update(self, bbox: np.ndarray, score: float) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = float(score)
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self) -> np.ndarray:
        if self.kf.x[6, 0] + self.kf.x[2, 0] <= 0:
            self.kf.x[6, 0] = 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x)

    def state(self) -> np.ndarray:
        return convert_x_to_bbox(self.kf.x)


def iou_batch(detections: np.ndarray, trackers: np.ndarray) -> np.ndarray:
    if len(detections) == 0 or len(trackers) == 0:
        return np.empty((len(detections), len(trackers)), dtype=float)
    det = np.expand_dims(detections, 1)
    trk = np.expand_dims(trackers, 0)
    xx1 = np.maximum(det[..., 0], trk[..., 0])
    yy1 = np.maximum(det[..., 1], trk[..., 1])
    xx2 = np.minimum(det[..., 2], trk[..., 2])
    yy2 = np.minimum(det[..., 3], trk[..., 3])
    intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    det_area = (det[..., 2] - det[..., 0]) * (det[..., 3] - det[..., 1])
    trk_area = (trk[..., 2] - trk[..., 0]) * (trk[..., 3] - trk[..., 1])
    return intersection / np.maximum(det_area + trk_area - intersection, 1e-6)


def associate_detections_to_trackers(
    detections: np.ndarray, trackers: np.ndarray, iou_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections), dtype=int),
            np.empty((0,), dtype=int),
        )
    iou_matrix = iou_batch(detections, trackers)
    det_indices, trk_indices = linear_sum_assignment(-iou_matrix)
    matched = []
    unmatched_detections = set(range(len(detections)))
    unmatched_trackers = set(range(len(trackers)))
    for det_index, trk_index in zip(det_indices, trk_indices):
        if iou_matrix[det_index, trk_index] < iou_threshold:
            continue
        matched.append((det_index, trk_index))
        unmatched_detections.discard(int(det_index))
        unmatched_trackers.discard(int(trk_index))
    return (
        np.asarray(matched, dtype=int).reshape(-1, 2),
        np.asarray(sorted(unmatched_detections), dtype=int),
        np.asarray(sorted(unmatched_trackers), dtype=int),
    )


class Sort:
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        KalmanBoxTracker.next_id = 1

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Accept Nx5 [x1,y1,x2,y2,score], return Nx6 [...,id,score]."""
        self.frame_count += 1
        predicted = []
        valid_trackers = []
        for tracker in self.trackers:
            position = tracker.predict()
            if np.all(np.isfinite(position)):
                predicted.append(position)
                valid_trackers.append(tracker)
        self.trackers = valid_trackers
        predicted_array = np.asarray(predicted, dtype=float).reshape(-1, 4)

        matches, unmatched_detections, _ = associate_detections_to_trackers(
            detections[:, :4], predicted_array, self.iou_threshold
        )
        for det_index, tracker_index in matches:
            self.trackers[tracker_index].update(detections[det_index, :4], detections[det_index, 4])
        for det_index in unmatched_detections:
            self.trackers.append(KalmanBoxTracker(detections[det_index, :4], detections[det_index, 4]))

        results = []
        alive = []
        for tracker in self.trackers:
            if tracker.time_since_update <= self.max_age:
                alive.append(tracker)
            if tracker.time_since_update == 0 and (
                tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                results.append(np.r_[tracker.state(), tracker.id, tracker.score])
        self.trackers = alive
        return np.asarray(results, dtype=float).reshape(-1, 6)


def open_video_writer(output_path: Path, fps: float, frame: np.ndarray) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {output_path}")
    return writer


def boxes_to_numpy(result) -> np.ndarray:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 5), dtype=float)
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidence = boxes.conf.detach().cpu().numpy().reshape(-1, 1)
    return np.concatenate((xyxy, confidence), axis=1)


def write_mot_rows(handle, frame_id: int, tracks: np.ndarray) -> None:
    for x1, y1, x2, y2, track_id, score in tracks:
        width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        handle.write(
            f"{frame_id},{int(track_id)},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},"
            f"{score:.5f},1,1.0\n"
        )


def tracks_in_original_coordinates(
    tracks: np.ndarray,
    original_width: int,
    original_height: int,
    inference_width: int,
    inference_height: int,
) -> np.ndarray:
    """Scale tracker boxes back to the decoded source-frame coordinate system."""
    if len(tracks) == 0 or (original_width, original_height) == (inference_width, inference_height):
        return tracks
    scaled = tracks.copy()
    scale_x = original_width / inference_width
    scale_y = original_height / inference_height
    scaled[:, [0, 2]] *= scale_x
    scaled[:, [1, 3]] *= scale_y
    return scaled


def tracker_config_for(tracker_name: str) -> str:
    return (
        "bytetrack.yaml"
        if tracker_name == "bytetrack"
        else str(CODE_DIR / "botsort_reid.yaml")
    )


def synchronize_cuda(device: str) -> None:
    """Wait for queued GPU work so wall-clock timing includes all CUDA kernels."""
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def warm_up_model(
    model: YOLO,
    frames: list[Path],
    tracker_name: str,
    args: argparse.Namespace,
    device: str,
) -> int:
    """Warm up the full inference path, then clear any tracker state it created."""
    warmup_count = min(args.warmup_frames, len(frames))
    for path in frames[:warmup_count]:
        source_frame = cv2.imread(str(path))
        if source_frame is None:
            raise RuntimeError(f"Could not read warm-up image: {path}")
        frame = (
            cv2.resize(source_frame, args.pre_resize, interpolation=cv2.INTER_AREA)
            if args.pre_resize is not None
            else source_frame
        )
        if tracker_name in {"bytetrack", "botsort_reid"}:
            model.track(
                frame,
                persist=True,
                tracker=tracker_config_for(tracker_name),
                classes=[0],
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )
        else:
            model.predict(
                frame,
                classes=[0],
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )

    synchronize_cuda(device)
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", ()):
        tracker.reset()
    return warmup_count


def process_sequence(
    sequence_dir: Path,
    tracker_name: str,
    args: argparse.Namespace,
    device: str,
) -> tuple[Path, Path, float]:
    frames = list_frames(sequence_dir, args.max_frames)
    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read {frames[0]}")
    if args.pre_resize is not None:
        first_frame = cv2.resize(first_frame, args.pre_resize, interpolation=cv2.INTER_AREA)

    output_dir = args.output_dir.resolve()
    model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(args.weights).stem)
    run_suffix = ""
    if args.run_name:
        safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_name)
        run_suffix = f"_{safe_run_name}"
    output_stem = f"{sequence_dir.name}_{model_name}_{tracker_name}{run_suffix}"
    video_path = output_dir / f"{output_stem}.mp4"
    mot_path = output_dir / f"{output_stem}.txt"
    writer = open_video_writer(video_path, args.fps, first_frame)
    mot_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    sort_tracker = (
        Sort(args.sort_max_age, args.sort_min_hits, args.sort_iou_threshold)
        if tracker_name == "sort"
        else None
    )

    warmup_count = warm_up_model(model, frames, tracker_name, args, device)
    print(f"Warm-up complete: {warmup_count} frame(s), excluded from timing; tracker state reset")

    elapsed = 0.0
    try:
        with mot_path.open("w", encoding="utf-8") as mot_file:
            progress = tqdm(
                frames,
                desc=f"{tracker_name.upper()} {sequence_dir.name}",
                unit="frame",
                dynamic_ncols=True,
            )
            synchronize_cuda(device)
            started = time.perf_counter()
            for path in progress:
                source_frame = cv2.imread(str(path))
                if source_frame is None:
                    progress.write(f"Warning: skipping unreadable image: {path}")
                    continue
                original_height, original_width = source_frame.shape[:2]
                frame = (
                    cv2.resize(source_frame, args.pre_resize, interpolation=cv2.INTER_AREA)
                    if args.pre_resize is not None
                    else source_frame
                )
                inference_height, inference_width = frame.shape[:2]
                frame_id = int(path.stem)
                if tracker_name in {"bytetrack", "botsort_reid"}:
                    result = model.track(
                        frame,
                        persist=True,
                        tracker=tracker_config_for(tracker_name),
                        classes=[0],
                        conf=args.conf,
                        iou=args.iou,
                        imgsz=args.imgsz,
                        device=device,
                        verbose=False,
                    )[0]
                    boxes = result.boxes
                    if boxes is None or len(boxes) == 0 or boxes.id is None:
                        tracks = np.empty((0, 6), dtype=float)
                    else:
                        xyxy = boxes.xyxy.detach().cpu().numpy()
                        ids = boxes.id.detach().cpu().numpy().reshape(-1, 1)
                        scores = boxes.conf.detach().cpu().numpy().reshape(-1, 1)
                        tracks = np.concatenate((xyxy, ids, scores), axis=1)
                else:
                    result = model.predict(
                        frame,
                        classes=[0],
                        conf=args.conf,
                        iou=args.iou,
                        imgsz=args.imgsz,
                        device=device,
                        verbose=False,
                    )[0]
                    assert sort_tracker is not None
                    tracks = sort_tracker.update(boxes_to_numpy(result))

                mot_tracks = tracks_in_original_coordinates(
                    tracks,
                    original_width,
                    original_height,
                    inference_width,
                    inference_height,
                )
                write_mot_rows(mot_file, frame_id, mot_tracks)
                annotated = draw_tracks(frame, tracks, tracker_name, frame_id, args.line_thickness)
                writer.write(annotated)
                progress.set_postfix(tracks=len(tracks), device=device)
            synchronize_cuda(device)
            elapsed = time.perf_counter() - started
    finally:
        writer.release()

    return video_path, mot_path, elapsed


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.imgsz <= 0:
        raise ValueError("--fps and --imgsz must be positive")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be non-negative")
    device = resolve_device(args.device)
    tracker_names = ["bytetrack", "sort"] if args.tracker == "both" else [args.tracker]
    resize_label = (
        f"{args.pre_resize[0]}x{args.pre_resize[1]}" if args.pre_resize is not None else "original"
    )
    print(
        f"Device: {device} | Detector: {args.weights} | Pre-resize: {resize_label} | "
        "Person class only (COCO class 0)"
    )

    for sequence_name in args.sequences:
        sequence_dir = args.dataset_root.resolve() / sequence_name
        for tracker_name in tracker_names:
            video_path, mot_path, elapsed = process_sequence(sequence_dir, tracker_name, args, device)
            frame_count = len(list_frames(sequence_dir, args.max_frames))
            speed = frame_count / elapsed if elapsed else 0.0
            print(
                f"Saved {tracker_name}: {video_path}\n"
                f"MOT results: {mot_path}\n"
                f"Elapsed: {elapsed:.1f}s ({speed:.2f} frames/s)"
            )


if __name__ == "__main__":
    main()

