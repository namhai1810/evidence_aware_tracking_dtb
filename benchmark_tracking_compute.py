#!/usr/bin/env python3
"""Benchmark warmed YOLO tracking compute without disk/video pipeline noise."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO

from track_people import (
    DEFAULT_DATASET_ROOT,
    list_frames,
    parse_frame_size,
    resolve_device,
    synchronize_cuda,
    tracker_config_for,
)


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "bandwidth_downsample" / "compute_benchmark_warmup3.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure YOLO + tracker compute on frames preloaded at their physical input size. "
            "Image I/O, resize, drawing, MOT writing, and video encoding are excluded."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--sequence", default="0015")
    parser.add_argument("--weights", default="yolo11l.pt")
    parser.add_argument(
        "--trackers",
        nargs="+",
        choices=("bytetrack", "botsort_reid"),
        default=["bytetrack", "botsort_reid"],
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=parse_frame_size,
        default=[(1024, 240), (768, 180), (512, 120)],
        metavar="WIDTHxHEIGHT",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def preload_frames(frame_paths: list[Path], size: tuple[int, int]) -> list:
    frames = []
    for path in frame_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read {path}")
        frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    return frames


def run_tracker_frame(
    model: YOLO,
    frame,
    tracker_name: str,
    imgsz: int,
    args: argparse.Namespace,
    device: str,
) -> None:
    model.track(
        frame,
        persist=True,
        tracker=tracker_config_for(tracker_name),
        classes=[0],
        conf=args.conf,
        iou=args.iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )


def reset_trackers(model: YOLO) -> None:
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", ())
    if not trackers:
        raise RuntimeError("Ultralytics did not initialize a tracker during warm-up")
    for tracker in trackers:
        tracker.reset()


def benchmark_once(
    frame_paths: list[Path],
    tracker_name: str,
    size: tuple[int, int],
    args: argparse.Namespace,
    device: str,
) -> tuple[float, float]:
    # Loading and physical resizing deliberately happen before the measured region.
    frames = preload_frames(frame_paths, size)
    model = YOLO(args.weights)
    warmup_count = min(args.warmup_frames, len(frames))
    for frame in frames[:warmup_count]:
        run_tracker_frame(model, frame, tracker_name, size[0], args, device)
    synchronize_cuda(device)
    reset_trackers(model)

    synchronize_cuda(device)
    started = time.perf_counter()
    for frame in frames:
        run_tracker_frame(model, frame, tracker_name, size[0], args, device)
    synchronize_cuda(device)
    elapsed = time.perf_counter() - started

    del model
    del frames
    gc.collect()
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return elapsed, len(frame_paths) / elapsed


def main() -> None:
    args = parse_args()
    if args.repeats <= 0 or args.warmup_frames < 0:
        raise ValueError("--repeats must be positive and --warmup-frames must be non-negative")
    device = resolve_device(args.device)
    frame_paths = list_frames(args.dataset_root.resolve() / args.sequence, args.max_frames)
    configurations = [
        (tracker_name, size)
        for tracker_name in args.trackers
        for size in args.sizes
    ]
    raw_results: dict[str, list[dict[str, float | int]]] = {
        f"{tracker}_{size[0]}x{size[1]}": [] for tracker, size in configurations
    }

    # Rotate configuration order between rounds to reduce temporal/thermal order bias.
    progress = tqdm(total=args.repeats * len(configurations), desc="Compute benchmark", unit="run")
    for repeat_index in range(args.repeats):
        offset = repeat_index % len(configurations)
        ordered = configurations[offset:] + configurations[:offset]
        for tracker_name, size in ordered:
            elapsed, fps = benchmark_once(frame_paths, tracker_name, size, args, device)
            key = f"{tracker_name}_{size[0]}x{size[1]}"
            raw_results[key].append(
                {"repeat": repeat_index + 1, "elapsed_seconds": elapsed, "fps": fps}
            )
            progress.set_postfix(config=key, fps=f"{fps:.2f}")
            progress.update()
    progress.close()

    results = {}
    for key, runs in raw_results.items():
        rates = [float(run["fps"]) for run in runs]
        results[key] = {
            "runs": runs,
            "median_fps": statistics.median(rates),
            "min_fps": min(rates),
            "max_fps": max(rates),
        }

    payload = {
        "protocol": {
            "weights": args.weights,
            "sequence": args.sequence,
            "device": device,
            "measured_frames_per_run": len(frame_paths),
            "warmup_frames_per_run": args.warmup_frames,
            "warmup_included_in_timing": False,
            "tracker_reset_after_warmup": True,
            "cuda_synchronized_before_and_after_timing": True,
            "repeats": args.repeats,
            "summary_statistic": "median_fps",
            "timed_scope": ["YOLO inference and NMS", "tracker update"],
            "excluded_scope": [
                "model loading",
                "image decoding",
                "physical resize",
                "drawing",
                "MOT writing",
                "video encoding",
                "progress rendering",
            ],
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved benchmark: {args.output.resolve()}")
    for key, metrics in results.items():
        print(
            f"{key}: median {metrics['median_fps']:.2f} FPS "
            f"(range {metrics['min_fps']:.2f}-{metrics['max_fps']:.2f})"
        )


if __name__ == "__main__":
    main()
