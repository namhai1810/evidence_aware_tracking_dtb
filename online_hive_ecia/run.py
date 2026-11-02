"""Run causal BoT-SORT with optional Online HIVE--ECIA identity recovery."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(CODE_ROOT / ".ultralytics"))

import yaml


DEFAULT_DATASET = (
    CODE_ROOT.parent
    / "cvat_packages_trim300_last100"
    / "object_tracking_prepared_manual_dataset"
    / "mot_challenge"
)


def parse_frame_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)[xX](\d+)", value.strip())
    if match is None:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, for example 512x120")
    width, height = (int(part) for part in match.groups())
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("WIDTH and HEIGHT must be positive")
    return width, height


def resolve_config_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (CODE_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dataset-label", default="dataset")
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "baseline",
            "online",
            "hr_disabled",
            "base_guided_hr_disabled",
            "base_guided_online",
            "base_guided_safety_online",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=CODE_ROOT / "yolo11l.pt")
    parser.add_argument("--tracker-config", type=Path, default=CODE_ROOT / "botsort_reid.yaml")
    parser.add_argument("--pre-resize", type=parse_frame_size, default=(512, 120))
    parser.add_argument(
        "--allow-resolution-mismatch",
        action="store_true",
        help=(
            "Allow a physical LR size different from the frozen HIVE reference size; "
            "the manifest records the run as zero-shot resolution transfer."
        ),
    )
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse sequences that already have complete MOT, diagnostic and timing "
            "artifacts; overwrite only incomplete per-sequence artifacts."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate paths and CLI contract without loading models or processing frames.",
    )
    return parser.parse_args()


def resolve_sequence_dir(root: Path, sequence: str) -> Path:
    candidates = [root / sequence, *(root / split / sequence for split in ("train", "val", "test"))]
    matches = [path for path in candidates if (path / "img1").is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one directory for sequence {sequence!r} below {root}; "
            f"found {[str(path) for path in matches]}"
        )
    return matches[0]


def list_frames(sequence_dir: Path, max_frames: int | None) -> list[Path]:
    frames = sorted(
        path
        for path in (sequence_dir / "img1").iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    frames = frames[:max_frames] if max_frames is not None else frames
    if not frames:
        raise RuntimeError(f"No image frames in {sequence_dir / 'img1'}")
    return frames


def validate(args: argparse.Namespace, config: dict) -> dict[str, Path]:
    if args.imgsz <= 0 or args.fps <= 0 or args.warmup_frames < 0:
        raise ValueError("imgsz/fps must be positive and warmup-frames non-negative")
    if not 0.0 < args.conf <= 1.0 or not 0.0 < args.iou <= 1.0:
        raise ValueError("conf and iou must be in (0,1]")
    calibration = config.get("base_guided_trigger", {}).get("calibration", {})
    calibration_split = str(calibration.get("split", "")).lower()
    if args.mode.startswith("base_guided") and (
        "validation" not in calibration_split
        or "test" in calibration_split
        or "dtb" in calibration_split
    ):
        raise ValueError(
            "Base-guided thresholds must declare a validation-only calibration split"
        )
    expected_size = tuple(map(int, config["input"]["physical_lr_size"]))
    if tuple(args.pre_resize) != expected_size and not args.allow_resolution_mismatch:
        raise ValueError(
            f"Frozen HIVE protocol requires physical LR {expected_size[0]}x{expected_size[1]}, "
            f"got {args.pre_resize[0]}x{args.pre_resize[1]}; pass "
            "--allow-resolution-mismatch for an explicitly logged zero-shot run"
        )
    required = {
        "weights": args.weights.resolve(),
        "tracker_config": args.tracker_config.resolve(),
    }
    if args.mode != "baseline":
        required.update(
            {
                "hive_checkpoint": resolve_config_path(config["hive"]["checkpoint"]),
                "ecia_checkpoint": resolve_config_path(config["ecia"]["checkpoint"]),
                "ecia_thresholds": resolve_config_path(config["ecia"]["thresholds"]),
                "reid_checkpoint": resolve_config_path(config["ecia"]["reid_checkpoint"]),
            }
        )
    missing = {name: path for name, path in required.items() if not path.is_file()}
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")
    root = args.dataset_root.resolve()
    for sequence in args.sequences:
        resolve_sequence_dir(root, sequence)
    if args.output_dir.exists() and not args.validate_only and not args.resume:
        raise FileExistsError(
            f"Output already exists: {args.output_dir}; use a new run directory"
        )
    return required


def reset_tracker(model) -> None:
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", ())
    for tracker in trackers:
        tracker.reset()


def track_frame(model, frame: np.ndarray, args: argparse.Namespace, device: str):
    return model.track(
        frame,
        persist=True,
        tracker=str(args.tracker_config.resolve()),
        classes=[0],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        verbose=False,
    )[0]


def result_tracks(result) -> np.ndarray:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0 or boxes.id is None:
        return np.empty((0, 6), dtype=float)
    return np.concatenate(
        (
            boxes.xyxy.detach().cpu().numpy(),
            boxes.id.detach().cpu().numpy().reshape(-1, 1),
            boxes.conf.detach().cpu().numpy().reshape(-1, 1),
        ),
        axis=1,
    )


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else 0.0


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def process_sequence(
    *,
    model,
    bundle: OnlineModelBundle | None,
    sequence: str,
    sequence_dir: Path,
    args: argparse.Namespace,
    config: dict,
    device: str,
) -> dict:
    frames = list_frames(sequence_dir, args.max_frames)
    for path in frames[: min(args.warmup_frames, len(frames))]:
        source = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(path)
        track_frame(model, cv2.resize(source, args.pre_resize, interpolation=cv2.INTER_AREA), args, device)
    synchronize_cuda(device)
    reset_tracker(model)

    engine = OnlineRecoveryEngine(config, mode=args.mode, model_bundle=bundle)
    canonical_path = args.output_dir / "mot" / f"{sequence}.txt"
    raw_path = args.output_dir / "raw_mot" / f"{sequence}.txt"
    diagnostic_path = args.output_dir / "diagnostics" / f"{sequence}.jsonl"
    timing_path = args.output_dir / "timing" / f"{sequence}.json"
    for path in (canonical_path, raw_path, diagnostic_path, timing_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    tracking_ms: list[float] = []
    total_ms: list[float] = []
    diagnostic_events = 0
    flush_every = max(1, int(config.get("logging", {}).get("flush_every_events", 1)))
    pipeline_started = time.perf_counter()
    with canonical_path.open("w", encoding="utf-8") as canonical_file, raw_path.open(
        "w", encoding="utf-8"
    ) as raw_file, diagnostic_path.open("w", encoding="utf-8") as diagnostic_file:
        progress = tqdm(frames, desc=f"{args.mode} {sequence}", unit="frame", dynamic_ncols=True)
        for source_path in progress:
            source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if source is None:
                raise FileNotFoundError(source_path)
            lr = cv2.resize(source, args.pre_resize, interpolation=cv2.INTER_AREA)
            frame_id = int(source_path.stem)
            synchronize_cuda(device)
            frame_started = time.perf_counter()
            result = track_frame(model, lr, args, device)
            synchronize_cuda(device)
            tracked_at = time.perf_counter()
            raw_tracks = result_tracks(result)
            active, lost = inspect_botsort(model) if args.mode != "baseline" else ({}, {})
            canonical_tracks, diagnostics = engine.process_frame(
                frame_id=frame_id,
                source_path=source_path,
                lr_frame=lr,
                hr_frame=source,
                raw_tracks=raw_tracks,
                active_signals=active,
                lost_signals=lost,
            )
            synchronize_cuda(device)
            finished = time.perf_counter()
            tracking_ms.append((tracked_at - frame_started) * 1000.0)
            total_ms.append((finished - frame_started) * 1000.0)

            original_h, original_w = source.shape[:2]
            raw_mot = tracks_in_original_coordinates(
                raw_tracks, original_w, original_h, lr.shape[1], lr.shape[0]
            )
            canonical_mot = tracks_in_original_coordinates(
                canonical_tracks, original_w, original_h, lr.shape[1], lr.shape[0]
            )
            write_mot_rows(raw_file, frame_id, raw_mot)
            write_mot_rows(canonical_file, frame_id, canonical_mot)
            for row in diagnostics:
                diagnostic_file.write(json.dumps(row, default=json_default) + "\n")
                if row.get("type") in {"recovery_event", "base_guided_event"}:
                    diagnostic_events += 1
            if diagnostics and diagnostic_events % flush_every == 0:
                diagnostic_file.flush()
            progress.set_postfix(tracks=len(raw_tracks), triggers=engine.counters["triggers"])
    pipeline_wall_seconds = time.perf_counter() - pipeline_started
    compute_seconds = sum(total_ms) / 1000.0
    engine_summary = engine.summary()
    timing = {
        "sequence": sequence,
        "mode": args.mode,
        "frames": len(frames),
        "wall_seconds": compute_seconds,
        "pipeline_wall_seconds": pipeline_wall_seconds,
        "fps": len(frames) / compute_seconds if compute_seconds else 0.0,
        "latency_ms": {
            "tracking_mean": statistics.fmean(tracking_ms) if tracking_ms else 0.0,
            "total_mean": statistics.fmean(total_ms) if total_ms else 0.0,
            "total_p50": percentile(total_ms, 50),
            "total_p95": percentile(total_ms, 95),
        },
        "engine": engine_summary,
        "timed_scope": "detector+BoT-SORT+online controller; excludes decode, resize and file writes",
    }
    timing_path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    return timing


def completed_sequence_timing(output_dir: Path, sequence: str) -> dict | None:
    """Return timing only when every artifact for a sequence is complete."""

    required = (
        output_dir / "mot" / f"{sequence}.txt",
        output_dir / "raw_mot" / f"{sequence}.txt",
        output_dir / "diagnostics" / f"{sequence}.jsonl",
        output_dir / "timing" / f"{sequence}.json",
    )
    if not all(path.is_file() for path in required):
        return None
    return json.loads(required[-1].read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    required = validate(args, config)
    if args.validate_only:
        print("CLI and paths valid; no model was loaded and no frame was processed.")
        print(json.dumps({key: str(value) for key, value in required.items()}, indent=2))
        return 0

    # Heavy CV/GPU dependencies are intentionally imported only after the
    # validation-only exit, so CLI/path checks cannot initialize a model.
    global cv2, np, torch, tqdm
    global resolve_device, synchronize_cuda, tracks_in_original_coordinates, write_mot_rows
    global OnlineRecoveryEngine, OnlineModelBundle, inspect_botsort
    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm
    from ultralytics import YOLO
    from track_people import (
        resolve_device,
        synchronize_cuda,
        tracks_in_original_coordinates,
        write_mot_rows,
    )
    from .engine import OnlineRecoveryEngine
    from .models import OnlineModelBundle
    from .tracker_adapter import inspect_botsort

    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    device = resolve_device(args.device)
    torch_device = torch.device(f"cuda:{device}" if device.isdigit() else device)
    bundle = OnlineModelBundle(config, torch_device) if args.mode != "baseline" else None
    model = YOLO(str(args.weights.resolve()))
    results = {}
    for sequence in args.sequences:
        completed = (
            completed_sequence_timing(args.output_dir, sequence)
            if args.resume
            else None
        )
        if completed is not None:
            print(f"Resume: reuse completed {args.mode}/{sequence}")
            results[sequence] = completed
            continue
        results[sequence] = process_sequence(
            model=model,
            bundle=bundle,
            sequence=sequence,
            sequence_dir=resolve_sequence_dir(args.dataset_root.resolve(), sequence),
            args=args,
            config=config,
            device=device,
        )
        reset_tracker(model)
    manifest = {
        "protocol": "causal_online_no_gt_no_future",
        "mode": args.mode,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_label": args.dataset_label,
        "sequences": args.sequences,
        "detector": str(args.weights.resolve()),
        "tracker_config": str(args.tracker_config.resolve()),
        "physical_lr_size": list(args.pre_resize),
        "hive_reference_lr_size": list(map(int, config["input"]["physical_lr_size"])),
        "resolution_zero_shot": tuple(args.pre_resize)
        != tuple(map(int, config["input"]["physical_lr_size"])),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": device,
        "frozen_models": bundle.checkpoint_audit if bundle is not None else None,
        "trigger_configuration": {
            "implementation": (
                "ecia_base_guided"
                if args.mode.startswith("base_guided")
                else ("rule_based" if args.mode != "baseline" else "none")
            ),
            "base_guided": (
                config.get("base_guided_trigger")
                if args.mode.startswith("base_guided")
                else None
            ),
        },
        "sequence_timing": results,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Completed {args.mode}: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

