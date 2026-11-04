"""Lightweight path, dependency and sample-byte validation."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import cv2
import yaml

from mot_baseline_benchmark.protocol import list_frames, resolve_sequence_dir

from .bandwidth import encode_jpeg, periodic_is_hr, scale_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--quadtrack-root", type=Path, required=True)
    parser.add_argument("--dtb-root", type=Path, required=True)
    parser.add_argument("--fair-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = []
    for name in ("cv2", "filterpy", "matplotlib", "numpy", "scipy", "torch", "torchreid", "ultralytics", "yaml"):
        try:
            importlib.import_module(name)
        except Exception as error:
            missing.append(f"{name}: {error}")
    if missing:
        raise RuntimeError(f"Missing dependencies: {missing}")
    config = yaml.safe_load(args.config.resolve().read_text())
    roots = {"quadtrack": args.quadtrack_root.resolve(), "dtb": args.dtb_root.resolve()}
    sample_audit = {}
    for key in ("quadtrack", "dtb"):
        dataset = config[key]
        for sequence in map(str, dataset["sequences"]):
            directory = resolve_sequence_dir(roots[key], sequence)
            if not (directory / "gt/gt.txt").is_file():
                raise FileNotFoundError(directory / "gt/gt.txt")
        sample_dir = resolve_sequence_dir(roots[key], str(dataset["sequences"][0]))
        frame = cv2.imread(str(list_frames(sample_dir, 1)[0]), cv2.IMREAD_COLOR)
        lr = cv2.resize(frame, tuple(map(int, dataset["lr_size"])), interpolation=cv2.INTER_AREA)
        hr_bytes, _ = encode_jpeg(frame, int(config["protocol"]["jpeg_quality"]))
        lr_bytes, _ = encode_jpeg(lr, int(config["protocol"]["jpeg_quality"]))
        if hr_bytes <= lr_bytes:
            raise RuntimeError(f"Sample HR JPEG is not larger than LR JPEG for {key}")
        mapped = scale_boxes(
            [[0, 0, lr.shape[1], lr.shape[0]]],
            (lr.shape[1], lr.shape[0]), (frame.shape[1], frame.shape[0]),
        )[0]
        if tuple(mapped.astype(int)) != (0, 0, frame.shape[1], frame.shape[0]):
            raise RuntimeError("Coordinate mapping preflight failed")
        sample_audit[key] = {
            "hr_shape": list(frame.shape[:2][::-1]), "lr_shape": list(lr.shape[:2][::-1]),
            "hr_jpeg_bytes": hr_bytes, "lr_jpeg_bytes": lr_bytes,
        }
        hive_root = args.fair_root / dataset["label"] / "hive_ecia"
        if not (hive_root / "run_manifest.json").is_file():
            raise FileNotFoundError(hive_root / "run_manifest.json")
    if not periodic_is_hr(0, 5) or periodic_is_hr(1, 5) or not periodic_is_hr(5, 5):
        raise RuntimeError("Periodic schedule preflight failed")
    print(json.dumps({"status": "passed", "sample_bandwidth_audit": sample_audit}, indent=2))
    print("Bandwidth benchmark preflight passed; no detector inference executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

