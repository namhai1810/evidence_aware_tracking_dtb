"""Dependency, path and frozen-checkpoint preflight without model inference."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import yaml

from .protocol import DATASETS, resolve_sequence_dir, sha256


CODE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "detector": ("yolo11l.pt", "9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f"),
    "botsort": ("botsort_reid.yaml", "f9df0ab3ec318b8656ffaf42edf883343db6d4977f722489be57e8f0f9b82d1b"),
    "hive": (
        "hr_finding/runs/hive_history_v2_seed42/checkpoints/best.pt",
        "954b23039cc590f0abdb37ab4edd9c77b3f0d37f3a9c573933065ff540b6d0dd",
    ),
    "ecia": (
        "outputs/hive_ecia_recovery_aware/20261030_osnet_ain_recovery_v1/training/best.pt",
        "283653561d877fe31b79a870b84d2d117f62b10c97cc4240617d7d698955272e",
    ),
    "thresholds": (
        "outputs/hive_ecia_recovery_aware/20261030_osnet_ain_recovery_v1/validation_thresholds.json",
        "9b8006ab8739c43592a5325f77614f7561c6d2da49d446602ceaca4a0d6bd2ba",
    ),
    "reid": (
        "models/reid_benchmark/osnet_ain_x1_0_d_m_c.pth",
        "9e610511b54d2f4fb62aaaec86348b768157332dac3ab2263672f4769ee74069",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadtrack-root", type=Path, required=True)
    parser.add_argument("--dtb-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CODE_ROOT / "online_hive_ecia/config.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing_dependencies = []
    loaded = {}
    for module in ("cv2", "filterpy", "numpy", "scipy", "torch", "torchreid", "tqdm", "ultralytics", "yaml"):
        try:
            loaded[module] = importlib.import_module(module)
        except Exception as error:  # pragma: no cover - host dependent
            missing_dependencies.append(f"{module}: {error}")
    if missing_dependencies:
        raise RuntimeError(f"Missing Python dependencies: {missing_dependencies}")
    roots = {"quadtrack": args.quadtrack_root.resolve(), "dtb": args.dtb_root.resolve()}
    for key, spec in DATASETS.items():
        for sequence in spec.sequences:
            sequence_dir = resolve_sequence_dir(roots[key], sequence)
            if not (sequence_dir / "gt" / "gt.txt").is_file():
                raise FileNotFoundError(sequence_dir / "gt" / "gt.txt")
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    configured = {
        "hive": config["hive"]["checkpoint"],
        "ecia": config["ecia"]["checkpoint"],
        "thresholds": config["ecia"]["thresholds"],
        "reid": config["ecia"]["reid_checkpoint"],
    }
    if "online_ecia_finetune" in json.dumps(configured):
        raise RuntimeError("Online-aware v1/balanced-v2 checkpoint is forbidden")
    audits = {}
    for name, (relative, expected_hash) in EXPECTED.items():
        path = CODE_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Frozen {name} artifact changed: expected {expected_hash}, got {actual_hash}"
            )
        audits[name] = {"path": str(path), "sha256": actual_hash}
    for name, configured_path in configured.items():
        if Path(configured_path).as_posix() != EXPECTED[name][0]:
            raise RuntimeError(
                f"Existing Base-Guided Safety config points to unexpected {name}: {configured_path}"
            )
    if config["ecia"]["min_old_probability"] != 0.80 or config["ecia"]["min_old_margin"] != 0.40:
        raise RuntimeError("Existing Base-Guided Safety fallback thresholds changed")
    print(json.dumps({
        "status": "passed",
        "runtime": {
            "python": sys.executable,
            "ultralytics_version": getattr(loaded["ultralytics"], "__version__", "unknown"),
            "ultralytics_path": getattr(loaded["ultralytics"], "__file__", "unknown"),
            "torch_version": getattr(loaded["torch"], "__version__", "unknown"),
            "opencv_version": getattr(loaded["cv2"], "__version__", "unknown"),
            "torchreid_path": getattr(loaded["torchreid"], "__file__", "unknown"),
        },
        "frozen_artifacts": audits,
    }, indent=2))
    print("Fair MOT benchmark preflight passed; no model inference was executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

