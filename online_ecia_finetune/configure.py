"""Materialize isolated training and inference configs for one online ECIA run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


CODE_ROOT = Path(__file__).resolve().parents[1]


def absolute(value: str | Path) -> str:
    path = Path(value)
    return str(path.resolve() if path.is_absolute() else (CODE_ROOT / path).resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("training")
    training.add_argument("--template", type=Path, required=True)
    training.add_argument("--run-root", type=Path, required=True)
    training.add_argument("--device", required=True)
    training.add_argument("--output", type=Path, required=True)
    inference = subparsers.add_parser("inference")
    inference.add_argument("--base-config", type=Path, required=True)
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--thresholds", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "training":
        config = yaml.safe_load(args.template.read_text(encoding="utf-8"))
        run_root = args.run_root.resolve()
        config["data"].update(
            train_cache=str(run_root / "cache" / "train.npz"),
            validation_cache=str(run_root / "cache" / "validation.npz"),
            calibration_cache=str(run_root / "cache" / "calibration.npz"),
        )
        config["fine_tuning"]["initial_checkpoint"] = absolute(
            config["fine_tuning"]["initial_checkpoint"]
        )
        config["training"]["device"] = args.device
    else:
        config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
        thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
        config["ecia"]["checkpoint"] = str(args.checkpoint.resolve())
        config["ecia"]["thresholds"] = str(args.thresholds.resolve())
        config["base_guided_trigger"]["policy"] = thresholds["base_guided_policy"]
        config["base_guided_trigger"]["calibration"] = {
            "split": thresholds["selection_split"],
            "sequences": thresholds["calibration_sequences"],
            "status": "selected_and_frozen",
            "source": str(args.thresholds.resolve()),
        }
        config["experiment"]["name"] = (
            "online_aware_recovery_ecia_balanced_v2_inference"
            if thresholds["selection_split"] == "quadtrack_video_disjoint_calibration"
            else "online_aware_recovery_ecia_inference"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Config: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
