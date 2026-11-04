"""Read-only dependency and path checks for the two online-aware workflows."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import yaml


CODE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", choices=("train", "evaluate"), required=True)
    parser.add_argument("--quadtrack-root", type=Path, required=True)
    parser.add_argument("--dtb-root", type=Path)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--allow-missing-training-artifacts", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=CODE_ROOT / "online_ecia_finetune/config.yaml",
    )
    return parser.parse_args()


def sequence_exists(root: Path, sequence: str) -> bool:
    return any(
        (root / prefix / sequence / "img1").is_dir()
        for prefix in (Path("."), Path("train"), Path("val"), Path("test"))
    )


def main() -> int:
    args = parse_args()
    modules = ("numpy", "torch", "cv2", "yaml", "ultralytics", "tqdm")
    missing_modules = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing_modules:
        raise RuntimeError(f"Missing Python dependencies: {missing_modules}")
    # Use the same resolver as actual OSNet-AIN inference. It first checks the
    # active conda environment, then the project-local pure-Python Torchreid
    # package under code/.venv without replacing the active Torch/CUDA stack.
    from hive_ecia.reid_encoders import import_torchreid_build_model

    import_torchreid_build_model()
    config = yaml.safe_load(args.config.read_text())
    sequences = sum(
        (list(map(str, config["data"][key])) for key in (
            "train_sequences", "validation_sequences", "calibration_sequences", "test_sequences"
        )), []
    )
    missing_sequences = [seq for seq in sequences if not sequence_exists(args.quadtrack_root, seq)]
    required_files = (
        CODE_ROOT / "yolo11l.pt",
        CODE_ROOT / "botsort_reid.yaml",
        CODE_ROOT / "online_hive_ecia/config.yaml",
        CODE_ROOT / config["fine_tuning"]["initial_checkpoint"],
        CODE_ROOT / "hr_finding/runs/hive_history_v2_seed42/checkpoints/best.pt",
        CODE_ROOT / "models/reid_benchmark/osnet_ain_x1_0_d_m_c.pth",
    )
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_sequences or missing_files:
        raise FileNotFoundError(
            f"Missing sequences={missing_sequences}; missing files={missing_files}"
        )
    if args.workflow == "evaluate":
        if args.dtb_root is None or args.training_root is None:
            raise ValueError("evaluate preflight requires --dtb-root and --training-root")
        dtb_sequences = ("normal_walking", "crossing", "partial_occ", "seam", "full_occ")
        missing_dtb = [seq for seq in dtb_sequences if not sequence_exists(args.dtb_root, seq)]
        artifacts = (
            args.training_root / "training/best.pt",
            args.training_root / "calibration/thresholds.json",
            args.training_root / "inference_config.yaml",
            args.training_root / "leakage_audit.json",
        )
        missing_artifacts = [str(path) for path in artifacts if not path.is_file()]
        if missing_dtb or (missing_artifacts and not args.allow_missing_training_artifacts):
            raise FileNotFoundError(
                f"Missing DTB sequences={missing_dtb}; training artifacts={missing_artifacts}"
            )
        if missing_artifacts:
            print(
                "Evaluation training artifacts are not present yet; they will be "
                "required after the selected training script completes."
            )
    print(f"{args.workflow} dependency and path preflight passed; no inference/training executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
