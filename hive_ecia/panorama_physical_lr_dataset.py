"""Build DTB ECIA events from a tracker that consumed physical low-res frames.

Unlike :mod:`panorama_dataset`, this builder does not reuse the existing
full-resolution YOLO12l + ByteTrack predictions.  It consumes fresh MOT files
produced by ``track_people.py --pre-resize WIDTHxHEIGHT``.  Those MOT boxes have
already been mapped back to the original panorama coordinate system, allowing
evaluation against the manual GT and HR cropping without privileged HR
detections.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import yaml

from evaluate_mot import load_mot
from hive_hr_reid import run_experiment as baseline

from .panorama_dataset import (
    audit_splits,
    build_sequence_events,
    resolve_config_path,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("panorama_config.yaml"),
    )
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument(
        "--prediction-pattern",
        default=None,
    )
    parser.add_argument(
        "--physical-size",
        default="512x120",
        help="Actual detector/tracker input size as WIDTHxHEIGHT",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-events-per-sequence", type=int, default=None)
    return parser.parse_args()


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid physical size {value!r}; expected WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Physical size must be positive, got {value!r}")
    return width, height


def seq_length(path: Path) -> int:
    return int(
        next(
            line.split("=", 1)[1]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("seqLength=")
        )
    )


def build(args: argparse.Namespace) -> Path:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    splits = {key: list(value) for key, value in data["splits"].items()}
    audit_splits(splits)
    manual_root = resolve_config_path(data["manual_root"])
    recorded_splits = json.loads((manual_root / "splits.json").read_text(encoding="utf-8"))
    if recorded_splits != splits:
        raise ValueError(f"Configured split differs from audited manual split: {recorded_splits}")

    prediction_dir = args.prediction_dir.resolve()
    physical_width, physical_height = parse_size(args.physical_size)
    size_slug = f"{physical_width}x{physical_height}"
    prediction_pattern = args.prediction_pattern or (
        f"{{sequence}}_yolo11l_botsort_reid_pre{size_slug}.txt"
    )
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(
            f"Physical-LR protocol output already exists: {output}. "
            "Choose a new run directory; overwrite is intentionally unsupported."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    root = temporary / output.name
    root.mkdir()

    try:
        sequence_split = {
            sequence: split for split, sequences in splits.items() for sequence in sequences
        }
        all_events: dict[str, list[dict]] = {name: [] for name in splits}
        audits = {}
        baseline_metrics = {}
        source_hashes = {}

        for sequence, split in sequence_split.items():
            source_prediction = prediction_dir / prediction_pattern.format(
                sequence=sequence
            )
            if not source_prediction.is_file():
                raise FileNotFoundError(f"Missing physical-LR prediction: {source_prediction}")
            source_dataset = manual_root / "mot_challenge" / split / sequence
            flat = root / "dataset" / sequence
            (flat / "gt").mkdir(parents=True)
            shutil.copy2(source_dataset / "gt" / "gt.txt", flat / "gt" / "gt.txt")
            shutil.copy2(source_dataset / "seqinfo.ini", flat / "seqinfo.ini")
            os.symlink(
                (source_dataset / "img1").resolve(),
                flat / "img1",
                target_is_directory=True,
            )

            # track_people.py stores MOT coordinates after mapping the physical
            # detector boxes back to the original panorama coordinate
            # system.  Do not scale or shift them again here.
            predictions = load_mot(source_prediction, ground_truth=False)
            expected_length = seq_length(source_dataset / "seqinfo.ini")
            outside = sorted(frame for frame in predictions if not 1 <= int(frame) <= expected_length)
            if outside:
                raise ValueError(
                    f"Prediction frames outside 1..{expected_length} for {sequence}: "
                    f"{outside[:10]}"
                )
            baseline.write_mot(root / "predictions" / f"{sequence}.txt", predictions)

            gt = load_mot(flat / "gt" / "gt.txt", ground_truth=True)
            metric_args = argparse.Namespace(
                iou=float(config["event_generation"]["evaluation_iou"])
            )
            baseline_metrics[sequence] = baseline.basic_metrics(
                metric_args,
                [sequence],
                {sequence: predictions},
                {sequence: gt},
                include_hota=True,
            )["aggregate"]
            event_config = dict(config["event_generation"])
            event_config["lowres_width"] = physical_width
            event_config["lowres_height"] = physical_height
            events, event_audit = build_sequence_events(
                sequence=sequence,
                split=split,
                dataset_dir=flat,
                predictions=predictions,
                output_root=root,
                event_config=event_config,
                max_events=args.max_events_per_sequence,
            )
            all_events[split].extend(events)
            audits[sequence] = {
                "split": split,
                "frames": expected_length,
                **event_audit,
            }
            source_hashes[sequence] = baseline.sha256(source_prediction)

        for split, rows in all_events.items():
            rows.sort(key=lambda row: (row["sequence"], row["target_frame"], row["event_id"]))
            write_jsonl(root / f"{split}_data" / "events.jsonl", rows)
            (root / f"{split}_data" / "sequences.txt").write_text(
                "\n".join(splits[split]) + "\n", encoding="utf-8"
            )

        split_config = {
            "dataset_name": f"dtb_panorama_physical_lr_{size_slug}_ecia_events",
            "protocol": (
                f"physical_{size_slug}_detection_tracking; oracle_switch_events; "
                "LR-only proposals; GT labels joined after pool freeze"
            ),
            "split_unit": "complete_sequence",
            "train_sequences": splits["train"],
            "validation_sequences": splits["val"],
            "test_sequences": splits["test"],
            "physical_lr_size": [physical_width, physical_height],
            "prediction_pattern": prediction_pattern,
            "source_prediction_sha256": source_hashes,
            "source_files": {
                "physical_lr_predictions": str(prediction_dir),
                # HIVERankingDataset still reads the historical key name
                # ``lowres_predictions``.  Point it at the normalized copies
                # inside the final (post-atomic-rename) protocol directory.
                "lowres_predictions": str(output / "predictions"),
                "original_frames": str(output / "dataset"),
            },
        }
        (root / "split_config.json").write_text(
            json.dumps(split_config, indent=2) + "\n", encoding="utf-8"
        )
        audit = {
            "passed": True,
            "sequence_disjoint": True,
            "configured_splits": splits,
            "manual_splits_match": True,
            "prediction_frame_offset": 0,
            "physical_lr_detection_tracking": True,
            "physical_lr_size": [physical_width, physical_height],
            "boxes_mapped_to_original_before_evaluation": True,
            "candidate_pool_frozen_before_gt_label": True,
            "sequences": audits,
            "baseline_metrics": baseline_metrics,
        }
        (root / "leakage_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        root.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    shutil.rmtree(temporary, ignore_errors=True)
    print(f"Physical-LR DTB event protocol: {output}")
    print(f"Leakage/protocol audit: {output / 'leakage_audit.json'}")
    return output


def main() -> int:
    build(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
