#!/usr/bin/env python3
"""Build a leakage-safe, video-grouped dataset for HR-candidate ranking.

The package contains materialized low-resolution query/target crops, high-resolution
candidate crops, event-grouped JSONL manifests, flat CSV tables, split statistics,
and provenance metadata. Splitting is performed only at the sequence/video level.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from evaluate_mot import load_mot


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
OUTPUTS = CODE_DIR / "outputs"
DEFAULT_EVENTS = OUTPUTS / "highres_retrieval_reid_embedding" / "switch_events_top5.json"
DEFAULT_SUMMARY = OUTPUTS / "highres_retrieval_reid_embedding" / "summary.json"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "QuadTrack" / "train"
DEFAULT_LOWRES_DIR = OUTPUTS / "all_sequences_reid_512x120" / "predictions"
DEFAULT_OUTPUT = OUTPUTS / "reid_ranking_dataset_v1"
DEFAULT_TEST_SEQUENCES = ("0016", "0017", "0025", "0031")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-json", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--retrieval-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--lowres-dir", type=Path, default=DEFAULT_LOWRES_DIR)
    parser.add_argument(
        "--lowres-pattern",
        default="{sequence}_yolo11l_botsort_reid_pre512x120.txt",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-sequences", nargs="+", default=list(DEFAULT_TEST_SEQUENCES))
    parser.add_argument("--lowres-width", type=int, default=512)
    parser.add_argument("--lowres-height", type=int, default=120)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write manifests without materializing LR/HR crop images.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict | list:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_project(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


@lru_cache(maxsize=32)
def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return image


def clamp_box(box: list[float] | np.ndarray, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = (float(value) for value in box)
    left = max(0, min(width - 1, int(np.floor(x1))))
    top = max(0, min(height - 1, int(np.floor(y1))))
    right = max(left + 1, min(width, int(np.ceil(x2))))
    bottom = max(top + 1, min(height, int(np.ceil(y2))))
    return [left, top, right, bottom]


def save_crop(image: np.ndarray, box: list[float] | np.ndarray, path: Path, quality: int) -> list[int]:
    height, width = image.shape[:2]
    clipped = clamp_box(box, width, height)
    x1, y1, x2, y2 = clipped
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Empty crop for {path}: box={box}, clipped={clipped}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(f"Failed to write crop: {path}")
    return clipped


def find_track_row(frames: dict, frame_id: int, track_id: int) -> tuple:
    matches = [row for row in frames.get(frame_id, []) if int(row[0]) == track_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one row for frame={frame_id}, track={track_id}; found {len(matches)}"
        )
    return matches[0]


def rounded_box(box: list[float] | np.ndarray) -> list[float]:
    return [round(float(value), 2) for value in box]


def json_line(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def split_statistics(events: list[dict], sequences: list[str], crop_counts: dict) -> dict:
    candidates = [candidate for event in events for candidate in event["candidates"]]
    positives = sum(candidate["label"] for candidate in candidates)
    candidate_events = sum(bool(event["candidates"]) for event in events)
    top5_positive = sum(event["has_positive_candidate"] for event in events)
    return {
        "sequences": sequences,
        "sequence_count": len(sequences),
        "event_count": len(events),
        "candidate_event_count": candidate_events,
        "no_candidate_event_count": len(events) - candidate_events,
        "correct_identity_available_count": sum(
            event["correct_identity_available_in_window"] for event in events
        ),
        "top1_correct_count": sum(event["top1_correct"] for event in events),
        "top1_accuracy_all_events": (
            sum(event["top1_correct"] for event in events) / len(events) if events else 0.0
        ),
        "positive_in_top5_event_count": top5_positive,
        "positive_in_top5_rate_all_events": top5_positive / len(events) if events else 0.0,
        "candidate_pair_count": len(candidates),
        "positive_pair_count": positives,
        "negative_pair_count": len(candidates) - positives,
        "positive_pair_rate": positives / len(candidates) if candidates else 0.0,
        "materialized_crop_counts": dict(crop_counts),
    }


def write_split_files(root: Path, name: str, events: list[dict], sequences: list[str]) -> None:
    split_dir = root / f"{name}_data"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "sequences.txt").write_text("\n".join(sequences) + "\n", encoding="utf-8")
    (split_dir / "events.jsonl").write_text(
        "\n".join(json_line(event) for event in events) + ("\n" if events else ""),
        encoding="utf-8",
    )

    event_fields = (
        "split", "sequence", "event_id", "target_frame", "gt_id", "lr_old_pred_id",
        "lr_new_pred_id", "lr_last_match_frame", "lr_gap_frames", "window_start", "window_end",
        "query_crop_count", "target_crop_available", "candidate_count",
        "correct_identity_available_in_window", "top1_correct", "has_positive_candidate",
    )
    with (split_dir / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=event_fields)
        writer.writeheader()
        for event in events:
            writer.writerow({field: event[field] for field in event_fields})

    candidate_fields = (
        "split", "sequence", "event_id", "target_frame", "gt_id", "lr_old_pred_id",
        "lr_new_pred_id", "rank", "candidate_id", "candidate_frame", "seconds_before",
        "hr_pred_id", "hr_bbox_xyxy", "hr_confidence", "similarity_to_old_track",
        "similarity_to_target", "bridge_similarity", "label", "candidate_crop_path",
        "source_image",
    )
    with (split_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_fields)
        writer.writeheader()
        for event in events:
            base = {field: event[field] for field in candidate_fields if field in event}
            for candidate in event["candidates"]:
                row = {**base, **candidate}
                row["hr_bbox_xyxy"] = json.dumps(row["hr_bbox_xyxy"], separators=(",", ":"))
                writer.writerow({field: row[field] for field in candidate_fields})


def write_dataset_card(root: Path, config: dict, statistics: dict) -> None:
    train = statistics["train"]
    test = statistics["test"]
    lines = [
        "# QuadTrack cross-resolution ReID ranking dataset v1",
        "",
        "## Purpose",
        "",
        "Train and evaluate a model that ranks historical 2048x480 person crops for a",
        "512x120 ID-switch query. Each event is one ranking group and must never be split",
        "across datasets.",
        "",
        "## Leakage-safe split",
        "",
        "The split is grouped by complete video sequence. Train and test have no shared",
        "sequence, event, frame source, or crop file.",
        "",
        f"- Train sequences: `{', '.join(config['train_sequences'])}`",
        f"- Test sequences: `{', '.join(config['test_sequences'])}`",
        "",
        "| Split | Videos | Events | Candidate pairs | Positive | Negative | Positive in top-5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Train | {train['sequence_count']} | {train['event_count']} | {train['candidate_pair_count']} | {train['positive_pair_count']} | {train['negative_pair_count']} | {train['positive_in_top5_event_count']} |",
        f"| Test | {test['sequence_count']} | {test['event_count']} | {test['candidate_pair_count']} | {test['positive_pair_count']} | {test['negative_pair_count']} | {test['positive_in_top5_event_count']} |",
        "",
        "## Sample definition",
        "",
        "For a switch at frame `T`, the input group contains up to five old-track LR crops,",
        "one switched-target LR crop, and up to five ranked HR person crops from",
        f"`{config['candidate_window']['exact_frame_interval']}`. A candidate label is `1` when its HR box overlaps the target GT",
        "identity by IoU >= 0.50; otherwise it is `0`. GT is applied after ReID ranking.",
        "",
        "## Layout",
        "",
        "```text",
        "reid_ranking_dataset_v1/",
        "├── split_config.json",
        "├── statistics.json",
        "├── dataset_card.md",
        "├── train_data/",
        "│   ├── sequences.txt",
        "│   ├── events.jsonl",
        "│   ├── events.csv",
        "│   ├── candidates.csv",
        "│   └── crops/<event_id>/{query,target,candidate}_*.jpg",
        "└── test_data/",
        "    └── ... same schema ...",
        "```",
        "",
        "All crop paths in manifests are relative to this package root. `events.jsonl` is the",
        "canonical grouped format for learning-to-rank; `candidates.csv` is a convenient flat",
        "view for analysis. Do not randomly split rows from either file.",
        "",
        "## Recommended evaluation",
        "",
        "Report event-level top-1 accuracy, Recall@5, MRR, and no-match/reject performance.",
        "Events without a positive candidate cannot be solved by reranking alone and should",
        "be retained for reject-head or end-to-end coverage evaluation.",
        "",
        "## Reproduction",
        "",
        "Run `python build_reid_ranking_dataset.py` inside the project environment. If this",
        "version already exists, pass a new versioned `--output-root`; the script refuses to",
        "overwrite an existing package. Exact source checksums and retrieval configuration",
        "are recorded in `split_config.json`.",
        "",
    ]
    (root / "dataset_card.md").write_text("\n".join(lines), encoding="utf-8")


def validate_package(root: Path, config: dict, statistics: dict) -> None:
    train_sequences = set(config["train_sequences"])
    test_sequences = set(config["test_sequences"])
    if train_sequences & test_sequences:
        raise AssertionError("Train/test sequence leakage detected")

    event_ids: dict[str, set[str]] = {}
    crop_paths: dict[str, set[str]] = {}
    source_paths: dict[str, set[str]] = {}
    for name in ("train", "test"):
        records = [
            json.loads(line)
            for line in (root / f"{name}_data" / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        expected_sequences = set(config[f"{name}_sequences"])
        if any(record["sequence"] not in expected_sequences for record in records):
            raise AssertionError(f"Unexpected sequence in {name} manifest")
        event_ids[name] = {record["event_id"] for record in records}
        crop_paths[name] = {
            path
            for record in records
            for path in (
                [item["crop_path"] for item in record["query_crops"]]
                + ([record["target_crop"]["crop_path"]] if record["target_crop"] else [])
                + [item["candidate_crop_path"] for item in record["candidates"]]
            )
            if path
        }
        source_paths[name] = {
            path
            for record in records
            for path in (
                [item["source_image"] for item in record["query_crops"]]
                + ([record["target_crop"]["source_image"]] if record["target_crop"] else [])
                + [item["source_image"] for item in record["candidates"]]
            )
        }
        if len(records) != statistics[name]["event_count"]:
            raise AssertionError(f"Event count mismatch for {name}")

    if event_ids["train"] & event_ids["test"]:
        raise AssertionError("Train/test event leakage detected")
    if crop_paths["train"] & crop_paths["test"]:
        raise AssertionError("Train/test crop leakage detected")
    if source_paths["train"] & source_paths["test"]:
        raise AssertionError("Train/test source-frame leakage detected")


def build(args: argparse.Namespace) -> Path:
    events_path = args.events_json.resolve()
    summary_path = args.retrieval_summary.resolve()
    dataset_root = args.dataset_root.resolve()
    lowres_dir = args.lowres_dir.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists: {output_root}. Choose a new versioned --output-root."
        )
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")

    source_events = load_json(events_path)
    retrieval_summary = load_json(summary_path)
    all_sequences = list(retrieval_summary["sequences"])
    test_sequences = sorted(set(args.test_sequences))
    unknown = sorted(set(test_sequences) - set(all_sequences))
    if unknown:
        raise ValueError(f"Unknown test sequences: {unknown}")
    train_sequences = sorted(set(all_sequences) - set(test_sequences))
    if not train_sequences or not test_sequences:
        raise ValueError("Both train and test must contain at least one sequence")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.building-", dir=output_root.parent) as tmp:
        root = Path(tmp) / output_root.name
        root.mkdir()
        prepared: dict[str, list[dict]] = {"train": [], "test": []}
        crop_counts: dict[str, dict[str, int]] = {
            "train": defaultdict(int),
            "test": defaultdict(int),
        }
        lowres_cache: dict[str, dict] = {}

        for source_event in source_events:
            sequence = source_event["sequence"]
            split = "test" if sequence in test_sequences else "train"
            if sequence not in lowres_cache:
                lowres_path = lowres_dir / args.lowres_pattern.format(sequence=sequence)
                lowres_cache[sequence] = load_mot(lowres_path, ground_truth=False)
            lowres_frames = lowres_cache[sequence]
            event_id = source_event["event_id"]
            event_crop_dir = Path(f"{split}_data") / "crops" / event_id

            query_crops = []
            for index, frame_id in enumerate(source_event["query_frames"], start=1):
                row = find_track_row(lowres_frames, frame_id, int(source_event["lr_old_pred_id"]))
                source_image = dataset_root / sequence / "img1" / f"{frame_id:06d}.jpg"
                image = read_image(str(source_image))
                scale_x = args.lowres_width / image.shape[1]
                scale_y = args.lowres_height / image.shape[0]
                lr_image = cv2.resize(
                    image,
                    (args.lowres_width, args.lowres_height),
                    interpolation=cv2.INTER_AREA,
                )
                original_box = np.asarray(row[1], dtype=float)
                lr_box = original_box.copy()
                lr_box[[0, 2]] *= scale_x
                lr_box[[1, 3]] *= scale_y
                relative_crop = event_crop_dir / f"query_old_{index:02d}_f{frame_id:06d}.jpg"
                clipped = None
                if not args.manifest_only:
                    clipped = save_crop(lr_image, lr_box, root / relative_crop, args.jpeg_quality)
                    crop_counts[split]["query_lr"] += 1
                query_crops.append(
                    {
                        "frame": frame_id,
                        "track_id": int(source_event["lr_old_pred_id"]),
                        "bbox_original_xyxy": rounded_box(original_box),
                        "bbox_lr_xyxy": rounded_box(lr_box),
                        "bbox_lr_clipped_xyxy": clipped,
                        "crop_path": relative_crop.as_posix() if not args.manifest_only else None,
                        "source_image": relative_to_project(source_image),
                    }
                )

            target_crop = None
            if source_event["target_embedding_available"]:
                target_frame = int(source_event["target_frame"])
                row = find_track_row(
                    lowres_frames, target_frame, int(source_event["lr_new_pred_id"])
                )
                source_image = dataset_root / sequence / "img1" / f"{target_frame:06d}.jpg"
                image = read_image(str(source_image))
                scale_x = args.lowres_width / image.shape[1]
                scale_y = args.lowres_height / image.shape[0]
                lr_image = cv2.resize(
                    image,
                    (args.lowres_width, args.lowres_height),
                    interpolation=cv2.INTER_AREA,
                )
                original_box = np.asarray(row[1], dtype=float)
                lr_box = original_box.copy()
                lr_box[[0, 2]] *= scale_x
                lr_box[[1, 3]] *= scale_y
                relative_crop = event_crop_dir / f"target_lr_f{target_frame:06d}.jpg"
                clipped = None
                if not args.manifest_only:
                    clipped = save_crop(lr_image, lr_box, root / relative_crop, args.jpeg_quality)
                    crop_counts[split]["target_lr"] += 1
                target_crop = {
                    "frame": target_frame,
                    "track_id": int(source_event["lr_new_pred_id"]),
                    "bbox_original_xyxy": rounded_box(original_box),
                    "bbox_lr_xyxy": rounded_box(lr_box),
                    "bbox_lr_clipped_xyxy": clipped,
                    "crop_path": relative_crop.as_posix() if not args.manifest_only else None,
                    "source_image": relative_to_project(source_image),
                }

            candidates = []
            for rank, source_candidate in enumerate(source_event["top_frames"], start=1):
                frame_id = int(source_candidate["candidate_frame"])
                source_image = dataset_root / sequence / "img1" / f"{frame_id:06d}.jpg"
                image = read_image(str(source_image))
                relative_crop = event_crop_dir / (
                    f"candidate_r{rank:02d}_f{frame_id:06d}_id{int(source_candidate['hr_pred_id'])}.jpg"
                )
                clipped = None
                if not args.manifest_only:
                    clipped = save_crop(
                        image,
                        source_candidate["hr_bbox_xyxy"],
                        root / relative_crop,
                        args.jpeg_quality,
                    )
                    crop_counts[split]["candidate_hr"] += 1
                candidates.append(
                    {
                        "rank": rank,
                        "candidate_id": f"{event_id}_r{rank:02d}",
                        "candidate_frame": frame_id,
                        "seconds_before": source_candidate["seconds_before"],
                        "hr_pred_id": int(source_candidate["hr_pred_id"]),
                        "hr_bbox_xyxy": source_candidate["hr_bbox_xyxy"],
                        "hr_bbox_clipped_xyxy": clipped,
                        "hr_confidence": source_candidate["hr_confidence"],
                        "similarity_to_old_track": source_candidate["similarity_to_old_track"],
                        "similarity_to_target": source_candidate["similarity_to_target"],
                        "bridge_similarity": source_candidate["bridge_similarity"],
                        "label": int(source_candidate["correct_gt_identity"]),
                        "candidate_crop_path": (
                            relative_crop.as_posix() if not args.manifest_only else None
                        ),
                        "source_image": relative_to_project(source_image),
                    }
                )

            record = {
                "schema_version": "1.0",
                "split": split,
                "sequence": sequence,
                "event_id": event_id,
                "target_frame": int(source_event["target_frame"]),
                "gt_id": int(source_event["gt_id"]),
                "lr_old_pred_id": int(source_event["lr_old_pred_id"]),
                "lr_new_pred_id": int(source_event["lr_new_pred_id"]),
                "lr_last_match_frame": int(source_event["lr_last_match_frame"]),
                "lr_gap_frames": int(source_event["lr_gap_frames"]),
                "window_start": int(source_event["window_start"]),
                "window_end": int(source_event["window_end"]),
                "query_crop_count": len(query_crops),
                "query_crops": query_crops,
                "target_crop_available": target_crop is not None,
                "target_crop": target_crop,
                "candidate_count": len(candidates),
                "candidates": candidates,
                "correct_identity_available_in_window": bool(
                    source_event["correct_identity_available_in_window"]
                ),
                "top1_correct": bool(candidates and candidates[0]["label"]),
                "has_positive_candidate": bool(any(item["label"] for item in candidates)),
            }
            prepared[split].append(record)

        for name in prepared:
            prepared[name].sort(key=lambda item: (item["sequence"], item["target_frame"], item["event_id"]))

        split_sequences = {"train": train_sequences, "test": test_sequences}
        statistics = {
            name: split_statistics(prepared[name], split_sequences[name], crop_counts[name])
            for name in ("train", "test")
        }
        statistics["all"] = split_statistics(
            prepared["train"] + prepared["test"], all_sequences, {
                key: crop_counts["train"].get(key, 0) + crop_counts["test"].get(key, 0)
                for key in set(crop_counts["train"]) | set(crop_counts["test"])
            }
        )

        max_lookback_frames = int(retrieval_summary["retrieval_frames"][1])
        config = {
            "dataset_name": "quadtrack_cross_resolution_reid_ranking",
            "dataset_version": "1.0",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "split_unit": "sequence",
            "train_sequences": train_sequences,
            "test_sequences": test_sequences,
            "materialized_crops": not args.manifest_only,
            "lowres_physical_size": [args.lowres_width, args.lowres_height],
            "candidate_window": {
                "seconds_before": retrieval_summary["retrieval_seconds"],
                "frames_before": retrieval_summary["retrieval_frames"],
                "exact_frame_interval": f"T-{max_lookback_frames} through T-1",
            },
            "top_k": retrieval_summary["top_k"],
            "evaluation_iou": retrieval_summary["evaluation_iou"],
            "reid_model": Path(retrieval_summary["model"]).name,
            "source_files": {
                "events_json": relative_to_project(events_path),
                "events_json_sha256": sha256(events_path),
                "retrieval_summary": relative_to_project(summary_path),
                "retrieval_summary_sha256": sha256(summary_path),
                "lowres_predictions": relative_to_project(lowres_dir),
                "original_frames": relative_to_project(dataset_root),
            },
        }

        for name in ("train", "test"):
            write_split_files(root, name, prepared[name], split_sequences[name])
        (root / "split_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (root / "statistics.json").write_text(
            json.dumps(statistics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        write_dataset_card(root, config, statistics)
        validate_package(root, config, statistics)
        root.rename(output_root)

    return output_root


def main() -> None:
    args = parse_args()
    output = build(args)
    statistics = load_json(output / "statistics.json")
    print(json.dumps(statistics, indent=2, ensure_ascii=False))
    print(f"Dataset: {output}")


if __name__ == "__main__":
    main()
