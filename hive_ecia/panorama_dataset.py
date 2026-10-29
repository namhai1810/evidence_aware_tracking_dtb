"""Build leakage-audited panorama switch events for zero-shot ECIA evaluation.

Manual GT is used only to identify oracle ID-switch events and, after the
LR-only candidate pool is frozen, to label candidates for post-hoc analysis.
It is never an operational HIVE or ECIA feature.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm.auto import tqdm

from build_hive_history_ranking_v2 import (
    appearance_descriptor,
    rounded_box,
    select_candidate_pool,
)
from evaluate_mot import load_mot
from hr_finding.datasets.preprocessing import select_track_history
from rank_highres_retrieval_frames import build_match_index, find_switch_events

from hive_hr_reid import run_experiment as baseline


CODE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("panorama_config.yaml"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-events-per-sequence", type=int, default=None)
    return parser.parse_args()


def resolve_config_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (CODE_ROOT / path).resolve()


def audit_splits(splits: dict[str, list[str]]) -> None:
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = set(splits[left]) & set(splits[right])
            if overlap:
                raise ValueError(f"Sequence leakage between {left} and {right}: {sorted(overlap)}")


def shift_prediction_frames(source: dict, offset: int, length: int) -> dict:
    """Map source frames offset+1..offset+length to prepared frames 1..length."""

    return {
        int(frame) - offset: rows
        for frame, rows in source.items()
        if offset + 1 <= int(frame) <= offset + length
    }


def crop_lr(
    image: np.ndarray,
    box: np.ndarray,
    lowres_size: tuple[int, int],
) -> tuple[np.ndarray, list[float], list[int]]:
    width, height = lowres_size
    lr = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    scaled = np.asarray(box, dtype=np.float32).copy()
    scaled[[0, 2]] *= width / image.shape[1]
    scaled[[1, 3]] *= height / image.shape[0]
    x1 = max(0, min(width - 1, int(np.floor(scaled[0]))))
    y1 = max(0, min(height - 1, int(np.floor(scaled[1]))))
    x2 = max(x1 + 1, min(width, int(np.ceil(scaled[2]))))
    y2 = max(y1 + 1, min(height, int(np.ceil(scaled[3]))))
    return lr[y1:y2, x1:x2].copy(), rounded_box(scaled), [x1, y1, x2, y2]


def save_jpeg(path: Path, crop: np.ndarray) -> None:
    if crop is None or not crop.size:
        raise ValueError(f"Empty crop: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"Could not write {path}")


def find_track_row(frames: dict, frame: int, track_id: int) -> tuple | None:
    return next((row for row in frames.get(frame, []) if int(row[0]) == int(track_id)), None)


def materialize_lr_record(
    *,
    image_path: Path,
    frame: int,
    track_id: int,
    box: np.ndarray,
    crop_path: Path,
    relative_crop_path: Path,
    lowres_size: tuple[int, int],
) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    crop, scaled, clipped = crop_lr(image, box, lowres_size)
    save_jpeg(crop_path, crop)
    return {
        "frame": int(frame),
        "track_id": int(track_id),
        "bbox_original_xyxy": rounded_box(box),
        "bbox_lr_xyxy": scaled,
        "bbox_lr_clipped_xyxy": clipped,
        "crop_path": relative_crop_path.as_posix(),
        "source_image": str(image_path.resolve()),
    }


def build_sequence_events(
    *,
    sequence: str,
    split: str,
    dataset_dir: Path,
    predictions: dict,
    output_root: Path,
    event_config: dict,
    max_events: int | None,
) -> tuple[list[dict], dict]:
    gt = load_mot(dataset_dir / "gt" / "gt.txt", ground_truth=True)
    matched_by_frame, _ = build_match_index(gt, predictions, float(event_config["evaluation_iou"]))
    switches = find_switch_events(matched_by_frame)
    if max_events is not None:
        switches = switches[:max_events]
    pred_to_gt = {
        (int(frame), int(record["pred_id"])): int(gt_id)
        for frame, matches in matched_by_frame.items()
        for gt_id, record in matches.items()
    }
    lowres_size = (int(event_config["lowres_width"]), int(event_config["lowres_height"]))
    fps = float(event_config["fps"])
    window = int(event_config["candidate_window_frames"])
    events = []
    skipped = Counter()

    # Cache decoded images and LR appearance descriptors across overlapping
    # event windows; proposal construction is LR-only.
    image_cache = baseline.ImageCache(max_items=64)
    descriptor_cache: dict[tuple[int, int], np.ndarray] = {}

    def image(frame: int) -> np.ndarray:
        return image_cache.read(dataset_dir / "img1" / f"{frame:06d}.jpg")

    def descriptor(frame: int, row: tuple) -> np.ndarray:
        key = (int(frame), int(row[0]))
        if key not in descriptor_cache:
            crop, _, _ = crop_lr(image(frame), row[1], lowres_size)
            descriptor_cache[key] = appearance_descriptor(crop)
        return descriptor_cache[key]

    for number, source in enumerate(
        tqdm(
            switches,
            desc=f"build {split}/{sequence}",
            unit="event",
            dynamic_ncols=True,
        ),
        start=1,
    ):
        target_frame = int(source["target_frame"])
        old_id = int(source["lr_old_pred_id"])
        new_id = int(source["lr_new_pred_id"])
        target_row = find_track_row(predictions, target_frame, new_id)
        history = select_track_history(
            predictions,
            old_id,
            target_frame,
            int(event_config["history_frames"]),
            int(event_config["max_history"]),
        )
        if target_row is None or not history:
            skipped["missing_target_or_old_history"] += 1
            continue

        event_id = f"{sequence}_{number:04d}"
        relative_dir = Path(f"{split}_data") / "crops" / event_id
        query_crops = []
        for slot, (frame, box, _) in enumerate(history, start=1):
            relative = relative_dir / f"query_old_{slot:02d}_f{frame:06d}.jpg"
            query_crops.append(
                materialize_lr_record(
                    image_path=dataset_dir / "img1" / f"{frame:06d}.jpg",
                    frame=frame,
                    track_id=old_id,
                    box=box,
                    crop_path=output_root / relative,
                    relative_crop_path=relative,
                    lowres_size=lowres_size,
                )
            )
        target_relative = relative_dir / f"target_lr_f{target_frame:06d}.jpg"
        target_crop = materialize_lr_record(
            image_path=dataset_dir / "img1" / f"{target_frame:06d}.jpg",
            frame=target_frame,
            track_id=new_id,
            box=target_row[1],
            crop_path=output_root / target_relative,
            relative_crop_path=target_relative,
            lowres_size=lowres_size,
        )

        query_descriptor = np.mean(
            [descriptor(int(frame), find_track_row(predictions, int(frame), old_id)) for frame, _, _ in history],
            axis=0,
        )
        query_descriptor /= max(float(np.linalg.norm(query_descriptor)), 1e-12)
        target_descriptor = descriptor(target_frame, target_row)

        frame_best = []
        for frame in range(max(1, target_frame - window), target_frame):
            scored = []
            for row in predictions.get(frame, []):
                value = descriptor(frame, row)
                old_similarity = float(np.dot(query_descriptor, value))
                target_similarity = float(np.dot(target_descriptor, value))
                scored.append(
                    {
                        "candidate_frame": int(frame),
                        "lr_pred_id": int(row[0]),
                        "candidate_bbox_xyxy": rounded_box(row[1]),
                        "lr_confidence": round(float(row[2]), 6),
                        "lr_similarity_to_old": round(old_similarity, 6),
                        "lr_similarity_to_target": round(target_similarity, 6),
                        "lr_bridge_similarity": round(min(old_similarity, target_similarity), 6),
                        "source_image": str((dataset_dir / "img1" / f"{frame:06d}.jpg").resolve()),
                    }
                )
            if scored:
                frame_best.append(
                    max(scored, key=lambda item: (item["lr_bridge_similarity"], item["lr_confidence"]))
                )

        class CandidateArgs:
            appearance_candidates = int(event_config["appearance_candidates"])
            recent_candidates = int(event_config["recent_candidates"])
            diverse_candidates = int(event_config["diverse_candidates"])
            max_candidates = int(event_config["max_candidates"])
            min_spacing_frames = int(event_config["min_spacing_frames"])

        # Candidate pool is finalized before GT identities are consulted.
        selected = select_candidate_pool(frame_best, CandidateArgs())
        candidates = []
        for rank, candidate in enumerate(selected, start=1):
            matched_gt = pred_to_gt.get((candidate["candidate_frame"], candidate["lr_pred_id"]))
            item = dict(candidate)
            item.update(
                rank=rank,
                candidate_id=f"{event_id}_lr{rank:02d}",
                seconds_before=round((target_frame - candidate["candidate_frame"]) / fps, 3),
                matched_gt_id=matched_gt,
                label=int(matched_gt == int(source["gt_id"])),
                negative_kind=(
                    None
                    if matched_gt == int(source["gt_id"])
                    else "wrong_identity" if matched_gt is not None else "unmatched_detection"
                ),
            )
            candidates.append(item)

        events.append(
            {
                "schema_version": "panorama-ecia-1.0",
                "split": split,
                "sequence": sequence,
                "event_id": event_id,
                "target_frame": target_frame,
                "gt_id": int(source["gt_id"]),
                "lr_old_pred_id": old_id,
                "lr_new_pred_id": new_id,
                "lr_last_match_frame": int(source["lr_last_match_frame"]),
                "lr_gap_frames": int(source["lr_gap_frames"]),
                "query_crops": query_crops,
                "target_crop": target_crop,
                "candidates": candidates,
                "candidate_count": len(candidates),
                "has_positive_candidate": bool(any(row["label"] for row in candidates)),
            }
        )
    return events, {
        "oracle_switches": len(switches),
        "materialized_events": len(events),
        "events_with_positive_candidate": sum(row["has_positive_candidate"] for row in events),
        "candidate_pairs": sum(len(row["candidates"]) for row in events),
        "skipped": dict(skipped),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def build(args: argparse.Namespace) -> Path:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    splits = {key: list(value) for key, value in data["splits"].items()}
    audit_splits(splits)
    manual_root = resolve_config_path(data["manual_root"])
    benchmark_root = resolve_config_path(data["benchmark_root"])
    recorded_splits = json.loads((manual_root / "splits.json").read_text())
    if recorded_splits != splits:
        raise ValueError(f"Configured split differs from audited manual split: {recorded_splits}")

    output = (args.output_root or resolve_config_path(config["output_root"])).resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}; pass --overwrite")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    root = temporary / output.name
    root.mkdir()
    try:
        sequence_split = {sequence: split for split, values in splits.items() for sequence in values}
        all_events: dict[str, list[dict]] = {name: [] for name in splits}
        audits = {}
        baseline_metrics = {}
        for sequence, split in sequence_split.items():
            print(f"Preparing {split}/{sequence}", flush=True)
            source_sequence = str(data["sources"][sequence])
            source_prediction = benchmark_root / source_sequence / "pred" / "tracks.txt"
            source_dataset = manual_root / "mot_challenge" / split / sequence
            length = int(
                next(
                    line.split("=", 1)[1]
                    for line in (source_dataset / "seqinfo.ini").read_text().splitlines()
                    if line.startswith("seqLength=")
                )
            )
            predictions = shift_prediction_frames(
                load_mot(source_prediction, ground_truth=False),
                int(data["trim_start_frames"]),
                length,
            )
            flat = root / "dataset" / sequence
            (flat / "gt").mkdir(parents=True)
            shutil.copy2(source_dataset / "gt" / "gt.txt", flat / "gt" / "gt.txt")
            os.symlink(source_dataset.resolve() / "img1", flat / "img1", target_is_directory=True)
            shutil.copy2(source_dataset / "seqinfo.ini", flat / "seqinfo.ini")
            baseline.write_mot(root / "predictions" / f"{sequence}.txt", predictions)
            gt = load_mot(flat / "gt" / "gt.txt", ground_truth=True)
            metric_args = argparse.Namespace(iou=float(config["event_generation"]["evaluation_iou"]))
            baseline_metrics[sequence] = baseline.basic_metrics(
                metric_args,
                [sequence],
                {sequence: predictions},
                {sequence: gt},
                include_hota=True,
            )["aggregate"]
            events, event_audit = build_sequence_events(
                sequence=sequence,
                split=split,
                dataset_dir=flat,
                predictions=predictions,
                output_root=root,
                event_config=config["event_generation"],
                max_events=args.max_events_per_sequence,
            )
            all_events[split].extend(events)
            audits[sequence] = {"split": split, "frames": length, **event_audit}

        for split, rows in all_events.items():
            rows.sort(key=lambda row: (row["sequence"], row["target_frame"], row["event_id"]))
            write_jsonl(root / f"{split}_data" / "events.jsonl", rows)
            (root / f"{split}_data" / "sequences.txt").write_text(
                "\n".join(splits[split]) + "\n", encoding="utf-8"
            )

        split_config = {
            "dataset_name": "panorama_manual_mot_ecia_events",
            "protocol": "oracle_switch_events; LR-only proposals; GT labels joined after pool freeze",
            "split_unit": "complete_sequence",
            "train_sequences": splits["train"],
            "validation_sequences": splits["val"],
            "test_sequences": splits["test"],
            "source_files": {
                "lowres_predictions": str((output / "predictions").resolve()),
                "original_frames": str((output / "dataset").resolve()),
            },
        }
        (root / "split_config.json").write_text(json.dumps(split_config, indent=2) + "\n")
        audit = {
            "passed": True,
            "sequence_disjoint": True,
            "configured_splits": splits,
            "manual_splits_match": True,
            "prediction_frame_offset": int(data["trim_start_frames"]),
            "candidate_pool_frozen_before_gt_label": True,
            "sequences": audits,
            "baseline_metrics": baseline_metrics,
        }
        (root / "leakage_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        root.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    shutil.rmtree(temporary, ignore_errors=True)
    print(f"Panorama ECIA event dataset: {output}")
    print(f"Leakage audit: {output / 'leakage_audit.json'}")
    return output


def main() -> int:
    build(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
