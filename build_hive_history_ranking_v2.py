#!/usr/bin/env python3
"""Build a video-disjoint, LR-proposed history-ranking dataset for HIVE.

Every event remains a known LR identity switch. Candidate proposals are selected
without GT, HR detections, or HR ReID: only LR tracker boxes, LR pixels, confidence,
time, and deterministic temporal diversity are used. GT is joined only after the
candidate pool is frozen to produce identity-correctness labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from evaluate_mot import load_mot
from rank_highres_retrieval_frames import build_match_index


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
OUTPUTS = CODE_DIR / "outputs"
DEFAULT_INPUT = OUTPUTS / "reid_ranking_dataset_v1"
DEFAULT_OUTPUT = OUTPUTS / "hive_history_ranking_v2"
DEFAULT_DATASET = PROJECT_ROOT / "QuadTrack" / "train"
DEFAULT_PREDICTIONS = OUTPUTS / "all_sequences_reid_512x120" / "predictions"
DEFAULT_VAL_SEQUENCES = ("0019", "0028", "0030")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--prediction-pattern",
        default="{sequence}_yolo11l_botsort_reid_pre512x120.txt",
    )
    parser.add_argument("--validation-sequences", nargs="+", default=list(DEFAULT_VAL_SEQUENCES))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--window-frames", type=int, default=150)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--appearance-candidates", type=int, default=10)
    parser.add_argument("--recent-candidates", type=int, default=5)
    parser.add_argument("--diverse-candidates", type=int, default=5)
    parser.add_argument("--min-spacing-frames", type=int, default=3)
    parser.add_argument("--evaluation-iou", type=float, default=0.5)
    parser.add_argument("--lowres-width", type=int, default=512)
    parser.add_argument("--lowres-height", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def json_line(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rounded_box(box: np.ndarray | list[float]) -> list[float]:
    return [round(float(value), 2) for value in box]


def clamp_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in box)
    left = max(0, min(width - 1, math.floor(x1)))
    top = max(0, min(height - 1, math.floor(y1)))
    right = max(left + 1, min(width, math.ceil(x2)))
    bottom = max(top + 1, min(height, math.ceil(y2)))
    return left, top, right, bottom


def lr_crop(
    lr_frame: np.ndarray,
    box_original: np.ndarray,
    original_width: int,
    original_height: int,
) -> np.ndarray:
    scaled = np.asarray(box_original, dtype=np.float32).copy()
    scaled[[0, 2]] *= lr_frame.shape[1] / original_width
    scaled[[1, 3]] *= lr_frame.shape[0] / original_height
    x1, y1, x2, y2 = clamp_box(scaled, lr_frame.shape[1], lr_frame.shape[0])
    return lr_frame[y1:y2, x1:x2]


def appearance_descriptor(image: np.ndarray) -> np.ndarray:
    """Small LR-only descriptor used for proposal mining, never as a GT label."""
    if image.size == 0:
        return np.zeros(134, dtype=np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).reshape(-1)
    resized = cv2.resize(image, (16, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    means = resized.mean(axis=(0, 1))
    stds = resized.std(axis=(0, 1))
    vector = np.concatenate([histogram.astype(np.float32), means, stds])
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def descriptor_from_file(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return appearance_descriptor(image)


def evenly_spaced(items: list[dict], count: int) -> list[dict]:
    if count <= 0 or not items:
        return []
    ordered = sorted(items, key=lambda item: item["candidate_frame"], reverse=True)
    if len(ordered) <= count:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, num=count, dtype=int)
    return [ordered[int(index)] for index in indices]


def append_unique(
    selected: list[dict],
    candidates: list[dict],
    limit: int,
    role: str,
    spacing: int,
) -> None:
    for candidate in candidates:
        if len(selected) >= limit:
            return
        key = (candidate["candidate_frame"], candidate["lr_pred_id"])
        if any((item["candidate_frame"], item["lr_pred_id"]) == key for item in selected):
            continue
        if spacing > 0 and any(
            abs(candidate["candidate_frame"] - item["candidate_frame"]) < spacing
            for item in selected
        ):
            continue
        chosen = dict(candidate)
        chosen["proposal_role"] = role
        selected.append(chosen)


def select_candidate_pool(frame_best: list[dict], args: argparse.Namespace) -> list[dict]:
    """Freeze an LR-only pool before any GT identity label is attached."""
    selected: list[dict] = []
    appearance = sorted(
        frame_best,
        key=lambda item: (
            item["lr_bridge_similarity"], item["lr_confidence"], item["candidate_frame"]
        ),
        reverse=True,
    )
    recent = sorted(frame_best, key=lambda item: item["candidate_frame"], reverse=True)
    diverse = evenly_spaced(frame_best, args.diverse_candidates * 2)
    append_unique(
        selected,
        appearance,
        min(args.appearance_candidates, args.max_candidates),
        "appearance_hard",
        args.min_spacing_frames,
    )
    append_unique(
        selected,
        recent,
        min(args.appearance_candidates + args.recent_candidates, args.max_candidates),
        "recent",
        args.min_spacing_frames,
    )
    append_unique(
        selected,
        diverse,
        args.max_candidates,
        "temporal_diverse",
        args.min_spacing_frames,
    )
    # Fill only if spacing prevented the requested pool size.
    append_unique(selected, appearance, args.max_candidates, "appearance_fill", 0)
    return selected[: args.max_candidates]


def build_sequence_candidates(
    sequence: str,
    events: list[dict],
    args: argparse.Namespace,
) -> tuple[dict[str, list[dict]], dict]:
    gt_frames = load_mot(args.dataset_root / sequence / "gt" / "gt.txt", ground_truth=True)
    prediction_path = args.prediction_dir / args.prediction_pattern.format(sequence=sequence)
    prediction_frames = load_mot(prediction_path, ground_truth=False)
    matched_by_frame, _ = build_match_index(gt_frames, prediction_frames, args.evaluation_iou)
    pred_to_gt = {
        (frame, int(record["pred_id"])): int(gt_id)
        for frame, matches in matched_by_frame.items()
        for gt_id, record in matches.items()
    }

    required_frames = {
        frame
        for event in events
        for frame in range(
            max(0, int(event["target_frame"]) - args.window_frames),
            int(event["target_frame"]),
        )
    }
    descriptor_rows: dict[int, list[dict]] = defaultdict(list)
    source_size = None
    for frame in sorted(required_frames):
        rows = prediction_frames.get(frame, [])
        if not rows:
            continue
        image_path = args.dataset_root / sequence / "img1" / f"{frame:06d}.jpg"
        original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError(image_path)
        source_size = (original.shape[1], original.shape[0])
        lr_frame = cv2.resize(
            original,
            (args.lowres_width, args.lowres_height),
            interpolation=cv2.INTER_AREA,
        )
        for pred_id, box, confidence in rows:
            crop = lr_crop(lr_frame, box, original.shape[1], original.shape[0])
            descriptor_rows[frame].append(
                {
                    "candidate_frame": int(frame),
                    "lr_pred_id": int(pred_id),
                    "candidate_bbox_xyxy": rounded_box(box),
                    "lr_confidence": round(float(confidence), 6),
                    "descriptor": appearance_descriptor(crop),
                    "source_image": f"QuadTrack/train/{sequence}/img1/{frame:06d}.jpg",
                }
            )

    result: dict[str, list[dict]] = {}
    event_audit = Counter()
    for event in events:
        query_descriptors = [
            descriptor_from_file(args.input_root / query["crop_path"])
            for query in event.get("query_crops", [])
        ]
        if query_descriptors:
            query = np.mean(query_descriptors, axis=0)
            query /= max(float(np.linalg.norm(query)), 1e-12)
        else:
            query = np.zeros(134, dtype=np.float32)
        if event.get("target_crop"):
            target = descriptor_from_file(args.input_root / event["target_crop"]["crop_path"])
        else:
            target = query

        start = max(0, int(event["target_frame"]) - args.window_frames)
        end = int(event["target_frame"]) - 1
        frame_best = []
        for frame in range(start, end + 1):
            scored = []
            for row in descriptor_rows.get(frame, []):
                old_similarity = cosine(query, row["descriptor"])
                target_similarity = cosine(target, row["descriptor"])
                candidate = {key: value for key, value in row.items() if key != "descriptor"}
                candidate["lr_similarity_to_old"] = round(old_similarity, 6)
                candidate["lr_similarity_to_target"] = round(target_similarity, 6)
                candidate["lr_bridge_similarity"] = round(
                    min(old_similarity, target_similarity), 6
                )
                scored.append(candidate)
            if scored:
                frame_best.append(
                    max(
                        scored,
                        key=lambda item: (
                            item["lr_bridge_similarity"], item["lr_confidence"]
                        ),
                    )
                )

        # IMPORTANT: selection is complete before pred_to_gt or event gt_id is consulted.
        selected = select_candidate_pool(frame_best, args)
        labeled = []
        for rank, candidate in enumerate(selected, start=1):
            matched_gt = pred_to_gt.get(
                (candidate["candidate_frame"], candidate["lr_pred_id"])
            )
            item = dict(candidate)
            item.update(
                {
                    "rank": rank,
                    "candidate_id": f"{event['event_id']}_lr{rank:02d}",
                    "seconds_before": round(
                        (int(event["target_frame"]) - candidate["candidate_frame"]) / args.fps,
                        3,
                    ),
                    "matched_gt_id": matched_gt,
                    "label": int(matched_gt == int(event["gt_id"])),
                    "negative_kind": (
                        None
                        if matched_gt == int(event["gt_id"])
                        else "wrong_identity" if matched_gt is not None else "unmatched_detection"
                    ),
                }
            )
            labeled.append(item)
        result[event["event_id"]] = labeled
        event_audit["events"] += 1
        event_audit["events_with_candidates"] += int(bool(labeled))
        event_audit["events_with_positive"] += int(any(item["label"] for item in labeled))
        event_audit["candidate_pairs"] += len(labeled)
        event_audit["positive_candidates"] += sum(item["label"] for item in labeled)
        event_audit["negative_candidates"] += sum(not item["label"] for item in labeled)
        event_audit["wrong_identity_negatives"] += sum(
            item["negative_kind"] == "wrong_identity" for item in labeled
        )
        event_audit["unmatched_detection_negatives"] += sum(
            item["negative_kind"] == "unmatched_detection" for item in labeled
        )
    return result, {"source_size": source_size, **dict(event_audit)}


def copy_event_inputs(event: dict, input_root: Path, output_root: Path) -> dict:
    copied = json.loads(json.dumps(event))
    for query in copied.get("query_crops", []):
        source = input_root / query["crop_path"]
        destination = output_root / query["crop_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
    if copied.get("target_crop"):
        source = input_root / copied["target_crop"]["crop_path"]
        destination = output_root / copied["target_crop"]["crop_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
    return copied


def split_stats(events: list[dict], sequences: list[str]) -> dict:
    candidates = [candidate for event in events for candidate in event["candidates"]]
    positives = sum(int(candidate["label"]) for candidate in candidates)
    roles = Counter(candidate["proposal_role"] for candidate in candidates)
    kinds = Counter(
        candidate["negative_kind"]
        for candidate in candidates
        if not candidate["label"]
    )
    return {
        "sequences": sequences,
        "sequence_count": len(sequences),
        "event_count": len(events),
        "candidate_pair_count": len(candidates),
        "positive_pair_count": positives,
        "negative_pair_count": len(candidates) - positives,
        "positive_pair_rate": positives / len(candidates) if candidates else 0.0,
        "events_with_positive": sum(event["has_positive_candidate"] for event in events),
        "events_without_positive": sum(not event["has_positive_candidate"] for event in events),
        "oracle_coverage": (
            sum(event["has_positive_candidate"] for event in events) / len(events)
            if events else 0.0
        ),
        "proposal_roles": dict(roles),
        "negative_kinds": {str(key): value for key, value in kinds.items()},
    }


def write_split(root: Path, name: str, events: list[dict], sequences: list[str]) -> None:
    directory = root / f"{name}_data"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sequences.txt").write_text("\n".join(sequences) + "\n", encoding="utf-8")
    (directory / "events.jsonl").write_text(
        "\n".join(json_line(event) for event in events) + "\n", encoding="utf-8"
    )
    event_fields = [
        "split", "sequence", "event_id", "target_frame", "gt_id", "lr_old_pred_id",
        "lr_new_pred_id", "candidate_count", "has_positive_candidate",
    ]
    with (directory / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=event_fields)
        writer.writeheader()
        for event in events:
            writer.writerow({field: event[field] for field in event_fields})
    candidate_fields = [
        "split", "sequence", "event_id", "target_frame", "candidate_id", "rank",
        "candidate_frame", "seconds_before", "lr_pred_id", "lr_confidence",
        "lr_bridge_similarity", "proposal_role", "matched_gt_id", "label", "negative_kind",
        "source_image",
    ]
    with (directory / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_fields)
        writer.writeheader()
        for event in events:
            for candidate in event["candidates"]:
                row = {field: candidate.get(field) for field in candidate_fields}
                row.update(
                    split=event["split"], sequence=event["sequence"],
                    event_id=event["event_id"], target_frame=event["target_frame"],
                )
                writer.writerow(row)


def leakage_audit(train: list[dict], test: list[dict], validation_sequences: set[str]) -> dict:
    groups = {
        "train": [event for event in train if event["sequence"] not in validation_sequences],
        "val": [event for event in train if event["sequence"] in validation_sequences],
        "test": test,
    }
    audit = {"passed": True, "pairs": {}}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_events = groups[left]
        right_events = groups[right]
        left_sequences = {event["sequence"] for event in left_events}
        right_sequences = {event["sequence"] for event in right_events}
        left_ids = {event["event_id"] for event in left_events}
        right_ids = {event["event_id"] for event in right_events}
        left_frames = {
            (event["sequence"], candidate["candidate_frame"])
            for event in left_events for candidate in event["candidates"]
        }
        right_frames = {
            (event["sequence"], candidate["candidate_frame"])
            for event in right_events for candidate in event["candidates"]
        }
        result = {
            "sequence_overlap": sorted(left_sequences & right_sequences),
            "event_overlap_count": len(left_ids & right_ids),
            "candidate_source_frame_overlap_count": len(left_frames & right_frames),
        }
        audit["pairs"][f"{left}_vs_{right}"] = result
        audit["passed"] &= not any(
            (result["sequence_overlap"], result["event_overlap_count"], result["candidate_source_frame_overlap_count"])
        )
    return audit


def report_text(config: dict, statistics: dict, leakage: dict) -> str:
    rows = []
    for split in ("train", "validation", "test", "all"):
        item = statistics[split]
        rows.append(
            f"| {split.title()} | {item['sequence_count']} | {item['event_count']} | "
            f"{item['candidate_pair_count']:,} | {item['positive_pair_count']:,} | "
            f"{item['negative_pair_count']:,} | {item['oracle_coverage']:.1%} |"
        )
    return f"""# HIVE history-ranking dataset v2

Generated: `{config['generated_at']}`

## Scope

This dataset trains **historical candidate retrieval after an ID switch is already known**.
It intentionally contains no normal/non-switch events and does not train a switch trigger.

## Construction

1. Preserve the original 925 GT-confirmed LR switch events and video-level split.
2. Search LR tracker detections only in `T-{config['window_frames']} ... T-1`.
3. Downsample the complete panorama to `{config['lowres_width']}x{config['lowres_height']}` before cropping.
4. Build an LR-only HSV appearance descriptor for proposal mining.
5. Select at most {config['max_candidates']} candidates using appearance-hard, recent, and temporally diverse routes.
6. Freeze the candidate pool, then use evaluator-identical Hungarian GT matching at IoU {config['evaluation_iou']:.2f} to assign candidate labels.

Neither HR detections nor HR ReID embeddings are used to form the v2 candidate pool. GT is used only for offline labels after selection.

## Change from v1

| Dataset | Candidate source | Max/event | Candidate pairs | Positive | Negative | Oracle coverage |
|---|---|---:|---:|---:|---:|---:|
| v1 | HR detections + HR ReID top-5 | 5 | 4,606 | 1,927 | 2,679 | 76.3% |
| v2 | LR tracker + LR-only proposal | 20 | {statistics['all']['candidate_pair_count']:,} | {statistics['all']['positive_pair_count']:,} | {statistics['all']['negative_pair_count']:,} | {statistics['all']['oracle_coverage']:.1%} |

The candidate sources differ, so v2 is not merely v1 with extra rows. It replaces the privileged HR proposal stage with a reproducible LR-only pool and adds {statistics['all']['negative_pair_count'] - 2679:,} negative pairs. V2 contains {statistics['all']['negative_kinds'].get('wrong_identity', 0):,} wrong-identity negatives and {statistics['all']['negative_kinds'].get('unmatched_detection', 0):,} unmatched-detection negatives.

## Dataset size

| Split | Videos | Switch events | Candidate pairs | Positive | Negative | Events with ≥1 positive |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Negative candidates are LR detections that either match a different GT identity or do not match any GT at IoU {config['evaluation_iou']:.2f}. They are not non-switch frames.

## Leakage audit

- Video-disjoint split: **{'PASS' if leakage['passed'] else 'FAIL'}**.
- Train/validation/test sequence overlap: **0**.
- Cross-split event overlap: **0**.
- Cross-split candidate source-frame overlap: **0**.
- Validation videos: {', '.join(config['validation_sequences'])}.
- Test videos: {', '.join(config['test_sequences'])}.

The split is applied at video level before model training. Candidate mining is performed independently inside each video's own history window. Test labels and test statistics must not be used for model selection.

## Files

- `train_data/events.jsonl`, `test_data/events.jsonl`: event-grouped manifests.
- `train_data/candidates.csv`, `test_data/candidates.csv`: flat candidate tables.
- `statistics.json`: full counts by learning-train, validation and test subsets.
- `leakage_audit.json`: machine-readable split audit.
- `split_config.json`: immutable provenance and construction parameters.

Rebuild command from the code directory:

```bash
python build_hive_history_ranking_v2.py
```

## Important limitation

The event trigger remains GT-confirmed offline: this dataset assumes a switch event has already been supplied. It evaluates LR-only historical proposal and ranking, not online switch detection and not measured IDSW correction.
"""


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.prediction_dir = args.prediction_dir.resolve()
    if args.output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset: {args.output_root}. Choose a new --output-root."
        )
    if args.appearance_candidates + args.recent_candidates + args.diverse_candidates != args.max_candidates:
        raise ValueError("Candidate route counts must sum to --max-candidates")

    split_config_v1 = json.loads(
        (args.input_root / "split_config.json").read_text(encoding="utf-8")
    )
    train_events = read_jsonl(args.input_root / "train_data" / "events.jsonl")
    test_events = read_jsonl(args.input_root / "test_data" / "events.jsonl")
    all_events = train_events + test_events
    by_sequence: dict[str, list[dict]] = defaultdict(list)
    for event in all_events:
        by_sequence[event["sequence"]].append(event)

    candidate_map: dict[str, list[dict]] = {}
    sequence_audit = {}
    for sequence in sorted(by_sequence):
        print(f"[{sequence}] mining LR candidates for {len(by_sequence[sequence])} events", flush=True)
        sequence_candidates, audit = build_sequence_candidates(
            sequence, by_sequence[sequence], args
        )
        candidate_map.update(sequence_candidates)
        sequence_audit[sequence] = audit

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.tmp-", dir=args.output_root.parent)
    )
    try:
        prepared = {"train": [], "test": []}
        for split, events in (("train", train_events), ("test", test_events)):
            for source_event in events:
                event = copy_event_inputs(source_event, args.input_root, temporary)
                candidates = candidate_map[event["event_id"]]
                event["schema_version"] = "2.0"
                event["split"] = split
                event["candidate_generation"] = "lr_only_hybrid_proposal"
                event["candidate_count"] = len(candidates)
                event["candidates"] = candidates
                event["correct_identity_available_in_window"] = bool(
                    any(candidate["label"] for candidate in candidates)
                )
                event["top1_correct"] = bool(candidates and candidates[0]["label"])
                event["has_positive_candidate"] = bool(
                    any(candidate["label"] for candidate in candidates)
                )
                prepared[split].append(event)

        train_sequences = sorted(split_config_v1["train_sequences"])
        test_sequences = sorted(split_config_v1["test_sequences"])
        validation_sequences = set(args.validation_sequences)
        learning_sequences = sorted(set(train_sequences) - validation_sequences)
        validation_events = [
            event for event in prepared["train"] if event["sequence"] in validation_sequences
        ]
        learning_events = [
            event for event in prepared["train"] if event["sequence"] in learning_sequences
        ]
        statistics = {
            "train": split_stats(learning_events, learning_sequences),
            "validation": split_stats(validation_events, sorted(validation_sequences)),
            "test": split_stats(prepared["test"], test_sequences),
            "all": split_stats(
                prepared["train"] + prepared["test"],
                sorted(set(train_sequences) | set(test_sequences)),
            ),
            "sequence_mining": sequence_audit,
        }
        leakage = leakage_audit(prepared["train"], prepared["test"], validation_sequences)
        if not leakage["passed"]:
            raise RuntimeError(f"Leakage audit failed: {leakage}")

        write_split(temporary, "train", prepared["train"], train_sequences)
        write_split(temporary, "test", prepared["test"], test_sequences)
        config = {
            "dataset_name": "hive_lr_history_ranking",
            "dataset_version": "2.0",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "seed": args.seed,
            "split_unit": "sequence",
            "train_sequences": train_sequences,
            "validation_sequences": sorted(validation_sequences),
            "learning_sequences": learning_sequences,
            "test_sequences": test_sequences,
            "window_frames": args.window_frames,
            "candidate_window": "T-window_frames through T-1",
            "max_candidates": args.max_candidates,
            "candidate_routes": {
                "appearance_hard": args.appearance_candidates,
                "recent": args.recent_candidates,
                "temporal_diverse": args.diverse_candidates,
            },
            "min_spacing_frames": args.min_spacing_frames,
            "evaluation_iou": args.evaluation_iou,
            "lowres_width": args.lowres_width,
            "lowres_height": args.lowres_height,
            "candidate_generation": "LR tracker + LR pixels only; GT joined after selection",
            "source_files": {
                "v1_dataset": str(args.input_root),
                "v1_split_config_sha256": sha256(args.input_root / "split_config.json"),
                "lowres_predictions": "code/outputs/all_sequences_reid_512x120/predictions",
                "original_frames": "QuadTrack/train",
            },
        }
        (temporary / "split_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "statistics.json").write_text(
            json.dumps(statistics, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "leakage_audit.json").write_text(
            json.dumps(leakage, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "DATASET_REPORT.md").write_text(
            report_text(config, statistics, leakage), encoding="utf-8"
        )
        # tempfile.mkdtemp creates mode 0700. The dataset lives in the shared
        # workspace and must remain readable from both host and container.
        temporary.chmod(0o755)
        temporary.rename(args.output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps(statistics, indent=2))
    print(f"Dataset: {args.output_root}")
    print(f"Report:  {args.output_root / 'DATASET_REPORT.md'}")


if __name__ == "__main__":
    main()
