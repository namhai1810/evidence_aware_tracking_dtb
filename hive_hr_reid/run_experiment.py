#!/usr/bin/env python3
"""Run HIVE v2 + one-shot HR ReID identity correction on QuadTrack.

Protocol
--------
* HIVE v2 is loaded from its frozen checkpoint; it is never trained here.
* The correction policy never reads GT. GT is used after replay for validation
  threshold selection and final held-out-test measurement only.
* Nearest-frame and HIVE request exactly one 2048x480 historical crop for every
  event with candidates. Both use the same ReID encoder and shared thresholds.
* LR-only, nearest+HR-ReID, and HIVE+HR-ReID are replayed from the same MOT files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import OrderedDict, Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parent
HIVE_DIR = CODE_DIR / "hr_finding"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(HIVE_DIR))

from datasets import HIVERankingDataset, collate_event_batch  # noqa: E402
from evaluate_mot import evaluate, load_mot  # noqa: E402
from evaluate_quadtrack_dataset import aggregate_metrics, collect_failure_frames  # noqa: E402
from models import HIVE  # noqa: E402
from retrieve_highres_with_reid_embeddings import normalized  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402

from hive_hr_reid.hota import combine_hota_raw, evaluate_hota_raw  # noqa: E402
from hive_hr_reid.replay import (  # noqa: E402
    annotate_operation_outcomes,
    replay_identity_merges,
)


DEFAULT_DATASET = PROJECT_ROOT / "QuadTrack" / "train"
DEFAULT_HIVE_DATASET = CODE_DIR / "outputs" / "hive_history_ranking_v2"
DEFAULT_CHECKPOINT = HIVE_DIR / "runs" / "hive_history_v2_seed42" / "checkpoints" / "best.pt"
DEFAULT_LR_PREDICTIONS = CODE_DIR / "outputs" / "all_sequences_reid_512x120" / "predictions"
DEFAULT_REID = CODE_DIR / "yolo26n-reid.onnx"
DEFAULT_OUTPUT = CODE_DIR / "outputs" / "hive_v2_hr_reid_replay"
METHODS = ("lr_only", "nearest_hr_reid", "hive_v2_hr_reid")
INPUT_KEYS = (
    "target_crop",
    "old_crops",
    "old_mask",
    "candidate_crops",
    "candidate_mask",
    "metadata",
    "competitor_crops",
    "competitor_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--hive-dataset-root", type=Path, default=DEFAULT_HIVE_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--lr-pred-dir", type=Path, default=DEFAULT_LR_PREDICTIONS)
    parser.add_argument(
        "--lr-pred-pattern", default="{sequence}_yolo11l_botsort_reid_pre512x120.txt"
    )
    parser.add_argument("--reid-model", type=Path, default=DEFAULT_REID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--lowres-width", type=int, default=512)
    parser.add_argument("--lowres-height", type=int, default=120)
    parser.add_argument("--competitor-history-frames", type=int, default=30)
    parser.add_argument("--competitor-samples", type=int, default=3)
    parser.add_argument("--max-competitors", type=int, default=12)
    parser.add_argument("--max-segment-gap", type=int, default=30)
    parser.add_argument("--visual-examples", type=int, default=6)
    parser.add_argument(
        "--max-events-per-split",
        type=int,
        default=None,
        help="Smoke-test mode only; keep the first N events in validation and test.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path, description in (
        (args.dataset_root, "QuadTrack dataset"),
        (args.hive_dataset_root, "HIVE v2 event dataset"),
        (args.checkpoint, "frozen HIVE v2 checkpoint"),
        (args.lr_pred_dir, "LR prediction directory"),
        (args.reid_model, "ReID model"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path.resolve()}")
    if "{sequence}" not in args.lr_pred_pattern:
        raise ValueError("--lr-pred-pattern must contain {sequence}")
    if not 0 < args.iou <= 1:
        raise ValueError("--iou must be in (0, 1]")
    if args.max_events_per_split is not None and args.max_events_per_split <= 0:
        raise ValueError("--max-events-per-split must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")


def prepare_output(path: Path, overwrite: bool) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Pass --overwrite or choose another --output-dir."
        )
    path.mkdir(parents=True, exist_ok=True)
    for name in ("predictions", "events", "visuals"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_hive_dataset(
    args: argparse.Namespace,
    config: dict,
    sequences: list[str],
    manifest: str,
) -> HIVERankingDataset:
    data_config = dict(config["data"])
    data_config["teacher_margin_field"] = None
    data_config["include_competitor_history"] = True
    dataset = HIVERankingDataset(
        args.hive_dataset_root.resolve(),
        manifest,
        PROJECT_ROOT.resolve(),
        data_config,
        include_sequences=sequences,
    )
    if args.max_events_per_split is not None:
        dataset.events = dataset.events[: args.max_events_per_split]
    return dataset


@torch.no_grad()
def select_frames_with_hive(
    model: HIVE,
    dataset: HIVERankingDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict], dict[str, dict]]:
    """Return model selections. Labels are removed before the model call."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_event_batch,
    )
    event_lookup = {event["event_id"]: event for event in dataset.events}
    selections: dict[str, dict] = {}
    model.eval()
    for raw_batch in loader:
        # Explicit allowlist is the no-GT decision boundary. labels, teacher
        # targets, gt_id and candidate labels never cross into model.forward().
        model_batch = to_device({key: raw_batch[key] for key in INPUT_KEYS}, device)
        logits = model(model_batch).detach().float().cpu()
        masks = raw_batch["candidate_mask"].bool()
        for row, event_id in enumerate(raw_batch["event_id"]):
            valid = torch.nonzero(masks[row], as_tuple=False).flatten()
            if len(valid):
                slot = int(valid[torch.argmax(logits[row, valid])])
                probability = float(torch.sigmoid(logits[row, slot]))
                candidate = event_lookup[event_id]["candidates"][slot]
                selections[event_id] = {
                    "candidate_id": candidate["candidate_id"],
                    "candidate_slot": slot,
                    "candidate_frame": int(candidate["candidate_frame"]),
                    "utility_probability": probability,
                }
            else:
                selections[event_id] = {}
    return dataset.events, selections


def nearest_selection(event: dict) -> dict:
    candidates = event.get("candidates", [])
    if not candidates:
        return {}
    slot, candidate = max(
        enumerate(candidates),
        key=lambda pair: (int(pair[1]["candidate_frame"]), -int(pair[0])),
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_slot": slot,
        "candidate_frame": int(candidate["candidate_frame"]),
        "utility_probability": None,
    }


class ImageCache:
    def __init__(self, max_items: int = 48) -> None:
        self.max_items = max_items
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def read(self, path: Path) -> np.ndarray:
        key = str(path.resolve())
        if key in self.cache:
            image = self.cache.pop(key)
            self.cache[key] = image
            return image
        image = cv2.imread(key)
        if image is None:
            raise FileNotFoundError(f"Could not decode image: {path}")
        self.cache[key] = image
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return image


class OpenCVONNXReID:
    """Small ONNX ReID runner with an ONNX Runtime compatibility path.

    It matches Ultralytics' exported-model preprocessing: BGR-to-RGB, bilinear
    resize to 224x224, float32 scaling to [0, 1], and BCHW layout. OpenCV DNN
    was the original backend.  Recent OpenCV builds cannot execute the dynamic
    attention ``Reshape`` nodes in this model, so ONNX Runtime is preferred
    when installed and OpenCV remains the dependency-free fallback.
    """

    def __init__(self, model: Path, device: torch.device, image_size: int = 224) -> None:
        if model.suffix.lower() != ".onnx":
            raise ValueError("The dependency-free ReID runner currently expects an ONNX model")
        self.image_size = int(image_size)
        self.session = None
        self.input_name = None
        self.output_names = None
        try:
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            available = set(ort.get_available_providers())
            if device.type == "cuda" and "CUDAExecutionProvider" in available:
                providers.insert(0, "CUDAExecutionProvider")
            self.session = ort.InferenceSession(str(model.resolve()), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.backend = f"onnxruntime_{self.session.get_providers()[0]}"
            self.net = None
        except ImportError:
            self.net = cv2.dnn.readNetFromONNX(str(model.resolve()))
            self.backend = "opencv_dnn_cpu"
            if device.type == "cuda":
                # Many opencv-python wheels are CPU-only. Select CUDA only when
                # the runtime exposes a CUDA device.
                try:
                    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                        self.backend = "opencv_dnn_cuda"
                except (AttributeError, cv2.error):
                    pass

    def encode(self, crops: list[np.ndarray], batch_size: int = 128) -> list[np.ndarray | None]:
        valid = [crop is not None and bool(crop.size) for crop in crops]
        valid_crops = [crop for crop, keep in zip(crops, valid) if keep]
        features: list[np.ndarray] = []
        for start in range(0, len(valid_crops), batch_size):
            group = valid_crops[start : start + batch_size]
            tensors = []
            for crop in group:
                resized = cv2.resize(
                    crop,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                tensors.append(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
            inputs = np.stack(tensors, axis=0)
            if self.session is not None:
                output = self.session.run(self.output_names, {self.input_name: inputs})[0]
            else:
                self.net.setInput(inputs)
                output = self.net.forward()
            output = np.asarray(output, dtype=np.float32).reshape(len(group), -1)
            features.extend(normalized(vector) for vector in output)
        iterator = iter(features)
        return [next(iterator) if keep else None for keep in valid]


def run_crop_batches(encoder: OpenCVONNXReID, crops: list[np.ndarray]) -> list[np.ndarray | None]:
    return encoder.encode(crops)


def clipped_crop(image: np.ndarray, bbox: list[float] | np.ndarray, pad_fraction: float = 0.05) -> np.ndarray:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    width, height = x2 - x1, y2 - y1
    x1 -= width * pad_fraction
    x2 += width * pad_fraction
    y1 -= height * pad_fraction
    y2 += height * pad_fraction
    h, w = image.shape[:2]
    left = max(0, min(w, int(math.floor(x1))))
    top = max(0, min(h, int(math.floor(y1))))
    right = max(0, min(w, int(math.ceil(x2))))
    bottom = max(0, min(h, int(math.ceil(y2))))
    if right <= left or bottom <= top:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return image[top:bottom, left:right].copy()


def source_path(relative: str | Path) -> Path:
    path = Path(relative)
    path = path if path.is_absolute() else PROJECT_ROOT / path
    if path.exists():
        return path
    path = path.resolve(strict=False)
    value = str(path)
    for source, target in (
        ("/workspace/", "/home/dtb/workspace/"),
        ("/home/dtb/workspace/", "/workspace/"),
    ):
        if value.startswith(source):
            candidate = Path(target + value[len(source) :])
            if candidate.exists():
                return candidate
    return path


def hr_candidate_crop(cache: ImageCache, candidate: dict) -> np.ndarray:
    image = cache.read(source_path(candidate["source_image"]))
    return clipped_crop(image, candidate["candidate_bbox_xyxy"])


def lowres_track_crop(
    cache: ImageCache,
    dataset_root: Path,
    sequence: str,
    frame: int,
    bbox: np.ndarray,
    lowres_size: tuple[int, int],
) -> np.ndarray:
    image = cache.read(source_path(dataset_root / sequence / "img1" / f"{frame:06d}.jpg"))
    resized = cv2.resize(image, lowres_size, interpolation=cv2.INTER_AREA)
    scale_x = lowres_size[0] / image.shape[1]
    scale_y = lowres_size[1] / image.shape[0]
    scaled = np.asarray(bbox, dtype=float).copy()
    scaled[[0, 2]] *= scale_x
    scaled[[1, 3]] *= scale_y
    return clipped_crop(resized, scaled, pad_fraction=0.05)


def choose_history_rows(
    observations: list[tuple[int, tuple]],
    start: int,
    end: int,
    samples: int,
) -> list[tuple[int, tuple]]:
    eligible = [item for item in observations if start <= item[0] <= end]
    eligible.sort(
        key=lambda item: (
            float(item[1][2])
            * math.sqrt(max(1.0, (item[1][1][2] - item[1][1][0]) * (item[1][1][3] - item[1][1][1]))),
            item[0],
        ),
        reverse=True,
    )
    selected: list[tuple[int, tuple]] = []
    for item in eligible:
        if all(abs(item[0] - previous[0]) >= 2 for previous in selected):
            selected.append(item)
            if len(selected) == samples:
                break
    return sorted(selected)


def competitor_histories(
    frames: dict[int, list[tuple]],
    event: dict,
    history_frames: int,
    samples: int,
    max_competitors: int,
) -> list[tuple[int, list[tuple[int, tuple]]]]:
    target = int(event["target_frame"])
    old_id = int(event["lr_old_pred_id"])
    new_id = int(event["lr_new_pred_id"])
    observations: dict[int, list[tuple[int, tuple]]] = defaultdict(list)
    for frame in range(max(1, target - history_frames), target):
        for row in frames.get(frame, []):
            track_id = int(row[0])
            if track_id != old_id:
                observations[track_id].append((frame, row))
    ordered_ids = sorted(
        observations,
        key=lambda track_id: (
            track_id == new_id,
            max(frame for frame, _ in observations[track_id]),
            len(observations[track_id]),
        ),
        reverse=True,
    )[:max_competitors]
    return [
        (
            track_id,
            choose_history_rows(
                observations[track_id], max(1, target - history_frames), target - 1, samples
            ),
        )
        for track_id in ordered_ids
    ]


def feature_mean(features: list[np.ndarray | None]) -> np.ndarray | None:
    available = [feature for feature in features if feature is not None]
    return normalized(np.mean(available, axis=0)) if available else None


def cosine(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    return float(np.dot(left, right))


def event_reid_scores(
    args: argparse.Namespace,
    encoder: OpenCVONNXReID,
    cache: ImageCache,
    event: dict,
    selections: dict[str, dict],
    lr_frames: dict[int, list[tuple]],
) -> dict[str, dict]:
    """Embed LR context and one HR crop per method in one model batch."""
    crops: list[np.ndarray] = []
    groups: dict[str, list[int]] = defaultdict(list)

    def add(group: str, crop: np.ndarray) -> None:
        groups[group].append(len(crops))
        crops.append(crop)

    for query in event.get("query_crops", []):
        add("old", cv2.imread(str(args.hive_dataset_root / query["crop_path"])))
    if event.get("target_crop"):
        add("target", cv2.imread(str(args.hive_dataset_root / event["target_crop"]["crop_path"])))

    candidates = {candidate["candidate_id"]: candidate for candidate in event.get("candidates", [])}
    for method in ("nearest_hr_reid", "hive_v2_hr_reid"):
        selected = selections.get(method, {})
        candidate = candidates.get(selected.get("candidate_id"))
        if candidate is not None:
            add(f"hr:{method}", hr_candidate_crop(cache, candidate))

    competitors = competitor_histories(
        lr_frames,
        event,
        args.competitor_history_frames,
        args.competitor_samples,
        args.max_competitors,
    )
    for track_id, history in competitors:
        for frame, row in history:
            add(
                f"competitor:{track_id}",
                lowres_track_crop(
                    cache,
                    args.dataset_root,
                    event["sequence"],
                    frame,
                    row[1],
                    (args.lowres_width, args.lowres_height),
                ),
            )

    # cv2.imread returns None on malformed paths; preserve an explicit empty crop
    # so the encoder helper reports a missing embedding instead of crashing.
    crops = [crop if crop is not None else np.empty((0, 0, 3), dtype=np.uint8) for crop in crops]
    features = run_crop_batches(encoder, crops)
    old = feature_mean([features[index] for index in groups.get("old", [])])
    target = feature_mean([features[index] for index in groups.get("target", [])])
    competitor_vectors = {
        int(group.split(":", 1)[1]): feature_mean([features[index] for index in indices])
        for group, indices in groups.items()
        if group.startswith("competitor:")
    }
    competitor_vectors = {key: value for key, value in competitor_vectors.items() if value is not None}

    old_target = cosine(old, target)
    method_scores: dict[str, dict] = {}
    for method in ("nearest_hr_reid", "hive_v2_hr_reid"):
        hr = feature_mean([features[index] for index in groups.get(f"hr:{method}", [])])
        hr_old = cosine(hr, old)
        hr_target = cosine(hr, target)
        evidence_competitors = {
            str(track_id): float(np.dot(hr, vector))
            for track_id, vector in competitor_vectors.items()
            if hr is not None
        }
        target_competitors = {
            str(track_id): float(np.dot(target, vector))
            for track_id, vector in competitor_vectors.items()
            if target is not None
        }
        best_evidence_comp = max(evidence_competitors.values(), default=-1.0)
        best_target_comp = max(target_competitors.values(), default=-1.0)
        bridge = min(hr_old, hr_target) if hr_old is not None and hr_target is not None else None
        method_scores[method] = {
            "embedding_available": bridge is not None and old_target is not None,
            "similarity_hr_old": hr_old,
            "similarity_hr_target": hr_target,
            "similarity_old_target": old_target,
            "best_competitor_similarity_to_hr": best_evidence_comp,
            "best_competitor_similarity_to_target": best_target_comp,
            "evidence_bridge": bridge,
            "evidence_margin": bridge - best_evidence_comp if bridge is not None else None,
            "target_margin": old_target - best_target_comp if old_target is not None else None,
            "competitor_similarity_to_hr": evidence_competitors,
            "competitor_similarity_to_target": target_competitors,
            "competitor_count": len(competitor_vectors),
        }
    return method_scores


def build_split_evidence(
    args: argparse.Namespace,
    encoder: OpenCVONNXReID,
    cache: ImageCache,
    events: list[dict],
    hive_selections: dict[str, dict],
    prediction_cache: dict[str, dict],
) -> list[dict]:
    evidence = []
    total = len(events)
    for index, event in enumerate(events, start=1):
        sequence = event["sequence"]
        nearest = nearest_selection(event)
        selections = {
            "nearest_hr_reid": nearest,
            "hive_v2_hr_reid": hive_selections.get(event["event_id"], {}),
        }
        scores = event_reid_scores(
            args, encoder, cache, event, selections, prediction_cache[sequence]
        )
        methods = {}
        candidates = {candidate["candidate_id"]: candidate for candidate in event["candidates"]}
        for method, selection in selections.items():
            selected_candidate = candidates.get(selection.get("candidate_id"))
            methods[method] = {
                **selection,
                **scores[method],
                # This label is retained only for evaluation/reporting and is
                # never read by decide_merge().
                "evaluation_selected_hr_correct": (
                    bool(int(selected_candidate["label"])) if selected_candidate is not None else None
                ),
                "selected_bbox_xyxy": (
                    selected_candidate.get("candidate_bbox_xyxy") if selected_candidate else None
                ),
                "selected_source_image": (
                    selected_candidate.get("source_image") if selected_candidate else None
                ),
            }
        evidence.append(
            {
                "event_id": event["event_id"],
                "sequence": sequence,
                "target_frame": int(event["target_frame"]),
                "gt_id": int(event["gt_id"]),
                "lr_old_pred_id": int(event["lr_old_pred_id"]),
                "lr_new_pred_id": int(event["lr_new_pred_id"]),
                "lr_last_match_frame": int(event["lr_last_match_frame"]),
                "target_crop": event.get("target_crop"),
                "query_crops": event.get("query_crops", []),
                "methods": methods,
            }
        )
        if index == 1 or index % 10 == 0 or index == total:
            print(f"  ReID evidence {index}/{total}", flush=True)
    return evidence


def decide_merge(scores: dict, thresholds: dict[str, float]) -> tuple[bool, str]:
    """No-GT correction policy used identically on validation and test."""
    if not scores.get("embedding_available", False):
        return False, "missing_embedding"
    checks = (
        (scores["evidence_bridge"] >= thresholds["bridge_min"], "bridge_below_threshold"),
        (scores["similarity_old_target"] >= thresholds["old_target_min"], "old_target_below_threshold"),
        (scores["evidence_margin"] >= thresholds["evidence_margin_min"], "competitor_evidence_wins"),
        (scores["target_margin"] >= thresholds["target_margin_min"], "competitor_target_wins"),
    )
    for passed, reason in checks:
        if not passed:
            return False, reason
    return True, "reid_merge"


def decisions_for_method(
    evidence: list[dict], method: str, thresholds: dict[str, float]
) -> list[dict]:
    decisions = []
    for event in evidence:
        scores = event["methods"][method]
        merge, reason = decide_merge(scores, thresholds)
        decisions.append(
            {
                "event_id": event["event_id"],
                "sequence": event["sequence"],
                "target_frame": event["target_frame"],
                "gt_id": event["gt_id"],
                "lr_old_pred_id": event["lr_old_pred_id"],
                "lr_new_pred_id": event["lr_new_pred_id"],
                "lr_last_match_frame": event["lr_last_match_frame"],
                "merge": merge,
                "decision_reason": reason,
                "selected_candidate_id": scores.get("candidate_id"),
                "selected_frame": scores.get("candidate_frame"),
                "scores": scores,
            }
        )
    return decisions


def replay_split(
    args: argparse.Namespace,
    evidence: list[dict],
    method: str,
    thresholds: dict[str, float],
    prediction_cache: dict[str, dict],
) -> tuple[dict[str, dict], list[dict], list[dict]]:
    decisions = decisions_for_method(evidence, method, thresholds)
    decisions_by_sequence: dict[str, list[dict]] = defaultdict(list)
    for decision in decisions:
        decisions_by_sequence[decision["sequence"]].append(decision)
    replayed: dict[str, dict] = {}
    operations = []
    for sequence in sorted({row["sequence"] for row in evidence}):
        frames, sequence_operations = replay_identity_merges(
            prediction_cache[sequence],
            decisions_by_sequence.get(sequence, []),
            args.max_segment_gap,
        )
        replayed[sequence] = frames
        for operation in sequence_operations:
            operation["sequence"] = sequence
        operations.extend(sequence_operations)
    return replayed, decisions, operations


def basic_metrics(
    args: argparse.Namespace,
    sequences: list[str],
    predictions: dict[str, dict],
    gt_cache: dict[str, dict],
    include_hota: bool,
) -> dict:
    per_sequence = {
        sequence: evaluate(gt_cache[sequence], predictions[sequence], args.iou)
        for sequence in sequences
    }
    aggregate = aggregate_metrics(per_sequence)
    if include_hota:
        hota_rows = []
        for sequence in sequences:
            raw = evaluate_hota_raw(gt_cache[sequence], predictions[sequence])
            per_sequence[sequence].update(combine_hota_raw([raw]))
            hota_rows.append(raw)
        aggregate.update(combine_hota_raw(hota_rows))
    return {"aggregate": aggregate, "sequences": per_sequence}


def grid_values() -> list[dict[str, float]]:
    rows = []
    for bridge in (0.35, 0.50, 0.65, 0.80):
        for old_target in (0.30, 0.50, 0.70):
            for evidence_margin in (-0.10, 0.00, 0.10):
                for target_margin in (-0.10, 0.00, 0.10):
                    rows.append(
                        {
                            "bridge_min": bridge,
                            "old_target_min": old_target,
                            "evidence_margin_min": evidence_margin,
                            "target_margin_min": target_margin,
                        }
                    )
    return rows


def tune_thresholds(
    args: argparse.Namespace,
    validation_evidence: list[dict],
    sequences: list[str],
    prediction_cache: dict[str, dict],
    gt_cache: dict[str, dict],
) -> tuple[dict[str, float], list[dict]]:
    baseline = basic_metrics(
        args,
        sequences,
        {sequence: prediction_cache[sequence] for sequence in sequences},
        gt_cache,
        include_hota=False,
    )["aggregate"]
    search_rows = []
    for index, thresholds in enumerate(grid_values(), start=1):
        method_results = {}
        for method in ("nearest_hr_reid", "hive_v2_hr_reid"):
            replayed, decisions, operations = replay_split(
                args, validation_evidence, method, thresholds, prediction_cache
            )
            metrics = basic_metrics(
                args, sequences, replayed, gt_cache, include_hota=False
            )["aggregate"]
            applied = sum(operation["applied"] for operation in operations)
            selected_wrong = sum(
                operation["applied"]
                and not bool(decision["scores"].get("evaluation_selected_hr_correct"))
                for operation, decision in zip(operations, decisions)
            )
            method_results[method] = {
                "idf1": metrics["idf1"],
                "idsw": metrics["id_switches"],
                "applied": applied,
                "wrong_hr_evidence_applied": selected_wrong,
            }
        mean_idf1 = float(np.mean([row["idf1"] for row in method_results.values()]))
        mean_idsw = float(np.mean([row["idsw"] for row in method_results.values()]))
        applied = sum(row["applied"] for row in method_results.values())
        wrong_evidence = sum(row["wrong_hr_evidence_applied"] for row in method_results.values())
        idsw_reduction = (baseline["id_switches"] - mean_idsw) / max(1, baseline["id_switches"])
        wrong_rate = wrong_evidence / max(1, applied)
        # Shared validation objective: identity accuracy dominates, IDSW gives a
        # smaller reward, and accepted wrong HR evidence is explicitly penalized.
        objective = mean_idf1 + 0.10 * idsw_reduction - 0.05 * wrong_rate
        search_rows.append(
            {
                "thresholds": thresholds,
                "objective": objective,
                "mean_idf1": mean_idf1,
                "mean_idsw": mean_idsw,
                "validation_baseline_idf1": baseline["idf1"],
                "validation_baseline_idsw": baseline["id_switches"],
                "wrong_evidence_rate": wrong_rate,
                "methods": method_results,
            }
        )
        if index % 18 == 0 or index == 108:
            print(f"  threshold grid {index}/108", flush=True)
    best = max(
        search_rows,
        key=lambda row: (
            row["objective"],
            row["mean_idf1"],
            -row["mean_idsw"],
            -row["wrong_evidence_rate"],
            sum(row["thresholds"].values()),
        ),
    )
    return best["thresholds"], sorted(search_rows, key=lambda row: row["objective"], reverse=True)


def write_mot(path: Path, frames: dict[int, list[tuple]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for frame in sorted(frames):
            for track_id, box, confidence in frames[frame]:
                x1, y1, x2, y2 = (float(value) for value in box)
                writer.writerow(
                    [
                        frame,
                        int(track_id),
                        f"{x1:.3f}",
                        f"{y1:.3f}",
                        f"{x2 - x1:.3f}",
                        f"{y2 - y1:.3f}",
                        f"{float(confidence):.6f}",
                        -1,
                        -1,
                        -1,
                    ]
                )


def switch_points(failures: list[dict]) -> set[tuple[str, int, int]]:
    return {
        (row["sequence"], int(row["frame_id"]), int(switch["gt_id"]))
        for row in failures
        for switch in row["switches"]
    }


def final_evaluate_method(
    args: argparse.Namespace,
    output_dir: Path,
    method: str,
    test_evidence: list[dict],
    test_sequences: list[str],
    thresholds: dict[str, float],
    prediction_cache: dict[str, dict],
    gt_cache: dict[str, dict],
    baseline_switches: set[tuple[str, int, int]],
) -> tuple[dict, list[dict], dict[str, dict]]:
    if method == "lr_only":
        replayed = {sequence: prediction_cache[sequence] for sequence in test_sequences}
        decisions: list[dict] = []
        operations: list[dict] = []
    else:
        replayed, decisions, operations = replay_split(
            args, test_evidence, method, thresholds, prediction_cache
        )

    metric_payload = basic_metrics(args, test_sequences, replayed, gt_cache, include_hota=True)
    failures = []
    outcomes = []
    if method != "lr_only":
        events_by_sequence: dict[str, list[dict]] = defaultdict(list)
        operations_by_sequence: dict[str, list[dict]] = defaultdict(list)
        for event in test_evidence:
            events_by_sequence[event["sequence"]].append(event)
        for operation in operations:
            operations_by_sequence[operation["sequence"]].append(operation)
        for sequence in test_sequences:
            sequence_outcomes = annotate_operation_outcomes(
                events_by_sequence[sequence],
                operations_by_sequence[sequence],
                gt_cache[sequence],
                prediction_cache[sequence],
                replayed[sequence],
                args.iou,
            )
            outcomes.extend(sequence_outcomes)
        decision_map = {row["event_id"]: row for row in decisions}
        for outcome in outcomes:
            decision = decision_map[outcome["event_id"]]
            outcome["decision_reason"] = decision["decision_reason"]
            outcome["selected_frame"] = decision["selected_frame"]
            outcome["selected_candidate_id"] = decision["selected_candidate_id"]
            outcome["scores"] = decision["scores"]

    for sequence in test_sequences:
        failures.extend(
            collect_failure_frames(sequence, gt_cache[sequence], replayed[sequence], args.iou)
        )
        write_mot(
            output_dir / "predictions" / method / f"{sequence}.txt", replayed[sequence]
        )
    current_switches = switch_points(failures)
    fixed_points = baseline_switches - current_switches
    introduced_points = current_switches - baseline_switches
    applied = sum(row["applied"] for row in outcomes)
    fixed_events = sum(row["actually_fixed"] for row in outcomes)
    wrong_corrections = sum(row["wrong_correction"] for row in outcomes)
    selected_correct = sum(
        bool(row["scores"].get("evaluation_selected_hr_correct")) for row in outcomes
    )
    summary = {
        **metric_payload,
        "correction": {
            "events": len(test_evidence) if method != "lr_only" else 0,
            "hr_requests": sum(
                row["scores"].get("candidate_id") is not None for row in outcomes
            ),
            "policy_merge_requests": sum(row["requested_merge"] for row in outcomes),
            "applied_merges": applied,
            "collision_or_missing_rejections": sum(
                row["requested_merge"] and not row["applied"] for row in outcomes
            ),
            "selected_hr_identity_correct": selected_correct,
            "actual_event_switches_fixed": fixed_events,
            "wrong_corrections": wrong_corrections,
            "full_replay_switch_points_removed": len(fixed_points),
            "full_replay_switch_points_introduced": len(introduced_points),
            "fixed_switch_points": [list(value) for value in sorted(fixed_points)],
            "introduced_switch_points": [list(value) for value in sorted(introduced_points)],
        },
    }
    write_json(output_dir / "events" / f"{method}_metrics.json", summary)
    write_jsonl(output_dir / "events" / f"{method}_decisions.jsonl", outcomes)
    write_json(output_dir / "events" / f"{method}_failure_frames.json", failures)
    return summary, outcomes, replayed


def plot_metric_comparison(output_dir: Path, results: dict[str, dict]) -> Path:
    labels = ["LR-only", "Nearest + HR ReID", "HIVE v2 + HR ReID"]
    colors = ["#64748b", "#f59e0b", "#2563eb"]
    metrics = [
        ("id_switches", "ID switches (lower is better)", False),
        ("idf1", "IDF1 (higher is better)", True),
        ("hota", "HOTA (higher is better)", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, (key, title, fraction) in zip(axes, metrics):
        values = [results[method]["aggregate"][key] for method in METHODS]
        bars = axis.bar(labels, values, color=colors, width=0.65)
        axis.set_title(title, fontsize=11, weight="bold")
        axis.tick_params(axis="x", labelrotation=16)
        axis.grid(axis="y", alpha=0.25)
        if fraction:
            axis.set_ylim(0, max(1.0, max(values) * 1.18))
        for bar, value in zip(bars, values):
            text = f"{value:.3f}" if fraction else f"{int(value)}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                text,
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
            )
    fig.suptitle("Held-out QuadTrack test: tracking after identity replay", fontsize=14, weight="bold")
    fig.tight_layout()
    path = output_dir / "visuals" / "tracking_metrics.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_correction_outcomes(output_dir: Path, results: dict[str, dict]) -> Path:
    methods = ("nearest_hr_reid", "hive_v2_hr_reid")
    labels = ["Nearest + HR ReID", "HIVE v2 + HR ReID"]
    requested = [results[method]["correction"]["selected_hr_identity_correct"] for method in methods]
    fixed = [results[method]["correction"]["actual_event_switches_fixed"] for method in methods]
    wrong = [results[method]["correction"]["wrong_corrections"] for method in methods]
    introduced = [
        results[method]["correction"]["full_replay_switch_points_introduced"] for method in methods
    ]
    x = np.arange(len(methods))
    width = 0.19
    fig, axis = plt.subplots(figsize=(9, 5.2))
    for offset, values, label, color in (
        (-1.5, requested, "Correct HR evidence", "#38bdf8"),
        (-0.5, fixed, "Actually fixed after replay", "#22c55e"),
        (0.5, wrong, "Wrong corrections", "#ef4444"),
        (1.5, introduced, "New switch points", "#a855f7"),
    ):
        bars = axis.bar(x + offset * width, values, width, label=label, color=color)
        axis.bar_label(bars, padding=2, fontsize=9)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Events / switch points")
    axis.set_title("Frame retrieval is not treated as correction success", weight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, frameon=False)
    fig.tight_layout()
    path = output_dir / "visuals" / "correction_outcomes.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_threshold_search(output_dir: Path, rows: list[dict], selected: dict) -> Path:
    x = [row["mean_idsw"] for row in rows]
    y = [row["mean_idf1"] for row in rows]
    colors = [row["wrong_evidence_rate"] for row in rows]
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    scatter = axis.scatter(x, y, c=colors, cmap="magma_r", alpha=0.75, s=34)
    best = rows[0]
    axis.scatter(
        [best["mean_idsw"]],
        [best["mean_idf1"]],
        marker="*",
        s=260,
        c="#22c55e",
        edgecolors="black",
        label="Selected on validation",
        zorder=5,
    )
    axis.set_xlabel("Mean validation IDSW (lower is better)")
    axis.set_ylabel("Mean validation IDF1 (higher is better)")
    axis.set_title("Validation-only ReID threshold search", weight="bold")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("Accepted wrong-HR-evidence rate")
    threshold_text = "\n".join(f"{key}={value:.2f}" for key, value in selected.items())
    axis.text(
        0.02,
        0.02,
        threshold_text,
        transform=axis.transAxes,
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#94a3b8"},
    )
    fig.tight_layout()
    path = output_dir / "visuals" / "validation_threshold_search.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def fit_panel(image: np.ndarray, width: int = 320, height: int = 220) -> np.ndarray:
    if image is None or not image.size:
        return np.full((height, width, 3), 35, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def label_panel(panel: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    output = panel.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(output, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output, subtitle, (8, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 220, 255), 1, cv2.LINE_AA)
    return output


def render_evidence_examples(
    args: argparse.Namespace,
    output_dir: Path,
    method: str,
    evidence: list[dict],
    outcomes: list[dict],
) -> Path | None:
    if not outcomes:
        return None
    evidence_map = {row["event_id"]: row for row in evidence}
    fixed = [row for row in outcomes if row["actually_fixed"]]
    wrong = [row for row in outcomes if row["wrong_correction"]]
    rejected = [row for row in outcomes if row["requested_merge"] and not row["applied"]]
    selected = (fixed[: args.visual_examples] + wrong[: args.visual_examples] + rejected[:2])
    selected = selected[: max(args.visual_examples, 1)]
    if not selected:
        selected = outcomes[: args.visual_examples]
    rows = []
    for outcome in selected:
        event = evidence_map[outcome["event_id"]]
        method_row = event["methods"][method]
        query = event["query_crops"][-1] if event["query_crops"] else None
        old_crop = (
            cv2.imread(str(args.hive_dataset_root / query["crop_path"])) if query else None
        )
        target = event.get("target_crop")
        target_crop = (
            cv2.imread(str(args.hive_dataset_root / target["crop_path"])) if target else None
        )
        source = method_row.get("selected_source_image")
        bbox = method_row.get("selected_bbox_xyxy")
        hr_crop = None
        if source and bbox:
            image = cv2.imread(str(source_path(source)))
            if image is not None:
                hr_crop = clipped_crop(image, bbox)
        status = (
            "FIXED" if outcome["actually_fixed"] else "WRONG" if outcome["wrong_correction"] else outcome["reason"].upper()
        )
        bridge = method_row.get("evidence_bridge")
        margin = method_row.get("evidence_margin")
        row = np.hstack(
            [
                label_panel(fit_panel(old_crop), "Old LR history", f"event {outcome['event_id']}"),
                label_panel(fit_panel(target_crop), "Target LR", f"old -> new at T={event['target_frame']}"),
                label_panel(
                    fit_panel(hr_crop),
                    f"Selected HR evidence | {status}",
                    f"frame={method_row.get('candidate_frame')} bridge={bridge:.3f} margin={margin:.3f}"
                    if bridge is not None and margin is not None
                    else "embedding unavailable",
                ),
            ]
        )
        rows.append(row)
    contact_sheet = np.vstack(rows)
    path = output_dir / "visuals" / f"{method}_examples.jpg"
    cv2.imwrite(str(path), contact_sheet)
    return path


def percent_delta(current: float, baseline: float, *, lower_better: bool = False) -> float:
    if baseline == 0:
        return 0.0
    return ((baseline - current) if lower_better else (current - baseline)) / abs(baseline)


def report_lines(
    args: argparse.Namespace,
    output_dir: Path,
    validation_sequences: list[str],
    test_sequences: list[str],
    thresholds: dict[str, float],
    results: dict[str, dict],
    checkpoint_hash: str,
    test_event_count: int,
) -> list[str]:
    baseline = results["lr_only"]["aggregate"]
    names = {
        "lr_only": "LR-only",
        "nearest_hr_reid": "Nearest-frame + HR ReID",
        "hive_v2_hr_reid": "HIVE v2 + HR ReID",
    }
    lines = [
        "# HIVE v2 + HR ReID: sửa ID Switch trên QuadTrack",
        "",
        f"Generated: `{datetime.now().astimezone().isoformat(timespec='seconds')}`",
        "",
        "## Kết luận chính",
        "",
        "Bảng dưới đây là kết quả **sau khi replay toàn bộ tracking**, không phải accuracy chọn frame. HIVE đúng frame không tự động được tính là đã sửa ID; một event chỉ được tính `actually fixed` khi GT identity mang cùng output ID ở historical match và target frame sau replay.",
        "",
        "| Method | HR requests | IDSW ↓ | ΔIDSW | IDF1 ↑ | ΔIDF1 | HOTA ↑ | ΔHOTA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        aggregate = results[method]["aggregate"]
        requests = int(results[method]["correction"]["hr_requests"])
        lines.append(
            f"| {names[method]} | {requests} | {aggregate['id_switches']} | "
            f"{percent_delta(aggregate['id_switches'], baseline['id_switches'], lower_better=True):+.1%} | "
            f"{aggregate['idf1']:.4f} | {aggregate['idf1'] - baseline['idf1']:+.4f} | "
            f"{aggregate['hota']:.4f} | {aggregate['hota'] - baseline['hota']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "![Tracking metrics](visuals/tracking_metrics.png)",
            "",
            "## Retrieval và correction outcome",
            "",
            "| Method | HR đúng identity | Policy yêu cầu merge | Merge thực thi | IDSW event thực sự sửa | Sửa sai | Switch point mới tạo |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("nearest_hr_reid", "hive_v2_hr_reid"):
        correction = results[method]["correction"]
        outcomes_path = output_dir / "events" / f"{method}_decisions.jsonl"
        outcomes = [json.loads(line) for line in outcomes_path.read_text(encoding="utf-8").splitlines() if line]
        lines.append(
            f"| {names[method]} | {correction['selected_hr_identity_correct']} | "
            f"{sum(row['requested_merge'] for row in outcomes)} | {sum(row['applied'] for row in outcomes)} | "
            f"**{correction['actual_event_switches_fixed']}** | **{correction['wrong_corrections']}** | "
            f"{correction['full_replay_switch_points_introduced']} |"
        )
    lines.extend(
        [
            "",
            "`Sửa sai` nghĩa là merge đã được áp dụng nhưng không khôi phục continuity tại event, hoặc segment bị đổi ID có dominant GT identity khác identity của event. Chỉ số này được tính hậu nghiệm bằng GT; policy không nhìn thấy nó.",
            "",
            "![Correction outcomes](visuals/correction_outcomes.png)",
            "",
            "## Protocol và chống leakage",
            "",
            f"- HIVE checkpoint giữ nguyên, không train/fine-tune: `{args.checkpoint.resolve()}`.",
            f"- SHA-256 checkpoint: `{checkpoint_hash}`.",
            f"- Validation sequences để tune threshold: `{', '.join(validation_sequences)}`.",
            f"- Held-out test sequences chỉ đánh giá một lần: `{', '.join(test_sequences)}`.",
            "- HIVE input allowlist chỉ gồm LR target/history/candidates/competitors và temporal metadata. `gt_id`, `label`, `matched_gt_id` không được truyền vào model hay hàm quyết định merge.",
            "- GT được dùng ở hai nơi: (1) chọn shared thresholds trên validation; (2) đo IDSW/IDF1/HOTA và outcome trên test sau khi mọi quyết định đã đóng băng.",
            "- Event timestamps/old-new pair đến từ evaluator-created ID-switch manifest. Vì vậy đây là thí nghiệm correction với oracle switch trigger, chưa phải bộ phát hiện switch online hoàn chỉnh.",
            f"- Cả Nearest và HIVE dùng đúng một HR crop từ source panorama `2048×480` cho mỗi event ({test_event_count} requests/method). Target, old history và competitors vẫn là LR `512×120`.",
            "- ReID so HR evidence với old identity prototype, target và tối đa 12 competing identity prototypes. Merge bị từ chối nếu ReID không qua threshold hoặc nếu replay tạo duplicate ID trong cùng frame.",
            "",
            "## Threshold đã tune trên validation",
            "",
            "| Parameter | Value |",
            "|---|---:|",
        ]
    )
    for key, value in thresholds.items():
        lines.append(f"| `{key}` | {value:.2f} |")
    lines.extend(
        [
            "",
            "Cùng một bộ threshold được dùng cho Nearest và HIVE để tách ảnh hưởng của frame selector khỏi decision policy. Objective validation kết hợp mean IDF1, IDSW reduction và penalty khi policy chấp nhận HR evidence sai identity.",
            "",
            "![Validation threshold search](visuals/validation_threshold_search.png)",
            "",
            "## Ví dụ trực quan",
            "",
            "### Nearest-frame + HR ReID",
            "",
            "![Nearest examples](visuals/nearest_hr_reid_examples.jpg)",
            "",
            "### HIVE v2 + HR ReID",
            "",
            "![HIVE examples](visuals/hive_v2_hr_reid_examples.jpg)",
            "",
            "Mỗi hàng lần lượt là old LR history, target LR và HR crop được request. Nhãn `FIXED/WRONG` chỉ được thêm sau replay để phân tích.",
            "",
            "## Per-sequence held-out test",
            "",
            "| Sequence | Method | IDSW | IDF1 | MOTA |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for sequence in test_sequences:
        for method in METHODS:
            row = results[method]["sequences"][sequence]
            lines.append(
                f"| {sequence} | {names[method]} | {row['id_switches']} | {row['idf1']:.4f} | {row['mota']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `summary.json`: toàn bộ config, split, threshold và aggregate results.",
            "- `events/*_decisions.jsonl`: score, quyết định, replay outcome cho từng event.",
            "- `events/*_metrics.json`: IDSW/IDF1/HOTA và correction counts.",
            "- `events/*_failure_frames.json`: evaluator-identical FN/FP/IDSW logs.",
            "- `predictions/<method>/<sequence>.txt`: MOT result sau replay.",
            "- `visuals/`: biểu đồ và evidence contact sheets trong report.",
            "",
        ]
    )
    return lines


def main() -> int:
    args = parse_args()
    validate_args(args)
    output_dir = prepare_output(args.output_dir, args.overwrite)
    device = torch.device(args.device)
    print(f"Output: {output_dir}", flush=True)
    print(f"Device: {device}", flush=True)

    checkpoint_state = load_checkpoint(args.checkpoint.resolve(), "cpu")
    config = checkpoint_state["config"]
    split_config = json.loads(
        (args.hive_dataset_root / "split_config.json").read_text(encoding="utf-8")
    )
    validation_sequences = [str(value) for value in config["data"]["validation_sequences"]]
    test_sequences = [str(value) for value in split_config["test_sequences"]]
    validation_dataset = build_hive_dataset(
        args, config, validation_sequences, config["data"]["train_manifest"]
    )
    test_dataset = build_hive_dataset(
        args, config, test_sequences, config["data"]["test_manifest"]
    )
    validation_sequences = sorted({event["sequence"] for event in validation_dataset.events})
    test_sequences = sorted({event["sequence"] for event in test_dataset.events})
    if not validation_sequences or not test_sequences:
        raise RuntimeError("Validation or test event split is empty")

    model = HIVE(config["model"], metadata_dim=validation_dataset.metadata_dim).to(device)
    model.load_state_dict(checkpoint_state["model"], strict=True)
    print("Selecting validation frames with frozen HIVE v2...", flush=True)
    validation_events, validation_hive = select_frames_with_hive(
        model, validation_dataset, device, args.batch_size
    )
    print("Selecting held-out test frames with frozen HIVE v2...", flush=True)
    test_events, test_hive = select_frames_with_hive(model, test_dataset, device, args.batch_size)

    sequences = sorted(set(validation_sequences) | set(test_sequences))
    prediction_cache = {}
    gt_cache = {}
    for sequence in sequences:
        prediction_cache[sequence] = load_mot(
            args.lr_pred_dir / args.lr_pred_pattern.format(sequence=sequence), ground_truth=False
        )
        gt_cache[sequence] = load_mot(
            args.dataset_root / sequence / "gt" / "gt.txt", ground_truth=True
        )

    print(f"Loading ReID encoder: {args.reid_model.name}", flush=True)
    encoder = OpenCVONNXReID(args.reid_model, device)
    print(f"ReID backend: {encoder.backend}", flush=True)
    image_cache = ImageCache()
    print("Building validation ReID evidence (GT is not read by the policy)...", flush=True)
    validation_evidence = build_split_evidence(
        args, encoder, image_cache, validation_events, validation_hive, prediction_cache
    )
    print("Building held-out test ReID evidence (decisions remain GT-free)...", flush=True)
    test_evidence = build_split_evidence(
        args, encoder, image_cache, test_events, test_hive, prediction_cache
    )
    test_hr_budgets = {
        method: sum(
            bool(event["methods"][method].get("candidate_id")) for event in test_evidence
        )
        for method in ("nearest_hr_reid", "hive_v2_hr_reid")
    }
    if len(set(test_hr_budgets.values())) != 1:
        raise RuntimeError(f"Unequal held-out HR budgets: {test_hr_budgets}")
    write_jsonl(output_dir / "events" / "validation_evidence.jsonl", validation_evidence)
    write_jsonl(output_dir / "events" / "test_evidence.jsonl", test_evidence)

    print("Tuning one shared ReID policy on validation only...", flush=True)
    thresholds, tuning_rows = tune_thresholds(
        args, validation_evidence, validation_sequences, prediction_cache, gt_cache
    )
    write_json(
        output_dir / "events" / "validation_threshold_search.json",
        {"selected_thresholds": thresholds, "rows": tuning_rows},
    )
    print(f"Frozen thresholds: {thresholds}", flush=True)

    baseline_failures = []
    for sequence in test_sequences:
        baseline_failures.extend(
            collect_failure_frames(
                sequence, gt_cache[sequence], prediction_cache[sequence], args.iou
            )
        )
    baseline_switches = switch_points(baseline_failures)

    results: dict[str, dict] = {}
    outcomes: dict[str, list[dict]] = {}
    print("Replaying and evaluating held-out test...", flush=True)
    for method in METHODS:
        print(f"  {method}", flush=True)
        result, method_outcomes, _ = final_evaluate_method(
            args,
            output_dir,
            method,
            test_evidence,
            test_sequences,
            thresholds,
            prediction_cache,
            gt_cache,
            baseline_switches,
        )
        results[method] = result
        outcomes[method] = method_outcomes

    print("Rendering report visualizations...", flush=True)
    plot_metric_comparison(output_dir, results)
    plot_correction_outcomes(output_dir, results)
    plot_threshold_search(output_dir, tuning_rows, thresholds)
    for method in ("nearest_hr_reid", "hive_v2_hr_reid"):
        render_evidence_examples(args, output_dir, method, test_evidence, outcomes[method])

    checkpoint_hash = sha256(args.checkpoint)
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": {
            "hive_checkpoint": str(args.checkpoint.resolve()),
            "hive_checkpoint_sha256": checkpoint_hash,
            "hive_retrained": False,
            "reid_model": str(args.reid_model.resolve()),
            "validation_sequences": validation_sequences,
            "held_out_test_sequences": test_sequences,
            "validation_events": len(validation_evidence),
            "test_events": len(test_evidence),
            "held_out_hr_requests": test_hr_budgets,
            "hr_budget_equal": len(set(test_hr_budgets.values())) == 1,
            "hr_source_resolution": [2048, 480],
            "lr_context_resolution": [args.lowres_width, args.lowres_height],
            "gt_decision_features": [],
            "oracle_switch_trigger": True,
            "evaluation_iou": args.iou,
        },
        "selected_validation_thresholds": thresholds,
        "results": results,
    }
    write_json(output_dir / "summary.json", summary)
    report = output_dir / "REPORT.md"
    report.write_text(
        "\n".join(
            report_lines(
                args,
                output_dir,
                validation_sequences,
                test_sequences,
                thresholds,
                results,
                checkpoint_hash,
                len(test_evidence),
            )
        ),
        encoding="utf-8",
    )
    print("\nHeld-out result:")
    for method in METHODS:
        row = results[method]["aggregate"]
        print(
            f"  {method}: IDSW={row['id_switches']} IDF1={row['idf1']:.4f} HOTA={row['hota']:.4f}"
        )
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

