"""Frozen HIVE v2 + frozen ReID + Lightweight ECIA inference.

No GT field is read by this module.  The only operational output is a proposed
old/new ID merge, later protected by the existing replay collision check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch
import yaml

from hive_hr_reid import run_experiment as baseline

from .dataset import normalize_embedding
from .model import ECIA, count_trainable_parameters


class CropRequest(NamedTuple):
    key: str
    crop: np.ndarray


class DiskEmbeddingCache:
    """Content-addressed `.npy` cache for frozen ReID embeddings."""

    def __init__(self, root: Path, model_fingerprint: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_fingerprint = model_fingerprint

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(
            f"{self.model_fingerprint}|{key}".encode("utf-8")
        ).hexdigest()
        return self.root / digest[:2] / f"{digest}.npy"

    def encode(
        self,
        encoder: baseline.OpenCVONNXReID,
        requests: list[CropRequest],
    ) -> list[np.ndarray | None]:
        """Load cache hits and batch-encode only missing, non-empty crops."""

        results: list[np.ndarray | None] = [None] * len(requests)
        missing_indices = []
        missing_crops = []
        for index, request in enumerate(requests):
            path = self._path(request.key)
            if path.exists():
                results[index] = normalize_embedding(np.load(path, allow_pickle=False))
            elif request.crop is not None and request.crop.size:
                missing_indices.append(index)
                missing_crops.append(request.crop)

        encoded = encoder.encode(missing_crops) if missing_crops else []
        for index, embedding in zip(missing_indices, encoded):
            if embedding is None:
                continue
            embedding = normalize_embedding(embedding)
            results[index] = embedding
            path = self._path(requests[index].key)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, embedding, allow_pickle=False)
        return results


def crop_key(source: str, bbox: list[float] | np.ndarray, resolution: str) -> str:
    rounded = [round(float(value), 3) for value in bbox]
    return json.dumps(
        {"source": source, "bbox": rounded, "resolution": resolution},
        sort_keys=True,
        separators=(",", ":"),
    )


def mean_embedding(values: list[np.ndarray | None]) -> np.ndarray | None:
    available = [value for value in values if value is not None]
    if not available:
        return None
    return normalize_embedding(np.mean(available, axis=0))


def extract_event_embeddings(
    *,
    args: argparse.Namespace,
    event: dict,
    selection: dict,
    lr_frames: dict[int, list[tuple]],
    image_cache: baseline.ImageCache,
    embedding_cache: DiskEmbeddingCache,
    encoder: baseline.OpenCVONNXReID,
) -> dict | None:
    """Create the exact ECIA inputs while using at most one HR crop.

    Old/target/competitor crops follow the 512×120 baseline preprocessing.  HR
    is the single original-resolution candidate selected by frozen HIVE v2.
    Candidate GT labels present in an offline manifest are intentionally never
    accessed here.
    """

    requests: list[CropRequest] = []
    groups: dict[str, list[int]] = {"old": [], "target": [], "hr": []}

    def add(group: str, key: str, crop: np.ndarray) -> None:
        groups.setdefault(group, []).append(len(requests))
        requests.append(CropRequest(key, crop))

    # LR old prototype: up to five existing HIVE query crops.
    for query in event.get("query_crops", [])[:5]:
        path = args.hive_dataset_root / query["crop_path"]
        add(
            "old",
            crop_key(str(path), query["bbox_lr_clipped_xyxy"], "lr_512x120"),
            cv2.imread(str(path)),
        )

    # LR target crop at the switch event.
    target_record = event.get("target_crop")
    if target_record:
        path = args.hive_dataset_root / target_record["crop_path"]
        add(
            "target",
            crop_key(str(path), target_record["bbox_lr_clipped_xyxy"], "lr_512x120"),
            cv2.imread(str(path)),
        )

    # Exactly one HR crop: the historical candidate already selected by HIVE.
    candidate_by_id = {
        candidate["candidate_id"]: candidate
        for candidate in event.get("candidates", [])
    }
    candidate = candidate_by_id.get(selection.get("candidate_id"))
    if candidate is not None:
        source = baseline.source_path(candidate["source_image"])
        add(
            "hr",
            crop_key(
                str(source), candidate["candidate_bbox_xyxy"], "hr_source_2048x480"
            ),
            baseline.hr_candidate_crop(image_cache, candidate),
        )

    # Variable competitor prototypes are built from the same LR tracker history
    # as the baseline. Each prototype may aggregate up to three cached crops.
    competitors = baseline.competitor_histories(
        lr_frames,
        event,
        args.competitor_history_frames,
        args.competitor_samples,
        args.max_competitors,
    )
    competitor_track_ids = []
    for track_id, history in competitors:
        competitor_track_ids.append(int(track_id))
        group = f"competitor:{track_id}"
        for frame, row in history:
            source = args.dataset_root / event["sequence"] / "img1" / f"{frame:06d}.jpg"
            add(
                group,
                crop_key(str(source), row[1], "lr_512x120"),
                baseline.lowres_track_crop(
                    image_cache,
                    args.dataset_root,
                    event["sequence"],
                    frame,
                    row[1],
                    (args.lowres_width, args.lowres_height),
                ),
            )

    features = embedding_cache.encode(encoder, requests)
    old = mean_embedding([features[index] for index in groups["old"]])
    target = mean_embedding([features[index] for index in groups["target"]])
    hr = mean_embedding([features[index] for index in groups["hr"]])
    competitor_embeddings = []
    competitor_member_embeddings = []
    retained_track_ids = []
    for track_id in competitor_track_ids:
        members = [
            features[index]
            for index in groups.get(f"competitor:{track_id}", [])
            if features[index] is not None
        ]
        prototype = mean_embedding(members)
        if prototype is not None:
            competitor_embeddings.append(prototype)
            competitor_member_embeddings.append(members)
            retained_track_ids.append(track_id)

    # Old and target are mandatory. Missing HR is supported and forces ECIA to
    # base scores, but the normal HIVE protocol should provide exactly one.
    if old is None or target is None:
        return None
    embedding_dim = old.size
    return {
        "old": old,
        "target": target,
        "hr": hr if hr is not None else np.zeros(embedding_dim, dtype=np.float32),
        "hr_available": hr is not None,
        "competitors": (
            np.stack(competitor_embeddings).astype(np.float32)
            if competitor_embeddings
            else np.zeros((0, embedding_dim), dtype=np.float32)
        ),
        "competitor_track_ids": retained_track_ids,
        # Used only by the offline training-cache miner to hold one real LR
        # crop out of its prototype. Operational inference ignores this field.
        "competitor_member_embeddings": competitor_member_embeddings,
        "hr_crop_count": int(hr is not None),
    }


def tensorize_event(embeddings: dict, device: torch.device) -> dict[str, torch.Tensor]:
    competitors = embeddings["competitors"]
    # Keep one invalid padding slot when K=0 so reduction operations stay valid.
    if competitors.shape[0] == 0:
        competitors = np.zeros((1, embeddings["old"].size), dtype=np.float32)
        mask = np.zeros((1, 1), dtype=np.bool_)
    else:
        mask = np.ones((1, competitors.shape[0]), dtype=np.bool_)
    return {
        "old": torch.from_numpy(embeddings["old"]).unsqueeze(0).to(device),
        "target": torch.from_numpy(embeddings["target"]).unsqueeze(0).to(device),
        "hr": torch.from_numpy(embeddings["hr"]).unsqueeze(0).to(device),
        "competitors": torch.from_numpy(competitors).unsqueeze(0).to(device),
        "competitor_mask": torch.from_numpy(mask).to(device),
        "hr_available": torch.tensor([embeddings["hr_available"]], device=device),
    }


def recovery_decision(
    decision_logits: torch.Tensor,
    *,
    min_old_probability: float,
    min_old_margin: float,
) -> dict:
    """Turn learned Old/Competitor/Abstain logits into a GT-free merge proposal."""

    probabilities = torch.softmax(decision_logits.detach().float(), dim=-1)[0]
    predicted_class = int(probabilities.argmax())
    old_probability = float(probabilities[0])
    strongest_other = float(probabilities[1:].max())
    old_margin = old_probability - strongest_other
    merge = (
        predicted_class == 0
        and old_probability >= min_old_probability
        and old_margin >= min_old_margin
    )
    if predicted_class != 0:
        reason = "competitor_or_abstain_wins"
    elif old_probability < min_old_probability:
        reason = "old_probability_below_minimum"
    elif old_margin < min_old_margin:
        reason = "old_margin_below_minimum"
    else:
        reason = "ecia_merge"
    return {
        "merge": merge,
        "reason": reason,
        "predicted_class": predicted_class,
        "old_probability": old_probability,
        "old_margin": old_margin,
        "probabilities": probabilities.cpu().tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--ecia-checkpoint", type=Path, required=True)
    parser.add_argument("--hive-checkpoint", type=Path, default=baseline.DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=baseline.DEFAULT_DATASET)
    parser.add_argument("--hive-dataset-root", type=Path, default=baseline.DEFAULT_HIVE_DATASET)
    parser.add_argument("--reid-model", type=Path, default=baseline.DEFAULT_REID)
    parser.add_argument("--lr-pred-dir", type=Path, default=baseline.DEFAULT_LR_PREDICTIONS)
    parser.add_argument(
        "--lr-pred-pattern", default="{sequence}_yolo11l_botsort_reid_pre512x120.txt"
    )
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument(
        "--sequences",
        default=None,
        help="Optional comma-separated subset of the selected split.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds-json",
        type=Path,
        default=None,
        help="Validation-selected min_old_probability/min_old_margin JSON.",
    )
    parser.add_argument("--min-old-probability", type=float, default=None)
    parser.add_argument("--min-old-margin", type=float, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lowres-width", type=int, default=512)
    parser.add_argument("--lowres-height", type=int, default=120)
    parser.add_argument("--competitor-history-frames", type=int, default=30)
    parser.add_argument("--competitor-samples", type=int, default=3)
    parser.add_argument("--max-competitors", type=int, default=12)
    parser.add_argument("--max-events-per-split", type=int, default=None)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-segment-gap", type=int, default=30)
    parser.add_argument("--visual-examples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; pass --overwrite")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    inference_config = dict(config["inference"])
    if args.thresholds_json is not None:
        calibrated = json.loads(args.thresholds_json.read_text(encoding="utf-8"))
        selected = calibrated.get("selected_thresholds", calibrated)
        inference_config["min_old_probability"] = float(selected["min_old_probability"])
        inference_config["min_old_margin"] = float(selected["min_old_margin"])
    if args.min_old_probability is not None:
        inference_config["min_old_probability"] = args.min_old_probability
    if args.min_old_margin is not None:
        inference_config["min_old_margin"] = args.min_old_margin

    # Load ECIA only. HIVE/ReID checkpoints remain frozen and are used under
    # no-grad inference; there is no optimizer in this module.
    checkpoint = torch.load(args.ecia_checkpoint, map_location="cpu", weights_only=False)
    model = ECIA(**{
        key: config["model"][key]
        for key in (
            "embedding_dim",
            "projection_dim",
            "relation_hidden_dim",
            "event_hidden_dim",
            "parameter_limit",
        )
    }).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    if count_trainable_parameters(model) > int(config["model"]["parameter_limit"]):
        raise RuntimeError("Loaded ECIA exceeds configured parameter limit")

    hive_state = baseline.load_checkpoint(args.hive_checkpoint.resolve(), "cpu")
    hive_config = hive_state["config"]
    split_config = json.loads(
        (args.hive_dataset_root / "split_config.json").read_text(encoding="utf-8")
    )
    if args.split == "validation":
        sequences = list(hive_config["data"]["validation_sequences"])
        manifest = hive_config["data"]["train_manifest"]
    else:
        sequences = list(split_config["test_sequences"])
        manifest = hive_config["data"]["test_manifest"]
    if args.sequences:
        requested = [value.strip() for value in args.sequences.split(",") if value.strip()]
        outside = set(requested) - set(map(str, sequences))
        if outside:
            raise ValueError(
                f"Requested sequences are outside {args.split} split: {sorted(outside)}"
            )
        sequences = requested
    hive_dataset = baseline.build_hive_dataset(args, hive_config, sequences, manifest)
    hive = baseline.HIVE(hive_config["model"], metadata_dim=hive_dataset.metadata_dim).to(device)
    hive.load_state_dict(hive_state["model"], strict=True)
    hive.eval()
    events, selections = baseline.select_frames_with_hive(
        hive, hive_dataset, device, args.batch_size
    )

    prediction_cache = {
        sequence: baseline.load_mot(
            args.lr_pred_dir / args.lr_pred_pattern.format(sequence=sequence),
            ground_truth=False,
        )
        for sequence in sorted({event["sequence"] for event in events})
    }
    encoder = baseline.OpenCVONNXReID(args.reid_model, device)
    model_hash = baseline.sha256(args.reid_model)
    embedding_cache = DiskEmbeddingCache(
        Path(config["inference"]["embedding_cache_dir"]), model_hash
    )
    image_cache = baseline.ImageCache()
    decisions = []

    for event in events:
        selection = selections.get(event["event_id"], {})
        embeddings = extract_event_embeddings(
            args=args,
            event=event,
            selection=selection,
            lr_frames=prediction_cache[event["sequence"]],
            image_cache=image_cache,
            embedding_cache=embedding_cache,
            encoder=encoder,
        )
        common = {
            "event_id": event["event_id"],
            "sequence": event["sequence"],
            "target_frame": int(event["target_frame"]),
            "lr_old_pred_id": int(event["lr_old_pred_id"]),
            "lr_new_pred_id": int(event["lr_new_pred_id"]),
            "lr_last_match_frame": int(event["lr_last_match_frame"]),
            "selected_candidate_id": selection.get("candidate_id"),
            "selected_frame": selection.get("candidate_frame"),
        }
        if embeddings is None:
            decisions.append({**common, "merge": False, "reason": "missing_old_or_target"})
            continue
        if embeddings["hr_crop_count"] > int(config["inference"]["max_hr_crops_per_event"]):
            raise RuntimeError("One-HR-crop budget violated")

        tensors = tensorize_event(embeddings, device)
        with torch.no_grad():
            output = model(**tensors, ablation=checkpoint.get("ablation", "full"))
        decision = recovery_decision(
            output.decision_logits,
            min_old_probability=float(inference_config["min_old_probability"]),
            min_old_margin=float(inference_config["min_old_margin"]),
        )
        predicted = decision["predicted_class"]
        competitor_ids = embeddings["competitor_track_ids"]
        decision["predicted_identity_hypothesis"] = (
            "old"
            if predicted == 0
            else (
                f"competitor:{competitor_ids[predicted - 1]}"
                if 1 <= predicted <= len(competitor_ids)
                else "abstain"
            )
        )
        decisions.append(
            {
                **common,
                **decision,
                "gate": float(output.gate[0, 0]),
                "hr_available": bool(embeddings["hr_available"]),
                "hr_crop_count": embeddings["hr_crop_count"],
                "competitor_track_ids": competitor_ids,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision, separators=(",", ":")) + "\n")
    print(f"Wrote {len(decisions)} GT-free decisions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
