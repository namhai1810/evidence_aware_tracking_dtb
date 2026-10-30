"""Prepare train/validation ECIA embedding caches without touching test videos.

This is the only stage allowed to read GT identity fields for training-label
construction. HIVE v2 and ReID are loaded frozen; exactly one selected HR crop
is used per source event.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from hive_hr_reid import run_experiment as baseline

from .dataset import (
    ECIASample,
    assert_sample_coverage,
    audit_named_splits,
    build_identity_sample,
    normalize_embedding,
    save_embedding_cache,
)
from .inference import DiskEmbeddingCache, extract_event_embeddings
from .reid_encoders import (
    ReIDEncoder,
    build_reid_encoder,
    encoder_fingerprint,
    resolve_reid_settings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=baseline.DEFAULT_DATASET)
    parser.add_argument("--hive-dataset-root", type=Path, default=baseline.DEFAULT_HIVE_DATASET)
    parser.add_argument("--hive-checkpoint", type=Path, default=baseline.DEFAULT_CHECKPOINT)
    parser.add_argument("--reid-model", type=Path, default=None)
    parser.add_argument(
        "--reid-encoder",
        choices=("onnx", "osnet_x1_0", "osnet_ain_x1_0"),
        default=None,
    )
    parser.add_argument("--lr-pred-dir", type=Path, default=baseline.DEFAULT_LR_PREDICTIONS)
    parser.add_argument(
        "--lr-pred-pattern", default="{sequence}_yolo11l_botsort_reid_pre512x120.txt"
    )
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


def selected_hr_identity(event: dict, selection: dict) -> int | None:
    """Evaluation-only identity lookup used to label misleading HR evidence."""

    for candidate in event.get("candidates", []):
        if candidate["candidate_id"] == selection.get("candidate_id"):
            value = candidate.get("matched_gt_id")
            return int(value) if value is not None else None
    return None


def make_original_sample(
    split: str,
    event: dict,
    selection: dict,
    embeddings: dict,
) -> ECIASample:
    competitor_count = embeddings["competitors"].shape[0]
    competitor_identity_codes = list(range(100, 100 + competitor_count))
    gt_id = int(event["gt_id"])
    hr_gt_id = selected_hr_identity(event, selection)
    # Synthetic identity code 0 represents the event's true old/target person;
    # competitor codes are local to this sample and never model inputs.
    return build_identity_sample(
        event_id=event["event_id"],
        sequence=event["sequence"],
        split=split,
        old_embedding=embeddings["old"],
        target_embedding=embeddings["target"],
        hr_embedding=embeddings["hr"] if embeddings["hr_available"] else None,
        competitor_embeddings=list(embeddings["competitors"]),
        old_identity=0,
        target_identity=0,
        hr_identity=0 if hr_gt_id == gt_id else -1,
        competitor_identities=competitor_identity_codes,
        competitor_track_ids=embeddings["competitor_track_ids"],
    )


def counterfactual_competitor_sample(
    source: ECIASample,
    competitor_slot: int,
    *,
    kind: str,
    target_embedding: np.ndarray,
    prototype_embedding: np.ndarray,
) -> ECIASample:
    """Use a real competitor prototype as a counterfactual target hypothesis.

    These samples teach competitor and hard-negative classes without importing
    any crop from validation/test. They are training-time augmentation only;
    final validation/test replay always uses untouched real events.
    """

    competitor_count = source.competitors.shape[0]
    identities = list(range(100, 100 + competitor_count))
    competitor_prototypes = source.competitors.copy()
    competitor_prototypes[competitor_slot] = prototype_embedding
    return build_identity_sample(
        event_id=f"{source.event_id}__{kind}_c{competitor_slot}",
        sequence=source.sequence,
        split=source.split,
        old_embedding=source.old,
        target_embedding=target_embedding,
        hr_embedding=source.hr,
        competitor_embeddings=list(competitor_prototypes),
        old_identity=0,
        target_identity=identities[competitor_slot],
        hr_identity=-1 if source.hr is not None else None,
        competitor_identities=identities,
        competitor_track_ids=source.competitor_track_ids,
        hard_negative=kind == "hard_negative",
    )


def add_counterfactual_samples(
    originals: list[ECIASample],
    identity_keys: dict[str, tuple[str, int]],
    hypothesis_identity_sets: dict[str, set[int]],
    eligible_negative_slots: dict[str, list[int]],
    held_out_competitor_pairs: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
) -> list[ECIASample]:
    """Add easy negative, hardest competitor, and real-embedding abstain cases."""

    augmented = list(originals)
    for sample in originals:
        eligible = eligible_negative_slots.get(sample.event_id, [])
        if eligible:
            similarities = sample.competitors[eligible] @ sample.target
            easiest = eligible[int(np.argmin(similarities))]
            hardest = eligible[int(np.argmax(similarities))]
            augmented.append(
                counterfactual_competitor_sample(
                    sample,
                    easiest,
                    kind="negative",
                    target_embedding=held_out_competitor_pairs[sample.event_id][easiest][0],
                    prototype_embedding=held_out_competitor_pairs[sample.event_id][easiest][1],
                )
            )
            augmented.append(
                counterfactual_competitor_sample(
                    sample,
                    hardest,
                    kind="hard_negative",
                    target_embedding=held_out_competitor_pairs[sample.event_id][hardest][0],
                    prototype_embedding=held_out_competitor_pairs[sample.event_id][hardest][1],
                )
            )

    # No-merge targets are borrowed only within the same split. Prefer another
    # GT event from the same sequence; fall back to a different sequence.
    for index, sample in enumerate(originals):
        candidates = [
            other
            for other in originals
            if other.event_id != sample.event_id
            and other.sequence == sample.sequence
            and identity_keys[other.event_id][1]
            not in hypothesis_identity_sets[sample.event_id]
        ]
        if not candidates:
            # MOT identities are sequence-local. A target borrowed from another
            # video cannot be the source event's Old or Competitor identity,
            # even if the integer ID happens to be reused in both videos.
            candidates = [
                other
                for other in originals
                if other.event_id != sample.event_id
                and other.sequence != sample.sequence
            ]
        if not candidates:
            continue
        borrowed = candidates[index % len(candidates)]
        competitor_count = sample.competitors.shape[0]
        augmented.append(
            build_identity_sample(
                event_id=f"{sample.event_id}__no_merge",
                sequence=sample.sequence,
                split=sample.split,
                old_embedding=sample.old,
                target_embedding=borrowed.target,
                hr_embedding=sample.hr,
                competitor_embeddings=list(sample.competitors),
                old_identity=0,
                target_identity=-999,
                hr_identity=-1 if sample.hr is not None else None,
                competitor_identities=list(range(100, 100 + competitor_count)),
                competitor_track_ids=sample.competitor_track_ids,
            )
        )
    return augmented


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def competitor_gt_identities(
    args: argparse.Namespace,
    event: dict,
    lr_frames: dict[int, list[tuple]],
    gt_frames: dict[int, list[tuple]],
) -> dict[int, int | None]:
    """Post-hoc GT match used only to avoid mislabeled synthetic negatives."""

    histories = baseline.competitor_histories(
        lr_frames,
        event,
        args.competitor_history_frames,
        args.competitor_samples,
        args.max_competitors,
    )
    result = {}
    for track_id, history in histories:
        matched = []
        for frame, prediction in history:
            best_id = None
            best_iou = 0.0
            for gt_id, gt_box, _ in gt_frames.get(frame, []):
                overlap = box_iou(np.asarray(prediction[1]), np.asarray(gt_box))
                if overlap > best_iou:
                    best_iou, best_id = overlap, int(gt_id)
            if best_id is not None and best_iou >= args.iou:
                matched.append(best_id)
        # Synthetic competitor targets are allowed only when every sampled crop
        # is matched and all agree on one GT identity. Mixed/unknown tracklets
        # are retained as inference distractors but never used as labeled
        # counterfactual targets.
        result[int(track_id)] = (
            matched[0]
            if len(matched) >= 2
            and len(matched) == len(history)
            and len(set(matched)) == 1
            else None
        )
    return result


def build_split(
    *,
    args: argparse.Namespace,
    split: str,
    sequences: list[str],
    manifest: str,
    hive_config: dict,
    hive_state: dict,
    encoder: ReIDEncoder,
    embedding_cache: DiskEmbeddingCache,
    image_cache: baseline.ImageCache,
    device: torch.device,
) -> tuple[list[ECIASample], dict]:
    dataset = baseline.build_hive_dataset(args, hive_config, sequences, manifest)
    hive = baseline.HIVE(hive_config["model"], metadata_dim=dataset.metadata_dim).to(device)
    hive.load_state_dict(hive_state["model"], strict=True)
    hive.eval()
    events, selections = baseline.select_frames_with_hive(
        hive, dataset, device, args.batch_size
    )
    prediction_cache = {
        sequence: baseline.load_mot(
            args.lr_pred_dir / args.lr_pred_pattern.format(sequence=sequence),
            ground_truth=False,
        )
        for sequence in sorted({event["sequence"] for event in events})
    }
    gt_cache = {
        sequence: baseline.load_mot(
            args.dataset_root / sequence / "gt" / "gt.txt", ground_truth=True
        )
        for sequence in sorted({event["sequence"] for event in events})
    }

    originals = []
    identity_keys: dict[str, tuple[str, int]] = {}
    hypothesis_identity_sets: dict[str, set[int]] = {}
    eligible_negative_slots: dict[str, list[int]] = {}
    held_out_competitor_pairs: dict[
        str, dict[int, tuple[np.ndarray, np.ndarray]]
    ] = {}
    skipped = Counter()
    with torch.no_grad():
        for index, event in enumerate(events, start=1):
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
            if embeddings is None:
                skipped["missing_old_or_target"] += 1
                continue
            if embeddings["hr_crop_count"] > 1:
                raise RuntimeError("One-HR-crop budget violated while building cache")
            sample = make_original_sample(split, event, selection, embeddings)
            originals.append(sample)
            identity_keys[sample.event_id] = (event["sequence"], int(event["gt_id"]))
            competitor_gt = competitor_gt_identities(
                args,
                event,
                prediction_cache[event["sequence"]],
                gt_cache[event["sequence"]],
            )
            hypothesis_identity_sets[sample.event_id] = {
                int(event["gt_id"]),
                *(int(value) for value in competitor_gt.values() if value is not None),
            }
            held_out_competitor_pairs[sample.event_id] = {}
            for slot, members in enumerate(embeddings["competitor_member_embeddings"]):
                if len(members) < 2:
                    continue
                # The latest sampled crop becomes a real target; the prototype
                # is recomputed without it to prevent target-in-prototype leakage.
                held_out_competitor_pairs[sample.event_id][slot] = (
                    normalize_embedding(members[-1]),
                    normalize_embedding(np.mean(members[:-1], axis=0)),
                )
            eligible_negative_slots[sample.event_id] = [
                slot
                for slot, track_id in enumerate(embeddings["competitor_track_ids"])
                if competitor_gt.get(track_id) is not None
                and competitor_gt[track_id] != int(event["gt_id"])
                and slot in held_out_competitor_pairs[sample.event_id]
            ]
            if index == 1 or index % 25 == 0 or index == len(events):
                print(f"  {split} frozen evidence {index}/{len(events)}", flush=True)

    samples = add_counterfactual_samples(
        originals,
        identity_keys,
        hypothesis_identity_sets,
        eligible_negative_slots,
        held_out_competitor_pairs,
    )
    return samples, {
        "source_events": len(events),
        "original_samples": len(originals),
        "materialized_samples": len(samples),
        "sample_kinds": dict(Counter(sample.sample_kind for sample in samples)),
        "skipped": dict(skipped),
        "sequences": sorted({sample.sequence for sample in samples}),
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reid_kind, reid_checkpoint = resolve_reid_settings(args, config)
    data_config = config["data"]
    audit_named_splits(
        train=data_config["train_sequences"],
        validation=data_config["validation_sequences"],
        calibration=data_config["calibration_sequences"],
        test=data_config["test_sequences"],
    )
    train_path = Path(data_config["train_cache"])
    validation_path = Path(data_config["validation_cache"])
    for path in (train_path, validation_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Cache exists: {path}; pass --overwrite")

    device = torch.device(args.device)
    hive_state = baseline.load_checkpoint(args.hive_checkpoint.resolve(), "cpu")
    hive_config = hive_state["config"]
    encoder = build_reid_encoder(reid_kind, reid_checkpoint, device)
    embedding_cache = DiskEmbeddingCache(
        Path(config["inference"]["embedding_cache_dir"]),
        encoder_fingerprint(reid_kind, reid_checkpoint),
    )
    image_cache = baseline.ImageCache()

    train_samples, train_audit = build_split(
        args=args,
        split="train",
        sequences=list(data_config["train_sequences"]),
        manifest=hive_config["data"]["train_manifest"],
        hive_config=hive_config,
        hive_state=hive_state,
        encoder=encoder,
        embedding_cache=embedding_cache,
        image_cache=image_cache,
        device=device,
    )
    validation_samples, validation_audit = build_split(
        args=args,
        split="validation",
        sequences=list(data_config["validation_sequences"]),
        manifest=hive_config["data"]["train_manifest"],
        hive_config=hive_config,
        hive_state=hive_state,
        encoder=encoder,
        embedding_cache=embedding_cache,
        image_cache=image_cache,
        device=device,
    )
    assert_sample_coverage(train_samples, data_config["required_sample_kinds"])
    assert_sample_coverage(validation_samples, data_config["required_sample_kinds"])
    save_embedding_cache(
        train_path, train_samples, max_competitors=int(config["model"]["max_competitors"])
    )
    save_embedding_cache(
        validation_path,
        validation_samples,
        max_competitors=int(config["model"]["max_competitors"]),
    )

    audit = {
        "hive_checkpoint": str(args.hive_checkpoint.resolve()),
        "hive_checkpoint_sha256": baseline.sha256(args.hive_checkpoint),
        "reid_encoder": reid_kind,
        "reid_backend": encoder.backend,
        "reid_model": str(reid_checkpoint),
        "reid_model_sha256": baseline.sha256(reid_checkpoint),
        "reid_load_audit": getattr(encoder, "load_audit", None),
        "hive_frozen": True,
        "reid_frozen": True,
        "calibration_sequences_accessed": False,
        "test_sequences_accessed": False,
        "train": train_audit,
        "validation": validation_audit,
        "configured_calibration_sequences": data_config["calibration_sequences"],
        "configured_test_sequences": data_config["test_sequences"],
    }
    audit_path = train_path.parent / "cache_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Train cache: {train_path}")
    print(f"Validation cache: {validation_path}")
    print(f"Leakage audit: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
