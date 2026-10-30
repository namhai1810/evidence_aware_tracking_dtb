"""Zero-shot evaluation of a frozen QuadTrack ECIA ablation on panorama MOT.

The panorama train/validation videos are not used for optimization, early
stopping, threshold selection, or method selection. All model weights and
operational thresholds remain frozen from QuadTrack before the panorama test
sequence is opened.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from hive_hr_reid import run_experiment as baseline
from hive_hr_reid.replay import annotate_operation_outcomes, replay_identity_merges

from .inference import (
    DiskEmbeddingCache,
    extract_event_embeddings,
    recovery_decision,
    tensorize_event,
)
from .model import ECIA, count_trainable_parameters
from .recovery_aware import (
    ARCHITECTURE_NAME as RECOVERY_AWARE_ARCHITECTURE,
    build_from_checkpoint as build_recovery_aware_from_checkpoint,
)
from .panorama_dataset import CODE_ROOT, resolve_config_path
from .reid_encoders import CacheOnlyEncoder, build_reid_encoder, encoder_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("panorama_config.yaml"))
    parser.add_argument("--event-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--ablation",
        choices=("base_only", "base_hr_gain", "full"),
        default="full",
        help="Frozen QuadTrack ECIA ablation to evaluate.",
    )
    parser.add_argument(
        "--ecia-checkpoint",
        type=Path,
        default=None,
        help="Override frozen_models.ecia_checkpoint without modifying the protocol config.",
    )
    parser.add_argument(
        "--ecia-thresholds",
        type=Path,
        default=None,
        help="Override frozen_models.ecia_thresholds with QuadTrack-calibrated thresholds.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reid-model", type=Path, default=None)
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=None,
        help="Writable overlay cache; the event-root cache remains a read-only fallback.",
    )
    parser.add_argument(
        "--reid-encoder",
        choices=("onnx", "osnet_x1_0", "osnet_ain_x1_0"),
        default="onnx",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Read checkpoint-fingerprinted embeddings only and fail on any cache miss.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--gate-override",
        type=float,
        default=None,
        help="Recovery-aware causal ablation; 0 disables HR using the same checkpoint.",
    )
    parser.add_argument(
        "--all-sequences",
        action="store_true",
        help=(
            "Evaluate all DTB train/val/test sequences as one cross-dataset zero-shot "
            "set. This is valid only while all model/threshold choices remain frozen "
            "from QuadTrack."
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sanitize_event(event: dict) -> dict:
    """Remove all evaluation labels before ECIA crop extraction."""

    forbidden = {"gt_id", "has_positive_candidate", "top1_correct"}
    result = {key: value for key, value in event.items() if key not in forbidden and key != "candidates"}
    result["candidates"] = [
        {
            key: value
            for key, value in candidate.items()
            if key not in {"label", "matched_gt_id", "negative_kind"}
        }
        for candidate in event.get("candidates", [])
    ]
    return result


def display_name(ablation: str) -> str:
    return {
        "base_only": "ECIA Base-only",
        "base_hr_gain": "ECIA Base + HR Gain",
        "full": "ECIA Full",
    }[ablation]


def plot_results(
    output: Path,
    baseline_metrics: dict,
    ecia_metrics: dict,
    correction: dict,
    ablation: str,
    scope_label: str,
    baseline_name: str,
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    specs = (
        ("id_switches", "IDSW ↓", "{:.0f}"),
        ("idf1", "IDF1 ↑", "{:.4f}"),
        ("hota", "HOTA ↑", "{:.4f}"),
        ("mota", "MOTA ↑", "{:.4f}"),
    )
    for axis, (key, title, fmt) in zip(axes, specs):
        values = [float(baseline_metrics[key]), float(ecia_metrics[key])]
        bars = axis.bar(
            [baseline_name, display_name(ablation)],
            values,
            color=["#64748b", "#10b981"],
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, labels=[fmt.format(value) for value in values], padding=3)
    figure.suptitle(f"Zero-shot {display_name(ablation)} on {scope_label}")
    figure.savefig(output / "tracking_metrics.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    labels = ["Requested", "Applied", "Immediately fixed", "Wrong segment"]
    values = [
        correction["merge_requests"],
        correction["applied_merges"],
        correction["actually_fixed"],
        correction["wrong_corrections"],
    ]
    bars = axis.bar(labels, values, color=["#38bdf8", "#8b5cf6", "#10b981", "#ef4444"])
    axis.bar_label(bars, padding=3)
    axis.set_ylabel("Events")
    axis.set_title("Identity-recovery outcomes")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / "correction_outcomes.png", dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    event_root = (args.event_root or resolve_config_path(config["output_root"])).resolve()
    if args.all_sequences:
        default_output_name = f"zero_shot_all_{args.ablation}"
    else:
        default_output_name = (
            "zero_shot_evaluation"
            if args.ablation == "full"
            else f"zero_shot_evaluation_{args.ablation}"
        )
    output = (args.output_dir or event_root / default_output_name).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    audit = json.loads((event_root / "leakage_audit.json").read_text())
    if not audit.get("passed") or not audit.get("sequence_disjoint"):
        raise RuntimeError("Panorama split leakage audit did not pass")
    splits = audit["configured_splits"]
    configured_test_sequences = list(splits["test"])
    if set(configured_test_sequences) & (set(splits["train"]) | set(splits["val"])):
        raise RuntimeError("Test sequence overlaps panorama train/validation")
    evaluation_splits = ("train", "val", "test") if args.all_sequences else ("test",)
    evaluation_sequences = [
        sequence for split in evaluation_splits for sequence in splits[split]
    ]
    physical_size = audit.get("physical_lr_size")
    physical_size_label = (
        f"{int(physical_size[0])}x{int(physical_size[1])}"
        if physical_size and len(physical_size) == 2
        else "unknown"
    )

    model_paths = {
        key: resolve_config_path(value)
        for key, value in config["frozen_models"].items()
    }
    if args.ecia_checkpoint is not None:
        model_paths["ecia_checkpoint"] = args.ecia_checkpoint.resolve()
    if args.ecia_thresholds is not None:
        model_paths["ecia_thresholds"] = args.ecia_thresholds.resolve()
    if args.reid_model is not None:
        model_paths["reid_model"] = args.reid_model.resolve()
    for name, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen {name}: {path}")
    device = torch.device(args.device)

    # Load frozen HIVE v2. Labels exist in the offline dataset for evaluation,
    # but baseline.select_frames_with_hive forwards an explicit GT-free key list.
    hive_state = baseline.load_checkpoint(model_paths["hive_checkpoint"], "cpu")
    hive_config = hive_state["config"]
    data_config = dict(hive_config["data"])
    data_config.update(
        teacher_margin_field=None,
        include_competitor_history=True,
        tracking_pattern="{sequence}.txt",
    )
    events = []
    selections = {}
    metadata_dim = None
    hive = None
    for split in evaluation_splits:
        split_dataset = baseline.HIVERankingDataset(
            event_root,
            f"{split}_data/events.jsonl",
            baseline.PROJECT_ROOT,
            data_config,
            include_sequences=list(splits[split]),
        )
        if metadata_dim is None:
            metadata_dim = split_dataset.metadata_dim
            hive = baseline.HIVE(hive_config["model"], metadata_dim=metadata_dim).to(device)
            hive.load_state_dict(hive_state["model"], strict=True)
            hive.eval()
        elif split_dataset.metadata_dim != metadata_dim:
            raise RuntimeError("HIVE metadata dimensions differ across DTB splits")
        split_events, split_selections = baseline.select_frames_with_hive(
            hive,
            split_dataset,
            device,
            int(config["inference"]["batch_size"]),
        )
        duplicate_ids = set(selections) & set(split_selections)
        if duplicate_ids:
            raise RuntimeError(f"Duplicate event IDs across DTB splits: {sorted(duplicate_ids)}")
        events.extend(split_events)
        selections.update(split_selections)

    # Load the exact ablation checkpoint and its QuadTrack-calibrated thresholds.
    # No DTB/panorama label participates in checkpoint or threshold selection.
    checkpoint = torch.load(model_paths["ecia_checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("ablation") != args.ablation:
        raise ValueError(
            f"Requested {args.ablation!r}, but checkpoint contains "
            f"{checkpoint.get('ablation')!r}"
        )
    ecia_config = checkpoint["config"]
    recovery_aware = checkpoint.get("architecture") == RECOVERY_AWARE_ARCHITECTURE
    if recovery_aware:
        ecia = build_recovery_aware_from_checkpoint(checkpoint, device)
    else:
        if args.gate_override is not None:
            raise ValueError("--gate-override is supported only by Recovery-Aware ECIA")
        ecia = ECIA(**{
            key: ecia_config["model"][key]
            for key in ("embedding_dim", "projection_dim", "relation_hidden_dim", "event_hidden_dim", "parameter_limit")
        }).to(device)
        ecia.load_state_dict(checkpoint["model"], strict=True)
        ecia.eval()
        if count_trainable_parameters(ecia) != int(ecia_config["model"]["expected_trainable_parameters"]):
            raise RuntimeError("ECIA parameter-count mismatch")
    threshold_payload = json.loads(model_paths["ecia_thresholds"].read_text())
    thresholds = threshold_payload["selected_thresholds"]

    predictions = {
        sequence: baseline.load_mot(event_root / "predictions" / f"{sequence}.txt", ground_truth=False)
        for sequence in evaluation_sequences
    }
    actual_lr_size = audit.get("physical_lr_size") or (
        int(config["event_generation"]["lowres_width"]),
        int(config["event_generation"]["lowres_height"]),
    )
    extraction_args = argparse.Namespace(
        hive_dataset_root=event_root,
        dataset_root=event_root / "dataset",
        competitor_history_frames=int(config["inference"]["competitor_history_frames"]),
        competitor_samples=int(config["inference"]["competitor_samples"]),
        max_competitors=int(config["inference"]["max_competitors"]),
        lowres_width=int(actual_lr_size[0]),
        lowres_height=int(actual_lr_size[1]),
    )
    encoder = CacheOnlyEncoder() if args.cache_only else build_reid_encoder(
        args.reid_encoder, model_paths["reid_model"], device
    )
    original_embedding_cache = event_root / "embedding_cache"
    embedding_cache_root = (
        args.embedding_cache_dir.resolve()
        if args.embedding_cache_dir is not None
        else original_embedding_cache
    )
    embedding_cache = DiskEmbeddingCache(
        embedding_cache_root,
        encoder_fingerprint(args.reid_encoder, model_paths["reid_model"]),
        fallback_roots=(
            [original_embedding_cache]
            if embedding_cache_root != original_embedding_cache
            else []
        ),
    )
    image_cache = baseline.ImageCache()
    decisions = []
    for index, raw_event in enumerate(events, start=1):
        event = sanitize_event(raw_event)
        selection = selections.get(event["event_id"], {})
        embeddings = extract_event_embeddings(
            args=extraction_args,
            event=event,
            selection=selection,
            lr_frames=predictions[event["sequence"]],
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
        tensors = tensorize_event(embeddings, device)
        with torch.no_grad():
            forward_kwargs = {"ablation": args.ablation}
            if recovery_aware:
                forward_kwargs["gate_override"] = args.gate_override
            result = ecia(**tensors, **forward_kwargs)
        decision = recovery_decision(
            result.decision_logits,
            min_old_probability=float(thresholds["min_old_probability"]),
            min_old_margin=float(thresholds["min_old_margin"]),
        )
        predicted = decision["predicted_class"]
        competitors = embeddings["competitor_track_ids"]
        decision["predicted_identity_hypothesis"] = (
            "old"
            if predicted == 0
            else f"competitor:{competitors[predicted - 1]}"
            if 1 <= predicted <= len(competitors)
            else "abstain"
        )
        decisions.append(
            {
                **common,
                **decision,
                "gate": float(result.gate[0, 0]),
                "gate_override": args.gate_override,
                "model_architecture": checkpoint.get("architecture", "lightweight_ecia_v1"),
                "hr_available": bool(embeddings["hr_available"]),
                "hr_crop_count": int(embeddings["hr_crop_count"]),
                "competitor_track_ids": competitors,
            }
        )
        if index == 1 or index % 10 == 0 or index == len(events):
            print(f"Zero-shot ECIA decisions {index}/{len(events)}", flush=True)

    # Freeze the GT-free artifact before loading manual GT for evaluation.
    decision_path = output / "decisions_gt_free.jsonl"
    with decision_path.open("w", encoding="utf-8") as handle:
        for row in decisions:
            if "gt_id" in row:
                raise RuntimeError("GT identity leaked into decision artifact")
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    raw_events = {}
    for split in evaluation_splits:
        for row in load_jsonl(event_root / f"{split}_data/events.jsonl"):
            if row["event_id"] in raw_events:
                raise RuntimeError(f"Duplicate raw event ID: {row['event_id']}")
            raw_events[row["event_id"]] = row
    evaluation_events = [{**row, "gt_id": int(raw_events[row["event_id"]]["gt_id"])} for row in decisions]
    gt = {
        sequence: baseline.load_mot(event_root / "dataset" / sequence / "gt" / "gt.txt", ground_truth=True)
        for sequence in evaluation_sequences
    }
    decisions_by_sequence: dict[str, list[dict]] = defaultdict(list)
    events_by_sequence: dict[str, list[dict]] = defaultdict(list)
    for decision, event in zip(decisions, evaluation_events):
        decisions_by_sequence[decision["sequence"]].append(decision)
        events_by_sequence[event["sequence"]].append(event)
    replayed = {}
    operations_by_sequence = {}
    outcomes = []
    for sequence in evaluation_sequences:
        replayed[sequence], operations_by_sequence[sequence] = replay_identity_merges(
            predictions[sequence],
            decisions_by_sequence[sequence],
            int(config["inference"]["max_segment_gap"]),
        )
        outcomes.extend(
            annotate_operation_outcomes(
                events_by_sequence[sequence],
                operations_by_sequence[sequence],
                gt[sequence],
                predictions[sequence],
                replayed[sequence],
                float(config["event_generation"]["evaluation_iou"]),
            )
        )
        baseline.write_mot(
            output / "predictions" / f"{sequence}_ecia_{args.ablation}.txt",
            replayed[sequence],
        )

    metric_args = argparse.Namespace(iou=float(config["event_generation"]["evaluation_iou"]))
    before = baseline.basic_metrics(
        metric_args, evaluation_sequences, predictions, gt, include_hota=True
    )
    after = baseline.basic_metrics(
        metric_args, evaluation_sequences, replayed, gt, include_hota=True
    )
    operations = [row for values in operations_by_sequence.values() for row in values]
    correction = {
        "events": len(decisions),
        "hr_requests": sum(row.get("selected_candidate_id") is not None for row in decisions),
        "encoded_hr_crops": sum(int(row.get("hr_crop_count", 0)) for row in decisions),
        "merge_requests": sum(bool(row.get("merge")) for row in decisions),
        "applied_merges": sum(bool(row["applied"]) for row in operations),
        "actually_fixed": sum(bool(row["actually_fixed"]) for row in outcomes),
        "wrong_corrections": sum(bool(row["wrong_correction"]) for row in outcomes),
        "collision_or_missing_rejections": sum(row["requested_merge"] and not row["applied"] for row in operations),
    }

    # Baseline-only split statistics are descriptive. They do not influence any
    # checkpoint, threshold, or test decision.
    baseline_splits = {}
    for split, sequences in splits.items():
        split_predictions = {
            sequence: baseline.load_mot(event_root / "predictions" / f"{sequence}.txt", ground_truth=False)
            for sequence in sequences
        }
        split_gt = {
            sequence: baseline.load_mot(event_root / "dataset" / sequence / "gt" / "gt.txt", ground_truth=True)
            for sequence in sequences
        }
        baseline_splits[split] = baseline.basic_metrics(
            metric_args, sequences, split_predictions, split_gt, include_hota=True
        )

    payload = {
        "protocol": {
            "zero_shot": True,
            "panorama_train_used_for_fitting": False,
            "panorama_validation_used_for_fitting_or_thresholds": False,
            "test_decisions_frozen_before_gt_join": True,
            "oracle_switch_trigger": True,
            # test_sequences is retained for comparison-artifact compatibility;
            # evaluation_scope disambiguates the all-DTB zero-shot protocol.
            "test_sequences": evaluation_sequences,
            "configured_test_sequences": configured_test_sequences,
            "evaluation_sequences": evaluation_sequences,
            "evaluation_scope": (
                "all_dtb_physical_lr_zero_shot"
                if args.all_sequences and audit.get("physical_lr_detection_tracking")
                else "all_dtb_zero_shot"
                if args.all_sequences
                else "held_out_test"
            ),
            "physical_lr_detection_tracking": bool(
                audit.get("physical_lr_detection_tracking", False)
            ),
            "physical_lr_size": audit.get("physical_lr_size"),
            "frozen_quadtrack_ecia_epoch": int(checkpoint["epoch"]),
            "ecia_ablation": args.ablation,
            "ecia_architecture": checkpoint.get("architecture", "lightweight_ecia_v1"),
            "gate_override": args.gate_override,
            "frozen_quadtrack_thresholds": thresholds,
            "hive_checkpoint_sha256": baseline.sha256(model_paths["hive_checkpoint"]),
            "ecia_checkpoint_sha256": baseline.sha256(model_paths["ecia_checkpoint"]),
            "reid_model_sha256": baseline.sha256(model_paths["reid_model"]),
            "reid_encoder": args.reid_encoder,
            "reid_backend": encoder.backend,
            "reid_load_audit": getattr(encoder, "load_audit", None),
        },
        "baseline_splits": baseline_splits,
        "test_baseline": before,
        f"test_ecia_{args.ablation}": after,
        "correction": correction,
        "outcomes": outcomes,
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output / "outcomes.jsonl").open("w", encoding="utf-8") as handle:
        for row in outcomes:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    scope_label = (
        f"all five DTB physical-LR {physical_size_label} sequences"
        if args.all_sequences and audit.get("physical_lr_detection_tracking")
        else "all five unseen DTB panorama sequences"
        if args.all_sequences
        else "the held-out panorama full-occlusion sequence"
    )
    baseline_name = (
        f"YOLO11l + BoT-SORT/ReID @ {physical_size_label}"
        if audit.get("physical_lr_detection_tracking")
        else "YOLO12l + ByteTrack"
    )
    plot_results(
        output,
        before["aggregate"],
        after["aggregate"],
        correction,
        args.ablation,
        scope_label,
        baseline_name,
    )

    b, a = before["aggregate"], after["aggregate"]
    report = [
        f"# Zero-shot {display_name(args.ablation)} evaluation on manual panorama MOT",
        "",
        f"The {display_name(args.ablation)} checkpoint, HIVE v2 checkpoint, ReID model, and merge thresholds were frozen on QuadTrack. Panorama train/validation videos were not used for fitting or calibration.",
        "",
        "## Leakage-safe split",
        "",
        f"- Train (unused by zero-shot model): `{', '.join(splits['train'])}`",
        f"- Validation (unused by zero-shot model): `{', '.join(splits['val'])}`",
        f"- Held-out test: `{', '.join(splits['test'])}`",
        "",
        "## Zero-shot result" if args.all_sequences else "## Held-out result",
        "",
        "| Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | MOTA ↑ |",
        "|---|---:|---:|---:|---:|",
        f"| {baseline_name} | {b['id_switches']} | {b['idf1']:.4f} | {b['hota']:.4f} | {b['mota']:.4f} |",
        f"| Frozen HIVE v2 + {display_name(args.ablation)} | {a['id_switches']} | {a['idf1']:.4f} | {a['hota']:.4f} | {a['mota']:.4f} |",
        "",
        "## Per-sequence result",
        "",
        "| Sequence | Method | IDSW ↓ | IDF1 ↑ | MOTA ↑ |",
        "|---|---|---:|---:|---:|",
        *[
            row
            for sequence in evaluation_sequences
            for row in (
                f"| {sequence} | {baseline_name} | {before['sequences'][sequence]['id_switches']} | {before['sequences'][sequence]['idf1']:.4f} | {before['sequences'][sequence]['mota']:.4f} |",
                f"| {sequence} | {display_name(args.ablation)} | {after['sequences'][sequence]['id_switches']} | {after['sequences'][sequence]['idf1']:.4f} | {after['sequences'][sequence]['mota']:.4f} |",
            )
        ],
        "",
        "## Recovery outcomes",
        "",
        f"- Oracle switch events: {correction['events']}",
        f"- HR requests / encoded crops: {correction['hr_requests']} / {correction['encoded_hr_crops']}",
        f"- Merge requests / applied: {correction['merge_requests']} / {correction['applied_merges']}",
        f"- Immediately fixed: {correction['actually_fixed']}",
        f"- Segment-level wrong corrections: {correction['wrong_corrections']}",
        f"- Collision/missing rejections: {correction['collision_or_missing_rejections']}",
        "",
        "![Tracking metrics](tracking_metrics.png)",
        "",
        "![Correction outcomes](correction_outcomes.png)",
        "",
        "## Scope",
        "",
        (
            f"This is a cross-dataset zero-shot aggregate over all five DTB videos. The detector/tracker physically consumed {physical_size_label} frames, and its boxes were mapped back to the original panorama coordinates before GT evaluation and HR cropping. The legacy DTB train/val/test names are not used for fitting or model selection; all HIVE, ReID, and ECIA weights and thresholds remain frozen from QuadTrack. Because these results have now been observed, these videos must not later be used for tuning and still be claimed as held-out evaluation."
            if args.all_sequences and audit.get("physical_lr_detection_tracking")
            else "This is a cross-dataset zero-shot aggregate over all five DTB videos. The legacy DTB train/val/test names are not used for fitting or model selection; all weights and thresholds remain frozen from QuadTrack. Because these results have now been observed, these videos must not later be used for tuning and still be claimed as held-out evaluation."
            if args.all_sequences
            else "This is a zero-shot domain-generalization test with an oracle switch trigger. It evaluates recovery after a GT-confirmed switch is supplied; it does not evaluate online switch detection. The baseline tracker is YOLO12l + ByteTrack with appearance_reid=false. ReID is introduced only by the frozen ECIA evidence encoder."
        ),
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Zero-shot report: {output / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
