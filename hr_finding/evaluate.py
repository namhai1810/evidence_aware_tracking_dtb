#!/usr/bin/env python3
"""Evaluate HIVE on a video-disjoint split and export inspectable rankings."""

from __future__ import annotations

import argparse
import csv
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import HIVERankingDataset, collate_event_batch
from losses import HIVEMultiTaskLoss
from models import HIVE, count_trainable_parameters
from utils.checkpoint import load_checkpoint
from utils.config import load_config, resolve_path
from utils.metrics import MetricAccumulator, move_batch_to_device
from utils.seed import seed_everything


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML configuration. By default the configuration embedded in the checkpoint is used.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--baseline", choices=("none", "temporal"), default="none")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Default 0 avoids small container /dev/shm failures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write metrics, per-event rankings, failures and a Markdown report here.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Backward-compatible path for metrics JSON.",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace, state: dict[str, Any] | None) -> dict:
    if args.config is not None:
        config = load_config(args.config)
        if state is not None and isinstance(state.get("config"), dict):
            checkpoint_model = state["config"].get("model", {})
            if config.get("model", {}) != checkpoint_model:
                raise ValueError(
                    "The supplied config model section differs from the checkpoint. "
                    "Remove --config to use the checkpoint configuration."
                )
        return config
    if state is not None and isinstance(state.get("config"), dict):
        return deepcopy(state["config"])
    return load_config(SCRIPT_DIR / "configs" / "hive.yaml")


def candidate_prediction_rows(
    logits: torch.Tensor,
    batch: dict,
    event_lookup: dict[str, dict],
    request_threshold: float,
) -> list[dict]:
    """Create JSON-serializable rankings while labels are used only for evaluation."""
    predictions = []
    logits = logits.detach().float().cpu()
    labels = batch["labels"].detach().float().cpu()
    masks = batch["candidate_mask"].detach().bool().cpu()
    for row_index, event_id in enumerate(batch["event_id"]):
        event = event_lookup[event_id]
        valid_slots = torch.nonzero(masks[row_index], as_tuple=False).flatten().tolist()
        ordered_slots = sorted(valid_slots, key=lambda slot: float(logits[row_index, slot]), reverse=True)
        rank_by_slot = {slot: rank for rank, slot in enumerate(ordered_slots, start=1)}
        candidates = []
        for slot, candidate in enumerate(event.get("candidates", [])[: masks.shape[1]]):
            available = slot in rank_by_slot
            logit = float(logits[row_index, slot]) if available else None
            probability = float(torch.sigmoid(logits[row_index, slot])) if available else None
            candidates.append(
                {
                    "candidate_id": candidate.get("candidate_id", f"{event_id}_slot{slot + 1}"),
                    "candidate_frame": candidate.get("candidate_frame"),
                    "seconds_before": candidate.get("seconds_before"),
                    "source_image": candidate.get("source_image"),
                    "lr_input_available": available,
                    "predicted_rank": rank_by_slot.get(slot),
                    "utility_logit": logit,
                    "utility_probability": probability,
                    "gt_label": int(labels[row_index, slot].item()) if available else int(candidate["label"]),
                }
            )
        ranked = sorted(
            (candidate for candidate in candidates if candidate["lr_input_available"]),
            key=lambda candidate: int(candidate["predicted_rank"]),
        )
        selected = ranked[0] if ranked else None
        has_positive = any(candidate["gt_label"] == 1 for candidate in ranked)
        top1_correct = bool(selected is not None and selected["gt_label"] == 1)
        requested = bool(
            selected is not None
            and float(selected["utility_probability"]) >= request_threshold
        )
        predictions.append(
            {
                "event_id": event_id,
                "sequence": event["sequence"],
                "target_frame": event["target_frame"],
                "lr_old_pred_id": event.get("lr_old_pred_id"),
                "lr_new_pred_id": event.get("lr_new_pred_id"),
                "candidate_count": len(ranked),
                "oracle_retrievable": has_positive,
                "top1_correct": top1_correct,
                "requested": requested,
                "selected_candidate_id": selected["candidate_id"] if selected else None,
                "selected_frame": selected["candidate_frame"] if selected else None,
                "selected_seconds_before": selected["seconds_before"] if selected else None,
                "selected_probability": selected["utility_probability"] if selected else None,
                "candidates": ranked,
            }
        )
    return predictions


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def export_results(
    output_dir: Path,
    metrics: dict,
    predictions: list[dict],
    checkpoint: Path | None,
    config: dict,
    sequences: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    failures = [
        row for row in predictions if row["oracle_retrievable"] and not row["top1_correct"]
    ]
    write_jsonl(output_dir / "top1_failures.jsonl", failures)

    fields = [
        "event_id", "sequence", "target_frame", "oracle_retrievable", "top1_correct",
        "requested", "candidate_id", "candidate_frame", "seconds_before", "predicted_rank",
        "utility_logit", "utility_probability", "gt_label", "source_image",
    ]
    with (output_dir / "candidate_rankings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in predictions:
            for candidate in event["candidates"]:
                writer.writerow(
                    {
                        "event_id": event["event_id"],
                        "sequence": event["sequence"],
                        "target_frame": event["target_frame"],
                        "oracle_retrievable": event["oracle_retrievable"],
                        "top1_correct": event["top1_correct"],
                        "requested": event["requested"],
                        "candidate_id": candidate["candidate_id"],
                        "candidate_frame": candidate["candidate_frame"],
                        "seconds_before": candidate["seconds_before"],
                        "predicted_rank": candidate["predicted_rank"],
                        "utility_logit": candidate["utility_logit"],
                        "utility_probability": candidate["utility_probability"],
                        "gt_label": candidate["gt_label"],
                        "source_image": candidate["source_image"],
                    }
                )

    percent_metrics = [
        ("Candidate accuracy", "accuracy"),
        ("Candidate precision", "precision"),
        ("Candidate recall", "recall"),
        ("Candidate F1", "f1"),
        ("Success@1 (all events)", "success_at_1"),
        ("Success@3 (all events)", "success_at_3"),
        ("Success@5 (all events)", "success_at_5"),
        ("Success@1 (retrievable only)", "success_at_1_when_retrievable"),
        ("MRR", "mrr"),
        ("NDCG@5", "ndcg_at_5"),
        ("Oracle candidate coverage", "oracle_coverage"),
        ("Oracle gap at 1", "oracle_gap_at_1"),
        ("Request rate", "request_rate"),
        ("Request precision", "request_precision"),
    ]
    metric_lines = [
        f"| {name} | {float(metrics[key]):.4f} ({float(metrics[key]) * 100:.2f}%) |"
        for name, key in percent_metrics
    ]
    report = f"""# HIVE held-out {metrics['split']} evaluation

## Evaluation identity

| Field | Value |
|---|---|
| Checkpoint | `{checkpoint if checkpoint is not None else 'temporal baseline'}` |
| Checkpoint epoch | {metrics.get('checkpoint_epoch', 'n/a')} |
| Model variant | `{config['model']['variant']}` |
| Video sequences | {', '.join(sequences)} |
| Events | {metrics['event_count']} |
| Candidates | {metrics['candidate_count']} |
| Trainable parameters | {metrics['parameters']:,} |

## Results

| Metric | Value |
|---|---:|
{chr(10).join(metric_lines)}

The model selected a GT-positive HR frame at rank 1 for **{metrics['success_at_1'] * metrics['event_count']:.0f}/{metrics['event_count']}** test events. The fixed candidate pool contained at least one GT-positive frame for **{metrics['events_with_positive']}/{metrics['event_count']}** events. There are **{len(failures)}** retrievable events where the model's top-1 choice was wrong.

## Runtime

| Metric | Value |
|---|---:|
| Model latency per event | {metrics['latency_per_event_ms']:.3f} ms |
| Encoder time per event | {metrics['encoder_time_per_event_ms']:.3f} ms |
| Scorer time per candidate | {metrics['scorer_time_per_candidate_ms']:.3f} ms |
| End-to-end wall time | {metrics['wall_time_seconds']:.3f} s |

## Output files

- `predictions.jsonl`: complete model ranking for every switch event.
- `candidate_rankings.csv`: one row per candidate for analysis or plotting.
- `top1_failures.jsonl`: cases where a valid positive existed but top-1 was wrong.
- `metrics.json`: machine-readable aggregate metrics.

## Interpretation boundary

This is a video-disjoint offline test of **LR-only frame selection**. HR images and GT labels are not model inputs; GT is read after inference to score the ranking. Success@K measures selection correctness, not a measured reduction in ID switches. IDSW, IDF1, MOTA and HOTA require a separate tracker correction/replay experiment.
"""
    (output_dir / "test_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.baseline == "none" and args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --baseline is selected")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    state = load_checkpoint(args.checkpoint, "cpu") if args.checkpoint else None
    config = resolve_config(args, state)
    seed_everything(int(config["experiment"]["seed"]), True)
    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root
        else resolve_path(config["data"]["dataset_root"], SCRIPT_DIR)
    )
    project_root = (
        args.project_root.resolve()
        if args.project_root
        else resolve_path(config["data"]["project_root"], SCRIPT_DIR)
    )
    if args.split == "val":
        manifest = config["data"]["train_manifest"]
        sequences = [str(value) for value in config["data"]["validation_sequences"]]
    else:
        manifest = config["data"]["test_manifest"]
        split_config = json.loads(
            (dataset_root / "split_config.json").read_text(encoding="utf-8")
        )
        sequences = [str(value) for value in split_config["test_sequences"]]

    data_config = dict(config["data"])
    data_config["teacher_margin_field"] = config["loss"].get("margin_teacher_field")
    data_config["include_competitor_history"] = bool(
        data_config.get("include_competitor_history", True)
        and config["model"].get("variant") == "hive_full"
    )
    dataset = HIVERankingDataset(
        dataset_root, manifest, project_root, data_config, include_sequences=sequences
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(config["data"].get("pin_memory", True)),
        collate_fn=collate_event_batch,
    )
    event_lookup = {event["event_id"]: event for event in dataset.events}
    model = None
    criterion = HIVEMultiTaskLoss(config["loss"], None).to(device)
    if args.baseline == "none":
        assert state is not None
        model = HIVE(config["model"], metadata_dim=dataset.metadata_dim).to(device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()

    accumulator = MetricAccumulator(
        config["evaluation"]["classification_threshold"],
        config["evaluation"].get("no_request_threshold"),
    )
    totals = {"total": 0.0, "classification": 0.0, "ranking": 0.0, "margin": 0.0}
    timings = {"encoder_seconds": 0.0, "scorer_seconds": 0.0, "total_seconds": 0.0}
    batches = timed_events = timed_candidates = 0
    warmup_batches = int(config["evaluation"].get("latency_warmup_batches", 5))
    predictions: list[dict] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(
            tqdm(loader, desc=f"evaluate {args.split}", dynamic_ncols=True)
        ):
            batch = move_batch_to_device(batch, device)
            if args.baseline == "temporal":
                logits = -batch["metadata"][..., 0]
                logits = logits.masked_fill(~batch["candidate_mask"], -1e4)
            else:
                assert model is not None
                # Always retain at least the final batch for timing on small splits.
                if batch_index < warmup_batches and batch_index < len(loader) - 1:
                    logits = model(batch)
                else:
                    logits, measured = model.forward_with_timing(batch)
                    for key in timings:
                        timings[key] += measured[key]
                    timed_events += len(batch["event_id"])
                    timed_candidates += int(batch["candidate_mask"].sum())
            losses = criterion(logits, batch)
            for key in totals:
                totals[key] += float(losses[key])
            batches += 1
            accumulator.update(logits, batch["labels"], batch["candidate_mask"], batch["event_id"])
            predictions.extend(
                candidate_prediction_rows(
                    logits,
                    batch,
                    event_lookup,
                    float(config["evaluation"].get("no_request_threshold", 0.5)),
                )
            )

    metrics = accumulator.compute()
    metrics.update({f"loss_{key}": value / max(1, batches) for key, value in totals.items()})
    metrics["wall_time_seconds"] = time.perf_counter() - started
    metrics["parameters"] = count_trainable_parameters(model) if model is not None else 0
    metrics["latency_per_event_ms"] = timings["total_seconds"] * 1000 / max(1, timed_events)
    metrics["encoder_time_per_event_ms"] = timings["encoder_seconds"] * 1000 / max(1, timed_events)
    metrics["scorer_time_per_candidate_ms"] = timings["scorer_seconds"] * 1000 / max(1, timed_candidates)
    metrics["latency_timed_events"] = timed_events
    metrics["split"] = args.split
    metrics["baseline"] = args.baseline
    metrics["checkpoint_epoch"] = int(state["epoch"]) if state is not None else None
    metrics["checkpoint_best_validation_metric"] = (
        float(state["best_metric"]) if state is not None else None
    )
    print(json.dumps(metrics, indent=2))

    if args.output_dir:
        export_results(
            args.output_dir.resolve(), metrics, predictions, args.checkpoint, config, sequences
        )
        print(f"Results written to: {args.output_dir.resolve()}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
