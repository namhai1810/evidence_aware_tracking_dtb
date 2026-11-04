"""Robustly calibrate online ECIA thresholds on disjoint QuadTrack videos."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import statistics

import torch
from torch.utils.data import DataLoader
import yaml

from hive_ecia.dataset import collate_ecia, load_embedding_cache
from hive_ecia.recovery_aware import build_from_checkpoint
from .cache import load_raw_candidate_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--collection-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def raw_safety_index(collection_run: Path, sequences: list[str]) -> dict[str, bool]:
    result = {}
    for sequence in sequences:
        payload = load_raw_candidate_cache(
            collection_run / "candidate_cache" / f"{sequence}.npz"
        )
        for value in payload["metadata"]:
            row = json.loads(str(value))
            result[str(row["event_id"])] = bool(row["safety_feasible"])
    return result


def policy_requests(row: dict, policy: dict) -> bool:
    direct = (
        row["base_predicted"] == 0
        and row["base_old_probability"] >= policy["direct_merge_min_old_probability"]
        and row["base_old_margin"] >= policy["direct_merge_min_old_margin"]
    )
    if direct:
        return False
    plausible = row["base_old_probability"] >= policy["request_min_old_probability"]
    uncertain = (
        row["base_old_probability"] <= policy["request_max_old_probability"]
        or row["base_competitor_margin"] <= policy["request_max_competitor_margin"]
        or row["base_entropy"] >= policy["request_min_normalized_entropy"]
    )
    allowed_top = row["base_identity_predicted"] == 0 or not policy["require_old_identity_top1"]
    return plausible and uncertain and allowed_top


def merge_from(row: dict, *, use_full: bool, probability: float, margin: float) -> bool:
    prefix = "full" if use_full else "base"
    return (
        row["safety_feasible"]
        and row[f"{prefix}_predicted"] == 0
        and row[f"{prefix}_old_probability"] >= probability
        and row[f"{prefix}_old_margin"] >= margin
    )


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    expected_sequences = list(map(str, config["data"]["calibration_sequences"]))
    required = (args.checkpoint, args.calibration_cache, args.collection_run / "run_manifest.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing calibration inputs: {missing}")
    if set(expected_sequences) & set(map(str, config["data"]["test_sequences"])):
        raise ValueError("Calibration overlaps QuadTrack test sequences")
    if set(expected_sequences) & set(map(str, config["data"]["validation_sequences"])):
        raise ValueError("Calibration must be video-disjoint from model validation")
    if args.validate_only:
        print("Calibration CLI and video-disjoint boundary valid; calibration not started.")
        return 0

    dataset = load_embedding_cache(args.calibration_cache)
    if {sample.sequence for sample in dataset.samples} != set(expected_sequences):
        raise ValueError("Calibration cache contains a non-calibration sequence")
    safety = raw_safety_index(args.collection_run, expected_sequences)
    if set(safety) != {sample.event_id for sample in dataset.samples}:
        raise ValueError("Raw safety metadata and labeled calibration events differ")

    device = torch.device(args.device or config["training"]["device"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_from_checkpoint(checkpoint, device).eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_ecia)
    rows = []
    with torch.inference_mode():
        for raw in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in raw.items()
            }
            base = model.forward_base(
                batch["old"], batch["target"], batch["competitors"],
                batch["competitor_mask"],
            )
            output = model.forward_from_base(base, batch["hr"], batch["hr_available"])
            base_decision = torch.softmax(base.base_decision_logits.float(), dim=-1)
            full_decision = torch.softmax(output.decision_logits.float(), dim=-1)
            identity = model._masked_softmax(base.base_scores.float(), base.hypothesis_mask)
            for index, event_id in enumerate(batch["event_id"]):
                valid = int(base.hypothesis_mask[index].sum())
                best_competitor = float(identity[index, 1:valid].max()) if valid > 1 else 0.0
                target_merge = bool(batch["merge_target"][index].item())
                rows.append(
                    {
                        "event_id": event_id,
                        "sequence": str(batch["sequence"][index]),
                        "target_merge": target_merge,
                        "safety_feasible": safety[event_id],
                        "hr_available": bool(batch["hr_available"][index]),
                        "base_predicted": int(base_decision[index].argmax()),
                        "base_identity_predicted": int(identity[index].argmax()),
                        "base_old_probability": float(base_decision[index, 0]),
                        "base_old_margin": float(base_decision[index, 0] - base_decision[index, 1:].max()),
                        "base_competitor_margin": float(identity[index, 0]) - best_competitor,
                        "base_entropy": float(base.normalized_entropy[index]),
                        "full_predicted": int(full_decision[index].argmax()),
                        "full_old_probability": float(full_decision[index, 0]),
                        "full_old_margin": float(full_decision[index, 0] - full_decision[index, 1:].max()),
                    }
                )

    calibration = config["calibration"]
    positives = sum(row["target_merge"] for row in rows)
    minimum_events = int(calibration.get("min_events", 1))
    minimum_positives = int(calibration.get("min_positive_merges", 1))
    if len(rows) < minimum_events or positives < minimum_positives:
        raise RuntimeError(
            "Calibration support is too small: "
            f"events={len(rows)} (min={minimum_events}), "
            f"positive_merges={positives} (min={minimum_positives})"
        )
    grid = []
    final_probability_grid = list(calibration["final_probability_grid"])
    final_margin_grid = list(calibration["final_margin_grid"])
    # Runs created before the balanced-v2 grid fix have a materialized config
    # that ends at p=0.80/margin=0.20. Extend only balanced-v2 checkpoints at
    # runtime so a completed training run can resume calibration without
    # mutating or retraining its checkpoint.
    if checkpoint.get("architecture") == "online_aware_recovery_ecia_balanced_v2":
        final_probability_grid = sorted(set(map(float, final_probability_grid)) | {0.95})
        final_margin_grid = sorted(set(map(float, final_margin_grid)) | {0.40})
    dimensions = (
        calibration["direct_probability_grid"], calibration["direct_margin_grid"],
        calibration["request_min_probability_grid"], final_probability_grid,
        final_margin_grid,
    )
    def score_rows(selected_rows, policy, final_probability, final_margin):
        tp = fp = fn = requests = actual_hr = 0
        for row in selected_rows:
            request = policy_requests(row, policy)
            requests += int(request)
            use_full = request and row["safety_feasible"] and row["hr_available"]
            actual_hr += int(use_full)
            merge = False if request and not use_full else merge_from(
                row,
                use_full=use_full,
                probability=float(final_probability),
                margin=float(final_margin),
            )
            tp += int(merge and row["target_merge"])
            fp += int(merge and not row["target_merge"])
            fn += int(not merge and row["target_merge"])
        row_positives = sum(row["target_merge"] for row in selected_rows)
        row_negatives = len(selected_rows) - row_positives
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        wrong_rate = fp / max(1, row_negatives)
        objective = (
            f1
            - float(calibration["false_merge_penalty"]) * wrong_rate
            - float(calibration["missed_merge_penalty"]) * fn / max(1, row_positives)
            - float(calibration["hr_request_penalty"]) * actual_hr / max(1, len(selected_rows))
        )
        return {
            "objective": objective,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "wrong_merge_rate": wrong_rate,
            "would_request_hr": requests,
            "actual_hr_requests": actual_hr,
            "events": len(selected_rows),
            "positive_merges": row_positives,
        }

    for direct_probability, direct_margin, request_min, final_probability, final_margin in itertools.product(*dimensions):
        policy = {
            "direct_merge_min_old_probability": float(direct_probability),
            "direct_merge_min_old_margin": float(direct_margin),
            "request_min_old_probability": float(request_min),
            "request_max_old_probability": float(calibration["request_max_old_probability"]),
            "request_max_competitor_margin": float(calibration["request_max_competitor_margin"]),
            "request_min_normalized_entropy": float(calibration["request_min_normalized_entropy"]),
            "require_old_identity_top1": bool(calibration["require_old_identity_top1"]),
        }
        aggregate = score_rows(rows, policy, final_probability, final_margin)
        per_sequence = {
            sequence: score_rows(
                [row for row in rows if row["sequence"] == sequence],
                policy,
                final_probability,
                final_margin,
            )
            for sequence in expected_sequences
        }
        sequence_objectives = [row["objective"] for row in per_sequence.values()]
        robust_objective = (
            statistics.fmean(sequence_objectives)
            - float(calibration.get("per_video_std_penalty", 0.0))
            * (statistics.pstdev(sequence_objectives) if len(sequence_objectives) > 1 else 0.0)
        )
        eligible = (
            aggregate["recall"] >= float(calibration.get("min_merge_recall", 0.0))
            and aggregate["wrong_merge_rate"]
            <= float(calibration.get("max_wrong_merge_rate", 1.0))
            and aggregate["fp"] <= int(calibration.get("max_wrong_merges", 10**9))
        )
        grid.append(
            {
                "eligible": eligible,
                "base_guided_policy": policy,
                "selected_thresholds": {
                    "min_old_probability": float(final_probability),
                    "min_old_margin": float(final_margin),
                },
                **{key: value for key, value in aggregate.items() if key != "objective"},
                "aggregate_objective": aggregate["objective"],
                "objective": robust_objective,
                "per_sequence": per_sequence,
            }
        )
    grid.sort(
        key=lambda row: (
            row["eligible"], row["objective"], -row["fp"], row["f1"],
            -row["actual_hr_requests"],
        ),
        reverse=True,
    )
    best = grid[0]
    if not best["eligible"]:
        raise RuntimeError(
            "No threshold combination passed calibration Merge Recall/Wrong Merge constraints"
        )
    payload = {
        "selection_split": "quadtrack_video_disjoint_calibration",
        "calibration_sequences": expected_sequences,
        "checkpoint": str(args.checkpoint.resolve()),
        "gt_usage": "offline target_class calibration only; never inference input",
        "test_data_accessed": False,
        "dtb_data_accessed": False,
        "events": len(rows),
        "positive_executable_merges": positives,
        "anti_overfit": {
            "coarse_grid_size": len(grid),
            "minimum_events": minimum_events,
            "minimum_positive_merges": minimum_positives,
            "per_video_robust_objective": True,
            "per_video_std_penalty": float(
                calibration.get("per_video_std_penalty", 0.0)
            ),
        },
        "selected_thresholds": best["selected_thresholds"],
        "base_guided_policy": best["base_guided_policy"],
        "selected_result": {key: value for key, value in best.items() if key not in {"selected_thresholds", "base_guided_policy"}},
        "grid": grid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"thresholds": payload["selected_thresholds"], "policy": payload["base_guided_policy"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
