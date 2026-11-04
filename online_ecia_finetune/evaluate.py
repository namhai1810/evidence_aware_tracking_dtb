"""Paired full-video evaluation for online-aware ECIA and frozen baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from online_hive_ecia.compare import resolve_sequence_dir, sha256, timing_summary, trigger_pool


METHODS = (
    "botsort_baseline",
    "rule_based_full",
    "existing_base_guided_safety",
    "fine_tuned_hr_disabled",
    "fine_tuned_safety_full",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--rule-run", type=Path, required=True)
    parser.add_argument("--existing-safety-run", type=Path, required=True)
    parser.add_argument("--fine-hr-disabled-run", type=Path, required=True)
    parser.add_argument("--fine-full-run", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.50)
    return parser.parse_args()


def protocol(manifest: dict) -> dict:
    fields = (
        "dataset_root", "dataset_label", "sequences", "detector", "tracker_config",
        "physical_lr_size", "imgsz", "conf", "iou",
    )
    return {key: manifest.get(key) for key in fields}


def _raw_id(token: str) -> int:
    return int(str(token).split(":", 1)[0])


def identity_recovery_metrics(run: Path, sequences: list[str], gt, raw_predictions, iou: float) -> dict:
    """Measure exact applied recovery merges against post-hoc oracle IDSW.

    GT is consumed only after inference artifacts are frozen. A wrong merge is
    an applied online identity transfer that is not an exact raw IDSW tuple.
    """

    from rank_highres_retrieval_frames import build_match_index, find_switch_events

    totals = {"oracle_switches": 0, "applied_merges": 0, "true_merges": 0, "wrong_merges": 0}
    per_sequence = {}
    for sequence in sequences:
        matches, _ = build_match_index(gt[sequence], raw_predictions[sequence], iou)
        oracle = {
            (
                int(row["target_frame"]),
                int(row["lr_old_pred_id"]),
                int(row["lr_new_pred_id"]),
            )
            for row in find_switch_events(matches)
        }
        applied = set()
        diagnostic = run / "diagnostics" / f"{sequence}.jsonl"
        for line in diagnostic.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("type") not in {"base_guided_event", "recovery_event"}:
                continue
            if bool(row.get("safety", {}).get("applied")):
                applied.add(
                    (
                        int(row["frame"]),
                        _raw_id(row["old_token"]),
                        _raw_id(row["new_token"]),
                    )
                )
        true_merges = len(applied & oracle)
        wrong_merges = len(applied - oracle)
        item = {
            "oracle_switches": len(oracle),
            "applied_merges": len(applied),
            "true_merges": true_merges,
            "wrong_merges": wrong_merges,
            "merge_recall": true_merges / len(oracle) if oracle else 0.0,
            "merge_precision": true_merges / len(applied) if applied else 0.0,
        }
        per_sequence[sequence] = item
        for key in totals:
            totals[key] += item[key]
    totals["merge_recall"] = (
        totals["true_merges"] / totals["oracle_switches"]
        if totals["oracle_switches"] else 0.0
    )
    totals["merge_precision"] = (
        totals["true_merges"] / totals["applied_merges"]
        if totals["applied_merges"] else 0.0
    )
    return {
        "definition": "exact (sequence, frame, old raw ID, new raw ID) applied merge",
        "gt_usage": "posthoc evaluation only",
        "aggregate": totals,
        "sequences": per_sequence,
    }


def main() -> int:
    args = parse_args()
    from evaluate_mot import load_mot
    from hive_hr_reid.run_experiment import basic_metrics

    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    runs = {
        "botsort_baseline": args.baseline_run.resolve(),
        "rule_based_full": args.rule_run.resolve(),
        "existing_base_guided_safety": args.existing_safety_run.resolve(),
        "fine_tuned_hr_disabled": args.fine_hr_disabled_run.resolve(),
        "fine_tuned_safety_full": args.fine_full_run.resolve(),
    }
    expected_modes = {
        "botsort_baseline": "baseline",
        "rule_based_full": "online",
        "existing_base_guided_safety": "base_guided_safety_online",
        "fine_tuned_hr_disabled": "base_guided_hr_disabled",
        "fine_tuned_safety_full": "base_guided_safety_online",
    }
    manifests = {
        method: json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        for method, run in runs.items()
    }
    for method, expected in expected_modes.items():
        if manifests[method].get("mode") != expected:
            raise ValueError(f"{method} must use mode={expected}")
    protocols = {method: protocol(manifest) for method, manifest in manifests.items()}
    if len({json.dumps(value, sort_keys=True) for value in protocols.values()}) != 1:
        raise RuntimeError(f"Detector/tracker protocols differ: {protocols}")

    raw_hashes = {
        method: {
            sequence: sha256(run / "raw_mot" / f"{sequence}.txt")
            for sequence in args.sequences
        }
        for method, run in runs.items()
    }
    identical_raw = all(
        len({raw_hashes[method][sequence] for method in METHODS}) == 1
        for sequence in args.sequences
    )
    if not identical_raw:
        raise RuntimeError("Raw detector/BoT-SORT MOT differs across paired methods")

    fine_hr0_events = trigger_pool(
        runs["fine_tuned_hr_disabled"], args.sequences,
        event_type="base_guided_event",
    )
    fine_full_events = trigger_pool(
        runs["fine_tuned_safety_full"], args.sequences,
        event_type="base_guided_event",
    )
    fine_hr0_requests = trigger_pool(
        runs["fine_tuned_hr_disabled"], args.sequences,
        event_type="base_guided_event", request_only=True,
    )
    fine_full_requests = trigger_pool(
        runs["fine_tuned_safety_full"], args.sequences,
        event_type="base_guided_event", request_only=True,
    )
    identical_event_pool = fine_hr0_events == fine_full_events
    identical_request_schedule = fine_hr0_requests == fine_full_requests
    if not identical_event_pool or not identical_request_schedule:
        raise RuntimeError("Fine-tuned HR-on/off is not a paired event-schedule ablation")

    leakage = json.loads(
        (args.training_root / "leakage_audit.json").read_text(encoding="utf-8")
    )
    thresholds = json.loads(
        (args.training_root / "calibration/thresholds.json").read_text(encoding="utf-8")
    )
    if leakage.get("status") != "passed" or leakage.get("test_sequences_accessed") is not False:
        raise RuntimeError("Training leakage audit did not pass")
    calibration_sequences = set(map(str, thresholds.get("calibration_sequences", [])))
    leakage_splits = leakage.get("splits", {})
    if (
        not calibration_sequences
        or calibration_sequences & set(map(str, leakage_splits.get("test", [])))
        or calibration_sequences & set(map(str, leakage_splits.get("validation", [])))
        or thresholds.get("dtb_data_accessed") is not False
    ):
        raise RuntimeError("Threshold calibration is not video-disjoint from validation/test")

    gt = {
        sequence: load_mot(
            resolve_sequence_dir(args.dataset_root.resolve(), sequence) / "gt" / "gt.txt",
            ground_truth=True,
        )
        for sequence in args.sequences
    }
    raw_predictions = {
        sequence: load_mot(
            runs["botsort_baseline"] / "raw_mot" / f"{sequence}.txt",
            ground_truth=False,
        )
        for sequence in args.sequences
    }
    metrics = {}
    for method, run in runs.items():
        predictions = {
            sequence: load_mot(run / "mot" / f"{sequence}.txt", ground_truth=False)
            for sequence in args.sequences
        }
        metrics[method] = basic_metrics(
            argparse.Namespace(iou=args.iou), args.sequences, predictions, gt,
            include_hota=True,
        )
        metrics[method]["timing"] = timing_summary(run, args.sequences)
        metrics[method]["identity_recovery"] = identity_recovery_metrics(
            run, args.sequences, gt, raw_predictions, args.iou
        )

    payload = {
        "protocol": {
            "dataset_label": args.dataset_label,
            "settings": protocols["botsort_baseline"],
            "identical_raw_botsort_predictions": identical_raw,
            "identical_fine_tuned_event_pool": identical_event_pool,
            "identical_fine_tuned_hr_request_schedule": identical_request_schedule,
            "fine_tuned_proposed_events": len(fine_hr0_events),
            "fine_tuned_would_request_hr": len(fine_hr0_requests),
            "gt_used_only_for_posthoc_evaluation": True,
            "training_leakage_audit": str((args.training_root / "leakage_audit.json").resolve()),
            "threshold_selection_split": thresholds["selection_split"],
            "raw_prediction_sha256": raw_hashes,
        },
        "methods": metrics,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    columns = (
        "Method", "IDSW", "IDF1", "HOTA", "AssA", "MOTA", "IDTP", "IDFP",
        "IDFN", "Merge recall", "Wrong merge", "FPS", "Latency ms",
        "HR requests", "Bandwidth bytes",
    )
    lines = [
        f"# Online-aware ECIA — {args.dataset_label}", "",
        "All methods have byte-identical raw detector/BoT-SORT predictions. GT is loaded only after online predictions are complete.", "",
        f"Fine-tuned HR-on/off share `{len(fine_hr0_events)}` proposed events and `{len(fine_hr0_requests)}` would-request-HR events.", "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(columns) - 1)) + "|",
    ]
    for method in METHODS:
        aggregate = metrics[method]["aggregate"]
        timing = metrics[method]["timing"]
        recovery = metrics[method]["identity_recovery"]["aggregate"]
        hr = timing["hr_cost"]
        lines.append(
            "| " + " | ".join(
                [method, str(aggregate["id_switches"]), f"{aggregate['idf1']:.4f}",
                 f"{aggregate['hota']:.4f}", f"{aggregate['assa']:.4f}",
                 f"{aggregate['mota']:.4f}", str(aggregate["idtp"]),
                 str(aggregate["idfp"]), str(aggregate["idfn"]),
                 f"{recovery['merge_recall']:.4f}", str(recovery["wrong_merges"]),
                 f"{timing['fps']:.2f}",
                 f"{timing['mean_latency_ms']:.2f}", str(hr["requests"]),
                 str(hr["bytes_read"])]
            ) + " |"
        )
    lines.extend(["", "## Per-sequence metrics", ""])
    for sequence in args.sequences:
        lines.extend([
            f"### {sequence}", "",
            "| Method | IDSW | IDF1 | HOTA | AssA | MOTA | IDTP | IDFP | IDFN | Merge recall | Wrong merge |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for method in METHODS:
            row = metrics[method]["sequences"][sequence]
            recovery = metrics[method]["identity_recovery"]["sequences"][sequence]
            lines.append(
                f"| {method} | {row['id_switches']} | {row['idf1']:.4f} | "
                f"{row['hota']:.4f} | {row['assa']:.4f} | {row['mota']:.4f} | "
                f"{row['idtp']} | {row['idfp']} | {row['idfn']} | "
                f"{recovery['merge_recall']:.4f} | {recovery['wrong_merges']} |"
            )
        lines.append("")
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Dataset report: {(args.output_dir / 'REPORT.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
