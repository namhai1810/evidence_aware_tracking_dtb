"""Paired post-hoc evaluation of rule and ECIA-Base-guided online triggers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compare import (
    resolve_sequence_dir,
    sha256,
    timing_summary,
    trigger_accuracy,
    trigger_pool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--rule-run", type=Path, required=True)
    parser.add_argument("--base-hr-disabled-run", type=Path, required=True)
    parser.add_argument("--base-full-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from evaluate_mot import load_mot
    from hive_hr_reid.run_experiment import basic_metrics

    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    runs = {
        "rule_based_full": args.rule_run.resolve(),
        "base_guided_hr_disabled": args.base_hr_disabled_run.resolve(),
        "base_guided_full": args.base_full_run.resolve(),
    }
    manifests = {
        name: json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
        for name, path in runs.items()
    }
    expected_modes = {
        "rule_based_full": "online",
        "base_guided_hr_disabled": "base_guided_hr_disabled",
        "base_guided_full": "base_guided_online",
    }
    for name, mode in expected_modes.items():
        if manifests[name].get("mode") != mode:
            raise RuntimeError(f"{name} must use --mode {mode}")

    paired_fields = (
        "dataset_root", "dataset_label", "sequences", "detector",
        "tracker_config", "physical_lr_size", "imgsz", "conf", "iou",
    )
    protocols = {
        name: {field: manifest.get(field) for field in paired_fields}
        for name, manifest in manifests.items()
    }
    if len({json.dumps(row, sort_keys=True) for row in protocols.values()}) != 1:
        raise RuntimeError(f"Run protocols differ: {protocols}")

    raw_hashes = {
        name: {
            sequence: sha256(path / "raw_mot" / f"{sequence}.txt")
            for sequence in args.sequences
        }
        for name, path in runs.items()
    }
    if any(
        len({raw_hashes[name][sequence] for name in runs}) != 1
        for sequence in args.sequences
    ):
        raise RuntimeError("Raw BoT-SORT outputs are not byte-identical")

    # Candidate proposals and Base HR-request decisions must be identical in
    # the HR-on/off ablation; otherwise HR is not the only changed variable.
    hr0_all = trigger_pool(
        runs["base_guided_hr_disabled"], args.sequences,
        event_type="base_guided_event",
    )
    full_all = trigger_pool(
        runs["base_guided_full"], args.sequences,
        event_type="base_guided_event",
    )
    hr0_requests = trigger_pool(
        runs["base_guided_hr_disabled"], args.sequences,
        event_type="base_guided_event", request_only=True,
    )
    full_requests = trigger_pool(
        runs["base_guided_full"], args.sequences,
        event_type="base_guided_event", request_only=True,
    )
    if hr0_all != full_all:
        raise RuntimeError("Base-guided HR-on/off proposed event pools differ")
    if hr0_requests != full_requests:
        raise RuntimeError("Base-guided HR-on/off request schedules differ")

    gt = {
        sequence: load_mot(
            resolve_sequence_dir(args.dataset_root.resolve(), sequence) / "gt" / "gt.txt",
            ground_truth=True,
        )
        for sequence in args.sequences
    }
    raw = {
        sequence: load_mot(
            runs["rule_based_full"] / "raw_mot" / f"{sequence}.txt",
            ground_truth=False,
        )
        for sequence in args.sequences
    }
    trigger_metrics = {
        "rule_based": trigger_accuracy(
            run=runs["rule_based_full"], sequences=args.sequences,
            gt_cache=gt, raw_predictions=raw, iou=args.iou,
        ),
        "base_guided_hr_request": trigger_accuracy(
            run=runs["base_guided_full"], sequences=args.sequences,
            gt_cache=gt, raw_predictions=raw, iou=args.iou,
            event_type="base_guided_event", request_only=True,
        ),
    }
    metrics = {}
    for name, run in runs.items():
        predictions = {
            sequence: load_mot(run / "mot" / f"{sequence}.txt", ground_truth=False)
            for sequence in args.sequences
        }
        metrics[name] = basic_metrics(
            argparse.Namespace(iou=args.iou), args.sequences, predictions, gt,
            include_hota=True,
        )
        metrics[name]["timing"] = timing_summary(run, args.sequences)

    args.output_dir.mkdir(parents=True)
    payload = {
        "protocol": {
            "paired_settings": protocols["rule_based_full"],
            "identical_raw_botsort_predictions": True,
            "identical_base_guided_proposed_events": True,
            "identical_base_guided_hr_request_schedule": True,
            "base_guided_proposed_events": len(hr0_all),
            "base_guided_hr_requests": len(hr0_requests),
            "gt_used_only_for_posthoc_evaluation": True,
            "raw_prediction_sha256": raw_hashes,
        },
        "trigger_accuracy": trigger_metrics,
        "methods": metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    protocol = protocols["rule_based_full"]
    lines = [
        f"# Base-guided online HR trigger — {protocol['dataset_label']}", "",
        "All methods use byte-identical raw BoT-SORT predictions. GT is joined only after inference for metrics and trigger scoring.", "",
        f"Base-guided HR-on/off share `{len(hr0_all)}` proposed events and `{len(hr0_requests)}` would-request-HR events exactly.", "",
        "| Method | IDSW | IDF1 | HOTA | AssA | MOTA | FPS | Latency ms | HR requests | HR crops | Bandwidth bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in metrics.items():
        aggregate = result["aggregate"]
        timing = result["timing"]
        cost = timing["hr_cost"]
        lines.append(
            f"| {name} | {aggregate['id_switches']} | {aggregate['idf1']:.4f} | "
            f"{aggregate['hota']:.4f} | {aggregate['assa']:.4f} | {aggregate['mota']:.4f} | "
            f"{timing['fps']:.2f} | {timing['mean_latency_ms']:.2f} | "
            f"{cost['requests']} | {cost['crops']} | {cost['bytes_read']} |"
        )
    lines += ["", "## Trigger precision/recall", "",
              "| Trigger | Events | Oracle IDSW | TP | FP | FN | Precision | Recall | F1 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in trigger_metrics.items():
        row = result["aggregate"]
        lines.append(
            f"| {name} | {row['triggers']} | {row['oracle_switches']} | {row['tp']} | "
            f"{row['fp']} | {row['fn']} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} |"
        )
    lines += ["", "## Per-sequence tracking", ""]
    for sequence in args.sequences:
        lines += [f"### {sequence}", "", "| Method | IDSW | IDF1 | HOTA | AssA | MOTA |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, result in metrics.items():
            row = result["sequences"][sequence]
            lines.append(
                f"| {name} | {row['id_switches']} | {row['idf1']:.4f} | "
                f"{row['hota']:.4f} | {row['assa']:.4f} | {row['mota']:.4f} |"
            )
        lines.append("")
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Comparison: {(args.output_dir / 'REPORT.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
