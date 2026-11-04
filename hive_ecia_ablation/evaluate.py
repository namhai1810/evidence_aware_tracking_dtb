"""Evaluate one frozen A0--A3 dataset ablation and write auditable artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_mot import load_mot
from hive_hr_reid.run_experiment import basic_metrics
from mot_baseline_benchmark.protocol import resolve_sequence_dir

from .bandwidth import dataset_bandwidth
from .protocol import (
    EXPECTED_METRICS,
    EXPECTED_MODES,
    VARIANTS,
    baseline_timing,
    logged_events,
    mot_equivalent,
    mot_path,
    normalized_protocol,
    online_timing,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=("quadtrack", "dtb"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--a0-run", type=Path, required=True)
    parser.add_argument("--a1-run", type=Path, required=True)
    parser.add_argument("--a2-run", type=Path, required=True)
    parser.add_argument("--a3-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lr-size", required=True, help="WIDTHxHEIGHT")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    return parser.parse_args()


def metric_subset(result: dict) -> dict:
    row = result["aggregate"]
    return {key: row[key] for key in ("id_switches", "idf1", "hota", "assa", "mota", "idtp", "idfp", "idfn")}


def expected_check(actual: dict, expected: dict) -> dict:
    differences = {}
    for key, value in expected.items():
        if key == "id_switches":
            matches = int(actual[key]) == int(value)
        else:
            matches = abs(float(actual[key]) - float(value)) <= 5e-5
        if not matches:
            differences[key] = {"expected": value, "actual": actual[key]}
    return {"passed": not differences, "differences": differences}


def event_summary(run: Path, sequences: list[str]) -> dict:
    rows = logged_events(run, sequences)
    by_type = {}
    for row in rows:
        by_type[row["type"]] = by_type.get(row["type"], 0) + 1
    return {
        "events": len(rows),
        "event_types": by_type,
        "would_request_hr": sum(bool(row.get("would_request_hr")) for row in rows),
        "actual_hr_requests": sum(bool(row.get("decision", {}).get("hr", {}).get("requested")) for row in rows),
    }


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    width, height = (int(part) for part in args.lr_size.lower().split("x", 1))
    runs = {"a0": args.a0_run.resolve(), "a1": args.a1_run.resolve(), "a2": args.a2_run.resolve(), "a3": args.a3_run.resolve()}

    for variant, mode in EXPECTED_MODES.items():
        manifest = json.loads((runs[variant] / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("mode") != mode:
            raise RuntimeError(f"{variant.upper()} must use mode={mode}, got {manifest.get('mode')}")

    protocols = {variant: normalized_protocol(variant, run) for variant, run in runs.items()}
    protocol_paired = len({json.dumps(row, sort_keys=True) for row in protocols.values()}) == 1
    if not protocol_paired:
        raise RuntimeError(f"Ablation protocols differ: {protocols}")

    gt = {
        sequence: load_mot(resolve_sequence_dir(args.dataset_root.resolve(), sequence) / "gt" / "gt.txt", ground_truth=True)
        for sequence in args.sequences
    }
    results = {}
    for variant, run in runs.items():
        predictions = {sequence: load_mot(mot_path(variant, run, sequence), ground_truth=False) for sequence in args.sequences}
        results[variant] = basic_metrics(argparse.Namespace(iou=args.iou), args.sequences, predictions, gt, include_hota=True)
        results[variant]["timing"] = baseline_timing(run, args.sequences) if variant == "a0" else online_timing(run, args.sequences)

    raw_checks = {}
    for variant in ("a1", "a2", "a3"):
        raw_checks[variant] = {}
        for sequence in args.sequences:
            equivalent, differences = mot_equivalent(
                mot_path("a0", runs["a0"], sequence),
                mot_path(variant, runs[variant], sequence, raw=True),
            )
            raw_checks[variant][sequence] = {"equivalent": equivalent, "differences": differences}
    raw_paired = all(row["equivalent"] for variant in raw_checks.values() for row in variant.values())

    a2_a3_hashes = {
        variant: {sequence: sha256(mot_path(variant, runs[variant], sequence)) for sequence in args.sequences}
        for variant in ("a2", "a3")
    }
    identical_a2_a3 = {sequence: a2_a3_hashes["a2"][sequence] == a2_a3_hashes["a3"][sequence] for sequence in args.sequences}
    a2_a3_metric_identical = metric_subset(results["a2"]) == metric_subset(results["a3"])

    bandwidth = dataset_bandwidth(
        dataset_root=args.dataset_root.resolve(), sequences=args.sequences,
        lr_size=(width, height), quality=args.jpeg_quality, fps=args.fps,
        online_runs={key: runs[key] for key in ("a1", "a2", "a3")},
    )
    reference = {
        variant: expected_check(metric_subset(results[variant]), EXPECTED_METRICS[args.dataset_key][variant])
        for variant in ("a0", "a3")
    }
    a3_fewer_hr = bandwidth["a3"]["hr_requests"] < bandwidth["a2"]["hr_requests"]
    verification = {
        "protocol_paired": protocol_paired,
        "raw_botsort_equivalent_to_a0": raw_paired,
        "a0_reference_reproduced": reference["a0"],
        "a3_reference_reproduced": reference["a3"],
        "a2_a3_byte_identical_by_sequence": identical_a2_a3,
        "a2_a3_metrics_identical": a2_a3_metric_identical,
        "a3_uses_fewer_hr_requests": a3_fewer_hr,
        "a2_a3_same_predictions_and_a3_fewer_hr": all(identical_a2_a3.values()) and a3_fewer_hr,
        "gt_used_only_for_posthoc_evaluation": True,
    }
    verification["required_checks_passed"] = (
        raw_paired and reference["a0"]["passed"] and reference["a3"]["passed"]
    )

    payload = {
        "dataset": args.dataset_key,
        "protocol": protocols["a0"],
        "variants": VARIANTS,
        "verification": verification,
        "events": {variant: event_summary(runs[variant], args.sequences) for variant in ("a1", "a2", "a3")},
        "methods": results,
        "bandwidth": bandwidth,
        "raw_checks": raw_checks,
        "prediction_sha256": a2_a3_hashes,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Frozen HIVE–ECIA ablation — {protocols['a0'].get('physical_lr_size')}", "",
        f"Required verification: **{'PASS' if verification['required_checks_passed'] else 'FAIL'}**", "",
        "| Variant | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | MOTA ↑ | HR requests | Encoded HR MB | Total MB | FPS ↑ | Latency ms ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        metric = results[variant]["aggregate"]
        timing = results[variant]["timing"]
        bw = bandwidth[variant]
        lines.append(
            f"| {variant.upper()} | {metric['id_switches']} | {metric['idf1']:.4f} | {metric['hota']:.4f} | {metric['assa']:.4f} | {metric['mota']:.4f} | "
            f"{bw['hr_requests']} | {bw['hr_bytes']/1_000_000:.3f} | {bw['total_mb']:.3f} | {timing['fps']:.2f} | {timing['latency_ms']:.2f} |"
        )
    lines += [
        "", "## Verification", "",
        f"- A0 reference reproduced: `{reference['a0']['passed']}`.",
        f"- A3 reference reproduced: `{reference['a3']['passed']}`.",
        f"- A1/A2/A3 raw BoT-SORT equivalent to cached A0: `{raw_paired}`.",
        f"- A2/A3 byte-identical final MOT: `{all(identical_a2_a3.values())}`; per sequence: `{identical_a2_a3}`.",
        f"- A2/A3 aggregate metrics identical: `{a2_a3_metric_identical}`.",
        f"- A3 uses fewer HR requests: `{a3_fewer_hr}` ({bandwidth['a2']['hr_requests']} → {bandwidth['a3']['hr_requests']}).",
        "", "A2 intentionally uses the original rule-based ambiguity trigger. A3 changes candidate proposal/request policy to ECIA Base-guided gating and adds pre-query/final registry safety; therefore candidate pools are not asserted identical.",
        "", "## Per-sequence tracking", "",
    ]
    for sequence in args.sequences:
        lines += [f"### {sequence}", "", "| Variant | IDSW | IDF1 | HOTA | AssA | MOTA |", "|---|---:|---:|---:|---:|---:|"]
        for variant in VARIANTS:
            row = results[variant]["sequences"][sequence]
            lines.append(f"| {variant.upper()} | {row['id_switches']} | {row['idf1']:.4f} | {row['hota']:.4f} | {row['assa']:.4f} | {row['mota']:.4f} |")
        lines.append("")
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Dataset ablation report: {(args.output_dir / 'REPORT.md').resolve()}")
    return 0 if verification["required_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

