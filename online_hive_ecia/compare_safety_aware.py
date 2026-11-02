"""Compare Base-Guided Full with its pre-query safety optimization.

The comparison is deliberately paired.  It checks raw BoT-SORT predictions,
the complete Base-guided proposal/request schedule, and final canonical MOT
predictions before reporting the HR I/O saved by the safety-aware run.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

from .compare import resolve_sequence_dir, timing_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--safety-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.50)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_events(run: Path, sequences: list[str]) -> list[dict]:
    events = []
    for sequence in sequences:
        path = run / "diagnostics" / f"{sequence}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("type") != "base_guided_event":
                continue
            row["sequence"] = sequence
            events.append(row)
    return events


def event_key(row: dict) -> tuple[str, int, str, str]:
    return (
        str(row["sequence"]),
        int(row["frame"]),
        str(row["old_token"]),
        str(row["new_token"]),
    )


def request_schedule(events: list[dict]) -> list[tuple[str, int, str, str]]:
    return [event_key(row) for row in events if bool(row.get("would_request_hr"))]


def mot_line_differences(left: Path, right: Path, limit: int = 100) -> list[dict]:
    left_lines = left.read_text(encoding="utf-8").splitlines()
    right_lines = right.read_text(encoding="utf-8").splitlines()
    differences = []
    for line_number, (before, after) in enumerate(
        itertools.zip_longest(left_lines, right_lines), start=1
    ):
        if before == after:
            continue
        differences.append(
            {"line": line_number, "baseline": before, "safety_aware": after}
        )
        if len(differences) >= limit:
            break
    return differences


def event_differences(baseline: list[dict], safety: list[dict]) -> list[dict]:
    baseline_by_key = {event_key(row): row for row in baseline}
    safety_by_key = {event_key(row): row for row in safety}
    differences = []
    for key in sorted(set(baseline_by_key) | set(safety_by_key)):
        before = baseline_by_key.get(key)
        after = safety_by_key.get(key)
        if before is None or after is None:
            differences.append(
                {
                    "event": key,
                    "reason": "event_missing_from_one_run",
                    "baseline_present": before is not None,
                    "safety_present": after is not None,
                }
            )
            continue
        before_decision = before.get("decision", {})
        after_decision = after.get("decision", {})
        before_safety = before.get("safety", {})
        after_safety = after.get("safety", {})
        fields = {
            "would_request_hr": (
                before.get("would_request_hr"), after.get("would_request_hr")
            ),
            "merge_decision": (
                before_decision.get("merge"), after_decision.get("merge")
            ),
            "decision_reason": (
                before_decision.get("reason"), after_decision.get("reason")
            ),
            "final_safety_applied": (
                before_safety.get("applied"), after_safety.get("applied")
            ),
            "final_safety_reason": (
                before_safety.get("reason"), after_safety.get("reason")
            ),
        }
        changed = {name: value for name, value in fields.items() if value[0] != value[1]}
        # A safety skip intentionally changes the internal decision/reason but
        # must preserve the externally applied merge outcome.
        if changed:
            differences.append(
                {
                    "event": key,
                    "changed": changed,
                    "pre_query_safety": after.get("pre_query_safety"),
                }
            )
    return differences


def paired_protocol(manifests: dict[str, dict]) -> tuple[dict, dict]:
    fields = (
        "dataset_root", "dataset_label", "sequences", "detector",
        "tracker_config", "physical_lr_size", "imgsz", "conf", "iou",
    )
    values = {
        method: {field: manifest.get(field) for field in fields}
        for method, manifest in manifests.items()
    }
    return values["base_guided_full"], values


def metric_row(result: dict) -> dict:
    aggregate = result["aggregate"]
    return {
        key: aggregate[key]
        for key in ("id_switches", "idf1", "hota", "assa", "mota")
    }


def main() -> int:
    args = parse_args()
    from evaluate_mot import load_mot
    from hive_hr_reid.run_experiment import basic_metrics

    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    runs = {
        "base_guided_full": args.baseline_run.resolve(),
        "safety_aware_full": args.safety_run.resolve(),
    }
    manifests = {
        method: json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        for method, run in runs.items()
    }
    expected_modes = {
        "base_guided_full": "base_guided_online",
        "safety_aware_full": "base_guided_safety_online",
    }
    actual_modes = {method: manifest.get("mode") for method, manifest in manifests.items()}
    if actual_modes != expected_modes:
        raise RuntimeError(f"Unexpected modes: {actual_modes}; expected {expected_modes}")

    protocol, protocol_values = paired_protocol(manifests)
    protocol_paired = len(
        {json.dumps(value, sort_keys=True) for value in protocol_values.values()}
    ) == 1
    if not protocol_paired:
        raise RuntimeError(f"Run protocols differ: {protocol_values}")

    raw_hashes = {
        method: {
            sequence: sha256(run / "raw_mot" / f"{sequence}.txt")
            for sequence in args.sequences
        }
        for method, run in runs.items()
    }
    identical_raw = all(
        raw_hashes["base_guided_full"][sequence]
        == raw_hashes["safety_aware_full"][sequence]
        for sequence in args.sequences
    )
    raw_mot_differences = {
        sequence: mot_line_differences(
            runs["base_guided_full"] / "raw_mot" / f"{sequence}.txt",
            runs["safety_aware_full"] / "raw_mot" / f"{sequence}.txt",
        )
        for sequence in args.sequences
        if raw_hashes["base_guided_full"][sequence]
        != raw_hashes["safety_aware_full"][sequence]
    }

    events = {method: load_events(run, args.sequences) for method, run in runs.items()}
    proposal_paired = [event_key(row) for row in events["base_guided_full"]] == [
        event_key(row) for row in events["safety_aware_full"]
    ]
    request_paired = request_schedule(events["base_guided_full"]) == request_schedule(
        events["safety_aware_full"]
    )
    skipped = [
        row for row in events["safety_aware_full"]
        if bool(row.get("pre_query_safety", {}).get("checked"))
        and not bool(row.get("pre_query_safety", {}).get("feasible"))
    ]
    skip_reasons = Counter(
        str(row["pre_query_safety"].get("reason", "unknown")) for row in skipped
    )

    mot_hashes = {
        method: {
            sequence: sha256(run / "mot" / f"{sequence}.txt")
            for sequence in args.sequences
        }
        for method, run in runs.items()
    }
    identical_mot_by_sequence = {
        sequence: mot_hashes["base_guided_full"][sequence]
        == mot_hashes["safety_aware_full"][sequence]
        for sequence in args.sequences
    }
    mot_differences = {
        sequence: mot_line_differences(
            runs["base_guided_full"] / "mot" / f"{sequence}.txt",
            runs["safety_aware_full"] / "mot" / f"{sequence}.txt",
        )
        for sequence in args.sequences
        if not identical_mot_by_sequence[sequence]
    }

    gt = {
        sequence: load_mot(
            resolve_sequence_dir(args.dataset_root.resolve(), sequence) / "gt" / "gt.txt",
            ground_truth=True,
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

    tracking_metrics_identical = metric_row(metrics["base_guided_full"]) == metric_row(
        metrics["safety_aware_full"]
    )
    internal_event_differences = event_differences(
        events["base_guided_full"], events["safety_aware_full"]
    )
    baseline_hr = metrics["base_guided_full"]["timing"]["hr_cost"]
    safety_hr = metrics["safety_aware_full"]["timing"]["hr_cost"]
    saved_requests = baseline_hr["requests"] - safety_hr["requests"]
    reduction = saved_requests / baseline_hr["requests"] if baseline_hr["requests"] else 0.0
    request_accounting_exact = saved_requests == len(skipped)
    hr_requests_not_increased = saved_requests >= 0
    verification_passed = all(
        (
            protocol_paired,
            identical_raw,
            proposal_paired,
            request_paired,
            all(identical_mot_by_sequence.values()),
            tracking_metrics_identical,
            request_accounting_exact,
            hr_requests_not_increased,
        )
    )
    optimization_goal_met = verification_passed and saved_requests > 0

    payload = {
        "protocol": {
            "paired_settings": protocol,
            "iou": args.iou,
            "gt_used_only_for_posthoc_evaluation": True,
            "identical_raw_botsort_predictions": identical_raw,
            "identical_base_proposed_events": proposal_paired,
            "identical_base_hr_request_schedule": request_paired,
            "identical_final_mot_by_sequence": identical_mot_by_sequence,
            "identical_tracking_metrics": tracking_metrics_identical,
            "hr_skip_accounting_exact": request_accounting_exact,
            "hr_requests_not_increased": hr_requests_not_increased,
            "verification_passed": verification_passed,
            "optimization_goal_met": optimization_goal_met,
            "raw_prediction_sha256": raw_hashes,
            "mot_prediction_sha256": mot_hashes,
        },
        "savings": {
            "baseline_hr_requests": baseline_hr["requests"],
            "safety_hr_requests": safety_hr["requests"],
            "saved_hr_requests": saved_requests,
            "hr_request_reduction": reduction,
            "prequery_skips": len(skipped),
            "prequery_skip_reasons": dict(skip_reasons),
            "bytes_saved": baseline_hr["bytes_read"] - safety_hr["bytes_read"],
            "crop_pixels_saved": baseline_hr["crop_pixels"] - safety_hr["crop_pixels"],
        },
        "methods": metrics,
        "differences": {
            "raw_mot_lines": raw_mot_differences,
            "mot_lines": mot_differences,
            "events": internal_event_differences,
            "skipped_events": [
                {
                    "event": event_key(row),
                    "reason": row["pre_query_safety"].get("reason"),
                    "pair_kind": row.get("pair_kind"),
                }
                for row in skipped
            ],
        },
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    status = "PASS" if verification_passed else "FAIL"
    lines = [
        f"# Pre-Query Safety Check — {protocol['dataset_label']}", "",
        f"Verification: **{status}**", "",
        (
            "PASS requires identical raw BoT-SORT predictions, Base candidate/request "
            "schedules, final MOT files and tracking metrics. GT is used only after "
            "prediction for evaluation."
        ), "",
        "## Tracking equivalence and HR cost", "",
        "| Method | IDSW | IDF1 | HOTA | AssA | MOTA | HR requests | HR crops | Bytes read | Mean latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("base_guided_full", "safety_aware_full"):
        result = metrics[method]
        aggregate = result["aggregate"]
        timing = result["timing"]
        hr = timing["hr_cost"]
        lines.append(
            f"| {method} | {aggregate['id_switches']} | {aggregate['idf1']:.4f} | "
            f"{aggregate['hota']:.4f} | {aggregate['assa']:.4f} | "
            f"{aggregate['mota']:.4f} | {hr['requests']} | {hr['crops']} | "
            f"{hr['bytes_read']} | {timing['mean_latency_ms']:.2f} |"
        )
    lines.extend([
        "", "## Savings", "",
        f"- HR requests saved: **{saved_requests}** ({reduction:.2%}).",
        f"- HR bytes saved: **{payload['savings']['bytes_saved']}**.",
        f"- Pre-query skips: **{len(skipped)}**; reasons: `{dict(skip_reasons)}`.",
        f"- Skip/request accounting exact: `{request_accounting_exact}`.",
        f"- Optimization goal (same output and fewer HR requests): `{optimization_goal_met}`.",
        "", "## Equivalence checks", "",
        f"- Raw BoT-SORT byte-identical: `{identical_raw}`.",
        f"- Base proposal pool identical: `{proposal_paired}`.",
        f"- Base would-request-HR schedule identical: `{request_paired}`.",
        f"- Final MOT byte-identical per sequence: `{identical_mot_by_sequence}`.",
        f"- Aggregate tracking metrics identical: `{tracking_metrics_identical}`.",
    ])
    if not verification_passed:
        lines.extend([
            "", "## Exact divergences", "",
            "See `metrics.json -> differences` for line-level raw/canonical MOT "
            "differences and event-level decision/safety reasons.",
        ])
    lines.extend(["", "## Per-sequence tracking", ""])
    for sequence in args.sequences:
        lines.extend([
            f"### {sequence}", "",
            "| Method | IDSW | IDF1 | HOTA | AssA | MOTA |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for method in ("base_guided_full", "safety_aware_full"):
            row = metrics[method]["sequences"][sequence]
            lines.append(
                f"| {method} | {row['id_switches']} | {row['idf1']:.4f} | "
                f"{row['hota']:.4f} | {row['assa']:.4f} | {row['mota']:.4f} |"
            )
        lines.append("")
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Safety comparison: {(args.output_dir / 'REPORT.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
