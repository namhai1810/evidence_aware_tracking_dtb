"""Paired evaluation of G0 gating-off and frozen G1 gating-on."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_mot import load_mot
from hive_hr_reid.run_experiment import basic_metrics
from hive_ecia_ablation.bandwidth import hr_sequence, lr_sequence
from hive_ecia_ablation.protocol import mot_equivalent, online_timing, sha256
from mot_baseline_benchmark.protocol import resolve_sequence_dir


EXPECTED = {
    "quadtrack": {"id_switches": 78, "idf1": 0.4663, "hr_requests": 29},
    "dtb": {"id_switches": 89, "idf1": 0.3722, "hr_requests": 18},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=("quadtrack", "dtb"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--g0-run", type=Path, required=True)
    parser.add_argument("--g1-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lr-size", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    return parser.parse_args()


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


def candidate_key(row: dict) -> tuple[str, int, str, str, str]:
    return (
        str(row["sequence"]), int(row["frame"]), str(row["old_token"]),
        str(row["new_token"]), str(row.get("pair_kind", "unknown")),
    )


def unsafe_hr_violations(events: list[dict]) -> list[dict]:
    violations = []
    for row in events:
        safety = row.get("pre_query_safety", {})
        requested = bool(row.get("decision", {}).get("hr", {}).get("requested"))
        if bool(safety.get("checked")) and not bool(safety.get("feasible")) and requested:
            violations.append({"event": candidate_key(row), "reason": safety.get("reason")})
    return violations


def normalized_manifest(run: Path) -> dict:
    row = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    return {
        "dataset_root": row["dataset_root"], "sequences": row["sequences"],
        "detector": row["detector"], "tracker_config": row["tracker_config"],
        "physical_lr_size": row["physical_lr_size"], "imgsz": row["imgsz"],
        "conf": row["conf"], "iou": row["iou"],
        "frozen_models": row["frozen_models"],
        "base_guided": row["trigger_configuration"]["base_guided"],
    }


def bandwidth(*, dataset_root: Path, sequences: list[str], lr_size: tuple[int, int], quality: int, fps: float, runs: dict[str, Path]) -> dict:
    lr_rows = {sequence: lr_sequence(sequence=sequence, dataset_root=dataset_root, lr_size=lr_size, quality=quality) for sequence in sequences}
    lr_bytes = sum(row["lr_bytes"] for row in lr_rows.values())
    frames = sum(row["frames"] for row in lr_rows.values())
    duration = frames / fps
    result = {}
    for method, run in runs.items():
        rows = {sequence: hr_sequence(sequence=sequence, dataset_root=dataset_root, run=run, quality=quality) for sequence in sequences}
        hr_bytes = sum(row["hr_bytes"] for row in rows.values())
        total = lr_bytes + hr_bytes
        result[method] = {
            "lr_bytes": lr_bytes, "hr_bytes": hr_bytes, "total_bytes": total,
            "total_mb": total / 1e6,
            "mbps": 8.0 * total / 1e6 / duration if duration else 0.0,
            "hr_requests": sum(row["hr_requests"] for row in rows.values()),
            "hr_crops": sum(row["hr_crops"] for row in rows.values()),
            "jpeg_quality": quality,
            "measurement": "in-memory JPEG encoded payload; not network throughput",
        }
    return result


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    runs = {"g0_gating_off": args.g0_run.resolve(), "g1_gating_on": args.g1_run.resolve()}
    modes = {
        method: json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))["mode"]
        for method, run in runs.items()
    }
    expected_modes = {
        "g0_gating_off": "base_guided_gating_off_safety_online",
        "g1_gating_on": "base_guided_safety_online",
    }
    if modes != expected_modes:
        raise RuntimeError(f"Unexpected modes: {modes}; expected {expected_modes}")

    protocols = {method: normalized_manifest(run) for method, run in runs.items()}
    protocol_paired = protocols["g0_gating_off"] == protocols["g1_gating_on"]
    if not protocol_paired:
        raise RuntimeError(f"G0/G1 frozen protocols differ: {protocols}")

    events = {method: load_events(run, args.sequences) for method, run in runs.items()}
    pools = {method: [candidate_key(row) for row in rows] for method, rows in events.items()}
    candidate_pools_identical = pools["g0_gating_off"] == pools["g1_gating_on"]
    candidate_pool_difference = {
        "only_g0": [list(key) for key in sorted(set(pools["g0_gating_off"]) - set(pools["g1_gating_on"]))],
        "only_g1": [list(key) for key in sorted(set(pools["g1_gating_on"]) - set(pools["g0_gating_off"]))],
    }

    raw_checks = {}
    final_hashes = {}
    for sequence in args.sequences:
        equivalent, differences = mot_equivalent(
            runs["g0_gating_off"] / "raw_mot" / f"{sequence}.txt",
            runs["g1_gating_on"] / "raw_mot" / f"{sequence}.txt",
        )
        raw_checks[sequence] = {
            "semantic_equivalent": equivalent,
            "byte_identical": sha256(runs["g0_gating_off"] / "raw_mot" / f"{sequence}.txt") == sha256(runs["g1_gating_on"] / "raw_mot" / f"{sequence}.txt"),
            "differences": differences,
        }
        final_hashes[sequence] = {
            method: sha256(run / "mot" / f"{sequence}.txt") for method, run in runs.items()
        }
    raw_equivalent = all(row["semantic_equivalent"] for row in raw_checks.values())
    final_identical = {sequence: len(set(hashes.values())) == 1 for sequence, hashes in final_hashes.items()}

    gt = {sequence: load_mot(resolve_sequence_dir(args.dataset_root.resolve(), sequence) / "gt" / "gt.txt", ground_truth=True) for sequence in args.sequences}
    metrics = {}
    for method, run in runs.items():
        predictions = {sequence: load_mot(run / "mot" / f"{sequence}.txt", ground_truth=False) for sequence in args.sequences}
        metrics[method] = basic_metrics(argparse.Namespace(iou=args.iou), args.sequences, predictions, gt, include_hota=True)
        metrics[method]["timing"] = online_timing(run, args.sequences)

    width, height = (int(part) for part in args.lr_size.lower().split("x", 1))
    encoded = bandwidth(dataset_root=args.dataset_root.resolve(), sequences=args.sequences, lr_size=(width, height), quality=args.jpeg_quality, fps=args.fps, runs=runs)
    expected = EXPECTED[args.dataset_key]
    g1_metric = metrics["g1_gating_on"]["aggregate"]
    reproduction = {
        "id_switches": int(g1_metric["id_switches"]) == expected["id_switches"],
        "idf1": abs(float(g1_metric["idf1"]) - expected["idf1"]) <= 5e-5,
        "hr_requests": int(encoded["g1_gating_on"]["hr_requests"]) == expected["hr_requests"],
    }
    safety_violations = {method: unsafe_hr_violations(rows) for method, rows in events.items()}
    safety_passed = all(not rows for rows in safety_violations.values())
    required_checks_passed = protocol_paired and candidate_pools_identical and raw_equivalent and all(reproduction.values()) and safety_passed

    g0_bw, g1_bw = encoded["g0_gating_off"], encoded["g1_gating_on"]
    savings = {
        "hr_requests": g0_bw["hr_requests"] - g1_bw["hr_requests"],
        "hr_request_fraction": (g0_bw["hr_requests"] - g1_bw["hr_requests"]) / g0_bw["hr_requests"] if g0_bw["hr_requests"] else 0.0,
        "hr_bytes": g0_bw["hr_bytes"] - g1_bw["hr_bytes"],
        "hr_byte_fraction": (g0_bw["hr_bytes"] - g1_bw["hr_bytes"]) / g0_bw["hr_bytes"] if g0_bw["hr_bytes"] else 0.0,
        "total_bytes": g0_bw["total_bytes"] - g1_bw["total_bytes"],
        "total_byte_fraction": (g0_bw["total_bytes"] - g1_bw["total_bytes"]) / g0_bw["total_bytes"] if g0_bw["total_bytes"] else 0.0,
    }
    payload = {
        "dataset": args.dataset_key, "protocol": protocols["g1_gating_on"],
        "verification": {
            "protocol_paired": protocol_paired,
            "candidate_pools_identical": candidate_pools_identical,
            "raw_botsort_equivalent": raw_equivalent,
            "raw_checks": raw_checks,
            "g1_reference_reproduction": reproduction,
            "unsafe_hr_request_violations": safety_violations,
            "prequery_safety_passed": safety_passed,
            "final_mot_identical_by_sequence": final_identical,
            "required_checks_passed": required_checks_passed,
        },
        "candidate_pool": {"counts": {method: len(rows) for method, rows in events.items()}, "difference": candidate_pool_difference},
        "methods": metrics, "bandwidth": encoded, "savings": savings,
        "final_prediction_sha256": final_hashes,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Base-Guided Gating Ablation — {args.dataset_key}", "",
        f"Verification: **{'PASS' if required_checks_passed else 'FAIL'}**", "",
        "| Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | HR requests ↓ | Encoded HR MB ↓ | Total MB ↓ | FPS ↑ | Latency ms ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, label in (("g0_gating_off", "G0 — Gating OFF"), ("g1_gating_on", "G1 — Gating ON (Ours)")):
        row = metrics[method]["aggregate"]
        bw = encoded[method]
        timing = metrics[method]["timing"]
        lines.append(f"| {label} | {row['id_switches']} | {row['idf1']:.4f} | {row['hota']:.4f} | {row['assa']:.4f} | {bw['hr_requests']} | {bw['hr_bytes']/1e6:.3f} | {bw['total_mb']:.3f} | {timing['fps']:.2f} | {timing['latency_ms']:.2f} |")
    lines += [
        "", "## Paired verification", "",
        f"- Candidate pools identical: `{candidate_pools_identical}` ({len(events['g0_gating_off'])} events each expected).",
        f"- Raw BoT-SORT equivalent: `{raw_equivalent}`.",
        f"- G1 frozen reference reproduced: `{reproduction}`.",
        f"- Safety-infeasible candidates requested HR: `{sum(len(rows) for rows in safety_violations.values())}`.",
        f"- Final MOT identical per sequence: `{final_identical}`.",
        "", "## Savings: acquisition versus total transmission", "",
        f"- Physical HR requests: {g0_bw['hr_requests']} → {g1_bw['hr_requests']} (**−{savings['hr_request_fraction']:.2%}**).",
        f"- Encoded HR payload: {g0_bw['hr_bytes']/1e6:.3f} → {g1_bw['hr_bytes']/1e6:.3f} MB (**−{savings['hr_byte_fraction']:.2%}**).",
        f"- Total transmitted payload including continuous LR: {g0_bw['total_mb']:.3f} → {g1_bw['total_mb']:.3f} MB (**−{savings['total_byte_fraction']:.4%}**).",
        "", "HR-acquisition savings can be large while total-bandwidth savings remain small because every method continuously transmits LR frames.",
        "", "## Per-sequence metrics", "",
    ]
    for sequence in args.sequences:
        lines += [f"### {sequence}", "", "| Method | IDSW | IDF1 | HOTA | AssA |", "|---|---:|---:|---:|---:|"]
        for method, label in (("g0_gating_off", "G0"), ("g1_gating_on", "G1")):
            row = metrics[method]["sequences"][sequence]
            lines.append(f"| {label} | {row['id_switches']} | {row['idf1']:.4f} | {row['hota']:.4f} | {row['assa']:.4f} |")
        lines.append("")
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Gating ablation report: {(args.output_dir / 'REPORT.md').resolve()}")
    return 0 if required_checks_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

