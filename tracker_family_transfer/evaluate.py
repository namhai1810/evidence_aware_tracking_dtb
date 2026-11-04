"""Evaluate four tracker families before and after frozen HIVE–ECIA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_mot import load_mot
from hive_hr_reid.run_experiment import basic_metrics

from mot_baseline_benchmark.protocol import DATASETS, resolve_sequence_dir, sha256, write_json


PAIRS = (
    ("sort", "sort_hive", "SORT"),
    ("bytetrack", "bytetrack_hive", "ByteTrack"),
    ("ocsort", "ocsort_hive", "OC-SORT"),
    ("botsort", "hive_ecia", "BoT-SORT"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-metrics", type=Path, required=True)
    parser.add_argument("--reference-standard", type=Path, required=True)
    parser.add_argument("--sort-run", type=Path, required=True)
    parser.add_argument("--ocsort-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.50)
    return parser.parse_args()


def evaluate_root(root: Path, sequences: tuple[str, ...], gt: dict, iou: float) -> dict:
    predictions = {
        sequence: load_mot(root / f"{sequence}.txt", ground_truth=False)
        for sequence in sequences
    }
    return basic_metrics(argparse.Namespace(iou=iou), list(sequences), predictions, gt, include_hota=True)


def identity_only_audit(raw: Path, canonical: Path) -> dict:
    left = raw.read_text(encoding="utf-8").splitlines()
    right = canonical.read_text(encoding="utf-8").splitlines()
    result = {"rows": len(left), "row_count_equal": len(left) == len(right), "identity_only": True, "changed_ids": 0, "first_failure": None}
    if len(left) != len(right):
        result.update(identity_only=False, first_failure="row_count")
        return result
    for index, (raw_line, canonical_line) in enumerate(zip(left, right), start=1):
        raw_parts, canonical_parts = raw_line.split(","), canonical_line.split(",")
        if len(raw_parts) != len(canonical_parts) or raw_parts[0] != canonical_parts[0] or raw_parts[2:] != canonical_parts[2:]:
            result.update(identity_only=False, first_failure={"line": index, "raw": raw_line, "canonical": canonical_line})
            return result
        result["changed_ids"] += int(raw_parts[1] != canonical_parts[1])
    return result


def timing(run: Path, sequences: tuple[str, ...], recovery: bool) -> dict:
    rows = [json.loads((run / "timing" / f"{sequence}.json").read_text(encoding="utf-8")) for sequence in sequences]
    frames = sum(int(row["frames"]) for row in rows)
    detector_ms = sum(float(row["shared_detector_ms_total"]) for row in rows)
    processing_ms = sum(float(row["recovery_replay_ms_total"] if recovery else row["tracker_only_ms_total"]) for row in rows)
    total_ms = detector_ms + processing_ms
    output = {
        "frames": frames,
        "end_to_end_fps": frames / (total_ms / 1000.0) if total_ms else 0.0,
        "end_to_end_latency_ms": total_ms / frames if frames else 0.0,
        "scope": "shared cached YOLO detector timing + tracker/controller compute; decode, resize and writes excluded",
    }
    if recovery:
        output["hr_requests"] = sum(int(row["engine"]["hr_cost"]["requests"]) for row in rows)
        output["hr_bytes_read"] = sum(int(row["engine"]["hr_cost"]["bytes_read"]) for row in rows)
    else:
        output.update(hr_requests=0, hr_bytes_read=0)
    return output


def improvement(methods: dict, baseline: str, recovery: str) -> dict:
    left, right = methods[baseline]["aggregate"], methods[recovery]["aggregate"]
    return {
        "baseline": baseline,
        "recovery": recovery,
        "idsw_reduction": int(left["id_switches"] - right["id_switches"]),
        "idsw_relative_reduction": (left["id_switches"] - right["id_switches"]) / left["id_switches"] if left["id_switches"] else 0.0,
        **{f"delta_{key}": float(right[key] - left[key]) for key in ("idf1", "hota", "assa", "mota")},
    }


def report(payload: dict) -> str:
    methods = payload["methods"]
    lines = [
        f"# Tracker-family HIVE–ECIA transfer — {payload['protocol']['dataset_label']}", "",
        "All methods consume the same frozen YOLO11l detections. SORT/OC-SORT use the frozen BoT-SORT-trained HIVE–ECIA checkpoint and thresholds without tracker-specific or test/DTB tuning.", "",
        "| Tracker | Baseline IDSW | +HIVE IDSW | Reduction | Baseline IDF1 | +HIVE IDF1 | ΔHOTA | ΔAssA | HR requests |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for base, full, label in PAIRS:
        left, right = methods[base]["aggregate"], methods[full]["aggregate"]
        delta = payload["improvements"][base]
        lines.append(
            f"| {label} | {left['id_switches']} | {right['id_switches']} | {delta['idsw_reduction']:+d} ({delta['idsw_relative_reduction']:+.2%}) | "
            f"{left['idf1']:.4f} | {right['idf1']:.4f} | {delta['delta_hota']:+.4f} | {delta['delta_assa']:+.4f} | {methods[full]['timing'].get('hr_requests', 0)} |"
        )
    lines += ["", "## Fairness verification", ""]
    for key, value in payload["fairness"].items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", "## Per-sequence metrics", ""]
    for sequence in payload["protocol"]["sequences"]:
        lines += [f"### {sequence}", "", "| Method | IDSW | IDF1 | HOTA | AssA |", "|---|---:|---:|---:|---:|"]
        for base, full, label in PAIRS:
            for method, suffix in ((base, "baseline"), (full, "+ HIVE–ECIA")):
                row = methods[method]["sequences"][sequence]
                lines.append(f"| {label} {suffix} | {row['id_switches']} | {row['idf1']:.4f} | {row['hota']:.4f} | {row['assa']:.4f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    spec = DATASETS[args.dataset_key]
    reference = json.loads(args.reference_metrics.read_text(encoding="utf-8"))
    if tuple(reference["protocol"]["sequences"]) != spec.sequences:
        raise RuntimeError("Reference metric sequence set differs")
    gt = {
        sequence: load_mot(resolve_sequence_dir(args.dataset_root.resolve(), sequence) / "gt" / "gt.txt", ground_truth=True)
        for sequence in spec.sequences
    }
    sort_manifest = json.loads((args.sort_run / "run_manifest.json").read_text(encoding="utf-8"))
    oc_manifest = json.loads((args.ocsort_run / "run_manifest.json").read_text(encoding="utf-8"))
    for manifest, tracker in ((sort_manifest, "sort"), (oc_manifest, "ocsort")):
        if manifest["tracker"] != tracker or manifest["mode"] != "base_guided_safety_online":
            raise RuntimeError(f"Unexpected {tracker} transfer manifest")
        if tuple(manifest["sequences"]) != spec.sequences or tuple(manifest["physical_lr_size"]) != spec.physical_lr_size:
            raise RuntimeError(f"{tracker} transfer protocol differs")
    methods = {
        key: reference["methods"][key]
        for key in ("bytetrack", "bytetrack_hive", "botsort", "hive_ecia")
    }
    methods["sort"] = evaluate_root(args.sort_run / "raw_mot", spec.sequences, gt, args.iou)
    methods["sort_hive"] = evaluate_root(args.sort_run / "mot", spec.sequences, gt, args.iou)
    methods["ocsort"] = evaluate_root(args.ocsort_run / "raw_mot", spec.sequences, gt, args.iou)
    methods["ocsort_hive"] = evaluate_root(args.ocsort_run / "mot", spec.sequences, gt, args.iou)
    methods["sort"]["timing"] = timing(args.sort_run, spec.sequences, False)
    methods["sort_hive"]["timing"] = timing(args.sort_run, spec.sequences, True)
    methods["ocsort"]["timing"] = timing(args.ocsort_run, spec.sequences, False)
    methods["ocsort_hive"]["timing"] = timing(args.ocsort_run, spec.sequences, True)
    sort_raw_equal = all(
        sha256(args.sort_run / "raw_mot" / f"{sequence}.txt") == sha256(args.reference_standard / "mot" / "sort" / f"{sequence}.txt")
        for sequence in spec.sequences
    )
    audits = {
        tracker: {
            sequence: identity_only_audit(run / "raw_mot" / f"{sequence}.txt", run / "mot" / f"{sequence}.txt")
            for sequence in spec.sequences
        }
        for tracker, run in (("sort", args.sort_run), ("ocsort", args.ocsort_run))
    }
    identity_only = all(row["identity_only"] for values in audits.values() for row in values.values())
    if not sort_raw_equal or not identity_only:
        raise RuntimeError(f"Paired raw/identity-only verification failed: sort_equal={sort_raw_equal}, audits={audits}")
    payload = {
        "protocol": {
            "dataset_key": args.dataset_key,
            "dataset_label": spec.name,
            "dataset_root": str(args.dataset_root.resolve()),
            "sequences": list(spec.sequences),
            "physical_lr_size": list(spec.physical_lr_size),
            "imgsz": spec.imgsz,
            "conf": reference["protocol"]["conf"],
            "nms_iou": reference["protocol"]["nms_iou"],
            "evaluation_iou": args.iou,
            "zero_shot_tracker_transfer": ["sort", "ocsort"],
        },
        "fairness": {
            "same_yolo_detection_cache": True,
            "sort_raw_byte_identical_to_fair_baseline": sort_raw_equal,
            "recovery_changes_identity_only": identity_only,
            "fresh_tracker_per_sequence": True,
            "gt_used_only_after_predictions_frozen": True,
            "no_tracker_specific_threshold_tuning": True,
        },
        "identity_only_audits": audits,
        "methods": methods,
    }
    payload["improvements"] = {base: improvement(methods, base, full) for base, full, _ in PAIRS}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", payload)
    (args.output_dir / "REPORT.md").write_text(report(payload), encoding="utf-8")
    print(f"Tracker-family dataset report: {(args.output_dir / 'REPORT.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

