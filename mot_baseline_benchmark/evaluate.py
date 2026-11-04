"""Evaluate five MOT methods and enforce all fairness/reference guards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_mot import load_mot
from hive_hr_reid.run_experiment import basic_metrics

from .protocol import DATASETS, resolve_sequence_dir, sha256, write_json
from .run_trackers import compare_mot_replay


METHOD_LABELS = {
    "sort": "SORT",
    "bytetrack": "YOLO11l + ByteTrack",
    "botsort": "YOLO11l + BoT-SORT",
    "bytetrack_hive": "YOLO11l + ByteTrack + HIVE–ECIA",
    "hive_ecia": "YOLO11l + BoT-SORT + HIVE–ECIA",
}


def best_value(methods: dict, metric: str, *, lower: bool = False) -> float:
    values = [float(methods[name]["aggregate"][metric]) for name in methods]
    return min(values) if lower else max(values)


def bold_if_best(value: float | int, best: float, formatted: str) -> str:
    return f"**{formatted}**" if abs(float(value) - float(best)) <= 1e-12 else formatted


def format_best(value: float | int, best: float, spec: str = "") -> str:
    return bold_if_best(value, best, format(value, spec))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--standard-run", type=Path, required=True)
    parser.add_argument("--hive-run", type=Path, default=None)
    parser.add_argument("--bytetrack-hive-run", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--reference-only", action="store_true")
    return parser.parse_args()


def evaluate_method(
    *, method: str, mot_root: Path, sequences: tuple[str, ...], gt: dict, iou: float
) -> dict:
    predictions = {
        sequence: load_mot(mot_root / f"{sequence}.txt", ground_truth=False)
        for sequence in sequences
    }
    return basic_metrics(
        argparse.Namespace(iou=iou), list(sequences), predictions, gt, include_hota=True
    )


def expected_differences(actual: dict, expected: dict) -> dict:
    differences = {}
    for key, value in expected.items():
        observed = actual[key]
        if isinstance(value, int):
            if int(observed) != value:
                differences[key] = {"expected": value, "actual": observed}
        elif round(float(observed), 4) != round(float(value), 4):
            differences[key] = {"expected": value, "actual": observed}
    return differences


def aggregate_timing(standard_run: Path, method: str, sequences: tuple[str, ...]) -> dict:
    rows = [
        json.loads((standard_run / "timing" / method / f"{sequence}.json").read_text(encoding="utf-8"))
        for sequence in sequences
    ]
    frames = sum(int(row["frames"]) for row in rows)
    tracker_ms = sum(float(row["tracker_only_ms_total"]) for row in rows)
    end_to_end_ms = sum(float(row["end_to_end_ms_total"]) for row in rows)
    return {
        "frames": frames,
        "tracker_only_fps": frames / (tracker_ms / 1000.0) if tracker_ms else 0.0,
        "tracker_only_latency_ms": tracker_ms / frames if frames else 0.0,
        "end_to_end_fps": frames / (end_to_end_ms / 1000.0) if end_to_end_ms else 0.0,
        "end_to_end_latency_ms": end_to_end_ms / frames if frames else 0.0,
        "scope": (
            "end_to_end is shared cached YOLO detector latency plus tracker-only replay latency; "
            "decode, physical resize, cache serialization and MOT writes are excluded"
        ),
    }


def hive_timing(hive_run: Path, sequences: tuple[str, ...]) -> dict:
    manifest = json.loads((hive_run / "run_manifest.json").read_text(encoding="utf-8"))
    tracker_type = str(manifest.get("tracker_type", "tracker"))
    rows = [
        json.loads((hive_run / "timing" / f"{sequence}.json").read_text(encoding="utf-8"))
        for sequence in sequences
    ]
    frames = sum(int(row["frames"]) for row in rows)
    wall = sum(float(row["wall_seconds"]) for row in rows)
    requests = sum(int(row["engine"]["hr_cost"].get("requests", 0)) for row in rows)
    bytes_read = sum(int(row["engine"]["hr_cost"].get("bytes_read", 0)) for row in rows)
    return {
        "frames": frames,
        "tracker_only_fps": None,
        "tracker_only_latency_ms": None,
        "end_to_end_fps": frames / wall if wall else 0.0,
        "end_to_end_latency_ms": 1000.0 * wall / frames if frames else 0.0,
        "hr_requests": requests,
        "hr_bytes_read": bytes_read,
        "scope": (
            f"measured detector + {tracker_type} + online HIVE–ECIA controller; "
            "decode, physical resize and file writes are excluded"
        ),
    }


def relative_change(full: float, baseline: float, higher_is_better: bool) -> float:
    if baseline == 0:
        return 0.0
    return (
        (full - baseline) / abs(baseline)
        if higher_is_better else (baseline - full) / abs(baseline)
    )


def improvement_rows(methods: dict) -> dict:
    full = methods["hive_ecia"]["aggregate"]
    rows = {}
    for baseline_name in ("sort", "bytetrack", "botsort"):
        baseline = methods[baseline_name]["aggregate"]
        rows[baseline_name] = {
            "comparison_type": "direct" if baseline_name == "botsort" else "indirect",
            "idsw_reduction": baseline["id_switches"] - full["id_switches"],
            "idsw_relative_reduction": relative_change(
                full["id_switches"], baseline["id_switches"], False
            ),
            **{
                f"{metric}_absolute": full[metric] - baseline[metric]
                for metric in ("idf1", "hota", "assa", "mota")
            },
            **{
                f"{metric}_relative": relative_change(full[metric], baseline[metric], True)
                for metric in ("idf1", "hota", "assa", "mota")
            },
        }
    return rows


def paired_improvement(methods: dict, baseline_name: str, full_name: str) -> dict:
    baseline = methods[baseline_name]["aggregate"]
    full = methods[full_name]["aggregate"]
    return {
        "baseline": baseline_name,
        "full": full_name,
        "idsw_reduction": baseline["id_switches"] - full["id_switches"],
        "idsw_relative_reduction": relative_change(
            full["id_switches"], baseline["id_switches"], False
        ),
        **{
            f"{metric}_absolute": full[metric] - baseline[metric]
            for metric in ("idf1", "hota", "assa", "mota")
        },
        **{
            f"{metric}_relative": relative_change(full[metric], baseline[metric], True)
            for metric in ("idf1", "hota", "assa", "mota")
        },
    }


def normalize_recovery_geometry(
    *,
    baseline_path: Path,
    recovery_raw_path: Path,
    recovery_canonical_path: Path,
    output_path: Path,
) -> dict:
    """Combine baseline geometry with HIVE--ECIA canonical identities."""

    baseline = baseline_path.read_text(encoding="utf-8").splitlines()
    raw = recovery_raw_path.read_text(encoding="utf-8").splitlines()
    canonical = recovery_canonical_path.read_text(encoding="utf-8").splitlines()
    if not (len(baseline) == len(raw) == len(canonical)):
        raise RuntimeError("Cannot normalize ByteTrack recovery: MOT row counts differ")
    output = []
    changed_ids = 0
    for line_number, (baseline_line, raw_line, canonical_line) in enumerate(
        zip(baseline, raw, canonical), start=1
    ):
        base_parts = baseline_line.split(",")
        raw_parts = raw_line.split(",")
        canonical_parts = canonical_line.split(",")
        if (
            len(base_parts) < 9
            or len(raw_parts) != len(base_parts)
            or len(canonical_parts) != len(base_parts)
        ):
            raise RuntimeError(f"Malformed paired MOT row {line_number}")
        # The recovery controller may alter identity only. Frame, confidence
        # and MOT metadata stay exact; coordinate scaling may differ by one
        # serialized rounding unit on otherwise identical float boxes.
        geometry_delta = max(
            abs(float(raw_parts[index]) - float(canonical_parts[index]))
            for index in (2, 3, 4, 5)
        )
        if (
            raw_parts[0] != canonical_parts[0]
            or raw_parts[6:] != canonical_parts[6:]
            or geometry_delta > 0.011
        ):
            raise RuntimeError(
                f"Recovery controller changed non-identity data at MOT row {line_number}"
            )
        if base_parts[0] != raw_parts[0] or base_parts[6:] != raw_parts[6:]:
            raise RuntimeError(
                f"Baseline/recovery frame, score or metadata differs at MOT row {line_number}"
            )
        normalized = list(base_parts)
        normalized[1] = canonical_parts[1]
        changed_ids += int(normalized[1] != base_parts[1])
        output.append(",".join(normalized))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(output) + ("\n" if output else ""), encoding="utf-8"
    )
    return {
        "rows": len(output),
        "rows_with_recovered_identity": changed_ids,
        "output": str(output_path.resolve()),
        "definition": (
            "baseline ByteTrack geometry/confidence/metadata + HIVE-ECIA canonical ID"
        ),
    }


def report_lines(payload: dict) -> list[str]:
    methods = payload["methods"]
    protocol = payload["protocol"]
    lines = [
        f"# Fair MOT benchmark — {protocol['dataset_label']}",
        "",
        f"Physical LR `{protocol['physical_lr_size'][0]}×{protocol['physical_lr_size'][1]}`, "
        f"YOLO `imgsz={protocol['imgsz']}`, confidence `{protocol['conf']}`, "
        f"NMS IoU `{protocol['nms_iou']}`, evaluation IoU `{protocol['evaluation_iou']}`.",
        "",
        "## Aggregate comparison",
        "",
        "| Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | MOTA ↑ | IDTP | IDFP | IDFN | FPS ↑ | Latency ms ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    best = {
        "id_switches": best_value(methods, "id_switches", lower=True),
        **{
            metric: best_value(methods, metric)
            for metric in ("idf1", "hota", "assa", "mota")
        },
        "idtp": best_value(methods, "idtp"),
        "idfp": best_value(methods, "idfp", lower=True),
        "idfn": best_value(methods, "idfn", lower=True),
    }
    best_fps = max(float(row["timing"]["end_to_end_fps"]) for row in methods.values())
    best_latency = min(
        float(row["timing"]["end_to_end_latency_ms"]) for row in methods.values()
    )
    for method in ("sort", "bytetrack", "bytetrack_hive", "botsort", "hive_ecia"):
        row = methods[method]["aggregate"]
        timing = methods[method]["timing"]
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{format_best(row['id_switches'], best['id_switches'])} | "
            f"{format_best(row['idf1'], best['idf1'], '.4f')} | "
            f"{format_best(row['hota'], best['hota'], '.4f')} | "
            f"{format_best(row['assa'], best['assa'], '.4f')} | "
            f"{format_best(row['mota'], best['mota'], '.4f')} | "
            f"{format_best(row['idtp'], best['idtp'])} | "
            f"{format_best(row['idfp'], best['idfp'])} | "
            f"{format_best(row['idfn'], best['idfn'])} | "
            f"{format_best(timing['end_to_end_fps'], best_fps, '.2f')} | "
            f"{format_best(timing['end_to_end_latency_ms'], best_latency, '.2f')} |"
        )
    lines.extend([
        "",
        "Direct paired comparisons are ByteTrack vs ByteTrack + HIVE–ECIA and "
        "BoT-SORT vs BoT-SORT + HIVE–ECIA. Comparisons with SORT or across "
        "tracker families are indirect.",
        "",
        "## Direct paired improvements",
        "",
        "| Tracker family | Baseline IDSW | + HIVE–ECIA IDSW | Reduction | Relative | ΔIDF1 | ΔHOTA | ΔAssA | ΔMOTA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for family, baseline_name, full_name in (
        ("ByteTrack", "bytetrack", "bytetrack_hive"),
        ("BoT-SORT", "botsort", "hive_ecia"),
    ):
        baseline = methods[baseline_name]["aggregate"]
        full = methods[full_name]["aggregate"]
        row = payload["direct_improvements"][baseline_name]
        lines.append(
            f"| {family} | {baseline['id_switches']} | {full['id_switches']} | "
            f"{row['idsw_reduction']:+d} | {row['idsw_relative_reduction']:+.2%} | "
            f"{row['idf1_absolute']:+.4f} | {row['hota_absolute']:+.4f} | "
            f"{row['assa_absolute']:+.4f} | {row['mota_absolute']:+.4f} |"
        )
    lines.extend([
        "",
        "ByteTrack + HIVE–ECIA is a zero-shot tracker-transfer integration: the frozen ECIA/HIVE checkpoints and thresholds remain those selected with the BoT-SORT pipeline; no ByteTrack/test/DTB tuning is performed.",
        "",
        "## BoT-SORT + HIVE–ECIA against all standard baselines",
        "",
        "| Baseline | Type | ΔIDSW ↓ | Relative IDSW ↓ | ΔIDF1 | Relative IDF1 | ΔHOTA | Relative HOTA | ΔAssA | Relative AssA | ΔMOTA | Relative MOTA |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method in ("sort", "bytetrack", "botsort"):
        row = payload["improvements"][method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['comparison_type']} | "
            f"{row['idsw_reduction']:+d} | {row['idsw_relative_reduction']:+.2%} | "
            f"{row['idf1_absolute']:+.4f} | {row['idf1_relative']:+.2%} | "
            f"{row['hota_absolute']:+.4f} | {row['hota_relative']:+.2%} | "
            f"{row['assa_absolute']:+.4f} | {row['assa_relative']:+.2%} | "
            f"{row['mota_absolute']:+.4f} | {row['mota_relative']:+.2%} |"
        )
    lines.extend(["", "## Per-sequence metrics", ""])
    for sequence in protocol["sequences"]:
        lines.extend([
            f"### {sequence}",
            "",
            "| Method | IDSW | IDF1 | HOTA | AssA | MOTA | IDTP | IDFP | IDFN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        sequence_best = {
            "id_switches": min(
                methods[name]["sequences"][sequence]["id_switches"] for name in methods
            ),
            **{
                metric: max(
                    methods[name]["sequences"][sequence][metric] for name in methods
                )
                for metric in ("idf1", "hota", "assa", "mota", "idtp")
            },
            **{
                metric: min(
                    methods[name]["sequences"][sequence][metric] for name in methods
                )
                for metric in ("idfp", "idfn")
            },
        }
        for method in ("sort", "bytetrack", "bytetrack_hive", "botsort", "hive_ecia"):
            row = methods[method]["sequences"][sequence]
            lines.append(
                f"| {METHOD_LABELS[method]} | "
                f"{format_best(row['id_switches'], sequence_best['id_switches'])} | "
                f"{format_best(row['idf1'], sequence_best['idf1'], '.4f')} | "
                f"{format_best(row['hota'], sequence_best['hota'], '.4f')} | "
                f"{format_best(row['assa'], sequence_best['assa'], '.4f')} | "
                f"{format_best(row['mota'], sequence_best['mota'], '.4f')} | "
                f"{format_best(row['idtp'], sequence_best['idtp'])} | "
                f"{format_best(row['idfp'], sequence_best['idfp'])} | "
                f"{format_best(row['idfn'], sequence_best['idfn'])} |"
            )
        lines.append("")
    checks = payload["fairness_checks"]
    lines.extend([
        "## Fairness verification",
        "",
        f"- BoT-SORT reference metric gate: **{'PASS' if checks['botsort_reference_passed'] else 'FAIL'}**.",
        f"- Cached BoT-SORT replay is association-equivalent to original `model.track`: **{'PASS' if checks['cached_botsort_replay_equivalent'] else 'FAIL'}**.",
        f"- BoT-SORT raw predictions equal the HIVE–ECIA input: **{'PASS' if checks['hive_raw_equals_botsort'] else 'FAIL'}**.",
        f"- ByteTrack raw predictions are association-equivalent to the ByteTrack + HIVE–ECIA input: **{'PASS' if checks['bytetrack_hive_raw_association_equivalent'] else 'FAIL'}**.",
        "- ByteTrack + HIVE–ECIA metrics use baseline ByteTrack geometry with recovered canonical IDs, isolating identity recovery from ≤0.01 px float rounding.",
        "- SORT, ByteTrack and BoT-SORT consume the same immutable per-sequence YOLO cache. GT is loaded only by this evaluator.",
        "- Every tracker instance is recreated at each sequence boundary.",
        "- ByteTrack/BoT-SORT retain `track_low_thresh=0.10`, but the shared established detector confidence is 0.20; therefore no tracker sees detections in `[0.10,0.20)`. This is explicit and identical across methods.",
        "",
        "## Timing scope",
        "",
        "For the three standard trackers, end-to-end latency is shared cached YOLO detector latency plus tracker-only replay latency. Each HIVE–ECIA row uses measured detector + underlying tracker + recovery-controller latency. Decode, physical resize and output writes are excluded from both scopes.",
    ])
    return lines


def main() -> int:
    args = parse_args()
    spec = DATASETS[args.dataset_key]
    standard_run = args.standard_run.resolve()
    manifest = json.loads((standard_run / "run_manifest.json").read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    if tuple(protocol["sequences"]) != spec.sequences:
        raise RuntimeError("Benchmark sequence list differs from frozen protocol")
    if tuple(protocol["physical_lr_size"]) != spec.physical_lr_size or int(protocol["imgsz"]) != spec.imgsz:
        raise RuntimeError("Benchmark LR resolution differs from frozen protocol")
    gt = {
        sequence: load_mot(
            resolve_sequence_dir(args.dataset_root.resolve(), sequence) / "gt" / "gt.txt",
            ground_truth=True,
        )
        for sequence in spec.sequences
    }
    botsort = evaluate_method(
        method="botsort",
        mot_root=standard_run / "mot" / "botsort",
        sequences=spec.sequences,
        gt=gt,
        iou=args.iou,
    )
    differences = expected_differences(botsort["aggregate"], spec.expected_botsort)
    reference_payload = {
        "dataset": spec.name,
        "expected": spec.expected_botsort,
        "actual": botsort["aggregate"],
        "passed": not differences,
        "differences": differences,
    }
    write_json(args.output_dir / "botsort_reference.json", reference_payload)
    if differences:
        raise RuntimeError(
            "Frozen BoT-SORT reference metrics differ; HIVE–ECIA/full comparison is blocked: "
            f"{differences}"
        )
    if args.reference_only:
        print(f"BoT-SORT reference gate passed: {args.output_dir / 'botsort_reference.json'}")
        return 0
    if args.hive_run is None or args.bytetrack_hive_run is None:
        raise ValueError(
            "--hive-run and --bytetrack-hive-run are required unless --reference-only is used"
        )
    hive_run = args.hive_run.resolve()
    bytetrack_hive_run = args.bytetrack_hive_run.resolve()
    hive_manifest = json.loads((hive_run / "run_manifest.json").read_text(encoding="utf-8"))
    bytetrack_hive_manifest = json.loads(
        (bytetrack_hive_run / "run_manifest.json").read_text(encoding="utf-8")
    )
    if hive_manifest.get("mode") != "base_guided_safety_online":
        raise RuntimeError("HIVE–ECIA run is not Existing Base-Guided Safety mode")
    if bytetrack_hive_manifest.get("mode") != "base_guided_safety_online":
        raise RuntimeError("ByteTrack HIVE–ECIA run is not Base-Guided Safety mode")
    if hive_manifest.get("tracker_type") not in (None, "botsort"):
        raise RuntimeError(f"Expected BoT-SORT HIVE tracker, got {hive_manifest.get('tracker_type')}")
    if bytetrack_hive_manifest.get("tracker_type") != "bytetrack":
        raise RuntimeError(
            f"Expected ByteTrack HIVE tracker, got {bytetrack_hive_manifest.get('tracker_type')}"
        )
    paired_fields = {
        "dataset_root": str(args.dataset_root.resolve()),
        "sequences": list(spec.sequences),
        "physical_lr_size": list(spec.physical_lr_size),
        "imgsz": spec.imgsz,
        "conf": float(protocol["conf"]),
        "iou": float(protocol["nms_iou"]),
    }
    mismatched = {
        key: {"standard": value, "hive": hive_manifest.get(key)}
        for key, value in paired_fields.items()
        if hive_manifest.get(key) != value
    }
    if mismatched:
        raise RuntimeError(f"Standard/HIVE protocols differ: {mismatched}")
    bytetrack_mismatched = {
        key: {"standard": value, "bytetrack_hive": bytetrack_hive_manifest.get(key)}
        for key, value in paired_fields.items()
        if bytetrack_hive_manifest.get(key) != value
    }
    if bytetrack_mismatched:
        raise RuntimeError(
            f"Standard/ByteTrack-HIVE protocols differ: {bytetrack_mismatched}"
        )
    raw_hashes = {
        sequence: {
            "botsort": sha256(standard_run / "mot" / "botsort" / f"{sequence}.txt"),
            "hive_raw": sha256(hive_run / "raw_mot" / f"{sequence}.txt"),
        }
        for sequence in spec.sequences
    }
    hive_raw_equals_botsort = all(
        values["botsort"] == values["hive_raw"] for values in raw_hashes.values()
    )
    if not hive_raw_equals_botsort:
        raise RuntimeError(
            "Existing HIVE–ECIA did not receive byte-identical BoT-SORT predictions; "
            f"full comparison is blocked: {raw_hashes}"
        )
    bytetrack_raw_hashes = {
        sequence: {
            "bytetrack": sha256(
                standard_run / "mot" / "bytetrack" / f"{sequence}.txt"
            ),
            "bytetrack_hive_raw": sha256(
                bytetrack_hive_run / "raw_mot" / f"{sequence}.txt"
            ),
        }
        for sequence in spec.sequences
    }
    bytetrack_hive_raw_equals_bytetrack = all(
        values["bytetrack"] == values["bytetrack_hive_raw"]
        for values in bytetrack_raw_hashes.values()
    )
    bytetrack_association_audit = {
        sequence: compare_mot_replay(
            standard_run / "mot" / "bytetrack" / f"{sequence}.txt",
            bytetrack_hive_run / "raw_mot" / f"{sequence}.txt",
        )
        for sequence in spec.sequences
    }
    bytetrack_hive_raw_association_equivalent = all(
        bool(row["equivalent"]) for row in bytetrack_association_audit.values()
    )
    if not bytetrack_hive_raw_association_equivalent:
        raise RuntimeError(
            "ByteTrack + HIVE–ECIA did not receive association-equivalent ByteTrack "
            f"predictions; comparison is blocked: {bytetrack_association_audit}"
        )
    normalized_bytetrack_root = args.output_dir / "normalized_bytetrack_hive_mot"
    bytetrack_geometry_normalization = {
        sequence: normalize_recovery_geometry(
            baseline_path=standard_run / "mot" / "bytetrack" / f"{sequence}.txt",
            recovery_raw_path=bytetrack_hive_run / "raw_mot" / f"{sequence}.txt",
            recovery_canonical_path=bytetrack_hive_run / "mot" / f"{sequence}.txt",
            output_path=normalized_bytetrack_root / f"{sequence}.txt",
        )
        for sequence in spec.sequences
    }
    methods = {
        method: evaluate_method(
            method=method,
            mot_root=standard_run / "mot" / method,
            sequences=spec.sequences,
            gt=gt,
            iou=args.iou,
        )
        for method in ("sort", "bytetrack", "botsort")
    }
    methods["hive_ecia"] = evaluate_method(
        method="hive_ecia",
        mot_root=hive_run / "mot",
        sequences=spec.sequences,
        gt=gt,
        iou=args.iou,
    )
    methods["bytetrack_hive"] = evaluate_method(
        method="bytetrack_hive",
        mot_root=normalized_bytetrack_root,
        sequences=spec.sequences,
        gt=gt,
        iou=args.iou,
    )
    for method in ("sort", "bytetrack", "botsort"):
        methods[method]["timing"] = aggregate_timing(standard_run, method, spec.sequences)
    methods["hive_ecia"]["timing"] = hive_timing(hive_run, spec.sequences)
    methods["bytetrack_hive"]["timing"] = hive_timing(
        bytetrack_hive_run, spec.sequences
    )
    cached_replay_equivalent = all(
        bool(manifest["botsort_replay_audit"][sequence]["equivalent"])
        for sequence in spec.sequences
    )
    official_equals_live = all(
        manifest["live_botsort_sha256"][sequence]
        == manifest["official_botsort_sha256"][sequence]
        for sequence in spec.sequences
    )
    payload = {
        "protocol": {
            "dataset_key": args.dataset_key,
            "dataset_label": spec.name,
            "dataset_root": str(args.dataset_root.resolve()),
            "sequences": list(spec.sequences),
            "physical_lr_size": list(spec.physical_lr_size),
            "imgsz": spec.imgsz,
            "conf": protocol["conf"],
            "nms_iou": protocol["nms_iou"],
            "evaluation_iou": args.iou,
            "gt_used_only_for_offline_evaluation": True,
            "hive_mode": hive_manifest["mode"],
            "bytetrack_hive_mode": bytetrack_hive_manifest["mode"],
            "bytetrack_transfer": (
                "zero-shot tracker transfer; frozen BoT-SORT-trained ECIA/HIVE/thresholds"
            ),
            "frozen_models": hive_manifest.get("frozen_models"),
        },
        "fairness_checks": {
            "botsort_reference_passed": True,
            "botsort_reference": reference_payload,
            "cached_botsort_replay_equivalent": cached_replay_equivalent,
            "cached_botsort_replay_audit": manifest["botsort_replay_audit"],
            "official_botsort_equals_live": official_equals_live,
            "hive_raw_equals_botsort": hive_raw_equals_botsort,
            "bytetrack_hive_raw_equals_bytetrack": bytetrack_hive_raw_equals_bytetrack,
            "bytetrack_hive_raw_association_equivalent": (
                bytetrack_hive_raw_association_equivalent
            ),
            "bytetrack_association_audit": bytetrack_association_audit,
            "bytetrack_geometry_normalization": bytetrack_geometry_normalization,
            "raw_prediction_sha256": raw_hashes,
            "bytetrack_raw_prediction_sha256": bytetrack_raw_hashes,
            "same_detection_cache_for_standard_trackers": True,
            "sequence_reset_verified_by_fresh_tracker_instances": True,
            "no_gt_during_tracking": True,
        },
        "methods": methods,
    }
    payload["improvements"] = improvement_rows(methods)
    payload["direct_improvements"] = {
        "botsort": paired_improvement(methods, "botsort", "hive_ecia"),
        "bytetrack": paired_improvement(
            methods, "bytetrack", "bytetrack_hive"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", payload)
    (args.output_dir / "REPORT.md").write_text(
        "\n".join(report_lines(payload)) + "\n", encoding="utf-8"
    )
    print(f"Dataset report: {(args.output_dir / 'REPORT.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

