"""Evaluate and compare full-video online HIVE--ECIA runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    CODE_ROOT.parent
    / "cvat_packages_trim300_last100"
    / "object_tracking_prepared_manual_dataset"
    / "mot_challenge"
)


def resolve_sequence_dir(root: Path, sequence: str) -> Path:
    candidates = [root / sequence, *(root / split / sequence for split in ("train", "val", "test"))]
    matches = [path for path in candidates if (path / "img1").is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one directory for {sequence!r}; found {matches}"
        )
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--online-run", type=Path, required=True)
    parser.add_argument("--hr-disabled-run", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.50)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def timing_summary(run: Path, sequences: list[str]) -> dict:
    rows = [
        json.loads((run / "timing" / f"{sequence}.json").read_text(encoding="utf-8"))
        for sequence in sequences
    ]
    frames = sum(int(row["frames"]) for row in rows)
    wall = sum(float(row["wall_seconds"]) for row in rows)
    trigger_count = sum(int(row["engine"]["counters"].get("triggers", 0)) for row in rows)
    hr = {
        key: sum(int(row["engine"]["hr_cost"].get(key, 0)) for row in rows)
        for key in ("requests", "crops", "bytes_read", "crop_pixels")
    }
    return {
        "frames": frames,
        "wall_seconds": wall,
        "fps": frames / wall if wall else 0.0,
        "mean_latency_ms": 1000.0 * wall / frames if frames else 0.0,
        "triggers": trigger_count,
        "hr_cost": hr,
    }


def trigger_pool(run: Path, sequences: list[str]) -> list[tuple[str, int, str, str]]:
    events = []
    for sequence in sequences:
        path = run / "diagnostics" / f"{sequence}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("type") == "recovery_event":
                events.append(
                    (
                        sequence,
                        int(row["frame"]),
                        str(row["old_token"]),
                        str(row["new_token"]),
                    )
                )
    return events


def trigger_schedule(
    events: list[tuple[str, int, str, str]],
) -> list[tuple[str, int, str]]:
    """Ambiguity moments independent of the chosen old-ID hypothesis."""

    return [(sequence, frame, new_token) for sequence, frame, _, new_token in events]


def _raw_id(token: str) -> int:
    return int(token.split(":", 1)[0])


def trigger_accuracy(
    *,
    run: Path,
    sequences: list[str],
    gt_cache: dict,
    raw_predictions: dict,
    iou: float,
) -> dict:
    """Score the frozen online trigger after inference; never feed GT back."""

    from rank_highres_retrieval_frames import build_match_index, find_switch_events

    logged_rows = trigger_pool(run, sequences)
    logged_by_sequence = {sequence: set() for sequence in sequences}
    for sequence, frame, old_token, new_token in logged_rows:
        logged_by_sequence[sequence].add(
            (int(frame), _raw_id(old_token), _raw_id(new_token))
        )

    per_sequence = {}
    totals = {"triggers": 0, "oracle_switches": 0, "tp": 0, "fp": 0, "fn": 0}
    for sequence in sequences:
        matches, _ = build_match_index(gt_cache[sequence], raw_predictions[sequence], iou)
        oracle = {
            (
                int(event["target_frame"]),
                int(event["lr_old_pred_id"]),
                int(event["lr_new_pred_id"]),
            )
            for event in find_switch_events(matches)
        }
        triggered = logged_by_sequence[sequence]
        tp = len(triggered & oracle)
        fp = len(triggered - oracle)
        fn = len(oracle - triggered)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        row = {
            "triggers": len(triggered),
            "oracle_switches": len(oracle),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        per_sequence[sequence] = row
        totals["triggers"] += len(triggered)
        totals["oracle_switches"] += len(oracle)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    return {
        "definition": "exact (sequence, target_frame, old_raw_id, new_raw_id) match",
        "gt_usage": "posthoc evaluation only; unavailable to trigger/HIVE/ECIA",
        "aggregate": {
            **totals,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        },
        "sequences": per_sequence,
        "duplicate_logged_events": len(logged_rows)
        - sum(len(rows) for rows in logged_by_sequence.values()),
    }


def main() -> int:
    args = parse_args()
    # Heavy evaluator dependencies are loaded only after argparse, keeping
    # ``--help`` usable in a minimal host shell.
    from evaluate_mot import load_mot
    from hive_hr_reid.run_experiment import basic_metrics

    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    runs = {"baseline": args.baseline_run.resolve(), "online_hive_ecia": args.online_run.resolve()}
    if args.hr_disabled_run is not None:
        runs["hr_disabled"] = args.hr_disabled_run.resolve()

    manifests = {
        method: json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        for method, run in runs.items()
    }
    paired_fields = (
        "dataset_root",
        "dataset_label",
        "sequences",
        "detector",
        "tracker_config",
        "physical_lr_size",
        "imgsz",
        "conf",
        "iou",
    )
    protocol_values = {
        method: {field: manifest.get(field) for field in paired_fields}
        for method, manifest in manifests.items()
    }
    if len({json.dumps(value, sort_keys=True) for value in protocol_values.values()}) != 1:
        raise RuntimeError(f"Run protocols differ across methods: {protocol_values}")
    run_protocol = protocol_values["baseline"]

    gt_cache = {
        sequence: load_mot(
            resolve_sequence_dir(args.dataset_root.resolve(), sequence) / "gt" / "gt.txt",
            ground_truth=True,
        )
        for sequence in args.sequences
    }
    raw_hashes = {}
    for method, run in runs.items():
        raw_hashes[method] = {
            sequence: sha256(run / "raw_mot" / f"{sequence}.txt")
            for sequence in args.sequences
        }
    identical_raw = all(
        len({raw_hashes[method][sequence] for method in runs}) == 1
        for sequence in args.sequences
    )
    if not identical_raw:
        raise RuntimeError(
            "Raw BoT-SORT outputs differ across methods; detector/tracker comparison is not paired"
        )
    identical_trigger_schedule = True
    identical_trigger_hypotheses = True
    trigger_hypothesis_divergences = []
    if "hr_disabled" in runs:
        online_events = trigger_pool(runs["online_hive_ecia"], args.sequences)
        hr0_events = trigger_pool(runs["hr_disabled"], args.sequences)
        identical_trigger_schedule = trigger_schedule(online_events) == trigger_schedule(hr0_events)
        if not identical_trigger_schedule:
            raise RuntimeError(
                "Online and HR-disabled trigger schedules differ; HR ablation is not paired"
            )
        identical_trigger_hypotheses = online_events == hr0_events
        trigger_hypothesis_divergences = [
            {
                "sequence": online[0],
                "frame": online[1],
                "new_token": online[3],
                "online_old_token": online[2],
                "hr_disabled_old_token": hr0[2],
            }
            for online, hr0 in zip(online_events, hr0_events)
            if online[2] != hr0[2]
        ]

    raw_predictions = {
        sequence: load_mot(
            runs["baseline"] / "raw_mot" / f"{sequence}.txt",
            ground_truth=False,
        )
        for sequence in args.sequences
    }
    trigger_metrics = trigger_accuracy(
        run=runs["online_hive_ecia"],
        sequences=args.sequences,
        gt_cache=gt_cache,
        raw_predictions=raw_predictions,
        iou=args.iou,
    )

    metrics = {}
    for method, run in runs.items():
        predictions = {
            sequence: load_mot(run / "mot" / f"{sequence}.txt", ground_truth=False)
            for sequence in args.sequences
        }
        metrics[method] = basic_metrics(
            argparse.Namespace(iou=args.iou),
            args.sequences,
            predictions,
            gt_cache,
            include_hota=True,
        )
        metrics[method]["timing"] = timing_summary(run, args.sequences)

    args.output_dir.mkdir(parents=True)
    payload = {
        "protocol": {
            "iou": args.iou,
            "sequences": args.sequences,
            "identical_raw_botsort_predictions": identical_raw,
            "identical_online_trigger_schedule": identical_trigger_schedule,
            "identical_online_trigger_hypotheses": identical_trigger_hypotheses,
            "trigger_hypothesis_divergences": trigger_hypothesis_divergences,
            "raw_prediction_sha256": raw_hashes,
            "gt_used_only_for_posthoc_evaluation": True,
            "paired_run_settings": run_protocol,
        },
        "ambiguity_trigger": trigger_metrics,
        "methods": metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    columns = ["Method", "IDSW", "IDF1", "HOTA", "AssA", "MOTA", "FPS", "Latency ms", "HR crops"]
    lines = [
        f"# Online HIVE--ECIA full-video comparison — {run_protocol['dataset_label']}",
        "",
        f"Physical LR: `{run_protocol['physical_lr_size'][0]}x{run_protocol['physical_lr_size'][1]}`; detector `imgsz={run_protocol['imgsz']}`.",
        "",
        "Raw detector/BoT-SORT MOT files are byte-identical across methods. GT was loaded only after all online predictions were written.",
        "",
        (
            "Online and HR-disabled use the same trigger schedule. "
            f"Closed-loop corrections changed the selected old-ID hypothesis in "
            f"{len(trigger_hypothesis_divergences)} event(s); these are recorded in metrics.json."
        ),
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(columns) - 1)) + "|",
    ]
    for method, result in metrics.items():
        aggregate = result["aggregate"]
        timing = result["timing"]
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(aggregate["id_switches"]),
                    f"{aggregate['idf1']:.4f}",
                    f"{aggregate['hota']:.4f}",
                    f"{aggregate['assa']:.4f}",
                    f"{aggregate['mota']:.4f}",
                    f"{timing['fps']:.2f}",
                    f"{timing['mean_latency_ms']:.2f}",
                    str(timing["hr_cost"]["crops"]),
                ]
            )
            + " |"
        )
    trigger_aggregate = trigger_metrics["aggregate"]
    lines.extend(
        [
            "",
            "## Ambiguity-trigger accuracy",
            "",
            "GT is joined only here, after all MOT predictions and trigger logs are frozen. A true positive requires an exact `(sequence, target frame, old raw ID, new raw ID)` match to a BoT-SORT ID switch at the evaluation IoU.",
            "",
            "| Triggers | Oracle IDSW events | TP | FP | FN | Precision | Recall | F1 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {trigger_aggregate['triggers']} | {trigger_aggregate['oracle_switches']} | "
                f"{trigger_aggregate['tp']} | {trigger_aggregate['fp']} | {trigger_aggregate['fn']} | "
                f"{trigger_aggregate['precision']:.4f} | {trigger_aggregate['recall']:.4f} | "
                f"{trigger_aggregate['f1']:.4f} |"
            ),
            "",
            "### Trigger accuracy per sequence",
            "",
            "| Sequence | Triggers | Oracle events | TP | FP | FN | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sequence in args.sequences:
        row = trigger_metrics["sequences"][sequence]
        lines.append(
            f"| {sequence} | {row['triggers']} | {row['oracle_switches']} | "
            f"{row['true_positive']} | {row['false_positive']} | {row['false_negative']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )
    lines.extend(["", "## Per-sequence tracking metrics", ""])
    for sequence in args.sequences:
        lines.extend(
            [
                f"### {sequence}",
                "",
                "| Method | IDSW | IDF1 | HOTA | AssA | MOTA |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method, result in metrics.items():
            row = result["sequences"][sequence]
            lines.append(
                f"| {method} | {row['id_switches']} | {row['idf1']:.4f} | "
                f"{row['hota']:.4f} | {row['assa']:.4f} | {row['mota']:.4f} |"
            )
        lines.append("")
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Comparison: {(args.output_dir / 'REPORT.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

