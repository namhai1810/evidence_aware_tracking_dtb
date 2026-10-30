"""Compare frozen ECIA Base-only and Full on paired unseen DTB sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("baseline", "YOLO12l + ByteTrack", "#64748b"),
    ("base_only", "ECIA Base-only", "#f59e0b"),
    ("full", "ECIA Full", "#10b981"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-only", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_metrics(path: Path, expected: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = payload["protocol"].get("ecia_ablation", "full")
    if actual != expected:
        raise ValueError(f"Expected {expected!r} metrics at {path}, found {actual!r}")
    if not payload["protocol"].get("zero_shot"):
        raise ValueError(f"Not a zero-shot evaluation: {path}")
    return payload


def assert_paired_protocol(base: dict, full: dict) -> None:
    """Reject comparisons where anything except the ECIA ablation changed."""

    base_protocol, full_protocol = base["protocol"], full["protocol"]
    paired_fields = (
        "test_sequences",
        "hive_checkpoint_sha256",
        "reid_model_sha256",
        "oracle_switch_trigger",
    )
    for field in paired_fields:
        if base_protocol[field] != full_protocol[field]:
            raise ValueError(f"Unpaired protocol field {field!r}")
    if base["test_baseline"]["aggregate"] != full["test_baseline"]["aggregate"]:
        raise ValueError("Base-only and Full do not share the exact same tracker baseline")
    if base["correction"]["events"] != full["correction"]["events"]:
        raise ValueError("Base-only and Full were not evaluated on the same event count")
    if base["correction"]["hr_requests"] != full["correction"]["hr_requests"]:
        raise ValueError("Base-only and Full do not have the same HIVE/HR request budget")


def metric_row(name: str, metrics: dict, correction: dict | None = None) -> dict:
    row = {
        "method": name,
        "idsw": int(metrics["id_switches"]),
        "idf1": float(metrics["idf1"]),
        "hota": float(metrics["hota"]),
        "assa": float(metrics["assa"]),
        "mota": float(metrics["mota"]),
        "pred_tracks": int(metrics["pred_tracks"]),
        "fixed": 0,
        "wrong": 0,
    }
    if correction is not None:
        row["fixed"] = int(correction["actually_fixed"])
        row["wrong"] = int(correction["wrong_corrections"])
    return row


def posthoc_hive_selection_summary(full_metrics_path: Path) -> dict | None:
    """Join frozen HIVE choices to labels only after inference for reporting."""

    run_root = full_metrics_path.resolve().parent.parent
    decisions_path = full_metrics_path.resolve().parent / "decisions_gt_free.jsonl"
    event_root = run_root / "event_protocol"
    if not decisions_path.is_file() or not event_root.is_dir():
        return None
    events = {}
    for path in sorted(event_root.glob("*_data/events.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            events[event["event_id"]] = event
    correct = 0
    total = 0
    per_sequence: dict[str, dict[str, int]] = {}
    for line in decisions_path.read_text(encoding="utf-8").splitlines():
        decision = json.loads(line)
        event = events[decision["event_id"]]
        selected = next(
            candidate
            for candidate in event["candidates"]
            if candidate["candidate_id"] == decision["selected_candidate_id"]
        )
        is_correct = int(selected["label"] == 1)
        correct += is_correct
        total += 1
        row = per_sequence.setdefault(event["sequence"], {"correct": 0, "total": 0})
        row["correct"] += is_correct
        row["total"] += 1
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "per_sequence": per_sequence,
        "gt_used_only_for_posthoc_reporting": True,
    }


def plot_metrics(
    output: Path, rows: list[dict], scope_label: str, methods: tuple[tuple[str, str, str], ...]
) -> None:
    specs = (
        ("idsw", "IDSW ↓", "{:.0f}"),
        ("idf1", "IDF1 ↑", "{:.4f}"),
        ("hota", "HOTA ↑", "{:.4f}"),
        ("assa", "AssA ↑", "{:.4f}"),
    )
    labels = [row["method"] for row in rows]
    colors = [entry[2] for entry in methods]
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for axis, (key, title, fmt) in zip(axes.flat, specs):
        values = [float(row[key]) for row in rows]
        bars = axis.bar(np.arange(len(rows)), values, color=colors)
        axis.set_title(title)
        axis.set_xticks(np.arange(len(rows)), labels, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, labels=[fmt.format(value) for value in values], padding=3)
    figure.suptitle(f"{scope_label}: contribution of HR evidence", fontsize=15)
    figure.savefig(output / "base_only_vs_full_metrics.png", dpi=180)
    plt.close(figure)


def plot_per_sequence(
    output: Path,
    sequence_rows: list[dict],
    methods: tuple[tuple[str, str, str], ...],
) -> None:
    sequences = [row["sequence"] for row in sequence_rows]
    x = np.arange(len(sequences))
    width = 0.25
    figure, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    for index, (key, label, color) in enumerate(methods):
        bars = axis.bar(
            x + (index - 1) * width,
            [row[key]["id_switches"] for row in sequence_rows],
            width,
            label=label,
            color=color,
        )
        axis.bar_label(bars, padding=2, fontsize=8)
    axis.set_xticks(x, sequences, rotation=15, ha="right")
    axis.set_ylabel("ID switches (lower is better)")
    axis.set_title("Base-only vs Full for every DTB video")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output / "per_sequence_idsw.png", dpi=180)
    plt.close(figure)


def plot_corrections(output: Path, rows: list[dict]) -> None:
    """Compare successful and harmful recovery operations for both ablations."""

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ablations = rows[1:]
    x = np.arange(len(ablations))
    width = 0.34
    fixed = axis.bar(
        x - width / 2,
        [row["fixed"] for row in ablations],
        width,
        label="Actually fixed",
        color="#10b981",
    )
    wrong = axis.bar(
        x + width / 2,
        [row["wrong"] for row in ablations],
        width,
        label="Wrong correction",
        color="#ef4444",
    )
    axis.set_xticks(x, [row["method"] for row in ablations])
    axis.set_ylabel("Events")
    axis.set_title("Recovery outcomes under the paired event protocol")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.bar_label(fixed, padding=3)
    axis.bar_label(wrong, padding=3)
    figure.savefig(output / "base_only_vs_full_corrections.png", dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    base = load_metrics(args.base_only.resolve(), "base_only")
    full = load_metrics(args.full.resolve(), "full")
    assert_paired_protocol(base, full)
    physical_lr = bool(full["protocol"].get("physical_lr_detection_tracking", False))
    physical_size = full["protocol"].get("physical_lr_size")
    physical_size_label = (
        f"{int(physical_size[0])}x{int(physical_size[1])}"
        if physical_size and len(physical_size) == 2
        else "unknown"
    )
    methods = (
        (
            "baseline",
            f"YOLO11l + BoT-SORT/ReID @ {physical_size_label}"
            if physical_lr
            else METHODS[0][1],
            METHODS[0][2],
        ),
        METHODS[1],
        METHODS[2],
    )

    baseline_metrics = full["test_baseline"]["aggregate"]
    base_metrics = base["test_ecia_base_only"]["aggregate"]
    full_metrics = full["test_ecia_full"]["aggregate"]
    rows = [
        metric_row(methods[0][1], baseline_metrics),
        metric_row(methods[1][1], base_metrics, base["correction"]),
        metric_row(methods[2][1], full_metrics, full["correction"]),
    ]
    delta = {
        "idsw_reduction": int(base_metrics["id_switches"] - full_metrics["id_switches"]),
        "idf1_absolute": float(full_metrics["idf1"] - base_metrics["idf1"]),
        "hota_absolute": float(full_metrics["hota"] - base_metrics["hota"]),
        "assa_absolute": float(full_metrics["assa"] - base_metrics["assa"]),
        "mota_absolute": float(full_metrics["mota"] - base_metrics["mota"]),
        "additional_fixed": int(
            full["correction"]["actually_fixed"] - base["correction"]["actually_fixed"]
        ),
        "additional_wrong": int(
            full["correction"]["wrong_corrections"] - base["correction"]["wrong_corrections"]
        ),
    }
    # This is deliberately phrased as evaluation-set evidence, not universal proof.
    delta["hr_helpful_on_this_evaluation_set"] = bool(
        delta["idsw_reduction"] > 0
        and delta["hota_absolute"] > 0
        and delta["additional_wrong"] <= 0
    )
    sequences = list(full["protocol"].get("evaluation_sequences", full["protocol"]["test_sequences"]))
    sequence_rows = []
    for sequence in sequences:
        sequence_rows.append(
            {
                "sequence": sequence,
                "baseline": full["test_baseline"]["sequences"][sequence],
                "base_only": base["test_ecia_base_only"]["sequences"][sequence],
                "full": full["test_ecia_full"]["sequences"][sequence],
            }
        )
    scope = full["protocol"].get("evaluation_scope", "held_out_test")
    scope_label = (
        f"All DTB videos with physical {physical_size_label} tracking"
        if scope == "all_dtb_physical_lr_zero_shot"
        else "All unseen DTB videos"
        if scope == "all_dtb_zero_shot"
        else "Held-out DTB test"
    )
    comparison = {
        "protocol": {
            "paired_events": True,
            "zero_shot": True,
            "panorama_used_for_training_or_tuning": False,
            "same_hive_and_reid": True,
            "same_hr_request_budget": True,
            "test_sequences": full["protocol"]["test_sequences"],
            "configured_test_sequences": full["protocol"].get(
                "configured_test_sequences", full["protocol"]["test_sequences"]
            ),
            "evaluation_scope": scope,
            "physical_lr_detection_tracking": bool(
                full["protocol"].get("physical_lr_detection_tracking", False)
            ),
            "physical_lr_size": full["protocol"].get("physical_lr_size"),
            "events": full["correction"]["events"],
            "hr_requests_per_method": full["correction"]["hr_requests"],
        },
        "rows": rows,
        "per_sequence": sequence_rows,
        "full_minus_base_only": delta,
        "hive_selection_posthoc": posthoc_hive_selection_summary(args.full),
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    plot_metrics(output, rows, scope_label, methods)
    plot_corrections(output, rows)
    plot_per_sequence(output, sequence_rows, methods)

    verdict = (
        "HR evidence improved the frozen Full model on this evaluation set."
        if delta["hr_helpful_on_this_evaluation_set"]
        else "This evaluation set does not show a clean benefit from HR evidence."
    )
    report = [
        f"# ECIA Base-only vs Full on {scope_label}",
        "",
        "This paired zero-shot ablation isolates the HR-enabled ECIA path: both methods use the same switch events, frozen HIVE/ReID inputs, replay code, and QuadTrack-selected checkpoints/thresholds. DTB train/validation/test labels do not select either method.",
        "",
        "| Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | MOTA ↑ | Pred. tracks ↓ | Fixed | Wrong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['method']} | {row['idsw']} | {row['idf1']:.4f} | "
            f"{row['hota']:.4f} | {row['assa']:.4f} | {row['mota']:.4f} | "
            f"{row['pred_tracks']} | {row['fixed']} | {row['wrong']} |"
        )
    report.extend(
        [
            "",
            "## Per-video comparison",
            "",
            "| Video | Method | IDSW ↓ | IDF1 ↑ | MOTA ↑ |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for sequence_row in sequence_rows:
        for key, label, _ in methods:
            metrics = sequence_row[key]
            report.append(
                f"| {sequence_row['sequence']} | {label} | {metrics['id_switches']} | "
                f"{metrics['idf1']:.4f} | {metrics['mota']:.4f} |"
            )
    report.extend(
        [
            "",
            "## Physical-LR protocol and event coverage",
            "",
            (
                f"- Detector/tracker input: physical `{physical_size_label.replace('x', '×')}` frames; HR panorama is used only for manual GT evaluation and the single HIVE-selected HR crop."
                if physical_lr
                else "- Detector/tracker protocol: legacy full-resolution panorama baseline."
            ),
            f"- Tracker: `{methods[0][1]}`.",
            f"- Aggregate detection precision / recall: **{baseline_metrics['precision']:.4f} / {baseline_metrics['recall']:.4f}**.",
            f"- Oracle ID switches found / eligible recovery events: **{baseline_metrics['id_switches']} / {full['correction']['events']}**.",
            "- DTB labels are joined only after the candidate pool and decisions are frozen; they do not tune HIVE, ReID, ECIA, or thresholds.",
        ]
    )
    hive_summary = comparison["hive_selection_posthoc"]
    if hive_summary is not None:
        report.extend(
            [
                f"- Post-hoc HIVE correct-frame selections: **{hive_summary['correct']}/{hive_summary['total']} ({100.0 * hive_summary['accuracy']:.1f}%)**.",
                "  This diagnostic uses GT only after inference and does not imply that ECIA will recover the ID.",
            ]
        )
    configured_test = set(comparison["protocol"]["configured_test_sequences"])
    held_out_rows = [
        row for row in sequence_rows if row["sequence"] in configured_test
    ]
    if held_out_rows and scope in {"all_dtb_zero_shot", "all_dtb_physical_lr_zero_shot"}:
        report.extend(
            [
                "",
                "## Strict held-out interpretation",
                "",
                "The aggregate above is an exploratory zero-shot result over all five videos. Under the predeclared DTB split, the strict held-out test contains only:",
                "",
                "| Video | Baseline IDSW | Base-only IDSW | Full IDSW | Baseline IDF1 | Base-only IDF1 | Full IDF1 |",
                "|---|---:|---:|---:|---:|---:|---:|",
                *[
                    f"| {row['sequence']} | {row['baseline']['id_switches']} | {row['base_only']['id_switches']} | {row['full']['id_switches']} | {row['baseline']['idf1']:.4f} | {row['base_only']['idf1']:.4f} | {row['full']['idf1']:.4f} |"
                    for row in held_out_rows
                ],
                "",
                "Therefore, improvement on the five-video aggregate must not be reported as improvement on the strict held-out test.",
            ]
        )
    report.extend(
        [
            "",
            "## Full minus Base-only",
            "",
            f"- Additional ID switches removed: **{delta['idsw_reduction']}**",
            f"- IDF1 absolute change: **{delta['idf1_absolute']:+.4f}**",
            f"- HOTA absolute change: **{delta['hota_absolute']:+.4f}**",
            f"- AssA absolute change: **{delta['assa_absolute']:+.4f}**",
            f"- Additional correct / wrong corrections: **{delta['additional_fixed']:+d} / {delta['additional_wrong']:+d}**",
            "",
            f"**Zero-shot verdict:** {verdict}",
            "",
            (
                "This is cross-dataset zero-shot evidence over all available DTB videos, not statistical proof across the full deployment distribution. These videos can no longer be used for tuning while also being claimed as unseen evaluation. The oracle switch trigger means the experiment evaluates recovery, not online switch detection."
                if scope in {"all_dtb_zero_shot", "all_dtb_physical_lr_zero_shot"}
                else "This is evidence on one held-out DTB sequence, not statistical proof across the full deployment distribution. The oracle switch trigger means the experiment evaluates recovery, not online switch detection."
            ),
            "",
            "![Tracking ablation](base_only_vs_full_metrics.png)",
            "",
            "![Correction ablation](base_only_vs_full_corrections.png)",
            "",
            "![Per-video ID switches](per_sequence_idsw.png)",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"DTB ECIA ablation report: {output / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
