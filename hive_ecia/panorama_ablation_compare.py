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


def plot_metrics(output: Path, rows: list[dict], scope_label: str) -> None:
    specs = (
        ("idsw", "IDSW ↓", "{:.0f}"),
        ("idf1", "IDF1 ↑", "{:.4f}"),
        ("hota", "HOTA ↑", "{:.4f}"),
        ("assa", "AssA ↑", "{:.4f}"),
    )
    labels = [row["method"] for row in rows]
    colors = [entry[2] for entry in METHODS]
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


def plot_per_sequence(output: Path, sequence_rows: list[dict]) -> None:
    sequences = [row["sequence"] for row in sequence_rows]
    x = np.arange(len(sequences))
    width = 0.25
    figure, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    for index, (key, label, color) in enumerate(METHODS):
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

    baseline_metrics = full["test_baseline"]["aggregate"]
    base_metrics = base["test_ecia_base_only"]["aggregate"]
    full_metrics = full["test_ecia_full"]["aggregate"]
    rows = [
        metric_row(METHODS[0][1], baseline_metrics),
        metric_row(METHODS[1][1], base_metrics, base["correction"]),
        metric_row(METHODS[2][1], full_metrics, full["correction"]),
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
    scope_label = "All unseen DTB videos" if scope == "all_dtb_zero_shot" else "Held-out DTB test"
    comparison = {
        "protocol": {
            "paired_events": True,
            "zero_shot": True,
            "panorama_used_for_training_or_tuning": False,
            "same_hive_and_reid": True,
            "same_hr_request_budget": True,
            "test_sequences": full["protocol"]["test_sequences"],
            "evaluation_scope": scope,
            "events": full["correction"]["events"],
            "hr_requests_per_method": full["correction"]["hr_requests"],
        },
        "rows": rows,
        "per_sequence": sequence_rows,
        "full_minus_base_only": delta,
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    plot_metrics(output, rows, scope_label)
    plot_corrections(output, rows)
    plot_per_sequence(output, sequence_rows)

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
        for key, label, _ in METHODS:
            metrics = sequence_row[key]
            report.append(
                f"| {sequence_row['sequence']} | {label} | {metrics['id_switches']} | "
                f"{metrics['idf1']:.4f} | {metrics['mota']:.4f} |"
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
                if scope == "all_dtb_zero_shot"
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
