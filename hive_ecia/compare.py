"""Aggregate Base-only, HR-Gain, and Full ECIA held-out evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DISPLAY = {
    "base_only": "ECIA Base-only",
    "base_hr_gain": "ECIA Base + HR Gain",
    "full": "ECIA Full (Gain + Gate)",
}


def plot_tracking_metrics(output_dir: Path, rows: list[dict]) -> None:
    """Visualize tracking quality without collapsing unlike metric scales."""

    labels = [row["display"] for row in rows]
    colors = ["#64748b", "#f59e0b", "#38bdf8", "#8b5cf6", "#10b981"]
    specs = (
        ("idsw", "ID Switches (lower is better)"),
        ("idf1", "IDF1 (higher is better)"),
        ("hota", "HOTA (higher is better)"),
        ("mota", "MOTA (higher is better)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, specs):
        values = [float(row[key]) for row in rows]
        bars = axis.bar(np.arange(len(rows)), values, color=colors)
        axis.set_title(title)
        axis.set_xticks(np.arange(len(rows)), labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(
            bars,
            labels=[str(int(value)) if key == "idsw" else f"{value:.4f}" for value in values],
            padding=3,
        )
    figure.suptitle("Held-out QuadTrack tracking comparison", fontsize=15)
    figure.savefig(output_dir / "tracking_metrics.png", dpi=180)
    plt.close(figure)


def plot_corrections(output_dir: Path, rows: list[dict]) -> None:
    """Show true recoveries and harmful corrections side by side."""

    labels = [row["display"] for row in rows]
    fixed = [int(row["actually_fixed"]) for row in rows]
    wrong = [int(row["wrong_corrections"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    fixed_bars = axis.bar(x - width / 2, fixed, width, label="Actually fixed", color="#10b981")
    wrong_bars = axis.bar(x + width / 2, wrong, width, label="Wrong corrections", color="#ef4444")
    axis.set_title("Identity-recovery outcomes on held-out test")
    axis.set_ylabel("Number of events")
    axis.set_xticks(x, labels, rotation=18, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.bar_label(fixed_bars, padding=3)
    axis.bar_label(wrong_bars, padding=3)
    figure.savefig(output_dir / "correction_outcomes.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="mode=/absolute/or/relative/metrics.json",
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        required=True,
        help="summary.json from the unchanged hive_hr_reid baseline.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = {}
    for spec in args.run:
        mode, path = spec.split("=", 1)
        if mode not in DISPLAY:
            raise ValueError(f"Unknown mode {mode}")
        runs[mode] = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(runs) != set(DISPLAY):
        raise ValueError(f"Expected all ablations: {sorted(DISPLAY)}")

    baseline = next(iter(runs.values()))["baseline"]["aggregate"]
    baseline_payload = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    hive_baseline = baseline_payload["results"]["hive_v2_hr_reid"]
    baseline_hr_requests = int(
        baseline_payload["protocol"]["held_out_hr_requests"]["hive_v2_hr_reid"]
    )
    rows = []
    for mode in ("base_only", "base_hr_gain", "full"):
        payload = runs[mode]
        metrics = payload["ecia"]["aggregate"]
        correction = payload["correction"]
        rows.append(
            {
                "mode": mode,
                "display": DISPLAY[mode],
                "idsw": metrics["id_switches"],
                "idf1": metrics["idf1"],
                "hota": metrics["hota"],
                "mota": metrics["mota"],
                "hr_requests": int(payload["protocol"]["hr_requests"]),
                **correction,
            }
        )
    unequal_budgets = {
        row["mode"]: row["hr_requests"]
        for row in rows
        if row["hr_requests"] != baseline_hr_requests
    }
    if unequal_budgets:
        raise RuntimeError(
            "Held-out HR budgets differ from the unchanged baseline "
            f"({baseline_hr_requests}): {unequal_budgets}"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_rows = [
        {
            "display": "LR-only",
            "idsw": baseline["id_switches"],
            "idf1": baseline["idf1"],
            "hota": baseline["hota"],
            "mota": baseline["mota"],
            "actually_fixed": 0,
            "wrong_corrections": 0,
        },
        {
            "display": "HIVE v2 + HR ReID",
            "idsw": hive_baseline["aggregate"]["id_switches"],
            "idf1": hive_baseline["aggregate"]["idf1"],
            "hota": hive_baseline["aggregate"]["hota"],
            "mota": hive_baseline["aggregate"]["mota"],
            "actually_fixed": hive_baseline["correction"]["actual_event_switches_fixed"],
            "wrong_corrections": hive_baseline["correction"]["wrong_corrections"],
        },
        *rows,
    ]
    plot_tracking_metrics(output_dir, visual_rows)
    plot_corrections(output_dir, visual_rows[1:])
    (output_dir / "comparison.json").write_text(
        json.dumps(
            {
                "baseline": baseline,
                "hive_v2_hr_reid_baseline": hive_baseline,
                "held_out_hr_requests_per_method": baseline_hr_requests,
                "hr_budget_equal": True,
                "predeclared_primary_method": "full",
                "ablations": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = [
        "# HIVE + Lightweight ECIA ablation comparison",
        "",
        "All checkpoints use the same frozen HIVE v2/ReID inputs, video-disjoint split, and validation-only threshold calibration.",
        "",
        "| Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | MOTA ↑ | Fixed | Wrong |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| LR-only | {baseline['id_switches']} | {baseline['idf1']:.4f} | {baseline['hota']:.4f} | {baseline['mota']:.4f} | 0 | 0 |",
        f"| HIVE v2 + HR ReID baseline | {hive_baseline['aggregate']['id_switches']} | {hive_baseline['aggregate']['idf1']:.4f} | {hive_baseline['aggregate']['hota']:.4f} | {hive_baseline['aggregate']['mota']:.4f} | {hive_baseline['correction']['actual_event_switches_fixed']} | {hive_baseline['correction']['wrong_corrections']} |",
    ]
    for row in rows:
        report.append(
            f"| {row['display']} | {row['idsw']} | {row['idf1']:.4f} | "
            f"{row['hota']:.4f} | {row['mota']:.4f} | {row['actually_fixed']} | "
            f"{row['wrong_corrections']} |"
        )
    report.extend(
        [
            "",
            "The predeclared primary method is **ECIA Full**; Base-only and Base + HR Gain are explanatory ablations, not candidates selected by test performance.",
            "",
            "The test split was not used to choose the architecture, checkpoint, loss weights, or merge thresholds.",
            "",
            f"All HIVE-based methods used the same held-out HR budget: **{baseline_hr_requests} requests**, with at most one HR crop per event.",
            "",
            "## Visual comparison",
            "",
            "![Held-out tracking metrics](tracking_metrics.png)",
            "",
            "![Correct and wrong identity corrections](correction_outcomes.png)",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Comparison report: {output_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
