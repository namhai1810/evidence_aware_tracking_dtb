"""Consolidate QuadTrack and DTB tracker-family transfer results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAIRS = (
    ("sort", "sort_hive", "SORT"),
    ("bytetrack", "bytetrack_hive", "ByteTrack"),
    ("ocsort", "ocsort_hive", "OC-SORT"),
    ("botsort", "hive_ecia", "BoT-SORT"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadtrack", type=Path, required=True)
    parser.add_argument("--dtb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--figure",
        type=Path,
        default=None,
        help="Optional paper-ready before/after comparison figure.",
    )
    return parser.parse_args()


def plot_tracker_family_transfer(datasets: dict[str, dict], output: Path) -> None:
    """Visualize the controlled baseline versus +HIVE comparison."""
    labels = [label for _, _, label in PAIRS]
    metrics = (
        ("idf1", "IDF1 ↑"),
        ("hota", "HOTA ↑"),
        ("id_switches", "ID switches ↓"),
    )
    baseline_color = "#8fa3b8"
    hive_color = "#e76f51"
    x = np.arange(len(labels))
    width = 0.34

    plt.rcParams.update({
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.3), constrained_layout=False)
    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.07, right=0.98, hspace=0.52, wspace=0.28)

    for row, (dataset_name, payload) in enumerate(datasets.items()):
        for col, (metric, title) in enumerate(metrics):
            ax = axes[row, col]
            baseline = [payload["methods"][base]["aggregate"][metric] for base, _, _ in PAIRS]
            recovered = [payload["methods"][full]["aggregate"][metric] for _, full, _ in PAIRS]
            left = ax.bar(
                x - width / 2,
                baseline,
                width,
                color=baseline_color,
                edgecolor="#5d7185",
                linewidth=0.8,
                label="Tracker baseline",
                zorder=3,
            )
            right = ax.bar(
                x + width / 2,
                recovered,
                width,
                color=hive_color,
                edgecolor="#b83d21",
                linewidth=1.1,
                hatch="//",
                label="+ HIVE–ECIA",
                zorder=3,
            )
            ax.set_title(title, pad=9)
            ax.set_xticks(x, labels)
            ax.tick_params(axis="x", labelsize=9)
            ax.grid(axis="y", alpha=0.22, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)

            all_values = baseline + recovered
            if metric != "id_switches":
                low = max(0.0, min(all_values) - 0.035)
                high = max(all_values) + 0.035
                ax.set_ylim(low, high)
                fmt = "%.3f"
            else:
                ax.set_ylim(0, max(all_values) * 1.28)
                fmt = "%d"

            for bars in (left, right):
                for bar in bars:
                    value = bar.get_height()
                    label = fmt % value
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.018,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90 if metric != "id_switches" else 0,
                    )

            if metric == "id_switches":
                for index, (before, after) in enumerate(zip(baseline, recovered)):
                    reduction = 100.0 * (before - after) / before
                    ax.text(
                        index,
                        max(before, after) + ax.get_ylim()[1] * 0.105,
                        f"−{reduction:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="#a63219",
                        fontweight="bold",
                    )

            if col == 0:
                short_name = "QuadTrack held-out (512×120)" if row == 0 else "DTB zero-shot (640×320)"
                ax.set_ylabel(short_name, labelpad=10)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.suptitle(
        "Frozen HIVE–ECIA transfer across association trackers",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    datasets = {
        "QuadTrack held-out test (512×120)": json.loads(args.quadtrack.read_text(encoding="utf-8")),
        "DTB zero-shot (640×320)": json.loads(args.dtb.read_text(encoding="utf-8")),
    }
    lines = [
        "# HIVE–ECIA tracker-family transfer benchmark", "",
        "The frozen Existing Base-Guided Safety pipeline is attached to four association backends. Every pair uses the same YOLO11l detections, LR frames and evaluation settings; no tracker-specific test/DTB tuning is used.", "",
        "## Direct before/after comparison", "",
        "| Dataset | Association | IDSW baseline | IDSW + HIVE | Reduction | IDF1 baseline | IDF1 + HIVE | ΔIDF1 | ΔHOTA | ΔAssA | HR requests |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, payload in datasets.items():
        for base, full, label in PAIRS:
            left = payload["methods"][base]["aggregate"]
            right = payload["methods"][full]["aggregate"]
            delta = payload["improvements"][base]
            requests = payload["methods"][full]["timing"].get("hr_requests", 0)
            lines.append(
                f"| {dataset} | {label} | {left['id_switches']} | {right['id_switches']} | "
                f"{delta['idsw_reduction']:+d} ({delta['idsw_relative_reduction']:+.2%}) | {left['idf1']:.4f} | {right['idf1']:.4f} | "
                f"{delta['delta_idf1']:+.4f} | {delta['delta_hota']:+.4f} | {delta['delta_assa']:+.4f} | {requests} |"
            )
    lines += [
        "", "## Interpretation boundary", "",
        "- BoT-SORT is the primary trained configuration. ByteTrack, SORT and OC-SORT are zero-shot tracker transfers of the same frozen HIVE/ECIA checkpoints and thresholds.",
        "- Baseline and +HIVE rows inside each family share the same raw tracker output; the recovery controller is permitted to change canonical IDs only.",
        "- OC-SORT uses repository-vendored reference defaults: detection threshold 0.25, max age 30, minimum hits 3, IoU threshold 0.30, delta_t 3, inertia 0.20 and Byte second pass disabled.",
        "- GT is loaded only for offline evaluation after every MOT prediction is written.",
        "", "## Fairness checks", "",
    ]
    for dataset, payload in datasets.items():
        lines.append(f"### {dataset}")
        for key, value in payload["fairness"].items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    lines += [
        "## Detailed reports", "",
        f"- QuadTrack: `{args.quadtrack.parent / 'REPORT.md'}`",
        f"- DTB: `{args.dtb.parent / 'REPORT.md'}`",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.figure is not None:
        plot_tracker_family_transfer(datasets, args.figure)
        print(f"Tracker-family figure: {args.figure.resolve()}")
    print(f"Consolidated tracker-family report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

