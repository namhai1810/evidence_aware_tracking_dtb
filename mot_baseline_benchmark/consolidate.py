"""Combine QuadTrack and DTB fair-MOT reports at the requested fixed path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import METHOD_LABELS


def bold(value, best, formatted: str) -> str:
    return f"**{formatted}**" if abs(float(value) - float(best)) <= 1e-12 else formatted


def format_best(value, best, spec: str = "") -> str:
    return bold(value, best, format(value, spec))


def paper_rank(value, candidates, *, higher_is_better: bool, spec: str = "") -> str:
    """Bold best and underline the second-best distinct paper result."""

    ordered = sorted({float(item) for item in candidates}, reverse=higher_is_better)
    formatted = format(value, spec)
    if abs(float(value) - ordered[0]) <= 1e-12:
        return f"**{formatted}**"
    if len(ordered) > 1 and abs(float(value) - ordered[1]) <= 1e-12:
        return f"<u>{formatted}</u>"
    return formatted


def plot_paper_summary(datasets: dict, output: Path) -> None:
    """Create a compact identity-centric figure for the main paper."""

    methods = ("sort", "bytetrack", "bytetrack_hive", "botsort", "hive_ecia")
    labels = ("SORT", "ByteTrack", "ByteTrack\n+ HIVE–ECIA", "BoT-SORT", "BoT-SORT\n+ HIVE–ECIA")
    colors = ("#9CA3AF", "#7AA6C2", "#356D8C", "#74A57F", "#E76F51")
    identity_metrics = (("idf1", "IDF1"), ("hota", "HOTA"), ("assa", "AssA"))
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), gridspec_kw={"width_ratios": [1.65, 1]})
    for row_index, (dataset_label, payload) in enumerate(datasets.items()):
        aggregate = {
            method: payload["methods"][method]["aggregate"] for method in methods
        }
        axis = axes[row_index, 0]
        x = np.arange(len(methods))
        bar_width = 0.23
        metric_colors = ("#355F8A", "#4FA3A5", "#8AB17D")
        # Emphasize both tracker variants that include our HIVE--ECIA flow.
        for method_index, fill in ((2, "#E8F1FB"), (4, "#FCE8E1")):
            axis.axvspan(
                x[method_index] - 0.48,
                x[method_index] + 0.48,
                color=fill,
                alpha=0.95,
                zorder=0,
            )
        for metric_index, ((metric, label), color) in enumerate(
            zip(identity_metrics, metric_colors)
        ):
            values = [float(aggregate[method][metric]) for method in methods]
            offset = (metric_index - 1) * bar_width
            bars = axis.bar(
                x + offset,
                values,
                width=bar_width,
                color=color,
                label=label,
                alpha=0.72,
                zorder=2,
            )
            for method_index, bar in enumerate(bars):
                bar.set_alpha(0.32 if method_index not in (2, 4) else 1.0)
            for method_index, edge in ((2, "#1D5D91"), (4, "#B33A20")):
                bars[method_index].set_edgecolor(edge)
                bars[method_index].set_linewidth(2.0)
                bars[method_index].set_hatch("//")
            # Numerical labels make the differences explicit without changing
            # the underlying values.
            for method_index in (1, 2, 3, 4):
                value = values[method_index]
                axis.text(
                    x[method_index] + offset,
                    value + 0.003,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=6.2,
                    color="#243447",
                    fontweight="bold" if method_index in (2, 4) else "normal",
                )
        axis.set_xticks(x, labels, fontsize=8)
        axis.set_ylim((0.25, 0.515) if row_index == 0 else (0.20, 0.40))
        axis.set_ylabel("Score ↑")
        axis.set_title(f"{dataset_label} — identity quality", loc="left", fontweight="bold")
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncol=3, loc="upper left")
        for method_index, color in ((2, "#1D5D91"), (4, "#B33A20")):
            axis.get_xticklabels()[method_index].set_color(color)
            axis.get_xticklabels()[method_index].set_fontweight("bold")

        axis = axes[row_index, 1]
        values = [int(aggregate[method]["id_switches"]) for method in methods]
        idsw_colors = ("#C7CBD1", "#C7CBD1", "#4E8FC1", "#C7CBD1", "#E76F51")
        bars = axis.bar(x, values, color=idsw_colors, width=0.72)
        axis.bar_label(bars, padding=3, fontsize=9)
        axis.set_xticks(x, labels, fontsize=8)
        axis.set_ylabel("ID switches ↓")
        axis.set_title(f"{dataset_label} — identity errors", loc="left", fontweight="bold")
        axis.set_ylim(0, max(values) * 1.30)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        # Outline the proposed tracker-transfer variants without changing
        # their values.
        for method_index, edge in ((2, "#1D5D91"), (4, "#8C2F1B")):
            bars[method_index].set_edgecolor(edge)
            bars[method_index].set_linewidth(2.0)
            bars[method_index].set_hatch("//")
            axis.get_xticklabels()[method_index].set_color(edge)
            axis.get_xticklabels()[method_index].set_fontweight("bold")
        comparisons = (
            (1, 2, "#356D8C", max(values) * 1.10),
            (3, 4, "#B33A20", max(values) * 1.20),
        )
        for baseline_index, improved_index, color, height in comparisons:
            reduction = values[baseline_index] - values[improved_index]
            relative = reduction / values[baseline_index]
            axis.annotate(
                "",
                xy=(improved_index, height),
                xytext=(baseline_index, height),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.8},
            )
            axis.text(
                (baseline_index + improved_index) / 2,
                height + max(values) * 0.015,
                f"−{reduction} ({relative:.1%})",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color=color,
            )

    fig.suptitle(
        "Fair MOT identity-tracking comparison",
        fontsize=16,
        fontweight="bold",
        x=0.055,
        ha="left",
    )
    fig.text(
        0.055,
        0.94,
        "Frozen detector and common evaluation protocol; lower IDSW and higher identity scores are better.",
        fontsize=9.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.91), h_pad=2.3, w_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadtrack", type=Path, required=True)
    parser.add_argument("--dtb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = {
        "QuadTrack held-out test (512×120)": json.loads(args.quadtrack.read_text(encoding="utf-8")),
        "DTB zero-shot (640×320)": json.loads(args.dtb.read_text(encoding="utf-8")),
    }
    summary_figure = args.output.resolve().parent / "figures" / "fair_mot_identity_summary.png"
    plot_paper_summary(datasets, summary_figure)
    lines = [
        "# Fair MOT baseline benchmark",
        "",
        "Frozen Existing Base-Guided Safety (HIVE–ECIA) is compared with standard SORT, ByteTrack and BoT-SORT. No online-aware v1 or balanced-v2 checkpoint is used.",
        "",
        "## Paper-ready summary",
        "",
        "| Method | QuadTrack IDF1 ↑ | HOTA ↑ | AssA ↑ | IDSW ↓ | DTB IDF1 ↑ | HOTA ↑ | AssA ↑ | IDSW ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    method_order = ("sort", "bytetrack", "bytetrack_hive", "botsort", "hive_ecia")
    short_labels = {
        "sort": "SORT",
        "bytetrack": "ByteTrack",
        "bytetrack_hive": "ByteTrack + HIVE–ECIA†",
        "botsort": "BoT-SORT",
        "hive_ecia": "BoT-SORT + HIVE–ECIA (Ours)",
    }
    ranked = {
        dataset_label: {
            metric: [
                payload["methods"][method]["aggregate"][metric]
                for method in method_order
            ]
            for metric in ("idf1", "hota", "assa", "id_switches")
        }
        for dataset_label, payload in datasets.items()
    }
    for method in method_order:
        cells = []
        for dataset_label, payload in datasets.items():
            row = payload["methods"][method]["aggregate"]
            for metric in ("idf1", "hota", "assa", "id_switches"):
                cells.append(
                    paper_rank(
                        row[metric],
                        ranked[dataset_label][metric],
                        higher_is_better=metric != "id_switches",
                        spec=".4f" if metric != "id_switches" else "",
                    )
                )
        lines.append(f"| {short_labels[method]} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "**Table:** Identity-centric comparison using the same YOLO11l detections and evaluation protocol. **Bold** denotes best and <u>underline</u> denotes second-best per dataset/metric. †Zero-shot tracker transfer; the primary proposed configuration is BoT-SORT + HIVE–ECIA.",
        "",
        "![Fair MOT identity comparison](figures/fair_mot_identity_summary.png)",
        "",
        "**Figure:** Identity quality and ID-switch comparison under the frozen common protocol. The orange-highlighted group denotes our primary BoT-SORT + HIVE–ECIA configuration.",
        "",
        "## Aggregate results",
        "",
        "| Dataset | Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | MOTA ↑ | IDTP | IDFP | IDFN | FPS ↑ | Latency ms ↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset_label, payload in datasets.items():
        methods = payload["methods"]
        best = {
            "id_switches": min(row["aggregate"]["id_switches"] for row in methods.values()),
            **{
                metric: max(row["aggregate"][metric] for row in methods.values())
                for metric in ("idf1", "hota", "assa", "mota", "idtp")
            },
            **{
                metric: min(row["aggregate"][metric] for row in methods.values())
                for metric in ("idfp", "idfn")
            },
        }
        best_fps = max(row["timing"]["end_to_end_fps"] for row in methods.values())
        best_latency = min(
            row["timing"]["end_to_end_latency_ms"] for row in methods.values()
        )
        for method in ("sort", "bytetrack", "bytetrack_hive", "botsort", "hive_ecia"):
            row = methods[method]["aggregate"]
            timing = methods[method]["timing"]
            lines.append(
                f"| {dataset_label} | {METHOD_LABELS[method]} | "
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
        "## Direct tracker-family comparisons",
        "",
        "| Dataset | Tracker | Baseline IDSW | + HIVE–ECIA IDSW | Reduction | Relative | ΔIDF1 | ΔHOTA | ΔAssA | ΔMOTA |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset_label, payload in datasets.items():
        for family, baseline_name, full_name in (
            ("ByteTrack", "bytetrack", "bytetrack_hive"),
            ("BoT-SORT", "botsort", "hive_ecia"),
        ):
            baseline = payload["methods"][baseline_name]["aggregate"]
            full = payload["methods"][full_name]["aggregate"]
            delta = payload["direct_improvements"][baseline_name]
            lines.append(
                f"| {dataset_label} | {family} | {baseline['id_switches']} | "
                f"{full['id_switches']} | {delta['idsw_reduction']:+d} | "
                f"{delta['idsw_relative_reduction']:+.2%} | "
                f"{delta['idf1_absolute']:+.4f} | {delta['hota_absolute']:+.4f} | "
                f"{delta['assa_absolute']:+.4f} | {delta['mota_absolute']:+.4f} |"
            )
    lines.extend([
        "",
        "BoT-SORT + HIVE–ECIA is the original Existing Base-Guided Safety pipeline. ByteTrack + HIVE–ECIA is a zero-shot tracker-transfer integration using the same frozen checkpoints, thresholds and safety policy without ByteTrack/test/DTB tuning.",
        "",
        "## Fairness checks",
        "",
    ])
    for dataset_label, payload in datasets.items():
        checks = payload["fairness_checks"]
        passed = all(
            checks[key]
            for key in (
                "botsort_reference_passed",
                "cached_botsort_replay_equivalent",
                "official_botsort_equals_live",
                "hive_raw_equals_botsort",
                "bytetrack_hive_raw_association_equivalent",
                "same_detection_cache_for_standard_trackers",
                "sequence_reset_verified_by_fresh_tracker_instances",
                "no_gt_during_tracking",
            )
        )
        lines.append(f"- {dataset_label}: **{'PASS' if passed else 'FAIL'}**.")
    lines.extend([
        "- Shared detector: YOLO11l, person class only, confidence 0.20 and NMS IoU 0.70.",
        "- QuadTrack sequences: 0016, 0017, 0025, 0031. DTB sequences: normal_walking, crossing, partial_occ, seam, full_occ.",
        "- GT is loaded only after MOT predictions are frozen.",
        "- ByteTrack + HIVE–ECIA uses baseline ByteTrack geometry and recovered canonical IDs; raw association equivalence is checked before normalization.",
        "- Standard tracker timing reports shared detector latency plus tracker replay latency; HIVE–ECIA timing measures detector + BoT-SORT + recovery controller. Decode/resize/write I/O is excluded.",
        "",
        "## Detailed reports",
        "",
        f"- QuadTrack: `{args.quadtrack.parent / 'REPORT.md'}`",
        f"- DTB: `{args.dtb.parent / 'REPORT.md'}`",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Consolidated report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

