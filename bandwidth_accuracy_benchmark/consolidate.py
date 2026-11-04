"""Create joint bandwidth report and update the project experimental report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


START = "<!-- BANDWIDTH_ACCURACY_BENCHMARK_START -->"
END = "<!-- BANDWIDTH_ACCURACY_BENCHMARK_END -->"
TRACKER_LABELS = {"sort": "SORT", "bytetrack": "ByteTrack", "botsort": "BoT-SORT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadtrack", type=Path, required=True)
    parser.add_argument("--dtb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experimental-report", type=Path, required=True)
    return parser.parse_args()


def bold(value, best, spec="") -> str:
    text = format(value, spec)
    return f"**{text}**" if abs(float(value) - float(best)) <= 1e-12 else text


def relative_link(target: Path, document_dir: Path) -> str:
    """Return a host/container-independent POSIX Markdown target."""

    return Path(os.path.relpath(target.resolve(), document_dir.resolve())).as_posix()


def plot_summary(datasets: dict, output: Path) -> None:
    """Paper-oriented comparison using the same BoT-SORT tracker family."""

    method_keys = [
        "full_hr_botsort",
        "periodic_5_botsort",
        "periodic_10_botsort",
        "periodic_30_botsort",
        "hive_ecia",
    ]
    labels = ["Full-HR", "Periodic\nN=5", "Periodic\nN=10", "Periodic\nN=30", "HIVE–ECIA"]
    colors = ["#9CA3AF", "#78A6C8", "#5F91B5", "#467A9F", "#E76F51"]
    metrics = [
        ("total_mb", "Transmitted data (MB) ↓", "bandwidth"),
        ("idf1", "IDF1 ↑", "metrics"),
        ("id_switches", "ID switches ↓", "metrics"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    for row_index, (dataset_label, payload) in enumerate(datasets.items()):
        methods = payload["methods"]
        for column_index, (key, title, source) in enumerate(metrics):
            axis = axes[row_index, column_index]
            values = []
            for method_key in method_keys:
                method = methods[method_key]
                if source == "bandwidth":
                    values.append(float(method["bandwidth"][key]))
                else:
                    values.append(float(method["metrics"]["aggregate"][key]))
            positions = np.arange(len(values))
            axis.axvspan(
                positions[-1] - 0.48,
                positions[-1] + 0.48,
                color="#FCE8E1",
                alpha=0.75,
                zorder=0,
            )
            bars = axis.bar(positions, values, color=colors, width=0.72, zorder=2)
            bars[-1].set_edgecolor("#9E321D")
            bars[-1].set_linewidth(2.0)
            bars[-1].set_hatch("//")
            axis.set_xticks(positions, labels, fontsize=8)
            axis.get_xticklabels()[-1].set_color("#B33A20")
            axis.get_xticklabels()[-1].set_fontweight("bold")
            axis.set_title(title, fontsize=11, fontweight="bold", loc="left")
            axis.grid(axis="y", alpha=0.22, linewidth=0.8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.bar_label(
                bars,
                labels=[f"{value:.2f}" if key == "total_mb" else (
                    f"{value:.3f}" if key == "idf1" else f"{int(value)}"
                ) for value in values],
                padding=3,
                fontsize=8,
            )
            upper = max(values) * 1.32 if values else 1.0
            axis.set_ylim(0, upper)
            n30, ours = values[-2], values[-1]
            if key in {"total_mb", "id_switches"}:
                relative_change = (n30 - ours) / n30 if n30 else 0.0
                change_label = f"−{relative_change:.1%} vs N=30"
            else:
                relative_change = (ours - n30) / n30 if n30 else 0.0
                change_label = f"+{relative_change:.1%} vs N=30"
            arrow_height = max(values) * 1.17
            axis.annotate(
                "",
                xy=(positions[-1], arrow_height),
                xytext=(positions[-2], arrow_height),
                arrowprops={"arrowstyle": "->", "color": "#B33A20", "lw": 1.9},
            )
            axis.text(
                (positions[-2] + positions[-1]) / 2,
                arrow_height + max(values) * 0.025,
                change_label,
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold",
                color="#B33A20",
            )
            if column_index == 0:
                axis.set_ylabel(dataset_label, fontsize=10, fontweight="bold")
    fig.suptitle(
        "Bandwidth–identity tracking trade-off (BoT-SORT fixed)",
        fontsize=15,
        fontweight="bold",
        x=0.06,
        ha="left",
    )
    fig.text(
        0.06,
        0.935,
        "Same tracker family; JPEG quality 90. HIVE–ECIA transmits continuous LR plus selected HR crops.",
        fontsize=9,
        color="#4B5563",
        ha="left",
    )
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_details(payload: dict, output_dir: Path) -> dict[str, Path]:
    """Create uncluttered same-tracker figures with explicit OURS deltas."""

    method_keys = [
        "full_hr_botsort",
        "periodic_5_botsort",
        "periodic_10_botsort",
        "periodic_30_botsort",
        "hive_ecia",
    ]
    labels = ["Full-HR", "Periodic N=5", "Periodic N=10", "Periodic N=30", "HIVE–ECIA (Ours)"]
    short_labels = ["Full-HR", "N=5", "N=10", "N=30", "OURS"]
    methods = payload["methods"]
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_key = payload["dataset_key"]
    dataset_label = {
        "quadtrack": "QuadTrack held-out test (512×120)",
        "dtb": "DTB zero-shot (640×320)",
    }.get(dataset_key, payload["dataset_label"])
    colors = ["#A3AAB5", "#91B4CD", "#6E9CBC", "#4F7FA1", "#E76F51"]

    bandwidth = [float(methods[key]["bandwidth"]["total_mb"]) for key in method_keys]
    figure, axis = plt.subplots(figsize=(10.5, 5.3))
    positions = np.arange(len(method_keys))
    axis.axvspan(positions[-1] - 0.48, positions[-1] + 0.48, color="#FCE8E1", alpha=0.8)
    bars = axis.bar(positions, bandwidth, color=colors, width=0.7, zorder=2)
    bars[-1].set_edgecolor("#9E321D")
    bars[-1].set_linewidth(2.3)
    axis.bar_label(bars, labels=[f"{value:.2f} MB" for value in bandwidth], padding=4, fontsize=9)
    axis.set_xticks(positions, labels, fontsize=9)
    axis.get_xticklabels()[-1].set_color("#B33A20")
    axis.get_xticklabels()[-1].set_fontweight("bold")
    axis.set_ylabel("Transmitted JPEG data (MB) ↓")
    axis.set_title(f"{dataset_label}: bandwidth by acquisition strategy", loc="left", fontweight="bold")
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylim(0, max(bandwidth) * 1.24)
    figure.tight_layout()
    bandwidth_path = output_dir / f"{dataset_key}_bandwidth_strategies.png"
    figure.savefig(bandwidth_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    mb = np.asarray(bandwidth)
    idf1 = np.asarray([methods[key]["metrics"]["aggregate"]["idf1"] for key in method_keys])
    idsw = np.asarray([methods[key]["metrics"]["aggregate"]["id_switches"] for key in method_keys])
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for axis, values, ylabel in (
        (axes[0], idf1, "IDF1 ↑"),
        (axes[1], idsw, "ID switches ↓"),
    ):
        axis.plot(mb[:-1], values[:-1], "o--", color="#668FAE", lw=1.5, ms=7, alpha=0.8)
        axis.scatter(mb[-1], values[-1], s=310, marker="*", color="#E76F51", edgecolor="#8C2F1B", linewidth=1.5, zorder=5)
        for index, label in enumerate(short_labels[:-1]):
            axis.annotate(label, (mb[index], values[index]), xytext=(5, 6), textcoords="offset points", fontsize=8, color="#4B5563")
        axis.annotate(
            "OURS",
            (mb[-1], values[-1]),
            xytext=(8, -15 if ylabel.startswith("ID switches") else 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#B33A20",
        )
        axis.annotate(
            "",
            xy=(mb[-1], values[-1]),
            xytext=(mb[-2], values[-2]),
            arrowprops={"arrowstyle": "->", "color": "#B33A20", "lw": 2.0},
        )
        axis.set_xlabel("Transmitted data (MB) ↓")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"{dataset_label}: identity accuracy versus bandwidth (BoT-SORT fixed)",
        x=0.055,
        ha="left",
        fontweight="bold",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.02, 0.02, 1, 0.93))
    tradeoff_path = output_dir / f"{dataset_key}_identity_tradeoff.png"
    figure.savefig(tradeoff_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return {"bandwidth": bandwidth_path, "tradeoff": tradeoff_path}


def section(
    datasets: dict,
    report_root: Path,
    document_dir: Path,
    summary_plot: Path,
    dataset_plots: dict[str, dict[str, Path]],
) -> str:
    summary_link = relative_link(summary_plot, document_dir)
    lines = [
        START,
        "## Bandwidth–accuracy benchmark",
        "",
        "Actual transmitted volume is modeled by in-memory JPEG encoding at quality 90. Computed Mbps is encoded bytes divided by video duration; it is not measured network throughput.",
        "",
        f"![BoT-SORT bandwidth and identity summary]({summary_link})",
        "",
        "**Figure:** Fair tracker-family comparison. HIVE–ECIA uses the least bandwidth on both datasets while giving the best QuadTrack IDSW; DTB remains a competitive zero-shot trade-off.",
        "",
    ]
    for label, payload in datasets.items():
        methods = payload["methods"]
        best = {
            "idsw": min(row["metrics"]["aggregate"]["id_switches"] for row in methods.values()),
            "idf1": max(row["metrics"]["aggregate"]["idf1"] for row in methods.values()),
            "hota": max(row["metrics"]["aggregate"]["hota"] for row in methods.values()),
            "assa": max(row["metrics"]["aggregate"]["assa"] for row in methods.values()),
            "mota": max(row["metrics"]["aggregate"]["mota"] for row in methods.values()),
            "mb": min(row["bandwidth"]["total_mb"] for row in methods.values()),
        }
        lines.extend([
            f"### {label}", "",
            "| Acquisition | Tracker | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | MOTA ↑ | Total MB ↓ | Mbps ↓ | HR requests | HR frames/crops | Saving vs Full-HR |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        full_bytes = payload["strategy_bandwidth"]["full_hr"]["total_bytes"]
        for row in methods.values():
            metric = row["metrics"]["aggregate"]
            bw = row["bandwidth"]
            saving = 1.0 - bw["total_bytes"] / full_bytes
            lines.append(
                f"| {row['display']} | {TRACKER_LABELS[row['tracker']]} | "
                f"{bold(metric['id_switches'], best['idsw'])} | "
                f"{bold(metric['idf1'], best['idf1'], '.4f')} | "
                f"{bold(metric['hota'], best['hota'], '.4f')} | "
                f"{bold(metric['assa'], best['assa'], '.4f')} | "
                f"{bold(metric['mota'], best['mota'], '.4f')} | "
                f"{bold(bw['total_mb'], best['mb'], '.2f')} | "
                f"{bw['megabits_per_second']:.2f} | {bw.get('hr_requests', 0)} | "
                f"{bw['hr_frames'] + bw['hr_crops']} | {saving:.2%} |"
            )
        dataset_key = payload["dataset_key"]
        bandwidth_plot = relative_link(dataset_plots[dataset_key]["bandwidth"], document_dir)
        accuracy_plot = relative_link(dataset_plots[dataset_key]["tradeoff"], document_dir)
        detailed_report = relative_link(
            report_root / dataset_key / "report" / "REPORT.md", document_dir
        )
        lines.extend([
            "",
            f"![{label} bandwidth]({bandwidth_plot})",
            "",
            f"![{label} accuracy versus bandwidth]({accuracy_plot})",
            "",
            f"Detailed report: [{dataset_key}]({detailed_report})",
            "",
        ])
    lines.extend([
        "### Interpretation boundary", "",
        "- Full-HR and Periodic-HR are system-level acquisition baselines, not matched-budget HIVE ablations.",
        "- Full/Periodic methods run real YOLO inference on the transmitted HR or LR frame. Tracker state stays in original panorama coordinates across resolution transitions.",
        "- HIVE–ECIA uses frozen Existing Base-Guided Safety and counts continuous LR JPEG frames plus every requested HR crop transmission. Repeated requests are counted again.",
        "- Ground truth is used only for offline MOT evaluation.",
        "",
        END,
    ])
    return "\n".join(lines)


def update_marked(path: Path, content: str) -> None:
    original = path.read_text(encoding="utf-8") if path.is_file() else ""
    if START in original and END in original:
        prefix = original.split(START, 1)[0].rstrip()
        suffix = original.split(END, 1)[1].lstrip()
        updated = prefix + "\n\n" + content + ("\n\n" + suffix if suffix else "\n")
    else:
        updated = original.rstrip() + "\n\n" + content + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")


def main() -> int:
    args = parse_args()
    datasets = {
        "QuadTrack held-out test (512×120)": json.loads(args.quadtrack.read_text()),
        "DTB zero-shot (640×320)": json.loads(args.dtb.read_text()),
    }
    report_root = args.output.resolve().parent
    summary_plot = (
        args.experimental_report.resolve().parent
        / "figures"
        / "bandwidth_accuracy_botsort_summary.png"
    )
    figure_dir = args.experimental_report.resolve().parent / "figures"
    plot_summary(datasets, summary_plot)
    dataset_plots = {}
    for payload in datasets.values():
        dataset_key = payload["dataset_key"]
        if dataset_key == "dtb":
            # Preserve the original neutral DTB zero-shot visualizations.
            plot_root = report_root / dataset_key / "report" / "plots"
            dataset_plots[dataset_key] = {
                "bandwidth": plot_root / "bandwidth_comparison.png",
                "tradeoff": plot_root / "accuracy_vs_bandwidth.png",
            }
        else:
            dataset_plots[dataset_key] = plot_dataset_details(payload, figure_dir)
    joint_content = section(
        datasets,
        report_root,
        args.output.resolve().parent,
        summary_plot,
        dataset_plots,
    )
    experimental_content = section(
        datasets,
        report_root,
        args.experimental_report.resolve().parent,
        summary_plot,
        dataset_plots,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# HIVE–ECIA bandwidth–accuracy report\n\n" + joint_content + "\n"
    )
    update_marked(args.experimental_report.resolve(), experimental_content)
    print(f"Bandwidth consolidated report: {args.output.resolve()}")
    print(f"Experimental report updated: {args.experimental_report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

