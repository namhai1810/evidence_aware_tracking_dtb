"""Consolidate QuadTrack and DTB ablation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .protocol import VARIANTS


def plot_ablation(data: dict, output: Path) -> None:
    """Create a paper-oriented component ablation summary."""

    variants = tuple(VARIANTS)
    labels = ("A0\nLR-only", "A1\nBase-only", "A2\nRule Full", "A3\nFinal")
    colors = ("#A8AFB9", "#79A98C", "#4E8FC1", "#E76F51")
    metrics = (("idf1", "IDF1"), ("hota", "HOTA"), ("assa", "AssA"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2))

    for row_index, (dataset, payload) in enumerate(data.items()):
        x = np.arange(len(variants))

        axis = axes[row_index, 0]
        width = 0.22
        for metric_index, (metric, label) in enumerate(metrics):
            values = [float(payload["methods"][variant]["aggregate"][metric]) for variant in variants]
            offset = (metric_index - 1) * width
            bars = axis.bar(x + offset, values, width=width, label=label, alpha=0.82)
            for variant_index, bar in enumerate(bars):
                bar.set_color(("#7891AE", "#71AFB2", "#9DBB91")[metric_index])
                if variant_index == 3:
                    bar.set_edgecolor("#A93620")
                    bar.set_linewidth(1.8)
                    bar.set_hatch("//")
            for variant_index, value in enumerate(values):
                axis.text(
                    x[variant_index] + offset, value + 0.0018, f"{value:.3f}",
                    ha="center", va="bottom", rotation=90, fontsize=6.2,
                    fontweight="bold" if variant_index == 3 else "normal",
                )
        axis.axvspan(2.55, 3.45, color="#FCE8E1", alpha=0.70, zorder=0)
        axis.set_ylim((0.34, 0.485) if row_index == 0 else (0.20, 0.39))
        axis.set_xticks(x, labels, fontsize=8)
        axis.get_xticklabels()[-1].set_color("#B33A20")
        axis.get_xticklabels()[-1].set_fontweight("bold")
        axis.set_ylabel(f"{dataset}\nScore ↑", fontweight="bold")
        axis.set_title("Identity quality", loc="left", fontweight="bold", pad=34)
        axis.legend(
            frameon=False,
            ncol=3,
            fontsize=8,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            borderaxespad=0.0,
        )
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)

        axis = axes[row_index, 1]
        idsw = [int(payload["methods"][variant]["aggregate"]["id_switches"]) for variant in variants]
        bars = axis.bar(x, idsw, color=colors, width=0.68)
        bars[-1].set_edgecolor("#8C2F1B")
        bars[-1].set_linewidth(2.0)
        bars[-1].set_hatch("//")
        axis.bar_label(bars, padding=3, fontsize=8.5)
        axis.set_ylim(0, max(idsw) * 1.32)
        axis.set_xticks(x, labels, fontsize=8)
        axis.get_xticklabels()[-1].set_color("#B33A20")
        axis.get_xticklabels()[-1].set_fontweight("bold")
        axis.set_ylabel("ID switches ↓")
        axis.set_title("Identity errors", loc="left", fontweight="bold")
        reduction = (idsw[0] - idsw[-1]) / idsw[0] if idsw[0] else 0.0
        arrow_y = max(idsw) * 1.16
        axis.annotate("", xy=(3, arrow_y), xytext=(0, arrow_y), arrowprops={"arrowstyle": "->", "color": "#B33A20", "lw": 1.9})
        axis.text(1.5, arrow_y + max(idsw) * 0.025, f"A0→A3: −{reduction:.1%}", ha="center", va="bottom", color="#B33A20", fontsize=8.5, fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)

        axis = axes[row_index, 2]
        requests = [int(payload["bandwidth"][variant]["hr_requests"]) for variant in variants]
        bars = axis.bar(x, requests, color=colors, width=0.68)
        bars[-1].set_edgecolor("#8C2F1B")
        bars[-1].set_linewidth(2.0)
        bars[-1].set_hatch("//")
        axis.bar_label(bars, padding=3, fontsize=8.5)
        upper = max(requests) * 1.32 if max(requests) else 1.0
        axis.set_ylim(0, upper)
        axis.set_xticks(x, labels, fontsize=8)
        axis.get_xticklabels()[-1].set_color("#B33A20")
        axis.get_xticklabels()[-1].set_fontweight("bold")
        axis.set_ylabel("Physical HR requests ↓")
        axis.set_title("HR acquisition cost", loc="left", fontweight="bold")
        if requests[2]:
            saving = (requests[2] - requests[3]) / requests[2]
            arrow_y = max(requests) * 1.16
            axis.annotate("", xy=(3, arrow_y), xytext=(2, arrow_y), arrowprops={"arrowstyle": "->", "color": "#B33A20", "lw": 1.9})
            axis.text(2.5, arrow_y + max(requests) * 0.025, f"A2→A3: −{saving:.1%}", ha="center", va="bottom", color="#B33A20", fontsize=8.5, fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Frozen HIVE–ECIA component ablation", x=0.055, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.055, 0.94, "A0: LR-only · A1: ECIA Base-only · A2: rule-trigger Full HR · A3: Base-Guided + Safety (final)", fontsize=9.5, color="#4B5563")
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.91), h_pad=4.0, w_pad=1.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadtrack", type=Path, required=True)
    parser.add_argument("--dtb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = {
        "QuadTrack held-out test (512×120)": json.loads(args.quadtrack.read_text(encoding="utf-8")),
        "DTB zero-shot (640×320)": json.loads(args.dtb.read_text(encoding="utf-8")),
    }
    figure = args.output.resolve().parent / "figures" / "ablation_summary.png"
    plot_ablation(data, figure)
    lines = [
        "# Frozen HIVE–ECIA Ablation Study", "",
        "A0/A3 reuse the frozen fair-benchmark artifacts. A1/A2 use the same YOLO11l, BoT-SORT, checkpoints and thresholds; GT is joined only after predictions are frozen.", "",
        "![Ablation summary](figures/ablation_summary.png)", "",
        "## Aggregate comparison", "",
        "| Dataset | Variant | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | MOTA ↑ | HR req. | HR MB | Total MB | FPS ↑ | Latency ms ↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, payload in data.items():
        for variant in VARIANTS:
            metric = payload["methods"][variant]["aggregate"]
            timing = payload["methods"][variant]["timing"]
            bw = payload["bandwidth"][variant]
            lines.append(
                f"| {dataset} | {variant.upper()} | {metric['id_switches']} | {metric['idf1']:.4f} | {metric['hota']:.4f} | {metric['assa']:.4f} | {metric['mota']:.4f} | "
                f"{bw['hr_requests']} | {bw['hr_bytes']/1_000_000:.3f} | {bw['total_mb']:.3f} | {timing['fps']:.2f} | {timing['latency_ms']:.2f} |"
            )
    lines += ["", "## Component contribution", ""]
    for dataset, payload in data.items():
        lines += [f"### {dataset}", "", "| Transition | ΔIDSW | ΔIDF1 | ΔHOTA | ΔAssA | ΔHR requests |", "|---|---:|---:|---:|---:|---:|"]
        for left, right, label in (("a0", "a1", "Base association"), ("a1", "a2", "HIVE + HR Full (also changes trigger)"), ("a2", "a3", "Base-guided gating + Safety")):
            lm, rm = payload["methods"][left]["aggregate"], payload["methods"][right]["aggregate"]
            lb, rb = payload["bandwidth"][left], payload["bandwidth"][right]
            lines.append(f"| {label} ({left.upper()}→{right.upper()}) | {rm['id_switches']-lm['id_switches']:+d} | {rm['idf1']-lm['idf1']:+.4f} | {rm['hota']-lm['hota']:+.4f} | {rm['assa']-lm['assa']:+.4f} | {rb['hr_requests']-lb['hr_requests']:+d} |")
        check = payload["verification"]
        lines += ["", f"A2/A3 same final predictions: `{all(check['a2_a3_byte_identical_by_sequence'].values())}`; A3 fewer HR: `{check['a3_uses_fewer_hr_requests']}`.", ""]
    lines += [
        "## Interpretation boundary", "",
        "- A0→A1 isolates LR ECIA Base decisions with HR physically disabled.",
        "- A1→A2 does not isolate HR alone: A1 uses Base-guided candidates while A2 is the preserved original rule-trigger Full pipeline. This necessary existing-pipeline difference is reported explicitly.",
        "- A2→A3 measures the final Base-guided + Safety policy against the original Full pipeline; equality of predictions is checked, not assumed.",
        "- Encoded bandwidth uses JPEG quality 90 on every LR frame and each actually retrieved HR crop. It is encoded payload size, not measured network throughput.",
        "- FPS/latency exclude decode, resize, JPEG accounting and file writes; A0 timing comes from the shared-cache benchmark while online variants include their recovery controller.",
        "", "## Variant definitions", "",
    ]
    lines.extend(f"- **{key.upper()}**: {value}." for key, value in VARIANTS.items())
    lines += ["", "## Detailed reports", "", f"- QuadTrack: `{args.quadtrack.parent / 'REPORT.md'}`", f"- DTB: `{args.dtb.parent / 'REPORT.md'}`"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Consolidated ablation report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

