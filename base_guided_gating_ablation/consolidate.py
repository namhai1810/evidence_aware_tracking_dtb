"""Consolidate paired QuadTrack and DTB gating-ablation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def plot_gating_ablation(data: dict, output: Path) -> None:
    """Visualize the controlled gate-only comparison for a paper/report."""

    colors = ("#8795A7", "#E76F51")
    labels = ("G0\nGating OFF", "G1\nGating ON\n(Ours)")
    score_metrics = (("idf1", "IDF1"), ("hota", "HOTA"), ("assa", "AssA"))
    figure, axes = plt.subplots(2, 3, figsize=(14.8, 7.8))

    for row, (dataset, payload) in enumerate(data.items()):
        dataset_short = "QuadTrack held-out (512×120)" if row == 0 else "DTB zero-shot (640×320)"
        g0 = payload["methods"]["g0_gating_off"]["aggregate"]
        g1 = payload["methods"]["g1_gating_on"]["aggregate"]
        methods = (g0, g1)

        # Identity quality: pair G0/G1 within each metric so small changes remain visible.
        axis = axes[row, 0]
        x = np.arange(len(score_metrics))
        width = 0.34
        for method_index, (method, label, color) in enumerate(zip(methods, labels, colors)):
            values = [float(method[key]) for key, _ in score_metrics]
            offset = (-0.5 if method_index == 0 else 0.5) * width
            bars = axis.bar(x + offset, values, width, label=label.replace("\n", " "), color=color, alpha=0.92)
            if method_index == 1:
                for bar in bars:
                    bar.set_edgecolor("#9E321D")
                    bar.set_linewidth(1.8)
                    bar.set_hatch("//")
            for xpos, value in zip(x + offset, values):
                axis.text(
                    xpos,
                    value + 0.0022,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    fontweight="bold" if method_index == 1 else "normal",
                )
        axis.set_xticks(x, [label for _, label in score_metrics])
        axis.set_ylim((0.35, 0.48) if row == 0 else (0.22, 0.39))
        axis.text(0.0, 1.02, "Score ↑", transform=axis.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold")
        axis.set_title(f"{dataset_short} — Identity quality", loc="left", fontweight="bold", pad=20)

        # IDSW: the DTB panel exposes the harmful forced-HR merge.
        axis = axes[row, 1]
        idsw = [int(method["id_switches"]) for method in methods]
        bars = axis.bar(np.arange(2), idsw, color=colors, width=0.62)
        bars[1].set_edgecolor("#9E321D")
        bars[1].set_linewidth(1.8)
        bars[1].set_hatch("//")
        axis.bar_label(bars, padding=4, fontsize=10, fontweight="bold")
        axis.set_xticks(np.arange(2), labels)
        axis.get_xticklabels()[1].set_color("#B33A20")
        axis.get_xticklabels()[1].set_fontweight("bold")
        axis.set_ylim(0, max(idsw) * 1.32)
        axis.text(0.0, 1.02, "ID switches ↓", transform=axis.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold")
        axis.set_title(f"{dataset_short} — Identity errors", loc="left", fontweight="bold", pad=20)
        delta = idsw[0] - idsw[1]
        if delta:
            y = max(idsw) * 1.16
            axis.annotate("", xy=(1, y), xytext=(0, y), arrowprops={"arrowstyle": "->", "color": "#B33A20", "lw": 1.9})
            axis.text(0.5, y + max(idsw) * 0.025, f"−{delta} IDSW", ha="center", color="#B33A20", fontsize=9, fontweight="bold")
        else:
            axis.text(0.5, max(idsw) * 1.15, "Identical", ha="center", color="#4B5563", fontsize=9, fontweight="bold")

        # Physical HR requests: this is the direct quantity controlled by gating.
        axis = axes[row, 2]
        requests = [int(payload["bandwidth"][key]["hr_requests"]) for key in ("g0_gating_off", "g1_gating_on")]
        bars = axis.bar(np.arange(2), requests, color=colors, width=0.62)
        bars[1].set_edgecolor("#9E321D")
        bars[1].set_linewidth(1.8)
        bars[1].set_hatch("//")
        axis.bar_label(bars, padding=4, fontsize=10, fontweight="bold")
        axis.set_xticks(np.arange(2), labels)
        axis.get_xticklabels()[1].set_color("#B33A20")
        axis.get_xticklabels()[1].set_fontweight("bold")
        axis.set_ylim(0, max(requests) * 1.38)
        axis.text(0.0, 1.02, "HR requests ↓", transform=axis.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold")
        axis.set_title(f"{dataset_short} — HR acquisition", loc="left", fontweight="bold", pad=20)
        saving = (requests[0] - requests[1]) / requests[0]
        y = max(requests) * 1.19
        axis.annotate("", xy=(1, y), xytext=(0, y), arrowprops={"arrowstyle": "->", "color": "#B33A20", "lw": 1.9})
        axis.text(0.5, y + max(requests) * 0.03, f"−{saving:.1%}", ha="center", color="#B33A20", fontsize=9.5, fontweight="bold")

        for axis in axes[row]:
            axis.grid(axis="y", alpha=0.20)
            axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle("Controlled Base-Guided HR gating ablation", x=0.055, ha="left", fontsize=16, fontweight="bold")
    figure.text(0.055, 0.94, "Only the pre-query HR gate changes; candidate pools, raw BoT-SORT tracks, checkpoints and safety are fixed.", fontsize=9.5, color="#4B5563")
    figure.legend(
        handles=(
            Patch(facecolor=colors[0], label="G0 Gating OFF"),
            Patch(facecolor=colors[1], edgecolor="#9E321D", hatch="//", linewidth=1.8, label="G1 Gating ON (Ours)"),
        ),
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0.03, 0.03, 1, 0.83), h_pad=3.0, w_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


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
    figure = args.output.resolve().parent / "figures" / "gating_ablation_summary.png"
    plot_gating_ablation(data, figure)
    lines = [
        "# Controlled Base-Guided Gating Ablation", "",
        "G0 and G1 use the same Base-Guided candidate generator, frozen checkpoints, thresholds, pre-query safety and final identity safety. Only the Base-controlled HR request gate changes.", "",
        "![Controlled gating ablation](figures/gating_ablation_summary.png)", "",
        "| Dataset | Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | AssA ↑ | HR requests ↓ | HR MB ↓ | Total MB ↓ | Latency ms ↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, payload in data.items():
        for method, label in (("g0_gating_off", "G0 — Gating OFF"), ("g1_gating_on", "G1 — Gating ON (Ours)")):
            metric = payload["methods"][method]["aggregate"]
            bw = payload["bandwidth"][method]
            timing = payload["methods"][method]["timing"]
            lines.append(f"| {dataset} | {label} | {metric['id_switches']} | {metric['idf1']:.4f} | {metric['hota']:.4f} | {metric['assa']:.4f} | {bw['hr_requests']} | {bw['hr_bytes']/1e6:.3f} | {bw['total_mb']:.3f} | {timing['latency_ms']:.2f} |")
    lines += ["", "## Verification and savings", ""]
    for dataset, payload in data.items():
        check, saving = payload["verification"], payload["savings"]
        final_equal = all(check["final_mot_identical_by_sequence"].values())
        lines += [
            f"### {dataset}", "",
            f"- Required paired checks: **{'PASS' if check['required_checks_passed'] else 'FAIL'}**.",
            f"- Identical candidate pool: `{check['candidate_pools_identical']}`; equivalent raw BoT-SORT: `{check['raw_botsort_equivalent']}`.",
            f"- G1 reproduces frozen A3: `{check['g1_reference_reproduction']}`.",
            f"- Final MOT predictions identical: `{final_equal}`; per sequence: `{check['final_mot_identical_by_sequence']}`.",
            f"- HR requests saved: **{saving['hr_requests']} ({saving['hr_request_fraction']:.2%})**.",
            f"- Encoded HR payload saved: **{saving['hr_bytes']/1e6:.3f} MB ({saving['hr_byte_fraction']:.2%})**.",
            f"- Total transmitted payload saved: **{saving['total_bytes']/1e6:.3f} MB ({saving['total_byte_fraction']:.4%})**.", "",
        ]
    lines += [
        "## Interpretation", "",
        "HR acquisition cost counts on-demand HR queries/crops. Total transmission additionally includes every continuous LR frame, so its percentage reduction is expected to be much smaller.", "",
        "## Detailed reports", "",
        f"- QuadTrack: `{args.quadtrack.parent / 'REPORT.md'}`",
        f"- DTB: `{args.dtb.parent / 'REPORT.md'}`",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Consolidated gating report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

