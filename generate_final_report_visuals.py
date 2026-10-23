#!/usr/bin/env python3
"""Generate the all-sequence tracking and retrieval figures used by final_report.md."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


CODE_DIR = Path(__file__).resolve().parent
OUTPUTS = CODE_DIR / "outputs"
ASSETS = OUTPUTS / "report_assets"


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def rounded_panel(ax, color: str = "#FFFFFF") -> None:
    ax.set_facecolor(color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color="#DFE4EE", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def annotate_bars(ax, bars, fmt="{:.3f}", dy=0.015, fontsize=8) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + dy,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#273142",
            fontweight="semibold",
        )


def dashboard() -> None:
    run_paths = {
        "512 / ByteTrack": OUTPUTS / "all_sequences_512x120" / "metrics_iou_0.5.json",
        "512 / ReID": OUTPUTS / "all_sequences_reid_512x120" / "metrics_iou_0.5.json",
        "2048 / ByteTrack": OUTPUTS / "all_sequences_original_2048x480" / "metrics_iou_0.5.json",
        "2048 / ReID": OUTPUTS / "all_sequences_reid_original_2048x480" / "metrics_iou_0.5.json",
    }
    runs = {name: load(path)["aggregate"] for name, path in run_paths.items()}
    summary = load(OUTPUTS / "highres_retrieval_reid_embedding" / "summary.json")
    events = load(OUTPUTS / "highres_retrieval_reid_embedding" / "switch_events_top5.json")
    candidate_rows = [candidate for event in events for candidate in event["top_frames"]]
    positives = sum(candidate["correct_gt_identity"] for candidate in candidate_rows)
    negatives = len(candidate_rows) - positives
    ranked_events = sum(bool(event["top_frames"]) for event in events)

    colors = ["#356AE6", "#24A887", "#F49D37", "#805AD5"]
    fig = plt.figure(figsize=(16, 10), facecolor="#F3F6FB")
    grid = fig.add_gridspec(2, 2, left=0.09, right=0.97, top=0.86, bottom=0.08, hspace=0.36, wspace=0.22)
    fig.suptitle(
        "QuadTrack all-sequence tracking and HR retrieval overview",
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=23,
        fontweight="bold",
        color="#172033",
    )
    fig.text(
        0.055,
        0.915,
        "17 sequences (0015–0031)  •  YOLO11l  •  person class  •  evaluation IoU ≥ 0.50",
        fontsize=11.5,
        color="#667085",
    )

    ax = fig.add_subplot(grid[0, 0])
    rounded_panel(ax)
    metrics = ["precision", "recall", "f1", "mota", "idf1"]
    labels = ["Precision", "Recall", "F1", "MOTA", "IDF1"]
    x = np.arange(len(metrics))
    width = 0.19
    for index, (name, values) in enumerate(runs.items()):
        bars = ax.bar(
            x + (index - 1.5) * width,
            [values[key] for key in metrics],
            width,
            color=colors[index],
            label=name,
            edgecolor="none",
        )
        annotate_bars(ax, bars, fontsize=7, dy=0.012)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="both", colors="#475467", labelsize=9)
    ax.set_title("A. End-to-end tracking quality", loc="left", fontsize=13, fontweight="bold", color="#172033")
    ax.legend(frameon=False, ncols=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13))

    ax = fig.add_subplot(grid[0, 1])
    rounded_panel(ax)
    names = list(runs)
    idsw = [runs[name]["id_switches"] for name in names]
    bars = ax.bar(np.arange(4), idsw, color=colors, width=0.64)
    for bar, value in zip(bars, idsw):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 45,
            f"{value:,}",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#273142",
        )
    ax.set_ylim(0, max(idsw) * 1.17)
    ax.set_xticks(np.arange(4), ["512\nByteTrack", "512\nReID", "2048\nByteTrack", "2048\nReID"])
    ax.tick_params(axis="both", colors="#475467", labelsize=9)
    ax.set_title("B. Identity switches (lower is better)", loc="left", fontsize=13, fontweight="bold", color="#172033")
    ax.text(
        0.01,
        -0.23,
        "Read with recall: 512 misses more people, so fewer switches can be observed.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#667085",
    )

    ax = fig.add_subplot(grid[1, 0])
    rounded_panel(ax)
    funnel_labels = [
        "All LR switches",
        "Any ranked HR candidate",
        "Correct HR available",
        "Positive in top-5",
        "Correct top-1",
    ]
    funnel_values = [
        summary["total_lr_switch_events"],
        ranked_events,
        summary["events_with_correct_identity_available"],
        summary["top5_improvable_events"],
        summary["top1_improvable_events"],
    ]
    y = np.arange(len(funnel_labels))[::-1]
    funnel_colors = ["#23395D", "#356AE6", "#50A7D9", "#24A887", "#17836A"]
    bars = ax.barh(y, funnel_values, color=funnel_colors, height=0.62)
    total = funnel_values[0]
    for bar, value in zip(bars, funnel_values):
        ax.text(
            value + 17,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}  ({value / total:.1%})",
            va="center",
            fontsize=9,
            fontweight="semibold",
            color="#273142",
        )
    ax.set_xlim(0, total * 1.23)
    ax.set_yticks(y, funnel_labels)
    ax.tick_params(axis="both", colors="#475467", labelsize=9)
    ax.set_title("C. Backward HR retrieval funnel", loc="left", fontsize=13, fontweight="bold", color="#172033")
    ax.set_xlabel("Events", fontsize=9, color="#667085")

    ax = fig.add_subplot(grid[1, 1])
    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_visible(False)
    wedges, _ = ax.pie(
        [positives, negatives],
        colors=["#24A887", "#E05A6F"],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.36, "edgecolor": "#FFFFFF", "linewidth": 3},
    )
    ax.text(0, 0.08, f"{len(candidate_rows):,}", ha="center", va="center", fontsize=23, fontweight="bold", color="#172033")
    ax.text(0, -0.15, "ranked pairs", ha="center", va="center", fontsize=10, color="#667085")
    ax.text(-1.38, -0.88, f"Positive\n{positives:,} ({positives / len(candidate_rows):.1%})", color="#17836A", fontsize=10, fontweight="bold")
    ax.text(0.70, -0.88, f"Negative\n{negatives:,} ({negatives / len(candidate_rows):.1%})", color="#B73E52", fontsize=10, fontweight="bold")
    ax.set_title("D. Saved top-5 ranking samples", loc="left", fontsize=13, fontweight="bold", color="#172033")
    ax.text(
        0.5,
        -0.12,
        f"{summary['top1_improvable_events']} correct top-1 events  •  {summary['top5_improvable_events']} events with a positive in top-5",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
        color="#667085",
    )

    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / "tracking_retrieval_dashboard.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def workflow() -> None:
    summary = load(OUTPUTS / "highres_retrieval_reid_embedding" / "summary.json")
    events = load(OUTPUTS / "highres_retrieval_reid_embedding" / "switch_events_top5.json")
    candidate_rows = [candidate for event in events for candidate in event["top_frames"]]
    positives = sum(candidate["correct_gt_identity"] for candidate in candidate_rows)
    negatives = len(candidate_rows) - positives
    max_frames = summary["retrieval_frames"][1]

    fig, ax = plt.subplots(figsize=(16, 5.2), facecolor="#F3F6FB")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.suptitle(
        "How the cross-resolution ReID ranking dataset is built",
        x=0.06,
        y=0.95,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color="#172033",
    )
    ax.text(
        0.3,
        5.22,
        "GT/Hungarian identifies evaluation switch events and labels outcomes; it never supplies the ranking score.",
        fontsize=10.5,
        color="#B54708",
        fontweight="semibold",
    )

    boxes = [
        (0.35, 2.05, 2.25, 2.15, "1  LR switch event", f"{summary['total_lr_switch_events']:,} events\nold ID → new ID", "#EAF0FF", "#356AE6"),
        (3.05, 2.05, 2.55, 2.15, "2  Build LR query", "≤5 old-track crops\n+ target crop\nphysical 512×120", "#E9F8F4", "#24A887"),
        (6.05, 2.05, 2.35, 2.15, "3  Extract embeddings", "YOLO26n-ReID\n512-D, L2 normalized", "#F3EDFF", "#805AD5"),
        (8.85, 2.05, 2.55, 2.15, "4  Search HR history", f"all tracked people\nT−{max_frames} … T−1\nnearest → oldest", "#FFF4E5", "#F49D37"),
        (11.85, 2.05, 1.85, 2.15, "5  Score", "min(cos old,\ncos target)", "#EAF4FC", "#2E90C7"),
        (14.15, 2.05, 1.5, 2.15, "6  Rank", "top-1 / top-5\nGT label after", "#E9F8F4", "#17836A"),
    ]
    for x, y, width, height, title, body, fill, border in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=1.7,
            edgecolor=border,
            facecolor=fill,
        )
        ax.add_patch(patch)
        ax.text(x + 0.16, y + height - 0.42, title, fontsize=10.5, fontweight="bold", color="#172033", va="top")
        ax.text(x + 0.16, y + height - 0.91, body, fontsize=9.5, color="#475467", va="top", linespacing=1.35)
    for first, second in zip(boxes, boxes[1:]):
        start = (first[0] + first[2] + 0.06, first[1] + first[3] / 2)
        end = (second[0] - 0.06, second[1] + second[3] / 2)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.5, color="#98A2B3"))

    footer = [
        (0.45, 3.15, f"{len(candidate_rows):,} ranked pairs", "#EAF0FF", "#356AE6"),
        (3.85, 2.65, f"{positives:,} positive ({positives / len(candidate_rows):.1%})", "#E9F8F4", "#17836A"),
        (6.75, 2.85, f"{negatives:,} negative ({negatives / len(candidate_rows):.1%})", "#FDECEF", "#B73E52"),
        (9.85, 5.75, f"{summary['top1_improvable_events']:,} correct top-1  •  {summary['top5_improvable_events']:,} positive in top-5", "#F3EDFF", "#6840B8"),
    ]
    for x, width, label, fill, border in footer:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.82),
                width,
                0.58,
                boxstyle="round,pad=0.04,rounding_size=0.1",
                linewidth=1.2,
                edgecolor=border,
                facecolor=fill,
            )
        )
        ax.text(x + width / 2, 1.11, label, ha="center", va="center", fontsize=9.4, color=border, fontweight="bold")
    ax.text(
        0.45,
        0.27,
        "Not yet implemented: online uncertainty trigger + identity merge + post-merge IDF1/MOTA evaluation.",
        fontsize=10,
        color="#B54708",
        fontweight="semibold",
    )
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / "reid_ranking_pipeline.png", dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    dashboard()
    workflow()
    print(f"Wrote figures to {ASSETS}")


if __name__ == "__main__":
    main()
