#!/usr/bin/env python3
"""Generate publication-ready statistics and figures for the manual MOT dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "cvat_packages_trim300_last100"
    / "object_tracking_prepared_manual_dataset"
)
DEFAULT_REPORT_ASSETS = CODE_DIR / "outputs" / "report_assets"

SCENARIOS = ["normal_walking", "crossing", "seam", "partial_occ", "full_occ"]
DISPLAY_NAMES = {
    "normal_walking": "Normal walking",
    "crossing": "Crossing",
    "seam": "Seam interaction",
    "partial_occ": "Partial occlusion",
    "full_occ": "Full occlusion",
}
COLORS = {
    "normal_walking": "#0072B2",
    "crossing": "#D55E00",
    "seam": "#009E73",
    "partial_occ": "#CC79A7",
    "full_occ": "#E69F00",
}
SPLITS = {
    "normal_walking": "train",
    "crossing": "train",
    "partial_occ": "train",
    "seam": "val",
    "full_occ": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-assets", type=Path, default=DEFAULT_REPORT_ASSETS)
    return parser.parse_args()


def load_annotations(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for key in ("frame", "track_id", "class_id"):
                row[key] = int(row[key])
            for key in ("x", "y", "width", "height", "visibility"):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def load_seqinfo(dataset_root: Path, scenario: str) -> dict:
    path = dataset_root / "mot_challenge" / SPLITS[scenario] / scenario / "seqinfo.ini"
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return {
        "frames": int(values["seqLength"]),
        "width": int(values["imWidth"]),
        "height": int(values["imHeight"]),
        "fps": int(values["frameRate"]),
    }


def track_gaps(rows: list[dict]) -> list[dict]:
    by_track: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in rows:
        by_track[(row["scenario"], row["track_id"])].append(row["frame"])
    gaps = []
    for (scenario, track_id), frames in sorted(by_track.items()):
        unique = sorted(set(frames))
        for before, after in zip(unique, unique[1:]):
            if after > before + 1:
                gaps.append(
                    {
                        "scenario": scenario,
                        "track_id": track_id,
                        "last_visible_frame": before,
                        "next_visible_frame": after,
                        "missing_frames": after - before - 1,
                    }
                )
    return gaps


def overlap_iou(left: dict, right: dict) -> float:
    x1 = max(left["x"], right["x"])
    y1 = max(left["y"], right["y"])
    x2 = min(left["x"] + left["width"], right["x"] + right["width"])
    y2 = min(left["y"] + left["height"], right["y"] + right["height"])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left["width"] * left["height"] + right["width"] * right["height"] - intersection
    return intersection / union if union else 0.0


def compute_statistics(rows: list[dict], seqinfo: dict[str, dict]) -> tuple[list[dict], list[dict], dict]:
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario"]].append(row)
    gaps = track_gaps(rows)
    gap_by_scenario: dict[str, list[dict]] = defaultdict(list)
    for gap in gaps:
        gap_by_scenario[gap["scenario"]].append(gap)
    sequence_stats = []
    box_stats = []
    all_areas = []
    all_heights = []
    all_aspects = []
    total_overlap_frames_01 = 0
    total_overlap_frames_03 = 0
    for scenario in SCENARIOS:
        scenario_rows = by_scenario[scenario]
        info = seqinfo[scenario]
        frame_counts = Counter(row["frame"] for row in scenario_rows)
        areas = np.asarray(
            [row["width"] * row["height"] / (info["width"] * info["height"]) for row in scenario_rows]
        )
        heights = np.asarray([row["height"] / info["height"] for row in scenario_rows])
        aspects = np.asarray([row["width"] / row["height"] for row in scenario_rows])
        all_areas.extend(areas.tolist())
        all_heights.extend(heights.tolist())
        all_aspects.extend(aspects.tolist())
        by_frame: dict[int, list[dict]] = defaultdict(list)
        for row in scenario_rows:
            by_frame[row["frame"]].append(row)
        overlap_values = []
        for frame_rows in by_frame.values():
            if len(frame_rows) >= 2:
                overlap_values.append(overlap_iou(frame_rows[0], frame_rows[1]))
        overlap_01 = sum(value >= 0.10 for value in overlap_values)
        overlap_03 = sum(value >= 0.30 for value in overlap_values)
        total_overlap_frames_01 += overlap_01
        total_overlap_frames_03 += overlap_03
        scenario_gaps = gap_by_scenario[scenario]
        track_lengths = Counter(row["track_id"] for row in scenario_rows)
        sequence_stats.append(
            {
                "scenario": scenario,
                "display_name": DISPLAY_NAMES[scenario],
                "split": SPLITS[scenario],
                "frames": info["frames"],
                "boxes": len(scenario_rows),
                "tracks": len(track_lengths),
                "annotated_frames": len(frame_counts),
                "empty_frames": info["frames"] - len(frame_counts),
                "mean_persons_per_frame": len(scenario_rows) / info["frames"],
                "track_1_length": track_lengths.get(1, 0),
                "track_2_length": track_lengths.get(2, 0),
                "gaps": len(scenario_gaps),
                "long_gaps_ge_30": sum(gap["missing_frames"] >= 30 for gap in scenario_gaps),
                "max_gap": max((gap["missing_frames"] for gap in scenario_gaps), default=0),
                "overlap_frames_iou_ge_0_1": overlap_01,
                "overlap_frames_iou_ge_0_3": overlap_03,
                "median_box_area_percent": float(np.median(areas) * 100),
                "median_box_height_percent": float(np.median(heights) * 100),
                "median_aspect_ratio": float(np.median(aspects)),
            }
        )
        box_stats.append(
            {
                "scenario": scenario,
                "area": areas,
                "height": heights,
                "aspect": aspects,
                "center_x": np.asarray([(row["x"] + row["width"] / 2) / info["width"] for row in scenario_rows]),
                "center_y": np.asarray([(row["y"] + row["height"] / 2) / info["height"] for row in scenario_rows]),
                "frame_counts": frame_counts,
            }
        )
    global_stats = {
        "sequences": len(SCENARIOS),
        "frames": sum(row["frames"] for row in sequence_stats),
        "boxes": len(rows),
        "sequence_local_tracks": sum(row["tracks"] for row in sequence_stats),
        "annotated_frames": sum(row["annotated_frames"] for row in sequence_stats),
        "empty_frames": sum(row["empty_frames"] for row in sequence_stats),
        "gaps": len(gaps),
        "long_gaps_ge_30": sum(gap["missing_frames"] >= 30 for gap in gaps),
        "max_gap": max(gap["missing_frames"] for gap in gaps),
        "overlap_frames_iou_ge_0_1": total_overlap_frames_01,
        "overlap_frames_iou_ge_0_3": total_overlap_frames_03,
        "median_box_area_percent": float(np.median(all_areas) * 100),
        "box_area_q1_percent": float(np.quantile(all_areas, 0.25) * 100),
        "box_area_q3_percent": float(np.quantile(all_areas, 0.75) * 100),
        "median_box_height_percent": float(np.median(all_heights) * 100),
        "median_aspect_ratio": float(np.median(all_aspects)),
    }
    return sequence_stats, box_stats, {"global": global_stats, "gaps": gaps}


def style_axis(axis, title: str, panel: str) -> None:
    axis.set_title(f"{panel}  {title}", loc="left", fontsize=12, weight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)


def save_figure(fig, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_overview(
    output_base: Path,
    sequence_stats: list[dict],
    rows: list[dict],
    gaps: list[dict],
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.5))
    names = [DISPLAY_NAMES[row["scenario"]] for row in sequence_stats]
    x = np.arange(len(names))
    width = 0.36
    frames = [row["frames"] for row in sequence_stats]
    boxes = [row["boxes"] for row in sequence_stats]
    axes[0, 0].bar(x - width / 2, frames, width, label="Frames", color="#4C78A8")
    axes[0, 0].bar(x + width / 2, boxes, width, label="Tracked boxes", color="#59A14F")
    axes[0, 0].set_xticks(x, names, rotation=18, ha="right")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend(frameon=False, ncol=2)
    style_axis(axes[0, 0], "Sequence scale", "A")

    by_track: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in rows:
        by_track[(row["scenario"], row["track_id"])].append(row["frame"])
    y = 0
    ylabels = []
    for scenario in SCENARIOS:
        for track_id in sorted(key[1] for key in by_track if key[0] == scenario):
            values = sorted(by_track[(scenario, track_id)])
            axes[0, 1].scatter(
                values,
                [y] * len(values),
                s=2.2,
                color=COLORS[scenario],
                rasterized=True,
            )
            ylabels.append(f"{DISPLAY_NAMES[scenario]} · ID {track_id}")
            y += 1
    axes[0, 1].set_yticks(range(len(ylabels)), ylabels, fontsize=7.5)
    axes[0, 1].set_xlabel("Frame (1-based)")
    axes[0, 1].grid(axis="x", alpha=0.22, linewidth=0.7)
    axes[0, 1].spines[["top", "right"]].set_visible(False)
    axes[0, 1].set_title("B  Track presence and temporal gaps", loc="left", fontsize=12, weight="bold")

    bins = [(1, 1, "1"), (2, 5, "2–5"), (6, 15, "6–15"), (16, 30, "16–30"), (31, 90, "31–90"), (91, 10_000, ">90")]
    bottom = np.zeros(len(bins))
    for scenario in SCENARIOS:
        values = [
            sum(low <= gap["missing_frames"] <= high for gap in gaps if gap["scenario"] == scenario)
            for low, high, _ in bins
        ]
        axes[1, 0].bar(
            [label for _, _, label in bins],
            values,
            bottom=bottom,
            color=COLORS[scenario],
            label=DISPLAY_NAMES[scenario],
        )
        bottom += np.asarray(values)
    axes[1, 0].set_xlabel("Missing frames per gap")
    axes[1, 0].set_ylabel("Gap count")
    axes[1, 0].legend(frameon=False, fontsize=7.5, ncol=2)
    style_axis(axes[1, 0], "Occlusion/absence gap distribution", "C")

    split_order = ["train", "val", "test"]
    split_frames = [sum(row["frames"] for row in sequence_stats if row["split"] == split) for split in split_order]
    split_boxes = [sum(row["boxes"] for row in sequence_stats if row["split"] == split) for split in split_order]
    colors = ["#0072B2", "#009E73", "#D55E00"]
    y_pos = np.arange(2)
    frame_left = 0
    box_left = 0
    for split, frame_value, box_value, color in zip(split_order, split_frames, split_boxes, colors):
        axes[1, 1].barh(0, frame_value, left=frame_left, color=color, label=split)
        axes[1, 1].barh(1, box_value, left=box_left, color=color)
        if frame_value > 250:
            axes[1, 1].text(frame_left + frame_value / 2, 0, f"{frame_value:,}", ha="center", va="center", color="white", weight="bold")
        if box_value > 400:
            axes[1, 1].text(box_left + box_value / 2, 1, f"{box_value:,}", ha="center", va="center", color="white", weight="bold")
        frame_left += frame_value
        box_left += box_value
    axes[1, 1].set_yticks(y_pos, ["Frames", "Tracked boxes"])
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Count")
    axes[1, 1].legend(frameon=False, ncol=3, loc="lower right")
    style_axis(axes[1, 1], "Sequence-disjoint split composition", "D")
    fig.suptitle("Human-reviewed panoramic MOT dataset: scale, continuity, and split", fontsize=15, weight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, output_base)


def make_geometry(output_base: Path, box_stats: list[dict], sequence_stats: list[dict]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.5))
    names = [DISPLAY_NAMES[row["scenario"]] for row in box_stats]
    area_data = [row["area"] * 100 for row in box_stats]
    parts = axes[0, 0].violinplot(area_data, showmedians=True, showextrema=False)
    for body, row in zip(parts["bodies"], box_stats):
        body.set_facecolor(COLORS[row["scenario"]])
        body.set_alpha(0.8)
    parts["cmedians"].set_color("black")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(np.arange(1, len(names) + 1), names, rotation=18, ha="right")
    axes[0, 0].set_ylabel("Box area / image area (%) · log scale")
    style_axis(axes[0, 0], "Person-scale distribution", "A")

    height_data = [row["height"] * 100 for row in box_stats]
    boxplot = axes[0, 1].boxplot(height_data, patch_artist=True, showfliers=False)
    for patch, row in zip(boxplot["boxes"], box_stats):
        patch.set_facecolor(COLORS[row["scenario"]])
        patch.set_alpha(0.78)
    axes[0, 1].set_xticks(np.arange(1, len(names) + 1), names, rotation=18, ha="right")
    axes[0, 1].set_ylabel("Box height / image height (%)")
    style_axis(axes[0, 1], "Vertical scale variation", "B")

    center_x = np.concatenate([row["center_x"] for row in box_stats])
    center_y = np.concatenate([row["center_y"] for row in box_stats])
    hist = axes[1, 0].hist2d(center_x, center_y, bins=(48, 24), range=((0, 1), (0, 1)), cmap="magma")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Normalized panorama x")
    axes[1, 0].set_ylabel("Normalized panorama y")
    axes[1, 0].axvline(0.5, color="white", linestyle="--", linewidth=1, alpha=0.8)
    axes[1, 0].set_title("C  Spatial density of person centers", loc="left", fontsize=12, weight="bold")
    fig.colorbar(hist[3], ax=axes[1, 0], label="Box centers per bin")

    distributions = []
    for row, sequence in zip(box_stats, sequence_stats):
        counts = Counter(row["frame_counts"])
        distributions.append(
            [
                (sequence["frames"] - len(row["frame_counts"])) / sequence["frames"] * 100,
                sum(value == 1 for value in row["frame_counts"].values()) / sequence["frames"] * 100,
                sum(value >= 2 for value in row["frame_counts"].values()) / sequence["frames"] * 100,
            ]
        )
    distributions = np.asarray(distributions)
    bottom = np.zeros(len(names))
    for index, (label, color) in enumerate((("0 persons", "#BAB0AC"), ("1 person", "#4C78A8"), ("2 persons", "#F28E2B"))):
        axes[1, 1].bar(names, distributions[:, index], bottom=bottom, label=label, color=color)
        bottom += distributions[:, index]
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].set_ylabel("Frames (%)")
    axes[1, 1].tick_params(axis="x", rotation=18)
    axes[1, 1].legend(frameon=False, ncol=3)
    style_axis(axes[1, 1], "Per-frame person-count composition", "D")
    fig.suptitle("Panoramic MOT annotation geometry and scene composition", fontsize=15, weight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, output_base)


def frame_image(dataset_root: Path, scenario: str, frame: int) -> np.ndarray:
    return cv2.imread(
        str(dataset_root / "mot_challenge" / SPLITS[scenario] / scenario / "img1" / f"{frame:06d}.jpg")
    )


def track_color(track_id: int) -> tuple[int, int, int]:
    return (70, 230, 80) if track_id == 1 else (255, 170, 40)


def annotate_image(image: np.ndarray, scenario: str, frame: int, frame_rows: list[dict], title: str) -> np.ndarray:
    output = image.copy()
    for row in frame_rows:
        color = track_color(row["track_id"])
        x1, y1 = int(round(row["x"])), int(round(row["y"]))
        x2, y2 = int(round(row["x"] + row["width"])), int(round(row["y"] + row["height"]))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 4)
        label = f"ID {row['track_id']}"
        cv2.rectangle(output, (x1, max(0, y1 - 28)), (x1 + 70, y1), color, -1)
        cv2.putText(output, label, (x1 + 5, max(19, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (8, 8, 8), 2, cv2.LINE_AA)
    cv2.rectangle(output, (0, 0), (output.shape[1], 55), (12, 12, 12), -1)
    cv2.putText(output, title, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, f"{DISPLAY_NAMES[scenario]} | frame {frame}", (16, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 225, 240), 1, cv2.LINE_AA)
    return output


def choose_example_frames(rows: list[dict], seqinfo: dict[str, dict], gaps: list[dict]) -> list[tuple[str, list[int], list[str]]]:
    by_scenario_frame: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_scenario_frame[row["scenario"]][row["frame"]].append(row)
    crossing_overlap = max(
        (
            (overlap_iou(frame_rows[0], frame_rows[1]), frame)
            for frame, frame_rows in by_scenario_frame["crossing"].items()
            if len(frame_rows) >= 2
        ),
        default=(0, seqinfo["crossing"]["frames"] // 2),
    )[1]
    seam_frames = [
        frame
        for frame, frame_rows in by_scenario_frame["seam"].items()
        if any(row["x"] <= seqinfo["seam"]["width"] / 2 <= row["x"] + row["width"] for row in frame_rows)
    ]
    seam_center = seam_frames[len(seam_frames) // 2] if seam_frames else seqinfo["seam"]["frames"] // 2
    longest = {}
    for scenario in ("partial_occ", "full_occ"):
        candidates = [gap for gap in gaps if gap["scenario"] == scenario]
        longest[scenario] = max(candidates, key=lambda value: value["missing_frames"])
    return [
        (
            "crossing",
            [max(1, crossing_overlap - 12), crossing_overlap, min(seqinfo["crossing"]["frames"], crossing_overlap + 12)],
            ["Before interaction", "Maximum box overlap", "After interaction"],
        ),
        (
            "seam",
            [max(1, seam_center - 20), seam_center, min(seqinfo["seam"]["frames"], seam_center + 20)],
            ["Approaching seam", "Seam intersection", "Leaving seam"],
        ),
        (
            "partial_occ",
            [
                longest["partial_occ"]["last_visible_frame"],
                (longest["partial_occ"]["last_visible_frame"] + longest["partial_occ"]["next_visible_frame"]) // 2,
                longest["partial_occ"]["next_visible_frame"],
            ],
            [
                f"ID {longest['partial_occ']['track_id']} last visible",
                f"ID {longest['partial_occ']['track_id']} absent ({longest['partial_occ']['missing_frames']}-frame gap)",
                f"ID {longest['partial_occ']['track_id']} returns",
            ],
        ),
        (
            "full_occ",
            [
                longest["full_occ"]["last_visible_frame"],
                (longest["full_occ"]["last_visible_frame"] + longest["full_occ"]["next_visible_frame"]) // 2,
                longest["full_occ"]["next_visible_frame"],
            ],
            [
                f"ID {longest['full_occ']['track_id']} last visible",
                f"ID {longest['full_occ']['track_id']} absent ({longest['full_occ']['missing_frames']}-frame gap)",
                f"ID {longest['full_occ']['track_id']} returns",
            ],
        ),
    ]


def make_examples(
    output_path: Path,
    dataset_root: Path,
    rows: list[dict],
    seqinfo: dict[str, dict],
    gaps: list[dict],
) -> None:
    by_scenario_frame: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_scenario_frame[row["scenario"]][row["frame"]].append(row)
    panels = []
    for scenario, frames, titles in choose_example_frames(rows, seqinfo, gaps):
        row_panels = []
        for frame, title in zip(frames, titles):
            image = frame_image(dataset_root, scenario, frame)
            annotated = annotate_image(image, scenario, frame, by_scenario_frame[scenario].get(frame, []), title)
            scale = 580 / annotated.shape[1]
            resized = cv2.resize(
                annotated,
                (580, int(round(annotated.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
            canvas = np.full((330, 580, 3), 250, dtype=np.uint8)
            top = (310 - resized.shape[0]) // 2
            canvas[top : top + resized.shape[0]] = resized
            cv2.putText(
                canvas,
                f"{DISPLAY_NAMES[scenario]} - {title}",
                (12, 325),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            row_panels.append(canvas)
        panels.append(np.hstack(row_panels))
    sheet = np.vstack(panels)
    sheet = cv2.copyMakeBorder(sheet, 0, 14, 0, 0, cv2.BORDER_CONSTANT, value=(250, 250, 250))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_dataset_card(output_root: Path, sequence_stats: list[dict], summary: dict) -> None:
    global_stats = summary["global"]
    lines = [
        "# Paper statistics: human-reviewed panoramic MOT dataset",
        "",
        "## Aggregate",
        "",
        f"- Sequences: **{global_stats['sequences']}**",
        f"- Frames: **{global_stats['frames']:,}**",
        f"- Tracked boxes: **{global_stats['boxes']:,}**",
        f"- Sequence-local tracks: **{global_stats['sequence_local_tracks']}**",
        f"- Temporal gaps: **{global_stats['gaps']}**; gaps ≥30 frames: **{global_stats['long_gaps_ge_30']}**; maximum: **{global_stats['max_gap']}** frames.",
        f"- Median normalized box area: **{global_stats['median_box_area_percent']:.3f}%** (IQR {global_stats['box_area_q1_percent']:.3f}–{global_stats['box_area_q3_percent']:.3f}%).",
        f"- Median normalized box height: **{global_stats['median_box_height_percent']:.2f}%**.",
        f"- Frames with two-person IoU ≥0.10: **{global_stats['overlap_frames_iou_ge_0_1']}**; IoU ≥0.30: **{global_stats['overlap_frames_iou_ge_0_3']}**.",
        "",
        "## Per sequence",
        "",
        "| Scenario | Split | Frames | Boxes | Empty | Gaps | Max gap | Median area |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sequence_stats:
        lines.append(
            f"| {row['display_name']} | {row['split']} | {row['frames']:,} | {row['boxes']:,} | "
            f"{row['empty_frames']} | {row['gaps']} | {row['max_gap']} | {row['median_box_area_percent']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Generated artifacts",
            "",
            "- `manual_mot_dataset_overview_30102026.{png,pdf,svg}`: sequence scale, track presence, gap distribution, and split composition.",
            "- `manual_mot_geometry_30102026.{png,pdf,svg}`: person scale, height, spatial density, and per-frame person count.",
            "- `manual_mot_tracking_examples_30102026.jpg`: qualitative crossing, seam, partial-occlusion, and full-occlusion examples.",
            "- `sequence_statistics.csv`, `gap_statistics.csv`, and `paper_statistics.json`: machine-readable values used by the report.",
            "",
            "## Interpretation boundary",
            "",
            "Track IDs are verified within each sequence. Cross-sequence physical identity is not verified. The MOT visibility column is constant at 1.0 for visible boxes; temporal absence is represented by missing annotation rows. Therefore the dataset supports MOT and long-gap evaluation but not supervised visibility-score regression.",
            "",
        ]
    )
    (output_root / "PAPER_DATASET_CARD.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = dataset_root / "paper_assets"
    report_assets = args.report_assets.resolve()
    rows = load_annotations(dataset_root / "annotations.csv")
    seqinfo = {scenario: load_seqinfo(dataset_root, scenario) for scenario in SCENARIOS}
    sequence_stats, box_stats, summary = compute_statistics(rows, seqinfo)

    overview_base = output_root / "manual_mot_dataset_overview_30102026"
    geometry_base = output_root / "manual_mot_geometry_30102026"
    examples_path = output_root / "manual_mot_tracking_examples_30102026.jpg"
    make_overview(overview_base, sequence_stats, rows, summary["gaps"])
    make_geometry(geometry_base, box_stats, sequence_stats)
    make_examples(examples_path, dataset_root, rows, seqinfo, summary["gaps"])

    scalar_stats = []
    for row in sequence_stats:
        scalar_stats.append({key: value for key, value in row.items() if key != "display_name"})
    write_csv(output_root / "sequence_statistics.csv", scalar_stats)
    write_csv(output_root / "gap_statistics.csv", summary["gaps"])
    (output_root / "paper_statistics.json").write_text(
        json.dumps(
            {"global": summary["global"], "sequences": sequence_stats, "gaps": summary["gaps"]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_dataset_card(output_root, sequence_stats, summary)

    report_assets.mkdir(parents=True, exist_ok=True)
    for source in (
        overview_base.with_suffix(".png"),
        overview_base.with_suffix(".pdf"),
        overview_base.with_suffix(".svg"),
        geometry_base.with_suffix(".png"),
        geometry_base.with_suffix(".pdf"),
        geometry_base.with_suffix(".svg"),
        examples_path,
    ):
        shutil.copy2(source, report_assets / source.name)
    print(json.dumps(summary["global"], indent=2))
    print(f"Paper assets: {output_root}")
    print(f"Report assets: {report_assets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
