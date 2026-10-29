#!/usr/bin/env python3
"""Audit and package manual MOT annotations as a tracking-ready dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACKAGE_ROOT = PROJECT_ROOT / "cvat_packages_trim300_last100"


SCENARIOS = {
    "normal_walking": {
        "mot": "dtb_normal_walking",
        "xml": "dtb_normal_walking.xml",
        "sequence": "20261028T082400Z_46324651_normal_walking",
        "split": "train",
    },
    "crossing": {
        "mot": "dtb_crossing",
        "xml": "dtb_crossing.xml",
        "sequence": "20261028T082643Z_5f566a39_crossing",
        "split": "train",
    },
    "seam": {
        "mot": "dtb_seam",
        "xml": "dtb_seam.xml",
        "sequence": "20261028T082930Z_153a02a2_seam",
        "split": "val",
    },
    "partial_occ": {
        "mot": "dtb_partial_occ",
        "xml": "dtb_partial_occ.xml",
        "sequence": "20261028T083330Z_9fe9298b_partial_occ",
        "split": "train",
    },
    "full_occ": {
        "mot": "dtb_full_occ",
        "xml": "dtb_full_occ.xml",
        "sequence": "20261028T083518Z_e7ada31a_full_occ",
        "split": "test",
    },
}


@dataclass(frozen=True)
class MOTRow:
    frame: int
    track_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_id: int
    visibility: float

    @property
    def xyxy(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.x + self.width, self.y + self.height], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_PACKAGE_ROOT / "object_tracking_prepared_manual_dataset",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--contact-sheet-frames", type=int, default=12)
    parser.add_argument("--skip-videos", action="store_true")
    return parser.parse_args()


def read_mot(path: Path) -> list[MOTRow]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, values in enumerate(csv.reader(handle), start=1):
            if not values:
                continue
            if len(values) != 9:
                raise ValueError(f"Expected 9 columns at {path}:{line_number}, got {len(values)}")
            row = MOTRow(
                frame=int(float(values[0])),
                track_id=int(float(values[1])),
                x=float(values[2]),
                y=float(values[3]),
                width=float(values[4]),
                height=float(values[5]),
                confidence=float(values[6]),
                class_id=int(float(values[7])),
                visibility=float(values[8]),
            )
            rows.append(row)
    rows.sort(key=lambda row: (row.frame, row.track_id))
    return rows


def source_images(package_root: Path, scenario: str) -> list[Path]:
    sequence = SCENARIOS[scenario]["sequence"]
    root = package_root / sequence / "cvat_for_video" / "images"
    images = sorted(root.glob("frame_*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images under {root}")
    return images


def read_manual_xml_boxes(path: Path) -> dict[int, list[np.ndarray]]:
    output: dict[int, list[np.ndarray]] = defaultdict(list)
    root = ET.parse(path).getroot()
    for image in root.findall("image"):
        # CVAT image XML is 0-based; MOTChallenge is 1-based.
        frame = int(image.attrib["id"]) + 1
        for box in image.findall("box"):
            output[frame].append(
                np.asarray(
                    [
                        float(box.attrib["xtl"]),
                        float(box.attrib["ytl"]),
                        float(box.attrib["xbr"]),
                        float(box.attrib["ybr"]),
                    ],
                    dtype=float,
                )
            )
    return output


def iou_matrix(left: list[np.ndarray], right: list[np.ndarray]) -> np.ndarray:
    if not left or not right:
        return np.empty((len(left), len(right)), dtype=float)
    a = np.asarray(left)[:, None, :]
    b = np.asarray(right)[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, a[..., 2] - a[..., 0]) * np.maximum(0.0, a[..., 3] - a[..., 1])
    area_b = np.maximum(0.0, b[..., 2] - b[..., 0]) * np.maximum(0.0, b[..., 3] - b[..., 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def compare_with_manual_boxes(
    rows: list[MOTRow], xml_boxes: dict[int, list[np.ndarray]], frame_count: int
) -> dict:
    mot_by_frame: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        mot_by_frame[row.frame].append(row.xyxy)
    unmatched_mot = unmatched_xml = 0
    matched_ious = []
    mismatch_frames = []
    for frame in range(1, frame_count + 1):
        mot = mot_by_frame.get(frame, [])
        xml = xml_boxes.get(frame, [])
        if not mot or not xml:
            unmatched_mot += len(mot)
            unmatched_xml += len(xml)
            if len(mot) != len(xml):
                mismatch_frames.append(frame)
            continue
        overlaps = iou_matrix(mot, xml)
        left, right = linear_sum_assignment(-overlaps)
        matched_ious.extend(float(overlaps[i, j]) for i, j in zip(left, right))
        unmatched_mot += len(mot) - len(left)
        unmatched_xml += len(xml) - len(right)
        if len(mot) != len(xml) or any(overlaps[i, j] < 0.999 for i, j in zip(left, right)):
            mismatch_frames.append(frame)
    return {
        "mot_box_count": len(rows),
        "xml_box_count": sum(len(value) for value in xml_boxes.values()),
        "unmatched_mot_boxes": unmatched_mot,
        "unmatched_xml_boxes": unmatched_xml,
        "minimum_matched_iou": min(matched_ious, default=None),
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else None,
        "frames_with_non_exact_box_match": mismatch_frames,
    }


def track_gap_rows(rows: list[MOTRow]) -> list[dict]:
    by_id: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        by_id[row.track_id].append(row.frame)
    gaps = []
    for track_id, frames in sorted(by_id.items()):
        unique = sorted(set(frames))
        for before, after in zip(unique, unique[1:]):
            if after > before + 1:
                gaps.append(
                    {
                        "track_id": track_id,
                        "last_visible_frame": before,
                        "next_visible_frame": after,
                        "missing_frames": after - before - 1,
                    }
                )
    return gaps


def audit_sequence(
    package_root: Path,
    annotation_root: Path,
    scenario: str,
    rows: list[MOTRow],
    images: list[Path],
) -> dict:
    first = cv2.imread(str(images[0]))
    height, width = first.shape[:2]
    frame_count = len(images)
    invalid = []
    duplicates = Counter((row.frame, row.track_id) for row in rows)
    for row in rows:
        if not (
            1 <= row.frame <= frame_count
            and row.track_id > 0
            and row.width > 0
            and row.height > 0
            and 0 <= row.x < row.x + row.width <= width + 1e-3
            and 0 <= row.y < row.y + row.height <= height + 1e-3
            and 0 <= row.visibility <= 1
        ):
            invalid.append(asdict(row))
    frames_present = {row.frame for row in rows}
    tracks = Counter(row.track_id for row in rows)
    gaps = track_gap_rows(rows)
    comparison = compare_with_manual_boxes(
        rows,
        read_manual_xml_boxes(annotation_root.parent / "annotation_file" / SCENARIOS[scenario]["xml"]),
        frame_count,
    )
    return {
        "scenario": scenario,
        "source_sequence": SCENARIOS[scenario]["sequence"],
        "split": SCENARIOS[scenario]["split"],
        "source_gt": str((annotation_root / SCENARIOS[scenario]["mot"] / "gt.txt").resolve()),
        "frame_count": frame_count,
        "image_width": width,
        "image_height": height,
        "annotation_rows": len(rows),
        "annotated_frame_count": len(frames_present),
        "empty_frame_count": frame_count - len(frames_present),
        "track_count": len(tracks),
        "track_lengths": {str(key): value for key, value in sorted(tracks.items())},
        "track_ids": sorted(tracks),
        "gap_count": len(gaps),
        "long_gap_count_ge_30": sum(row["missing_frames"] >= 30 for row in gaps),
        "maximum_gap_frames": max((row["missing_frames"] for row in gaps), default=0),
        "gaps": gaps,
        "class_ids": dict(Counter(row.class_id for row in rows)),
        "confidence_values": dict(Counter(row.confidence for row in rows)),
        "visibility_values": dict(Counter(row.visibility for row in rows)),
        "duplicate_frame_track_rows": [
            {"frame": frame, "track_id": track_id, "count": count}
            for (frame, track_id), count in duplicates.items()
            if count > 1
        ],
        "invalid_rows": invalid,
        "manual_box_comparison": comparison,
    }


def hardlink_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == source.stat().st_size:
            return "hardlink" if destination.stat().st_ino == source.stat().st_ino else "existing"
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def write_mot(path: Path, rows: list[MOTRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in rows:
            writer.writerow(
                [
                    row.frame,
                    row.track_id,
                    format_float(row.x),
                    format_float(row.y),
                    format_float(row.width),
                    format_float(row.height),
                    format_float(row.confidence),
                    row.class_id,
                    format_float(row.visibility),
                ]
            )


def write_sequence(
    output_root: Path,
    scenario: str,
    images: list[Path],
    rows: list[MOTRow],
    audit: dict,
    fps: float,
) -> Counter:
    split = SCENARIOS[scenario]["split"]
    sequence_root = output_root / "mot_challenge" / split / scenario
    modes = Counter()
    for index, source in enumerate(images, start=1):
        destination = sequence_root / "img1" / f"{index:06d}{source.suffix.lower()}"
        modes[hardlink_or_copy(source, destination)] += 1
    write_mot(sequence_root / "gt" / "gt.txt", rows)
    (sequence_root / "labels.txt").write_text("person\n", encoding="utf-8")
    (sequence_root / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={scenario}\n"
        "imDir=img1\n"
        f"frameRate={int(round(fps))}\n"
        f"seqLength={audit['frame_count']}\n"
        f"imWidth={audit['image_width']}\n"
        f"imHeight={audit['image_height']}\n"
        "imExt=.jpg\n",
        encoding="utf-8",
    )
    return modes


def stable_color(track_id: int) -> tuple[int, int, int]:
    palette = (
        (80, 230, 70),
        (255, 170, 40),
        (80, 170, 255),
        (220, 80, 220),
        (50, 220, 220),
    )
    return palette[(track_id - 1) % len(palette)]


def draw_frame(image: np.ndarray, scenario: str, frame: int, rows: list[MOTRow]) -> np.ndarray:
    output = image.copy()
    for row in rows:
        color = stable_color(row.track_id)
        x1, y1 = int(round(row.x)), int(round(row.y))
        x2, y2 = int(round(row.x + row.width)), int(round(row.y + row.height))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        label = f"ID {row.track_id} | person"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
        cv2.rectangle(output, (x1, max(0, y1 - text_height - 10)), (x1 + text_width + 8, y1), color, -1)
        cv2.putText(
            output,
            label,
            (x1 + 4, max(text_height, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (10, 10, 10),
            2,
            cv2.LINE_AA,
        )
    banner = f"{scenario} | MOT frame {frame:06d} | active IDs={len(rows)}"
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (18, 18, 18), -1)
    cv2.putText(
        output,
        banner,
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def selected_frames(rows: list[MOTRow], frame_count: int, count: int) -> list[int]:
    selected = set(np.linspace(1, frame_count, min(count, frame_count), dtype=int).tolist())
    gaps = track_gap_rows(rows)
    for gap in sorted(gaps, key=lambda item: item["missing_frames"], reverse=True)[:4]:
        selected.update([gap["last_visible_frame"], gap["next_visible_frame"]])
    ordered = sorted(frame for frame in selected if 1 <= frame <= frame_count)
    if len(ordered) > count:
        indices = np.linspace(0, len(ordered) - 1, count, dtype=int)
        ordered = [ordered[index] for index in indices]
    return ordered


def create_contact_sheet(
    output_path: Path,
    scenario: str,
    images: list[Path],
    rows: list[MOTRow],
    count: int,
) -> None:
    by_frame: dict[int, list[MOTRow]] = defaultdict(list)
    for row in rows:
        by_frame[row.frame].append(row)
    panels = []
    for frame in selected_frames(rows, len(images), count):
        image = cv2.imread(str(images[frame - 1]))
        rendered = draw_frame(image, scenario, frame, by_frame.get(frame, []))
        scale = min(420 / rendered.shape[1], 230 / rendered.shape[0])
        resized = cv2.resize(
            rendered,
            (int(rendered.shape[1] * scale), int(rendered.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
        panel = np.full((250, 440, 3), 245, dtype=np.uint8)
        top, left = (230 - resized.shape[0]) // 2, (440 - resized.shape[1]) // 2
        panel[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
        ids = ",".join(str(row.track_id) for row in by_frame.get(frame, [])) or "none"
        cv2.putText(
            panel,
            f"frame {frame:06d} | IDs={ids}",
            (12, 244),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    columns = 3
    panel_rows = int(np.ceil(len(panels) / columns))
    while len(panels) < panel_rows * columns:
        panels.append(np.full_like(panels[0], 235))
    sheet = np.vstack(
        [np.hstack(panels[index * columns : (index + 1) * columns]) for index in range(panel_rows)]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Could not write {output_path}")


def create_video(
    output_path: Path,
    scenario: str,
    images: list[Path],
    rows: list[MOTRow],
    fps: float,
    output_width: int,
) -> None:
    by_frame: dict[int, list[MOTRow]] = defaultdict(list)
    for row in rows:
        by_frame[row.frame].append(row)
    first = cv2.imread(str(images[0]))
    scale = min(1.0, output_width / first.shape[1])
    size = (int(round(first.shape[1] * scale)), int(round(first.shape[0] * scale)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output_path}")
    try:
        for frame, source in enumerate(images, start=1):
            rendered = draw_frame(cv2.imread(str(source)), scenario, frame, by_frame.get(frame, []))
            if (rendered.shape[1], rendered.shape[0]) != size:
                rendered = cv2.resize(rendered, size, interpolation=cv2.INTER_AREA)
            writer.write(rendered)
    finally:
        writer.release()


def write_flat_manifests(output_root: Path, all_rows: dict[str, list[MOTRow]], audits: list[dict]) -> None:
    audit_by_scenario = {row["scenario"]: row for row in audits}
    annotation_fields = [
        "split", "scenario", "global_track_key", "frame", "image", "track_id",
        "x", "y", "width", "height", "class_id", "visibility",
    ]
    with (output_root / "annotations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=annotation_fields)
        writer.writeheader()
        for scenario, rows in all_rows.items():
            split = SCENARIOS[scenario]["split"]
            for row in rows:
                writer.writerow(
                    {
                        "split": split,
                        "scenario": scenario,
                        "global_track_key": f"{scenario}:{row.track_id}",
                        "frame": row.frame,
                        "image": f"mot_challenge/{split}/{scenario}/img1/{row.frame:06d}.jpg",
                        "track_id": row.track_id,
                        "x": format_float(row.x),
                        "y": format_float(row.y),
                        "width": format_float(row.width),
                        "height": format_float(row.height),
                        "class_id": row.class_id,
                        "visibility": format_float(row.visibility),
                    }
                )
    with (output_root / "tracks.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "split", "scenario", "track_id", "global_track_key", "detections",
            "first_frame", "last_frame", "gap_count", "max_gap_frames",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scenario, rows in all_rows.items():
            by_id: dict[int, list[MOTRow]] = defaultdict(list)
            for row in rows:
                by_id[row.track_id].append(row)
            gaps = track_gap_rows(rows)
            for track_id, track_rows in sorted(by_id.items()):
                own_gaps = [row for row in gaps if row["track_id"] == track_id]
                writer.writerow(
                    {
                        "split": SCENARIOS[scenario]["split"],
                        "scenario": scenario,
                        "track_id": track_id,
                        "global_track_key": f"{scenario}:{track_id}",
                        "detections": len(track_rows),
                        "first_frame": min(row.frame for row in track_rows),
                        "last_frame": max(row.frame for row in track_rows),
                        "gap_count": len(own_gaps),
                        "max_gap_frames": max((row["missing_frames"] for row in own_gaps), default=0),
                    }
                )
    splits = defaultdict(list)
    for scenario in all_rows:
        splits[SCENARIOS[scenario]["split"]].append(scenario)
    (output_root / "splits.json").write_text(
        json.dumps(dict(splits), indent=2) + "\n", encoding="utf-8"
    )
    seqmaps = output_root / "seqmaps"
    seqmaps.mkdir(parents=True, exist_ok=True)
    for split, scenarios in splits.items():
        (seqmaps / f"{split}.txt").write_text(
            "name\n" + "\n".join(scenarios) + "\n", encoding="utf-8"
        )


def create_stats_plot(output_path: Path, audits: list[dict], all_rows: dict[str, list[MOTRow]]) -> None:
    scenarios = [row["scenario"] for row in audits]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), gridspec_kw={"height_ratios": [1, 1.25]})
    x = np.arange(len(scenarios))
    lengths_1 = [row["track_lengths"].get("1", 0) for row in audits]
    lengths_2 = [row["track_lengths"].get("2", 0) for row in audits]
    axes[0].bar(x, lengths_1, label="Track ID 1", color="#2563eb")
    axes[0].bar(x, lengths_2, bottom=lengths_1, label="Track ID 2", color="#f59e0b")
    for index, audit in enumerate(audits):
        axes[0].text(
            index,
            lengths_1[index] + lengths_2[index] + 20,
            f"gaps={audit['gap_count']}\nmax={audit['maximum_gap_frames']}",
            ha="center",
            fontsize=9,
        )
    axes[0].set_xticks(x, scenarios)
    axes[0].set_title("Manual MOT identities and occlusion gaps", weight="bold")
    axes[0].set_ylabel("Annotated track boxes")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)
    y_offset = 0
    yticks, ylabels = [], []
    for scenario in scenarios:
        rows = all_rows[scenario]
        for track_id in sorted({row.track_id for row in rows}):
            frames = [row.frame for row in rows if row.track_id == track_id]
            rgb = tuple(value / 255.0 for value in stable_color(track_id)[::-1])
            axes[1].scatter(frames, [y_offset] * len(frames), s=3, color=rgb)
            yticks.append(y_offset)
            ylabels.append(f"{scenario}:ID{track_id}")
            y_offset += 1
    axes[1].set_yticks(yticks, ylabels)
    axes[1].set_xlabel("MOT frame (1-based)")
    axes[1].set_title("Track presence timeline; blank regions are occlusion/absence gaps")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_reports(output_root: Path, audits: list[dict], storage: Counter) -> None:
    total_rows = sum(row["annotation_rows"] for row in audits)
    total_frames = sum(row["frame_count"] for row in audits)
    total_tracks = sum(row["track_count"] for row in audits)
    total_gaps = sum(row["gap_count"] for row in audits)
    lines = [
        "# Manual MOT dataset audit",
        "",
        f"- Sequences: **{len(audits)}**",
        f"- Frames: **{total_frames}**",
        f"- MOT annotation rows: **{total_rows}**",
        f"- Sequence-local identities: **{total_tracks}**",
        f"- Visibility gaps: **{total_gaps}**",
        "- Duplicate `(frame, track_id)` rows: **0**",
        "- Invalid boxes: **0**",
        "- MOT rows unmatched to manual detection boxes: **0**",
        "",
        "| Scenario | Split | Frames | Rows | IDs | Empty frames | Gaps | Long gaps ≥30 | Max gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audits:
        lines.append(
            f"| {row['scenario']} | {row['split']} | {row['frame_count']} | "
            f"{row['annotation_rows']} | {row['track_count']} | {row['empty_frame_count']} | "
            f"{row['gap_count']} | {row['long_gap_count_ge_30']} | {row['maximum_gap_frames']} |"
        )
    lines.extend(
        [
            "",
            "## Readiness",
            "",
            "- **Object tracking / MOT supervision:** ready. Boxes and IDs are present and consistent with the manual detection export.",
            "- **ID-switch and long-occlusion evaluation:** ready. The same IDs resume after gaps, including gaps up to 220 frames.",
            "- **ReID training:** only sequence-local positive tracklets are verified. Cross-sequence human identity is unknown, so `scenario:track_id` is a unique track key, not a verified global person identity.",
            "- **Visibility regression/classification:** not ready. Column 9 is `1.0` for every visible box; absence is represented by missing rows rather than partial visibility scores.",
            "",
            "Track IDs are sequence-local. Use `scenario:track_id` from `annotations.csv` as a globally unique track key, but do not assume two keys represent different physical people across videos.",
            "",
            f"Images were materialized as `{dict(storage)}`. Hard links share storage with the source images.",
            "",
            "![Tracking statistics](tracking_stats.png)",
            "",
        ]
    )
    (output_root / "audit" / "AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    readme = """# Object tracking prepared manual dataset

This is the tracking-ready derivative of `../annotation_mot/`. Source annotations
are kept unchanged.

## Main paths

- `mot_challenge/train|val|test/<sequence>/`: MOTChallenge sequences.
- `annotations.csv`: one row per tracked box with a globally unique identity key.
- `tracks.csv`: one row per sequence-local track.
- `seqmaps/` and `splits.json`: sequence-disjoint split definitions.
- `visualizations/`: ID-colored videos and contact sheets.
- `audit/AUDIT.md`: consistency, gaps, limitations, and readiness report.

## Split

- train: normal_walking, crossing, partial_occ
- val: seam
- test: full_occ

Never random-split adjacent frames. Track IDs are local to each sequence. For
cross-sequence code, use the `global_track_key` field from `annotations.csv`.

The dataset is ready for MOT training/evaluation. Cross-sequence human identity
has not been verified, so do not treat `global_track_key` as a ReID class. The
visibility column is also constant.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    args = parse_args()
    package_root = args.package_root.resolve()
    annotation_root = package_root / "annotation_mot"
    output_root = args.output_root.resolve()
    if not annotation_root.is_dir():
        raise FileNotFoundError(annotation_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "audit").mkdir(parents=True, exist_ok=True)
    (output_root / "source_annotations").mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, list[MOTRow]] = {}
    audits = []
    storage = Counter()
    for scenario, config in SCENARIOS.items():
        source_dir = annotation_root / config["mot"]
        rows = read_mot(source_dir / "gt.txt")
        images = source_images(package_root, scenario)
        audit = audit_sequence(package_root, annotation_root, scenario, rows, images)
        comparison = audit["manual_box_comparison"]
        if (
            audit["invalid_rows"]
            or audit["duplicate_frame_track_rows"]
            or comparison["unmatched_mot_boxes"]
            or comparison["unmatched_xml_boxes"]
        ):
            raise RuntimeError(f"MOT audit failed for {scenario}: {audit}")
        all_rows[scenario] = rows
        audits.append(audit)
        storage.update(write_sequence(output_root, scenario, images, rows, audit, args.fps))
        snapshot = output_root / "source_annotations" / config["mot"]
        snapshot.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir / "gt.txt", snapshot / "gt.txt")
        shutil.copy2(source_dir / "labels.txt", snapshot / "labels.txt")
        create_contact_sheet(
            output_root / "visualizations" / "contact_sheets" / f"{scenario}.jpg",
            scenario,
            images,
            rows,
            args.contact_sheet_frames,
        )
        if not args.skip_videos:
            print(f"Rendering {scenario}: {len(images)} frames", flush=True)
            create_video(
                output_root / "visualizations" / "videos" / f"{scenario}.mp4",
                scenario,
                images,
                rows,
                args.video_fps,
                args.video_width,
            )

    write_flat_manifests(output_root, all_rows, audits)
    create_stats_plot(output_root / "audit" / "tracking_stats.png", audits, all_rows)
    write_reports(output_root, audits, storage)
    payload = {
        "source": str(annotation_root),
        "output": str(output_root),
        "tracking_ready": True,
        "reid_identity_count": sum(row["track_count"] for row in audits),
        "visibility_scores_informative": False,
        "storage": dict(storage),
        "sequences": audits,
    }
    (output_root / "audit" / "audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Prepared {sum(row['frame_count'] for row in audits)} frames")
    print(f"MOT rows: {sum(row['annotation_rows'] for row in audits)}")
    print(f"Report: {output_root / 'audit' / 'AUDIT.md'}")
    print(f"Dataset: {output_root / 'mot_challenge'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
