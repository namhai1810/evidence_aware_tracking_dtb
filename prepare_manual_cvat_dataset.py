#!/usr/bin/env python3
"""Audit, visualize, and package manually edited CVAT image annotations.

The source XML files are never modified. The derived YOLO dataset is explicitly
detection-only because CVAT image annotations do not contain track identities.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACKAGE_ROOT = PROJECT_ROOT / "cvat_packages_trim300_last100"


SCENARIOS = {
    "normal_walking": {
        "xml": "dtb_normal_walking.xml",
        "sequence": "20261028T082400Z_46324651_normal_walking",
        "split": "train",
    },
    "crossing": {
        "xml": "dtb_crossing.xml",
        "sequence": "20261028T082643Z_5f566a39_crossing",
        "split": "train",
    },
    "seam": {
        "xml": "dtb_seam.xml",
        "sequence": "20261028T082930Z_153a02a2_seam",
        "split": "val",
    },
    "partial_occ": {
        "xml": "dtb_partial_occ.xml",
        "sequence": "20261028T083330Z_9fe9298b_partial_occ",
        "split": "train",
    },
    "full_occ": {
        "xml": "dtb_full_occ.xml",
        "sequence": "20261028T083518Z_e7ada31a_full_occ",
        "split": "test",
    },
}


@dataclass
class Box:
    label: str
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    occluded: bool
    attributes: dict[str, str]


@dataclass
class FrameAnnotation:
    frame_id: int
    name: str
    width: int
    height: int
    boxes: list[Box]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_PACKAGE_ROOT / "prepared_manual_dataset",
    )
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--contact-sheet-frames", type=int, default=12)
    parser.add_argument("--skip-videos", action="store_true")
    return parser.parse_args()


def load_xml(path: Path) -> tuple[list[FrameAnnotation], dict]:
    root = ET.parse(path).getroot()
    frames = []
    for image in root.findall("image"):
        boxes = []
        for node in image.findall("box"):
            boxes.append(
                Box(
                    label=node.attrib["label"],
                    xtl=float(node.attrib["xtl"]),
                    ytl=float(node.attrib["ytl"]),
                    xbr=float(node.attrib["xbr"]),
                    ybr=float(node.attrib["ybr"]),
                    occluded=node.attrib.get("occluded", "0") == "1",
                    attributes={
                        value.attrib.get("name", ""): value.text or ""
                        for value in node.findall("attribute")
                    },
                )
            )
        frames.append(
            FrameAnnotation(
                frame_id=int(image.attrib["id"]),
                name=image.attrib["name"],
                width=int(image.attrib["width"]),
                height=int(image.attrib["height"]),
                boxes=boxes,
            )
        )
    meta = {
        "version": root.findtext("version"),
        "track_count": len(root.findall("track")),
        "mode": root.findtext("./meta/job/mode"),
        "declared_size": int(root.findtext("./meta/job/size", "0")),
        "labels": [node.findtext("name") for node in root.findall("./meta/job/labels/label")],
    }
    frames.sort(key=lambda row: row.frame_id)
    return frames, meta


def image_path(package_root: Path, scenario: str, frame: FrameAnnotation) -> Path:
    sequence = SCENARIOS[scenario]["sequence"]
    return package_root / sequence / "cvat_for_video" / frame.name


def audit_sequence(
    package_root: Path,
    scenario: str,
    xml_path: Path,
    frames: list[FrameAnnotation],
    meta: dict,
) -> dict:
    missing_images = []
    invalid_boxes = []
    actual_dimension_mismatches = []
    labels = Counter()
    attributes = Counter()
    occluded = 0
    for frame in frames:
        source = image_path(package_root, scenario, frame)
        if not source.is_file():
            missing_images.append(str(source))
        else:
            image = cv2.imread(str(source))
            if image is None:
                missing_images.append(str(source))
            elif (image.shape[1], image.shape[0]) != (frame.width, frame.height):
                actual_dimension_mismatches.append(
                    {
                        "frame": frame.frame_id,
                        "xml": [frame.width, frame.height],
                        "image": [image.shape[1], image.shape[0]],
                    }
                )
        for box in frame.boxes:
            labels[box.label] += 1
            occluded += int(box.occluded)
            attributes.update(box.attributes)
            if not (
                0 <= box.xtl < box.xbr <= frame.width
                and 0 <= box.ytl < box.ybr <= frame.height
            ):
                invalid_boxes.append(
                    {
                        "frame": frame.frame_id,
                        "box": [box.xtl, box.ytl, box.xbr, box.ybr],
                        "size": [frame.width, frame.height],
                    }
                )
    ids = {frame.frame_id for frame in frames}
    expected_ids = set(range(min(ids), max(ids) + 1)) if ids else set()
    box_counts = [len(frame.boxes) for frame in frames]
    return {
        "scenario": scenario,
        "sequence": SCENARIOS[scenario]["sequence"],
        "split": SCENARIOS[scenario]["split"],
        "source_xml": str(xml_path.resolve()),
        "cvat_mode": meta["mode"],
        "declared_size": meta["declared_size"],
        "frame_count": len(frames),
        "box_count": sum(box_counts),
        "empty_frame_count": sum(value == 0 for value in box_counts),
        "min_boxes_per_frame": min(box_counts, default=0),
        "max_boxes_per_frame": max(box_counts, default=0),
        "mean_boxes_per_frame": float(np.mean(box_counts)) if box_counts else 0.0,
        "labels": dict(labels),
        "track_count": meta["track_count"],
        "occluded_box_count": occluded,
        "attribute_counts": dict(attributes),
        "missing_frame_ids": sorted(expected_ids - ids),
        "missing_images": missing_images,
        "invalid_boxes": invalid_boxes,
        "dimension_mismatches": actual_dimension_mismatches,
    }


def draw_frame(image: np.ndarray, frame: FrameAnnotation, scenario: str) -> np.ndarray:
    output = image.copy()
    color = (70, 230, 80)
    for index, box in enumerate(frame.boxes, start=1):
        x1, y1, x2, y2 = (int(round(value)) for value in (box.xtl, box.ytl, box.xbr, box.ybr))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        text = f"person {index}"
        cv2.rectangle(output, (x1, max(0, y1 - 24)), (x1 + 112, y1), color, -1)
        cv2.putText(
            output,
            text,
            (x1 + 4, max(17, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )
    banner = f"{scenario} | frame {frame.frame_id:06d} | persons={len(frame.boxes)}"
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


def sample_indices(frames: list[FrameAnnotation], count: int) -> list[int]:
    if not frames:
        return []
    selected = set(np.linspace(0, len(frames) - 1, min(count, len(frames)), dtype=int).tolist())
    counts = [len(frame.boxes) for frame in frames]
    for index in range(1, len(counts)):
        if counts[index] != counts[index - 1]:
            selected.add(index)
    empty = [index for index, value in enumerate(counts) if value == 0]
    if empty:
        selected.update([empty[0], empty[len(empty) // 2], empty[-1]])
    ordered = sorted(selected)
    if len(ordered) > count:
        priority = set(np.linspace(0, len(ordered) - 1, count, dtype=int).tolist())
        ordered = [value for index, value in enumerate(ordered) if index in priority]
    return ordered


def create_contact_sheet(
    package_root: Path,
    output_path: Path,
    scenario: str,
    frames: list[FrameAnnotation],
    count: int,
) -> None:
    panels = []
    for index in sample_indices(frames, count):
        frame = frames[index]
        image = cv2.imread(str(image_path(package_root, scenario, frame)))
        rendered = draw_frame(image, frame, scenario)
        scale = min(420 / rendered.shape[1], 230 / rendered.shape[0])
        resized = cv2.resize(
            rendered,
            (int(rendered.shape[1] * scale), int(rendered.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
        panel = np.full((250, 440, 3), 245, dtype=np.uint8)
        top = (230 - resized.shape[0]) // 2
        left = (440 - resized.shape[1]) // 2
        panel[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
        cv2.putText(
            panel,
            f"frame {frame.frame_id:06d} | boxes={len(frame.boxes)}",
            (12, 244),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    columns = 3
    rows = int(np.ceil(len(panels) / columns))
    while len(panels) < rows * columns:
        panels.append(np.full_like(panels[0], 235))
    sheet = np.vstack([np.hstack(panels[row * columns : (row + 1) * columns]) for row in range(rows)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Could not write {output_path}")


def create_video(
    package_root: Path,
    output_path: Path,
    scenario: str,
    frames: list[FrameAnnotation],
    fps: float,
    output_width: int,
) -> None:
    first = cv2.imread(str(image_path(package_root, scenario, frames[0])))
    scale = min(1.0, output_width / first.shape[1])
    size = (int(round(first.shape[1] * scale)), int(round(first.shape[0] * scale)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video {output_path}")
    try:
        for frame in frames:
            image = cv2.imread(str(image_path(package_root, scenario, frame)))
            rendered = draw_frame(image, frame, scenario)
            if (rendered.shape[1], rendered.shape[0]) != size:
                rendered = cv2.resize(rendered, size, interpolation=cv2.INTER_AREA)
            writer.write(rendered)
    finally:
        writer.release()


def yolo_line(box: Box, width: int, height: int) -> str:
    center_x = ((box.xtl + box.xbr) / 2) / width
    center_y = ((box.ytl + box.ybr) / 2) / height
    box_width = (box.xbr - box.xtl) / width
    box_height = (box.ybr - box.ytl) / height
    return f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}"


def hardlink_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == source.stat().st_size:
            return "existing"
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def build_yolo_dataset(
    package_root: Path,
    output_root: Path,
    parsed: dict[str, list[FrameAnnotation]],
) -> list[dict]:
    rows = []
    link_modes = Counter()
    yolo_root = output_root / "detection_yolo"
    for scenario, frames in parsed.items():
        split = SCENARIOS[scenario]["split"]
        for frame in frames:
            source = image_path(package_root, scenario, frame)
            stem = f"{scenario}__frame_{frame.frame_id:06d}"
            image_destination = yolo_root / "images" / split / f"{stem}{source.suffix.lower()}"
            label_destination = yolo_root / "labels" / split / f"{stem}.txt"
            link_modes[hardlink_or_copy(source, image_destination)] += 1
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            lines = [yolo_line(box, frame.width, frame.height) for box in frame.boxes]
            label_destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            rows.append(
                {
                    "split": split,
                    "scenario": scenario,
                    "sequence": SCENARIOS[scenario]["sequence"],
                    "frame_id": frame.frame_id,
                    "source_image": str(source.resolve()),
                    "dataset_image": str(image_destination.relative_to(yolo_root)),
                    "dataset_label": str(label_destination.relative_to(yolo_root)),
                    "width": frame.width,
                    "height": frame.height,
                    "person_count": len(frame.boxes),
                }
            )
    with (yolo_root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (yolo_root / "data.yaml").write_text(
        f"path: {yolo_root.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: person\n",
        encoding="utf-8",
    )
    split_stats = Counter(row["split"] for row in rows)
    box_stats = Counter()
    empty_stats = Counter()
    for row in rows:
        box_stats[row["split"]] += int(row["person_count"])
        empty_stats[row["split"]] += int(row["person_count"] == 0)
    readme = f"""# Manual CVAT annotations: YOLO detection dataset

This is a derived, detection-only dataset. The five source XML files are kept
unchanged under `{package_root / 'annotation_file'}`.

## Split policy

- train: `normal_walking`, `crossing`, `partial_occ`
- val: `seam`
- test: `full_occ`

Splits are sequence-disjoint to avoid adjacent-frame leakage.

| Split | Images | Person boxes | Empty images |
|---|---:|---:|---:|
| train | {split_stats['train']} | {box_stats['train']} | {empty_stats['train']} |
| val | {split_stats['val']} | {box_stats['val']} | {empty_stats['val']} |
| test | {split_stats['test']} | {box_stats['test']} | {empty_stats['test']} |

Images were materialized as: `{dict(link_modes)}`. Hard links consume no second
copy on this filesystem but behave as normal files to training code.

## Important limitation

The manually exported CVAT XML files contain per-image `person` boxes only.
They contain no `track_id`, no `<track>` elements, and no visibility attributes.
This dataset is ready for person detection training, but **not** for supervised
MOT/ReID/identity training. Do not derive identity ground truth from box order.
"""
    (yolo_root / "README.md").write_text(readme, encoding="utf-8")
    return rows


def create_stats_plot(output_path: Path, audits: list[dict], parsed: dict[str, list[FrameAnnotation]]) -> None:
    scenarios = [row["scenario"] for row in audits]
    frames = [row["frame_count"] for row in audits]
    boxes = [row["box_count"] for row in audits]
    empty = [row["empty_frame_count"] for row in audits]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 1.2]})
    x = np.arange(len(scenarios))
    width = 0.36
    axes[0].bar(x - width / 2, frames, width, label="Frames", color="#2563eb")
    axes[0].bar(x + width / 2, boxes, width, label="Person boxes", color="#16a34a")
    for index, value in enumerate(empty):
        axes[0].text(index, frames[index] + 20, f"empty={value}", ha="center", fontsize=9)
    axes[0].set_xticks(x, scenarios)
    axes[0].set_title("Manual CVAT annotation inventory", weight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)
    for scenario in scenarios:
        counts = [len(frame.boxes) for frame in parsed[scenario]]
        axes[1].plot(counts, label=scenario, linewidth=1.1)
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("Person boxes")
    axes[1].set_title("Box count over time")
    axes[1].set_yticks(sorted({0, 1, 2}))
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_audit_report(output_root: Path, audits: list[dict], total: dict) -> None:
    lines = [
        "# Audit of manually exported CVAT annotations",
        "",
        "## Summary",
        "",
        f"- XML files: **{len(audits)}**",
        f"- Frames: **{total['frames']}**",
        f"- Person boxes: **{total['boxes']}**",
        f"- Empty frames: **{total['empty_frames']}**",
        f"- Invalid boxes: **{total['invalid_boxes']}**",
        f"- Missing images: **{total['missing_images']}**",
        f"- Manual track identities: **0**",
        f"- Occluded flags set: **{total['occluded_boxes']}**",
        f"- Visibility attributes: **0**",
        "",
        "| Scenario | Split | Frames | Boxes | Empty | Tracks | Occluded flags |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in audits:
        lines.append(
            f"| {row['scenario']} | {row['split']} | {row['frame_count']} | "
            f"{row['box_count']} | {row['empty_frame_count']} | {row['track_count']} | "
            f"{row['occluded_box_count']} |"
        )
    lines.extend(
        [
            "",
            "## Readiness decision",
            "",
            "- **Ready for person detection:** yes. All boxes are valid and all source images exist.",
            "- **Ready for tracking/ReID identity training:** no. The manual XML files are CVAT image annotations and contain no track IDs.",
            "- **Ready for visibility/occlusion supervision:** no. Every existing box has `occluded=0` and no visibility attribute was exported.",
            "",
            "The original package XML files contain automatic tracks, but those IDs must not be treated as manually verified identity ground truth. If MOT/ReID training is required, identities need a second CVAT video/interpolation review or a separate recovery-and-verification pass.",
            "",
            "## Recommended storage policy",
            "",
            "Keep `annotation_file/` immutable as the raw manual export. Use `prepared_manual_dataset/detection_yolo/` as a reproducible derived training format. Never edit both copies independently.",
            "",
            "![Dataset statistics](stats.png)",
            "",
        ]
    )
    (output_root / "audit" / "AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def write_root_readme(output_root: Path) -> None:
    content = """# Prepared manual panorama dataset

This directory is derived from the five manually exported CVAT XML files. The
source XML files under `../annotation_file/` remain unchanged.

## What is ready

- `detection_yolo/`: sequence-disjoint YOLO person-detection dataset.
- `visualizations/videos/`: complete annotated videos for visual review.
- `visualizations/contact_sheets/`: sampled annotated frames per scenario.
- `audit/`: machine-readable and Markdown data-quality reports.
- `source_annotations/`: snapshot of the five source XML files used to build it.

## Important boundary

The manual XML files contain person boxes but no track IDs and no visibility
attributes. This dataset is ready for detection training. It is not yet valid
ground truth for supervised MOT, ReID, ID-switch evaluation, or visibility
classification.

Read `audit/AUDIT.md` first, then use `detection_yolo/data.yaml` for detection
training. Do not edit the generated labels by hand; update the source CVAT XML
and rerun `code/prepare_manual_cvat_dataset.py` instead.
"""
    (output_root / "README.md").write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    package_root = args.package_root.resolve()
    output_root = args.output_root.resolve()
    annotation_root = package_root / "annotation_file"
    if not annotation_root.is_dir():
        raise FileNotFoundError(annotation_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "audit").mkdir(parents=True, exist_ok=True)
    (output_root / "source_annotations").mkdir(parents=True, exist_ok=True)

    parsed: dict[str, list[FrameAnnotation]] = {}
    audits = []
    for scenario, config in SCENARIOS.items():
        xml_path = annotation_root / config["xml"]
        if not xml_path.is_file():
            raise FileNotFoundError(xml_path)
        frames, meta = load_xml(xml_path)
        parsed[scenario] = frames
        audit = audit_sequence(package_root, scenario, xml_path, frames, meta)
        audits.append(audit)
        shutil.copy2(xml_path, output_root / "source_annotations" / xml_path.name)
        create_contact_sheet(
            package_root,
            output_root / "visualizations" / "contact_sheets" / f"{scenario}.jpg",
            scenario,
            frames,
            args.contact_sheet_frames,
        )
        if not args.skip_videos:
            print(f"Rendering {scenario}: {len(frames)} frames", flush=True)
            create_video(
                package_root,
                output_root / "visualizations" / "videos" / f"{scenario}.mp4",
                scenario,
                frames,
                args.video_fps,
                args.video_width,
            )

    total = {
        "frames": sum(row["frame_count"] for row in audits),
        "boxes": sum(row["box_count"] for row in audits),
        "empty_frames": sum(row["empty_frame_count"] for row in audits),
        "invalid_boxes": sum(len(row["invalid_boxes"]) for row in audits),
        "missing_images": sum(len(row["missing_images"]) for row in audits),
        "occluded_boxes": sum(row["occluded_box_count"] for row in audits),
    }
    if total["invalid_boxes"] or total["missing_images"]:
        raise RuntimeError(f"Dataset audit failed: {total}")
    rows = build_yolo_dataset(package_root, output_root, parsed)
    create_stats_plot(output_root / "audit" / "stats.png", audits, parsed)
    write_audit_report(output_root, audits, total)
    write_root_readme(output_root)
    write_json = {
        "source_annotation_root": str(annotation_root),
        "output_root": str(output_root),
        "total": total,
        "identity_labels_available": False,
        "visibility_labels_available": False,
        "detection_training_ready": True,
        "tracking_training_ready": False,
        "sequences": audits,
        "derived_yolo_rows": len(rows),
    }
    (output_root / "audit" / "audit.json").write_text(
        json.dumps(write_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(total, indent=2))
    print(f"Audit: {output_root / 'audit' / 'AUDIT.md'}")
    print(f"Visualizations: {output_root / 'visualizations'}")
    print(f"Detection dataset: {output_root / 'detection_yolo'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
