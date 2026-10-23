#!/usr/bin/env python3
"""Build a direct ByteTrack versus BoT-SORT+ReID comparison report."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
OUTPUTS = CODE_DIR / "outputs"
REPORT = OUTPUTS / "all_sequences_reid_comparison_report.md"
HUNGARIAN_ORACLE_SUMMARY = (
    OUTPUTS / "highres_retrieval_hungarian_gt_oracle_botsort_reid" / "summary.json"
)
EMBEDDING_RETRIEVAL_SUMMARY = OUTPUTS / "highres_retrieval_reid_embedding" / "summary.json"

RUNS = {
    "byte_512": OUTPUTS / "all_sequences_512x120" / "metrics_iou_0.5.json",
    "reid_512": OUTPUTS / "all_sequences_reid_512x120" / "metrics_iou_0.5.json",
    "byte_original": OUTPUTS / "all_sequences_original_2048x480" / "metrics_iou_0.5.json",
    "reid_original": OUTPUTS / "all_sequences_reid_original_2048x480" / "metrics_iou_0.5.json",
}


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def metric_row(label: str, metrics: dict) -> str:
    return (
        f"| {label} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | "
        f"{metrics['f1']:.3f} | {metrics['mean_iou']:.3f} | {metrics['mota']:.3f} | "
        f"{metrics['idf1']:.3f} | {metrics['id_switches']} | {metrics['tp']} | "
        f"{metrics['fp']} | {metrics['fn']} |"
    )


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def delta_row(label: str, baseline: dict, candidate: dict) -> str:
    return (
        f"| {label} | {signed(candidate['precision'] - baseline['precision'])} | "
        f"{signed(candidate['recall'] - baseline['recall'])} | "
        f"{signed(candidate['f1'] - baseline['f1'])} | "
        f"{signed(candidate['mean_iou'] - baseline['mean_iou'])} | "
        f"{signed(candidate['mota'] - baseline['mota'])} | "
        f"{signed(candidate['idf1'] - baseline['idf1'])} | "
        f"{candidate['id_switches'] - baseline['id_switches']:+d} | "
        f"{candidate['tp'] - baseline['tp']:+d} | "
        f"{candidate['fp'] - baseline['fp']:+d} | "
        f"{candidate['fn'] - baseline['fn']:+d} |"
    )


def sequence_table(title: str, baseline: dict, candidate: dict) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Seq | ByteTrack MOTA | ReID MOTA | Δ MOTA | ByteTrack IDF1 | ReID IDF1 | Δ IDF1 | ByteTrack IDSW | ReID IDSW | Δ IDSW |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sequence in sorted(baseline):
        old = baseline[sequence]
        new = candidate[sequence]
        lines.append(
            f"| {sequence} | {old['mota']:.3f} | {new['mota']:.3f} | "
            f"{signed(new['mota'] - old['mota'])} | {old['idf1']:.3f} | "
            f"{new['idf1']:.3f} | {signed(new['idf1'] - old['idf1'])} | "
            f"{old['id_switches']} | {new['id_switches']} | "
            f"{new['id_switches'] - old['id_switches']:+d} |"
        )
    lines.append("")
    return lines


def relative(path: Path) -> str:
    return path.relative_to(REPORT.parent).as_posix()


def main() -> None:
    data = {name: load(path) for name, path in RUNS.items()}
    b512 = data["byte_512"]["aggregate"]
    r512 = data["reid_512"]["aggregate"]
    borg = data["byte_original"]["aggregate"]
    rorg = data["reid_original"]["aggregate"]

    idsw_drop_512 = b512["id_switches"] - r512["id_switches"]
    idsw_drop_original = borg["id_switches"] - rorg["id_switches"]
    idsw_pct_512 = 100 * idsw_drop_512 / b512["id_switches"]
    idsw_pct_original = 100 * idsw_drop_original / borg["id_switches"]

    lines = [
        "# QuadTrack: ByteTrack vs BoT-SORT + ReID",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Scope and configuration",
        "",
        "- Dataset: all 17 QuadTrack train sequences (`0015`–`0031`).",
        "- Detector: YOLO11l; person class only; confidence `0.20`; NMS IoU `0.70`.",
        "- Evaluation: matching IoU `>=0.50`, micro-aggregated across sequences.",
        "- Low-resolution run: physical input `512x120`, inference `imgsz=512`.",
        "- Original run: physical input `2048x480`, inference `imgsz=960`.",
        "- ReID run: Ultralytics BoT-SORT with `with_reid: true`, `model: auto`, and sparse optical-flow GMC.",
        "- Important: this compares the full trackers. Because ByteTrack and BoT-SORT also differ in matching and GMC, it is not a controlled appearance-ReID-only ablation.",
        "",
        "## Aggregate metrics",
        "",
        "| Input / tracker | Precision | Recall | F1 | mIoU | MOTA | IDF1 | IDSW | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        metric_row("512x120 / ByteTrack", b512),
        metric_row("512x120 / BoT-SORT+ReID", r512),
        metric_row("2048x480 / ByteTrack", borg),
        metric_row("2048x480 / BoT-SORT+ReID", rorg),
        "",
        "## ReID delta relative to ByteTrack",
        "",
        "| Input | Δ Precision | Δ Recall | Δ F1 | Δ mIoU | Δ MOTA | Δ IDF1 | Δ IDSW | Δ TP | Δ FP | Δ FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        delta_row("512x120", b512, r512),
        delta_row("2048x480", borg, rorg),
        "",
        "## Main observations",
        "",
        f"- At `512x120`, ReID reduces ID switches by **{idsw_drop_512} ({idsw_pct_512:.1f}%)**, raises IDF1 by **{r512['idf1'] - b512['idf1']:+.3f}**, and raises MOTA by **{r512['mota'] - b512['mota']:+.3f}**.",
        f"- At `2048x480`, ReID reduces ID switches by **{idsw_drop_original} ({idsw_pct_original:.1f}%)** and raises IDF1 by **{rorg['idf1'] - borg['idf1']:+.3f}**.",
        f"- Original-resolution MOTA changes by **{rorg['mota'] - borg['mota']:+.3f}** because the ReID run adds {rorg['fp'] - borg['fp']:+d} FP while recovering {borg['fn'] - rorg['fn']:+d} FN. This tracker setting improves identity continuity but not overall detection error at that resolution.",
        "- For the planned edge-retrieval system, IDF1 and IDSW are the most directly relevant aggregate signals; the frame-level logs below contain the exact switch/FN/FP evidence for ranking experiments.",
        "",
    ]

    lines.extend(
        sequence_table(
            "Per-sequence comparison: 512x120",
            data["byte_512"]["sequences"],
            data["reid_512"]["sequences"],
        )
    )
    lines.extend(
        sequence_table(
            "Per-sequence comparison: 2048x480",
            data["byte_original"]["sequences"],
            data["reid_original"]["sequences"],
        )
    )

    if HUNGARIAN_ORACLE_SUMMARY.is_file():
        retrieval = load(HUNGARIAN_ORACLE_SUMMARY)
        lines.extend(
            [
                "## Historical GT/Hungarian retrieval oracle",
                "",
                "This older experiment links the low- and full-resolution streams through GT identities assigned with Hungarian IoU matching. It does not compare cross-resolution appearance embeddings.",
                "",
                f"- Low-resolution BoT-SORT+ReID ID-switch events: **{retrieval['total_lr_switch_events']}**.",
                f"- Events recoverable using at least one original-resolution history frame: **{retrieval['recoverable_events']} ({retrieval['recoverable_event_fraction']:.1%})**.",
                f"- Recoverable unique target frames: **{retrieval['recoverable_unique_frames']} ({retrieval['recoverable_unique_frame_fraction']:.1%})**.",
                f"- Oracle IDSW upper bound: **{retrieval['total_lr_switch_events']} → {retrieval['oracle_remaining_idsw']}**; MOTA **{retrieval['lr_mota']:.3f} → {retrieval['oracle_mota_upper_bound']:.3f}** if FN and FP stay fixed.",
                "- This is supervision/ranking potential, not a measured online identity merge.",
                "",
                "See [the explicitly named GT/Hungarian oracle report](highres_retrieval_hungarian_gt_oracle_botsort_reid_report.md).",
                "",
            ]
        )

    if EMBEDDING_RETRIEVAL_SUMMARY.is_file():
        retrieval = load(EMBEDDING_RETRIEVAL_SUMMARY)
        lines.extend(
            [
                "## Cross-resolution ReID embedding retrieval",
                "",
                "This experiment ranks historical 2048x480 detections by cosine similarity to the old-track and target embeddings extracted from the physical 512x120 stream. GT is used only to score the ranking.",
                "",
                f"- Switch events: **{retrieval['total_lr_switch_events']}**; correct HR identity available: **{retrieval['events_with_correct_identity_available']} ({retrieval['correct_identity_availability_fraction']:.1%})**.",
                f"- Correct top-1 retrieval: **{retrieval['top1_improvable_events']} ({retrieval['top1_improvable_fraction']:.1%})**.",
                f"- Correct identity in top five: **{retrieval['top5_improvable_events']} ({retrieval['top5_improvable_fraction']:.1%})**.",
                f"- Retrieval-supported IDSW estimates: **{retrieval['total_lr_switch_events']} → {retrieval['estimated_remaining_idsw_top1']}** for top-1, or **→ {retrieval['estimated_remaining_idsw_top5']}** with an oracle choice among top five.",
                "",
                "See [the cross-resolution embedding report](highres_retrieval_reid_embedding_report.md).",
                "",
            ]
        )

    lines.extend(
        [
            "## ReID artifacts",
            "",
            f"- [512x120 detailed report]({relative(OUTPUTS / 'all_sequences_reid_512x120_report.md')})",
            f"- [512x120 metrics JSON]({relative(RUNS['reid_512'])})",
            f"- [512x120 failure CSV]({relative(OUTPUTS / 'all_sequences_reid_512x120' / 'failure_frames.csv')})",
            f"- [512x120 failure JSON]({relative(OUTPUTS / 'all_sequences_reid_512x120' / 'failure_frames.json')})",
            f"- [2048x480 detailed report]({relative(OUTPUTS / 'all_sequences_reid_original_2048x480_report.md')})",
            f"- [2048x480 metrics JSON]({relative(RUNS['reid_original'])})",
            f"- [2048x480 failure CSV]({relative(OUTPUTS / 'all_sequences_reid_original_2048x480' / 'failure_frames.csv')})",
            f"- [2048x480 failure JSON]({relative(OUTPUTS / 'all_sequences_reid_original_2048x480' / 'failure_frames.json')})",
            "- Annotated videos and MOT-format predictions are in each run's `predictions/` directory.",
            "",
        ]
    )

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
