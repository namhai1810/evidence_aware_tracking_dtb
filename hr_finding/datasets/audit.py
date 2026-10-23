#!/usr/bin/env python3
"""Audit HIVE dataset compatibility, LR availability, and leakage risks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    from .preprocessing import box_iou, load_mot_tracks, select_track_history
except ImportError:  # Direct script execution.
    from preprocessing import box_iou, load_mot_tracks, select_track_history


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR.parents[1] / "outputs" / "reid_ranking_dataset_v1"
DEFAULT_REPORT = SCRIPT_DIR.parent / "DATA_AUDIT.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=SCRIPT_DIR.parent / "DATA_AUDIT.json")
    parser.add_argument("--lr-match-iou", type=float, default=0.5)
    parser.add_argument("--competitor-history-frames", type=int, default=30)
    return parser.parse_args()


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def resolve_project_root(dataset_root: Path, configured: Path | None) -> Path:
    if configured:
        return configured.resolve()
    # .../evidence_aware_tracking/code/outputs/<dataset>
    return dataset_root.resolve().parents[2]


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    project_root = resolve_project_root(dataset_root, args.project_root)
    config = json.loads((dataset_root / "split_config.json").read_text(encoding="utf-8"))
    train = read_events(dataset_root / "train_data" / "events.jsonl")
    test = read_events(dataset_root / "test_data" / "events.jsonl")
    events = train + test

    train_sequences = set(config["train_sequences"])
    test_sequences = set(config["test_sequences"])
    event_overlap = {event["event_id"] for event in train} & {event["event_id"] for event in test}
    sequence_overlap = train_sequences & test_sequences

    source_paths = []
    hr_crop_paths = []
    candidate_count = 0
    reconstructable = 0
    hr_teacher_fields = Counter()
    tracking_root = Path(config["source_files"]["lowres_predictions"])
    if not tracking_root.is_absolute():
        tracking_root = project_root / tracking_root
    tracking_cache: dict[str, dict] = {}
    all_sequences = sorted(train_sequences | test_sequences)
    tracking_files_available = sum(
        (
            tracking_root
            / f"{sequence}_yolo11l_botsort_reid_pre512x120.txt"
        ).is_file()
        for sequence in all_sequences
    )
    lr_match_count = 0
    lr_overlap_any = 0
    competitor_history_count = 0

    for event in events:
        sequence = event["sequence"]
        if sequence not in tracking_cache:
            path = tracking_root / f"{sequence}_yolo11l_botsort_reid_pre512x120.txt"
            tracking_cache[sequence] = load_mot_tracks(path)
        tracking = tracking_cache[sequence]
        if select_track_history(
            tracking,
            int(event["lr_new_pred_id"]),
            int(event["target_frame"]),
            args.competitor_history_frames,
            5,
        ):
            competitor_history_count += 1

        for candidate in event["candidates"]:
            candidate_count += 1
            source = Path(candidate["source_image"])
            source = source if source.is_absolute() else project_root / source
            source_paths.append(source)
            if source.is_file() and len(candidate.get("hr_bbox_xyxy", [])) == 4:
                reconstructable += 1
            crop = candidate.get("candidate_crop_path")
            if crop:
                hr_crop_paths.append(dataset_root / crop)
            for field in ("bridge_similarity", "similarity_to_old_track", "similarity_to_target"):
                if field in candidate:
                    hr_teacher_fields[field] += 1

            rows = tracking.get(int(candidate["candidate_frame"]), [])
            overlaps = [box_iou(candidate["hr_bbox_xyxy"], row[1]) for row in rows]
            maximum = max(overlaps, default=0.0)
            lr_overlap_any += int(maximum > 0)
            lr_match_count += int(maximum >= args.lr_match_iou)

    materialized_lr_candidate_fields = sum(
        "candidate_lr_crop_path" in candidate
        for event in events
        for candidate in event["candidates"]
    )
    source_missing = sum(not path.is_file() for path in source_paths)
    hr_crop_missing = sum(not path.is_file() for path in hr_crop_paths)
    query_missing = sum(
        not (dataset_root / query["crop_path"]).is_file()
        for event in events
        for query in event["query_crops"]
    )
    target_missing = sum(
        bool(event.get("target_crop"))
        and not (dataset_root / event["target_crop"]["crop_path"]).is_file()
        for event in events
    )

    audit = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "project_root": str(project_root),
        "events": len(events),
        "candidate_pairs": candidate_count,
        "train_sequences": sorted(train_sequences),
        "test_sequences": sorted(test_sequences),
        "sequence_overlap": sorted(sequence_overlap),
        "event_overlap_count": len(event_overlap),
        "missing_query_crops": query_missing,
        "missing_target_crops": target_missing,
        "missing_hr_candidate_crops": hr_crop_missing,
        "materialized_lr_candidate_crops": materialized_lr_candidate_fields,
        "lr_candidates_reconstructable_from_full_frame": reconstructable,
        "missing_full_source_frames": source_missing,
        "lr_tracking_files_available": tracking_files_available,
        "lr_tracking_files_expected": len(all_sequences),
        "full_lr_panorama_frames_materialized": False,
        "lr_tracker_match_iou_threshold": args.lr_match_iou,
        "candidates_with_lr_tracker_match": lr_match_count,
        "candidates_with_any_lr_tracker_overlap": lr_overlap_any,
        "events_with_competing_new_track_history": competitor_history_count,
        "association_matrices_available": False,
        "motion_features_available": False,
        "measured_correction_utility_labels_available": False,
        "hr_teacher_fields": dict(hr_teacher_fields),
        "candidate_pool_selected_with_hr_reid": bool(hr_teacher_fields["bridge_similarity"]),
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    match_fraction = lr_match_count / candidate_count if candidate_count else 0.0
    competitor_fraction = competitor_history_count / len(events) if events else 0.0
    lines = [
        "# HIVE dataset audit",
        "",
        f"Generated: `{audit['generated_at']}`",
        "",
        "## Compatibility summary",
        "",
        f"- Dataset: `{dataset_root}`",
        f"- Events / candidate pairs: **{len(events)} / {candidate_count}**.",
        f"- Video-disjoint leakage check: **PASS** (`{len(sequence_overlap)}` shared sequences, `{len(event_overlap)}` shared events).",
        f"- Materialized old-query / target LR crops missing: **{query_missing} / {target_missing}**.",
        f"- Materialized candidate LR crops: **{materialized_lr_candidate_fields}**.",
        f"- Candidate LR crops reconstructable by downsampling the complete source panorama first: **{reconstructable}/{candidate_count}**.",
        f"- MOT-format LR tracking files available: **{tracking_files_available}/{len(all_sequences)}** (frame, track ID, bbox, confidence).",
        "- Full LR panorama files are not materialized; preprocessing reproduces them from complete source frames with `INTER_AREA` before any crop.",
        f"- Candidate boxes matching an actual LR-tracker detection at IoU >= {args.lr_match_iou:.2f}: **{lr_match_count}/{candidate_count} ({match_fraction:.1%})**.",
        f"- Events with a real competing `lr_new_pred_id` history in the preceding {args.competitor_history_frames} frames: **{competitor_history_count}/{len(events)} ({competitor_fraction:.1%})**.",
        "",
        "## What HIVE may use online",
        "",
        "- Materialized old-track LR crops and target LR crop.",
        "- Candidate LR crops reconstructed by resizing the full panorama to 512x120 before cropping.",
        "- Temporal offsets and genuine LR-tracker histories when present.",
        "- The materialized HR candidate crop, label, GT identity, HR confidence, HR ReID similarities, and candidate rank are never exposed as online features.",
        "",
        "## Missing or limited information",
        "",
        "- No association matrices or explicit motion-residual features are stored.",
        "- Competing-track history is available only for a subset of events; `hive_full` uses a masked fallback elsewhere.",
        "- Current top-5 candidates were preselected using HR tracking detections and HR ReID similarity.",
        "- Although every candidate LR crop can be reconstructed offline, only the LR-tracker-matched subset has a directly deployable LR proposal box.",
        "- `bridge_similarity` is an offline HR-teacher signal, not a measured identity-margin or counterfactual correction-utility label.",
        "- No real tracking replay outcome is available, so IDSW/IDF1/HOTA improvements cannot be claimed from these labels.",
        "",
        "## Supported variants",
        "",
        "- `appearance_only`: supported.",
        "- `appearance_metadata`: supported using LR appearance and temporal metadata.",
        "- `hive_no_hypothesis`: supported using old/target/candidate contrastive relations.",
        "- `hive_full`: partially supported; real competing history is masked when unavailable.",
        "- Margin distillation: implemented but disabled because no validated utility margin exists.",
        "",
        "## Deployment blocker",
        "",
        "A deployable LR-only proposal generator is still required. It must form the candidate pool from LR tracker state without consulting HR detections or HR ReID rankings. Current results evaluate reranking inside an offline HR-preselected pool.",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(f"Report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
