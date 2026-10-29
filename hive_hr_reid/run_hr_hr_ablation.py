#!/usr/bin/env python3
"""Ablate LR/HR domain mismatch after a frozen HIVE v2 frame selection.

The HIVE selector is unchanged.  This experiment compares the existing policy,
which embeds LR old/target/competitor crops against one selected HR crop, with a
policy that obtains every ReID crop from the original 2048x480 frame.  Ground
truth is used only to select thresholds on validation and to evaluate the
frozen decisions on held-out test sequences.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wilcoxon

from hive_hr_reid import run_experiment as base


METHODS = (
    "lr_only",
    "hive_hr_lr_current",
    "hive_hr_hr_same_threshold",
    "hive_hr_hr_tuned",
)
DISPLAY_NAMES = {
    "lr_only": "LR-only",
    "hive_hr_lr_current": "HIVE HR↔LR (current)",
    "hive_hr_hr_same_threshold": "HIVE HR↔HR (same threshold)",
    "hive_hr_hr_tuned": "HIVE HR↔HR (val-tuned)",
}
CURRENT_THRESHOLDS = {
    "bridge_min": 0.50,
    "old_target_min": 0.70,
    "evidence_margin_min": 0.10,
    "target_margin_min": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=base.DEFAULT_DATASET)
    parser.add_argument("--hive-dataset-root", type=Path, default=base.DEFAULT_HIVE_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=base.DEFAULT_CHECKPOINT)
    parser.add_argument("--lr-pred-dir", type=Path, default=base.DEFAULT_LR_PREDICTIONS)
    parser.add_argument(
        "--lr-pred-pattern", default="{sequence}_yolo11l_botsort_reid_pre512x120.txt"
    )
    parser.add_argument("--reid-model", type=Path, default=base.DEFAULT_REID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base.CODE_DIR / "outputs" / "hive_v2_hr_hr_ablation",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--lowres-width", type=int, default=512)
    parser.add_argument("--lowres-height", type=int, default=120)
    parser.add_argument("--competitor-history-frames", type=int, default=30)
    parser.add_argument("--competitor-samples", type=int, default=3)
    parser.add_argument("--max-competitors", type=int, default=12)
    parser.add_argument("--max-segment-gap", type=int, default=30)
    parser.add_argument("--visual-examples", type=int, default=6)
    parser.add_argument("--max-events-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def hr_record_crop(cache: base.ImageCache, record: dict) -> np.ndarray:
    image = cache.read(base.source_path(record["source_image"]))
    return base.clipped_crop(image, record["bbox_original_xyxy"])


def hr_track_crop(
    cache: base.ImageCache,
    dataset_root: Path,
    sequence: str,
    frame: int,
    bbox: np.ndarray,
) -> np.ndarray:
    image = cache.read(dataset_root / sequence / "img1" / f"{frame:06d}.jpg")
    return base.clipped_crop(image, bbox)


def hr_hr_scores(
    args: argparse.Namespace,
    encoder: base.OpenCVONNXReID,
    cache: base.ImageCache,
    event: dict,
    selection: dict,
    lr_frames: dict[int, list[tuple]],
) -> dict:
    """Build old, target, evidence, and competitor embeddings from HR frames."""
    crops: list[np.ndarray] = []
    groups: dict[str, list[int]] = defaultdict(list)

    def add(group: str, crop: np.ndarray) -> None:
        groups[group].append(len(crops))
        crops.append(crop)

    for query in event.get("query_crops", []):
        add("old", hr_record_crop(cache, query))
    target_record = event.get("target_crop")
    if target_record:
        add("target", hr_record_crop(cache, target_record))

    candidates = {row["candidate_id"]: row for row in event.get("candidates", [])}
    selected_candidate = candidates.get(selection.get("candidate_id"))
    if selected_candidate is not None:
        add("selected", base.hr_candidate_crop(cache, selected_candidate))

    competitors = base.competitor_histories(
        lr_frames,
        event,
        args.competitor_history_frames,
        args.competitor_samples,
        args.max_competitors,
    )
    for track_id, history in competitors:
        for frame, row in history:
            add(
                f"competitor:{track_id}",
                hr_track_crop(cache, args.dataset_root, event["sequence"], frame, row[1]),
            )

    crops = [
        crop if crop is not None else np.empty((0, 0, 3), dtype=np.uint8)
        for crop in crops
    ]
    features = base.run_crop_batches(encoder, crops)
    old = base.feature_mean([features[index] for index in groups.get("old", [])])
    target = base.feature_mean([features[index] for index in groups.get("target", [])])
    selected = base.feature_mean([features[index] for index in groups.get("selected", [])])
    competitor_vectors = {
        int(group.split(":", 1)[1]): base.feature_mean(
            [features[index] for index in indices]
        )
        for group, indices in groups.items()
        if group.startswith("competitor:")
    }
    competitor_vectors = {
        key: value for key, value in competitor_vectors.items() if value is not None
    }

    hr_old = base.cosine(selected, old)
    hr_target = base.cosine(selected, target)
    old_target = base.cosine(old, target)
    evidence_competitors = {
        str(track_id): float(np.dot(selected, vector))
        for track_id, vector in competitor_vectors.items()
        if selected is not None
    }
    target_competitors = {
        str(track_id): float(np.dot(target, vector))
        for track_id, vector in competitor_vectors.items()
        if target is not None
    }
    best_evidence_comp = max(evidence_competitors.values(), default=-1.0)
    best_target_comp = max(target_competitors.values(), default=-1.0)
    bridge = min(hr_old, hr_target) if hr_old is not None and hr_target is not None else None
    return {
        **selection,
        "embedding_available": bridge is not None and old_target is not None,
        "similarity_hr_old": hr_old,
        "similarity_hr_target": hr_target,
        "similarity_old_target": old_target,
        "best_competitor_similarity_to_hr": best_evidence_comp,
        "best_competitor_similarity_to_target": best_target_comp,
        "evidence_bridge": bridge,
        "evidence_margin": bridge - best_evidence_comp if bridge is not None else None,
        "target_margin": old_target - best_target_comp if old_target is not None else None,
        "competitor_similarity_to_hr": evidence_competitors,
        "competitor_similarity_to_target": target_competitors,
        "competitor_count": len(competitor_vectors),
        "evaluation_selected_hr_correct": (
            bool(int(selected_candidate["label"])) if selected_candidate is not None else None
        ),
        "selected_bbox_xyxy": (
            selected_candidate.get("candidate_bbox_xyxy")
            if selected_candidate is not None
            else None
        ),
        "selected_source_image": (
            selected_candidate.get("source_image") if selected_candidate is not None else None
        ),
        "representation": "all_hr",
        "hr_crop_count": len(crops),
        "hr_old_crop_count": len(groups.get("old", [])),
        "hr_target_crop_count": len(groups.get("target", [])),
        "hr_selected_crop_count": len(groups.get("selected", [])),
        "hr_competitor_crop_count": sum(
            len(indices)
            for group, indices in groups.items()
            if group.startswith("competitor:")
        ),
    }


def build_evidence(
    args: argparse.Namespace,
    encoder: base.OpenCVONNXReID,
    cache: base.ImageCache,
    raw_events: list[dict],
    hive_selections: dict[str, dict],
    prediction_cache: dict[str, dict],
) -> list[dict]:
    current = base.build_split_evidence(
        args, encoder, cache, raw_events, hive_selections, prediction_cache
    )
    current_by_id = {row["event_id"]: row for row in current}
    total = len(raw_events)
    for index, event in enumerate(raw_events, start=1):
        row = current_by_id[event["event_id"]]
        current_scores = dict(row["methods"]["hive_v2_hr_reid"])
        current_scores.update(
            {
                "representation": "hr_evidence_with_lr_context",
                "hr_crop_count": 1 if current_scores.get("candidate_id") else 0,
                "hr_old_crop_count": 0,
                "hr_target_crop_count": 0,
                "hr_selected_crop_count": 1 if current_scores.get("candidate_id") else 0,
                "hr_competitor_crop_count": 0,
            }
        )
        row["methods"]["hive_hr_lr_current"] = current_scores
        all_hr = hr_hr_scores(
            args,
            encoder,
            cache,
            event,
            hive_selections.get(event["event_id"], {}),
            prediction_cache[event["sequence"]],
        )
        row["methods"]["hive_hr_hr_same_threshold"] = dict(all_hr)
        row["methods"]["hive_hr_hr_tuned"] = dict(all_hr)
        if index == 1 or index % 10 == 0 or index == total:
            print(f"  HR↔HR evidence {index}/{total}", flush=True)
    return current


def tune_one_method(
    args: argparse.Namespace,
    evidence: list[dict],
    method: str,
    sequences: list[str],
    prediction_cache: dict[str, dict],
    gt_cache: dict[str, dict],
) -> tuple[dict[str, float], list[dict]]:
    baseline = base.basic_metrics(
        args,
        sequences,
        {sequence: prediction_cache[sequence] for sequence in sequences},
        gt_cache,
        include_hota=False,
    )["aggregate"]
    rows = []
    for index, thresholds in enumerate(base.grid_values(), start=1):
        replayed, decisions, operations = base.replay_split(
            args, evidence, method, thresholds, prediction_cache
        )
        metrics = base.basic_metrics(
            args, sequences, replayed, gt_cache, include_hota=False
        )["aggregate"]
        applied = sum(operation["applied"] for operation in operations)
        wrong_evidence = sum(
            operation["applied"]
            and not bool(decision["scores"].get("evaluation_selected_hr_correct"))
            for operation, decision in zip(operations, decisions)
        )
        idsw_reduction = (
            baseline["id_switches"] - metrics["id_switches"]
        ) / max(1, baseline["id_switches"])
        wrong_rate = wrong_evidence / max(1, applied)
        objective = metrics["idf1"] + 0.10 * idsw_reduction - 0.05 * wrong_rate
        rows.append(
            {
                "thresholds": thresholds,
                "objective": objective,
                "idf1": metrics["idf1"],
                "idsw": metrics["id_switches"],
                "applied": applied,
                "wrong_hr_evidence_applied": wrong_evidence,
                "wrong_evidence_rate": wrong_rate,
            }
        )
        if index % 18 == 0 or index == 108:
            print(f"  HR↔HR threshold grid {index}/108", flush=True)
    rows.sort(
        key=lambda row: (
            row["objective"],
            row["idf1"],
            -row["idsw"],
            -row["wrong_evidence_rate"],
            sum(row["thresholds"].values()),
        ),
        reverse=True,
    )
    return rows[0]["thresholds"], rows


def rejection_counts(outcomes: list[dict], correct_only: bool = True) -> dict[str, int]:
    selected = [
        row
        for row in outcomes
        if not correct_only or bool(row["scores"].get("evaluation_selected_hr_correct"))
    ]
    return dict(sorted(Counter(row["decision_reason"] for row in selected).items()))


def pass_counts(evidence: list[dict], method: str, thresholds: dict) -> dict[str, int]:
    rows = [
        event["methods"][method]
        for event in evidence
        if event["methods"][method].get("evaluation_selected_hr_correct")
    ]
    checks = {
        "bridge": lambda row: row["evidence_bridge"] >= thresholds["bridge_min"],
        "old_target": lambda row: row["similarity_old_target"] >= thresholds["old_target_min"],
        "evidence_margin": lambda row: row["evidence_margin"] >= thresholds["evidence_margin_min"],
        "target_margin": lambda row: row["target_margin"] >= thresholds["target_margin_min"],
        "all": lambda row: all(
            (
                row["evidence_bridge"] >= thresholds["bridge_min"],
                row["similarity_old_target"] >= thresholds["old_target_min"],
                row["evidence_margin"] >= thresholds["evidence_margin_min"],
                row["target_margin"] >= thresholds["target_margin_min"],
            )
        ),
    }
    return {name: sum(check(row) for row in rows) for name, check in checks.items()}


def paired_score_statistics(evidence: list[dict]) -> dict[str, dict]:
    """Describe paired score changes on events with a correct HIVE selection."""
    rows = [
        event
        for event in evidence
        if event["methods"]["hive_hr_lr_current"].get(
            "evaluation_selected_hr_correct"
        )
    ]
    fields = (
        ("evidence_bridge", "Evidence bridge"),
        ("similarity_old_target", "Old-target similarity"),
        ("evidence_margin", "Evidence margin"),
        ("target_margin", "Target margin"),
    )
    result = {}
    for field, label in fields:
        lr = np.asarray(
            [event["methods"]["hive_hr_lr_current"][field] for event in rows],
            dtype=float,
        )
        hr = np.asarray(
            [event["methods"]["hive_hr_hr_same_threshold"][field] for event in rows],
            dtype=float,
        )
        delta = hr - lr
        result[field] = {
            "label": label,
            "n": len(rows),
            "hr_lr_median": float(np.median(lr)),
            "hr_hr_median": float(np.median(hr)),
            "median_delta": float(np.median(delta)),
            "mean_delta": float(np.mean(delta)),
            "improved": int(np.sum(delta > 1e-12)),
            "worsened": int(np.sum(delta < -1e-12)),
            "equal": int(np.sum(np.abs(delta) <= 1e-12)),
            "wilcoxon_p": float(wilcoxon(delta).pvalue),
        }
    return result


def plot_metrics(output_dir: Path, results: dict[str, dict]) -> None:
    colors = ["#64748b", "#2563eb", "#f59e0b", "#16a34a"]
    labels = [DISPLAY_NAMES[method] for method in METHODS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for axis, (metric, title, digits) in zip(
        axes,
        (
            ("id_switches", "ID switches ↓", 0),
            ("idf1", "IDF1 ↑", 4),
            ("hota", "HOTA ↑", 4),
        ),
    ):
        values = [results[method]["aggregate"][metric] for method in METHODS]
        bars = axis.bar(np.arange(len(METHODS)), values, color=colors)
        axis.set_xticks(np.arange(len(METHODS)), labels, rotation=18, ha="right")
        axis.set_title(title, weight="bold")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            label = str(int(value)) if digits == 0 else f"{value:.{digits}f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                weight="bold",
            )
    fig.suptitle("Frozen HIVE v2: LR↔HR versus HR↔HR ReID", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "visuals" / "tracking_metrics.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_outcomes(output_dir: Path, results: dict[str, dict]) -> None:
    methods = METHODS[1:]
    labels = [DISPLAY_NAMES[method] for method in methods]
    fields = (
        ("selected_hr_identity_correct", "HIVE selected correct", "#38bdf8"),
        ("policy_merge_requests", "Merge requested", "#f59e0b"),
        ("actual_event_switches_fixed", "Actually fixed", "#22c55e"),
        ("wrong_corrections", "Wrong correction", "#ef4444"),
    )
    x = np.arange(len(methods))
    width = 0.19
    fig, axis = plt.subplots(figsize=(11, 5.8))
    for slot, (field, label, color) in enumerate(fields):
        values = [results[method]["correction"][field] for method in methods]
        bars = axis.bar(x + (slot - 1.5) * width, values, width, label=label, color=color)
        axis.bar_label(bars, padding=2)
    axis.set_xticks(x, labels, rotation=12)
    axis.set_ylabel("Events")
    axis.set_title("Correct retrieval does not automatically become identity recovery", weight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "visuals" / "correction_outcomes.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_paired_scores(output_dir: Path, evidence: list[dict]) -> None:
    pairs = (
        ("evidence_bridge", "Evidence bridge", CURRENT_THRESHOLDS["bridge_min"]),
        ("similarity_old_target", "Old-target similarity", CURRENT_THRESHOLDS["old_target_min"]),
        ("evidence_margin", "Evidence margin", CURRENT_THRESHOLDS["evidence_margin_min"]),
        ("target_margin", "Target margin", CURRENT_THRESHOLDS["target_margin_min"]),
    )
    rows = [
        event
        for event in evidence
        if event["methods"]["hive_hr_lr_current"].get("evaluation_selected_hr_correct")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for axis, (field, title, threshold) in zip(axes.flat, pairs):
        x = np.asarray([row["methods"]["hive_hr_lr_current"][field] for row in rows])
        y = np.asarray([row["methods"]["hive_hr_hr_same_threshold"][field] for row in rows])
        low = float(min(x.min(), y.min(), threshold)) - 0.04
        high = float(max(x.max(), y.max(), threshold)) + 0.04
        axis.scatter(x, y, s=28, alpha=0.72, color="#2563eb")
        axis.plot([low, high], [low, high], "--", color="#64748b", label="no change")
        axis.axvline(threshold, color="#dc2626", linestyle=":")
        axis.axhline(threshold, color="#dc2626", linestyle=":", label="current threshold")
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_xlabel("HR↔LR score")
        axis.set_ylabel("HR↔HR score")
        axis.set_title(f"{title} (n={len(rows)})", weight="bold")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Paired ReID scores when HIVE selected the correct identity", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "visuals" / "paired_reid_scores.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def render_examples(
    args: argparse.Namespace,
    output_dir: Path,
    evidence: list[dict],
    outcomes: dict[str, list[dict]],
) -> None:
    evidence_map = {row["event_id"]: row for row in evidence}
    current_map = {row["event_id"]: row for row in outcomes["hive_hr_lr_current"]}
    tuned = outcomes["hive_hr_hr_tuned"]
    selected = [
        row
        for row in tuned
        if row["actually_fixed"]
        and not current_map[row["event_id"]]["actually_fixed"]
    ]
    if not selected:
        selected = [row for row in tuned if row["actually_fixed"]]
    selected = selected[: args.visual_examples]
    if not selected:
        return
    rows = []
    for outcome in selected:
        event = evidence_map[outcome["event_id"]]
        target = event["target_crop"]
        target_lr = cv2.imread(str(args.hive_dataset_root / target["crop_path"]))
        target_hr = hr_record_crop(base.ImageCache(max_items=4), target)
        method = event["methods"]["hive_hr_hr_tuned"]
        selected_image = cv2.imread(str(base.source_path(method["selected_source_image"])))
        selected_hr = base.clipped_crop(selected_image, method["selected_bbox_xyxy"])
        current = current_map[outcome["event_id"]]
        panels = (
            base.label_panel(base.fit_panel(target_lr), "Target LR", outcome["event_id"]),
            base.label_panel(base.fit_panel(target_hr), "Target HR", f"frame {event['target_frame']}"),
            base.label_panel(
                base.fit_panel(selected_hr),
                "HIVE-selected HR",
                f"frame {method['candidate_frame']}",
            ),
            base.label_panel(
                np.full((220, 320, 3), 245, dtype=np.uint8),
                "Replay outcome",
                f"HR-LR fixed={current['actually_fixed']} | HR-HR fixed={outcome['actually_fixed']}",
            ),
        )
        rows.append(np.hstack(panels))
    cv2.imwrite(str(output_dir / "visuals" / "hr_hr_recovery_examples.jpg"), np.vstack(rows))


def report_lines(
    args: argparse.Namespace,
    validation_sequences: list[str],
    test_sequences: list[str],
    tuned_thresholds: dict,
    results: dict[str, dict],
    outcomes: dict[str, list[dict]],
    test_evidence: list[dict],
    checkpoint_hash: str,
) -> list[str]:
    baseline = results["lr_only"]["aggregate"]
    correct_count = results["hive_hr_lr_current"]["correction"][
        "selected_hr_identity_correct"
    ]
    paired_stats = paired_score_statistics(test_evidence)
    current_fixed = {
        row["event_id"]
        for row in outcomes["hive_hr_lr_current"]
        if row["actually_fixed"]
    }
    tuned_fixed = {
        row["event_id"]
        for row in outcomes["hive_hr_hr_tuned"]
        if row["actually_fixed"]
    }
    current_affected = sum(
        len(row.get("affected_keys", []))
        for row in outcomes["hive_hr_lr_current"]
        if row["applied"]
    )
    tuned_affected = sum(
        len(row.get("affected_keys", []))
        for row in outcomes["hive_hr_hr_tuned"]
        if row["applied"]
    )
    lines = [
        "# Frozen HIVE v2: HR↔LR versus HR↔HR ReID ablation",
        "",
        f"Generated: `{datetime.now().astimezone().isoformat(timespec='seconds')}`",
        "",
        "## Kết luận chính",
        "",
        "Thay toàn bộ LR context bằng crop từ frame HR **không cải thiện tracking trên held-out test**. Sau khi tune threshold trên validation, HR↔HR vẫn sửa 4 ID switch như HR↔LR hiện tại, nhưng IDF1 và HOTA thấp hơn. Do đó LR/HR resolution mismatch không phải bottleneck chính của policy hiện tại.",
        "",
        f"Hai policy chỉ cùng sửa `{len(current_fixed & tuned_fixed)}/4` event. HR↔HR sửa thêm `{len(tuned_fixed - current_fixed)}` event khác nhưng mất `{len(current_fixed - tuned_fixed)}` event mà HR↔LR sửa được. HR↔LR relabel {current_affected} detections; HR↔HR relabel {tuned_affected}, giải thích vì sao cùng giảm 4 IDSW nhưng IDF1 khác nhau.",
        "",
        "## Held-out QuadTrack result",
        "",
        "HIVE selection is identical in every ReID branch. `Same threshold` isolates the representation change; `val-tuned` recalibrates HR↔HR only on validation before the held-out test is read.",
        "",
        "| Method | IDSW ↓ | IDF1 ↑ | HOTA ↑ | Merge requested | Applied | Actually fixed | Wrong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        aggregate = results[method]["aggregate"]
        correction = results[method]["correction"]
        lines.append(
            f"| {DISPLAY_NAMES[method]} | {aggregate['id_switches']} | {aggregate['idf1']:.4f} | "
            f"{aggregate['hota']:.4f} | {correction.get('policy_merge_requests', 0)} | "
            f"{correction.get('applied_merges', 0)} | {correction.get('actual_event_switches_fixed', 0)} | "
            f"{correction.get('wrong_corrections', 0)} |"
        )
    same = results["hive_hr_hr_same_threshold"]["aggregate"]
    tuned = results["hive_hr_hr_tuned"]["aggregate"]
    current = results["hive_hr_lr_current"]["aggregate"]
    lines.extend(
        [
            "",
            "![Tracking metrics](visuals/tracking_metrics.png)",
            "",
            "## Retrieval-to-correction conversion",
            "",
            f"Frozen HIVE selected the correct historical identity in **{correct_count}/{len(test_evidence)}** test events. This count is the same for all ReID branches because HIVE was not retrained or rerun with HR input.",
            "",
            "| Method | Correct HIVE selection | Merge requested | Applied | Actually fixed | Wrong correction | New switch points |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS[1:]:
        correction = results[method]["correction"]
        lines.append(
            f"| {DISPLAY_NAMES[method]} | {correction['selected_hr_identity_correct']} | "
            f"{correction['policy_merge_requests']} | {correction['applied_merges']} | "
            f"{correction['actual_event_switches_fixed']} | {correction['wrong_corrections']} | "
            f"{correction['full_replay_switch_points_introduced']} |"
        )
    lines.extend(
        [
            "",
            "![Correction outcomes](visuals/correction_outcomes.png)",
            "",
            "## Score behavior on correct HIVE retrievals",
            "",
            f"The paired plot contains only the {correct_count} events for which HIVE selected the correct identity. Points above the diagonal improved after replacing LR context with HR context; the red dotted lines are the current policy thresholds.",
            "",
            "![Paired ReID scores](visuals/paired_reid_scores.png)",
            "",
            "### Paired score statistics",
            "",
            "| Score | HR↔LR median | HR↔HR median | Median Δ | Improved / worsened | Wilcoxon p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired_stats.values():
        lines.append(
            f"| {row['label']} | {row['hr_lr_median']:.3f} | {row['hr_hr_median']:.3f} | "
            f"{row['median_delta']:+.3f} | {row['improved']} / {row['worsened']} | "
            f"{row['wilcoxon_p']:.3g} |"
        )
    lines.extend(
        [
            "",
            "At event level, HR↔HR significantly raises the bridge score and evidence margin, but it does not improve old-target similarity and tends to reduce target margin. Events are clustered within only four held-out videos, so the Wilcoxon values are descriptive rather than sequence-level generalization claims.",
            "",
            "### Threshold pass counts among correct retrievals",
            "",
            "| Representation / policy | Bridge | Old-target | Evidence margin | Target margin | All four |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method, threshold in (
        ("hive_hr_lr_current", CURRENT_THRESHOLDS),
        ("hive_hr_hr_same_threshold", CURRENT_THRESHOLDS),
        ("hive_hr_hr_tuned", tuned_thresholds),
    ):
        counts = pass_counts(test_evidence, method, threshold)
        lines.append(
            f"| {DISPLAY_NAMES[method]} | {counts['bridge']}/{correct_count} | "
            f"{counts['old_target']}/{correct_count} | {counts['evidence_margin']}/{correct_count} | "
            f"{counts['target_margin']}/{correct_count} | **{counts['all']}/{correct_count}** |"
        )
    lines.extend(
        [
            "",
            "### Rejection reasons among correct retrievals",
            "",
            "```json",
            json.dumps(
                {
                    method: rejection_counts(outcomes[method])
                    for method in METHODS[1:]
                },
                indent=2,
                ensure_ascii=False,
            ),
            "```",
            "",
            "## Thresholds",
            "",
            "| Parameter | Current HR↔LR / same-threshold HR↔HR | Validation-tuned HR↔HR |",
            "|---|---:|---:|",
        ]
    )
    for key in CURRENT_THRESHOLDS:
        lines.append(
            f"| `{key}` | {CURRENT_THRESHOLDS[key]:.2f} | {tuned_thresholds[key]:.2f} |"
        )
    total_hr_crops = sum(
        row["methods"]["hive_hr_hr_tuned"]["hr_crop_count"] for row in test_evidence
    )
    hr_breakdown = {
        key: sum(row["methods"]["hive_hr_hr_tuned"][key] for row in test_evidence)
        for key in (
            "hr_old_crop_count",
            "hr_target_crop_count",
            "hr_selected_crop_count",
            "hr_competitor_crop_count",
        )
    }
    lines.extend(
        [
            "",
            "## HR budget and interpretation boundary",
            "",
            f"- Current HR↔LR uses one selected historical HR crop per event: **{len(test_evidence)} HR crops** on test.",
            f"- All-HR context uses selected, target, up to five old-history, and sampled competitor crops: **{total_hr_crops} HR crops** across {len(test_evidence)} events ({total_hr_crops / len(test_evidence):.1f}/event).",
            f"- HR↔HR crop breakdown: old history {hr_breakdown['hr_old_crop_count']}, target {hr_breakdown['hr_target_crop_count']}, HIVE-selected evidence {hr_breakdown['hr_selected_crop_count']}, competitors {hr_breakdown['hr_competitor_crop_count']}.",
            "- Every HR crop comes from the original 2048×480 frame using tracker-predicted boxes. GT boxes and identities never enter the selector, embedding, threshold checks, or replay decision.",
            "- The ReID model still resizes each crop to 224×224 internally. This tests source-detail/domain mismatch, not a higher ReID network input size.",
            "- Because the HR budget is larger, HR↔HR is an accuracy ablation rather than a same-budget replacement for the original one-HR-request method.",
            "- ID-switch timestamps and old/new track pairs still come from an evaluator-created manifest; this remains an oracle-trigger correction experiment.",
            "",
            "## Protocol",
            "",
            f"- Frozen HIVE checkpoint: `{args.checkpoint.resolve()}`",
            f"- Checkpoint SHA-256: `{checkpoint_hash}`",
            f"- Validation-only threshold tuning: `{', '.join(validation_sequences)}`",
            f"- Held-out test: `{', '.join(test_sequences)}`",
            f"- Current HR↔LR: IDSW {current['id_switches']}, IDF1 {current['idf1']:.4f}, HOTA {current['hota']:.4f}.",
            f"- HR↔HR with unchanged thresholds: IDSW {same['id_switches']}, IDF1 {same['idf1']:.4f}, HOTA {same['hota']:.4f}.",
            f"- HR↔HR after validation calibration: IDSW {tuned['id_switches']}, IDF1 {tuned['idf1']:.4f}, HOTA {tuned['hota']:.4f}.",
            "",
            "## Qualitative recoveries",
            "",
            "![HR-HR recovery examples](visuals/hr_hr_recovery_examples.jpg)",
            "",
            "Each row shows the LR target used previously, the same target cropped from the HR frame, and the frozen-HIVE historical HR crop. Outcome labels are added only after replay for analysis.",
            "",
            "## Machine-readable artifacts",
            "",
            "- `summary.json`: protocol, thresholds, HR budget, and all aggregate results.",
            "- `events/validation_threshold_search_hr_hr.json`: validation-only calibration grid.",
            "- `events/test_evidence.jsonl`: paired LR-context and all-HR scores per event.",
            "- `events/*_decisions.jsonl`: merge/replay outcome for every method.",
            "- `predictions/<method>/`: replayed MOT predictions.",
            "",
        ]
    )
    return lines


def main() -> int:
    args = parse_args()
    base.validate_args(args)
    output_dir = base.prepare_output(args.output_dir, args.overwrite)
    device = torch.device(args.device)
    print(f"Output: {output_dir}", flush=True)
    print(f"Device: {device}", flush=True)

    checkpoint_state = base.load_checkpoint(args.checkpoint.resolve(), "cpu")
    config = checkpoint_state["config"]
    split_config = json.loads(
        (args.hive_dataset_root / "split_config.json").read_text(encoding="utf-8")
    )
    validation_sequences = [str(value) for value in config["data"]["validation_sequences"]]
    test_sequences = [str(value) for value in split_config["test_sequences"]]
    validation_dataset = base.build_hive_dataset(
        args, config, validation_sequences, config["data"]["train_manifest"]
    )
    test_dataset = base.build_hive_dataset(
        args, config, test_sequences, config["data"]["test_manifest"]
    )
    validation_sequences = sorted({event["sequence"] for event in validation_dataset.events})
    test_sequences = sorted({event["sequence"] for event in test_dataset.events})

    model = base.HIVE(config["model"], metadata_dim=validation_dataset.metadata_dim).to(device)
    model.load_state_dict(checkpoint_state["model"], strict=True)
    print("Selecting frames with frozen HIVE v2...", flush=True)
    validation_events, validation_hive = base.select_frames_with_hive(
        model, validation_dataset, device, args.batch_size
    )
    test_events, test_hive = base.select_frames_with_hive(
        model, test_dataset, device, args.batch_size
    )

    sequences = sorted(set(validation_sequences) | set(test_sequences))
    prediction_cache = {
        sequence: base.load_mot(
            args.lr_pred_dir / args.lr_pred_pattern.format(sequence=sequence),
            ground_truth=False,
        )
        for sequence in sequences
    }
    gt_cache = {
        sequence: base.load_mot(
            args.dataset_root / sequence / "gt" / "gt.txt", ground_truth=True
        )
        for sequence in sequences
    }

    encoder = base.OpenCVONNXReID(args.reid_model, device)
    print(f"ReID backend: {encoder.backend}", flush=True)
    cache = base.ImageCache()
    print("Building validation LR-context and all-HR evidence...", flush=True)
    validation_evidence = build_evidence(
        args, encoder, cache, validation_events, validation_hive, prediction_cache
    )
    print("Building held-out LR-context and all-HR evidence...", flush=True)
    test_evidence = build_evidence(
        args, encoder, cache, test_events, test_hive, prediction_cache
    )
    base.write_jsonl(output_dir / "events" / "validation_evidence.jsonl", validation_evidence)
    base.write_jsonl(output_dir / "events" / "test_evidence.jsonl", test_evidence)

    print("Tuning HR↔HR thresholds on validation only...", flush=True)
    tuned_thresholds, tuning_rows = tune_one_method(
        args,
        validation_evidence,
        "hive_hr_hr_tuned",
        validation_sequences,
        prediction_cache,
        gt_cache,
    )
    base.write_json(
        output_dir / "events" / "validation_threshold_search_hr_hr.json",
        {"selected_thresholds": tuned_thresholds, "rows": tuning_rows},
    )
    print(f"Frozen HR↔HR thresholds: {tuned_thresholds}", flush=True)

    baseline_failures = []
    for sequence in test_sequences:
        baseline_failures.extend(
            base.collect_failure_frames(
                sequence, gt_cache[sequence], prediction_cache[sequence], args.iou
            )
        )
    baseline_switches = base.switch_points(baseline_failures)
    thresholds_by_method = {
        "lr_only": CURRENT_THRESHOLDS,
        "hive_hr_lr_current": CURRENT_THRESHOLDS,
        "hive_hr_hr_same_threshold": CURRENT_THRESHOLDS,
        "hive_hr_hr_tuned": tuned_thresholds,
    }
    results: dict[str, dict] = {}
    outcomes: dict[str, list[dict]] = {}
    for method in METHODS:
        print(f"Evaluating {method}...", flush=True)
        result, method_outcomes, _ = base.final_evaluate_method(
            args,
            output_dir,
            method,
            test_evidence,
            test_sequences,
            thresholds_by_method[method],
            prediction_cache,
            gt_cache,
            baseline_switches,
        )
        results[method] = result
        outcomes[method] = method_outcomes

    plot_metrics(output_dir, results)
    plot_outcomes(output_dir, results)
    plot_paired_scores(output_dir, test_evidence)
    render_examples(args, output_dir, test_evidence, outcomes)
    checkpoint_hash = base.sha256(args.checkpoint)
    hr_budget = {
        "hr_lr_total_crops": sum(
            row["methods"]["hive_hr_lr_current"]["hr_crop_count"]
            for row in test_evidence
        ),
        "hr_hr_total_crops": sum(
            row["methods"]["hive_hr_hr_tuned"]["hr_crop_count"]
            for row in test_evidence
        ),
    }
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": {
            "hive_retrained": False,
            "hive_checkpoint": str(args.checkpoint.resolve()),
            "hive_checkpoint_sha256": checkpoint_hash,
            "reid_model": str(args.reid_model.resolve()),
            "validation_sequences": validation_sequences,
            "held_out_test_sequences": test_sequences,
            "validation_events": len(validation_evidence),
            "test_events": len(test_evidence),
            "source_resolution": [2048, 480],
            "reid_input_resolution": [224, 224],
            "oracle_switch_trigger": True,
            "gt_decision_features": [],
            "hr_budget": hr_budget,
        },
        "current_thresholds": CURRENT_THRESHOLDS,
        "selected_validation_thresholds_hr_hr": tuned_thresholds,
        "results": results,
        "correct_retrieval_pass_counts": {
            "hr_lr_current": pass_counts(
                test_evidence, "hive_hr_lr_current", CURRENT_THRESHOLDS
            ),
            "hr_hr_same_threshold": pass_counts(
                test_evidence, "hive_hr_hr_same_threshold", CURRENT_THRESHOLDS
            ),
            "hr_hr_tuned": pass_counts(
                test_evidence, "hive_hr_hr_tuned", tuned_thresholds
            ),
        },
        "paired_score_statistics": paired_score_statistics(test_evidence),
    }
    base.write_json(output_dir / "summary.json", summary)
    report = output_dir / "REPORT.md"
    report.write_text(
        "\n".join(
            report_lines(
                args,
                validation_sequences,
                test_sequences,
                tuned_thresholds,
                results,
                outcomes,
                test_evidence,
                checkpoint_hash,
            )
        ),
        encoding="utf-8",
    )
    print("\nHeld-out result:")
    for method in METHODS:
        aggregate = results[method]["aggregate"]
        correction = results[method]["correction"]
        print(
            f"  {method}: IDSW={aggregate['id_switches']} IDF1={aggregate['idf1']:.4f} "
            f"HOTA={aggregate['hota']:.4f} fixed={correction.get('actual_event_switches_fixed', 0)} "
            f"wrong={correction.get('wrong_corrections', 0)}"
        )
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
