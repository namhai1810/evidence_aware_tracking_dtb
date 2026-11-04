"""Join GT to frozen online candidates after collection and write ECIA labels.

GT is used only in this offline stage.  The resulting cache contains class
targets, frozen embeddings and masks; no GT identity is an inference feature.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from evaluate_mot import load_mot
from hive_ecia.dataset import ECIASample, normalize_embedding, save_embedding_cache
from rank_highres_retrieval_frames import build_match_index

from .cache import load_raw_candidate_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-run", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--split", choices=("train", "validation", "calibration"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--forbidden-sequences", nargs="*", default=[])
    parser.add_argument(
        "--separate-safety-infeasible",
        action="store_true",
        help=(
            "Store safety-blocked candidates as a distinct sample kind while "
            "retaining the Abstain target. Existing v1 behavior is unchanged "
            "unless this flag is supplied."
        ),
    )
    return parser.parse_args()


def pred_to_gt_index(gt: dict, predictions: dict, iou: float) -> dict[tuple[int, int], int]:
    matched, _ = build_match_index(gt, predictions, iou)
    return {
        (int(frame), int(row["pred_id"])): int(gt_id)
        for frame, rows in matched.items()
        for gt_id, row in rows.items()
    }


def main() -> int:
    args = parse_args()
    sequences = list(map(str, args.sequences))
    overlap = set(sequences) & set(map(str, args.forbidden_sequences))
    if overlap:
        raise ValueError(f"Leakage: {args.split} contains forbidden sequences {sorted(overlap)}")
    collection = args.collection_run.resolve()
    manifest = json.loads((collection / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("mode") != "base_guided_collect":
        raise ValueError("Labels require a GT-free base_guided_collect run")
    if set(sequences) - set(map(str, manifest.get("sequences", []))):
        raise ValueError("Requested sequence is absent from the collection manifest")

    samples: list[ECIASample] = []
    counts = Counter()
    per_sequence = {}
    for sequence in sequences:
        gt = load_mot(args.dataset_root / sequence / "gt" / "gt.txt", ground_truth=True)
        predictions = load_mot(
            collection / "raw_mot" / f"{sequence}.txt", ground_truth=False
        )
        identity = pred_to_gt_index(gt, predictions, args.iou)
        payload = load_raw_candidate_cache(collection / "candidate_cache" / f"{sequence}.npz")
        sequence_counts = Counter()
        for index, encoded_metadata in enumerate(payload["metadata"]):
            row = json.loads(str(encoded_metadata))
            mask = payload["competitor_mask"][index].astype(bool)
            old_identity = identity.get((row["old_ref_frame"], row["old_ref_raw_id"]))
            target_identity = identity.get((row["target_ref_frame"], row["target_ref_raw_id"]))
            hr_identity = identity.get((row["hr_ref_frame"], row["hr_ref_raw_id"]))
            competitor_identities = [
                identity.get((int(frame), int(raw_id)))
                for frame, raw_id in zip(
                    payload["competitor_ref_frames"][index, mask],
                    payload["competitor_ref_raw_ids"][index, mask],
                )
            ]
            competitor_count = int(mask.sum())
            abstain_index = competitor_count + 1
            safety_feasible = bool(row["safety_feasible"])
            if safety_feasible and target_identity is not None and target_identity == old_identity:
                target_class = 0
                if hr_identity == target_identity:
                    sample_kind = "positive"
                elif hr_identity is not None and hr_identity in competitor_identities:
                    sample_kind = "hard_negative"
                else:
                    sample_kind = "negative"
            elif safety_feasible and target_identity is not None and target_identity in competitor_identities:
                target_class = competitor_identities.index(target_identity) + 1
                sample_kind = "hard_negative"
            else:
                # Safety-infeasible Active-to-Active events are explicitly
                # supervised as non-executable, even when GT identities match.
                target_class = abstain_index
                sample_kind = (
                    "safety_infeasible"
                    if args.separate_safety_infeasible and not safety_feasible
                    else "no_merge"
                )

            hr = (
                normalize_embedding(payload["hr"][index])
                if bool(row["hr_available"])
                else None
            )
            samples.append(
                ECIASample(
                    event_id=row["event_id"],
                    sequence=sequence,
                    split=args.split,
                    old=normalize_embedding(payload["old"][index]),
                    target=normalize_embedding(payload["target"][index]),
                    hr=hr,
                    competitors=payload["competitors"][index, mask].astype(np.float32),
                    competitor_track_ids=tuple(
                        int(value) for value in payload["competitor_ref_raw_ids"][index, mask]
                    ),
                    target_class=target_class,
                    sample_kind=sample_kind,
                )
            )
            sequence_counts[sample_kind] += 1
            safety_counter = (
                "safety_infeasible_events"
                if args.separate_safety_infeasible
                else "safety_infeasible"
            )
            sequence_counts[safety_counter] += int(not safety_feasible)
            sequence_counts["hr_available"] += int(hr is not None)
        counts.update(sequence_counts)
        per_sequence[sequence] = dict(sequence_counts)

    save_embedding_cache(args.output, samples)
    audit = {
        "split": args.split,
        "sequences": sequences,
        "forbidden_sequences": sorted(map(str, args.forbidden_sequences)),
        "dataset_root": str(args.dataset_root.resolve()),
        "collection_manifest": str((collection / "run_manifest.json").resolve()),
        "collection_mode": "base_guided_collect_no_gt",
        "gt_usage": "offline_label_generation_only",
        "gt_identity_stored_as_feature": False,
        "safety_infeasible_policy": "force_abstain_non_executable",
        "safety_infeasible_separate_sample_kind": bool(
            args.separate_safety_infeasible
        ),
        "events": len(samples),
        "counts": dict(counts),
        "per_sequence": per_sequence,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"{args.split} labeled cache: {args.output} ({len(samples)} events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
