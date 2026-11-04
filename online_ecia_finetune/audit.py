"""Leakage and artifact-boundary audit for online-aware ECIA fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from hive_ecia.dataset import audit_named_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--collection-run", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    splits = {
        "train": list(map(str, data["train_sequences"])),
        "validation": list(map(str, data["validation_sequences"])),
        "calibration": list(map(str, data["calibration_sequences"])),
        "test": list(map(str, data["test_sequences"])),
    }
    audit_named_splits(**splits)
    if splits["test"] != ["0016", "0017", "0025", "0031"]:
        raise ValueError("Unexpected QuadTrack test boundary")
    manifest = json.loads(
        (args.collection_run / "run_manifest.json").read_text(encoding="utf-8")
    )
    collected = set(map(str, manifest.get("sequences", [])))
    allowed = set(splits["train"] + splits["validation"] + splits["calibration"])
    if manifest.get("mode") != "base_guided_collect" or collected != allowed:
        raise ValueError("Collector did not use exactly train+validation+calibration")
    if collected & set(splits["test"]):
        raise ValueError("Test leakage in collection run")

    audits = {}
    for split in ("train", "validation", "calibration"):
        path = args.audit_dir / f"{split}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if set(map(str, row["sequences"])) != set(splits[split]):
            raise ValueError(f"{split} label audit sequence mismatch")
        if row.get("gt_usage") != "offline_label_generation_only":
            raise ValueError(f"Unexpected GT policy for {split}")
        if row.get("gt_identity_stored_as_feature") is not False:
            raise ValueError(f"GT identity feature leakage in {split}")
        audits[split] = row
    safety_separated = all(
        bool(audits[split].get("safety_infeasible_separate_sample_kind"))
        for split in audits
    )
    if config.get("sampling", {}).get("strategy") == "controlled_identity" and not safety_separated:
        raise ValueError("Balanced training requires separate safety-infeasible labels")
    payload = {
        "status": "passed",
        "video_disjoint": True,
        "collector_gt_access": False,
        "gt_usage": "offline labels for train/validation/calibration only",
        "test_sequences_accessed": False,
        "dtb_sequences_accessed": False,
        "splits": splits,
        "collection_sequences": sorted(collected),
        "labeled_events": {split: audits[split]["events"] for split in audits},
        "safety_infeasible_policy": "force_abstain_non_executable",
        "safety_infeasible_separate_sample_kind": safety_separated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Leakage audit passed: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
