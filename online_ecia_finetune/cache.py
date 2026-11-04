"""Non-pickle cache format for causal online candidate evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_raw_candidate_cache(path: Path, records: list[dict], sequence: str, max_competitors: int = 12) -> None:
    if not records:
        raise ValueError(f"No online candidates collected for {sequence}")
    count = len(records)
    dimension = int(np.asarray(records[0]["old"]).size)
    old = np.zeros((count, dimension), np.float32)
    target = np.zeros_like(old)
    hr = np.zeros_like(old)
    competitors = np.zeros((count, max_competitors, dimension), np.float32)
    mask = np.zeros((count, max_competitors), bool)
    competitor_frames = np.full((count, max_competitors), -1, np.int64)
    competitor_raw_ids = np.full((count, max_competitors), -1, np.int64)
    metadata = []
    for index, row in enumerate(records):
        values = np.asarray(row["competitors"], np.float32).reshape(-1, dimension)
        if len(values) > max_competitors:
            raise ValueError(f"Candidate {index} exceeds max competitors={max_competitors}")
        frames = list(map(int, row["competitor_ref_frames"]))
        raw_ids = list(map(int, row["competitor_ref_raw_ids"]))
        if len(values) != len(frames) or len(values) != len(raw_ids):
            raise ValueError("Competitor embeddings and identity references are misaligned")
        old[index] = np.asarray(row["old"], np.float32)
        target[index] = np.asarray(row["target"], np.float32)
        hr[index] = np.asarray(row["hr"], np.float32)
        competitors[index, : len(values)] = values
        mask[index, : len(values)] = True
        competitor_frames[index, : len(values)] = frames
        competitor_raw_ids[index, : len(values)] = raw_ids
        metadata.append(
            json.dumps(
                {
                    "event_id": f"{sequence}:{row['frame']}:{row['old_token']}:{row['new_token']}",
                    "sequence": str(sequence),
                    "frame": int(row["frame"]),
                    "pair_kind": row["pair_kind"],
                    "old_token": row["old_token"],
                    "new_token": row["new_token"],
                    "old_ref_frame": int(row["old_ref_frame"]),
                    "old_ref_raw_id": int(row["old_ref_raw_id"]),
                    "target_ref_frame": int(row["target_ref_frame"]),
                    "target_ref_raw_id": int(row["target_ref_raw_id"]),
                    "hr_ref_frame": int(row["hr_ref_frame"]),
                    "hr_ref_raw_id": int(row["hr_ref_raw_id"]),
                    "hr_available": bool(row["hr_available"]),
                    "safety_feasible": bool(row["safety_feasible"]),
                    "safety_reason": row["safety_reason"],
                    "hive": row["hive"],
                    "base": row["base"],
                },
                sort_keys=True,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        old=old,
        target=target,
        hr=hr,
        competitors=competitors,
        competitor_mask=mask,
        competitor_ref_frames=competitor_frames,
        competitor_ref_raw_ids=competitor_raw_ids,
        metadata=np.asarray(metadata),
    )


def load_raw_candidate_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
