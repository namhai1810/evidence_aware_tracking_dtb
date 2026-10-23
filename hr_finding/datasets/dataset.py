"""Event-grouped HIVE dataset with strict LR-only online model inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from .preprocessing import (
    FrameCache,
    image_to_tensor,
    load_materialized_crop,
    load_mot_tracks,
    match_lr_detection,
    reconstruct_lr_crop,
    select_track_history,
    temporal_metadata,
)


class HIVERankingDataset(Dataset):
    """One item is one ID-switch event and its complete candidate ranking group."""

    def __init__(
        self,
        dataset_root: Path,
        manifest: str | Path,
        project_root: Path,
        data_config: dict,
        include_sequences: Iterable[str] | None = None,
        exclude_sequences: Iterable[str] | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.project_root = Path(project_root).resolve()
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = self.dataset_root / manifest_path
        include = set(include_sequences or [])
        exclude = set(exclude_sequences or [])
        if include & exclude:
            raise ValueError("A sequence cannot be both included and excluded")
        self.events = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            event = json.loads(line)
            sequence = event["sequence"]
            if include and sequence not in include:
                continue
            if sequence in exclude:
                continue
            self.events.append(event)
        self.events.sort(key=lambda row: (row["sequence"], row["target_frame"], row["event_id"]))

        self.lowres_hw = tuple(int(value) for value in data_config.get("lowres_size", [120, 512]))
        self.crop_hw = tuple(int(value) for value in data_config.get("crop_size", [128, 64]))
        self.max_history = int(data_config.get("max_history", 5))
        self.max_candidates = int(data_config.get("max_candidates", 5))
        self.window_frames = int(data_config.get("candidate_window_frames", 150))
        self.include_competitor = bool(data_config.get("include_competitor_history", True))
        self.competitor_history_frames = int(data_config.get("competitor_history_frames", 30))
        self.candidate_box_mode = str(data_config.get("candidate_box_mode", "manifest_bbox"))
        self.lr_match_iou = float(data_config.get("lr_match_iou", 0.5))
        self.teacher_margin_field = data_config.get("teacher_margin_field")
        if self.candidate_box_mode not in {"manifest_bbox", "lr_tracker_iou"}:
            raise ValueError("candidate_box_mode must be manifest_bbox or lr_tracker_iou")
        self.frame_cache = FrameCache(int(data_config.get("frame_cache_size", 32)))

        split_config = json.loads(
            (self.dataset_root / "split_config.json").read_text(encoding="utf-8")
        )
        tracking_root = Path(split_config["source_files"]["lowres_predictions"])
        if not tracking_root.is_absolute():
            tracking_root = self.project_root / tracking_root
        self.tracking_root = tracking_root
        original_frames_root = Path(split_config["source_files"]["original_frames"])
        if not original_frames_root.is_absolute():
            original_frames_root = self.project_root / original_frames_root
        self.original_frames_root = original_frames_root
        self.tracking_pattern = "{sequence}_yolo11l_botsort_reid_pre512x120.txt"
        self._tracking_cache: dict[str, dict] = {}
        self.metadata_dim = 3

    def __len__(self) -> int:
        return len(self.events)

    def _source_path(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else self.project_root / path

    def _tracking(self, sequence: str) -> dict:
        if sequence not in self._tracking_cache:
            path = self.tracking_root / self.tracking_pattern.format(sequence=sequence)
            self._tracking_cache[sequence] = load_mot_tracks(path)
        return self._tracking_cache[sequence]

    def _empty_images(self, count: int) -> torch.Tensor:
        return torch.zeros((count, 3, self.crop_hw[0], self.crop_hw[1]), dtype=torch.float32)

    def __getitem__(self, index: int) -> dict:
        event = self.events[index]
        sequence = event["sequence"]
        tracking = self._tracking(sequence) if (
            self.include_competitor or self.candidate_box_mode == "lr_tracker_iou"
        ) else None

        old_images = self._empty_images(self.max_history)
        old_mask = torch.zeros(self.max_history, dtype=torch.bool)
        for slot, query in enumerate(event["query_crops"][: self.max_history]):
            crop_path = self.dataset_root / query["crop_path"]
            old_images[slot] = load_materialized_crop(crop_path, self.crop_hw)
            old_mask[slot] = True

        target = torch.zeros((3, self.crop_hw[0], self.crop_hw[1]), dtype=torch.float32)
        target_available = event.get("target_crop") is not None
        if target_available:
            target = load_materialized_crop(
                self.dataset_root / event["target_crop"]["crop_path"], self.crop_hw
            )

        candidate_images = self._empty_images(self.max_candidates)
        candidate_mask = torch.zeros(self.max_candidates, dtype=torch.bool)
        labels = torch.full((self.max_candidates,), -1.0, dtype=torch.float32)
        metadata = torch.zeros((self.max_candidates, self.metadata_dim), dtype=torch.float32)
        teacher_targets = torch.zeros(self.max_candidates, dtype=torch.float32)
        teacher_mask = torch.zeros(self.max_candidates, dtype=torch.bool)

        for slot, candidate in enumerate(event["candidates"][: self.max_candidates]):
            source = self._source_path(candidate["source_image"])
            # v1 stores HR-proposal boxes; v2 uses deployable LR-tracker proposal
            # boxes expressed in original-panorama coordinates.
            box = candidate.get("candidate_bbox_xyxy", candidate.get("hr_bbox_xyxy"))
            if box is None:
                raise KeyError(
                    f"Candidate {candidate.get('candidate_id')} has no usable bounding box"
                )
            if self.candidate_box_mode == "lr_tracker_iou":
                assert tracking is not None
                matched, _ = match_lr_detection(
                    tracking.get(int(candidate["candidate_frame"]), []), box, self.lr_match_iou
                )
                if matched is None:
                    continue
                box = matched[1]
            crop = reconstruct_lr_crop(source, box, self.lowres_hw, self.frame_cache)
            candidate_images[slot] = image_to_tensor(crop, self.crop_hw)
            candidate_mask[slot] = True
            labels[slot] = float(candidate["label"])
            metadata[slot] = temporal_metadata(float(candidate["seconds_before"]))
            if self.teacher_margin_field and self.teacher_margin_field in candidate:
                teacher_targets[slot] = float(candidate[self.teacher_margin_field])
                teacher_mask[slot] = True

        competitor_images = self._empty_images(self.max_history)
        competitor_mask = torch.zeros(self.max_history, dtype=torch.bool)
        if self.include_competitor and tracking is not None:
            competitor_rows = select_track_history(
                tracking,
                int(event["lr_new_pred_id"]),
                int(event["target_frame"]),
                self.competitor_history_frames,
                self.max_history,
            )
            for slot, (frame, box, _) in enumerate(competitor_rows):
                source = self.original_frames_root / sequence / "img1" / f"{frame:06d}.jpg"
                crop = reconstruct_lr_crop(source, box, self.lowres_hw, self.frame_cache)
                competitor_images[slot] = image_to_tensor(crop, self.crop_hw)
                competitor_mask[slot] = True

        return {
            "event_id": event["event_id"],
            "sequence": sequence,
            "old_crops": old_images,
            "old_mask": old_mask,
            "target_crop": target,
            "target_available": torch.tensor(target_available, dtype=torch.bool),
            "candidate_crops": candidate_images,
            "candidate_mask": candidate_mask,
            "metadata": metadata,
            "labels": labels,
            "teacher_targets": teacher_targets,
            "teacher_mask": teacher_mask,
            "competitor_crops": competitor_images,
            "competitor_mask": competitor_mask,
        }

    def class_counts(self) -> tuple[int, int]:
        positives = negatives = 0
        for event in self.events:
            for candidate in event["candidates"]:
                if int(candidate["label"]) == 1:
                    positives += 1
                else:
                    negatives += 1
        return positives, negatives


def collate_event_batch(items: list[dict]) -> dict:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    batch: dict = {
        "event_id": [item["event_id"] for item in items],
        "sequence": [item["sequence"] for item in items],
    }
    for key in items[0]:
        if key not in batch:
            batch[key] = torch.stack([item[key] for item in items])
    return batch
