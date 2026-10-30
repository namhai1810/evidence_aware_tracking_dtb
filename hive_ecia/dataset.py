"""Embedding-only ECIA dataset and video-disjoint cache format.

The module deliberately stores frozen ReID embeddings rather than pixels.  GT
identity metadata is used once to create ``target_class`` and ``sample_kind``;
it is never returned as an inference feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


SAMPLE_KINDS = ("positive", "negative", "hard_negative", "no_merge")
SAMPLE_KIND_TO_ID = {name: index for index, name in enumerate(SAMPLE_KINDS)}


def normalize_embedding(value: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2 normalize one frozen ReID vector and reject malformed inputs."""

    array = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(array).all() or norm < eps:
        raise ValueError("Embedding must be finite and have non-zero norm")
    return array / norm


@dataclass(frozen=True)
class ECIASample:
    """One identity-recovery event before competitor padding."""

    event_id: str
    sequence: str
    split: str
    old: np.ndarray
    target: np.ndarray
    hr: np.ndarray | None
    competitors: np.ndarray  # [K, D]
    competitor_track_ids: tuple[int, ...]
    target_class: int  # 0=old, 1..K=competitor, K+1=abstain.
    sample_kind: str

    @property
    def hr_available(self) -> bool:
        return self.hr is not None

    @property
    def abstain_index(self) -> int:
        return len(self.competitor_track_ids) + 1

    @property
    def merge_target(self) -> bool:
        return self.target_class == 0


def build_identity_sample(
    *,
    event_id: str,
    sequence: str,
    split: str,
    old_embedding: np.ndarray,
    target_embedding: np.ndarray,
    hr_embedding: np.ndarray | None,
    competitor_embeddings: Sequence[np.ndarray],
    old_identity: int,
    target_identity: int,
    hr_identity: int | None,
    competitor_identities: Sequence[int],
    competitor_track_ids: Sequence[int],
    hard_negative: bool = False,
) -> ECIASample:
    """Convert GT identity metadata into a classification target.

    Identity IDs are consumed only here while materializing an offline cache.
    The resulting model sample contains embeddings, masks, and a class index.

    Sample taxonomy
    ---------------
    * ``positive``: target belongs to old and HR supports the same identity.
    * ``hard_negative``: misleading HR for an old target, or a mined difficult
      competitor explicitly marked by the event miner.
    * ``negative``: target belongs to one of the competitor hypotheses.
    * ``no_merge``: target matches neither old nor any available competitor.
    """

    if len(competitor_embeddings) != len(competitor_identities):
        raise ValueError("competitor embeddings and identity IDs must have equal length")
    if len(competitor_embeddings) != len(competitor_track_ids):
        raise ValueError("competitor embeddings and track IDs must have equal length")

    if target_identity == old_identity:
        target_class = 0
        misleading_hr = hr_identity is not None and hr_identity != target_identity
        sample_kind = "hard_negative" if hard_negative or misleading_hr else "positive"
    elif target_identity in competitor_identities:
        target_class = competitor_identities.index(target_identity) + 1
        sample_kind = "hard_negative" if hard_negative else "negative"
    else:
        target_class = len(competitor_identities) + 1
        sample_kind = "no_merge"

    competitors = np.stack(
        [normalize_embedding(value) for value in competitor_embeddings], axis=0
    ) if competitor_embeddings else np.zeros((0, np.asarray(old_embedding).size), dtype=np.float32)
    return ECIASample(
        event_id=str(event_id),
        sequence=str(sequence),
        split=str(split),
        old=normalize_embedding(old_embedding),
        target=normalize_embedding(target_embedding),
        hr=normalize_embedding(hr_embedding) if hr_embedding is not None else None,
        competitors=competitors,
        competitor_track_ids=tuple(int(value) for value in competitor_track_ids),
        target_class=int(target_class),
        sample_kind=sample_kind,
    )


def audit_video_disjoint(
    train_sequences: Iterable[str],
    validation_sequences: Iterable[str],
    test_sequences: Iterable[str],
) -> None:
    """Fail fast if any video appears in more than one split."""

    splits = {
        "train": set(map(str, train_sequences)),
        "validation": set(map(str, validation_sequences)),
        "test": set(map(str, test_sequences)),
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = splits[left] & splits[right]
        if overlap:
            raise ValueError(f"Video leakage between {left} and {right}: {sorted(overlap)}")


def audit_named_splits(**splits: Iterable[str]) -> None:
    """Audit any number of named sequence splits pairwise."""

    normalized = {name: set(map(str, values)) for name, values in splits.items()}
    names = list(normalized)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = normalized[left] & normalized[right]
            if overlap:
                raise ValueError(
                    f"Video leakage between {left} and {right}: {sorted(overlap)}"
                )


def assert_sample_coverage(samples: Sequence[ECIASample], required: Sequence[str]) -> None:
    """Ensure the training cache actually contains every requested sample type."""

    present = {sample.sample_kind for sample in samples}
    missing = set(required) - present
    if missing:
        raise ValueError(f"Missing required ECIA sample kinds: {sorted(missing)}")


class ECIADataset(Dataset):
    """In-memory embedding dataset; no image/ReID work occurs during training."""

    def __init__(self, samples: Sequence[ECIASample]) -> None:
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("ECIADataset cannot be empty")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ECIASample:
        return self.samples[index]


def collate_ecia(samples: Sequence[ECIASample]) -> dict[str, Tensor | list[str]]:
    """Pad variable competitors and remap each local abstain slot to batch K+1."""

    if not samples:
        raise ValueError("Cannot collate an empty ECIA batch")
    batch_size = len(samples)
    embedding_dim = samples[0].old.size
    # Retain one fully masked padding slot when every event has K=0. This keeps
    # max reductions in the model well-defined without inventing a competitor.
    max_competitors = max(1, max(len(sample.competitor_track_ids) for sample in samples))

    old = torch.zeros(batch_size, embedding_dim, dtype=torch.float32)
    target = torch.zeros_like(old)
    hr = torch.zeros_like(old)
    competitors = torch.zeros(batch_size, max_competitors, embedding_dim, dtype=torch.float32)
    competitor_mask = torch.zeros(batch_size, max_competitors, dtype=torch.bool)
    hr_available = torch.zeros(batch_size, dtype=torch.bool)
    target_class = torch.zeros(batch_size, dtype=torch.long)
    merge_target = torch.zeros(batch_size, dtype=torch.float32)
    sample_kind = torch.zeros(batch_size, dtype=torch.long)
    sample_weight = torch.ones(batch_size, dtype=torch.float32)

    for row, sample in enumerate(samples):
        if sample.old.size != embedding_dim:
            raise ValueError("All embeddings in a batch must have the same dimension")
        count = sample.competitors.shape[0]
        old[row] = torch.from_numpy(sample.old)
        target[row] = torch.from_numpy(sample.target)
        if sample.hr is not None:
            hr[row] = torch.from_numpy(sample.hr)
            hr_available[row] = True
        if count:
            competitors[row, :count] = torch.from_numpy(sample.competitors)
            competitor_mask[row, :count] = True
        is_abstain = sample.target_class == sample.abstain_index
        target_class[row] = max_competitors + 1 if is_abstain else sample.target_class
        merge_target[row] = float(sample.merge_target)
        sample_kind[row] = SAMPLE_KIND_TO_ID[sample.sample_kind]

    return {
        "old": old,
        "target": target,
        "hr": hr,
        "competitors": competitors,
        "competitor_mask": competitor_mask,
        "hr_available": hr_available,
        "target_class": target_class,
        "merge_target": merge_target,
        "sample_kind": sample_kind,
        "sample_weight": sample_weight,
        "event_id": [sample.event_id for sample in samples],
        "sequence": [sample.sequence for sample in samples],
    }


def save_embedding_cache(
    path: Path,
    samples: Sequence[ECIASample],
    max_competitors: int = 12,
) -> None:
    """Write a compact, non-pickle NPZ cache suitable for frozen embeddings."""

    if not samples:
        raise ValueError("Cannot save an empty embedding cache")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(samples)
    embedding_dim = samples[0].old.size
    old = np.zeros((count, embedding_dim), dtype=np.float32)
    target = np.zeros_like(old)
    hr = np.zeros_like(old)
    competitors = np.zeros((count, max_competitors, embedding_dim), dtype=np.float32)
    mask = np.zeros((count, max_competitors), dtype=np.bool_)
    hr_available = np.zeros(count, dtype=np.bool_)
    target_class = np.zeros(count, dtype=np.int64)
    sample_kind = np.zeros(count, dtype=np.int64)
    track_ids = np.full((count, max_competitors), -1, dtype=np.int64)

    for index, sample in enumerate(samples):
        competitor_count = sample.competitors.shape[0]
        if competitor_count > max_competitors:
            raise ValueError(
                f"Event {sample.event_id} has {competitor_count} competitors; max={max_competitors}"
            )
        old[index], target[index] = sample.old, sample.target
        if sample.hr is not None:
            hr[index] = sample.hr
            hr_available[index] = True
        competitors[index, :competitor_count] = sample.competitors
        mask[index, :competitor_count] = True
        track_ids[index, :competitor_count] = sample.competitor_track_ids
        target_class[index] = max_competitors + 1 if sample.target_class == sample.abstain_index else sample.target_class
        sample_kind[index] = SAMPLE_KIND_TO_ID[sample.sample_kind]

    np.savez_compressed(
        path,
        old=old,
        target=target,
        hr=hr,
        competitors=competitors,
        competitor_mask=mask,
        hr_available=hr_available,
        target_class=target_class,
        sample_kind=sample_kind,
        competitor_track_ids=track_ids,
        event_id=np.asarray([sample.event_id for sample in samples]),
        sequence=np.asarray([sample.sequence for sample in samples]),
        split=np.asarray([sample.split for sample in samples]),
    )


def load_embedding_cache(path: Path) -> ECIADataset:
    """Load the non-pickle NPZ cache and reconstruct variable-length samples."""

    # ``NpzFile.__getitem__`` decompresses an array on every access. Accessing
    # ``payload["old"]`` and friends inside the sample loop therefore expanded
    # the same 30 MB cache thousands of times. Materialize each member once so
    # parallel ECIA jobs spend time training rather than repeatedly inflating
    # identical ZIP entries.
    with np.load(Path(path), allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    samples = []
    max_competitors = payload["competitors"].shape[1]
    for index in range(payload["old"].shape[0]):
        mask = payload["competitor_mask"][index].astype(bool)
        competitor_count = int(mask.sum())
        stored_class = int(payload["target_class"][index])
        local_abstain = competitor_count + 1
        target_class = local_abstain if stored_class == max_competitors + 1 else stored_class
        kind = SAMPLE_KINDS[int(payload["sample_kind"][index])]
        samples.append(
            ECIASample(
                event_id=str(payload["event_id"][index]),
                sequence=str(payload["sequence"][index]),
                split=str(payload["split"][index]),
                old=payload["old"][index].astype(np.float32),
                target=payload["target"][index].astype(np.float32),
                hr=(payload["hr"][index].astype(np.float32) if payload["hr_available"][index] else None),
                competitors=payload["competitors"][index, mask].astype(np.float32),
                competitor_track_ids=tuple(
                    int(value) for value in payload["competitor_track_ids"][index, mask]
                ),
                target_class=target_class,
                sample_kind=kind,
            )
        )
    return ECIADataset(samples)
