#!/usr/bin/env python3
"""Fast loader/model/loss smoke test. This performs no optimizer step or training."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import HIVERankingDataset, collate_event_batch  # noqa: E402
from losses import HIVEMultiTaskLoss  # noqa: E402
from models import HIVE  # noqa: E402
from train import make_datasets  # noqa: E402
from utils.config import load_config, resolve_path  # noqa: E402
from utils.metrics import MetricAccumulator  # noqa: E402


def main() -> None:
    config = load_config(ROOT / "configs" / "hive.yaml")
    dataset_root = resolve_path(config["data"]["dataset_root"], ROOT)
    project_root = resolve_path(config["data"]["project_root"], ROOT)
    learning, validation, learning_sequences, validation_sequences = make_datasets(
        config, dataset_root, project_root
    )
    assert len(learning) == 757
    assert len(validation) == 76
    assert set(learning_sequences).isdisjoint(validation_sequences)
    dataset = HIVERankingDataset(
        dataset_root,
        config["data"]["train_manifest"],
        project_root,
        config["data"],
        include_sequences=["0015"],
    )
    batch = collate_event_batch([dataset[0], dataset[1]])
    for variant in sorted(HIVE.SUPPORTED_VARIANTS):
        model_config = deepcopy(config["model"])
        model_config["variant"] = variant
        model = HIVE(model_config, metadata_dim=dataset.metadata_dim).eval()
        with torch.no_grad():
            logits = model(batch)
            losses = HIVEMultiTaskLoss(config["loss"], positive_weight=1.0)(logits, batch)
        assert logits.shape == batch["labels"].shape
        assert torch.isfinite(losses["total"])
        assert torch.all(logits[~batch["candidate_mask"]] == -1e4)
    metrics = MetricAccumulator(0.5, 0.5)
    metrics.update(logits, batch["labels"], batch["candidate_mask"], batch["event_id"])
    computed = metrics.compute()
    for key in ("success_at_1", "mrr", "ndcg_at_5", "oracle_coverage", "request_rate"):
        assert key in computed
    print(
        f"PASS train={len(learning)} val={len(validation)} sample_sequence_events={len(dataset)} "
        f"batch={len(batch['event_id'])} "
        f"candidates={int(batch['candidate_mask'].sum())} variants={len(HIVE.SUPPORTED_VARIANTS)}"
    )


if __name__ == "__main__":
    main()
