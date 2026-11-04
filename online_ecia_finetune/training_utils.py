"""Balanced sampling and decision-level metrics for online ECIA.

The sampler is the *only* class-balancing mechanism in balanced-v2. Losses
receive unit weights, preventing the former sampler-plus-loss double weighting.
"""

from __future__ import annotations

from collections import Counter
import math
import random
from typing import Iterable, Iterator, Sequence

import torch
from torch import Tensor
from torch.utils.data import Sampler

from hive_ecia.dataset import ECIASample


DECISION_CLASSES = ("merge_old", "competitor", "abstain")
SAMPLING_GROUPS = ("merge_old", "competitor", "abstain", "safety_infeasible")


def sample_group(sample: ECIASample) -> str:
    """Map an event to a mutually exclusive training-sampling group."""

    if sample.sample_kind == "safety_infeasible":
        return "safety_infeasible"
    if sample.target_class == 0:
        return "merge_old"
    if sample.target_class == sample.abstain_index:
        return "abstain"
    return "competitor"


def _quota_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    if set(fractions) != set(SAMPLING_GROUPS):
        raise ValueError(f"sampling fractions must define {SAMPLING_GROUPS}")
    if any(float(value) < 0 for value in fractions.values()):
        raise ValueError("sampling fractions cannot be negative")
    denominator = sum(float(value) for value in fractions.values())
    if not math.isclose(denominator, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"sampling fractions must sum to 1, got {denominator}")
    exact = {name: total * float(fractions[name]) for name in SAMPLING_GROUPS}
    counts = {name: int(math.floor(value)) for name, value in exact.items()}
    remainder = total - sum(counts.values())
    order = sorted(SAMPLING_GROUPS, key=lambda name: exact[name] - counts[name], reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


class ControlledIdentitySampler(Sampler[int]):
    """Sample fixed decision quotas with a capped Old hard-negative share.

    Sampling is with replacement because the positive Old pool is much smaller
    than the genuine Abstain pool. Competitors are deliberately retained as
    hard negatives; misleading-HR Old samples are capped independently.
    """

    def __init__(
        self,
        samples: Sequence[ECIASample],
        *,
        fractions: dict[str, float],
        epoch_size: int,
        old_hard_negative_fraction: float,
        seed: int,
    ) -> None:
        if epoch_size <= 0:
            raise ValueError("epoch_size must be positive")
        if not 0.0 <= old_hard_negative_fraction <= 1.0:
            raise ValueError("old_hard_negative_fraction must be in [0,1]")
        self.samples = list(samples)
        self.epoch_size = int(epoch_size)
        self.fractions = {key: float(value) for key, value in fractions.items()}
        self.quotas = _quota_counts(self.epoch_size, self.fractions)
        self.old_hard_negative_fraction = float(old_hard_negative_fraction)
        self.seed = int(seed)
        self.epoch = 0
        self.pools: dict[str, list[int]] = {name: [] for name in SAMPLING_GROUPS}
        self.old_regular: list[int] = []
        self.old_hard: list[int] = []
        for index, sample in enumerate(self.samples):
            group = sample_group(sample)
            self.pools[group].append(index)
            if group == "merge_old":
                destination = self.old_hard if sample.sample_kind == "hard_negative" else self.old_regular
                destination.append(index)
        missing = [name for name, count in self.quotas.items() if count and not self.pools[name]]
        if missing:
            raise ValueError(f"cannot sample empty required pools: {missing}")
        if self.quotas["merge_old"] and not self.old_regular:
            raise ValueError("merge_old requires at least one non-hard example")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _draw(rng: random.Random, pool: Sequence[int], count: int) -> list[int]:
        if count <= 0:
            return []
        if not pool:
            raise ValueError("attempted to draw from an empty pool")
        return [pool[rng.randrange(len(pool))] for _ in range(count)]

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        indices: list[int] = []
        old_count = self.quotas["merge_old"]
        old_hard_count = min(
            old_count,
            int(round(old_count * self.old_hard_negative_fraction)),
        ) if self.old_hard else 0
        indices.extend(self._draw(rng, self.old_hard, old_hard_count))
        indices.extend(self._draw(rng, self.old_regular, old_count - old_hard_count))
        for group in ("competitor", "abstain", "safety_infeasible"):
            indices.extend(self._draw(rng, self.pools[group], self.quotas[group]))
        if len(indices) != self.epoch_size:
            raise RuntimeError("controlled sampler produced an invalid epoch size")
        rng.shuffle(indices)
        self.epoch += 1
        return iter(indices)

    def __len__(self) -> int:
        return self.epoch_size

    def audit(self) -> dict:
        old_hard = int(round(self.quotas["merge_old"] * self.old_hard_negative_fraction))
        old_hard = min(old_hard, self.quotas["merge_old"]) if self.old_hard else 0
        hard_total = self.quotas["competitor"] + old_hard
        return {
            "strategy": "controlled_identity_sampler_only",
            "epoch_size": self.epoch_size,
            "quotas": dict(self.quotas),
            "source_pool_counts": {name: len(values) for name, values in self.pools.items()},
            "old_hard_negative_quota": old_hard,
            "hard_negative_quota": hard_total,
            "hard_negative_fraction": hard_total / self.epoch_size,
            "loss_class_weights": "disabled_unit_weights",
        }


def decision_categories(indices: Tensor, abstain_index: int) -> Tensor:
    """Map padded local class indices to Old/Competitor/Abstain IDs."""

    result = torch.ones_like(indices, dtype=torch.long)
    result[indices == 0] = 0
    result[indices == int(abstain_index)] = 2
    return result


def confusion_update(confusion: Tensor, target: Tensor, prediction: Tensor, abstain_index: int) -> None:
    target_group = decision_categories(target.detach().cpu(), abstain_index)
    prediction_group = decision_categories(prediction.detach().cpu(), abstain_index)
    for truth, predicted in zip(target_group.tolist(), prediction_group.tolist()):
        confusion[int(truth), int(predicted)] += 1


def metrics_from_confusion(confusion: Tensor) -> dict[str, float | int | dict]:
    """Compute per-class PR/F1 and anti-collapse metrics.

    Rows are targets and columns are predictions.
    """

    matrix = confusion.to(torch.float64)
    per_class = {}
    f1_values = []
    for index, name in enumerate(DECISION_CLASSES):
        tp = float(matrix[index, index])
        predicted = float(matrix[:, index].sum())
        support = float(matrix[index, :].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            f1_values.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
            "predicted": int(predicted),
        }
    total = int(matrix.sum())
    non_merge = float(matrix[1:, :].sum())
    wrong_merge = int(matrix[1:, 0].sum())
    return {
        "confusion": confusion.tolist(),
        "per_class": per_class,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "merge_precision": per_class["merge_old"]["precision"],
        "merge_recall": per_class["merge_old"]["recall"],
        "wrong_merge": wrong_merge,
        "wrong_merge_rate": wrong_merge / non_merge if non_merge else 0.0,
        "abstain_prediction_fraction": (
            float(matrix[:, 2].sum()) / total if total else 0.0
        ),
        "accuracy": float(matrix.diag().sum()) / total if total else 0.0,
        "examples": total,
    }


def checkpoint_score(metrics: dict, config: dict) -> tuple[bool, float, list[str]]:
    """Return eligibility and validation-only selection score."""

    limits = config["eligibility"]
    reasons = []
    if metrics["merge_recall"] < float(limits["min_merge_recall"]):
        reasons.append("merge_recall_below_minimum")
    if metrics["macro_f1"] < float(limits["min_macro_f1"]):
        reasons.append("macro_f1_below_minimum")
    if metrics["wrong_merge_rate"] > float(limits["max_wrong_merge_rate"]):
        reasons.append("wrong_merge_rate_above_maximum")
    if metrics["abstain_prediction_fraction"] > float(limits["max_abstain_prediction_fraction"]):
        reasons.append("abstain_collapse")
    weights = config["score_weights"]
    score = (
        float(weights["macro_f1"]) * metrics["macro_f1"]
        + float(weights["merge_recall"]) * metrics["merge_recall"]
        + float(weights["merge_precision"]) * metrics["merge_precision"]
        - float(weights["wrong_merge_rate"]) * metrics["wrong_merge_rate"]
    )
    return not reasons, score, reasons
