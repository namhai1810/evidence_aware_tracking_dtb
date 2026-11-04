"""Objectives for executable online identity recovery."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class OnlineLossOutput:
    total: Tensor
    base_classification: Tensor
    fused_classification: Tensor
    ranking: Tensor
    wrong_merge: Tensor
    hr_preservation: Tensor

    def detached(self) -> dict[str, float]:
        return {name: float(value.detach()) for name, value in self.__dict__.items()}


class OnlineAwareLoss(nn.Module):
    def __init__(
        self,
        base_weight: float = 0.5,
        fused_weight: float = 1.0,
        ranking_weight: float = 0.25,
        wrong_merge_weight: float = 2.0,
        hr_preservation_weight: float = 0.75,
        ranking_margin: float = 0.20,
        false_merge_multiplier: float = 2.0,
    ) -> None:
        super().__init__()
        self.base_weight = base_weight
        self.fused_weight = fused_weight
        self.ranking_weight = ranking_weight
        self.wrong_merge_weight = wrong_merge_weight
        self.hr_preservation_weight = hr_preservation_weight
        self.ranking_margin = ranking_margin
        self.false_merge_multiplier = float(false_merge_multiplier)
        if self.false_merge_multiplier < 1.0:
            raise ValueError("false_merge_multiplier must be >= 1")

    @staticmethod
    def weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    def forward(
        self,
        output,
        target: Tensor,
        merge_target: Tensor,
        weights: Tensor | None = None,
    ) -> OnlineLossOutput:
        # Balanced-v2 passes no weights: the controlled sampler is the sole
        # class-balancing mechanism. Optional weights retain v1 compatibility.
        if weights is None:
            weights = torch.ones_like(merge_target, dtype=output.decision_logits.dtype)
        else:
            weights = weights.to(output.decision_logits.dtype)
        base_rows = F.cross_entropy(output.base_decision_logits, target, reduction="none")
        fused_rows = F.cross_entropy(output.decision_logits, target, reduction="none")
        base_loss = self.weighted_mean(base_rows, weights)
        fused_loss = self.weighted_mean(fused_rows, weights)

        correct = output.decision_logits.gather(1, target[:, None]).squeeze(1)
        alternatives = torch.ones_like(output.decision_logits, dtype=torch.bool)
        alternatives.scatter_(1, target[:, None], False)
        strongest = output.decision_logits.masked_fill(
            ~alternatives, torch.finfo(output.decision_logits.dtype).min
        ).max(dim=1).values
        ranking = self.weighted_mean(
            F.relu(self.ranking_margin - correct + strongest), weights
        )

        old_probability = torch.softmax(output.decision_logits, dim=-1)[:, 0]
        wrong_merge_rows = F.binary_cross_entropy(
            old_probability.clamp(1e-6, 1 - 1e-6), merge_target, reduction="none"
        )
        # Non-merge examples receive extra weight because a false identity
        # transfer can damage a complete future trajectory.
        false_merge_weights = weights * torch.where(
            merge_target.bool(),
            torch.ones_like(weights),
            torch.full_like(weights, self.false_merge_multiplier),
        )
        wrong_merge = self.weighted_mean(wrong_merge_rows, false_merge_weights)

        base_prob = torch.softmax(output.base_decision_logits, dim=-1)
        fused_prob = torch.softmax(output.decision_logits, dim=-1)
        base_correct = base_prob.argmax(dim=-1) == target
        preservation_rows = F.relu(
            base_prob.gather(1, target[:, None]).squeeze(1)
            - fused_prob.gather(1, target[:, None]).squeeze(1)
        )
        preservation = self.weighted_mean(
            preservation_rows, weights * (1.0 + base_correct.to(weights.dtype))
        )
        total = (
            self.base_weight * base_loss
            + self.fused_weight * fused_loss
            + self.ranking_weight * ranking
            + self.wrong_merge_weight * wrong_merge
            + self.hr_preservation_weight * preservation
        )
        return OnlineLossOutput(total, base_loss, fused_loss, ranking, wrong_merge, preservation)
