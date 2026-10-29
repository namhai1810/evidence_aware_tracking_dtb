"""Training objectives for Lightweight ECIA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import ECIAOutput


@dataclass
class ECIALossOutput:
    total: Tensor
    base_classification: Tensor
    fused_classification: Tensor
    margin_ranking: Tensor
    wrong_merge: Tensor

    def detached(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "base_classification": float(self.base_classification.detach()),
            "fused_classification": float(self.fused_classification.detach()),
            "margin_ranking": float(self.margin_ranking.detach()),
            "wrong_merge": float(self.wrong_merge.detach()),
        }


class ECIALoss(nn.Module):
    """Classification + ranking + explicit wrong-merge objective.

    Base classification is defined only when the target belongs to old or a
    listed competitor.  Fused classification includes learned abstention for
    every event.  The wrong-merge term puts extra cost on assigning old ID to a
    negative, hard-negative, or no-merge sample.
    """

    def __init__(
        self,
        base_classification_weight: float = 0.5,
        fused_classification_weight: float = 1.0,
        margin_ranking_weight: float = 0.25,
        wrong_merge_weight: float = 0.5,
        ranking_margin: float = 0.20,
    ) -> None:
        super().__init__()
        self.base_classification_weight = float(base_classification_weight)
        self.fused_classification_weight = float(fused_classification_weight)
        self.margin_ranking_weight = float(margin_ranking_weight)
        self.wrong_merge_weight = float(wrong_merge_weight)
        self.ranking_margin = float(ranking_margin)

    @staticmethod
    def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    def forward(
        self,
        output: ECIAOutput,
        target_class: Tensor,
        merge_target: Tensor,
        sample_weight: Tensor | None = None,
    ) -> ECIALossOutput:
        batch_size = target_class.shape[0]
        if sample_weight is None:
            sample_weight = torch.ones(batch_size, device=target_class.device)
        sample_weight = sample_weight.to(output.decision_logits.dtype)

        # Base head has no abstain slot. Exclude no-merge targets from this
        # auxiliary loss rather than forcing them onto an incorrect identity.
        hypothesis_count = output.base_scores.shape[1]
        base_valid = target_class < hypothesis_count
        if base_valid.any():
            masked_base = output.base_scores.masked_fill(
                ~output.hypothesis_mask, torch.finfo(output.base_scores.dtype).min
            )
            base_rows = F.cross_entropy(
                masked_base[base_valid], target_class[base_valid], reduction="none"
            )
            base_loss = self._weighted_mean(base_rows, sample_weight[base_valid])
        else:
            base_loss = output.base_scores.sum() * 0.0

        # The fused softmax includes old, valid competitors, and abstain.
        fused_rows = F.cross_entropy(
            output.decision_logits, target_class, reduction="none"
        )
        fused_loss = self._weighted_mean(fused_rows, sample_weight)

        # Margin ranking operates on the correct class versus the strongest
        # alternative, including abstain. Invalid competitors already carry a
        # very negative decision logit from the model.
        positive = output.decision_logits.gather(1, target_class[:, None]).squeeze(1)
        alternative_mask = torch.ones_like(output.decision_logits, dtype=torch.bool)
        alternative_mask.scatter_(1, target_class[:, None], False)
        strongest_alternative = output.decision_logits.masked_fill(
            ~alternative_mask, torch.finfo(output.decision_logits.dtype).min
        ).max(dim=1).values
        ranking_rows = F.relu(self.ranking_margin - positive + strongest_alternative)
        ranking_loss = self._weighted_mean(ranking_rows, sample_weight)

        # Explicitly supervise the operational merge decision. High old-class
        # probability on a non-old target is a wrong-merge risk.
        old_probability = torch.softmax(output.decision_logits, dim=-1)[:, 0]
        wrong_merge_rows = F.binary_cross_entropy(
            old_probability.clamp(1e-6, 1.0 - 1e-6),
            merge_target.to(old_probability.dtype),
            reduction="none",
        )
        wrong_merge_loss = self._weighted_mean(wrong_merge_rows, sample_weight)

        total = (
            self.base_classification_weight * base_loss
            + self.fused_classification_weight * fused_loss
            + self.margin_ranking_weight * ranking_loss
            + self.wrong_merge_weight * wrong_merge_loss
        )
        return ECIALossOutput(
            total=total,
            base_classification=base_loss,
            fused_classification=fused_loss,
            margin_ranking=ranking_loss,
            wrong_merge=wrong_merge_loss,
        )
