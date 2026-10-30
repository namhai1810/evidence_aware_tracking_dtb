"""Objectives for recovery-aware HR fusion with a frozen Base branch."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .dataset import SAMPLE_KIND_TO_ID
from .recovery_aware import RecoveryAwareOutput


@dataclass
class RecoveryAwareLossOutput:
    total: Tensor
    fused_classification: Tensor
    margin_ranking: Tensor
    wrong_merge: Tensor
    base_preservation: Tensor
    recovery: Tensor
    gate_supervision: Tensor

    def detached(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "fused_classification": float(self.fused_classification.detach()),
            "margin_ranking": float(self.margin_ranking.detach()),
            "wrong_merge": float(self.wrong_merge.detach()),
            "base_preservation": float(self.base_preservation.detach()),
            "recovery": float(self.recovery.detach()),
            "gate_supervision": float(self.gate_supervision.detach()),
        }


class RecoveryAwareLoss(nn.Module):
    """Prefer HR only when it recovers a frozen-Base error.

    `positive` cache samples are the only samples whose selected HR identity is
    verified to support the target.  Negative, hard-negative, and no-merge
    samples deliberately contain misleading/irrelevant HR and supervise gate
    suppression.  GT-derived labels are present only in the offline cache.
    """

    def __init__(
        self,
        fused_classification_weight: float = 1.0,
        margin_ranking_weight: float = 0.25,
        wrong_merge_weight: float = 1.0,
        base_preservation_weight: float = 1.0,
        recovery_weight: float = 1.0,
        gate_supervision_weight: float = 0.75,
        helpful_gate_weight: float = 3.0,
        harmful_gate_weight: float = 2.0,
        ranking_margin: float = 0.20,
    ) -> None:
        super().__init__()
        self.fused_classification_weight = float(fused_classification_weight)
        self.margin_ranking_weight = float(margin_ranking_weight)
        self.wrong_merge_weight = float(wrong_merge_weight)
        self.base_preservation_weight = float(base_preservation_weight)
        self.recovery_weight = float(recovery_weight)
        self.gate_supervision_weight = float(gate_supervision_weight)
        self.helpful_gate_weight = float(helpful_gate_weight)
        self.harmful_gate_weight = float(harmful_gate_weight)
        self.ranking_margin = float(ranking_margin)

    @staticmethod
    def _weighted_mean(values: Tensor, weights: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is not None:
            if not mask.any():
                return values.sum() * 0.0
            values = values[mask]
            weights = weights[mask]
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    def forward(
        self,
        output: RecoveryAwareOutput,
        target_class: Tensor,
        merge_target: Tensor,
        sample_kind: Tensor,
        sample_weight: Tensor | None = None,
    ) -> RecoveryAwareLossOutput:
        batch_size = target_class.shape[0]
        dtype = output.decision_logits.dtype
        if sample_weight is None:
            sample_weight = torch.ones(batch_size, device=target_class.device, dtype=dtype)
        else:
            sample_weight = sample_weight.to(dtype)

        fused_rows = F.cross_entropy(output.decision_logits, target_class, reduction="none")
        fused_loss = self._weighted_mean(fused_rows, sample_weight)

        positive = output.decision_logits.gather(1, target_class[:, None]).squeeze(1)
        alternative_mask = torch.ones_like(output.decision_logits, dtype=torch.bool)
        alternative_mask.scatter_(1, target_class[:, None], False)
        strongest_alternative = output.decision_logits.masked_fill(
            ~alternative_mask, torch.finfo(dtype).min
        ).max(dim=1).values
        ranking_rows = F.relu(self.ranking_margin - positive + strongest_alternative)
        ranking_loss = self._weighted_mean(ranking_rows, sample_weight)

        old_probability = torch.softmax(output.decision_logits, dim=-1)[:, 0]
        wrong_merge_rows = F.binary_cross_entropy(
            old_probability.clamp(1e-6, 1.0 - 1e-6),
            merge_target.to(dtype),
            reduction="none",
        )
        wrong_merge_loss = self._weighted_mean(wrong_merge_rows, sample_weight)

        with torch.no_grad():
            base_probabilities = torch.softmax(output.base_decision_logits, dim=-1)
            base_prediction = base_probabilities.argmax(dim=-1)
            base_correct = base_prediction == target_class
            hr_supports_target = sample_kind == SAMPLE_KIND_TO_ID["positive"]
            helpful = (~base_correct) & hr_supports_target & output.hr_available
            suppress = (~helpful) & output.hr_available

        # Directly penalize any reduction in the correct-class probability on
        # events the frozen Base already solved. This captures HR-caused damage
        # before it becomes a discrete wrong decision.
        final_probabilities = torch.softmax(output.decision_logits, dim=-1)
        base_target_probability = base_probabilities.gather(1, target_class[:, None]).squeeze(1)
        final_target_probability = final_probabilities.gather(1, target_class[:, None]).squeeze(1)
        preservation_rows = F.relu(base_target_probability - final_target_probability)
        preservation_loss = self._weighted_mean(
            preservation_rows, sample_weight * self.harmful_gate_weight, base_correct
        )

        # Recovery is concentrated on verified-HR events missed by frozen Base.
        recovery_loss = self._weighted_mean(
            fused_rows, sample_weight * self.helpful_gate_weight, helpful
        )

        gate_target = helpful.to(dtype)
        gate_rows = F.binary_cross_entropy(
            output.gate[:, 0].clamp(1e-6, 1.0 - 1e-6), gate_target, reduction="none"
        )
        gate_weights = sample_weight * torch.where(
            helpful,
            torch.full_like(sample_weight, self.helpful_gate_weight),
            torch.full_like(sample_weight, self.harmful_gate_weight),
        )
        gate_loss = self._weighted_mean(gate_rows, gate_weights, helpful | suppress)

        total = (
            self.fused_classification_weight * fused_loss
            + self.margin_ranking_weight * ranking_loss
            + self.wrong_merge_weight * wrong_merge_loss
            + self.base_preservation_weight * preservation_loss
            + self.recovery_weight * recovery_loss
            + self.gate_supervision_weight * gate_loss
        )
        return RecoveryAwareLossOutput(
            total=total,
            fused_classification=fused_loss,
            margin_ranking=ranking_loss,
            wrong_merge=wrong_merge_loss,
            base_preservation=preservation_loss,
            recovery=recovery_loss,
            gate_supervision=gate_loss,
        )
