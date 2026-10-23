"""Masked classification, event-normalized RankNet, and optional distillation losses."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def masked_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    positive_weight: torch.Tensor | None,
) -> torch.Tensor:
    if not mask.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        logits[mask], labels[mask], pos_weight=positive_weight
    )


def event_normalized_ranknet(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Mean RankNet loss per eligible event, then mean across events."""
    event_losses = []
    for event_logits, event_labels, event_mask in zip(logits, labels, mask):
        valid_logits = event_logits[event_mask]
        valid_labels = event_labels[event_mask]
        positives = valid_logits[valid_labels > 0.5]
        negatives = valid_logits[valid_labels <= 0.5]
        if positives.numel() and negatives.numel():
            differences = positives[:, None] - negatives[None, :]
            event_losses.append(F.softplus(-differences).mean())
    if not event_losses:
        return logits.sum() * 0.0, 0
    return torch.stack(event_losses).mean(), len(event_losses)


def masked_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    if not mask.any():
        return logits.sum() * 0.0
    if loss_type == "mse":
        return F.mse_loss(logits[mask], targets[mask])
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(logits[mask], targets[mask])
    raise ValueError(f"Unsupported margin loss: {loss_type}")


class HIVEMultiTaskLoss(nn.Module):
    def __init__(self, config: dict, positive_weight: float | None = None) -> None:
        super().__init__()
        self.lambda_cls = float(config.get("lambda_cls", 1.0))
        self.lambda_rank = float(config.get("lambda_rank", 0.5))
        self.lambda_margin = float(config.get("lambda_margin", 0.0))
        self.margin_loss_type = str(config.get("margin_loss", "smooth_l1"))
        value = torch.tensor(float(positive_weight)) if positive_weight is not None else None
        self.register_buffer("positive_weight", value)

    def forward(self, logits: torch.Tensor, batch: dict) -> dict[str, torch.Tensor | int]:
        candidate_mask = batch["candidate_mask"]
        classification = masked_bce_with_logits(
            logits,
            batch["labels"],
            candidate_mask,
            self.positive_weight,
        )
        ranking, eligible_events = event_normalized_ranknet(
            logits, batch["labels"], candidate_mask
        )
        margin_mask = candidate_mask & batch["teacher_mask"]
        margin = masked_margin_loss(
            logits,
            batch["teacher_targets"],
            margin_mask,
            self.margin_loss_type,
        )
        total = (
            self.lambda_cls * classification
            + self.lambda_rank * ranking
            + self.lambda_margin * margin
        )
        return {
            "total": total,
            "classification": classification,
            "ranking": ranking,
            "margin": margin,
            "ranking_eligible_events": eligible_events,
        }
