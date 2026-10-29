"""Four-layer Lightweight ECIA network.

Tensor convention
-----------------
``B`` is batch size, ``K`` is the padded number of competitors, ``D=512`` is
the frozen ReID embedding size, and ``P=64`` is the learned projection size.
Competitor padding is controlled exclusively by ``competitor_mask``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


AblationMode = Literal["base_only", "base_hr_gain", "full"]


@dataclass
class ECIAOutput:
    """Named model outputs used by the loss, inference, and ablation reports."""

    base_scores: Tensor  # [B, 1+K], old is slot 0.
    delta_scores: Tensor  # [B, 1+K].
    final_scores: Tensor  # [B, 1+K].
    abstain_logit: Tensor  # [B, 1].
    decision_logits: Tensor  # [B, 2+K], abstain is the last slot.
    gate: Tensor  # [B, 1], shared by all hypotheses in one event.
    gate_features: Tensor  # [B, 5].
    hypothesis_mask: Tensor  # [B, 1+K].


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the exact number of trainable scalar parameters."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class ECIA(nn.Module):
    """Evidence-Conditioned Identity Association without a visual backbone.

    The model consumes only L2-normalized embeddings produced by the frozen
    ReID encoder.  It never receives GT IDs, candidate labels, frame labels, or
    evaluator outcomes.
    """

    VALID_ABLATIONS = {"base_only", "base_hr_gain", "full"}

    def __init__(
        self,
        embedding_dim: int = 512,
        projection_dim: int = 64,
        relation_hidden_dim: int = 32,
        event_hidden_dim: int = 16,
        parameter_limit: int = 50_000,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.projection_dim = int(projection_dim)

        # Layer 1: a single projection instance is reused by old, target, HR,
        # and every competitor.  Sharing is important for a meaningful cosine.
        self.shared_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.GELU(),
        )

        # Layer 2A: |target-q|, target*q, and cosine(target,q).
        base_input_dim = 2 * self.projection_dim + 1
        self.base_association = nn.Sequential(
            nn.Linear(base_input_dim, relation_hidden_dim),
            nn.GELU(),
            nn.Linear(relation_hidden_dim, 1),
        )

        # Layer 2B: |hr-q|, |hr-target|, hr*q, two cosines, and base_score.
        gain_input_dim = 3 * self.projection_dim + 3
        self.hr_evidence_gain = nn.Sequential(
            nn.Linear(gain_input_dim, relation_hidden_dim),
            nn.GELU(),
            nn.Linear(relation_hidden_dim, 1),
        )

        # Layer 3: one event-level scalar controls HR influence for all IDs.
        self.evidence_gate = nn.Sequential(
            nn.Linear(5, event_hidden_dim),
            nn.GELU(),
            nn.Linear(event_hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Layer 4: a learned alternative to selecting any identity hypothesis.
        self.abstain_head = nn.Sequential(
            nn.Linear(5, event_hidden_dim),
            nn.GELU(),
            nn.Linear(event_hidden_dim, 1),
        )

        parameter_count = count_trainable_parameters(self)
        if parameter_count > parameter_limit:
            raise ValueError(
                f"ECIA has {parameter_count:,} parameters, above limit {parameter_limit:,}"
            )

    @staticmethod
    def _masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
        masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return torch.softmax(masked, dim=-1)

    @staticmethod
    def _masked_max(values: Tensor, mask: Tensor, fallback: Tensor) -> Tensor:
        """Return a stable max when an event contains no competitors."""

        if values.shape[-1] == 0:
            return fallback
        masked = values.masked_fill(~mask, torch.finfo(values.dtype).min)
        maximum = masked.max(dim=-1).values
        has_value = mask.any(dim=-1)
        return torch.where(has_value, maximum, fallback)

    def _project(self, embedding: Tensor) -> Tensor:
        # Frozen ReID vectors are normalized at the model boundary as a second
        # safety layer even when the cache already stores normalized vectors.
        return self.shared_projection(F.normalize(embedding, p=2, dim=-1, eps=1e-8))

    def forward(
        self,
        old: Tensor,
        target: Tensor,
        hr: Tensor,
        competitors: Tensor,
        competitor_mask: Tensor,
        hr_available: Tensor | None = None,
        ablation: AblationMode = "full",
    ) -> ECIAOutput:
        """Score old ID, variable competitors, and learned abstention.

        Parameters
        ----------
        old, target, hr:
            ``[B, 512]`` normalized internally.  ``hr`` may be zeros when HR is
            absent, but ``hr_available`` must then be false.
        competitors:
            ``[B, K, 512]`` padded embeddings.
        competitor_mask:
            ``[B, K]`` boolean mask where true means a real competitor.
        hr_available:
            ``[B]`` or ``[B,1]``.  Missing HR forces ``final=base``.
        ablation:
            ``base_only``; ``base_hr_gain`` with gate fixed to one; or ``full``.
        """

        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown ablation {ablation!r}; use {sorted(self.VALID_ABLATIONS)}")
        if competitors.ndim != 3 or competitor_mask.ndim != 2:
            raise ValueError("competitors must be [B,K,D] and competitor_mask [B,K]")
        if competitors.shape[:2] != competitor_mask.shape:
            raise ValueError("competitor mask shape must match competitors[:2]")

        batch_size, competitor_count, _ = competitors.shape
        competitor_mask = competitor_mask.bool()
        if hr_available is None:
            hr_available = torch.ones(batch_size, 1, dtype=torch.bool, device=old.device)
        else:
            hr_available = hr_available.reshape(batch_size, 1).bool()

        # Layer 1 — shared 512D→64D projection.
        old_p = self._project(old)
        target_p = self._project(target)
        hr_p = self._project(hr)
        competitor_p = self._project(competitors.reshape(-1, self.embedding_dim)).reshape(
            batch_size, competitor_count, self.projection_dim
        )
        hypotheses = torch.cat((old_p.unsqueeze(1), competitor_p), dim=1)
        hypothesis_mask = torch.cat(
            (
                torch.ones(batch_size, 1, dtype=torch.bool, device=old.device),
                competitor_mask,
            ),
            dim=1,
        )

        # Layer 2A — LR-only association for old and every competitor.
        target_expanded = target_p.unsqueeze(1).expand_as(hypotheses)
        base_cosine = F.cosine_similarity(target_expanded, hypotheses, dim=-1).unsqueeze(-1)
        base_relation = torch.cat(
            (
                torch.abs(target_expanded - hypotheses),
                target_expanded * hypotheses,
                base_cosine,
            ),
            dim=-1,
        )
        base_scores = self.base_association(base_relation).squeeze(-1)

        # Layer 2B — signed HR evidence gain.  Delta may be positive or negative;
        # the architecture does not assume that an HR crop must help.
        hr_expanded = hr_p.unsqueeze(1).expand_as(hypotheses)
        hr_target_difference = torch.abs(hr_p - target_p).unsqueeze(1).expand_as(hypotheses)
        hr_hypothesis_cosine = F.cosine_similarity(hr_expanded, hypotheses, dim=-1).unsqueeze(-1)
        hr_target_cosine = F.cosine_similarity(hr_p, target_p, dim=-1, eps=1e-8)
        # Missing HR must not leak the projection bias of a zero placeholder
        # into either the gate or learned Abstain logit.
        hr_target_cosine = torch.where(
            hr_available[:, 0], hr_target_cosine, torch.zeros_like(hr_target_cosine)
        )
        hr_target_cosine_expanded = hr_target_cosine[:, None, None].expand(
            -1, hypotheses.shape[1], -1
        )
        gain_relation = torch.cat(
            (
                torch.abs(hr_expanded - hypotheses),
                hr_target_difference,
                hr_expanded * hypotheses,
                hr_hypothesis_cosine,
                hr_target_cosine_expanded,
                base_scores.unsqueeze(-1),
            ),
            dim=-1,
        )
        delta_scores = self.hr_evidence_gain(gain_relation).squeeze(-1)

        # Layer 3 — five event features.  Competitor similarities are summarized
        # by the strongest valid competitor so K may vary without changing size.
        base_probabilities = self._masked_softmax(base_scores, hypothesis_mask)
        old_probability = base_probabilities[:, 0]
        top_probability = base_probabilities.max(dim=-1).values
        best_competitor_base = self._masked_max(
            base_scores[:, 1:], competitor_mask, fallback=base_scores[:, 0]
        )
        old_margin = base_scores[:, 0] - best_competitor_base
        competitor_hr_cosines = hr_hypothesis_cosine[:, 1:, 0]
        best_hr_competitor = self._masked_max(
            competitor_hr_cosines,
            competitor_mask,
            fallback=torch.zeros_like(hr_target_cosine),
        )
        best_hr_competitor = torch.where(
            hr_available[:, 0], best_hr_competitor, torch.zeros_like(best_hr_competitor)
        )
        gate_features = torch.stack(
            (
                old_probability,
                top_probability,
                old_margin,
                hr_target_cosine,
                best_hr_competitor,
            ),
            dim=-1,
        )

        learned_gate = self.evidence_gate(gate_features)
        if ablation == "base_only":
            gate = torch.zeros_like(learned_gate)
        elif ablation == "base_hr_gain":
            gate = torch.ones_like(learned_gate)
        else:
            gate = learned_gate
        gate = gate * hr_available.to(gate.dtype)
        final_scores = base_scores + gate * delta_scores

        # Layer 4 — Base-only is a strict LR ablation. Its Abstain head receives
        # the same five-slot layout, but the two HR-derived slots are zeroed so
        # changing the selected HR crop cannot affect any decision logit.
        abstain_features = gate_features
        if ablation == "base_only":
            abstain_features = gate_features.clone()
            abstain_features[:, 3:] = 0.0
        abstain_logit = self.abstain_head(abstain_features)
        masked_final = final_scores.masked_fill(
            ~hypothesis_mask, torch.finfo(final_scores.dtype).min
        )
        decision_logits = torch.cat((masked_final, abstain_logit), dim=-1)

        return ECIAOutput(
            base_scores=base_scores,
            delta_scores=delta_scores,
            final_scores=final_scores,
            abstain_logit=abstain_logit,
            decision_logits=decision_logits,
            gate=gate,
            gate_features=gate_features,
            hypothesis_mask=hypothesis_mask,
        )
