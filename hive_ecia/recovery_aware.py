"""Recovery-aware HR fusion with an exactly frozen ECIA Base branch.

The module keeps the Base-only projection, association head, and abstain head
bit-for-bit frozen.  Only the HR evidence gain and event gate are trainable.
No detector, HIVE, ReID, GT ID, or evaluator signal is consumed at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import ECIA


ARCHITECTURE_NAME = "recovery_aware_hr_fusion_v1"


@dataclass
class RecoveryAwareOutput:
    base_scores: Tensor
    delta_scores: Tensor
    final_scores: Tensor
    abstain_logit: Tensor
    base_decision_logits: Tensor
    decision_logits: Tensor
    gate: Tensor
    gate_features: Tensor
    hypothesis_mask: Tensor
    hr_available: Tensor


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def total_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class RecoveryAwareECIA(nn.Module):
    """Frozen LR association plus learnable recovery-aware HR residual fusion."""

    FROZEN_PREFIXES = ("shared_projection", "base_association", "abstain_head")

    def __init__(
        self,
        embedding_dim: int = 512,
        projection_dim: int = 64,
        relation_hidden_dim: int = 32,
        event_hidden_dim: int = 32,
        parameter_limit: int = 50_000,
        hr_distribution_temperature: float = 0.20,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.projection_dim = int(projection_dim)
        self.hr_distribution_temperature = float(hr_distribution_temperature)

        # These three modules have exactly the same topology and key names as
        # the original ECIA model, which allows an audited Base-only transfer.
        base = ECIA(
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            relation_hidden_dim=relation_hidden_dim,
            event_hidden_dim=16,
            parameter_limit=parameter_limit,
        )
        self.shared_projection = base.shared_projection
        self.base_association = base.base_association
        self.abstain_head = base.abstain_head

        # Per-hypothesis HR relation: the original visual relation (195D) plus
        # eight LR-confidence/disagreement values.  The signed residual is free
        # to support or oppose each identity hypothesis.
        gain_input_dim = 3 * self.projection_dim + 10
        self.hr_evidence_gain = nn.Sequential(
            nn.Linear(gain_input_dim, relation_hidden_dim),
            nn.GELU(),
            nn.Linear(relation_hidden_dim, 1),
        )

        # Event-level recovery gate.  A conservative negative bias starts HR
        # influence near zero until supervision demonstrates a recovery gain.
        self.evidence_gate = nn.Sequential(
            nn.Linear(12, event_hidden_dim),
            nn.GELU(),
            nn.Linear(event_hidden_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.evidence_gate[2].bias, -2.0)

        if total_parameter_count(self) > int(parameter_limit):
            raise ValueError(
                f"RecoveryAwareECIA has {total_parameter_count(self):,} parameters, "
                f"above limit {parameter_limit:,}"
            )
        self.freeze_base()

    def freeze_base(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad = not name.startswith(self.FROZEN_PREFIXES)
        for module in (self.shared_projection, self.base_association, self.abstain_head):
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen LayerNorm statistics/behavior must remain identical to Base.
        for module in (self.shared_projection, self.base_association, self.abstain_head):
            module.eval()
        return self

    def load_frozen_base(self, checkpoint_path: Path) -> dict:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("ablation") != "base_only":
            raise ValueError("Recovery-aware initialization requires a Base-only checkpoint")
        source = checkpoint["model"]
        transferred = []
        own = self.state_dict()
        for prefix in self.FROZEN_PREFIXES:
            keys = [key for key in own if key.startswith(prefix + ".")]
            if not keys or any(key not in source for key in keys):
                raise KeyError(f"Incomplete frozen Base state for {prefix}")
            for key in keys:
                if own[key].shape != source[key].shape:
                    raise ValueError(f"Frozen Base shape mismatch for {key}")
                own[key] = source[key].detach().clone()
                transferred.append(key)
        self.load_state_dict(own, strict=True)
        self.freeze_base()
        return {
            "base_checkpoint": str(Path(checkpoint_path).resolve()),
            "transferred_keys": transferred,
            "transferred_key_count": len(transferred),
        }

    @staticmethod
    def _masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
        return torch.softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=-1)

    @staticmethod
    def _masked_max(values: Tensor, mask: Tensor, fallback: Tensor) -> Tensor:
        if values.shape[-1] == 0:
            return fallback
        maximum = values.masked_fill(~mask, torch.finfo(values.dtype).min).max(dim=-1).values
        return torch.where(mask.any(dim=-1), maximum, fallback)

    def _project(self, embedding: Tensor) -> Tensor:
        return self.shared_projection(F.normalize(embedding, p=2, dim=-1, eps=1e-8))

    def forward(
        self,
        old: Tensor,
        target: Tensor,
        hr: Tensor,
        competitors: Tensor,
        competitor_mask: Tensor,
        hr_available: Tensor | None = None,
        ablation: str = "full",
        gate_override: float | None = None,
    ) -> RecoveryAwareOutput:
        if ablation != "full":
            raise ValueError("RecoveryAwareECIA supports only the Full path; use gate_override=0 for HR ablation")
        batch_size, competitor_count, _ = competitors.shape
        competitor_mask = competitor_mask.bool()
        if hr_available is None:
            hr_available = torch.ones(batch_size, 1, dtype=torch.bool, device=old.device)
        else:
            hr_available = hr_available.reshape(batch_size, 1).bool()

        old_p = self._project(old)
        target_p = self._project(target)
        hr_p = self._project(hr)
        competitor_p = self._project(competitors.reshape(-1, self.embedding_dim)).reshape(
            batch_size, competitor_count, self.projection_dim
        )
        hypotheses = torch.cat((old_p.unsqueeze(1), competitor_p), dim=1)
        hypothesis_mask = torch.cat(
            (torch.ones(batch_size, 1, dtype=torch.bool, device=old.device), competitor_mask),
            dim=1,
        )

        target_expanded = target_p.unsqueeze(1).expand_as(hypotheses)
        target_hypothesis_cosine = F.cosine_similarity(
            target_expanded, hypotheses, dim=-1, eps=1e-8
        )
        base_relation = torch.cat(
            (
                torch.abs(target_expanded - hypotheses),
                target_expanded * hypotheses,
                target_hypothesis_cosine.unsqueeze(-1),
            ),
            dim=-1,
        )
        base_scores = self.base_association(base_relation).squeeze(-1)
        base_probabilities = self._masked_softmax(base_scores, hypothesis_mask)
        old_probability = base_probabilities[:, 0]
        top_probability = base_probabilities.max(dim=-1).values
        sorted_probabilities = base_probabilities.sort(dim=-1, descending=True).values
        second_probability = (
            sorted_probabilities[:, 1]
            if sorted_probabilities.shape[1] > 1
            else torch.zeros_like(top_probability)
        )
        top2_margin = top_probability - second_probability
        valid_count = hypothesis_mask.sum(dim=-1).to(base_probabilities.dtype)
        entropy = -(base_probabilities.clamp_min(1e-8).log() * base_probabilities).sum(dim=-1)
        normalized_entropy = entropy / valid_count.clamp_min(2.0).log()
        best_competitor_base = self._masked_max(
            base_scores[:, 1:], competitor_mask, fallback=base_scores[:, 0]
        )
        old_margin = base_scores[:, 0] - best_competitor_base

        # Exact frozen Base-only abstention features from the original model.
        base_abstain_features = torch.stack(
            (old_probability, top_probability, old_margin, torch.zeros_like(old_margin), torch.zeros_like(old_margin)),
            dim=-1,
        )
        abstain_logit = self.abstain_head(base_abstain_features)
        masked_base = base_scores.masked_fill(~hypothesis_mask, torch.finfo(base_scores.dtype).min)
        base_decision_logits = torch.cat((masked_base, abstain_logit), dim=-1)

        hr_expanded = hr_p.unsqueeze(1).expand_as(hypotheses)
        hr_target_difference = torch.abs(hr_p - target_p).unsqueeze(1).expand_as(hypotheses)
        hr_hypothesis_cosine = F.cosine_similarity(hr_expanded, hypotheses, dim=-1, eps=1e-8)
        hr_target_cosine = F.cosine_similarity(hr_p, target_p, dim=-1, eps=1e-8)
        hr_target_cosine = torch.where(
            hr_available[:, 0], hr_target_cosine, torch.zeros_like(hr_target_cosine)
        )
        hr_old_cosine = torch.where(
            hr_available[:, 0], hr_hypothesis_cosine[:, 0], torch.zeros_like(hr_target_cosine)
        )
        best_hr_competitor = self._masked_max(
            hr_hypothesis_cosine[:, 1:], competitor_mask, fallback=hr_hypothesis_cosine[:, 0]
        )
        best_hr_competitor = torch.where(
            hr_available[:, 0], best_hr_competitor, torch.zeros_like(best_hr_competitor)
        )
        hr_identity_margin = hr_old_cosine - best_hr_competitor

        hr_distribution = self._masked_softmax(
            hr_hypothesis_cosine / self.hr_distribution_temperature, hypothesis_mask
        )
        midpoint = 0.5 * (base_probabilities + hr_distribution)
        js_disagreement = 0.5 * (
            (base_probabilities * (base_probabilities.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(dim=-1)
            + (hr_distribution * (hr_distribution.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(dim=-1)
        )
        top_disagreement = (base_probabilities.argmax(dim=-1) != hr_distribution.argmax(dim=-1)).to(base_scores.dtype)
        similarity_disagreement = torch.abs(
            hr_hypothesis_cosine - target_hypothesis_cosine
        ).mean(dim=-1)
        js_disagreement = torch.where(hr_available[:, 0], js_disagreement, torch.zeros_like(js_disagreement))
        top_disagreement = torch.where(hr_available[:, 0], top_disagreement, torch.zeros_like(top_disagreement))
        similarity_disagreement = torch.where(
            hr_available[:, 0], similarity_disagreement, torch.zeros_like(similarity_disagreement)
        )

        gate_features = torch.stack(
            (
                old_probability,
                top_probability,
                top2_margin,
                old_margin,
                normalized_entropy,
                hr_target_cosine,
                hr_old_cosine,
                best_hr_competitor,
                hr_identity_margin,
                js_disagreement,
                top_disagreement,
                similarity_disagreement,
            ),
            dim=-1,
        )

        expanded = lambda value: value[:, None, None].expand(-1, hypotheses.shape[1], -1)
        gain_relation = torch.cat(
            (
                torch.abs(hr_expanded - hypotheses),
                hr_target_difference,
                hr_expanded * hypotheses,
                hr_hypothesis_cosine.unsqueeze(-1),
                expanded(hr_target_cosine),
                base_scores.unsqueeze(-1),
                base_probabilities.unsqueeze(-1),
                expanded(old_probability),
                expanded(top2_margin),
                expanded(old_margin),
                target_hypothesis_cosine.unsqueeze(-1),
                expanded(js_disagreement),
                torch.abs(hr_hypothesis_cosine - target_hypothesis_cosine).unsqueeze(-1),
            ),
            dim=-1,
        )
        delta_scores = self.hr_evidence_gain(gain_relation).squeeze(-1)
        learned_gate = self.evidence_gate(gate_features)
        if gate_override is not None:
            if not 0.0 <= float(gate_override) <= 1.0:
                raise ValueError("gate_override must be in [0,1]")
            gate = torch.full_like(learned_gate, float(gate_override))
        else:
            gate = learned_gate
        gate = gate * hr_available.to(gate.dtype)
        final_scores = base_scores + gate * delta_scores
        masked_final = final_scores.masked_fill(~hypothesis_mask, torch.finfo(final_scores.dtype).min)
        decision_logits = torch.cat((masked_final, abstain_logit), dim=-1)
        return RecoveryAwareOutput(
            base_scores=base_scores,
            delta_scores=delta_scores,
            final_scores=final_scores,
            abstain_logit=abstain_logit,
            base_decision_logits=base_decision_logits,
            decision_logits=decision_logits,
            gate=gate,
            gate_features=gate_features,
            hypothesis_mask=hypothesis_mask,
            hr_available=hr_available[:, 0],
        )


def build_from_checkpoint(checkpoint: dict, device: torch.device) -> RecoveryAwareECIA:
    config = checkpoint["config"]
    model_config = config["model"]
    recovery_config = config["recovery_aware"]
    model = RecoveryAwareECIA(
        embedding_dim=int(model_config["embedding_dim"]),
        projection_dim=int(model_config["projection_dim"]),
        relation_hidden_dim=int(model_config["relation_hidden_dim"]),
        event_hidden_dim=int(recovery_config["event_hidden_dim"]),
        parameter_limit=int(model_config["parameter_limit"]),
        hr_distribution_temperature=float(recovery_config["hr_distribution_temperature"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.freeze_base()
    model.eval()
    return model
