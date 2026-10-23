"""HIVE lightweight utility-ranking network."""

from __future__ import annotations

import time

import torch
from torch import nn

from .appearance_encoder import (
    MaskedAttentionAggregator,
    TinyAppearanceEncoder,
    encode_masked_images,
)
from .hypothesis_encoder import (
    CandidateRelationEncoder,
    CompetingHypothesisEncoder,
    SimpleAppearanceRelation,
)


class HIVE(nn.Module):
    SUPPORTED_VARIANTS = {
        "appearance_only",
        "appearance_metadata",
        "hive_no_hypothesis",
        "hive_full",
    }

    def __init__(self, model_config: dict, metadata_dim: int = 3) -> None:
        super().__init__()
        self.variant = str(model_config.get("variant", "hive_full"))
        if self.variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported HIVE variant: {self.variant}")
        self.embedding_dim = int(model_config.get("embedding_dim", 64))
        relation_dim = int(model_config.get("relation_dim", 96))
        dropout = float(model_config.get("dropout", 0.15))
        self.encoder = TinyAppearanceEncoder(
            embedding_dim=self.embedding_dim,
            dropout=dropout,
            normalize=bool(model_config.get("normalize_embeddings", True)),
        )
        if bool(model_config.get("freeze_appearance_encoder", False)):
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        self.history_aggregator = MaskedAttentionAggregator(self.embedding_dim)
        self.competitor_aggregator = MaskedAttentionAggregator(self.embedding_dim)
        self.simple_relation = SimpleAppearanceRelation(
            self.embedding_dim, relation_dim, dropout
        )
        self.contrastive_relation = CandidateRelationEncoder(
            self.embedding_dim, relation_dim, dropout
        )
        self.competitor_relation = CompetingHypothesisEncoder(
            self.embedding_dim, relation_dim // 2, dropout
        )
        metadata_hidden = int(model_config.get("metadata_hidden_dim", 16))
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, metadata_hidden),
            nn.LayerNorm(metadata_hidden),
            nn.SiLU(),
        )

        relation_features = relation_dim
        if self.variant in {"appearance_metadata", "hive_no_hypothesis", "hive_full"}:
            relation_features += metadata_hidden
        if self.variant == "hive_full":
            relation_features += relation_dim // 2
        scorer_hidden = int(model_config.get("scorer_hidden_dim", 96))
        self.scorer = nn.Sequential(
            nn.Linear(relation_features, scorer_hidden),
            nn.LayerNorm(scorer_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(scorer_hidden, 1),
        )

    def encode_inputs(self, batch: dict) -> dict:
        target = self.encoder(batch["target_crop"])
        old_embeddings = encode_masked_images(
            self.encoder, batch["old_crops"], batch["old_mask"], self.embedding_dim
        )
        old, old_attention = self.history_aggregator(
            old_embeddings, batch["old_mask"], target
        )
        candidates = encode_masked_images(
            self.encoder,
            batch["candidate_crops"],
            batch["candidate_mask"],
            self.embedding_dim,
        )
        if self.variant == "hive_full":
            competitor_embeddings = encode_masked_images(
                self.encoder,
                batch["competitor_crops"],
                batch["competitor_mask"],
                self.embedding_dim,
            )
            competitor, competitor_attention = self.competitor_aggregator(
                competitor_embeddings, batch["competitor_mask"], target
            )
            competitor_available = batch["competitor_mask"].any(dim=1)
        else:
            competitor = target.new_zeros(target.shape)
            competitor_attention = target.new_zeros(batch["competitor_mask"].shape)
            competitor_available = torch.zeros(
                target.shape[0], dtype=torch.bool, device=target.device
            )
        return {
            "target": target,
            "old": old,
            "candidates": candidates,
            "competitor": competitor,
            "competitor_available": competitor_available,
            "old_attention": old_attention,
            "competitor_attention": competitor_attention,
        }

    def score_encoded(self, encoded: dict, batch: dict) -> torch.Tensor:
        candidates = encoded["candidates"]
        batch_size, candidate_count, _ = candidates.shape
        old = encoded["old"][:, None, :].expand(-1, candidate_count, -1)
        target = encoded["target"][:, None, :].expand(-1, candidate_count, -1)
        if self.variant in {"appearance_only", "appearance_metadata"}:
            relation = self.simple_relation(old, target, candidates)
        else:
            relation = self.contrastive_relation(old, target, candidates)
        features = [relation]
        if self.variant in {"appearance_metadata", "hive_no_hypothesis", "hive_full"}:
            features.append(self.metadata_encoder(batch["metadata"]))
        if self.variant == "hive_full":
            competitor = encoded["competitor"][:, None, :].expand(-1, candidate_count, -1)
            available = encoded["competitor_available"][:, None].expand(
                batch_size, candidate_count
            )
            features.append(
                self.competitor_relation(target, candidates, competitor, available)
            )
        logits = self.scorer(torch.cat(features, dim=-1)).squeeze(-1)
        return logits.masked_fill(~batch["candidate_mask"], -1e4)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.score_encoded(self.encode_inputs(batch), batch)

    @torch.no_grad()
    def forward_with_timing(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        device = batch["target_crop"].device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        encoded = self.encode_inputs(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encoded_at = time.perf_counter()
        logits = self.score_encoded(encoded, batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        scored_at = time.perf_counter()
        return logits, {
            "encoder_seconds": encoded_at - start,
            "scorer_seconds": scored_at - encoded_at,
            "total_seconds": scored_at - start,
        }


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
