"""Identity-contrastive feature construction without track-ID shortcuts."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1).unsqueeze(-1)


class SimpleAppearanceRelation(nn.Module):
    """Appearance baseline using only old, target, and candidate embeddings."""

    def __init__(self, embedding_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, old: torch.Tensor, target: torch.Tensor, candidate: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat([old, target, candidate], dim=-1))


class CandidateRelationEncoder(nn.Module):
    """Contrast old-track, target, and candidate hypotheses."""

    def __init__(self, embedding_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        input_dim = embedding_dim * 9 + 3
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.SiLU(),
        )

    def forward(
        self, old: torch.Tensor, target: torch.Tensor, candidate: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat(
            [
                old,
                target,
                candidate,
                torch.abs(old - target),
                torch.abs(old - candidate),
                torch.abs(target - candidate),
                old * target,
                old * candidate,
                target * candidate,
                cosine(old, target),
                cosine(old, candidate),
                cosine(target, candidate),
            ],
            dim=-1,
        )
        return self.network(features)


class CompetingHypothesisEncoder(nn.Module):
    """Compare candidate support for a real competing LR appearance history."""

    def __init__(self, embedding_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        input_dim = embedding_dim * 3 + 3
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        target: torch.Tensor,
        candidate: torch.Tensor,
        competitor: torch.Tensor,
        competitor_available: torch.Tensor,
    ) -> torch.Tensor:
        available = competitor_available.float().unsqueeze(-1)
        features = torch.cat(
            [
                competitor,
                torch.abs(candidate - competitor),
                candidate * competitor,
                cosine(candidate, competitor),
                cosine(target, competitor),
                available,
            ],
            dim=-1,
        )
        return self.network(features) * available
