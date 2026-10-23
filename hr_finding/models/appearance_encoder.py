"""Lightweight shared appearance encoder and masked history aggregation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class TinyAppearanceEncoder(nn.Module):
    """Small CNN fallback when cached detector ROI features are unavailable."""

    def __init__(
        self,
        embedding_dim: int = 64,
        dropout: float = 0.1,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            DepthwiseSeparableBlock(16, 24, stride=2),
            DepthwiseSeparableBlock(24, 32, stride=2),
            DepthwiseSeparableBlock(32, 48, stride=2),
            DepthwiseSeparableBlock(48, 64, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.projection(self.features(images))
        return F.normalize(embeddings, dim=-1) if self.normalize else embeddings


class MaskedAttentionAggregator(nn.Module):
    """Target-conditioned aggregation over a variable-length identity history."""

    def __init__(self, embedding_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.null_embedding = nn.Parameter(torch.zeros(embedding_dim))

    def forward(
        self,
        history: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count, dim = history.shape
        target_expanded = target[:, None, :].expand(batch, count, dim)
        attention_inputs = torch.cat(
            [history, target_expanded, torch.abs(history - target_expanded)], dim=-1
        )
        logits = self.attention(attention_inputs).squeeze(-1)
        has_history = mask.any(dim=1)
        safe_mask = mask.clone()
        safe_mask[~has_history, 0] = True
        logits = logits.masked_fill(~safe_mask, -1e4)
        weights = torch.softmax(logits, dim=1) * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        aggregate = torch.sum(history * weights.unsqueeze(-1), dim=1)
        aggregate = torch.where(
            has_history.unsqueeze(-1),
            aggregate,
            self.null_embedding.unsqueeze(0).expand(batch, -1),
        )
        return aggregate, weights


def encode_masked_images(
    encoder: nn.Module,
    images: torch.Tensor,
    mask: torch.Tensor,
    embedding_dim: int,
) -> torch.Tensor:
    """Avoid running the CNN on padded crops."""
    batch, count = images.shape[:2]
    flat_images = images.reshape(batch * count, *images.shape[2:])
    flat_mask = mask.reshape(-1)
    output = images.new_zeros((batch * count, embedding_dim))
    if flat_mask.any():
        output[flat_mask] = encoder(flat_images[flat_mask])
    return output.reshape(batch, count, embedding_dim)
