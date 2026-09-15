from __future__ import annotations

import torch
from torch import nn


class ResidualFFN(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class DenseProxyEncoder(nn.Module):
    """Standard methylation-proxy adapter for a frozen dense locus embedding."""

    def __init__(
        self,
        raw_dim: int,
        latent_dim: int = 256,
        n_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(raw_dim), nn.Linear(raw_dim, latent_dim), nn.GELU())
        self.blocks = nn.ModuleList([ResidualFFN(latent_dim, dropout) for _ in range(n_blocks)])
        self.output_norm = nn.LayerNorm(latent_dim)
        self.output_dim = latent_dim

    def forward(self, raw_embedding: torch.Tensor) -> torch.Tensor:
        h = self.input(raw_embedding)
        for block in self.blocks:
            h = block(h)
        return self.output_norm(h)


class FunctionalAnnotationEncoder(nn.Module):
    """Self-contained functional-locus branch extracted from the original RNA->DNAm model.

    The input is patient-agnostic reference information only: a bag of functional track IDs
    plus dense locus covariates. No RNA or patient methylation value enters this encoder.
    """

    def __init__(
        self,
        *,
        n_tracks: int = 4165,
        dense_dim: int = 23,
        latent_dim: int = 256,
        n_blocks: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.track_embedding = nn.EmbeddingBag(
            n_tracks, latent_dim, mode="mean", include_last_offset=True
        )
        self.dense_encoder = nn.Sequential(
            nn.LayerNorm(dense_dim), nn.Linear(dense_dim, latent_dim), nn.GELU()
        )
        self.locus_norm = nn.LayerNorm(latent_dim)
        self.blocks = nn.ModuleList([ResidualFFN(latent_dim, dropout) for _ in range(n_blocks)])
        self.output_dim = latent_dim

    def forward(
        self,
        track_indices: torch.Tensor,
        offsets: torch.Tensor,
        dense: torch.Tensor,
    ) -> torch.Tensor:
        h = self.locus_norm(self.track_embedding(track_indices, offsets) + self.dense_encoder(dense))
        for block in self.blocks:
            h = block(h)
        return h


class MeanMethylationProxy(nn.Module):
    def __init__(self, encoder: nn.Module, embedding_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embedding_dim, 1)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "raw_embedding" in batch:
            return self.encoder(batch["raw_embedding"])
        return self.encoder(batch["track_indices"], batch["offsets"], batch["dense"])

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(batch)
        prediction = torch.sigmoid(self.head(embedding).squeeze(-1))
        return prediction, embedding
