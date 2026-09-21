from __future__ import annotations

import torch
from torch import nn

from cpg_repr_benchmark.models.model import ObservedCpGTokenizer


class FeedForward(nn.Module):
    def __init__(self, dim: int, *, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * mult)
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention followed by a residual FFN."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        context_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_padding_mask = None if context_valid is None else ~context_valid.bool()
        q = self.query_norm(query)
        kv = self.context_norm(context)
        update, _ = self.attn(
            q,
            kv,
            kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query = query + self.attn_dropout(update)
        return self.ff(query)


class SelfAttentionBlock(nn.Module):
    """Pre-norm latent self-attention followed by a residual FFN."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        update, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.attn_dropout(update)
        return self.ff(x)


class PerceiverMaskedMethylomeReconstructor(nn.Module):
    """Perceiver-IO style sparse methylome reconstructor.

    Observed CpG/value pairs are compressed into a small latent set through cross-attention.
    Latents interact through self-attention, while each target CpG is decoded with its own
    representation-derived query. No positional encoding is used, so the observed methylome
    remains a permutation-invariant set.
    """

    def __init__(
        self,
        raw_locus_dim: int,
        *,
        model_dim: int = 256,
        num_latents: int = 64,
        num_latent_blocks: int = 4,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_latents < 1:
            raise ValueError("num_latents must be positive")
        if num_latent_blocks < 1:
            raise ValueError("num_latent_blocks must be positive")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")

        self.raw_locus_dim = int(raw_locus_dim)
        self.model_dim = int(model_dim)
        self.num_latents = int(num_latents)

        self.locus_adapter = nn.Sequential(
            nn.LayerNorm(raw_locus_dim),
            nn.Linear(raw_locus_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.tokenizer = ObservedCpGTokenizer(model_dim, model_dim)

        self.latents = nn.Parameter(torch.empty(num_latents, model_dim))
        nn.init.normal_(self.latents, mean=0.0, std=0.02)
        self.input_cross_attention = CrossAttentionBlock(
            model_dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
        )
        self.latent_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    model_dim,
                    num_heads=num_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(num_latent_blocks)
            ]
        )

        self.target_query = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
        )
        self.output_cross_attention = CrossAttentionBlock(
            model_dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, 1),
        )
        # Before training the network exactly reproduces the train-patient locus prior.
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def encode_patient(
        self,
        observed_locus: torch.Tensor,
        observed_residual: torch.Tensor,
        observed_valid: torch.Tensor,
    ) -> torch.Tensor:
        observed_locus = self.locus_adapter(observed_locus)
        observed_tokens = self.tokenizer(observed_locus, observed_residual)
        batch = observed_tokens.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)
        latents = self.input_cross_attention(
            latents,
            observed_tokens,
            context_valid=observed_valid,
        )
        for block in self.latent_blocks:
            latents = block(latents)
        return latents

    def decode_targets(
        self,
        patient: torch.Tensor,
        target_locus: torch.Tensor,
        target_prior_logit: torch.Tensor,
    ) -> torch.Tensor:
        target_locus = self.locus_adapter(target_locus)
        query = self.target_query(target_locus)
        decoded = self.output_cross_attention(query, patient)
        delta = self.delta_head(decoded).squeeze(-1)
        return torch.sigmoid(target_prior_logit + delta)

    def forward(
        self,
        observed_locus: torch.Tensor,
        observed_residual: torch.Tensor,
        observed_valid: torch.Tensor,
        target_locus: torch.Tensor,
        target_prior_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patient = self.encode_patient(observed_locus, observed_residual, observed_valid)
        prediction = self.decode_targets(patient, target_locus, target_prior_logit)
        return prediction, patient
