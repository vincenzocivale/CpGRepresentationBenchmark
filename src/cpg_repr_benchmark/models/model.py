from __future__ import annotations

import torch
from torch import nn


class ObservedCpGTokenizer(nn.Module):
    def __init__(self, locus_dim: int, token_dim: int):
        super().__init__()
        self.locus = nn.Sequential(nn.Linear(locus_dim, token_dim), nn.GELU())
        self.value = nn.Sequential(nn.Linear(1, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, locus: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.norm(self.locus(locus) + self.value(residual.unsqueeze(-1)))


class DeepSetsPatientEncoder(nn.Module):
    def __init__(self, token_dim: int, patient_dim: int, hidden_dim: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(token_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, patient_dim), nn.GELU()
        )
        self.rho = nn.Sequential(
            nn.LayerNorm(patient_dim), nn.Linear(patient_dim, patient_dim), nn.GELU(), nn.Linear(patient_dim, patient_dim)
        )

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        h = self.phi(tokens)
        weights = valid.unsqueeze(-1).to(h.dtype)
        pooled = (h * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.rho(pooled)


class ResidualQueryDecoder(nn.Module):
    def __init__(self, locus_dim: int, patient_dim: int, hidden_dim: int):
        super().__init__()
        self.locus_proj = nn.Sequential(nn.LayerNorm(locus_dim), nn.Linear(locus_dim, patient_dim), nn.GELU())
        self.net = nn.Sequential(
            nn.Linear(patient_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # An untrained model must exactly reproduce the prior baseline.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, patient: torch.Tensor, target_locus: torch.Tensor) -> torch.Tensor:
        query = self.locus_proj(target_locus)
        p = patient.unsqueeze(1).expand(-1, query.shape[1], -1)
        return self.net(torch.cat([p, query], dim=-1)).squeeze(-1)


class MaskedMethylomeReconstructor(nn.Module):
    """Representation-controlled methylome reconstructor.

    The same architecture is used for every representation arm. `encode_patient` and
    `decode_targets` are exposed so a sparse patient profile can be encoded once and an
    arbitrarily large missing locus set can be decoded in chunks at downstream time.
    """

    def __init__(
        self,
        raw_locus_dim: int,
        locus_latent_dim: int = 256,
        token_dim: int = 256,
        patient_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.locus_adapter = nn.Sequential(
            nn.LayerNorm(raw_locus_dim),
            nn.Linear(raw_locus_dim, locus_latent_dim),
            nn.GELU(),
            nn.LayerNorm(locus_latent_dim),
        )
        self.tokenizer = ObservedCpGTokenizer(locus_latent_dim, token_dim)
        self.patient_encoder = DeepSetsPatientEncoder(token_dim, patient_dim, hidden_dim)
        self.decoder = ResidualQueryDecoder(locus_latent_dim, patient_dim, hidden_dim)

    def encode_patient(
        self,
        observed_locus: torch.Tensor,
        observed_residual: torch.Tensor,
        observed_valid: torch.Tensor,
    ) -> torch.Tensor:
        observed_locus = self.locus_adapter(observed_locus)
        tokens = self.tokenizer(observed_locus, observed_residual)
        return self.patient_encoder(tokens, observed_valid)

    def decode_targets(
        self,
        patient: torch.Tensor,
        target_locus: torch.Tensor,
        target_prior_logit: torch.Tensor,
    ) -> torch.Tensor:
        target_locus = self.locus_adapter(target_locus)
        delta = self.decoder(patient, target_locus)
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


def build_reconstructor(raw_locus_dim: int, model_cfg: dict) -> nn.Module:
    """Build the configured reconstruction architecture.

    `deepsets` keeps the original benchmark model available as an architectural baseline;
    `perceiver_io` is the target-conditioned sparse reconstruction model.
    """
    architecture = str(model_cfg.get("architecture", "deepsets")).lower()
    if architecture in {"deepsets", "masked_methylome", "legacy"}:
        return MaskedMethylomeReconstructor(
            raw_locus_dim=raw_locus_dim,
            locus_latent_dim=int(model_cfg.get("locus_latent_dim", 256)),
            token_dim=int(model_cfg.get("token_dim", 256)),
            patient_dim=int(model_cfg.get("patient_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        )
    if architecture in {"perceiver", "perceiver_io"}:
        from cpg_repr_benchmark.models.perceiver import PerceiverMaskedMethylomeReconstructor

        return PerceiverMaskedMethylomeReconstructor(
            raw_locus_dim=raw_locus_dim,
            model_dim=int(model_cfg.get("model_dim", 256)),
            num_latents=int(model_cfg.get("num_latents", 64)),
            num_latent_blocks=int(model_cfg.get("num_latent_blocks", 4)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ff_mult=int(model_cfg.get("ff_mult", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    raise ValueError(f"unsupported reconstruction architecture: {architecture!r}")
