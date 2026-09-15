from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from .model import DeepSetsPatientEncoder, ObservedCpGTokenizer

TaskType = Literal["regression", "binary", "multiclass"]


class MethylomeTaskModel(nn.Module):
    """Shared downstream backbone for age, mortality and disease tasks.

    The architecture intentionally mirrors the observed-CpG side of
    :class:`MaskedMethylomeReconstructor`: raw locus adapter -> locus/value tokenization ->
    DeepSets patient encoder. Only the final task head changes. Therefore, within a task,
    the representation store is the only representation-specific input.
    """

    def __init__(
        self,
        *,
        raw_locus_dim: int,
        task_type: TaskType,
        output_dim: int,
        locus_latent_dim: int = 256,
        token_dim: int = 256,
        patient_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if task_type == "binary" and output_dim != 1:
            raise ValueError("binary tasks require output_dim=1")
        if task_type == "multiclass" and output_dim < 2:
            raise ValueError("multiclass tasks require output_dim>=2")
        if task_type == "regression" and output_dim != 1:
            raise ValueError("regression tasks require output_dim=1")
        self.task_type = task_type
        self.locus_adapter = nn.Sequential(
            nn.LayerNorm(raw_locus_dim),
            nn.Linear(raw_locus_dim, locus_latent_dim),
            nn.GELU(),
            nn.LayerNorm(locus_latent_dim),
        )
        self.tokenizer = ObservedCpGTokenizer(locus_latent_dim, token_dim)
        self.patient_encoder = DeepSetsPatientEncoder(token_dim, patient_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(patient_dim),
            nn.Linear(patient_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        observed_locus: torch.Tensor,
        observed_residual: torch.Tensor,
        observed_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        locus = self.locus_adapter(observed_locus)
        tokens = self.tokenizer(locus, observed_residual)
        patient = self.patient_encoder(tokens, observed_valid)
        output = self.head(patient)
        if self.task_type in {"binary", "regression"}:
            output = output.squeeze(-1)
        return output, patient
