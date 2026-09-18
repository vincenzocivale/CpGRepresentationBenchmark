from __future__ import annotations

import torch
from torch import nn


class DownstreamBetaMLP(nn.Module):
    """Frozen oracle downstream predictor: fixed-length regressor over raw beta values.

    Consumes the real/reconstructed beta vector restricted to the canonical CpG universe U
    (NOT embeddings) and regresses a phenotype (here, age). Trained once on real complete
    methylomes and reused, frozen, for every representation/observed-count arm.
    """

    def __init__(self, n_cpg: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_cpg),
            nn.Linear(n_cpg, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        return self.net(beta).squeeze(-1)
