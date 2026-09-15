from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


class FunctionalLocusEncoder(nn.Module):
    N_TRACKS = 4165
    DENSE_DIM = 23
    WIDTH = 256

    def __init__(self, feed_forward_residual_cls, n_functional_ffn_blocks: int = 8, dropout: float = 0.1):
        super().__init__()
        self.track_embedding = nn.EmbeddingBag(
            self.N_TRACKS, self.WIDTH, mode="mean", include_last_offset=True
        )
        self.dense_encoder = nn.Sequential(
            nn.LayerNorm(self.DENSE_DIM), nn.Linear(self.DENSE_DIM, self.WIDTH), nn.GELU()
        )
        self.locus_norm = nn.LayerNorm(self.WIDTH)
        self.functional_ffn = nn.ModuleList(
            [feed_forward_residual_cls(self.WIDTH, dropout) for _ in range(n_functional_ffn_blocks)]
        )

    def forward(self, track_indices, offsets, dense):
        h = self.locus_norm(self.track_embedding(track_indices, offsets) + self.dense_encoder(dense))
        for ffn in self.functional_ffn:
            h = ffn(h)
        return h


def _load_sibling(root: Path):
    src = Path(root) / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"MehylPredictor src directory not found: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from methylation_predictor.modeling.retrieval import FeedForwardResidual
    from methylation_predictor.storage import open_functional_locus_cache

    return FeedForwardResidual, open_functional_locus_cache


def _load_state(checkpoint: Path, model: nn.Module) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw = state.get("model_state", state.get("model", state))
    prefixes = ("track_embedding.", "dense_encoder.", "locus_norm.", "functional_ffn.")
    selected = {k: v for k, v in raw.items() if k.startswith(prefixes)}
    if not selected:
        raise ValueError(f"checkpoint {checkpoint} contains no functional-locus encoder weights")
    missing, unexpected = model.load_state_dict(selected, strict=False)
    required_missing = [k for k in missing if k.startswith(prefixes)]
    if required_missing or unexpected:
        raise ValueError(f"functional encoder state mismatch: missing={required_missing}, unexpected={unexpected}")


class FunctionalAnnotationOnlineEncoder:
    """Online version of the functional locus representation recovered from MehylPredictor."""

    def __init__(
        self,
        *,
        methylpredictor_root: Path,
        checkpoint: Path,
        locus_store: Path,
        cpg_registry: Path,
        device: str = "cuda",
        batch_size: int = 4096,
        n_functional_ffn_blocks: int = 8,
        dropout: float = 0.1,
    ):
        ff_cls, open_cache = _load_sibling(Path(methylpredictor_root))
        self.cache = open_cache(locus_store=Path(locus_store), cpg_registry=Path(cpg_registry))
        self.model = FunctionalLocusEncoder(ff_cls, n_functional_ffn_blocks, dropout)
        _load_state(Path(checkpoint), self.model)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"online functional provider requested {device}, but CUDA is unavailable")
        self.device = torch.device(device)
        self.model = self.model.to(self.device).eval()
        self.batch_size = int(batch_size)

    @property
    def dim(self) -> int:
        return self.model.WIDTH

    @torch.no_grad()
    def encode(self, cpg_ids: np.ndarray) -> np.ndarray:
        cpg_ids = np.asarray(cpg_ids, dtype=np.int64)
        chunks = []
        for start in range(0, len(cpg_ids), self.batch_size):
            ids = cpg_ids[start : start + self.batch_size]
            batch = self.cache.get(ids)
            track_indices = torch.from_numpy(batch["track_indices"]).to(self.device)
            offsets = torch.from_numpy(batch["offsets"]).to(self.device)
            dense = torch.from_numpy(batch["dense"].astype(np.float32)).to(self.device)
            chunks.append(self.model(track_indices, offsets, dense).float().cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, self.dim), dtype=np.float32)


def build_functional_online_provider(*, repo_root=None, registry_path=None, **kwargs):
    """Factory for ``representation.mode: online``.

    Config may omit ``methylpredictor_root`` when MehylPredictor is a sibling of this repo,
    and may omit ``cpg_registry`` when the dataset registry is already configured.
    """
    repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
    kwargs.setdefault("methylpredictor_root", repo_root.parent / "MehylPredictor")
    if registry_path is not None:
        kwargs.setdefault("cpg_registry", Path(registry_path))
    for key in ("methylpredictor_root", "checkpoint", "locus_store", "cpg_registry"):
        if key not in kwargs:
            raise ValueError(f"functional online provider requires {key}")
        kwargs[key] = Path(kwargs[key])
    return FunctionalAnnotationOnlineEncoder(**kwargs)
