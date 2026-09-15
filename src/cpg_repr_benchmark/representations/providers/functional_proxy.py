from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cpg_repr_benchmark.proxy.model import FunctionalAnnotationEncoder, MeanMethylationProxy
from cpg_repr_benchmark.proxy.stores import FunctionalProxyInputStore


class FunctionalProxyProvider:
    """Frozen proxy-aligned functional representation over a canonical raw feature store."""

    name = "functional_proxy_aligned"

    def __init__(
        self,
        *,
        raw_store_h5: Path,
        proxy_checkpoint: Path,
        requested_cpg_ids: np.ndarray,
        device: str,
        n_tracks: int = 4165,
        embedding_dim: int = 256,
        n_blocks: int = 8,
        dropout: float = 0.1,
    ) -> None:
        self.requested_cpg_ids = np.asarray(requested_cpg_ids, dtype=np.int64)
        self.store = FunctionalProxyInputStore(Path(raw_store_h5), self.requested_cpg_ids)
        if not self.store.coverage_mask.all():
            missing = self.requested_cpg_ids[~self.store.coverage_mask]
            raise ValueError(
                f"functional raw store misses {len(missing)} requested loci; examples={missing[:10].tolist()}"
            )
        encoder = FunctionalAnnotationEncoder(
            n_tracks=int(n_tracks),
            dense_dim=int(self.store.dense_dim),
            latent_dim=int(embedding_dim),
            n_blocks=int(n_blocks),
            dropout=float(dropout),
        )
        self.model = MeanMethylationProxy(encoder, encoder.output_dim)
        state = torch.load(Path(proxy_checkpoint), map_location="cpu", weights_only=False)
        raw_state = state.get("model", state.get("model_state", state))
        self.model.load_state_dict(raw_state)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested {device}, but CUDA is unavailable")
        self.device = torch.device(device)
        self.model = self.model.to(self.device).eval()
        self.dim = int(encoder.output_dim)
        self.raw_store_h5 = Path(raw_store_h5)
        self.proxy_checkpoint = Path(proxy_checkpoint)
        order = np.argsort(self.requested_cpg_ids)
        self._sorted_ids = self.requested_cpg_ids[order]
        self._sorted_columns = order.astype(np.int64)

    def _columns(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        slots = np.searchsorted(self._sorted_ids, ids)
        present = slots < len(self._sorted_ids)
        rows = np.flatnonzero(present)
        present[rows] &= self._sorted_ids[slots[rows]] == ids[rows]
        if not present.all():
            missing = ids[~present]
            raise ValueError(f"functional proxy provider received unexpected loci: {missing[:10].tolist()}")
        return self._sorted_columns[slots]

    @torch.no_grad()
    def encode(self, cpg_idx: np.ndarray, chrom: np.ndarray, pos: np.ndarray) -> np.ndarray:
        del chrom, pos
        columns = self._columns(cpg_idx)
        return self.model.encode(self.store.batch(columns, self.device)).float().cpu().numpy()

    def metadata(self) -> dict:
        return {
            "track": "proxy_aligned",
            "component": "locus_only",
            "proxy_checkpoint": str(self.proxy_checkpoint),
            "raw_functional_store": str(self.raw_store_h5),
            "proxy_objective": "train_patient_mean_beta",
        }

    def close(self) -> None:
        pass
