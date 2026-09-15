from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cpg_repr_benchmark.data.coordinates import encode_many


class LegacyFunctionalProvider:
    """Coordinate adapter around the historical functional-locus checkpoint.

    This provider exists only for smoke/regression checks.  It resolves canonical
    GRCh38 loci through the historical array registry and then asks the original
    functional provider to encode those legacy CpG IDs.  It must not be used for
    strict external-locus claims because loci absent from the legacy registry are
    intentionally unsupported.
    """

    name = "functional_legacy_checkpoint"

    def __init__(
        self,
        *,
        legacy_registry: Path,
        methylpredictor_root: Path,
        checkpoint: Path,
        locus_store: Path,
        device: str,
        encoder_batch_size: int,
        n_functional_ffn_blocks: int = 8,
        dropout: float = 0.1,
    ) -> None:
        from cpg_repr_benchmark.representations.functional_provider import (
            FunctionalAnnotationOnlineEncoder,
        )

        table = pd.read_parquet(legacy_registry, columns=["cpg_idx", "chr", "pos"])
        canonical = encode_many(table["chr"].astype(str), table["pos"].astype(int))
        legacy = table["cpg_idx"].to_numpy(dtype=np.int64)
        order = np.argsort(canonical)
        self._canonical = canonical[order]
        self._legacy = legacy[order]
        if len(self._canonical) != len(np.unique(self._canonical)):
            raise ValueError("legacy functional registry contains duplicate genomic coordinates")

        self.encoder = FunctionalAnnotationOnlineEncoder(
            methylpredictor_root=Path(methylpredictor_root),
            checkpoint=Path(checkpoint),
            locus_store=Path(locus_store),
            cpg_registry=Path(legacy_registry),
            device=device,
            batch_size=int(encoder_batch_size),
            n_functional_ffn_blocks=int(n_functional_ffn_blocks),
            dropout=float(dropout),
        )
        self.dim = int(self.encoder.dim)
        self.legacy_registry = Path(legacy_registry)
        self.checkpoint = Path(checkpoint)
        self.locus_store = Path(locus_store)

    def _legacy_ids(self, canonical_ids: np.ndarray) -> np.ndarray:
        canonical_ids = np.asarray(canonical_ids, dtype=np.int64)
        slots = np.searchsorted(self._canonical, canonical_ids)
        present = slots < len(self._canonical)
        valid_rows = np.flatnonzero(present)
        present[valid_rows] &= self._canonical[slots[valid_rows]] == canonical_ids[valid_rows]
        if not present.all():
            missing = canonical_ids[~present]
            raise ValueError(
                "legacy functional provider cannot resolve "
                f"{len(missing)} canonical loci; examples={missing[:10].tolist()}. "
                "This provider is smoke-test-only; use a coordinate-native functional raw store "
                "for external-locus experiments."
            )
        return self._legacy[slots]

    def encode(self, cpg_idx: np.ndarray, chrom: np.ndarray, pos: np.ndarray) -> np.ndarray:
        del chrom, pos
        return self.encoder.encode(self._legacy_ids(cpg_idx))

    def metadata(self) -> dict:
        return {
            "track": "legacy",
            "component": "locus_only",
            "checkpoint": str(self.checkpoint),
            "legacy_registry": str(self.legacy_registry),
            "locus_store": str(self.locus_store),
            "publication_ranking": False,
        }

    def close(self) -> None:
        close = getattr(self.encoder, "close", None)
        if callable(close):
            close()
