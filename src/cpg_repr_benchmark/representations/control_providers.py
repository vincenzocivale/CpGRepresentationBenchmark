from __future__ import annotations

import numpy as np


class _SparseColumnStore:
    """Base: stores embeddings only for a given relevant subset of matrix columns."""

    def __init__(self, relevant_matrix_columns: np.ndarray, embedding: np.ndarray):
        order = np.argsort(relevant_matrix_columns)
        self._sorted_columns = np.asarray(relevant_matrix_columns, dtype=np.int64)[order]
        self._sorted_embedding = np.asarray(embedding, dtype=np.float32)[order]
        self._dim = self._sorted_embedding.shape[1]
        self.coverage_mask = None  # not meaningful without the full matrix axis; unused by callers

    @property
    def dim(self) -> int:
        return self._dim

    def get_by_matrix_columns(self, matrix_columns: np.ndarray) -> np.ndarray:
        matrix_columns = np.asarray(matrix_columns, dtype=np.int64)
        pos = np.searchsorted(self._sorted_columns, matrix_columns)
        if (pos >= len(self._sorted_columns)).any() or not np.array_equal(
            self._sorted_columns[pos], matrix_columns
        ):
            raise KeyError("requested matrix columns outside the store's relevant subset")
        return self._sorted_embedding[pos]

    def close(self) -> None:
        pass


class RandomStableStore(_SparseColumnStore):
    """Fixed per-locus random embedding, seeded by CpG id (patient-agnostic, reproducible)."""

    def __init__(self, relevant_matrix_columns: np.ndarray, relevant_cpg_ids: np.ndarray, *, dim: int = 32, seed: int = 17):
        relevant_cpg_ids = np.asarray(relevant_cpg_ids, dtype=np.int64)
        embedding = np.empty((len(relevant_cpg_ids), dim), dtype=np.float32)
        for i, cpg in enumerate(relevant_cpg_ids):
            rng = np.random.default_rng((seed, int(cpg)))
            embedding[i] = rng.normal(size=dim).astype(np.float32)
        super().__init__(relevant_matrix_columns, embedding)


class ConstantStore(_SparseColumnStore):
    """Identical embedding for every locus: an ablation with zero locus-discriminative signal."""

    def __init__(self, relevant_matrix_columns: np.ndarray, *, dim: int = 32):
        n = len(np.asarray(relevant_matrix_columns))
        embedding = np.ones((n, dim), dtype=np.float32)
        super().__init__(relevant_matrix_columns, embedding)
