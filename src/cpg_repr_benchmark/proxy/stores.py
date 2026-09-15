from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from cpg_repr_benchmark.representations.hdf5_store import inspect_representation_h5


class _MatrixIndex:
    def __init__(self, store_ids: np.ndarray, matrix_ids: np.ndarray) -> None:
        order = np.argsort(store_ids)
        sorted_ids = store_ids[order]
        matrix_ids = np.asarray(matrix_ids, dtype=np.int64)
        pos = np.searchsorted(sorted_ids, matrix_ids)
        present = pos < len(sorted_ids)
        present[present] &= sorted_ids[pos[present]] == matrix_ids[present]
        self.coverage_mask = present
        self.rows = np.full(len(matrix_ids), -1, dtype=np.int64)
        self.rows[present] = order[pos[present]]

    def store_rows(self, matrix_columns: np.ndarray) -> np.ndarray:
        columns = np.asarray(matrix_columns, dtype=np.int64)
        if not self.coverage_mask[columns].all():
            missing = columns[~self.coverage_mask[columns]][:10].tolist()
            raise ValueError(f"proxy input store misses matrix columns: {missing}")
        return self.rows[columns]


class DenseProxyInputStore:
    """Frozen dense locus features aligned to the benchmark methylation matrix.

    Only rows needed by the methylation matrix are materialized. This is important for
    genomic-FM atlases that may contain millions of loci and would otherwise require
    loading the complete atlas into RAM.
    """

    kind = "dense_hdf5"

    def __init__(
        self,
        path: Path,
        matrix_cpg_ids: np.ndarray,
        *,
        id_key: str = "auto",
        embedding_key: str = "auto",
    ) -> None:
        self.path = Path(path)
        store_ids, self.raw_dim, self.id_key, self.embedding_key = inspect_representation_h5(
            self.path,
            id_key=id_key,
            embedding_key=embedding_key,
        )
        self.index = _MatrixIndex(store_ids, matrix_cpg_ids)
        matrix_rows = np.flatnonzero(self.index.coverage_mask)
        store_rows = self.index.store_rows(matrix_rows)
        read_order = np.argsort(store_rows)
        with h5py.File(self.path, "r") as handle:
            values_sorted = np.asarray(
                handle[self.embedding_key][store_rows[read_order]],
                dtype=np.float32,
            )
        inverse = np.empty_like(read_order)
        inverse[read_order] = np.arange(len(read_order))
        self._matrix_values = np.zeros(
            (len(matrix_cpg_ids), self.raw_dim),
            dtype=np.float32,
        )
        self._matrix_values[matrix_rows] = values_sorted[inverse]

    @property
    def coverage_mask(self) -> np.ndarray:
        return self.index.coverage_mask

    def batch(
        self,
        matrix_columns: np.ndarray,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        columns = np.asarray(matrix_columns, dtype=np.int64)
        if not self.coverage_mask[columns].all():
            missing = columns[~self.coverage_mask[columns]][:10].tolist()
            raise ValueError(f"proxy input store misses matrix columns: {missing}")
        values = self._matrix_values[columns]
        return {"raw_embedding": torch.from_numpy(values).to(device)}


class FunctionalProxyInputStore:
    """Canonical CSR-like store for patient-agnostic functional annotations.

    HDF5 contract:
      /cpg_idx       int64 [N]
      /track_indices int64 [nnz]
      /track_indptr  int64 [N+1]
      /dense         float [N, dense_dim]
    """

    kind = "functional_hdf5"

    def __init__(self, path: Path, matrix_cpg_ids: np.ndarray) -> None:
        self.path = Path(path)
        with h5py.File(self.path, "r") as handle:
            required = {"cpg_idx", "track_indices", "track_indptr", "dense"}
            if not required.issubset(handle):
                missing = sorted(required - set(handle.keys()))
                raise ValueError(f"functional store missing datasets: {missing}")
            store_ids = np.asarray(handle["cpg_idx"][:], dtype=np.int64)
            self.track_indices = np.asarray(handle["track_indices"][:], dtype=np.int64)
            self.track_indptr = np.asarray(handle["track_indptr"][:], dtype=np.int64)
            self.dense = np.asarray(handle["dense"][:], dtype=np.float32)
        if self.track_indptr.shape != (len(store_ids) + 1,):
            raise ValueError("track_indptr must have shape [N+1]")
        if self.dense.shape[0] != len(store_ids):
            raise ValueError("dense rows must align with cpg_idx")
        if self.track_indptr[0] != 0 or self.track_indptr[-1] != len(self.track_indices):
            raise ValueError("track_indptr is inconsistent with track_indices")
        self.dense_dim = int(self.dense.shape[1])
        self.index = _MatrixIndex(store_ids, matrix_cpg_ids)

    @property
    def coverage_mask(self) -> np.ndarray:
        return self.index.coverage_mask

    def batch(
        self,
        matrix_columns: np.ndarray,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        rows = self.index.store_rows(matrix_columns)
        chunks = [
            self.track_indices[self.track_indptr[row] : self.track_indptr[row + 1]]
            for row in rows
        ]
        sizes = np.asarray([len(chunk) for chunk in chunks], dtype=np.int64)
        offsets = np.concatenate([np.asarray([0], dtype=np.int64), np.cumsum(sizes)])
        indices = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
        return {
            "track_indices": torch.from_numpy(indices).to(device),
            "offsets": torch.from_numpy(offsets).to(device),
            "dense": torch.from_numpy(self.dense[rows]).to(device),
        }
