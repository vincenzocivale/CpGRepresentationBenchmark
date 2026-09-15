from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


class RepresentationCoverageError(ValueError):
    pass


ID_KEY_CANDIDATES = ("cpg_idx", "cpg_ids", "ids", "id")
EMBEDDING_KEY_CANDIDATES = ("embedding", "embeddings", "emb", "features")


def _resolve_dataset_key(handle: h5py.File, requested: str | None, candidates: tuple[str, ...], kind: str) -> str:
    if requested and requested != "auto":
        if requested not in handle:
            raise RepresentationCoverageError(
                f"requested {kind} dataset /{requested} is absent; available root datasets={list(handle.keys())}"
            )
        return requested
    matches = [key for key in candidates if key in handle and isinstance(handle[key], h5py.Dataset)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer the canonical name when it exists; otherwise ambiguity should be explicit.
        canonical = candidates[0]
        if canonical in matches:
            return canonical
        raise RepresentationCoverageError(
            f"ambiguous {kind} dataset in HDF5: {matches}; set representation.{kind}_key explicitly"
        )
    raise RepresentationCoverageError(
        f"cannot find a {kind} dataset; tried {list(candidates)}; available root datasets={list(handle.keys())}"
    )


def inspect_representation_h5(
    path: Path,
    *,
    id_key: str | None = "auto",
    embedding_key: str | None = "auto",
) -> tuple[np.ndarray, int, str, str]:
    """Validate an HDF5 representation store and return IDs/dim plus resolved dataset keys.

    Canonical stores use /cpg_idx and /embedding. Existing atlases may use common aliases
    such as /ids and /emb; these can be auto-detected or configured explicitly.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        resolved_id_key = _resolve_dataset_key(handle, id_key, ID_KEY_CANDIDATES, "id")
        resolved_embedding_key = _resolve_dataset_key(
            handle, embedding_key, EMBEDDING_KEY_CANDIDATES, "embedding"
        )
        cpg_ids = np.asarray(handle[resolved_id_key][:], dtype=np.int64)
        embedding = handle[resolved_embedding_key]
        if cpg_ids.ndim != 1:
            raise RepresentationCoverageError(
                f"/{resolved_id_key} must be 1-D, got shape={cpg_ids.shape} in {path}"
            )
        if embedding.ndim != 2 or embedding.shape[0] != len(cpg_ids):
            raise RepresentationCoverageError(
                f"invalid /{resolved_embedding_key} shape in {path}: {embedding.shape}; "
                f"expected ({len(cpg_ids)}, D)"
            )
        dim = int(embedding.shape[1])
    if len(cpg_ids) != len(np.unique(cpg_ids)):
        raise RepresentationCoverageError(f"{path} contains duplicate CpG IDs in /{resolved_id_key}")
    return cpg_ids, dim, resolved_id_key, resolved_embedding_key


def validate_canonical_h5(path: Path) -> tuple[np.ndarray, int]:
    cpg_ids, dim, id_key, embedding_key = inspect_representation_h5(
        path, id_key="cpg_idx", embedding_key="embedding"
    )
    if id_key != "cpg_idx" or embedding_key != "embedding":  # defensive; explicit keys above should guarantee this
        raise RepresentationCoverageError(f"{path} is not canonical")
    return cpg_ids, dim


class HDF5RepresentationStore:
    """CpG representation store mapped into methylation-matrix column order."""

    def __init__(
        self,
        path: Path,
        matrix_cpg_ids: np.ndarray,
        *,
        id_key: str | None = "auto",
        embedding_key: str | None = "auto",
    ):
        self.path = Path(path)
        store_ids, self._dim, self.id_key, self.embedding_key = inspect_representation_h5(
            self.path, id_key=id_key, embedding_key=embedding_key
        )
        order = np.argsort(store_ids)
        sorted_ids = store_ids[order]
        matrix_cpg_ids = np.asarray(matrix_cpg_ids, dtype=np.int64)
        pos = np.searchsorted(sorted_ids, matrix_cpg_ids)
        present = pos < len(sorted_ids)
        present[present] &= sorted_ids[pos[present]] == matrix_cpg_ids[present]
        self.coverage_mask = present
        store_rows = np.full(len(matrix_cpg_ids), -1, dtype=np.int64)
        store_rows[present] = order[pos[present]]

        # Random per-sample fancy-indexing against the on-disk store is extremely slow for
        # large, coarsely chunked embedding tables (each scattered row access can pull in an
        # entire multi-MB chunk). Load the (small) subset of rows this benchmark actually
        # needs into RAM once, aligned to matrix-column order, so __getitem__ never touches
        # the HDF5 file again.
        present_idx = np.flatnonzero(present)
        needed_store_rows = store_rows[present_idx]
        read_order = np.argsort(needed_store_rows)
        with h5py.File(self.path, "r") as handle:
            sorted_values = np.asarray(
                handle[self.embedding_key][needed_store_rows[read_order]], dtype=np.float32
            )
        inverse = np.empty_like(read_order)
        inverse[read_order] = np.arange(len(read_order))
        self._embedding_matrix = np.zeros((len(matrix_cpg_ids), self._dim), dtype=np.float32)
        self._embedding_matrix[present_idx] = sorted_values[inverse]

    @property
    def dim(self) -> int:
        return self._dim

    def get_by_matrix_columns(self, matrix_columns: np.ndarray) -> np.ndarray:
        matrix_columns = np.asarray(matrix_columns, dtype=np.int64)
        if not self.coverage_mask[matrix_columns].all():
            bad = matrix_columns[~self.coverage_mask[matrix_columns]][:10].tolist()
            raise RepresentationCoverageError(f"representation is missing requested matrix columns: {bad}")
        return self._embedding_matrix[matrix_columns]

    def close(self) -> None:
        pass
