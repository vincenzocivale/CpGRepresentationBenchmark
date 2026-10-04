"""Embedding loading + cosine computation for the bioval v2 launch (NEW module; read-only on stores).

Loader contract (identical to representations.hdf5_store.HDF5RepresentationStore): ``/cpg_idx`` is NOT sorted in any
store -> argsort + searchsorted; rows are cast float16 -> float32; a requested locus absent from the store raises
(no per-representation intersection); non-finite values raise.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from cpg_repr_benchmark.representations.hdf5_store import inspect_representation_h5

# CLI arm id -> id used in configs/experiments/regulatory_confirmation_matrix/matrix.yaml
ARM_ALIASES = {"regulatory_histone_dnase_v1": "regulatory_histone_dnase"}
ARMS = ("regulatory_histone_dnase_v1", "functional_annotations_pca", "cpgpt_large_locus", "deepcpg_dna_locus")
REFERENCE_ARM = "regulatory_histone_dnase_v1"
MATRIX_YAML = "configs/experiments/regulatory_confirmation_matrix/matrix.yaml"
FORBIDDEN_READ_SUBSTRINGS = ("tcga", "data/protocols")  # the TCGA test set / masking-benchmark protocols are never read


def assert_readable(path) -> None:
    s = str(path).replace("\\", "/").lower()
    for bad in FORBIDDEN_READ_SUBSTRINGS:
        if bad in s:
            raise PermissionError(f"read of {path} refused: path matches forbidden pattern {bad!r} (TCGA test set guard)")


def sha256_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


@dataclass
class ArmSpec:
    arm_id: str
    matrix_id: str
    store_h5: str
    sha256: str
    dim: int
    dtype: str
    catalog: str | None = None
    frozen_manifest: str | None = None
    extra: dict = field(default_factory=dict)


def _walk_arms(o):
    if isinstance(o, dict):
        if "id" in o and o.get("store_h5"):
            yield o
        for v in o.values():
            yield from _walk_arms(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk_arms(v)


def load_arm_specs(root: Path, arms=ARMS) -> dict[str, ArmSpec]:
    import yaml
    mx = yaml.safe_load((Path(root) / MATRIX_YAML).read_text())
    by_id = {x["id"]: x for x in _walk_arms(mx)}
    out = {}
    for a in arms:
        e = by_id[ARM_ALIASES.get(a, a)]
        out[a] = ArmSpec(a, e["id"], e["store_h5"], e["sha256"], int(e["dim"]), str(e["dtype"]),
                         e.get("catalog"), e.get("frozen_manifest"),
                         {"provenance": e.get("provenance", {}), "size_bytes": e.get("size_bytes")})
    return out


class EmbeddingStore:
    """Read-only view of one representation store restricted to requested loci."""

    def __init__(self, spec: ArmSpec, root: Path, verify_sha: bool = True):
        self.spec = spec
        self.path = Path(root) / spec.store_h5
        assert_readable(self.path)
        self.sha256_actual = sha256_file(self.path) if verify_sha else None
        if verify_sha and self.sha256_actual != spec.sha256:
            raise RuntimeError(f"{spec.arm_id}: store sha256 {self.sha256_actual} != registered {spec.sha256}")
        ids, dim, self.id_key, self.emb_key = inspect_representation_h5(self.path)
        if dim != spec.dim:
            raise RuntimeError(f"{spec.arm_id}: dim {dim} != registered {spec.dim}")
        self.ids = ids
        self.dim = dim
        self.order = np.argsort(ids)
        self.sorted_ids = ids[self.order]
        with h5py.File(self.path, "r") as h:
            self.raw_dtype = str(h[self.emb_key].dtype)
        self.cpg_idx_sorted = bool(np.all(np.diff(ids) > 0))

    def rows_for(self, cpg_ids) -> np.ndarray:
        q = np.asarray(cpg_ids, dtype=np.int64)
        pos = np.searchsorted(self.sorted_ids, q)
        ok = pos < len(self.sorted_ids)
        ok[ok] &= self.sorted_ids[pos[ok]] == q[ok]
        if not ok.all():
            raise ValueError(f"{self.spec.arm_id}: {int((~ok).sum())} requested loci absent from store "
                             f"(e.g. {q[~ok][:5].tolist()}); no per-representation intersection is allowed")
        return self.order[pos]

    def load(self, cpg_ids) -> np.ndarray:
        """float32 (n, dim) aligned to ``cpg_ids`` (may repeat); finite check. Reads only the needed rows."""
        q = np.asarray(cpg_ids, dtype=np.int64)
        rows = self.rows_for(q)
        urows, inv = np.unique(rows, return_inverse=True)
        with h5py.File(self.path, "r") as h:
            ds = h[self.emb_key]
            chunks = []
            step = 50_000
            for s in range(0, len(urows), step):
                chunks.append(np.asarray(ds[urows[s:s + step]], dtype=np.float32))
        u = np.concatenate(chunks) if chunks else np.zeros((0, self.dim), np.float32)
        if not np.isfinite(u).all():
            raise ValueError(f"{self.spec.arm_id}: non-finite embedding values among requested loci")
        return u[inv]

    def zero_row_ids(self, chunk_rows: int = 50_000) -> np.ndarray:
        """cpg_idx of rows that are exactly all-zero (full pass; also asserts every value is finite)."""
        out = []
        with h5py.File(self.path, "r") as h:
            ds = h[self.emb_key]
            for s in range(0, ds.shape[0], chunk_rows):
                a = np.asarray(ds[s:s + chunk_rows]).astype(np.float32)
                if not np.isfinite(a).all():
                    raise ValueError(f"{self.spec.arm_id}: non-finite values in rows {s}..{s + len(a)}")
                z = (a == 0).all(axis=1)
                if z.any():
                    out.append(self.ids[s:s + chunk_rows][z])
        return np.sort(np.concatenate(out)) if out else np.zeros(0, np.int64)

    def identity(self, n_zero_rows: int | None = None) -> dict:
        d = {"arm_id": self.spec.arm_id, "matrix_id": self.spec.matrix_id, "store_h5": self.spec.store_h5,
             "sha256_registered": self.spec.sha256, "sha256_actual": self.sha256_actual,
             "dim": self.dim, "dtype_on_disk": self.raw_dtype, "dtype_loaded": "float32",
             "n_rows": len(self.ids), "id_key": self.id_key, "embedding_key": self.emb_key,
             "cpg_idx_sorted": self.cpg_idx_sorted, "catalog": self.spec.catalog,
             "frozen_manifest": self.spec.frozen_manifest, "provenance": self.spec.extra.get("provenance")}
        if n_zero_rows is not None:
            d["n_all_zero_rows"] = int(n_zero_rows)
        return d


def union_zero_loci(zero_sets: dict[str, np.ndarray]) -> np.ndarray:
    """D2: loci that are all-zero in ANY store (sorted unique)."""
    if not zero_sets:
        return np.zeros(0, np.int64)
    return np.unique(np.concatenate(list(zero_sets.values())).astype(np.int64))


def exclusion_mask(cpg_cols: list[np.ndarray], zero_union: np.ndarray) -> np.ndarray:
    """True = KEEP. A pair/row is excluded if ANY of its loci is in ``zero_union`` (identical across arms)."""
    keep = np.ones(len(cpg_cols[0]), dtype=bool)
    for c in cpg_cols:
        keep &= ~np.isin(np.asarray(c, dtype=np.int64), zero_union)
    return keep


def pair_cosine(E: np.ndarray, ia: np.ndarray, ib: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Cosine of rows E[ia] vs E[ib] (float64 output), chunked. Zero-norm rows give NaN (callers exclude them)."""
    E = np.asarray(E)
    norm = np.sqrt(np.einsum("ij,ij->i", E, E, dtype=np.float64))
    out = np.empty(len(ia), dtype=np.float64)
    for s in range(0, len(ia), chunk):
        a, b = ia[s:s + chunk], ib[s:s + chunk]
        dot = np.einsum("ij,ij->i", E[a], E[b], dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[s:s + chunk] = dot / (norm[a] * norm[b])
    return out


def topk_cosine_neighbors(E: np.ndarray, k: int, chunk: int = 1000) -> np.ndarray:
    """Exact top-k cosine neighbours (self excluded) for every row of E (n, d), int32 (n, k). Rows must be nonzero."""
    E = np.ascontiguousarray(E, dtype=np.float32)
    n = len(E)
    nrm = np.linalg.norm(E, axis=1, keepdims=True)
    if (nrm == 0).any():
        raise ValueError("zero rows must be removed before kNN")
    Z = E / nrm
    out = np.empty((n, k), dtype=np.int32)
    for s in range(0, n, chunk):
        sim = Z[s:s + chunk] @ Z.T
        r = np.arange(len(sim))
        sim[r, s + r] = -np.inf
        part = np.argpartition(-sim, k, axis=1)[:, :k]
        ps = np.take_along_axis(sim, part, axis=1)
        o = np.argsort(-ps, axis=1, kind="stable")
        out[s:s + chunk] = np.take_along_axis(part, o, axis=1)
    return out
