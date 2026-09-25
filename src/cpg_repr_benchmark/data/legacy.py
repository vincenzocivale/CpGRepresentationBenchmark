from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .coordinates import encode_many


def legacy_ids_to_coordinate_ids(legacy_ids: np.ndarray, registry_path: Path) -> np.ndarray:
    """Translate a source-specific legacy CpG axis through an explicit chr/pos registry.

    This is a migration helper only.  New processed datasets and representation caches
    should store coordinate-native IDs directly.
    """
    ids = np.asarray(legacy_ids, dtype=np.int64)
    registry = pd.read_parquet(registry_path, columns=["cpg_idx", "chr", "pos"])
    if registry["cpg_idx"].duplicated().any():
        raise ValueError(f"legacy registry has duplicate cpg_idx: {registry_path}")
    aligned = registry.set_index("cpg_idx").reindex(ids)
    if aligned[["chr", "pos"]].isna().any().any():
        missing = ids[aligned["chr"].isna().to_numpy()]
        raise KeyError(f"registry misses {len(missing)} legacy CpGs; examples={missing[:10].tolist()}")
    canonical = encode_many(aligned["chr"], aligned["pos"])
    if len(canonical) != len(np.unique(canonical)):
        raise ValueError("legacy axis collapses to duplicate GRCh38 CpG coordinates")
    return canonical


def legacy_ids_to_coordinate_ids_lenient(legacy_ids: np.ndarray, registry_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Lenient counterpart to :func:`legacy_ids_to_coordinate_ids`: drops legacy IDs the
    registry has no chr/pos for instead of raising.

    For diagnostic probes (e.g. `bio_validation`) that only ever *read* a representation's
    embedding against external annotations, silently narrowing to the registry's coverage is
    an acceptable, explicit tradeoff — unlike the masking benchmark's evaluation universe,
    which `legacy_ids_to_coordinate_ids` must keep strict per the "never silently narrow the
    universe" invariant. Returns `(canonical_ids, keep_mask)`; `keep_mask` is aligned to the
    input order so the caller can subset any parallel array (e.g. embedding rows) the same way.
    """
    ids = np.asarray(legacy_ids, dtype=np.int64)
    registry = pd.read_parquet(registry_path, columns=["cpg_idx", "chr", "pos"])
    if registry["cpg_idx"].duplicated().any():
        raise ValueError(f"legacy registry has duplicate cpg_idx: {registry_path}")
    aligned = registry.set_index("cpg_idx").reindex(ids)
    keep_mask = aligned[["chr", "pos"]].notna().all(axis=1).to_numpy()
    canonical = encode_many(aligned.loc[keep_mask, "chr"], aligned.loc[keep_mask, "pos"])
    if len(canonical) != len(np.unique(canonical)):
        raise ValueError("legacy axis collapses to duplicate GRCh38 CpG coordinates")
    return canonical, keep_mask


def coordinate_ids_to_legacy_ids(coordinate_ids: np.ndarray, registry_path: Path) -> np.ndarray:
    """Inverse of :func:`legacy_ids_to_coordinate_ids`.

    Positions with no legacy counterpart in the registry get ``-1`` rather than raising,
    since callers typically probe a superset axis (e.g. a downstream dataset's full CpG
    list) and only need the legacy ID at the positions that do overlap.
    """
    ids = np.asarray(coordinate_ids, dtype=np.int64)
    registry = pd.read_parquet(registry_path, columns=["cpg_idx", "chr", "pos"])
    if registry["cpg_idx"].duplicated().any():
        raise ValueError(f"legacy registry has duplicate cpg_idx: {registry_path}")
    canonical = encode_many(registry["chr"], registry["pos"])
    if len(canonical) != len(np.unique(canonical)):
        raise ValueError("legacy registry collapses to duplicate GRCh38 CpG coordinates")
    lookup = pd.Series(registry["cpg_idx"].to_numpy(), index=canonical)
    return lookup.reindex(ids).fillna(-1).to_numpy(dtype=np.int64)
