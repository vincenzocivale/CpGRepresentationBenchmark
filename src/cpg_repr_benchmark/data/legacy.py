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
