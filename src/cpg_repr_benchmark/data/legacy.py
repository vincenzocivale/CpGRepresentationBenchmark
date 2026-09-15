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
