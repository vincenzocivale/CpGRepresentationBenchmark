"""Exact (chr, pos) join between the benchmark universe (global cpg_idx) and a coordinate-native cohort (torch-free)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def exact_coordinate_join(
    universe_chr, universe_pos, cohort_chr, cohort_pos
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (universe_rows, cohort_cols, unmapped_universe_rows).

    ``universe_rows``/``cohort_cols`` are aligned pairs with identical (chr, pos), ordered by universe row (universe
    order is preserved). Both coordinate sets must be duplicate-free (asserted). No tolerance, no offset.
    """
    u = pd.DataFrame({"chr": np.asarray(universe_chr, dtype=str), "pos": np.asarray(universe_pos, dtype=np.int64)})
    c = pd.DataFrame({"chr": np.asarray(cohort_chr, dtype=str), "pos": np.asarray(cohort_pos, dtype=np.int64)})
    if u.duplicated(["chr", "pos"]).any():
        raise ValueError("universe has duplicate (chr, pos)")
    if c.duplicated(["chr", "pos"]).any():
        raise ValueError("cohort has duplicate (chr, pos); collapse multi-probe loci before the join")
    u["urow"] = np.arange(len(u), dtype=np.int64)
    c["ccol"] = np.arange(len(c), dtype=np.int64)
    m = u.merge(c, on=["chr", "pos"], how="left").sort_values("urow", kind="stable")
    hit = m["ccol"].notna().to_numpy()
    return (
        m["urow"].to_numpy()[hit].astype(np.int64),
        m["ccol"].to_numpy()[hit].astype(np.int64),
        m["urow"].to_numpy()[~hit].astype(np.int64),
    )
