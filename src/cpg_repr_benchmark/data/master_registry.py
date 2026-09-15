from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .coordinates import CPG_NAMESPACE, COORDINATE_CONVENTION, REFERENCE_BUILD, encode_many, normalize_chromosome


def canonicalize_coordinate_table(table: pd.DataFrame, *, source: str) -> pd.DataFrame:
    required = {"chr", "pos"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"source {source!r} is missing columns: {sorted(missing)}")

    out = table.loc[:, ["chr", "pos"]].copy()
    out = out.dropna(subset=["chr", "pos"])
    out["chr"] = out["chr"].map(normalize_chromosome)
    out["pos"] = pd.to_numeric(out["pos"], errors="raise").astype("int64")
    out["cpg_idx"] = encode_many(out["chr"], out["pos"])
    out["source"] = str(source)
    return out.drop_duplicates(["source", "cpg_idx"])


def build_master_registry(sources: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not sources:
        raise ValueError("at least one coordinate source is required")
    membership = pd.concat(
        [canonicalize_coordinate_table(frame, source=name) for name, frame in sources.items()],
        ignore_index=True,
    )
    membership = membership.loc[:, ["cpg_idx", "chr", "pos", "source"]].drop_duplicates()

    coordinate_counts = membership.groupby("cpg_idx")[["chr", "pos"]].nunique()
    inconsistent = coordinate_counts[(coordinate_counts["chr"] > 1) | (coordinate_counts["pos"] > 1)]
    if len(inconsistent):
        raise ValueError(f"canonical ID collision across {len(inconsistent)} loci")

    registry = membership.loc[:, ["cpg_idx", "chr", "pos"]].drop_duplicates("cpg_idx").copy()
    registry["chr_pos"] = registry["chr"] + ":" + registry["pos"].astype(str)
    registry["reference_build"] = REFERENCE_BUILD
    registry["coordinate_convention"] = COORDINATE_CONVENTION
    registry["cpg_namespace"] = CPG_NAMESPACE
    counts = membership.groupby("cpg_idx")["source"].nunique().rename("source_count")
    registry = registry.join(counts, on="cpg_idx")
    registry = registry.sort_values("cpg_idx", kind="stable").reset_index(drop=True)
    membership = membership.sort_values(["cpg_idx", "source"], kind="stable").reset_index(drop=True)
    return registry, membership
