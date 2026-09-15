import pandas as pd

from cpg_repr_benchmark.data.coordinates import encode_cpg_id
from cpg_repr_benchmark.data.master_registry import build_master_registry


def test_master_registry_deduplicates_coordinates_across_sources():
    a = pd.DataFrame({"chr": ["chr1", "chr2"], "pos": [10, 20]})
    b = pd.DataFrame({"chr": ["1", "chr3"], "pos": [10, 30]})
    registry, membership = build_master_registry({"A": a, "B": b})
    assert len(registry) == 3
    assert len(membership) == 4
    row = registry.loc[registry["cpg_idx"] == encode_cpg_id("chr1", 10)].iloc[0]
    assert row["source_count"] == 2
