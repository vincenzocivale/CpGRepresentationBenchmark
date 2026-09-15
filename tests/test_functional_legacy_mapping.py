import numpy as np

from cpg_repr_benchmark.data.coordinates import encode_cpg_id


def test_coordinate_namespace_is_sufficient_for_legacy_registry_mapping():
    chrom = ["chr1", "chr2"]
    pos = [123, 456]
    canonical = np.asarray(
        [encode_cpg_id(c, p) for c, p in zip(chrom, pos)],
        dtype=np.int64,
    )
    assert canonical.tolist() == [1_000_000_123, 2_000_000_456]
    assert len(np.unique(canonical)) == 2
