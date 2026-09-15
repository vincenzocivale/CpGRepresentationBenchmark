import numpy as np

from cpg_repr_benchmark.data.coordinates import (
    CPG_ID_STRIDE,
    CPG_NAMESPACE,
    chromosome_from_ids,
    decode_cpg_id,
    encode_cpg_id,
    normalize_chromosome,
)


def test_coordinate_ids_are_deterministic_and_reversible():
    value = encode_cpg_id("1", 15865)
    assert value == CPG_ID_STRIDE + 15865
    assert encode_cpg_id("chr1", 15865) == value
    assert decode_cpg_id(value) == ("chr1", 15865)
    assert CPG_NAMESPACE == "grch38_cpg_cytosine_1based_v1"


def test_chromosome_normalization_and_vector_decode():
    assert normalize_chromosome("X") == "chrX"
    assert normalize_chromosome("MT") == "chrM"
    ids = np.array([encode_cpg_id("chr1", 10), encode_cpg_id("chrX", 20)])
    assert chromosome_from_ids(ids).tolist() == ["chr1", "chrX"]
