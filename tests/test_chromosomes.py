from cpg_repr_benchmark.data.chromosomes import normalize_chromosome


def test_normalize_chromosome_common_encodings():
    assert normalize_chromosome("chr1") == "chr1"
    assert normalize_chromosome("1") == "chr1"
    assert normalize_chromosome(1) == "chr1"
    assert normalize_chromosome("X") == "chrX"
    assert normalize_chromosome("chrX") == "chrX"
    assert normalize_chromosome("MT") == "chrM"
