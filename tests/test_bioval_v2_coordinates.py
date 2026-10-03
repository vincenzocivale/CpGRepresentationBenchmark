from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cpg_repr_benchmark.biological_validation_v2 import coordinates as co


def test_normalize_chrom():
    assert co.normalize_chrom("1") == "chr1"
    assert co.normalize_chrom("CHRX") == "chrX"
    assert co.normalize_chrom("chr23") == "chrX"
    assert co.normalize_chrom("MT") == "chrM"
    assert co.normalize_chrom("chrUn_gl000220") is None


def test_zero_to_one_based_and_dedup():
    df = pd.DataFrame({"c": ["1", "chr1", "chr1", "weird"], "s": [99, 99, 199, 5]})
    out, rep = co.harmonize(df, "c", "s", "hg38", pos_base=0)
    assert out["pos"].tolist() == [100, 200]
    assert rep["n_in"] == 4 and rep["n_bad_chrom"] == 1 and rep["n_dupe"] == 1 and rep["n_out"] == 2
    assert rep["build_out"] == "GRCh38"


def _fake_chain(path):
    # hg19 chr1 [0,1000) -> hg38 chr1 [500,1500); chr2 [0,1000) -> chr3 [0,1000), size 1000
    path.write_text(
        "chain 1000 chr1 2000 + 0 1000 chr1 5000 + 500 1500 1\n1000\n\n"
        "chain 1000 chr2 2000 + 0 1000 chr3 5000 + 0 1000 2\n1000\n\n"
        "chain 100 chr4 2000 + 0 100 chr4 5000 + 0 100 3\n100\n\n"
        "chain 100 chr4 2000 + 0 100 chr4 5000 + 0 100 4\n100\n\n")
    return path


def test_liftover_fake_chain(tmp_path):
    pytest.importorskip("pyliftover")
    chain = _fake_chain(tmp_path / "x.chain")
    df = pd.DataFrame({"chrom": ["chr1", "chr1", "chr2", "chr1"], "pos": [10, 20, 7, 1500]})
    out, rep = co.harmonize(df, "chrom", "pos", "hg19", chain_path=chain)
    assert out[["chrom", "pos"]].values.tolist() == [["chr1", 510], ["chr1", 520], ["chr3", 7]]
    assert rep["n_unmapped"] == 1 and rep["n_out"] == 3 and rep["build_in"] == "GRCh37"


def test_liftover_intervals(tmp_path):
    pytest.importorskip("pyliftover")
    chain = _fake_chain(tmp_path / "x.chain")
    df = pd.DataFrame({"chrom": ["chr1", "chr1"], "s": [10, 10], "e": [20, 1400]})
    out, rep = co.harmonize(df, "chrom", "s", "hg19", end_col="e", chain_path=chain)
    assert len(out) == 1 and out["pos"].iloc[0] == 510 and out["end"].iloc[0] == 520
    assert rep["n_unmapped"] == 1


def test_overlap_audit_counts():
    u = pd.DataFrame({"cpg_idx": [1, 2, 3, 4], "chrom": ["chr1", "chr1", "chr2", "chr2"],
                      "pos": [10, 20, 10, 20]})
    df = pd.DataFrame({"chrom": ["chr1", "chr2", "chr2"], "pos": [10, 20, 99]})
    a = co.overlap_audit(df, u)
    assert a["n_exact_matches"] == 2 and a["frac_universe_covered"] == 0.5
    assert abs(a["frac_df_in_universe"] - 2 / 3) < 1e-9
    assert a["per_chrom"].set_index("chrom").loc["chr2", "n_df_in_universe"] == 1
    assert co.overlap_audit(df, u, tol_bp=1)["n_exact_matches"] == 2
    assert co.overlap_audit(df, u, tol_bp=100)["n_exact_matches"] == 3


def test_require_full_universe_coverage():
    u = pd.DataFrame({"cpg_idx": [1, 2, 3]})
    assert co.require_full_universe_coverage(np.array([3, 2, 1, 9]), u)
    with pytest.raises(ValueError):
        co.require_full_universe_coverage(np.array([1, 2]), u)


def test_universe_fixed_length_if_data_present():
    try:
        u = co.load_benchmark_universe()
    except FileNotFoundError:
        pytest.skip("registry not available")
    assert len(u) == co.UNIVERSE_SIZE == 408_399
    assert u["cpg_idx"].is_unique


def test_coord_id_roundtrip_and_context_mapping():
    from cpg_repr_benchmark.biological_validation_v2.coordinates import (
        decode_coord_id,
        encode_coord_id,
        map_genomic_context,
    )

    chroms = ["chr1", "chr16", "chrX", "chrY", "chr22"]
    pos = np.array([15865, 53434200, 24072640, 5, 249_000_000])
    ids = encode_coord_id(chroms, pos)
    assert ids[1] == 16053434200 and ids[2] == 23_024_072_640
    c2, p2 = decode_coord_id(ids)
    assert list(c2) == chroms and (p2 == pos).all()

    universe = pd.DataFrame({"cpg_idx": [7, 8, 9], "chrom": ["chr16", "chrX", "chr2"],
                             "pos": [53434200, 24072640, 10]})
    ctx = pd.DataFrame({"cpg_idx": ids, "context": ["island", "shore", "shelf", "open_sea", "island"]})
    out, audit = map_genomic_context(ctx, universe)
    assert out["cpg_idx"].tolist() == [7, 8]
    assert out["context"].tolist() == ["shore", "shelf"]
    assert out["source_cpg_idx"].tolist() == [16053434200, 23024072640]
    assert audit["n_universe_unmapped"] == 1 and audit["output_cpg_idx_unique"]


def test_real_genomic_context_canonical_smoke():
    from cpg_repr_benchmark.biological_validation_v2.coordinates import (
        GENOMIC_CONTEXT_CANONICAL,
        UNIVERSE_SIZE,
        load_genomic_context_canonical,
    )

    root = Path(__file__).resolve().parents[1]
    if not (root / GENOMIC_CONTEXT_CANONICAL).exists():
        pytest.skip("canonical genomic context not materialized")
    d = load_genomic_context_canonical(root)
    assert d["cpg_idx"].is_unique and len(d) <= UNIVERSE_SIZE
    assert set(d["context"]) <= {"island", "shore", "shelf", "open_sea"}
