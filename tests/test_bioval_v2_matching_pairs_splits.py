import ast
import pathlib

import numpy as np
import pandas as pd
import pytest

from cpg_repr_benchmark.biological_validation_v2 import (
    assert_no_leak,
    chromosome_blocked_folds,
    match_controls,
    match_pairs,
    pair_split_by_chrom,
    sample_pairs,
)
from cpg_repr_benchmark.biological_validation_v2.splits import pair_test_mask, pair_train_mask


def make_cov(n=600, seed=0):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"cpg_idx": np.arange(n), "chrom": r.choice(["chr1", "chr2", "chr3"], n),
                      "cpg_density": r.normal(10, 3, n), "gc": r.normal(0.5, 0.1, n)})
    d["case"] = r.random(n) < 0.15
    d.loc[d.case, "cpg_density"] += 4  # imbalance
    return d


def test_matching_rejects_embedding_columns():
    d = make_cov()
    with pytest.raises(ValueError):
        match_controls(d[d.case].assign(pca_1=0.0), d[~d.case], numeric={"cpg_density": 1})
    with pytest.raises(TypeError):
        match_controls(np.zeros((3, 3)), d, numeric={"cpg_density": 1})


def test_matching_invariants():
    d = make_cov()
    cases, pool = d[d.case], d[~d.case]
    num = {"cpg_density": 1.0, "gc": 0.05}
    mp, bal = match_controls(cases, pool, numeric=num, seed=1)
    assert len(mp) > 20
    m = mp.merge(d, left_on="case", right_on="cpg_idx").merge(d, left_on="control", right_on="cpg_idx",
                                                              suffixes=("_a", "_b"))
    assert (m.chrom_a == m.chrom_b).all()
    assert ((m.cpg_density_a - m.cpg_density_b).abs() <= 1.0).all()
    assert ((m.gc_a - m.gc_b).abs() <= 0.05).all()
    assert mp["control"].is_unique
    b = bal.set_index("covariate")
    assert abs(b.loc["cpg_density", "smd_after"]) < abs(b.loc["cpg_density", "smd_before"])
    mp2, _ = match_controls(cases, pool, numeric=num, seed=1)
    pd.testing.assert_frame_equal(mp, mp2)
    mp3, _ = match_controls(cases.sample(frac=1, random_state=3), pool.sample(frac=1, random_state=4),
                            numeric=num, seed=1)
    pd.testing.assert_frame_equal(mp, mp3)
    mp4, _ = match_controls(cases, pool, numeric=num, seed=1, replace=True)
    assert len(mp4) >= len(mp)


def idx(n=3000, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({"cpg_idx": np.arange(n) + 100, "chrom": r.choice(["chr1", "chr2", "chr3"], n),
                         "pos": r.integers(1, 5_000_000, n)})


def test_pair_sampling():
    d = idx()
    a = sample_pairs(d, 120, kind="intra", seed=5)
    b = sample_pairs(d.sample(frac=1, random_state=9), 120, kind="intra", seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert (a.i < a.j).all() and not (a.i == a.j).any()
    assert not a.duplicated(["i", "j"]).any()
    assert (a.chrom_i == a.chrom_j).all()
    assert ((a.dist >= 0) & np.isfinite(a.dist)).all()
    assert a["bin"].nunique() >= 4
    c = sample_pairs(d, 100, kind="inter", seed=5)
    assert (c.chrom_i != c.chrom_j).all() and len(c) == 100
    assert not sample_pairs(d, 120, kind="intra", seed=6).equals(a)


def test_chromosome_folds():
    chroms = [f"chr{i}" for i in range(1, 23)]
    f = chromosome_blocked_folds(chroms * 3, 5, seed=1)
    assert f == chromosome_blocked_folds(list(reversed(chroms)), 5, seed=1)
    assert set(f) == set(chroms) and set(f.values()) == set(range(5))
    assert f != chromosome_blocked_folds(chroms, 5, seed=2)
    with pytest.raises(AssertionError):
        assert_no_leak(["chr1", "chr2"], ["chr2"])


def test_pair_split_straddle_and_leak():
    folds = {"a": 0, "b": 0, "c": 1}
    pdf = pd.DataFrame({"chrom_i": ["a", "a", "a", "c"], "chrom_j": ["a", "b", "c", "a"]})
    assert pair_split_by_chrom(pdf, folds, "drop").tolist() == [0, 0, -1, -1]
    assert pair_split_by_chrom(pdf, folds, "both_in_test").tolist() == [0, 0, 0, 0]
    train = pdf[pair_train_mask(pdf, folds, 1)]
    test = pdf[pair_test_mask(pdf, folds, 1)]
    assert_no_leak(["a", "b"], ["c"], train, test.assign(chrom_i="c", chrom_j="c"))
    assert len(test) == 2 and len(train) == 2


def test_match_pairs_separates_intra_inter():
    d = idx()
    pool = sample_pairs(d, 600, kind="intra", seed=1)
    pool_inter = sample_pairs(d, 600, kind="inter", seed=1)
    allp = pd.concat([pool, pool_inter], ignore_index=True)
    cases = sample_pairs(d, 40, kind="intra", seed=2)
    m = match_pairs(cases, allp, same_chrom=True, seed=0)
    ctrl = allp.loc[m.control_row]
    assert (ctrl.chrom_i == ctrl.chrom_j).all()
    ci = sample_pairs(d, 40, kind="inter", seed=2)
    m2 = match_pairs(ci, allp, same_chrom=False, seed=0)
    ctrl2 = allp.loc[m2.control_row]
    assert (ctrl2.chrom_i != ctrl2.chrom_j).all() and len(m2) > 0


def test_no_embedding_imports():
    root = pathlib.Path(__file__).resolve().parents[1] / "src/cpg_repr_benchmark/biological_validation_v2"
    bad = ("torch", "h5py", "cpg_repr_benchmark.representations", "cpg_repr_benchmark.models",
           "cpg_repr_benchmark.embedding")
    for f in root.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for n in names:
                assert not n.startswith(bad), f"{f.name} imports {n}"


def test_fasta_window_stats_and_covariates_complete(tmp_path):
    pytest.importorskip("pyfaidx")
    from cpg_repr_benchmark.biological_validation_v2.matching import build_covariates

    seq = "ACGTCGGGCCAACGTTTTCGAAAA" * 5
    fa = tmp_path / "t.fa"
    fa.write_text(">chr1\n" + seq + "\n")
    u = pd.DataFrame({"cpg_idx": [1, 2, 3], "chrom": ["chr1"] * 3, "pos": [3, 12, 100]})
    ctx = tmp_path / "c.parquet"
    pd.DataFrame({"cpg_idx": [1, 2, 3], "context": ["island", "shore", "open_sea"]}).to_parquet(ctx)
    cov, notes = build_covariates(u, context_path=ctx, fasta_path=fa, tss_dist_bp=pd.Series([0, 9, 99], index=[1, 2, 3]),
                                  probe_type=pd.Series(["I", "II", "II"], index=[1, 2, 3]), gc_window=5)
    assert not cov.isna().any().any(), cov
    assert notes == []
    assert cov.loc[0, "tss_dist"] == 0.0
    assert cov["cpg_density_hg38"].min() >= 1
