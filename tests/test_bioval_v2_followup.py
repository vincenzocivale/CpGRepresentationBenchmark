"""Synthetic-data tests of the bioval v2 follow-up kernels (no real data, no embeddings)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from cpg_repr_benchmark.bioval_v2_followup import compare as cmp
from cpg_repr_benchmark.bioval_v2_followup import drivers as dr
from cpg_repr_benchmark.bioval_v2_followup import gate as fg
from cpg_repr_benchmark.bioval_v2_followup import geometry as geo
from cpg_repr_benchmark.bioval_v2_followup import profile_ridge as pr
from cpg_repr_benchmark.bioval_v2_followup import registry as rg
from cpg_repr_benchmark.bioval_v2_followup import strata as st
from cpg_repr_benchmark.bioval_v2_followup import targets as tg
from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import BlockBootstrap
from cpg_repr_benchmark.bioval_v2_launch.embedding_eval import pair_cosine

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)


# ------------------------------------------------------------------ targets
def _profiles(tmp_path, n=60, G=30):
    beta = rng.uniform(0, 1, (n, G)).astype(np.float16)
    prim = rng.uniform(size=(n, G)) > 0.1
    len_ = prim | (rng.uniform(size=(n, G)) > 0.7)
    cpg = rng.permutation(np.arange(100, 100 + n)).astype(np.int64)  # unsorted ids
    p = tmp_path / "beta.h5"
    with h5py.File(p, "w") as h:
        h["cpg_idx"] = cpg
        h["chrom"] = np.array([f"chr{1 + i % 3}" for i in range(n)], dtype="S5")
        h["beta"] = beta
        h["mask_primary"] = prim
        h["mask_lenient"] = len_
        h["groups"] = np.array([f"g{i}".encode() for i in range(G)])
        h["n_samples_per_group"] = np.array([1, 1] + [3] * (G - 2))
    return tg.LoyferProfiles(p), cpg


def test_pair_pearson_matches_numpy_with_nans():
    P = rng.normal(size=(20, 12))
    P[rng.uniform(size=P.shape) > 0.8] = np.nan
    ia, ib = rng.integers(0, 20, 50), rng.integers(0, 20, 50)
    ns, r = tg.pair_pearson(P, ia, ib, min_shared=5)
    for k in range(50):
        ok = ~np.isnan(P[ia[k]]) & ~np.isnan(P[ib[k]])
        assert ns[k] == ok.sum()
        if ok.sum() >= 5:
            assert r[k] == pytest.approx(np.corrcoef(P[ia[k]][ok], P[ib[k]][ok])[0, 1], abs=1e-12)
        else:
            assert np.isnan(r[k])


def test_pair_pearson_constant_profile_undefined():
    P = np.vstack([np.full(10, 0.5), np.linspace(0, 1, 10)])
    ns, r = tg.pair_pearson(P, np.array([0]), np.array([1]), min_shared=3)
    assert ns[0] == 10 and np.isnan(r[0])


def test_reproduction_check_and_failure(tmp_path):
    prof, cpg = _profiles(tmp_path)
    ia, ib = rng.integers(0, 60, 80), rng.integers(0, 60, 80)
    ns, r = tg.pair_pearson(prof.matrix("primary"), prof.rows(cpg[ia]), prof.rows(cpg[ib]))
    pairs = pd.DataFrame({"cpg_i": cpg[ia], "cpg_j": cpg[ib], "n_shared_groups": ns, "loyfer_pearson": r})
    assert tg.assert_reproduces_frozen(pairs, prof)["max_abs_diff_vs_frozen_loyfer_pearson"] < 1e-12
    bad = pairs.copy()
    bad.loc[bad.index[np.isfinite(bad.loyfer_pearson)][0], "loyfer_pearson"] += 1e-3
    with pytest.raises(AssertionError):
        tg.assert_reproduces_frozen(bad, prof)


def test_variants_lenient_superset_and_group_drop(tmp_path):
    prof, cpg = _profiles(tmp_path)
    assert prof.single_sample_groups() == [0, 1]
    ia, ib = rng.integers(0, 60, 100), rng.integers(0, 60, 100)
    pairs = pd.DataFrame({"cpg_i": cpg[ia], "cpg_j": cpg[ib]})
    _, ns_l = tg.variant_target(pairs, prof, "mask_lenient")
    ns_p, _ = tg.pair_pearson(prof.matrix("primary"), prof.rows(pairs.cpg_i), prof.rows(pairs.cpg_j))
    assert (ns_l >= ns_p).all() and (ns_l > ns_p).any()
    assert prof.matrix("primary", drop_groups=[0, 1]).shape[1] == 28
    with pytest.raises(ValueError):
        tg.variant_target(pairs, prof, "nope")


# ------------------------------------------------------------------ strata
def test_cells_definition_and_edges():
    v = np.array([0.0, rg.V_EDGES[0], rg.V_EDGES[1] - 1e-9, rg.V_EDGES[2], rg.V_TOP10, 1.0])
    m = np.array([0.1, 0.3, 0.5, 0.8, 0.9, 0.0])
    strat = np.array(["<1kb", "1-10kb", ">1Mb", "interchromosomal", "interchromosomal", "10-100kb"])
    c = st.cells(strat, v, m)
    assert [int(c[f"V_Q{k}"].sum()) for k in (1, 2, 3, 4)] == [1, 2, 0, 3]  # edge value belongs to the upper bin
    assert c["V_top10pct"].tolist() == [False] * 4 + [True, True]
    assert sum(int(c[f"M_Q{k}"].sum()) for k in (1, 2, 3, 4)) == 6
    assert (c["chrom_intra"] ^ c["chrom_inter"]).all()
    assert sum(int(c[f"dist_{s}"].sum()) for s in rg.ALL_STRATA) == 6


def test_registered_edges_assertion_detects_tuning():
    vmin = np.random.default_rng(1).uniform(size=10_000)
    with pytest.raises(AssertionError):
        st.check_registered_edges(vmin, vmin)
    # data whose quantiles equal the registered edges passes
    u = np.linspace(0, 1, 100_001)
    v = np.interp(u, [0, 0.25, 0.5, 0.75, 0.9, 1], [0, *rg.V_EDGES, rg.V_TOP10, 1.0])
    m = np.interp(u, [0, 0.25, 0.5, 0.75, 1], [0, *rg.M_EDGES, 1.0])
    st.check_registered_edges(v, m)


def test_per_cpg_stats_nan_groups():
    P = np.array([[0.2, 0.4, np.nan], [0.5, 0.5, 0.5]])
    mean, var, n = st.per_cpg_stats(P)
    assert mean[0] == pytest.approx(0.3) and var[0] == pytest.approx(0.01) and n.tolist() == [2, 3] and var[1] == 0


# ------------------------------------------------------------------ geometry
def test_centered_and_pc1_cosine():
    E = rng.normal(size=(500, 8)).astype(np.float32) + 3.0
    E[:, 0] += 5 * rng.normal(size=500)
    mu, v1, ev = geo.universe_mean_pc1(E)
    assert np.allclose(mu, E.astype(np.float64).mean(0))
    assert v1[0] != 0 and abs(np.linalg.norm(v1) - 1) < 1e-12 and ev > 0.3
    Z = E.astype(np.float64) - mu
    u = np.linalg.svd(Z, full_matrices=False)[2][0]
    assert abs(abs(u @ v1) - 1) < 1e-8
    ia, ib = rng.integers(0, 500, 200), rng.integers(0, 500, 200)
    cc = geo.centered_cosine(E, ia, ib, mu)
    assert np.allclose(cc, pair_cosine(Z.astype(np.float32), ia, ib), atol=1e-6)
    cp = geo.pc1_removed_cosine(E, ia, ib, mu, v1)
    Zp = Z - np.outer(Z @ v1, v1)
    assert np.allclose(Zp @ v1, 0, atol=1e-9)
    assert np.allclose(cp, pair_cosine(Zp.astype(np.float32), ia, ib), atol=1e-6)
    eu = geo.neg_euclidean(E, ia, ib)
    assert np.allclose(eu, -np.linalg.norm(E[ia].astype(float) - E[ib], axis=1))


def test_geometry_fails_loudly_on_zero_norm():
    E = np.ones((3, 4), np.float32)
    with pytest.raises(ValueError):
        geo.centered_cosine(E, np.array([0]), np.array([1]), E[0].astype(float))


def test_euclid_rank_invariant_to_monotone_standardisation():
    E = rng.normal(size=(100, 5))
    ia, ib = rng.integers(0, 100, 300), rng.integers(0, 100, 300)
    d = -geo.neg_euclidean(E, ia, ib)
    x = rng.normal(size=300)
    alt = np.exp(-d ** 2 / np.median(d ** 2))
    assert spearmanr(x, -d)[0] == pytest.approx(spearmanr(x, alt)[0])


# ------------------------------------------------------------------ profile ridge
def _ridge_data(n=600, d=10, G=5):
    X = rng.normal(size=(n, d))
    W = rng.normal(size=(d, G))
    Y = X @ W + 0.5 * rng.normal(size=(n, G)) + 2
    chrom = rng.integers(0, 10, n)
    fold = (chrom % 5).astype(int)
    return X, Y, chrom, fold


def test_ridge_same_folds_no_leakage_and_arm_independence():
    X, Y, _chrom, fold = _ridge_data()
    t = pr.assemble_targets(Y)
    f1, folds = pr.fit_all(X, t, fold)
    assert folds.tolist() == [0, 1, 2, 3, 4]
    # changing the TARGETS of one test fold must not change that fold's out-of-fold predictions (no leakage)
    Y2 = Y.copy()
    Y2[fold == 2] += 100.0
    f2, _ = pr.fit_all(X, pr.assemble_targets(Y2), fold)
    assert np.allclose(f1["raw"]["oof"][fold == 2], f2["raw"]["oof"][fold == 2])
    assert not np.allclose(f1["raw"]["oof"][fold == 3], f2["raw"]["oof"][fold == 3])
    # centred + level targets have the registered shapes and consistent decomposition
    assert f1["centered"]["oof"].shape == Y.shape and f1["level"]["oof"].shape == (len(Y), 1)
    assert np.allclose(t["centered"].sum(1), 0)


def test_baseline_is_train_mean_and_r2_near_zero():
    _X, Y, chrom, fold = _ridge_data()
    b = pr.train_mean_baseline(Y, fold)
    te = fold == 1
    assert np.allclose(b[te][0], Y[~te].mean(0))
    S = pr.SuffStats(Y, b, chrom, 10)
    assert abs(S.metrics(np.ones(10))["r2_global"]) < 0.05


def test_suffstats_match_direct_and_bootstrap_equals_concatenation():
    _X, Y, chrom, _fold = _ridge_data()
    P = Y + rng.normal(size=Y.shape)
    S = pr.SuffStats(Y, P, chrom, 10)
    m = S.metrics(np.ones(10))
    sst = ((Y - Y.mean(0)) ** 2).sum(0)
    sse = ((Y - P) ** 2).sum(0)
    assert np.allclose(m["r2_per_group"], 1 - sse / sst)
    assert m["r2_global"] == pytest.approx(1 - sse.sum() / sst.sum())
    assert m["pearson_pooled"] == pytest.approx(np.corrcoef(Y.ravel(), P.ravel())[0, 1])
    rs = [np.corrcoef(Y[i], P[i])[0, 1] for i in range(len(Y))]
    assert m["pearson_mean_over_cpgs"] == pytest.approx(np.mean(rs))
    boot = BlockBootstrap(np.arange(10).astype(str), n_boot=5, seed=17)
    rep = pr.bootstrap_metrics(S, boot.mult)
    for b in range(5):
        idx = np.concatenate([np.flatnonzero(chrom == c) for c in boot.draws[b]])
        Yb, Pb = Y[idx], P[idx]
        sstb = ((Yb - Yb.mean(0)) ** 2).sum(0)
        sseb = ((Yb - Pb) ** 2).sum(0)
        assert rep["r2_global"][b] == pytest.approx(1 - sseb.sum() / sstb.sum())
        assert np.allclose(rep["r2_per_group"][b], 1 - sseb / sstb)
        assert rep["pearson_pooled"][b] == pytest.approx(np.corrcoef(Yb.ravel(), Pb.ravel())[0, 1])


def test_fold_metrics_shapes():
    _X, Y, _chrom, fold = _ridge_data()
    pf, mat = pr.fold_metrics(Y, Y + 0.1, fold)
    assert sorted(pf) == [0, 1, 2, 3, 4] and mat.shape == (5, Y.shape[1])


# ------------------------------------------------------------------ contrasts / verdict
def test_contrast_paired_and_categories():
    a, b = rng.normal(size=1000), rng.normal(size=1000)
    c = cmp.paired_contrast(0.1, a + 1, 0.0, b)
    assert c["descriptive_only_no_holm"] and "p" in c and c["delta"] == pytest.approx(0.1)
    assert cmp.contrast_category(0.1, 0.2, 0.1, 0.3) == cmp.SAME
    assert cmp.contrast_category(0.1, -0.2, -0.3, -0.1) == cmp.FLIP_RESOLVED
    assert cmp.contrast_category(0.1, -0.2, -0.3, 0.1) == cmp.FLIP_UNRESOLVED
    pdel = {"a": -0.1, "b": -0.2}
    rows = {"a": {"delta": -0.05, "ci_lo": -0.1, "ci_hi": 0.0}, "b": {"delta": -0.1, "ci_lo": -0.2, "ci_hi": -0.01}}
    pv = {"r": 0.0, "a": 0.1, "b": 0.2, "c": 0.15}
    v = cmp.item_verdict(pdel, rows, pv, pv)
    assert v["verdict"] == cmp.CONFIRMS and v["ordering_identical"]
    rows["a"] = {"delta": 0.05, "ci_lo": 0.01, "ci_hi": 0.1}
    assert cmp.item_verdict(pdel, rows, pv, pv)["verdict"] == cmp.CHANGES
    rows["a"] = {"delta": 0.05, "ci_lo": -0.01, "ci_hi": 0.1}
    assert cmp.item_verdict(pdel, rows, pv, pv)["verdict"] == cmp.WEAKENS


# ------------------------------------------------------------------ drivers helpers
def _ctx(blocks=6):
    boot = BlockBootstrap(np.arange(blocks).astype(str), n_boot=20, seed=17)
    return SimpleNamespace(boot=boot)


def test_pooled_stat_equal_stratum_weights_and_missing_stratum():
    n = 400
    strat = np.array([">1Mb"] * 100 + ["interchromosomal"] * 300)
    x, y = rng.normal(size=n), rng.normal(size=n)
    blk = rng.integers(0, 6, n).astype(np.int16)
    ctx = _ctx()
    s = dr._stat_pooled(ctx, "p", x, y, strat, blk, "x")
    # equal stratum weight: weighted Pearson of weighted mid-ranks, w = 1/n_stratum
    w = np.where(strat == ">1Mb", 1 / 100, 1 / 300)
    from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import RankPlan, weighted_spearman
    assert s.value == pytest.approx(weighted_spearman(RankPlan(x), RankPlan(y), w))
    assert s.rep.shape == (20,)
    assert dr._stat_pooled(ctx, "p", x, y, strat, blk, "x", mask=(strat == ">1Mb")) is None
    u = dr._stat_unw(ctx, "u", x, y, blk, "x")
    assert u.value == pytest.approx(spearmanr(x, y)[0])


def test_row_pearson_matches_numpy():
    A = rng.normal(size=(30, 15))
    ia, ib = rng.integers(0, 30, 40), rng.integers(0, 30, 40)
    cols = np.arange(2, 12)
    r = dr._row_pearson(A, ia, ib, cols)
    for k in range(40):
        assert r[k] == pytest.approx(np.corrcoef(A[ia[k], cols], A[ib[k], cols])[0, 1])
    A[0] = 1.0
    assert np.isnan(dr._row_pearson(A, np.array([0]), np.array([1]), cols)[0])


def test_registry_and_drivers_consistent():
    assert set(dr.SENS_DRIVERS) == set(rg.ITEMS)
    assert all(v[0] == rg.RUNNABLE for v in rg.ITEMS.values())
    assert not set(rg.NOT_RUNNABLE) & set(rg.ITEMS)
    assert all(v[0] in (rg.NEW_ARTIFACT, rg.AMBIGUOUS) for v in rg.NOT_RUNNABLE.values())
    assert set(fg.rg.FOLLOWUP_CODE_FILES) >= {"scripts/bioval_v2/run_followup.py"}
    for rel in rg.FOLLOWUP_CODE_FILES:
        assert (ROOT / rel).exists(), rel


# ------------------------------------------------------------------ gate + guarded writer
def test_writer_refuses_frozen_primary_results_and_protected(tmp_path):
    base = ROOT / rg.BASE_OUT
    for sub in ("regulatory_histone_dnase_v1/endpoints", "contrasts", "report", "cpgpt_large_locus/results.json", "."):
        assert fg.is_frozen_result_path(ROOT, base / sub)
    assert not fg.is_frozen_result_path(ROOT, base / "sensitivity" / "x" / "endpoints")
    assert not fg.is_frozen_result_path(ROOT, base / "exploratory_posthoc_loyfer")
    assert not fg.is_frozen_result_path(ROOT, tmp_path)
    with pytest.raises(PermissionError):
        fg.FollowupWriter(ROOT, base / "contrasts")
    with pytest.raises(PermissionError):
        fg.FollowupWriter(ROOT, base)
    with pytest.raises(PermissionError):
        fg.FollowupWriter(ROOT, ROOT / "data/derived/bioval_v2/x")
    w = fg.FollowupWriter(ROOT, base / "sensitivity")
    with pytest.raises(PermissionError):
        w.path("../contrasts/x.json")
    with pytest.raises(PermissionError):
        w.path("../regulatory_histone_dnase_v1/endpoints/loyfer_profile.json")
    w2 = fg.FollowupWriter(tmp_path, tmp_path / "out")
    w2.write_json("a/b.json", {"x": np.float32(1.5)})
    assert (tmp_path / "out/a/b.json").exists()


def test_out_root_check_failure_modes():
    base = ROOT / rg.BASE_OUT
    assert fg.check_out_root(ROOT, base / "sensitivity", "sensitivity").ok
    assert not fg.check_out_root(ROOT, base / "sensitivity", "exploratory-loyfer").ok
    assert fg.check_out_root(ROOT, base / "exploratory_posthoc_loyfer", "exploratory-loyfer").ok
    assert not fg.check_out_root(ROOT, base / "contrasts", "sensitivity").ok
    assert not fg.check_out_root(ROOT, ROOT / "data/derived/bioval_v2/x", "sensitivity").ok
    assert not fg.check_out_root(ROOT, ROOT / "src/x", "sensitivity").ok


def test_gate_failure_modes_in_scratch_repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    (r / "docs").mkdir()
    (r / rg.FOLLOWUP_REGISTRATION_DOC).write_text("x")
    c = fg._clean(r, [rg.FOLLOWUP_REGISTRATION_DOC], "n")
    assert not c.ok and "not tracked" in c.detail
    subprocess.run(["git", "-C", str(r), "add", "."], check=True)
    subprocess.run(["git", "-C", str(r), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "m"], check=True)
    assert fg._clean(r, [rg.FOLLOWUP_REGISTRATION_DOC], "n").ok
    (r / rg.FOLLOWUP_REGISTRATION_DOC).write_text("changed")
    c = fg._clean(r, [rg.FOLLOWUP_REGISTRATION_DOC], "n")
    assert not c.ok and "modified" in c.detail
    assert not fg._clean(r, ["docs/missing.md"], "n").ok
    assert not fg.lg.check_tag(r).ok  # no freeze tag in the scratch repo


def test_frozen_results_fingerprint_detects_change_and_ignores_followup_dirs(tmp_path):
    base = tmp_path / rg.BASE_OUT
    (base / "armA/endpoints").mkdir(parents=True)
    (base / "armA/endpoints/e.json").write_text("1")
    (base / "contrasts").mkdir()
    (base / "contrasts/c.csv").write_text("a")
    f0 = fg.frozen_results_fingerprint(tmp_path)
    (base / "sensitivity").mkdir()
    (base / "sensitivity/new.json").write_text("x")
    (base / "exploratory_posthoc_loyfer").mkdir()
    (base / "exploratory_posthoc_loyfer/new.json").write_text("x")
    assert fg.frozen_results_fingerprint(tmp_path) == f0
    (base / "armA/endpoints/e.json").write_text("2")
    assert fg.frozen_results_fingerprint(tmp_path) != f0


def test_inputs_guard_rejects_tcga():
    assert not fg.lg.check_inputs_not_tcga(["x/TCGA/test.parquet"]).ok
    assert not fg.lg.check_inputs_not_tcga(["data/protocols/a.npz"]).ok
    assert fg.lg.check_inputs_not_tcga([str(ROOT / "data/derived/bioval_v2/x.parquet")]).ok


# ------------------------------------------------------------------ end-to-end on a synthetic tree (no real data)
class _FakeStore:
    def __init__(self, ids, E):
        self.ids, self.E = ids, E

    def load(self, q):
        return self.E[np.searchsorted(self.ids, np.asarray(q))].astype(np.float32)


def _synthetic_tree(tmp_path, n=2200, G=39):
    import json

    from cpg_repr_benchmark.biological_validation_v2.splits import chromosome_blocked_folds
    from cpg_repr_benchmark.bioval_v2_launch import launch_endpoints as le
    chroms = [f"chr{i}" for i in range(1, 23)]
    ids = np.arange(1000, 1000 + n).astype(np.int64)
    chrom = np.array([chroms[i % 22] for i in range(n)])
    beta = np.clip(rng.normal(0.6, 0.25, (n, G)), 0, 1).astype(np.float16)
    prim = rng.uniform(size=(n, G)) > 0.03
    root = tmp_path / "root"
    ext = root / le.BV / "external_methylation_programs"
    ext.mkdir(parents=True)
    with h5py.File(ext / "loyfer_celltype_beta.h5", "w") as h:
        h["cpg_idx"], h["chrom"], h["beta"] = ids, chrom.astype("S5"), beta
        h["mask_primary"], h["mask_lenient"] = prim, prim | (rng.uniform(size=(n, G)) > 0.5)
        h["groups"] = np.array([f"g{i}".encode() for i in range(G)])
        h["n_samples_per_group"] = np.array([1] * 4 + [3] * (G - 4))
    (root / le.BV / "replication_domains").mkdir(parents=True)
    (root / le.BV / "replication_domains/FROZEN_ENDPOINT.json").write_text(
        json.dumps({"folds": {"chrom_to_fold": chromosome_blocked_folds(chroms, 5, 17)}}))
    E = rng.normal(size=(n, 6)) + 0.3 * beta.astype(float)[:, :6]
    boot = BlockBootstrap(np.array(sorted(chroms)), n_boot=15, seed=17)
    universe = pd.DataFrame({"cpg_idx": ids, "chrom": chrom})
    ctx = le.EvalContext(root, _FakeStore(ids, E), np.zeros(0, np.int64), boot, universe)
    return ctx, ids, chrom, ext, prof_pairs(ids, ext, n)


def prof_pairs(ids, ext, n):
    prof = tg.LoyferProfiles(ext / "loyfer_celltype_beta.h5")
    ia, ib = rng.integers(0, n, 3000), rng.integers(0, n, 3000)
    ia = np.where(ia == ib, (ib + 1) % n, ia)
    ns, r = tg.pair_pearson(prof.matrix("primary"), prof.rows(ids[ia]), prof.rows(ids[ib]))
    ok = np.isfinite(r)
    d = pd.DataFrame({"cpg_i": ids[ia], "cpg_j": ids[ib], "n_shared_groups": ns, "loyfer_pearson": r})
    d["stratum"] = np.where(np.arange(len(d)) % 2 == 0, ">1Mb", "interchromosomal")
    d = d[ok].reset_index(drop=True)
    d.to_parquet(ext / "frozen_pairs_seed17.parquet")
    return d


def test_end_to_end_synthetic_loyfer_variant_and_profile_ridge(tmp_path):
    ctx, _ids, _chrom, _ext, _pairs = _synthetic_tree(tmp_path)
    res = dr.loyfer_variant(ctx, "mask_lenient")
    assert rg.LOYFER_PRIMARY_STAT in res.stats and res.stats["spearman_all_pairs"].n == res.meta["exclusion"]["n_kept"]
    assert res.meta["target_reproduction_check"]["max_abs_diff_vs_frozen_loyfer_pearson"] < 1e-12
    res2 = dr.loyfer_variant(ctx, "drop_single_sample_groups")
    assert len(res2.meta["single_sample_groups_dropped"]) == 4
    body, reps = dr.exploratory_profile_ridge(ctx, "synthetic")
    assert body["EXPLORATORY_POST_HOC"] and body["n_cpgs_complete_case"] > 400
    for t in ("raw", "centered", "level"):
        cell = body["targets"][t]["subsets"]["all_cpgs"]
        assert cell["model"]["r2_global"] > -0.2
        assert abs(cell["train_mean_baseline"]["r2_global"]) < 0.05
        assert len(cell["model"]["r2_per_group"]) == (1 if t == "level" else 39)
    assert reps["raw__all_cpgs__r2_global"].shape == (15,) and reps["raw__all_cpgs__r2_per_group"].shape == (15, 39)
    # identical fold map regardless of the representation: folds recorded and equal to the frozen map
    assert body["folds"] == [0, 1, 2, 3, 4]


def test_end_to_end_synthetic_similarity_geometries(tmp_path, monkeypatch):
    ctx, _ids, _chrom, _ext, pairs = _synthetic_tree(tmp_path)
    monkeypatch.setattr(st, "check_registered_edges", lambda v, m: {"patched": True})
    strat = pairs.stratum.to_numpy()
    cos = ctx.cosine_pairs(pairs.cpg_i.to_numpy(), pairs.cpg_j.to_numpy())
    blk = ctx.blk(ctx.chrom_of(pairs.cpg_i.to_numpy()))
    expected = dr._stat_pooled(ctx, "x", pairs.loyfer_pearson.to_numpy(), cos, strat, blk, "x").value
    monkeypatch.setattr(dr, "frozen_value", lambda root, arm, ep, stat: expected)
    res = dr.exploratory_similarity(ctx, "synthetic")
    assert res.meta["EXPLORATORY_POST_HOC"] and res.meta["reproduces_frozen_primary"]["recomputed"] == pytest.approx(expected)
    for g in geo.GEOMETRIES:
        assert f"spearman__{g}__all" in res.stats and f"spearman__{g}__chrom_intra" in res.stats
    assert res.stats["spearman__cosine__all"].value == pytest.approx(spearmanr(pairs.loyfer_pearson, cos)[0])
    # a wrong frozen reference makes the run fail explicitly (pair-set / pipeline mismatch is never silent)
    monkeypatch.setattr(dr, "frozen_value", lambda root, arm, ep, stat: expected + 0.1)
    with pytest.raises(AssertionError):
        dr.exploratory_similarity(ctx, "synthetic")


# ------------------------------------------------------------------ interpretation rules (fixed in the registration)
def _row(d, lo, hi):
    return {"delta": d, "ci_lo": lo, "ci_hi": hi}


def test_ridge_outcome_rules():
    from cpg_repr_benchmark.bioval_v2_followup import interpret as ip
    seq = ip.SEQ
    vals = {ip.REF: 0.01, seq[0]: 0.2, seq[1]: 0.25, ip.FUNC: 0.02}
    below = {s: _row(-0.2, -0.3, -0.1) for s in seq}
    assert ip.ridge_outcome(vals, below) == "ABSENT"
    vals[ip.REF] = 0.1
    assert ip.ridge_outcome(vals, below) == "LOWER"
    assert ip.ridge_outcome(vals, {seq[0]: below[seq[0]], seq[1]: _row(-0.05, -0.2, 0.1)}) == "MIXED"
    assert ip.ridge_outcome(vals, {s: _row(0.0, -0.1, 0.1) for s in seq}) == "COMPARABLE"
    assert ip.ridge_outcome(vals, {seq[0]: _row(0.1, 0.05, 0.2), seq[1]: _row(0.0, -0.1, 0.1)}) == "HIGHER"
    assert ip.ridge_outcome({a: 0.0 for a in vals}, below) == "UNINFORMATIVE"


def test_geometry_and_functional_outcome_and_table():
    from cpg_repr_benchmark.bioval_v2_followup import interpret as ip
    seq = ip.SEQ
    within = {"g1": _row(0.1, 0.05, 0.15), "g2": _row(0.01, -0.01, 0.03)}
    open_gap = {"g1": {s: _row(-0.1, -0.2, -0.05) for s in seq}, "g2": {s: _row(-0.1, -0.2, -0.05) for s in seq}}
    closed = {"g1": {s: _row(0.0, -0.05, 0.05) for s in seq}, "g2": open_gap["g2"]}
    assert ip.geometry_outcome(within, open_gap) == "IMPROVES_ONLY"
    assert ip.geometry_outcome(within, closed) == "CLOSES"
    assert ip.geometry_outcome({"g": _row(0.01, -0.01, 0.03)}, open_gap) == "NONE"
    assert ip.geometry_outcome({"g": _row(0.1, -0.01, 0.2)}, open_gap) == "NONE"  # CI includes 0: no improvement claimed
    assert ip.functional_outcome({s: _row(-0.1, -0.2, -0.01) for s in seq}) == "BELOW_BOTH"
    assert ip.functional_outcome({seq[0]: _row(-0.1, -0.2, -0.01), seq[1]: _row(0.0, -0.1, 0.1)}) == "NOT_BELOW"
    assert ip.interpret("ABSENT", "NONE", "BELOW_BOTH")["A"].startswith("supported")
    assert ip.interpret("COMPARABLE", "CLOSES", "NOT_BELOW")["B"] == "supported (strong)"
    assert ip.interpret("COMPARABLE", "NONE", "NOT_BELOW")["C"].startswith("not supported")
    assert ip.interpret("UNINFORMATIVE", "NONE", "NOT_BELOW")["A"].startswith("not adjudicable")
    assert "ANOMALY" in ip.interpret("ABSENT", "CLOSES", "NOT_BELOW")["notes"][0]
    # every (ro, go, fo) combination yields a reading (no undefined cell)
    for ro in ("UNINFORMATIVE", "ABSENT", "LOWER", "MIXED", "COMPARABLE", "HIGHER"):
        for go in ("CLOSES", "IMPROVES_ONLY", "NONE"):
            for fo in ("BELOW_BOTH", "NOT_BELOW"):
                assert {"A", "B", "C"} <= set(ip.interpret(ro, go, fo))


def _load_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_followup_mod", ROOT / "scripts/bioval_v2/run_followup.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_contrast_pipeline_smoke_on_synthetic_arms(tmp_path, monkeypatch):
    """Runs both Part B drivers for 4 synthetic 'arms', writes through the guarded writer, then the exploratory contrast +
    A/B/C interpretation code (reads real frozen primary results read-only for the ranking reference)."""
    mod = _load_script()
    import json
    out_root = tmp_path / "expl"
    w = fg.FollowupWriter(tmp_path, out_root)
    monkeypatch.setattr(st, "check_registered_edges", lambda v, m: {"patched": True})
    for k, arm in enumerate(mod.ARMS):
        ctx, _ids, _c, _e, pairs = _synthetic_tree(tmp_path / f"t{k}", n=1500)
        cos = ctx.cosine_pairs(pairs.cpg_i.to_numpy(), pairs.cpg_j.to_numpy())
        blk = ctx.blk(ctx.chrom_of(pairs.cpg_i.to_numpy()))
        exp = dr._stat_pooled(ctx, "x", pairs.loyfer_pearson.to_numpy(), cos, pairs.stratum.to_numpy(), blk, "x").value
        monkeypatch.setattr(dr, "frozen_value", lambda root, a, ep, stat, exp=exp: exp)
        res = dr.exploratory_similarity(ctx, arm)
        mod.write_endpoint(w, arm, res)
        body, reps = dr.exploratory_profile_ridge(ctx, arm)
        w.write_json(f"{arm}/endpoints/B3_profile_ridge.json", body)
        w.save_npz(f"{arm}/endpoints/B3_profile_ridge.replicates.npz", **reps)
    monkeypatch.undo()
    mod.contrasts_exploratory(None, w, out_root)
    for f in ("similarity_ranking_by_cell.csv", "geometry_within_arm_deltas.csv", "ridge_summary_by_arm.csv",
              "ridge_functional_vs_sequence.csv", "interpretation_ABC.json"):
        assert (out_root / f).exists(), f
    j = json.loads((out_root / "interpretation_ABC.json").read_text())
    assert j["primary_reading"] == "centered/variable_cpgs" and set(j["readings"]) == {
        "centered/variable_cpgs", "centered/all_cpgs", "raw/all_cpgs"}
    assert all({"A", "B", "C"} <= set(r) for r in j["readings"].values())
    assert len(list((out_root / "contrasts").glob("*.csv"))) == 6


def test_sensitivity_contrast_and_verdict_smoke(tmp_path):
    mod = _load_script()
    import json
    out_root = tmp_path / "sens"
    w = fg.FollowupWriter(tmp_path, out_root)
    for k, arm in enumerate(mod.ARMS):
        ctx, *_ = _synthetic_tree(tmp_path / f"s{k}", n=1500)
        mod.write_endpoint(w, arm, dr.loyfer_variant(ctx, "mask_lenient"))
    mod.contrasts_sensitivity(None, w, out_root)
    s = json.loads((out_root / "SUMMARY.json").read_text())
    done = [r for r in s if r["status"] == "done"]
    assert [r["item"] for r in done] == ["loyfer_mask_lenient"]
    assert done[0]["verdict"] in (cmp.CONFIRMS, cmp.CHANGES, cmp.WEAKENS) and "ordering_identical" in done[0]
    assert sum("not runnable" in r["status"] for r in s) == len(rg.NOT_RUNNABLE)
    assert (out_root / "contrasts").is_dir()
