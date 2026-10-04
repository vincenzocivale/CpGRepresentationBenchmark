"""Synthetic-data tests for the bioval v2 launch code (no real embedding / frozen data is read)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cpg_repr_benchmark.biological_validation_v2.metrics import auroc_binary, bootstrap_ci_by_block
from cpg_repr_benchmark.biological_validation_v2.splits import chromosome_blocked_folds
from cpg_repr_benchmark.bioval_v2_launch import chrom_probes as cp
from cpg_repr_benchmark.bioval_v2_launch import launch_endpoints as le
from cpg_repr_benchmark.bioval_v2_launch import launch_gate as lg
from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import (
    BlockBootstrap,
    RankPlan,
    block_index,
    bootstrap_p_two_sided,
    contrast,
    holm_adjust,
    weighted_auroc,
    weighted_mean,
    weighted_r2,
    weighted_spearman,
)
from cpg_repr_benchmark.bioval_v2_launch.embedding_eval import (
    ArmSpec,
    EmbeddingStore,
    assert_readable,
    exclusion_mask,
    pair_cosine,
    sha256_file,
    topk_cosine_neighbors,
    union_zero_loci,
)

CHROMS = [f"chr{i}" for i in range(1, 23)]


# ----------------------------------------------------------------------------- weighted kernels / bootstrap
def test_weighted_kernels_match_scipy_and_frozen_metrics_with_ties():
    rng = np.random.default_rng(0)
    x = np.round(rng.normal(size=300), 1)  # ties on purpose
    y = np.round(0.4 * x + rng.normal(size=300), 1)
    lab = rng.integers(0, 2, 300).astype(bool)
    w = np.ones(300)
    assert weighted_spearman(RankPlan(x), RankPlan(y), w) == pytest.approx(stats.spearmanr(x, y).statistic, abs=1e-12)
    assert weighted_auroc(RankPlan(y), lab, w) == pytest.approx(auroc_binary(lab, y), abs=1e-12)


def test_integer_weights_equal_concatenated_resample():
    rng = np.random.default_rng(1)
    n = 200
    x, y = rng.normal(size=n), rng.normal(size=n)
    lab = rng.integers(0, 2, n).astype(bool)
    blk = rng.integers(0, 5, n)
    mult = np.array([2, 0, 1, 3, 1], dtype=float)
    w = mult[blk]
    idx = np.concatenate([np.flatnonzero(blk == b).repeat(int(m)) for b, m in enumerate(mult)])
    assert weighted_spearman(RankPlan(x), RankPlan(y), w) == pytest.approx(stats.spearmanr(x[idx], y[idx]).statistic, abs=1e-12)
    assert weighted_auroc(RankPlan(y), lab, w) == pytest.approx(auroc_binary(lab[idx], y[idx]), abs=1e-12)
    ss_res = ((x[idx] - y[idx]) ** 2).sum()
    assert weighted_r2(x, y, w) == pytest.approx(1 - ss_res / ((x[idx] - x[idx].mean()) ** 2).sum(), abs=1e-12)
    assert weighted_mean(x, w) == pytest.approx(x[idx].mean())


def test_bootstrap_draws_identical_to_frozen_function_and_shared_between_arms():
    rng = np.random.default_rng(2)
    blocks = np.array(CHROMS)[rng.integers(0, 22, 500)]
    vals = rng.normal(size=500)
    names = np.unique(blocks)
    boot = BlockBootstrap(names, n_boot=50, seed=17)
    bi = block_index(blocks, names)
    rep = boot.replicates(lambda w: weighted_mean(vals, w), bi)
    pt, lo, hi = bootstrap_ci_by_block(vals, blocks, np.mean, n_boot=50, seed=17)
    assert pt == pytest.approx(weighted_mean(vals, np.ones(500)))
    assert np.quantile(rep, [0.025, 0.975]) == pytest.approx([lo, hi])
    # two different arms evaluated on the same table share the resampled chromosomes (and the draw table is reproducible)
    assert (BlockBootstrap(names, 50, 17).draws == boot.draws).all()
    assert boot.fingerprint() == BlockBootstrap(names, 50, 17).fingerprint()
    assert boot.fingerprint() != BlockBootstrap(names, 50, 18).fingerprint()


def test_paired_contrast_uses_same_replicates_and_p_formula():
    rng = np.random.default_rng(3)
    names = np.array(CHROMS)
    blocks = names[rng.integers(0, 22, 800)]
    base = rng.normal(size=800)
    a, b = base + 0.5, base  # arm A uniformly better: paired delta is exactly 0.5 in every replicate
    boot = BlockBootstrap(np.unique(blocks), 100, 17)
    bi = block_index(blocks, boot.names)
    ra = boot.replicates(lambda w: weighted_mean(a, w), bi)
    rb = boot.replicates(lambda w: weighted_mean(b, w), bi)
    c = contrast(a.mean(), ra, b.mean(), rb)
    assert c["delta"] == pytest.approx(0.5)
    assert c["ci_lo"] == pytest.approx(0.5) and c["ci_hi"] == pytest.approx(0.5)  # unpaired resampling would not give this
    assert c["p"] == pytest.approx(2 * 1 / 101)  # k(D<=0)=0 -> (1+0)/(1+100), two-sided x2
    assert bootstrap_p_two_sided(np.zeros(9))["p"] == 1.0
    with pytest.raises(ValueError):
        contrast(0, np.zeros(3), 0, np.zeros(4))


def test_holm_known_values():
    adj = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    assert adj == pytest.approx({"d": 0.02, "a": 0.03, "c": 0.06, "b": 0.06})
    assert holm_adjust({"x": 0.9, "y": 0.6})["y"] == pytest.approx(1.0)  # capped
    assert holm_adjust({}) == {}


# ----------------------------------------------------------------------------- embeddings / cosine / zero rows
def _write_store(path, ids, emb, dtype=np.float16, order=None):
    order = np.arange(len(ids)) if order is None else order
    with h5py.File(path, "w") as h:
        h.create_dataset("cpg_idx", data=np.asarray(ids, dtype=np.int64)[order])
        h.create_dataset("embedding", data=np.asarray(emb, dtype=dtype)[order])


def _spec(path, root, dim, arm="arm"):
    return ArmSpec(arm, arm, str(Path(path).relative_to(root)), sha256_file(path), dim, "float16")


def test_store_unsorted_ids_float16_cast_and_failure_modes(tmp_path):
    ids = np.arange(100, 130)
    emb = np.random.default_rng(0).normal(size=(30, 6)).astype(np.float16)
    order = np.random.default_rng(1).permutation(30)
    p = tmp_path / "s.h5"
    _write_store(p, ids, emb, order=order)
    st = EmbeddingStore(_spec(p, tmp_path, 6), tmp_path)
    assert not st.cpg_idx_sorted
    q = np.array([129, 100, 115, 100])
    got = st.load(q)
    assert got.dtype == np.float32 and np.array_equal(got, emb[q - 100].astype(np.float32))
    with pytest.raises(ValueError, match="absent"):
        st.load(np.array([100, 999]))
    bad = ArmSpec("arm", "arm", "s.h5", "0" * 64, 6, "float16")
    with pytest.raises(RuntimeError, match="sha256"):
        EmbeddingStore(bad, tmp_path)
    emb2 = emb.copy()
    emb2[3, 2] = np.inf
    p2 = tmp_path / "s2.h5"
    _write_store(p2, ids, emb2)
    with pytest.raises(ValueError, match="non-finite"):
        EmbeddingStore(_spec(p2, tmp_path, 6), tmp_path).load(ids)


def test_cosine_matches_numpy_and_zero_rows_give_identical_pair_sets_across_arms(tmp_path):
    rng = np.random.default_rng(0)
    ids = np.arange(1000, 1040)
    zero_sets, stores = {}, {}
    for k, zr in enumerate([[3, 17], [], [17, 30]]):  # per-arm all-zero rows (30 only in arm 2)
        e = rng.normal(size=(40, 8)).astype(np.float32)
        e[zr] = 0
        p = tmp_path / f"a{k}.h5"
        _write_store(p, ids, e, dtype=np.float32, order=rng.permutation(40))
        st = EmbeddingStore(_spec(p, tmp_path, 8, f"a{k}"), tmp_path)
        stores[k], zero_sets[k] = st, st.zero_row_ids()
    assert zero_sets[0].tolist() == [1003, 1017] and zero_sets[1].size == 0
    union = union_zero_loci(zero_sets)
    assert union.tolist() == [1003, 1017, 1030]
    ci, cj = rng.choice(ids, 200), rng.choice(ids, 200)
    keep = exclusion_mask([ci, cj], union)
    assert keep.sum() < 200
    for st in stores.values():  # same kept pair set for every arm; cosine correct and finite on it
        E = st.load(ids)
        c = pair_cosine(E, ci[keep] - 1000, cj[keep] - 1000)
        ref = np.array([E[a - 1000] @ E[b - 1000] / (np.linalg.norm(E[a - 1000]) * np.linalg.norm(E[b - 1000]))
                        for a, b in zip(ci[keep], cj[keep], strict=True)])
        assert np.isfinite(c).all() and np.allclose(c, ref, atol=1e-9)
    # without exclusion a zero row gives NaN (undefined), proving why the union rule is needed
    E0 = stores[0].load(ids)
    assert np.isnan(pair_cosine(E0, np.array([3]), np.array([5]))[0])


def test_topk_neighbors_match_bruteforce():
    rng = np.random.default_rng(5)
    E = rng.normal(size=(60, 5)).astype(np.float32)
    nn = topk_cosine_neighbors(E, 4, chunk=17)
    Z = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)
    assert (nn == np.argsort(-S, axis=1)[:, :4]).all()
    with pytest.raises(ValueError):
        E[0] = 0
        topk_cosine_neighbors(E, 3)


# ----------------------------------------------------------------------------- probes
def test_ridge_matches_sklearn_pipeline_and_does_not_see_test_fold():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    n, d = 400, 6
    X = rng.normal(size=(n, d)) * rng.uniform(0.5, 3, d) + rng.normal(size=d)
    y = X @ rng.normal(size=d) + rng.normal(size=n)
    fold = rng.integers(0, 5, n)
    out = cp.ridge_blocked_cv(X, y, fold, alphas=(10.0,))
    f = 2
    tr, te = fold != f, fold == f
    sc = StandardScaler().fit(X[tr])
    ref = Ridge(alpha=10.0).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    assert np.allclose(out["oof"][te, 0], ref, atol=1e-8)
    # leak test: corrupting y of the test fold must not change that fold's out-of-fold predictions nor its chosen alpha
    full = cp.ridge_blocked_cv(X, y, fold)
    y2 = y.copy()
    y2[fold == f] = 1e6 * rng.normal(size=(fold == f).sum())
    pert = cp.ridge_blocked_cv(X, y2, fold)
    fi = list(full["folds"]).index(f)
    assert np.allclose(full["oof"][fold == f], pert["oof"][fold == f])
    assert full["alpha"][fi, 0] == pert["alpha"][fi, 0]
    assert np.isfinite(full["oof"]).all()


def test_logistic_blocked_no_leak_and_grid_constants():
    rng = np.random.default_rng(1)
    n = 300
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    fold = rng.integers(0, 5, n)
    a = cp.logistic_blocked_cv(X, y, fold, Cs=(0.01, 1.0), max_iter=500)
    y2 = y.copy()
    y2[fold == 1] = 1 - y2[fold == 1]
    b = cp.logistic_blocked_cv(X, y2, fold, Cs=(0.01, 1.0), max_iter=500)
    assert np.allclose(a["oof"][fold == 1], b["oof"][fold == 1])
    assert auroc_binary(y, a["oof"]) > 0.8
    assert cp.RIDGE_ALPHAS == (1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4)
    assert cp.LOGREG_CS == (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    with pytest.raises(AssertionError):
        cp.assert_blocked([0, 1], [1, 2])
    with pytest.raises(AssertionError):
        cp.assert_blocked([0], [1], ["chr1", "chr2"], ["chr2"])
    with pytest.raises(ValueError):
        cp.ridge_blocked_cv(X, y, np.where(fold == 0, -1, fold))


# ----------------------------------------------------------------------------- synthetic frozen tree + endpoint drivers
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthroot")
    rng = np.random.default_rng(0)
    per = 30
    chrom = np.repeat(CHROMS, per)
    n = len(chrom)
    ids = np.arange(1, n + 1) * 10
    pos = np.tile(np.arange(per) * 1000 + 1, 22)
    c2f = chromosome_blocked_folds(CHROMS, 5, 17)
    fold = np.array([c2f[c] for c in chrom])
    base = root / le.BV
    for sub in ("external_methylation_programs", "3d_genome", "regulatory_activity", "replication_domains", "pmd_decato2020"):
        (base / sub).mkdir(parents=True)
    latent = rng.normal(size=(n, 3))

    def make_store(name, dim, zero_rows=(), dtype=np.float16):
        e = latent @ rng.normal(size=(3, dim)) + 0.3 * rng.normal(size=(n, dim))
        e = e.astype(np.float32)
        e[list(zero_rows)] = 0
        p = root / f"{name}.h5"
        _write_store(p, ids, e, dtype=dtype, order=rng.permutation(n))
        return p

    stores = {"A": make_store("A", 8, zero_rows=[5, 77], dtype=np.float32), "B": make_store("B", 5), "C": make_store("C", 4)}
    # RT universe (+2 ineligible), consensus depends on latent
    rt = pd.DataFrame({"cpg_idx": ids, "chrom": chrom, "pos": pos})
    for ln in le.RT_LINES:
        rt[f"rt_{ln}"] = (latent[:, 0] + 0.5 * rng.normal(size=n)).astype(np.float32)
    rt["rt_consensus_z"] = np.mean([rt[f"rt_{ln}"] for ln in le.RT_LINES], axis=0).astype(np.float32)
    rt.loc[[0, 1], "rt_consensus_z"] = np.nan
    rt.loc[[0, 1], [f"rt_{ln}" for ln in le.RT_LINES]] = np.nan
    rt.to_parquet(base / le.FROZEN_FILES["rt"])
    (base / le.FROZEN_FILES["rt_frozen"]).write_text(json.dumps({"folds": {"chrom_to_fold": c2f}}))
    # Loyfer pairs: 6 strata x 40 pairs
    rows = []
    for s in le.LOYFER_ALL_STRATA:
        a = rng.integers(0, n, 40)
        b = rng.integers(0, n, 40)
        for i, j in zip(a, b, strict=True):
            rows.append({"cpg_i": ids[i], "cpg_j": ids[j], "stratum": s, "distance_bp": 1.0, "n_shared_groups": int(rng.integers(20, 40)),
                         "loyfer_pearson": float(np.clip(latent[i] @ latent[j] / 3 + 0.2 * rng.normal(), -1, 1)),
                         "loyfer_spearman": float(rng.normal()), "pearson_defined": True, "draw_order": 0})
    pd.DataFrame(rows).to_parquet(base / le.FROZEN_FILES["loyfer_pairs"])
    # markers
    mk = pd.DataFrame({"cpg_idx": ids[rng.choice(n, 24, replace=False)], "chrom": "chr1", "pos": 1,
                       "cell_type": np.tile(["T", "B", "Liver"], 8), "source_table": ["U25"] * 12 + ["U250"] * 12,
                       "marker_source": "github_UXM_deconv_hg38@x", "provenance_strength": "x", "in_present_group": True})
    mk.to_parquet(base / le.FROZEN_FILES["loyfer_markers"])

    def micro(inter):
        pr = []
        m = 60
        for t in range(m):
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            k, l_ = rng.integers(0, n), rng.integers(0, n)
            f = int(fold[i]) if not (inter and t < 5) else -1
            pr.append({"pair_id": f"P{t}", "cpg_i": ids[i], "cpg_j": ids[j], "chrom_i": chrom[i], "chrom_j": chrom[j], "distance": float(rng.integers(2e4, 2e6)),
                       "label": 1, "matched_to": f"C{t}", "fold": f})
            pr.append({"pair_id": f"C{t}", "cpg_i": ids[k], "cpg_j": ids[l_], "chrom_i": chrom[i], "chrom_j": chrom[l_], "distance": float(rng.integers(2e4, 2e6)),
                       "label": 0, "matched_to": f"P{t}", "fold": f})
        return pd.DataFrame(pr)

    micro(False).to_parquet(base / le.FROZEN_FILES["microc_H1_intra"])
    micro(False).to_parquet(base / le.FROZEN_FILES["microc_HFFc6_intra"])
    micro(True).to_parquet(base / le.FROZEN_FILES["microc_H1_inter"])
    for k in ("comp_H1", "comp_GM12878"):
        e1 = latent[:, 1] + 0.3 * rng.normal(size=n)
        pd.DataFrame({"cpg_idx": ids, "chrom": chrom, "pos": pos, "bin": 0, "E1": np.where(np.arange(n) < 3, np.nan, e1),
                      "compartment": "A", "bin_valid": True, "fold": fold}).to_parquet(base / le.FROZEN_FILES[k])
    for k in ("f5_membership", "f5_win500", "f5_tss5k"):
        sel = rng.choice(n, 200, replace=False)
        lab = np.r_[np.ones(100), np.zeros(100)].astype(int)
        order = np.argsort(-latent[sel, 2] + 0.1 * rng.normal(size=200))
        sel = sel[np.r_[order[:100], order[100:]]]
        pd.DataFrame({"cpg_idx": ids[sel], "chrom": chrom[sel], "pos": 1, "label": lab, "matched_to": 0, "enhancer_id": "x", "fold": fold[sel]}) \
            .to_parquet(base / le.FROZEN_FILES[k])
    a_i, a_j = rng.integers(0, n, 150), rng.integers(0, n, 150)
    pd.DataFrame({"i": ids[a_i], "j": ids[a_j], "chrom_i": chrom[a_i], "chrom_j": chrom[a_j],
                  "profile_sim": rng.normal(size=150), "kind": np.where(np.arange(150) % 2, "intra", "inter")}) \
        .to_parquet(base / le.FROZEN_FILES["f5_activity"])
    pd.DataFrame({"cpg_idx": ids, "chrom": chrom, "pmd_in_any": (latent[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)}) \
        .to_parquet(base / le.FROZEN_FILES["pmd"])
    universe = pd.DataFrame({"cpg_idx": ids, "chrom": chrom})
    return SimpleNamespace(root=root, stores=stores, ids=ids, universe=universe, n=n)


def _ctx(synth, key, zero_union, n_boot=15):
    spec = ArmSpec(key, key, f"{key}.h5", sha256_file(synth.root / f"{key}.h5"), {"A": 8, "B": 5, "C": 4}[key], "float16")
    store = EmbeddingStore(spec, synth.root)
    boot = BlockBootstrap(np.array(CHROMS) if False else np.unique(synth.universe.chrom), n_boot=n_boot, seed=17)
    return le.EvalContext(synth.root, store, zero_union, boot, synth.universe)


def _union(synth):
    zs = {}
    for k in synth.stores:
        spec = ArmSpec(k, k, f"{k}.h5", sha256_file(synth.root / f"{k}.h5"), {"A": 8, "B": 5, "C": 4}[k], "x")
        zs[k] = EmbeddingStore(spec, synth.root).zero_row_ids()
    return union_zero_loci(zs)


def test_loyfer_equal_stratum_weight_and_identical_exclusions(synth):
    union = _union(synth)
    assert set(union.tolist()) == {synth.ids[5], synth.ids[77]}
    res = {k: le.loyfer_profile(_ctx(synth, k, union)) for k in "ABC"}
    ex = [r.meta["exclusion"] for r in res.values()]
    assert ex[0] == ex[1] == ex[2]  # identical pair set / counts for every arm
    assert ex[0]["n_total"] == 240 and ex[0]["n_excluded_zero_locus"] == ex[0]["n_total"] - ex[0]["n_kept"]
    assert res["A"].primary_stat in res["A"].stats
    names = set(res["A"].stats)
    for s in le.LOYFER_ALL_STRATA:
        assert f"spearman_{s}" in names
    assert {"spearman_all_pairs", "spearman_pooled_eqw__profile_spearman", "spearman_pooled_eqw__min30_shared_groups"} <= names
    # equal weighting: independently recompute and verify that duplicating the interchromosomal pairs does not change the primary
    ctx = _ctx(synth, "B", union)
    prim = res["B"].stats[res["B"].primary_stat].value
    p = ctx.path("loyfer_pairs")
    df = pd.read_parquet(p)
    d = df[exclusion_mask([df.cpg_i.to_numpy(), df.cpg_j.to_numpy()], union)]
    cos = ctx.cosine_pairs(d.cpg_i.to_numpy(), d.cpg_j.to_numpy())
    pooled = d.stratum.isin(le.LOYFER_POOLED_STRATA).to_numpy()
    sub, c = d[pooled], cos[pooled]
    w = np.where(sub.stratum == ">1Mb", 1 / (sub.stratum == ">1Mb").sum(), 1 / (sub.stratum == "interchromosomal").sum())

    def wsp(x, y, w):
        return weighted_spearman(RankPlan(x), RankPlan(y), w)
    assert prim == pytest.approx(wsp(sub.loyfer_pearson.to_numpy(), c, w), abs=1e-12)
    dup = pd.concat([sub, sub[sub.stratum == "interchromosomal"]])
    cdup = np.r_[c, c[(sub.stratum == "interchromosomal").to_numpy()]]
    wdup = np.where(dup.stratum == ">1Mb", 1 / (dup.stratum == ">1Mb").sum(), 1 / (dup.stratum == "interchromosomal").sum())
    assert wsp(dup.loyfer_pearson.to_numpy(), cdup, wdup) == pytest.approx(prim, abs=1e-12)
    # strata are reported separately and the pooled value differs from the plain all-pairs descriptive value
    assert res["B"].stats["spearman_all_pairs"].cls == "descriptive"


def test_microc_endpoints_auroc_delta_cosine_and_fold_filter(synth):
    union = _union(synth)
    r = le.microc_H1_inter(_ctx(synth, "B", union))
    ex = r.meta["exclusion"]
    assert ex["n_rows_fold_negative_removed"] == 10 and ex["n_rows_file"] == 120  # 5 matched pairs have fold -1
    assert ex["n_matched_pairs"] == 55
    assert ex["n_matched_pairs_kept_for_delta_cosine"] <= 55
    assert 0 <= r.stats["auroc_contact_vs_noncontact"].value <= 1
    h = le.microc_H1_intra(_ctx(synth, "A", union))
    assert h.primary_stat == "auroc_contact_vs_noncontact" and h.cls == "primary"
    assert any(k.startswith("auroc_distance_bin_") for k in h.stats)
    assert h.stats["delta_cosine_paired_mean"].rep.shape == (15,)
    # same kept rows for every arm
    h2 = le.microc_H1_intra(_ctx(synth, "C", union))
    assert h.meta["exclusion"] == h2.meta["exclusion"]


def test_probe_endpoints_run_on_synthetic_and_keep_zero_rows(synth):
    union = _union(synth)
    ctx = _ctx(synth, "A", union)  # arm A has the all-zero rows: probe endpoints must NOT exclude them
    rt = le.rt_consensus(ctx)
    assert rt.meta["n_eligible"] == synth.n - 2
    assert rt.meta["n_zero_embedding_rows_in_probe_set"] == 2
    assert rt.stats["spearman_oof_consensus"].value > 0.3
    assert "per_fold_spearman" in rt.stats["spearman_oof_consensus"].meta and "per_fold_r2" in rt.stats["r2_oof_consensus"].meta
    assert sum(k.startswith("spearman_oof_line_") for k in rt.stats) == 15
    f5 = le.fantom5_membership(ctx)
    assert f5.stats["auroc_oof_bg_enh5k"].n == 200 and f5.stats["auroc_oof__win500"].cls == "sensitivity"
    assert le.compartment_H1(ctx).meta["n_rows_used"] == synth.n - 3
    assert 0 <= le.pmd_probe(ctx).stats["auroc_oof_pmd_in_any"].value <= 1
    act = le.fantom5_activity(ctx)
    assert act.meta["exclusion"]["n_total"] == 150


def test_marker_knn_endpoint(synth, monkeypatch):
    monkeypatch.setattr(le, "KNN_N_PERM", 9)
    union = _union(synth)
    r = le.loyfer_marker_knn(_ctx(synth, "B", union))
    assert r.meta["n_candidates"] == synth.n - 2 and r.meta["k"] == 10
    s = r.stats["enrichment_github_U25_U250"]
    assert s.value > 0 and 0 < s.meta["perm_p_value_one_sided"] <= 1  # kernel == frozen knn_enrichment (asserted inside)


def test_frozen_knn_enrichment_permutation_bug_is_documented_and_replacement_is_sound():
    from cpg_repr_benchmark.biological_validation_v2.metrics import knn_enrichment
    rng = np.random.default_rng(0)
    n, k = 200, 4
    chroms = np.repeat(["a", "b"], 100)
    lab = np.array([""] * n, dtype=object)
    lab[[3, 50, 120, 160]] = "X"
    # planted structure: markers are each other's neighbours
    nn = rng.integers(0, n, size=(n, k))
    for i in (3, 50, 120, 160):
        nn[i] = [j for j in (3, 50, 120, 160) if j != i] + [3]
    bg = {"X": 4 / n}
    # frozen function: point estimate fine, permutation path breaks (KeyError '') -- reported, never silently fixed
    assert knn_enrichment(lab, nn, k, bg)["enrichment"] > 1
    with pytest.raises(KeyError):
        knn_enrichment(lab, nn, k, bg, chroms=chroms, n_perm=20, seed=1)
    pp = le.knn_permutation_pvalue(lab, nn, k, bg, chroms, 199, 1)
    assert pp["observed"] == pytest.approx(knn_enrichment(lab, nn, k, bg)["enrichment"])
    assert pp["p_value"] < 0.05 and pp["null_mean"] < pp["observed"]
    # null data: random neighbours -> large p
    nn_r = rng.integers(0, n, size=(n, k))
    assert le.knn_permutation_pvalue(lab, nn_r, k, bg, chroms, 199, 1)["p_value"] > 0.05


# ----------------------------------------------------------------------------- gate, writer, manifest
def _git_repo(tmp_path):
    def g(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    g("init", "-q")
    g("config", "user.email", "t@t")
    g("config", "user.name", "t")
    return g


def test_gate_failure_modes_and_pass(tmp_path, monkeypatch):
    g = _git_repo(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/x.txt").write_text("x")
    g("add", "-A")
    g("commit", "-qm", "c1")
    c1 = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert not lg.check_tag(tmp_path).ok  # tag missing
    g("tag", "-a", lg.PROTOCOL_TAG, "-m", "t")
    assert not lg.check_tag(tmp_path).ok  # tag resolves to the wrong commit
    monkeypatch.setattr(lg, "PROTOCOL_COMMIT", c1)
    monkeypatch.setattr(lg, "PROTOCOL_COMMIT_PREFIX", c1[:7])
    assert lg.check_tag(tmp_path).ok
    # registration doc: missing -> untracked -> committed -> modified
    assert not lg.check_registration_committed(tmp_path).ok
    reg = tmp_path / lg.REGISTRATION_DOC
    reg.write_text("v1")
    assert "not tracked" in lg.check_registration_committed(tmp_path).detail
    g("add", "-A")
    g("commit", "-qm", "reg")
    assert lg.check_registration_committed(tmp_path).ok
    reg.write_text("v2")
    assert "modified" in lg.check_registration_committed(tmp_path).detail
    g("checkout", "--", lg.REGISTRATION_DOC)
    # frozen paths: adding a file is fine, modifying a pre-existing frozen file is not
    fz = tmp_path / "configs/biological_validation_v2/m.yaml"
    fz.parent.mkdir(parents=True)
    fz.write_text("a: 1")
    assert lg.check_frozen_unchanged(tmp_path).ok  # untracked addition tolerated
    g("add", "-A")
    g("commit", "-qm", "freeze-like")
    monkeypatch.setattr(lg, "PROTOCOL_COMMIT", subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip())
    assert lg.check_frozen_unchanged(tmp_path).ok
    fz.write_text("a: 2")
    assert not lg.check_frozen_unchanged(tmp_path).ok
    g("commit", "-qam", "modify frozen")
    assert not lg.check_frozen_unchanged(tmp_path).ok  # committed modification still detected vs the freeze commit
    # enforce() raises SystemExit listing failures
    with pytest.raises(SystemExit, match="PRE-RUN GATE FAILED"):
        lg.enforce([lg.Check("a", True), lg.Check("b", False, "boom")])
    lg.enforce([lg.Check("a", True)])


def test_gate_store_sha_and_tcga_and_protected_out_root(tmp_path):
    p = tmp_path / "s.h5"
    _write_store(p, np.arange(3), np.zeros((3, 2)))
    spec_ok = ArmSpec("a", "a", "s.h5", sha256_file(p), 2, "float16")
    assert lg.check_stores(tmp_path, ["a"], {"a": spec_ok}).ok
    spec_bad = ArmSpec("a", "a", "s.h5", "0" * 64, 2, "float16")
    assert not lg.check_stores(tmp_path, ["a"], {"a": spec_bad}).ok
    assert not lg.check_inputs_not_tcga(["data/x/TCGA_test.h5"]).ok
    assert not lg.check_inputs_not_tcga(["data/protocols/foo.npz"]).ok
    assert lg.check_inputs_not_tcga(["data/derived/bioval_v2/x.parquet"]).ok
    with pytest.raises(PermissionError):
        assert_readable("outputs/tcga_x/matrix.h5")
    for bad in (lg.BV, "configs/foo", "src/x", "docs/bioval_v2_prep/a"):
        assert not lg.check_out_root(tmp_path, tmp_path / bad).ok
        with pytest.raises(PermissionError):
            lg.GuardedWriter(tmp_path, tmp_path / bad)
    assert lg.check_out_root(tmp_path, tmp_path / "outputs/biological_validation_v2").ok


def test_guarded_writer_never_writes_outside_root_or_into_frozen(tmp_path):
    w = lg.GuardedWriter(tmp_path, tmp_path / "outputs/o")
    w.write_json("arm/x.json", {"a": np.float32(1.5), "b": np.arange(2)})
    assert json.loads((tmp_path / "outputs/o/arm/x.json").read_text()) == {"a": 1.5, "b": [0, 1]}
    with pytest.raises(PermissionError, match="escapes"):
        w.path("..", "elsewhere.txt")
    with pytest.raises(PermissionError, match="escapes"):
        w.path("..", "..", "..", lg.BV, "FROZEN.json")
    # an output root that is an ancestor of a protected path cannot write into it either
    w2 = lg.GuardedWriter(tmp_path, tmp_path)
    with pytest.raises(PermissionError, match="protected"):
        w2.path(lg.BV, "MANIFEST.json")
    with pytest.raises(PermissionError, match="protected"):
        w2.path("configs", "x.yaml")
    with pytest.raises(PermissionError, match="protected"):
        w2.path("src", "mod.py")
    assert not (tmp_path / lg.BV).exists()


def test_run_manifest_content(tmp_path, monkeypatch):
    g = _git_repo(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / lg.REGISTRATION_DOC).write_text("reg")
    g("add", "-A")
    g("commit", "-qm", "c")
    g("tag", "-a", lg.PROTOCOL_TAG, "-m", "t")
    m = lg.build_run_manifest(tmp_path, "arm", {"sha256_actual": "abc", "dim": 4, "dtype_on_disk": "float16"}, seed=17, n_boot=1000,
                              boot_fingerprint="fp", threads=2, exclusion_counts={"x": 1}, checks=[lg.Check("c", True)],
                              endpoints=["e"], command="cmd", manifest_checksums_sha256="m", out_root=tmp_path / "o")
    for k in ("head_commit", "protocol_tag", "protocol_commit", "registration_doc", "representation", "timestamp_utc", "seed",
              "n_bootstrap", "library_versions", "exclusion_counts", "no_freeze_command_used", "tcga_test_set_referenced",
              "frozen_manifest_checksums_sha256", "bootstrap_draw_table_sha256", "gate_checks"):
        assert k in m
    assert m["seed"] == 17 and m["n_bootstrap"] == 1000 and m["no_freeze_command_used"] is True
    assert m["registration_doc"]["sha256"] and m["representation"]["sha256_actual"] == "abc"
    assert {"numpy", "scipy", "pandas", "sklearn", "h5py"} <= set(m["library_versions"])
    monkeypatch.setitem(sys.modules, "freeze_4dn_pairs", SimpleNamespace())  # simulated import of a freeze command
    m2 = lg.build_run_manifest(tmp_path, "arm", {}, seed=17, n_boot=1000, boot_fingerprint="fp", threads=1, exclusion_counts={},
                               checks=[], endpoints=[], command="c")
    assert m2["no_freeze_command_used"] is False and m2["freeze_modules_imported"] == ["freeze_4dn_pairs"]


def test_new_modules_do_not_import_or_call_freeze_generators():
    root = Path(__file__).resolve().parents[1]
    forbidden = ("sample_pairs", "match_controls", "match_pairs", "build_covariates", "freeze_4dn_pairs", "freeze_fantom5_matching",
                 "loyfer_freeze_pairs", "build_checksum_manifest", "prepare_", "TCGA", ".write_text(", "to_parquet(")
    for rel in ("block_bootstrap", "embedding_eval", "chrom_probes", "launch_endpoints"):
        text = (root / f"src/cpg_repr_benchmark/bioval_v2_launch/{rel}.py").read_text()
        for f in forbidden:
            if f in ("TCGA",):
                assert f"'{f}" not in text
            else:
                assert f not in text.replace('"""', ""), (rel, f)


def test_runner_contrasts_end_to_end_on_synthetic(synth, tmp_path):
    spec = importlib.util.spec_from_file_location("run_evaluation", Path(__file__).resolve().parents[1] / "scripts/bioval_v2/run_evaluation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    union = _union(synth)
    out = tmp_path / "out"
    w = lg.GuardedWriter(tmp_path, out)
    for arm, key in ((mod.REFERENCE_ARM, "A"), ("functional_annotations_pca", "B")):
        ctx = _ctx(synth, key, union, n_boot=30)
        for eid in ("loyfer_profile", "microc_H1_intra10kb", "fantom5_membership", "rt_consensus"):
            mod.write_endpoint(w, arm, le.ENDPOINTS[eid][1](ctx))
        mod.assemble_results(w, arm)
        assert (out / arm / "results.csv").exists() and (out / arm / "results.json").exists()
    mod.cmd_contrasts(SimpleNamespace(out_root=str(out), n_boot=30), [mod.REFERENCE_ARM, "functional_annotations_pca"])
    c = json.loads((out / "contrasts" / f"{mod.REFERENCE_ARM}__vs__functional_annotations_pca.json").read_text())
    fam = [r for r in c["rows"] if r["in_holm_family"]]
    assert len(fam) == 4 and c["holm_family_missing_endpoints"] == []
    assert all(r["holm_p_adjusted"] >= r["p"] and r["holm_p_adjusted"] <= 1 for r in fam)
    assert all(r["holm_p_adjusted"] is None for r in c["rows"] if not r["in_holm_family"])
    assert (out / "contrasts" / f"{mod.REFERENCE_ARM}__vs__functional_annotations_pca.csv").exists()
    assert "regulatory - comparator" in fam[0]["sign_convention"]
