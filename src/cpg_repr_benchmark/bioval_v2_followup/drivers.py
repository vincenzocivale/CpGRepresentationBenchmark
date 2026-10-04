"""Endpoint drivers of the follow-up (Part A frozen sensitivities, Part B EXPLORATORY POST-HOC). Reuses launch kernels.

Every driver reads frozen files only (through ``fpath`` -> assert_readable), never writes to data/derived, never calls a
freeze/prepare command. Cosine-based drivers use the D2 zero-row exclusion (identical pair sets across arms); linear probes
keep all rows. Contrast/Holm: descriptive only (see compare.py).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cpg_repr_benchmark.biological_validation_v2.metrics import (
    auroc_binary,
    compartment_eigenvector_corr,
    spearman_profile_vs_embedding_sim,
)
from cpg_repr_benchmark.bioval_v2_launch import chrom_probes as cp
from cpg_repr_benchmark.bioval_v2_launch import launch_endpoints as le
from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import (
    RankPlan,
    percentile_ci,
    weighted_auroc,
    weighted_r2,
    weighted_spearman,
)
from cpg_repr_benchmark.bioval_v2_launch.embedding_eval import assert_readable, exclusion_mask, pair_cosine

from . import geometry as geo
from . import profile_ridge as pr
from . import registry as rg
from . import strata as st
from . import targets as tg

BV = le.BV
F = {
    "loyfer_pairs": "external_methylation_programs/frozen_pairs_seed17.parquet",
    "loyfer_beta": "external_methylation_programs/loyfer_celltype_beta.h5",
    "comp_H1_50": "3d_genome/H1_MicroC_cpg_compartment_50000.parquet",
    "comp_GM_50": "3d_genome/GM12878_HiC_cpg_compartment_50000.parquet",
    "f5_membership": "regulatory_activity/frozen_enhancer_membership_seed17.parquet",
    "f5_enhancers": "regulatory_activity/enhancers.parquet",
    "f5_activity": "regulatory_activity/frozen_activity_pairs_seed17.parquet",
    "f5_act_sample": "regulatory_activity/activity_sample_log2tpm.npy",
    "f5_act_coll": "regulatory_activity/activity_collapsed_log2tpm.npy",
    "f5_act_coll_idx": "regulatory_activity/activity_collapsed_index.parquet",
    "rt_frozen": "replication_domains/FROZEN_ENDPOINT.json",
}
RESULTS_BASE = "outputs/biological_validation_v2/" + rg.PROTOCOL_TAG


def fpath(ctx, key):
    p = ctx.root / BV / F[key]
    assert_readable(p)
    return p


def frozen_value(root, arm, endpoint, stat):
    """Value of a frozen primary statistic (read-only; used as run-time reproduction check and as verdict reference)."""
    p = root / RESULTS_BASE / arm / "endpoints" / f"{endpoint}.json"
    return json.loads(p.read_text())["stats"][stat]["value"]


def _stat_unw(ctx, name, x, y, blk, cls, mask=None):
    if mask is not None:
        x, y, blk = x[mask], y[mask], blk[mask]
    if len(x) < 3:
        return None
    px, py = RankPlan(x), RankPlan(y)
    return le.make_stat(ctx, name, lambda w: weighted_spearman(px, py, w), blk, len(x), cls=cls)


def _stat_pooled(ctx, name, x, y, strat, blk, cls, mask=None):
    sel = np.isin(strat, rg.POOLED_STRATA)
    if mask is not None:
        sel &= mask
    s = strat[sel]
    if any((s == k).sum() == 0 for k in rg.POOLED_STRATA):
        return None
    w0 = np.zeros(sel.sum())
    for k in rg.POOLED_STRATA:
        m = s == k
        w0[m] = 1.0 / m.sum()
    px, py = RankPlan(x[sel]), RankPlan(y[sel])
    return le.make_stat(ctx, name, lambda w: weighted_spearman(px, py, w), blk[sel], sel.sum(), base_w=w0, cls=cls)


def _add(res, s):
    if s is not None:
        res.add(s)
    return s


# ======================================================================================== Part A: Loyfer target variants
def loyfer_variant(ctx, variant: str):
    pairs = pd.read_parquet(fpath(ctx, "loyfer_pairs"))
    prof = tg.LoyferProfiles(fpath(ctx, "loyfer_beta"))
    repro = tg.assert_reproduces_frozen(pairs, prof)
    r, _ns = tg.variant_target(pairs, prof, variant)
    defined = np.isfinite(r)
    zkeep = exclusion_mask([pairs.cpg_i.to_numpy(), pairs.cpg_j.to_numpy()], ctx.zero_union)
    keep = defined & zkeep
    eid = f"loyfer_{variant}"
    res = le.EndpointResult(eid, "sensitivity", "external_methylation_programs", rg.LOYFER_PRIMARY_STAT, meta={
        "target_reproduction_check": repro, "variant": variant,
        "single_sample_groups_dropped": ([prof.groups[i] for i in prof.single_sample_groups()]
                                         if variant == "drop_single_sample_groups" else []),
        "exclusion": le.exclusion_summary(len(pairs), zkeep & defined, {
            "n_target_undefined_(<20_shared_or_constant)": int((~defined).sum()),
            "n_zero_locus_excluded_among_defined": int((defined & ~zkeep).sum()),
            "per_stratum_kept": {s: int((keep & (pairs.stratum == s).to_numpy()).sum()) for s in rg.ALL_STRATA}}),
        "similarity": "cosine(embedding)", "descriptive_only": "no Holm; raw p in contrasts"})
    d = pairs[keep].reset_index(drop=True)
    x = r[keep]
    cos = ctx.cosine_pairs(d.cpg_i.to_numpy(), d.cpg_j.to_numpy())
    blk = ctx.blk(ctx.chrom_of(d.cpg_i.to_numpy()))
    strat = d.stratum.to_numpy()
    for s in rg.ALL_STRATA:
        _add(res, _stat_unw(ctx, f"spearman_{s}", x, cos, blk, "descriptive", strat == s))
    _add(res, _stat_pooled(ctx, rg.LOYFER_PRIMARY_STAT, x, cos, strat, blk, "sensitivity"))
    a = _add(res, _stat_unw(ctx, "spearman_all_pairs", x, cos, blk, "descriptive"))
    le._check("loyfer variant all pairs", a.value, spearman_profile_vs_embedding_sim(x, cos))
    return res


# ======================================================================================== Part A: compartments 50 kb
def compartment_50kb(ctx, endpoint_id, key):
    d = pd.read_parquet(fpath(ctx, key))
    n0 = len(d)
    d = d[d.E1.notna()].reset_index(drop=True)
    c2f = le._chrom_to_fold_frozen(ctx)
    fold = d.chrom.map(c2f).to_numpy()
    X = ctx.store.load(d.cpg_idx.to_numpy())
    y = d.E1.to_numpy(dtype=np.float64)
    fit = cp.ridge_blocked_cv(X, y, fold)
    oof = fit["oof"][:, 0]
    blk = ctx.blk(d.chrom.to_numpy())
    px, py = RankPlan(y), RankPlan(oof)
    res = le.EndpointResult(endpoint_id, "sensitivity", "3d_genome", "spearman_E1_vs_oof_pred", meta={
        "n_rows_file": n0, "n_rows_used": len(d), "probe": "ridge (D3), fold via frozen chrom->fold map",
        "alpha_per_fold": fit["alpha"][:, 0].tolist(), "zero_rows_kept": True,
        "resolution": "50 kb frozen E1 table (not in the primary 100 kb results)"})
    s = res.add(le.make_stat(ctx, "spearman_E1_vs_oof_pred", lambda w: weighted_spearman(px, py, w), blk, len(d), cls="sensitivity"))
    le._check(endpoint_id, s.value, compartment_eigenvector_corr(y, oof))
    res.add(le.make_stat(ctx, "r2_oof", lambda w: weighted_r2(y, oof, w), blk, len(d), cls="sensitivity"))
    return res


# ======================================================================================== Part A: FANTOM5 membership N
def fantom5_membership_N(ctx, N: int):
    m = pd.read_parquet(fpath(ctx, "f5_membership"))
    enh = pd.read_parquet(fpath(ctx, "f5_enhancers"), columns=["enhancer_id", "n_samples_expressed"]).set_index("enhancer_id")
    nexp = enh.n_samples_expressed
    pos = m[m.label == 1]
    kept = pos[pos.enhancer_id.map(nexp).to_numpy() >= N]
    ids = set(kept.cpg_idx) | set(kept.matched_to)
    d = m[m.cpg_idx.isin(ids)].reset_index(drop=True)
    assert (d.label == 1).sum() == (d.label == 0).sum() == len(kept)
    name = f"auroc_oof__N{N}"
    res = le.EndpointResult(f"fantom5_membership_N{N}", "sensitivity", "regulatory_activity", name, meta={
        "N": N, "definition": "positives = frozen positives whose enhancer is expressed (TPM>=1) in >= N libraries; their FROZEN matched "
                              "controls kept; NO re-matching (pair-level exact strata preserved by construction)",
        "n_positives_frozen": len(pos), "n_positives_kept": len(kept), "n_pairs": len(kept)})
    X = ctx.store.load(d.cpg_idx.to_numpy())
    fit = cp.logistic_blocked_cv(X, d.label.to_numpy(), d.fold.to_numpy())
    y = d.label.to_numpy().astype(bool)
    oof = fit["oof"]
    plan = RankPlan(oof)
    blk = ctx.blk(d.chrom.to_numpy())
    s = res.add(le.make_stat(ctx, name, lambda w: weighted_auroc(plan, y, w), blk, len(d), cls="sensitivity", meta={
        "C_per_fold": fit["C"].tolist(), "n_iter_max": fit["n_iter_max"], "n_nonconverged_fits": fit["n_nonconverged_fits"]}))
    le._check(name, s.value, auroc_binary(y, oof))
    return res


# ======================================================================================== Part A: FANTOM5 activity variants
def _row_pearson(A, ia, ib, cols, chunk=20_000):
    out = np.full(len(ia), np.nan)
    for s in range(0, len(ia), chunk):
        X = np.asarray(A[ia[s:s + chunk]][:, cols], dtype=np.float64)
        Y = np.asarray(A[ib[s:s + chunk]][:, cols], dtype=np.float64)
        X -= X.mean(1, keepdims=True)
        Y -= Y.mean(1, keepdims=True)
        den = np.sqrt((X * X).sum(1) * (Y * Y).sum(1))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[s:s + chunk] = np.where(den > 0, (X * Y).sum(1) / den, np.nan)
    return out


def fantom5_activity_variant(ctx, endpoint_id: str, mode: str, tol=1e-6):
    pairs = pd.read_parquet(fpath(ctx, "f5_activity"))
    enh = pd.read_parquet(fpath(ctx, "f5_enhancers"), columns=["enhancer_id", "n_samples_expressed"])
    row = pd.Series(np.arange(len(enh)), index=enh.enhancer_id.to_numpy())
    ia, ib = row.reindex(pairs.enh_i).to_numpy().astype(np.int64), row.reindex(pairs.enh_j).to_numpy().astype(np.int64)
    Ac = np.load(fpath(ctx, "f5_act_coll"), mmap_mode="r")
    ref = _row_pearson(Ac, ia, ib, np.arange(Ac.shape[1]))
    dif = float(np.nanmax(np.abs(ref - pairs.profile_sim.to_numpy())))
    if dif > tol or np.isnan(ref).any():
        raise AssertionError(f"collapsed profile similarity not reproduced from the frozen activity matrix (max diff {dif})")
    if mode == "sample":
        A = np.load(fpath(ctx, "f5_act_sample"), mmap_mode="r")
        cols = np.arange(A.shape[1])
        desc = "Pearson over all 1,829 libraries (log2(TPM+1), activity_sample_log2tpm.npy)"
    else:
        ci = pd.read_parquet(fpath(ctx, "f5_act_coll_idx"))
        cols = np.flatnonzero((ci.category == mode).to_numpy())
        A = Ac
        if len(cols) < 3:
            raise AssertionError(f"category {mode}: fewer than 3 groups")
        desc = f"Pearson over the {len(cols)} collapsed groups of category '{mode}' (min_groups=3, as in the frozen PREP helper)"
    sim = _row_pearson(A, ia, ib, cols)
    defined = np.isfinite(sim)
    zkeep = exclusion_mask([pairs.i.to_numpy(), pairs.j.to_numpy()], ctx.zero_union)
    keep = defined & zkeep
    res = le.EndpointResult(endpoint_id, "sensitivity", "regulatory_activity", "spearman_all_pairs", meta={
        "definition": desc, "collapsed_reproduction_max_abs_diff": dif,
        "exclusion": le.exclusion_summary(len(pairs), keep, {"n_target_undefined": int((~defined).sum()),
                                                             "n_zero_locus_excluded_among_defined": int((defined & ~zkeep).sum())})})
    d = pairs[keep].reset_index(drop=True)
    x = sim[keep]
    cos = ctx.cosine_pairs(d.i.to_numpy(), d.j.to_numpy())
    blk = ctx.blk(d.chrom_i.to_numpy())
    a = _add(res, _stat_unw(ctx, "spearman_all_pairs", x, cos, blk, "sensitivity"))
    le._check(endpoint_id, a.value, spearman_profile_vs_embedding_sim(x, cos))
    for kind in ("intra", "inter"):
        _add(res, _stat_unw(ctx, f"spearman_{kind}", x, cos, blk, "descriptive", (d.kind == kind).to_numpy()))
    return res


SENS_DRIVERS = {
    "loyfer_mask_lenient": lambda c: loyfer_variant(c, "mask_lenient"),
    "loyfer_drop_single_sample_groups": lambda c: loyfer_variant(c, "drop_single_sample_groups"),
    "compartment_E1_probe_H1_50kb": lambda c: compartment_50kb(c, "compartment_E1_probe_H1_50kb", "comp_H1_50"),
    "compartment_E1_probe_GM12878_50kb": lambda c: compartment_50kb(c, "compartment_E1_probe_GM12878_50kb", "comp_GM_50"),
    **{f"fantom5_membership_N{n}": (lambda c, n=n: fantom5_membership_N(c, n)) for n in (1, 3, 10, 20)},
    "fantom5_activity_sample_level": lambda c: fantom5_activity_variant(c, "fantom5_activity_sample_level", "sample"),
    "fantom5_activity_cat_cell_lines": lambda c: fantom5_activity_variant(c, "fantom5_activity_cat_cell_lines", "cell lines"),
    "fantom5_activity_cat_primary_cells": lambda c: fantom5_activity_variant(c, "fantom5_activity_cat_primary_cells", "primary cells"),
    "fantom5_activity_cat_tissues": lambda c: fantom5_activity_variant(c, "fantom5_activity_cat_tissues", "tissues"),
}
assert set(SENS_DRIVERS) == set(rg.ITEMS)


# ======================================================================================== Part B1 + B2
def exploratory_similarity(ctx, arm_id: str):
    pairs = pd.read_parquet(fpath(ctx, "loyfer_pairs"))
    prof = tg.LoyferProfiles(fpath(ctx, "loyfer_beta"))
    repro = tg.assert_reproduces_frozen(pairs, prof)
    vmin_all, mmean_all = st.pair_strata_values(pairs, prof)
    edge_check = st.check_registered_edges(vmin_all, mmean_all)
    zkeep = exclusion_mask([pairs.cpg_i.to_numpy(), pairs.cpg_j.to_numpy()], ctx.zero_union)
    d = pairs[zkeep].reset_index(drop=True)
    vm, mm = vmin_all[zkeep], mmean_all[zkeep]
    x = d.loyfer_pearson.to_numpy()
    strat = d.stratum.to_numpy()
    cl = st.cells(strat, vm, mm)
    blk = ctx.blk(ctx.chrom_of(d.cpg_i.to_numpy()))
    # universe mean / PC1 (registered: all loci minus D2 union-zero loci)
    uni = ctx.universe.cpg_idx.to_numpy()
    uni = uni[~np.isin(uni, ctx.zero_union)]
    Eu = ctx.store.load(uni)
    mu, v1, ev1 = geo.universe_mean_pc1(Eu)
    del Eu
    uniq, inv = np.unique(np.r_[d.cpg_i.to_numpy(), d.cpg_j.to_numpy()], return_inverse=True)
    E = ctx.store.load(uniq)
    ia, ib = inv[:len(d)], inv[len(d):]
    sims = {
        "cosine": pair_cosine(E, ia, ib),
        "centered_cosine": geo.centered_cosine(E, ia, ib, mu),
        "pc1_removed_cosine": geo.pc1_removed_cosine(E, ia, ib, mu, v1),
        "neg_euclidean": geo.neg_euclidean(E, ia, ib),
    }
    del E
    res = le.EndpointResult("exploratory_similarity", "exploratory_posthoc", "external_methylation_programs", None, meta={
        "EXPLORATORY_POST_HOC": True, "target_reproduction_check": repro, "registered_edges_check": edge_check,
        "exclusion": le.exclusion_summary(len(pairs), zkeep),
        "n_pairs_per_cell": {k: int(m.sum()) for k, m in cl.items()},
        "pc1_explained_variance_ratio": ev1, "universe_n_loci": len(uni),
        "mu_sha256": __import__("hashlib").sha256(mu.tobytes()).hexdigest(),
        "pc1_sha256": __import__("hashlib").sha256(v1.tobytes()).hexdigest(),
        "target": "frozen loyfer_pearson (primary mask)"})
    pooled_cells = ["all", *[k for k in cl if k.startswith(("V_", "M_"))]]
    for g, y in sims.items():
        for cname, mask in cl.items():
            _add(res, _stat_unw(ctx, f"spearman__{g}__{cname}", x, y, blk, "exploratory_posthoc", mask))
        for cname in pooled_cells:
            _add(res, _stat_pooled(ctx, f"spearman_pooled_eqw__{g}__{cname}", x, y, strat, blk, "exploratory_posthoc",
                                   None if cname == "all" else cl[cname]))
    fv = frozen_value(ctx.root, arm_id, "loyfer_profile", rg.LOYFER_PRIMARY_STAT)
    got = res.stats["spearman_pooled_eqw__cosine__all"].value
    if abs(fv - got) > 1e-9:
        raise AssertionError(f"B1/B2 cosine pooled statistic {got} does not reproduce the frozen primary {fv}")
    res.meta["reproduces_frozen_primary"] = {"frozen": fv, "recomputed": got}
    return res


# ======================================================================================== Part B3
def exploratory_profile_ridge(ctx, arm_id: str):
    prof = tg.LoyferProfiles(fpath(ctx, "loyfer_beta"))
    P = prof.matrix("primary")
    full = ~np.isnan(P).any(1)
    Pf = P[full]
    ids, chrom = prof.cpg[full], prof.chrom[full]
    c2f = le._chrom_to_fold_frozen(ctx)
    fold = pd.Series(chrom).map(c2f).to_numpy()
    if np.isnan(fold.astype(float)).any():
        raise AssertionError("chromosome outside the frozen fold map")
    fold = fold.astype(int)
    X = ctx.store.load(ids)
    targets = pr.assemble_targets(Pf)
    fit, folds = pr.fit_all(X, targets, fold)
    blk = ctx.blk(chrom)
    nb = len(ctx.boot.names)
    _, cpg_var, _ = st.per_cpg_stats(Pf)
    subsets = {"all_cpgs": np.ones(len(ids), bool), "variable_cpgs": cpg_var >= rg.VARIABLE_CPG_VAR}
    out = {"EXPLORATORY_POST_HOC": True, "arm": arm_id, "n_cpgs_complete_case": len(ids),
           "n_cpgs_variable": int(subsets["variable_cpgs"].sum()), "groups": prof.groups, "folds": folds.tolist(),
           "n_zero_embedding_rows_in_probe_set": int((~np.any(X != 0, axis=1)).sum()),
           "n_per_fold": {int(f): int((fold == f).sum()) for f in folds}, "targets": {}}
    reps = {}
    base = {k: pr.train_mean_baseline(v, fold) for k, v in targets.items()}
    for tname, Y in targets.items():
        pred = fit[tname]["oof"]
        out["targets"][tname] = {"alpha_per_fold_median_over_targets": np.median(fit[tname]["alpha"], axis=1).tolist(), "subsets": {}}
        for sname, sm in subsets.items():
            for kind, PP in (("model", pred), ("train_mean_baseline", base[tname])):
                S = pr.SuffStats(Y[sm], PP[sm], blk[sm], nb)
                point = S.metrics(np.ones(nb))
                cell = {k: v for k, v in point.items() if k != "r2_per_group"}
                cell["r2_per_group"] = point["r2_per_group"].tolist()
                if kind == "model":
                    rep = pr.bootstrap_metrics(S, ctx.boot.mult)
                    for k, v in rep.items():
                        reps[f"{tname}__{sname}__{k}"] = v
                        if v.ndim == 1:
                            lo, hi = percentile_ci(v)
                            cell[f"{k}_ci"] = [lo, hi]
                        else:
                            cell[f"{k}_ci"] = [[float(a), float(b)] for a, b in zip(*np.nanquantile(v, [0.025, 0.975], axis=0), strict=True)]
                    pf, mat = pr.fold_metrics(Y[sm], PP[sm], fold[sm])
                    cell["per_fold"] = pf
                    cell["per_fold_r2_per_group"] = mat.tolist()
                else:
                    cell = {k: cell[k] for k in ("r2_global", "r2_macro", "pearson_pooled", "r2_per_group")}
                out["targets"][tname]["subsets"].setdefault(sname, {})[kind] = cell
    return out, reps
