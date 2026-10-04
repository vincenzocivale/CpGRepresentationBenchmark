"""Endpoint drivers for the bioval v2 launch (NEW module; consumes frozen lists read-only, never re-filters them
beyond the registered rules: fold >= 0 for Micro-C, zero-row exclusion for COSINE endpoints only).

Registered definitions: docs/BIOLOGICAL_VALIDATION_V2_LAUNCH_REGISTRATION.md (decisions D1-D4 + launch defaults).
Every statistic carries bootstrap replicates drawn from ONE shared table (block_bootstrap.BlockBootstrap), so any two
arms can be contrasted pairwise afterwards. Frozen metric functions in metrics.py are used for the point estimates
(and cross-checked against the weighted kernels at run time).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cpg_repr_benchmark.biological_validation_v2.metrics import (
    auroc_binary,
    compartment_eigenvector_corr,
    contact_vs_noncontact_auroc,
    knn_enrichment,
    spearman_profile_vs_embedding_sim,
    stratified_by_distance,
)
from cpg_repr_benchmark.biological_validation_v2.splits import chromosome_blocked_folds

from . import chrom_probes as cp
from .block_bootstrap import (
    BlockBootstrap,
    RankPlan,
    block_index,
    percentile_ci,
    weighted_auroc,
    weighted_mean,
    weighted_pearson,
    weighted_r2,
    weighted_spearman,
)
from .embedding_eval import EmbeddingStore, exclusion_mask, pair_cosine, topk_cosine_neighbors

BV = "data/derived/bioval_v2"
CHROMS = [f"chr{i}" for i in range(1, 23)]
LOYFER_POOLED_STRATA = (">1Mb", "interchromosomal")
LOYFER_ALL_STRATA = ("<1kb", "1-10kb", "10-100kb", "100kb-1Mb", ">1Mb", "interchromosomal")
KNN_K = 10
KNN_N_PERM = 999
KNN_SEED = 17
RT_LINES = ["BG02ES", "BJ", "GM06990", "GM12801", "GM12812", "GM12813", "GM12878", "HELAS3", "HEPG2", "HUVEC",
            "IMR90", "K562", "MCF7", "NHEK", "SKNSH"]
CHECK_TOL = 1e-8

FROZEN_FILES = {
    "loyfer_pairs": "external_methylation_programs/frozen_pairs_seed17.parquet",
    "loyfer_markers": "external_methylation_programs/loyfer_markers_reconciled.parquet",
    "microc_H1_intra": "3d_genome/frozen_pairs_H1_intra10kb_seed17.parquet",
    "microc_HFFc6_intra": "3d_genome/frozen_pairs_HFFc6_intra10kb_seed17.parquet",
    "microc_H1_inter": "3d_genome/frozen_pairs_H1_inter1Mb_seed17.parquet",
    "comp_H1": "3d_genome/frozen_compartments_H1_100kb.parquet",
    "comp_GM12878": "3d_genome/frozen_compartments_GM12878_100kb.parquet",
    "f5_membership": "regulatory_activity/frozen_enhancer_membership_seed17.parquet",
    "f5_win500": "regulatory_activity/frozen_enhancer_membership_win500_seed17.parquet",
    "f5_tss5k": "regulatory_activity/frozen_enhancer_membership_tss5kpool_seed17.parquet",
    "f5_activity": "regulatory_activity/frozen_activity_pairs_seed17.parquet",
    "rt": "replication_domains/cpg_rt_universe.parquet",
    "rt_frozen": "replication_domains/FROZEN_ENDPOINT.json",
    "pmd": "pmd_decato2020/cpg_pmd_summary_universe.parquet",
}


# ----------------------------------------------------------------------------- containers
@dataclass
class Stat:
    name: str
    value: float
    ci_lo: float
    ci_hi: float
    n: int
    cls: str = "primary"
    rep: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def as_dict(self):
        return {"name": self.name, "value": self.value, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi, "n": self.n,
                "class": self.cls, "meta": self.meta}


@dataclass
class EndpointResult:
    endpoint_id: str
    cls: str
    axis: str
    primary_stat: str | None
    stats: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def add(self, s: Stat):
        self.stats[s.name] = s
        return s


@dataclass
class EvalContext:
    root: Path
    store: EmbeddingStore
    zero_union: np.ndarray
    boot: BlockBootstrap
    universe: pd.DataFrame  # cpg_idx (sorted), chrom
    log: callable = print

    def path(self, key):
        from .embedding_eval import assert_readable
        p = self.root / BV / FROZEN_FILES[key]
        assert_readable(p)
        return p

    def chrom_of(self, ids):
        u = self.universe
        pos = np.searchsorted(u.cpg_idx.to_numpy(), ids)
        if (pos >= len(u)).any() or (u.cpg_idx.to_numpy()[np.minimum(pos, len(u) - 1)] != ids).any():
            raise ValueError("cpg id outside the benchmark universe")
        return u.chrom.to_numpy()[pos]

    def blk(self, chroms):
        return block_index(chroms, self.boot.names)

    def cosine_pairs(self, ci, cj):
        uniq, inv = np.unique(np.r_[ci, cj], return_inverse=True)
        E = self.store.load(uniq)
        return pair_cosine(E, inv[:len(ci)], inv[len(ci):])


def make_stat(ctx: EvalContext, name, point_fn, blk, n, *, base_w=None, cls="primary", meta=None, ci=True):
    w0 = np.ones(len(blk)) if base_w is None else np.asarray(base_w, dtype=np.float64)
    value = float(point_fn(w0))
    if ci:
        rep = ctx.boot.replicates(point_fn, blk, base_w=None if base_w is None else w0)
        lo, hi = percentile_ci(rep)
    else:
        rep, lo, hi = None, float("nan"), float("nan")
    return Stat(name, value, lo, hi, int(n), cls, rep, meta or {})


def _check(name, a, b):
    if not (np.isfinite(a) and np.isfinite(b)) or abs(a - b) > CHECK_TOL:
        raise AssertionError(f"weighted kernel disagrees with frozen metric for {name}: {a} vs {b}")


def exclusion_summary(n_total, keep, extra=None):
    d = {"n_total": int(n_total), "n_excluded_zero_locus": int((~keep).sum()), "n_kept": int(keep.sum())}
    d.update(extra or {})
    return d


def _chrom_to_fold_frozen(ctx: EvalContext) -> dict:
    fz = json.loads(ctx.path("rt_frozen").read_text())["folds"]["chrom_to_fold"]
    rec = chromosome_blocked_folds(CHROMS, 5, seed=17)
    if {k: int(v) for k, v in fz.items()} != rec:
        raise AssertionError("frozen chrom_to_fold differs from chromosome_blocked_folds(22, 5, 17)")
    return rec


# ----------------------------------------------------------------------------- 1. Loyfer profile similarity
def loyfer_profile(ctx: EvalContext) -> EndpointResult:
    df = pd.read_parquet(ctx.path("loyfer_pairs"))
    keep = exclusion_mask([df.cpg_i.to_numpy(), df.cpg_j.to_numpy()], ctx.zero_union)
    per_stratum_excl = {s: int(((df.stratum == s) & ~keep).sum()) for s in LOYFER_ALL_STRATA}
    res = EndpointResult("loyfer_profile", "primary", "external_methylation_programs",
                         "spearman_pooled_gt1Mb_inter_eqw",
                         meta={"exclusion": exclusion_summary(len(df), keep, {"per_stratum": per_stratum_excl}),
                               "similarity": "cosine(embedding)",
                               "primary_definition": "weighted Spearman, pooled strata >1Mb + interchromosomal, each "
                                                     "stratum total weight 1/2 (weights 1/n_stratum)"})
    d = df[keep].reset_index(drop=True)
    cos = ctx.cosine_pairs(d.cpg_i.to_numpy(), d.cpg_j.to_numpy())
    chrom = ctx.chrom_of(d.cpg_i.to_numpy())
    blk = ctx.blk(chrom)
    pear, spear = d.loyfer_pearson.to_numpy(), d.loyfer_spearman.to_numpy()
    strat = d.stratum.to_numpy()
    n_s = {s: int((strat == s).sum()) for s in LOYFER_ALL_STRATA}
    res.meta["n_pairs_per_stratum_kept"] = n_s
    # per-stratum Spearman (descriptive strata)
    for s in LOYFER_ALL_STRATA:
        m = strat == s
        px, py = RankPlan(pear[m]), RankPlan(cos[m])
        st = make_stat(ctx, f"spearman_{s}", lambda w, px=px, py=py: weighted_spearman(px, py, w), blk[m], m.sum(),
                       cls="descriptive")
        _check(f"loyfer {s}", st.value, spearman_profile_vs_embedding_sim(pear[m], cos[m]))
        res.add(st)
    # primary: pooled with equal stratum weight
    pooled = np.isin(strat, LOYFER_POOLED_STRATA)
    plan_cos = RankPlan(cos[pooled])

    def eqw(mask_extra=None):
        w = np.zeros(pooled.sum())
        sp = strat[pooled]
        for s in LOYFER_POOLED_STRATA:
            m = sp == s
            if mask_extra is not None:
                m = m & mask_extra[pooled]
            if m.sum():
                w[m] = 1.0 / m.sum()
        return w

    def pooled_stat(name, x, cls, mask_extra=None, kind="spearman"):
        xp = x[pooled]
        w0 = eqw(mask_extra)
        if kind == "spearman":
            px = RankPlan(xp)
            fn = lambda w, px=px: weighted_spearman(px, plan_cos, w)
        else:
            fn = lambda w, xp=xp: weighted_pearson(xp, cos[pooled], w)
        return res.add(make_stat(ctx, name, fn, blk[pooled], (w0 > 0).sum(), base_w=w0, cls=cls))

    prim = pooled_stat("spearman_pooled_gt1Mb_inter_eqw", pear, "primary")
    # sanity: weights equal to 1/n_s reproduce the unweighted kernel when only one stratum is active
    res.meta["primary_stat"] = prim.name
    pooled_stat("spearman_pooled_eqw__profile_spearman", spear, "sensitivity")
    pooled_stat("pearsoncorr_pooled_eqw__profile_pearson", pear, "sensitivity", kind="pearson")
    ge30 = (d.n_shared_groups.to_numpy() >= 30)
    pooled_stat("spearman_pooled_eqw__min30_shared_groups", pear, "sensitivity", mask_extra=ge30)
    # mean of the two stratum Spearmans (alternative reading of 'equal weight per stratum'), descriptive
    reps = [res.stats[f"spearman_{s}"] for s in LOYFER_POOLED_STRATA]
    mval = float(np.mean([r.value for r in reps]))
    mrep = np.mean([r.rep for r in reps], axis=0)
    lo, hi = percentile_ci(mrep)
    res.add(Stat("spearman_mean_of_gt1Mb_and_inter", mval, lo, hi, sum(r.n for r in reps), "descriptive", mrep))
    # descriptive: all pairs (unweighted)
    px = RankPlan(pear)
    py = RankPlan(cos)
    st = make_stat(ctx, "spearman_all_pairs", lambda w: weighted_spearman(px, py, w), blk, len(d), cls="descriptive")
    _check("loyfer all", st.value, spearman_profile_vs_embedding_sim(pear, cos))
    res.add(st)
    return res


# ----------------------------------------------------------------------------- 2. marker kNN (secondary)
def knn_permutation_pvalue(labels, nn, k, background, chroms, n_perm, seed):
    """One-sided within-chromosome label-permutation p-value of the marker kNN enrichment.

    REPLACES ``metrics.knn_enrichment(n_perm>0)`` for the launch. FROZEN-MODULE BUG (reported, not fixed there):
    in ``knn_enrichment`` the marker positions ``is_marker`` are computed once from the ORIGINAL labels and reused to
    evaluate the PERMUTED labels, so (i) with a per-label ``background`` dict the loop raises KeyError('') as soon as a
    permuted label at an original marker position is empty, and (ii) with a scalar background the null statistic is
    evaluated at the wrong loci (original marker positions, permuted labels, '' == '' counted as same-label).
    Here markers are re-identified after each permutation. The point estimate is still ``knn_enrichment`` (n_perm=0).
    """
    labels = np.asarray(labels, dtype=object)
    uniq = [u for u in pd.unique(labels) if u != ""]
    code_of = {u: i + 1 for i, u in enumerate(uniq)}
    codes = np.array([code_of.get(x, 0) for x in labels], dtype=np.int32)
    bg = np.zeros(len(uniq) + 1)
    for u, c in code_of.items():
        bg[c] = background[u]
    chroms = np.asarray(chroms)
    groups = [np.flatnonzero(chroms == c) for c in np.unique(chroms)]
    nn = np.asarray(nn)[:, :k]

    def stat(cd):
        qi = np.flatnonzero(cd > 0)
        if len(qi) == 0:
            return float("nan")
        obs = (cd[nn[qi]] == cd[qi][:, None]).mean(axis=1)
        return float(obs.mean() / bg[cd[qi]].mean())

    observed = stat(codes)
    rng = np.random.Generator(np.random.PCG64(seed))
    null = np.empty(n_perm)
    for b in range(n_perm):
        cd = codes.copy()
        for g in groups:
            cd[g] = codes[g][rng.permutation(len(g))]
        null[b] = stat(cd)
    return {"observed": observed, "null_mean": float(null.mean()), "p_value": float((1 + (null >= observed).sum()) / (1 + n_perm))}


def _marker_sets(ctx: EvalContext):
    m = pd.read_parquet(ctx.path("loyfer_markers"))
    m = m[m.in_present_group]
    out = {}
    gh = m[m.marker_source.str.startswith("github")]
    paper = m[m.marker_source.str.startswith("paper")]
    for name, src in (("github_U25_U250", gh), ("github_U25_only", gh[gh.source_table == "U25"]),
                      ("paper_lifted_U25_U250", paper)):
        src = src.assign(prio=(src.source_table != "U25").astype(int)).sort_values(["prio", "cpg_idx"])
        nlab = src.groupby("cpg_idx").cell_type.nunique()
        conflict = nlab.index[nlab > 1]
        src = src[~src.cpg_idx.isin(conflict)].drop_duplicates("cpg_idx")
        out[name] = (src.set_index("cpg_idx").cell_type.to_dict(), len(conflict))
    return out


def loyfer_marker_knn(ctx: EvalContext) -> EndpointResult:
    k = KNN_K
    uni = ctx.universe.cpg_idx.to_numpy()
    cand_mask = ~np.isin(uni, ctx.zero_union)
    cand = uni[cand_mask]
    res = EndpointResult("loyfer_marker_knn", "secondary", "external_methylation_programs", "enrichment_github_U25_U250",
                         meta={"k": k, "n_perm": KNN_N_PERM, "perm_seed": KNN_SEED, "n_candidates": len(cand),
                               "n_zero_loci_removed_from_candidates": int((~cand_mask).sum()),
                               "neighbours": "exact top-k cosine over candidate loci (self excluded)"})
    E = ctx.store.load(cand)
    nn = topk_cosine_neighbors(E, k)
    del E
    chroms = ctx.chrom_of(cand)
    for set_name, (lab_by_cpg, n_conflict) in _marker_sets(ctx).items():
        pos = np.searchsorted(cand, np.fromiter(lab_by_cpg.keys(), dtype=np.int64))
        okp = (pos < len(cand))
        okp[okp] &= cand[pos[okp]] == np.fromiter(lab_by_cpg.keys(), dtype=np.int64)[okp]
        labels = np.array([""] * len(cand), dtype=object)
        keys = np.fromiter(lab_by_cpg.keys(), dtype=np.int64)
        for p_, key in zip(pos[okp], keys[okp], strict=True):
            labels[p_] = lab_by_cpg[int(key)]
        isl = labels != ""
        if not isl.any():
            res.meta.setdefault("skipped_empty_marker_sets", []).append(set_name)
            continue
        counts = pd.Series(labels[isl]).value_counts()
        bg = {lab: float(c) / len(cand) for lab, c in counts.items()}
        fr = knn_enrichment(labels, nn, k, bg, chroms=chroms, n_perm=0)  # frozen point estimate (n_perm=0 path only)
        pp = knn_permutation_pvalue(labels, nn, k, bg, chroms, KNN_N_PERM, KNN_SEED)
        qi = np.flatnonzero(isl)
        obs = np.array([np.mean([labels[j] == labels[i] for j in nn[i]]) for i in qi])
        exp = np.array([bg[labels[i]] for i in qi])
        blk = ctx.blk(chroms[qi])
        fn = lambda w, obs=obs, exp=exp: float((w * obs).sum() / (w * exp).sum())
        st = make_stat(ctx, f"enrichment_{set_name}", fn, blk, len(qi),
                       cls="secondary" if set_name == "github_U25_U250" else "sensitivity",
                       meta={"perm_p_value_one_sided": pp["p_value"], "perm_null_mean": pp["null_mean"],
                             "perm_implementation": "launch_endpoints.knn_permutation_pvalue (frozen knn_enrichment n_perm>0 is buggy)",
                             "n_markers_in_candidates": len(qi), "n_markers_total": len(lab_by_cpg),
                             "n_cpg_dropped_multi_label": n_conflict,
                             "n_markers_dropped_zero_locus": int(len(lab_by_cpg) - okp.sum())})
        _check(f"knn {set_name}", st.value, fr["enrichment"])
        _check(f"knn perm observed {set_name}", pp["observed"], fr["enrichment"])
        res.add(st)
    return res


# ----------------------------------------------------------------------------- 3. Micro-C
def _auc_stat(ctx, name, y, score, blk, cls, meta=None):
    plan = RankPlan(score)
    yb = np.asarray(y).astype(bool)
    st = make_stat(ctx, name, lambda w: weighted_auroc(plan, yb, w), blk, len(y), cls=cls, meta=meta)
    return st


def microc(ctx: EvalContext, endpoint_id: str, key: str, cls: str, with_bins: bool, primary_stat="auroc_contact_vs_noncontact"):
    df = pd.read_parquet(ctx.path(key), columns=["pair_id", "cpg_i", "cpg_j", "chrom_i", "distance", "label", "matched_to", "fold"])
    n_all = len(df)
    n_fold_neg = int((df.fold < 0).sum())
    df = df[df.fold >= 0].reset_index(drop=True)
    keep = exclusion_mask([df.cpg_i.to_numpy(), df.cpg_j.to_numpy()], ctx.zero_union)
    res = EndpointResult(endpoint_id, cls, "3d_genome", primary_stat,
                         meta={"exclusion": exclusion_summary(len(df), keep, {
                             "n_rows_file": n_all, "n_rows_fold_negative_removed": n_fold_neg,
                             "n_label1_kept": int(((df.label == 1) & keep).sum()),
                             "n_label0_kept": int(((df.label == 0) & keep).sum())}),
                             "score": "cosine(embedding)"})
    # matched pairs (positive, control) both surviving -> delta cosine
    pid = pd.Index(df.pair_id)
    pos = np.flatnonzero(df.label.to_numpy() == 1)
    ctrl = pid.get_indexer(df.matched_to.to_numpy()[pos])
    if (ctrl < 0).any() or (df.label.to_numpy()[ctrl] != 0).any():
        raise AssertionError("matched_to does not resolve to a label-0 row of the same file")
    both = keep[pos] & keep[ctrl]
    res.meta["exclusion"]["n_matched_pairs"] = len(pos)
    res.meta["exclusion"]["n_matched_pairs_kept_for_delta_cosine"] = int(both.sum())
    d = df[keep].reset_index(drop=True)
    cos_k = ctx.cosine_pairs(d.cpg_i.to_numpy(), d.cpg_j.to_numpy())
    cos = np.full(len(df), np.nan)
    cos[np.flatnonzero(keep)] = cos_k
    y = d.label.to_numpy()
    blk = ctx.blk(d.chrom_i.to_numpy())
    st = _auc_stat(ctx, "auroc_contact_vs_noncontact", y, cos_k, blk, "primary" if cls == "primary" else cls)
    _check(endpoint_id, st.value, contact_vs_noncontact_auroc(cos_k[y == 1], cos_k[y == 0]))
    res.add(st)
    # paired delta cosine (positive minus matched negative), mean over matched pairs, block = chrom of positive cpg_i
    pi, ci_ = pos[both], ctrl[both]
    dcos = cos[pi] - cos[ci_]
    dblk = ctx.blk(df.chrom_i.to_numpy()[pi])
    res.add(make_stat(ctx, "delta_cosine_paired_mean", lambda w: weighted_mean(dcos, w), dblk, len(dcos),
                      cls="secondary" if cls == "primary" else cls))
    if with_bins:
        dist = d.distance.to_numpy()
        pt = stratified_by_distance(y, cos_k, dist)
        edges = (0, 1e3, 1e4, 1e5, 1e6, 1e7, np.inf)
        for lo, hi in itertools.pairwise(edges):
            m = (dist >= lo) & (dist < hi)
            label = f"[{lo:g},{hi:g})"
            if m.sum() == 0 or len(np.unique(y[m])) < 2:
                continue
            s = _auc_stat(ctx, f"auroc_distance_bin_{label}", y[m], cos_k[m], blk[m], "secondary")
            _check(f"{endpoint_id} bin {label}", s.value, pt[label])
            res.add(s)
    return res


def microc_H1_intra(ctx):
    return microc(ctx, "microc_H1_intra10kb", "microc_H1_intra", "primary", True)


def microc_HFFc6_intra(ctx):
    return microc(ctx, "microc_HFFc6_intra10kb", "microc_HFFc6_intra", "sensitivity", False)


def microc_H1_inter(ctx):
    return microc(ctx, "microc_H1_inter1Mb", "microc_H1_inter", "exploratory", False)


# ----------------------------------------------------------------------------- 4. compartments (E1 probe, secondary)
def compartment_probe(ctx: EvalContext, endpoint_id: str, key: str):
    d = pd.read_parquet(ctx.path(key))
    n0 = len(d)
    d = d[d.E1.notna() & (d.fold >= 0)].reset_index(drop=True)
    X = ctx.store.load(d.cpg_idx.to_numpy())
    y = d.E1.to_numpy(dtype=np.float64)
    fit = cp.ridge_blocked_cv(X, y, d.fold.to_numpy())
    oof = fit["oof"][:, 0]
    blk = ctx.blk(d.chrom.to_numpy())
    px, py = RankPlan(y), RankPlan(oof)
    res = EndpointResult(endpoint_id, "secondary", "3d_genome", "spearman_E1_vs_oof_pred",
                         meta={"n_rows_file": n0, "n_rows_used": len(d), "probe": "ridge (D3)",
                               "alpha_per_fold": fit["alpha"][:, 0].tolist(), "folds": fit["folds"].tolist(),
                               "zero_rows_kept": True})
    st = make_stat(ctx, "spearman_E1_vs_oof_pred", lambda w: weighted_spearman(px, py, w), blk, len(d), cls="secondary")
    _check(endpoint_id, st.value, compartment_eigenvector_corr(y, oof))
    res.add(st)
    res.add(make_stat(ctx, "r2_oof", lambda w: weighted_r2(y, oof, w), blk, len(d), cls="secondary"))
    fold = d.fold.to_numpy()
    res.meta["per_fold_spearman"] = {int(f): float(compartment_eigenvector_corr(y[fold == f], oof[fold == f]))
                                     for f in fit["folds"]}
    return res


def compartment_H1(ctx):
    return compartment_probe(ctx, "compartment_E1_probe_H1_100kb", "comp_H1")


def compartment_GM12878(ctx):
    return compartment_probe(ctx, "compartment_E1_probe_GM12878_100kb", "comp_GM12878")


# ----------------------------------------------------------------------------- 5. FANTOM5
def _membership(ctx, key, name, cls, res):
    d = pd.read_parquet(ctx.path(key))
    if (d.fold < 0).any():
        raise AssertionError("unexpected fold < 0 in FANTOM5 membership file")
    X = ctx.store.load(d.cpg_idx.to_numpy())  # zero rows are valid features for the probe (no exclusion)
    fit = cp.logistic_blocked_cv(X, d.label.to_numpy(), d.fold.to_numpy())
    y = d.label.to_numpy().astype(bool)
    oof = fit["oof"]
    plan = RankPlan(oof)
    blk = ctx.blk(d.chrom.to_numpy())
    st = make_stat(ctx, name, lambda w: weighted_auroc(plan, y, w), blk, len(d), cls=cls,
                   meta={"file": key, "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
                         "C_per_fold": fit["C"].tolist(), "n_iter_max": fit["n_iter_max"],
                         "n_nonconverged_fits": fit["n_nonconverged_fits"],
                         "n_zero_embedding_rows_in_probe_set": int((~np.any(X != 0, axis=1)).sum()),
                         "per_fold_auroc": {int(f): float(auroc_binary(y[d.fold.to_numpy() == f], oof[d.fold.to_numpy() == f]))
                                            for f in fit["folds"]}})
    _check(name, st.value, auroc_binary(y, oof))
    res.add(st)
    return st


def fantom5_membership(ctx):
    res = EndpointResult("fantom5_membership", "primary", "regulatory_activity", "auroc_oof_bg_enh5k",
                         meta={"probe": "logistic L2 lbfgs, C grid (D3); zero-embedding rows kept"})
    _membership(ctx, "f5_membership", "auroc_oof_bg_enh5k", "primary", res)
    _membership(ctx, "f5_win500", "auroc_oof__win500", "sensitivity", res)
    _membership(ctx, "f5_tss5k", "auroc_oof__tss5k_pool", "sensitivity", res)
    return res


def fantom5_activity(ctx):
    df = pd.read_parquet(ctx.path("f5_activity"))
    keep = exclusion_mask([df.i.to_numpy(), df.j.to_numpy()], ctx.zero_union)
    res = EndpointResult("fantom5_activity_similarity", "secondary", "regulatory_activity", "spearman_all_pairs",
                         meta={"exclusion": exclusion_summary(len(df), keep)})
    d = df[keep].reset_index(drop=True)
    cos = ctx.cosine_pairs(d.i.to_numpy(), d.j.to_numpy())
    ps = d.profile_sim.to_numpy()
    blk = ctx.blk(d.chrom_i.to_numpy())
    px, py = RankPlan(ps), RankPlan(cos)
    st = make_stat(ctx, "spearman_all_pairs", lambda w: weighted_spearman(px, py, w), blk, len(d), cls="secondary")
    _check("f5 activity", st.value, spearman_profile_vs_embedding_sim(ps, cos))
    res.add(st)
    for kind in ("intra", "inter"):
        m = (d.kind == kind).to_numpy()
        pxk, pyk = RankPlan(ps[m]), RankPlan(cos[m])
        res.add(make_stat(ctx, f"spearman_{kind}", lambda w, a=pxk, b=pyk: weighted_spearman(a, b, w), blk[m], m.sum(),
                          cls="descriptive"))
    return res


# ----------------------------------------------------------------------------- 6. replication timing + PMD
def rt_consensus(ctx):
    d = pd.read_parquet(ctx.path("rt"))
    n0 = len(d)
    d = d[d.rt_consensus_z.notna()].reset_index(drop=True)
    c2f = _chrom_to_fold_frozen(ctx)
    fold = d.chrom.map(c2f).to_numpy()
    X = ctx.store.load(d.cpg_idx.to_numpy())
    cols = ["rt_consensus_z"] + [f"rt_{x}" for x in RT_LINES]
    Y = d[cols].to_numpy(dtype=np.float64)
    if not np.isfinite(Y).all():
        raise AssertionError("NaN in RT target among eligible CpGs")
    fit = cp.ridge_blocked_cv(X, Y, fold)
    oof = fit["oof"]
    blk = ctx.blk(d.chrom.to_numpy())
    res = EndpointResult("rt_consensus", "primary", "replication_domains", "spearman_oof_consensus",
                         meta={"n_universe": n0, "n_eligible": len(d), "probe": "ridge (D3), zero rows kept",
                               "alpha_per_fold_consensus": fit["alpha"][:, 0].tolist(), "folds": fit["folds"].tolist(),
                               "n_zero_embedding_rows_in_probe_set": int((~np.any(X != 0, axis=1)).sum())})
    for t, col in enumerate(cols):
        y, p = Y[:, t], oof[:, t]
        px, py = RankPlan(y), RankPlan(p)
        cls = "primary" if t == 0 else "sensitivity"
        suffix = "consensus" if t == 0 else f"line_{col[3:]}"
        s = make_stat(ctx, f"spearman_oof_{suffix}", lambda w, px=px, py=py: weighted_spearman(px, py, w), blk, len(d), cls=cls)
        if t == 0:
            _check("rt", s.value, spearman_profile_vs_embedding_sim(y, p))
            s.meta["per_fold_spearman"] = {int(f): float(spearman_profile_vs_embedding_sim(y[fold == f], p[fold == f]))
                                           for f in fit["folds"]}
        res.add(s)
        r2 = make_stat(ctx, f"r2_oof_{suffix}", lambda w, y=y, p=p: weighted_r2(y, p, w), blk, len(d),
                       cls="secondary" if t == 0 else "sensitivity")
        if t == 0:
            r2.meta["per_fold_r2"] = {int(f): float(1 - ((y[fold == f] - p[fold == f]) ** 2).sum()
                                                    / ((y[fold == f] - y[fold == f].mean()) ** 2).sum())
                                      for f in fit["folds"]}
            r2.meta["definition"] = "1 - SSE/SST on pooled out-of-fold predictions"
        res.add(r2)
    return res


def pmd_probe(ctx):
    d = pd.read_parquet(ctx.path("pmd"))
    c2f = _chrom_to_fold_frozen(ctx)
    fold = d.chrom.map(c2f).to_numpy()
    X = ctx.store.load(d.cpg_idx.to_numpy())
    y = d.pmd_in_any.to_numpy().astype(np.float64)
    fit = cp.ridge_blocked_cv(X, y, fold)
    oof = fit["oof"][:, 0]
    yb = y.astype(bool)
    plan = RankPlan(oof)
    blk = ctx.blk(d.chrom.to_numpy())
    res = EndpointResult("pmd_probe", "exploratory", "replication_domains", "auroc_oof_pmd_in_any",
                         meta={"probe": "ridge regression on the 0/1 label (D3 grid), AUROC of out-of-fold scores",
                               "alpha_per_fold": fit["alpha"][:, 0].tolist(), "n_pos": int(yb.sum()), "n": len(d)})
    st = make_stat(ctx, "auroc_oof_pmd_in_any", lambda w: weighted_auroc(plan, yb, w), blk, len(d), cls="exploratory")
    _check("pmd", st.value, auroc_binary(yb, oof))
    res.add(st)
    return res


ENDPOINTS = {
    "loyfer_profile": ("primary", loyfer_profile),
    "microc_H1_intra10kb": ("primary", microc_H1_intra),
    "fantom5_membership": ("primary", fantom5_membership),
    "rt_consensus": ("primary", rt_consensus),
    "loyfer_marker_knn": ("secondary", loyfer_marker_knn),
    "compartment_E1_probe_H1_100kb": ("secondary", compartment_H1),
    "compartment_E1_probe_GM12878_100kb": ("secondary", compartment_GM12878),
    "fantom5_activity_similarity": ("secondary", fantom5_activity),
    "microc_HFFc6_intra10kb": ("sensitivity", microc_HFFc6_intra),
    "microc_H1_inter1Mb": ("exploratory", microc_H1_inter),
    "pmd_probe": ("exploratory", pmd_probe),
}
# Holm family: exactly these (endpoint, statistic) pairs, applied per contrast.
PRIMARY_FAMILY = {
    "loyfer_profile": "spearman_pooled_gt1Mb_inter_eqw",
    "microc_H1_intra10kb": "auroc_contact_vs_noncontact",
    "fantom5_membership": "auroc_oof_bg_enh5k",
    "rt_consensus": "spearman_oof_consensus",
}
# The H1 micro-C primary statistic name differs from the registry primary label: keep a single source of truth.
for _k in PRIMARY_FAMILY:
    assert _k in ENDPOINTS
