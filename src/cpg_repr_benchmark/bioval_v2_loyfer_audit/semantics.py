"""Parts 3-4: target semantics (what high Pearson similarity means) and per-stratum pair-list audit. Descriptive, label-free."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata

from . import registry as rg

CLASSES = ("NEAR_CONSTANT", "SMALL_AMPLITUDE", "GLOBAL_LEVEL", "SPECIFIC_PROGRAM")


def per_cpg_profile_stats(P):
    """Per-CpG mean, variance (ddof=0), range, n observed groups over the non-NaN (mask_primary) entries."""
    ok = ~np.isnan(P)
    n = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(ok, P, 0).sum(1) / n
        var = (np.where(ok, P - mean[:, None], 0.0) ** 2).sum(1) / n
    hi = np.where(ok, P, -np.inf).max(1)
    lo = np.where(ok, P, np.inf).min(1)
    rng_ = np.where(n > 0, hi - lo, np.nan)
    return mean, var, rng_, n


def global_group_level(P):
    """L_g = mean beta of group g over all CpGs observed in g (39-vector)."""
    return np.nanmean(P, axis=0)


def r2_with_level(P, L):
    """Per-CpG squared Pearson between its observed profile and the global group-level vector L (NaN if < 3 observed / constant)."""
    ok = ~np.isnan(P)
    n = ok.sum(1)
    Lm = np.where(ok, L[None, :], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mp = np.where(ok, P, 0).sum(1) / n
        ml = Lm.sum(1) / n
        dp = np.where(ok, P - mp[:, None], 0.0)
        dl = np.where(ok, L[None, :] - ml[:, None], 0.0)
        r = (dp * dl).sum(1) / np.sqrt((dp * dp).sum(1) * (dl * dl).sum(1))
    r2 = r * r
    r2[(n < 3) | ~np.isfinite(r2)] = np.nan
    return r2


def pair_residual_pearson(P, ia, ib, L, min_shared=rg.MIN_SHARED, chunk=50_000):
    """Pearson of the two profiles after regressing each on L over the pairwise-complete groups."""
    out = np.full(len(ia), np.nan)
    for s in range(0, len(ia), chunk):
        a, b = P[ia[s:s + chunk]], P[ib[s:s + chunk]]
        ok = ~np.isnan(a) & ~np.isnan(b)
        m = ok.sum(1)
        Lb = np.broadcast_to(L, a.shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            ml = np.where(ok, Lb, 0).sum(1) / m
            dl = np.where(ok, Lb - ml[:, None], 0.0)
            sll = (dl * dl).sum(1)
            res = []
            for x in (a, b):
                mx = np.where(ok, x, 0).sum(1) / m
                dx = np.where(ok, x - mx[:, None], 0.0)
                slope = (dx * dl).sum(1) / sll
                res.append(dx - slope[:, None] * dl)
            ra, rb = res
            saa, sbb, sab = (ra * ra).sum(1), (rb * rb).sum(1), (ra * rb).sum(1)
            r = sab / np.sqrt(saa * sbb)
        out[s:s + chunk] = np.where((m >= min_shared) & (saa > 1e-12) & (sbb > 1e-12) & (sll > 1e-12), r, np.nan)
    return out


def classify_pairs(vmin, rmin, r2_i, r2_j, r_small, g_hi, v_edge=rg.V_EDGES[0]):
    """Hierarchical, mutually exclusive classes (first match): NEAR_CONSTANT (min variance < V_Q1 edge), SMALL_AMPLITUDE (min range <
    r_small), GLOBAL_LEVEL (both CpGs have R2 w.r.t. the global level >= g_hi), else SPECIFIC_PROGRAM. Returns int codes 0..3."""
    cls = np.full(len(vmin), 3, np.int8)
    glob = (r2_i >= g_hi) & (r2_j >= g_hi)
    cls[glob] = 2
    cls[rmin < r_small] = 1
    cls[vmin < v_edge] = 0
    return cls


def band_masks(target, bands=None):
    """top 1% (>= q99), bottom 1% (<= q01), median band (q49.5..q50.5) of the finite target values; thresholds returned."""
    t = np.asarray(target, float)
    fin = np.isfinite(t)
    q = lambda p: float(np.quantile(t[fin], p))
    lo_m, hi_m = rg.BAND_MEDIAN
    th = {"q99": q(rg.BAND_TOP), "q01": q(rg.BAND_BOTTOM), "q495": q(lo_m), "q505": q(hi_m)}
    m = {"top1pct": fin & (t >= th["q99"]), "bottom1pct": fin & (t <= th["q01"]), "median": fin & (t >= th["q495"]) & (t <= th["q505"]),
         "all": fin}
    return m, th


def select_examples(mask, n=rg.EXAMPLES_PER_BAND, seed=rg.SEMANTICS_SEED, tag=0):
    """First n of a seeded random permutation of the band members (no manual selection)."""
    ix = np.flatnonzero(mask)
    rng = np.random.default_rng([seed, tag])
    return np.sort(rng.permutation(ix)[:n])


def class_shares(cls, mask):
    n = int(mask.sum())
    return {c: (float((cls[mask] == i).mean()) if n else float("nan")) for i, c in enumerate(CLASSES)} | {"n": n}


def band_summary(values: dict, bands: dict, cls, resid, target):
    """Per band: counts, class shares, medians of indicators, fraction persisting after removing the global level."""
    out = {}
    for b, m in bands.items():
        d = {"class_shares": class_shares(cls, m)}
        for k, v in values.items():
            x = np.asarray(v, float)[m]
            x = x[np.isfinite(x)]
            d[k] = {"median": float(np.median(x)), "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75))} if len(x) else None
        t, r = target[m], resid[m]
        ok = np.isfinite(r)
        d["frac_resid_defined"] = float(ok.mean()) if len(ok) else float("nan")
        d["frac_persist_after_global_level_removal"] = (
            float(((np.sign(r[ok]) == np.sign(t[ok])) & (np.abs(r[ok]) >= rg.RESIDUAL_PERSIST * np.abs(t[ok]))).mean())
            if ok.any() else float("nan"))
        out[b] = d
    return out


def topband_dominance(top_shares: dict, all_shares: dict) -> dict:
    """Rule R2(c) input: share(near-constant U small-amplitude) in the top-1% band and its enrichment over all pairs."""
    s_top = top_shares["NEAR_CONSTANT"] + top_shares["SMALL_AMPLITUDE"]
    s_all = all_shares["NEAR_CONSTANT"] + all_shares["SMALL_AMPLITUDE"]
    enrich = s_top / s_all if s_all > 0 else float("inf")
    return {"share_top": s_top, "share_all": s_all, "enrichment": enrich,
            "dominated": bool(s_top >= rg.TOPBAND_DOMINANCE_SHARE and enrich >= rg.TOPBAND_DOMINANCE_ENRICH)}


# ------------------------------------------------------------------------------------------------ Part 4 helpers
def smd(x, y):
    """Standardised mean difference (Cohen's d, pooled sd) of finite values."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    sp = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    return float((x.mean() - y.mean()) / sp) if sp > 0 else float("nan")


def ks(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    return float(ks_2samp(x, y).statistic) if len(x) and len(y) else float("nan")


def compare_strata(values: dict, strat, ref_mask, strata=rg.STRATA):
    """values: name -> pair-level array. For every stratum: SMD and KS versus the reference set (pooled long range), and the full
    pairwise SMD / KS matrices between strata."""
    strat = np.asarray(strat)
    out = {}
    for name, v in values.items():
        v = np.asarray(v, float)
        d = {"per_stratum": {}, "vs_reference": {}, "smd_matrix": {}, "ks_matrix": {}}
        for s in strata:
            m = strat == s
            x = v[m & np.isfinite(v)]
            d["per_stratum"][s] = {"n": int(m.sum()), "mean": float(x.mean()) if len(x) else None,
                                   "sd": float(x.std(ddof=1)) if len(x) > 1 else None, "median": float(np.median(x)) if len(x) else None}
            d["vs_reference"][s] = {"smd": smd(v[m], v[ref_mask]), "ks": ks(v[m], v[ref_mask])}
        for a in strata:
            d["smd_matrix"][a] = {b: smd(v[strat == a], v[strat == b]) for b in strata}
            d["ks_matrix"][a] = {b: ks(v[strat == a], v[strat == b]) for b in strata}
        out[name] = d
    return out


def partial_spearman(x, y, Z):
    """Partial Spearman of x and y given covariates Z (n,k): Pearson of the residuals of the rank-transformed variables after OLS on
    the rank-transformed covariates (with intercept). Rows with any non-finite value are dropped."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = np.asarray(Z, float).reshape(len(x), -1)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(1)
    if ok.sum() < Z.shape[1] + 5:
        return float("nan")
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    RZ = np.column_stack([np.ones(ok.sum())] + [rankdata(Z[ok, j]) for j in range(Z.shape[1])])
    bx = np.linalg.lstsq(RZ, rx, rcond=None)[0]
    by = np.linalg.lstsq(RZ, ry, rcond=None)[0]
    ex, ey = rx - RZ @ bx, ry - RZ @ by
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def js_distance(p, q):
    p = np.asarray(p, float) / np.sum(p)
    q = np.asarray(q, float) / np.sum(q)
    m = (p + q) / 2

    def kl(a, b):
        k = a > 0
        return float((a[k] * np.log2(a[k] / b[k])).sum())
    return float(np.sqrt(max((kl(p, m) + kl(q, m)) / 2, 0.0)))


def chrom_composition(chrom_i, strat, strata=rg.STRATA, chroms=rg.CHROMS):
    chrom_i, strat = np.asarray(chrom_i).astype(str), np.asarray(strat)
    tab = {s: np.array([(chrom_i[strat == s] == c).sum() for c in chroms], float) for s in strata}
    ref = tab[">1Mb"] + tab["interchromosomal"]
    return {"counts": {s: dict(zip(chroms, v.astype(int).tolist(), strict=True)) for s, v in tab.items()},
            "js_distance_vs_reference": {s: js_distance(v, ref) for s, v in tab.items()}}


def pair_level(x_i, x_j):
    """pair summaries of a CpG-level covariate: mean and absolute difference."""
    x_i, x_j = np.asarray(x_i, float), np.asarray(x_j, float)
    return (x_i + x_j) / 2.0, np.abs(x_i - x_j)


def pairs_frame(values: dict) -> pd.DataFrame:
    return pd.DataFrame(values)
