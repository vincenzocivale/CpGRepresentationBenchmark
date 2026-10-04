"""Part B3 (EXPLORATORY POST-HOC): chromosome-blocked multi-output ridge from the CpG representation to the Loyfer profile.

Targets (complete-case CpGs = observed under mask_primary in ALL 39 groups; same rows for every arm; zero embedding rows
KEPT as in every D3 probe):
  T1 raw   : the 39 group betas;  T2 centred: beta - per-CpG mean over the 39 groups (profile shape);  T3 level: per-CpG mean (1 output).
Ridge = bioval_v2_launch.chrom_probes.ridge_blocked_cv (D3: frozen folds, scaler on outer-train, alpha grid, inner leave-one-fold-out CV,
per-target alpha) - imported unchanged, identical for all arms.
Metrics are functions of per-chromosome sufficient statistics, so the paired chromosome-block bootstrap (shared draw table) is exact and cheap.
"""
from __future__ import annotations

import numpy as np

from cpg_repr_benchmark.bioval_v2_launch import chrom_probes as cp


def assemble_targets(P_full):
    """P_full (n, 39) complete-case beta. Returns dict of target blocks."""
    ymean = P_full.mean(1)
    return {"raw": P_full, "centered": P_full - ymean[:, None], "level": ymean[:, None]}


def fit_all(X, targets: dict, fold):
    names = list(targets)
    Y = np.concatenate([targets[k] for k in names], axis=1)
    fit = cp.ridge_blocked_cv(X, Y, fold)
    out, c = {}, 0
    for k in names:
        w = targets[k].shape[1]
        out[k] = {"oof": fit["oof"][:, c:c + w], "alpha": fit["alpha"][:, c:c + w]}
        c += w
    return out, fit["folds"]


def train_mean_baseline(Y, fold):
    """Predict, for each held-out fold, the per-group mean of the other folds (the 'R2 = 0' reference)."""
    pred = np.empty_like(Y, dtype=np.float64)
    for f in np.unique(fold):
        te = fold == f
        pred[te] = Y[~te].mean(0)
    return pred


class SuffStats:
    """Per-block sums for (Y, P) of shape (n, G); block index in [0, nb)."""

    def __init__(self, Y, P, blk, nb):
        Y = np.asarray(Y, np.float64)
        P = np.asarray(P, np.float64)
        G = Y.shape[1]
        self.nb, self.G = nb, G
        z = lambda *s: np.zeros(s)
        self.n, self.Sy, self.Sp = z(nb), z(nb, G), z(nb, G)
        self.Syy, self.Spp, self.Syp, self.SSE = z(nb, G), z(nb, G), z(nb, G), z(nb, G)
        self.rsum, self.rcnt = z(nb), z(nb)
        yc, pc = Y - Y.mean(1, keepdims=True), P - P.mean(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (yc * pc).sum(1) / np.sqrt((yc * yc).sum(1) * (pc * pc).sum(1))
        valid = np.isfinite(r) & (G > 1)
        for b in range(nb):
            m = blk == b
            if not m.any():
                continue
            y, p = Y[m], P[m]
            self.n[b] = m.sum()
            self.Sy[b], self.Sp[b] = y.sum(0), p.sum(0)
            self.Syy[b], self.Spp[b], self.Syp[b] = (y * y).sum(0), (p * p).sum(0), (y * p).sum(0)
            self.SSE[b] = ((y - p) ** 2).sum(0)
            self.rsum[b] = r[m][valid[m]].sum()
            self.rcnt[b] = valid[m].sum()

    def metrics(self, w) -> dict:
        w = np.asarray(w, np.float64)
        N = (w * self.n).sum()
        if N <= 0:
            return {"r2_global": np.nan, "r2_macro": np.nan, "pearson_pooled": np.nan, "pearson_mean_over_cpgs": np.nan,
                    "r2_per_group": np.full(self.G, np.nan)}
        wc = w[:, None]
        Sy, Sp = (wc * self.Sy).sum(0), (wc * self.Sp).sum(0)
        Syy, Spp, Syp, SSE = [(wc * a).sum(0) for a in (self.Syy, self.Spp, self.Syp, self.SSE)]
        sst = Syy - Sy ** 2 / N
        with np.errstate(invalid="ignore", divide="ignore"):
            r2g = 1.0 - SSE / sst
            r2_global = 1.0 - SSE.sum() / sst.sum()
            T = N * self.G
            cov = Syp.sum() - Sy.sum() * Sp.sum() / T
            vy = Syy.sum() - Sy.sum() ** 2 / T
            vp = Spp.sum() - Sp.sum() ** 2 / T
            pear = cov / np.sqrt(vy * vp)
            rc = (w * self.rcnt).sum()
            rmean = (w * self.rsum).sum() / rc if rc > 0 else np.nan
        return {"r2_global": float(r2_global), "r2_macro": float(np.nanmean(r2g)), "pearson_pooled": float(pear),
                "pearson_mean_over_cpgs": float(rmean), "r2_per_group": r2g}


def bootstrap_metrics(S: SuffStats, mult):
    """mult (B, nb) chromosome multiplicities of the shared draw table. Returns dict name -> (B,) or (B, G) arrays."""
    res = [S.metrics(m) for m in mult]
    out = {k: np.array([r[k] for r in res]) for k in res[0]}
    return out


def fold_metrics(Y, P, fold):
    """Per-fold and per-fold x group R2 (point estimates)."""
    folds = np.unique(fold)
    S = SuffStats(Y, P, np.searchsorted(folds, fold), len(folds))
    per_fold, mat = {}, []
    for i, f in enumerate(folds):
        w = np.zeros(len(folds))
        w[i] = 1.0
        m = S.metrics(w)
        per_fold[int(f)] = {k: m[k] for k in ("r2_global", "pearson_pooled", "pearson_mean_over_cpgs")}
        mat.append(m["r2_per_group"])
    return per_fold, np.array(mat)
