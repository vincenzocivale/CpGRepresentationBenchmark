"""Parts 8-9: out-of-fold predicted-profile pair similarity and probe-fairness (extended alpha grid).

The ridge itself is `bioval_v2_launch.chrom_probes.ridge_blocked_cv` (imported unchanged; frozen chromosome->fold map, seed 17).
Here: leakage guards, OOF-profile pair similarity, same-fold restriction, alpha-edge rule, independent refit check.
"""
from __future__ import annotations

import numpy as np

from . import registry as rg


# ------------------------------------------------------------------------------------------------ leakage guards
def assert_train_excludes_test(chrom, fold, c2f):
    """For every fold f: the chromosomes of the training rows (fold != f) never include a chromosome of fold f. Also rows' fold ==
    frozen map fold of their chromosome."""
    chrom = np.asarray(chrom).astype(str)
    fold = np.asarray(fold)
    for c in np.unique(chrom):
        if (fold[chrom == c] != c2f[c]).any():
            raise AssertionError(f"row fold differs from the frozen chromosome->fold map for {c}")
    for f in np.unique(fold):
        test_chroms = set(chrom[fold == f])
        train_chroms = set(chrom[fold != f])
        if test_chroms & train_chroms:
            raise AssertionError(f"chromosome leakage between train and test for fold {f}: {sorted(test_chroms & train_chroms)}")
    return True


def refit_check(X, Y, fold, oof, alpha, fold_id: int, target_cols, n_rows: int = 200, seed: int = rg.SEED) -> dict:
    """Independent re-derivation of out-of-fold predictions of ``fold_id`` for a few target columns with the CHOSEN alpha: the model is
    trained ONLY on rows whose fold != fold_id (standardise on those rows, centred ridge, sklearn convention SSE + alpha |w|^2).
    Returns the max |difference| with the stored OOF predictions on a deterministic sample of held-out rows."""
    X = np.asarray(X)
    tr = np.flatnonzero(fold != fold_id)
    te = np.flatnonzero(fold == fold_id)
    rng = np.random.default_rng([seed, fold_id])
    te = np.sort(rng.choice(te, size=min(n_rows, len(te)), replace=False))
    Xt = X[tr].astype(np.float64)
    mu = Xt.mean(0)
    sd = Xt.std(0)
    sd[sd == 0] = 1.0
    Xs = (Xt - mu) / sd
    xm = Xs.mean(0)
    Xc = Xs - xm
    A = Xc.T @ Xc
    out = {}
    for t in target_cols:
        y = np.asarray(Y[tr, t], np.float64)
        ym = y.mean()
        a = float(alpha[t])
        w = np.linalg.solve(A + a * np.eye(A.shape[0]), Xc.T @ (y - ym))
        pred = ((X[te].astype(np.float64) - mu) / sd - xm) @ w + ym
        out[int(t)] = float(np.abs(pred - oof[te, t]).max())
    return {"fold": int(fold_id), "max_abs_diff_per_target": out, "max_abs_diff": max(out.values()), "n_rows": len(te)}


# ------------------------------------------------------------------------------------------------ OOF pair similarity
def oof_pair_pearson(oof, ia, ib, chunk: int = 100_000):
    """Pearson across the outputs (39 groups) of the predicted profiles of rows ia vs ib. NaN if a predicted profile is constant."""
    out = np.empty(len(ia))
    for s in range(0, len(ia), chunk):
        a = oof[ia[s:s + chunk]].astype(np.float64)
        b = oof[ib[s:s + chunk]].astype(np.float64)
        a -= a.mean(1, keepdims=True)
        b -= b.mean(1, keepdims=True)
        saa, sbb, sab = (a * a).sum(1), (b * b).sum(1), (a * b).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = sab / np.sqrt(saa * sbb)
        out[s:s + chunk] = np.where((saa > 1e-24) & (sbb > 1e-24), r, np.nan)
    return out


def same_fold_mask(fold_i, fold_j):
    return np.asarray(fold_i) == np.asarray(fold_j)


# ------------------------------------------------------------------------------------------------ alpha grid / edge rule
def alpha_edge_report(alpha_chosen, grid) -> dict:
    """alpha_chosen: (n_folds, n_targets) chosen alphas. Fractions of selections at the grid maximum / minimum."""
    a = np.asarray(alpha_chosen, float)
    g = np.asarray(grid, float)
    return {"frac_at_max": float((a == g.max()).mean()), "frac_at_min": float((a == g.min()).mean()),
            "n": int(a.size), "hit_max": bool((a == g.max()).mean() >= rg.EDGE_FRACTION),
            "hit_min": bool((a == g.min()).mean() >= rg.EDGE_FRACTION)}


def next_grid(grid, reports: dict, n_extensions_done: int):
    """Registered extension rule: if ANY arm hits an edge (>= EDGE_FRACTION of its selections), extend that edge side by 2 decades
    (identical grid for all arms); at most 2 extensions. Returns (new_grid, extended: bool)."""
    if n_extensions_done >= rg.EDGE_MAX_EXTENSIONS:
        return tuple(grid), False
    hit_max = any(r["hit_max"] for r in reports.values())
    hit_min = any(r["hit_min"] for r in reports.values())
    g = sorted(grid)
    if not (hit_max or hit_min):
        return tuple(g), False
    if hit_max:
        top = g[-1]
        g = g + [top * 10.0 ** j for j in range(1, rg.EDGE_EXTEND_DECADES + 1)]
    if hit_min:
        bot = g[0]
        g = [bot / 10.0 ** j for j in range(rg.EDGE_EXTEND_DECADES, 0, -1)] + g
    return tuple(g), True
