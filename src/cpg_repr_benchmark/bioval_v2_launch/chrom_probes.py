"""Chromosome-blocked linear probes for bioval v2 (NEW module, launch registration D3).

Identical for every arm. Outer folds = the FROZEN chromosome-blocked folds (``fold`` column / frozen chrom->fold map;
5 folds, seed 17). For each outer fold f: train = rows of the other folds, test = rows of fold f.
 * StandardScaler (mean, std ddof=0; zero std -> 1) is fitted on the outer-train rows only; it is also used (not
   refitted) inside the inner CV, which only ever sees outer-train rows.
 * Inner CV = leave-one-frozen-fold-out over the 4 outer-train folds (so inner folds are chromosome-blocked and
   consistent with the frozen fold structure). The regularisation grid is the same for all arms.
 * Ridge: alpha grid {1e-2..1e4}, sklearn convention (loss = SSE + alpha*|w|^2, intercept unpenalised), per-target
   alpha chosen by pooled inner out-of-fold SSE (ties -> larger alpha); solved from Gram matrices (exact).
 * Logistic: L2, lbfgs, class_weight=None, max_iter=5000, C grid {1e-4..10}; C chosen by mean inner-fold AUROC (ties ->
   smaller C).
Output = out-of-fold predictions for every row (test fold of each row's own fold), never fitted on its own chromosome.
"""
from __future__ import annotations

import warnings

import numpy as np

from cpg_repr_benchmark.biological_validation_v2.metrics import auroc_binary

RIDGE_ALPHAS = (1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4)
LOGREG_CS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
LOGREG_MAX_ITER = 5000


def assert_blocked(fold_train_ids, fold_test_ids, chrom_train=None, chrom_test=None):
    """No fold (and, if given, no chromosome) shared between train and test."""
    if set(np.unique(fold_train_ids)) & set(np.unique(fold_test_ids)):
        raise AssertionError("fold leakage between train and test")
    if chrom_train is not None and set(np.unique(chrom_train)) & set(np.unique(chrom_test)):
        raise AssertionError("chromosome leakage between train and test")


def _std_stats(X, rows):
    mu = np.zeros(X.shape[1])
    sq = np.zeros(X.shape[1])
    n = 0
    for s in range(0, len(rows), 100_000):
        a = X[rows[s:s + 100_000]].astype(np.float64)
        mu += a.sum(0)
        sq += (a * a).sum(0)
        n += len(a)
    mu /= n
    var = np.maximum(sq / n - mu * mu, 0.0)
    sd = np.sqrt(var)
    sd[sd == 0] = 1.0
    return mu, sd


def _scaled(X, rows, mu, sd):
    return ((X[rows].astype(np.float64) - mu) / sd)


class _Suff:
    """Sufficient statistics of (standardised X, Y) for a set of rows."""

    def __init__(self, Xs, Y):
        self.n = len(Xs)
        self.sx = Xs.sum(0)
        self.sy = Y.sum(0)
        self.xtx = Xs.T @ Xs
        self.xty = Xs.T @ Y

    def __add__(self, o):
        r = object.__new__(_Suff)
        r.n, r.sx, r.sy, r.xtx, r.xty = self.n + o.n, self.sx + o.sx, self.sy + o.sy, self.xtx + o.xtx, self.xty + o.xty
        return r


def _ridge_paths(s: _Suff, alphas):
    """Return (mu_x, mu_y, [W_alpha (d,T)]) with centred ridge solutions for all alphas via one eigh."""
    mx, my = s.sx / s.n, s.sy / s.n
    cxx = s.xtx - s.n * np.outer(mx, mx)
    cxy = s.xty - s.n * np.outer(mx, my)
    lam, Q = np.linalg.eigh(cxx)
    lam = np.maximum(lam, 0.0)
    qty = Q.T @ cxy
    return mx, my, [Q @ (qty / (lam[:, None] + a)) for a in alphas]


def ridge_blocked_cv(X, Y, fold, alphas=RIDGE_ALPHAS):
    """Chromosome(fold)-blocked ridge probe. X (n,d) float32/64, Y (n,T) (NaN-free), fold (n,) int >=0.

    Returns dict(oof=(n,T), alpha=(n_folds,T) chosen, folds=sorted fold ids)."""
    X = np.asarray(X)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    fold = np.asarray(fold)
    if (fold < 0).any():
        raise ValueError("rows with fold < 0 must be removed before probing")
    folds = np.unique(fold)
    oof = np.full(Y.shape, np.nan)
    chosen = np.zeros((len(folds), Y.shape[1]))
    alphas = tuple(alphas)
    for fi, f in enumerate(folds):
        te = np.flatnonzero(fold == f)
        tr_folds = [g for g in folds if g != f]
        assert_blocked(tr_folds, [f])
        tr_rows = {g: np.flatnonzero(fold == g) for g in tr_folds}
        all_tr = np.concatenate([tr_rows[g] for g in tr_folds])
        mu, sd = _std_stats(X, all_tr)  # scaler fitted on outer-train only
        suff, xs_cache = {}, {}
        for g in tr_folds:
            xs = _scaled(X, tr_rows[g], mu, sd)
            suff[g] = _Suff(xs, Y[tr_rows[g]])
            xs_cache[g] = xs
        sse = np.zeros((len(alphas), Y.shape[1]))
        for g in tr_folds:  # inner CV: leave one outer-train fold out
            s = None
            for h in tr_folds:
                if h != g:
                    s = suff[h] if s is None else s + suff[h]
            mx, my, Ws = _ridge_paths(s, alphas)
            for ai, W in enumerate(Ws):
                pred = (xs_cache[g] - mx) @ W + my
                sse[ai] += ((Y[tr_rows[g]] - pred) ** 2).sum(0)
        best = np.array([max(a for a, v in zip(alphas, sse[:, t], strict=True) if v == sse[:, t].min())
                         for t in range(Y.shape[1])])
        chosen[fi] = best
        s_all = None
        for g in tr_folds:
            s_all = suff[g] if s_all is None else s_all + suff[g]
        mx, my, Ws = _ridge_paths(s_all, alphas)
        xte = _scaled(X, te, mu, sd)
        for t in range(Y.shape[1]):
            W = Ws[alphas.index(best[t])]
            oof[te, t] = (xte - mx) @ W[:, t] + my[t]
        del xs_cache
    return {"oof": oof, "alpha": chosen, "folds": folds}


def logistic_blocked_cv(X, y, fold, Cs=LOGREG_CS, max_iter=LOGREG_MAX_ITER):
    """Chromosome(fold)-blocked L2 logistic probe. Returns dict(oof=decision scores (n,), C=(n_folds,), n_iter_max)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    fold = np.asarray(fold)
    if (fold < 0).any():
        raise ValueError("rows with fold < 0 must be removed before probing")
    folds = np.unique(fold)
    oof = np.full(len(y), np.nan)
    chosen = np.zeros(len(folds))
    n_iter_max = 0
    n_nonconv = 0

    def fit(Xa, ya, C):
        nonlocal n_iter_max, n_nonconv
        m = LogisticRegression(penalty="l2", solver="lbfgs", C=C, max_iter=max_iter, class_weight=None)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            m.fit(Xa, ya)
        n_nonconv += int(any(issubclass(x.category, ConvergenceWarning) for x in w))
        n_iter_max = max(n_iter_max, int(np.max(m.n_iter_)))
        return m

    for fi, f in enumerate(folds):
        te = np.flatnonzero(fold == f)
        tr_folds = [g for g in folds if g != f]
        assert_blocked(tr_folds, [f])
        tr_rows = {g: np.flatnonzero(fold == g) for g in tr_folds}
        all_tr = np.concatenate([tr_rows[g] for g in tr_folds])
        mu, sd = _std_stats(X, all_tr)
        xs = {g: _scaled(X, tr_rows[g], mu, sd) for g in tr_folds}
        score = []
        for C in Cs:
            aucs = []
            for g in tr_folds:
                xa = np.concatenate([xs[h] for h in tr_folds if h != g])
                ya = np.concatenate([y[tr_rows[h]] for h in tr_folds if h != g])
                m = fit(xa, ya, C)
                aucs.append(auroc_binary(y[tr_rows[g]], m.decision_function(xs[g])))
            score.append(np.nanmean(aucs))
        score = np.array(score)
        bestC = min(c for c, s in zip(Cs, score, strict=True) if s == np.nanmax(score))
        chosen[fi] = bestC
        m = fit(np.concatenate([xs[g] for g in tr_folds]), np.concatenate([y[tr_rows[g]] for g in tr_folds]), bestC)
        oof[te] = m.decision_function(_scaled(X, te, mu, sd))
    return {"oof": oof, "C": chosen, "folds": folds, "n_iter_max": n_iter_max, "n_nonconverged_fits": n_nonconv}
