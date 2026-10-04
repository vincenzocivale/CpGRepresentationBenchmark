"""Paired chromosome-block bootstrap, weighted metric kernels, bootstrap p-value and Holm (launch registration).

NEW module (phase 2a); frozen modules are untouched. Definitions follow docs/BIOLOGICAL_VALIDATION_V2.md section 0:
22 chromosome blocks, B=1000, seed 17, the SAME resampled chromosomes for every arm/endpoint (one shared draw
table), 95% percentile CI, two-sided p = min(1, 2*min(P*(D<=0), P*(D>=0))) with the (1+k)/(1+B) correction.

Resampling a chromosome c with multiplicity m_c is implemented as an item weight m_c[block(item)]. Nonlinear
statistics (AUROC, Spearman, R2) are then evaluated with weighted kernels that reuse one pre-computed sort per
variable (exactly equal to concatenating the resampled rows; tested against scipy / metrics.auroc_binary).
The draw table is bit-identical to ``metrics.bootstrap_ci_by_block(seed=17)`` block draws (tested).
"""
from __future__ import annotations

import hashlib

import numpy as np

B_DEFAULT = 1000
SEED_DEFAULT = 17
ALPHA = 0.05


def block_names(chroms) -> np.ndarray:
    """Sorted unique block names (np.unique order, as in metrics.bootstrap_ci_by_block)."""
    return np.unique(np.asarray(chroms).astype(str))


def block_index(chroms, names) -> np.ndarray:
    names = np.asarray(names)
    idx = np.searchsorted(names, np.asarray(chroms).astype(str))
    if (idx >= len(names)).any() or (names[np.minimum(idx, len(names) - 1)] != np.asarray(chroms).astype(str)).any():
        raise ValueError("item with a chromosome outside the declared block set")
    return idx.astype(np.int16)


class BlockBootstrap:
    """Shared draw table: ``draws[b]`` = n_blocks chromosome indices drawn with replacement."""

    def __init__(self, names, n_boot: int = B_DEFAULT, seed: int = SEED_DEFAULT):
        self.names = np.asarray(names)
        self.n_boot = int(n_boot)
        self.seed = int(seed)
        rng = np.random.Generator(np.random.PCG64(self.seed))
        self.draws = rng.integers(0, len(self.names), size=(self.n_boot, len(self.names)))
        self.mult = np.stack([np.bincount(r, minlength=len(self.names)) for r in self.draws]).astype(np.float64)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(",".join(map(str, self.names)).encode())
        h.update(self.draws.astype(np.int64).tobytes())
        return h.hexdigest()

    def replicates(self, stat_fn, blk_idx, base_w=None) -> np.ndarray:
        """stat_fn(w) -> float, with w = base_w * multiplicity[block(item)]. Returns (B,) array."""
        blk_idx = np.asarray(blk_idx)
        out = np.empty(self.n_boot)
        for b in range(self.n_boot):
            w = self.mult[b][blk_idx]
            if base_w is not None:
                w = w * base_w
            out[b] = stat_fn(w)
        return out


# ----------------------------------------------------------------------------- weighted kernels
class RankPlan:
    """One sort of a variable; ``midrank(w)`` gives the weighted mid-rank fraction for any weight vector."""

    def __init__(self, x):
        x = np.asarray(x)
        self.n = len(x)
        self.order = np.argsort(x, kind="mergesort")
        xs = x[self.order]
        new = np.empty(self.n, dtype=bool)
        new[0] = True
        new[1:] = xs[1:] != xs[:-1]
        self.gstart = np.flatnonzero(new)
        self.gend = np.r_[self.gstart[1:] - 1, self.n - 1]
        self.gid = np.cumsum(new) - 1

    def group_below(self, ws_sorted):
        cum = np.cumsum(ws_sorted)
        below = np.where(self.gstart > 0, cum[np.maximum(self.gstart - 1, 0)], 0.0)
        gtot = cum[self.gend] - below
        return below, gtot

    def midrank(self, w):
        ws = w[self.order]
        below, gtot = self.group_below(ws)
        tot = below[-1] + gtot[-1]
        u = np.empty(self.n)
        u[self.order] = ((below + 0.5 * gtot) / tot)[self.gid]
        return u


def _wpearson(a, b, w):
    sw = w.sum()
    if sw <= 0:
        return float("nan")
    ma, mb = (w * a).sum() / sw, (w * b).sum() / sw
    da, db = a - ma, b - mb
    va, vb = (w * da * da).sum(), (w * db * db).sum()
    if va <= 0 or vb <= 0:
        return float("nan")
    return float((w * da * db).sum() / np.sqrt(va * vb))


def weighted_spearman(plan_x: RankPlan, plan_y: RankPlan, w) -> float:
    """Weighted Spearman = weighted Pearson of weighted mid-rank fractions. w=1 equals scipy.spearmanr."""
    return _wpearson(plan_x.midrank(w), plan_y.midrank(w), w)


def weighted_pearson(x, y, w) -> float:
    return _wpearson(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(w, dtype=np.float64))


def weighted_auroc(plan_s: RankPlan, y_bool, w) -> float:
    """Weighted AUROC with tie handling (0.5); w=1 equals metrics.auroc_binary. plan_s = RankPlan(score)."""
    y = np.asarray(y_bool).astype(np.float64)
    o = plan_s.order
    wn = (w * (1.0 - y))[o]
    wp = (w * y)[o]
    w1, w0 = wp.sum(), wn.sum()
    if w1 <= 0 or w0 <= 0:
        return float("nan")
    below, gneg = plan_s.group_below(wn)
    return float((wp * (below + 0.5 * gneg)[plan_s.gid]).sum() / (w1 * w0))


def weighted_mean(v, w) -> float:
    sw = w.sum()
    return float((w * v).sum() / sw) if sw > 0 else float("nan")


def weighted_r2(y, pred, w) -> float:
    """1 - SSE/SST on (out-of-fold) predictions, SST around the (weighted) mean of y."""
    sw = w.sum()
    if sw <= 0:
        return float("nan")
    ybar = (w * y).sum() / sw
    sst = (w * (y - ybar) ** 2).sum()
    return float(1.0 - (w * (y - pred) ** 2).sum() / sst) if sst > 0 else float("nan")


# ----------------------------------------------------------------------------- inference
def percentile_ci(rep, alpha: float = ALPHA):
    r = np.asarray(rep, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(r, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_p_two_sided(delta_rep) -> dict:
    """p = min(1, 2*min((1+#(D*<=0))/(1+B), (1+#(D*>=0))/(1+B))), B = number of finite replicates."""
    d = np.asarray(delta_rep, dtype=float)
    d = d[np.isfinite(d)]
    b = len(d)
    k_le, k_ge = int((d <= 0).sum()), int((d >= 0).sum())
    p = min(1.0, 2.0 * min((1 + k_le) / (1 + b), (1 + k_ge) / (1 + b)))
    return {"p": float(p), "k_le0": k_le, "k_ge0": k_ge, "B_eff": b}


def contrast(point_a, rep_a, point_b, rep_b, alpha: float = ALPHA) -> dict:
    """delta = A - B (registered: A = regulatory). Replicates are paired by index (same resampled chromosomes)."""
    ra, rb = np.asarray(rep_a, dtype=float), np.asarray(rep_b, dtype=float)
    if ra.shape != rb.shape:
        raise ValueError("replicate vectors must come from the same draw table")
    d = ra - rb
    lo, hi = percentile_ci(d, alpha)
    out = {"delta": float(point_a - point_b), "ci_lo": lo, "ci_hi": hi}
    out.update(bootstrap_p_two_sided(d))
    return out


def holm_adjust(pvals: dict) -> dict:
    """Holm step-down adjusted p-values (monotone, capped at 1). Input {name: raw p}."""
    names = sorted(pvals, key=lambda k: (pvals[k], k))
    m = len(names)
    adj, run = {}, 0.0
    for i, k in enumerate(names):
        run = max(run, (m - i) * pvals[k])
        adj[k] = float(min(1.0, run))
    return adj
