from __future__ import annotations

import numpy as np


def _rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r per row over jointly finite entries (NaN if fewer than 3 or constant)."""
    m = np.isfinite(a) & np.isfinite(b)
    n = m.sum(axis=1)
    a0, b0 = np.where(m, a, 0.0), np.where(m, b, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = a0.sum(1) / n, b0.sum(1) / n
        da, db = np.where(m, a - ma[:, None], 0.0), np.where(m, b - mb[:, None], 0.0)
        r = (da * db).sum(1) / np.sqrt((da**2).sum(1) * (db**2).sum(1))
    r[n < 3] = np.nan
    return r


def per_sample_table(pred: np.ndarray, true: np.ndarray, prior: np.ndarray) -> dict[str, np.ndarray]:
    m = np.isfinite(true) & np.isfinite(pred)
    diff = np.where(m, pred - true, 0.0)
    pdiff = np.where(m, prior[None, :] - true, 0.0)
    n = np.maximum(m.sum(1), 1)
    return {
        "mse": (diff**2).sum(1) / n,
        "mae": np.abs(diff).sum(1) / n,
        "prior_mse": (pdiff**2).sum(1) / n,
        "pearson": _rowwise_pearson(pred, true),
        "prior_pearson": _rowwise_pearson(np.broadcast_to(prior[None, :], pred.shape), true),
        "n_targets": m.sum(1),
    }


def locus_correlation(pred: np.ndarray, true: np.ndarray, columns: np.ndarray | None = None) -> np.ndarray:
    """Across-sample Pearson r per locus (needs >= 3 samples)."""
    if columns is not None:
        pred, true = pred[:, columns], true[:, columns]
    return _rowwise_pearson(pred.T, true.T)


def summarize(
    pred: np.ndarray,
    true: np.ndarray,
    prior: np.ndarray,
    locus_std: np.ndarray | None = None,
    variable_quantile: float = 0.9,
    rows: np.ndarray | None = None,
) -> dict[str, float]:
    """Headline reconstruction metrics on a [samples x loci] block.

    prior      [loci] train-mean beta baseline for the same loci.
    locus_std  [loci] train-sample std; the top decile ("variable" loci) carries the between-sample signal
               that a per-locus mean cannot explain, so locus-wise correlation is reported there.
    """
    if rows is not None:
        pred, true = pred[rows], true[rows]
    tab = per_sample_table(pred, true, prior)
    m = np.isfinite(true) & np.isfinite(pred)
    sq = np.where(m, (pred - true) ** 2, 0.0).sum()
    psq = np.where(m, (prior[None, :] - true) ** 2, 0.0).sum()
    n = max(int(m.sum()), 1)
    out = {
        "mse": float(sq / n),
        "rmse": float(np.sqrt(sq / n)),
        "mae": float(np.where(m, np.abs(pred - true), 0.0).sum() / n),
        "prior_mse": float(psq / n),
        "skill_vs_prior": float(1.0 - sq / psq) if psq > 0 else float("nan"),
        "within_0.1": float(np.where(m, np.abs(pred - true) < 0.1, False).sum() / n),
        "sample_pearson": float(np.nanmean(tab["pearson"])),
        "prior_sample_pearson": float(np.nanmean(tab["prior_pearson"])),
        "n_samples": int(pred.shape[0]),
        "n_targets_per_sample": float(np.mean(tab["n_targets"])),
    }
    if pred.shape[0] >= 3:
        r_all = locus_correlation(pred, true)
        out["locus_pearson_all"] = float(np.nanmean(r_all))
        if locus_std is not None:
            cutoff = np.nanquantile(locus_std, variable_quantile)
            var_cols = np.flatnonzero(np.nan_to_num(locus_std, nan=-1.0) >= cutoff)
            out["locus_pearson_variable"] = float(np.nanmean(r_all[var_cols]))
            # Error on variable loci relative to the mean baseline: what the model adds beyond the prior.
            mv = m[:, var_cols]
            sqv = np.where(mv, (pred[:, var_cols] - true[:, var_cols]) ** 2, 0.0).sum()
            psqv = np.where(mv, (prior[None, var_cols] - true[:, var_cols]) ** 2, 0.0).sum()
            out["mse_variable"] = float(sqv / max(int(mv.sum()), 1))
            out["skill_vs_prior_variable"] = float(1.0 - sqv / psqv) if psqv > 0 else float("nan")
    return out


def bootstrap_ci(
    pred: np.ndarray,
    true: np.ndarray,
    prior: np.ndarray,
    locus_std: np.ndarray | None,
    variable_quantile: float = 0.9,
    n_boot: int = 200,
    seed: int = 17,
) -> dict[str, list[float]]:
    """95% percentile CI resampling *samples* (the unit of replication).

    Uses per-sample sufficient statistics (squared errors, correlations) and only the variable-locus
    block for locus-wise correlation, so a resample costs O(n * variable loci) instead of O(n * all loci).
    """
    rng = np.random.default_rng(seed)
    n = pred.shape[0]
    tab = per_sample_table(pred, true, prior)
    cnt = tab["n_targets"].astype(np.float64)
    sq, psq = tab["mse"] * cnt, tab["prior_mse"] * cnt
    var_cols = None
    if locus_std is not None and n >= 3:
        cutoff = np.nanquantile(locus_std, variable_quantile)
        var_cols = np.flatnonzero(np.nan_to_num(locus_std, nan=-1.0) >= cutoff)
        pv, tv = pred[:, var_cols], true[:, var_cols]
        mv = np.isfinite(pv) & np.isfinite(tv)
        sqv = np.where(mv, (pv - tv) ** 2, 0.0).sum(1)
        psqv = np.where(mv, (prior[None, var_cols] - tv) ** 2, 0.0).sum(1)
        cntv = mv.sum(1).astype(np.float64)
        # Sufficient statistics: a bootstrap replicate is a weight vector over samples, so weighted
        # per-locus Pearson r is six vector-matrix products instead of a fresh pass over the data.
        Mv = mv.astype(np.float64)
        Xv, Yv = np.where(mv, pv, 0.0).astype(np.float64), np.where(mv, tv, 0.0).astype(np.float64)
        stats = (Mv, Xv, Yv, Xv * Xv, Yv * Yv, Xv * Yv)

        def weighted_locus_r(w: np.ndarray) -> float:
            n_l, sx, sy, sxx, syy, sxy = (w @ a for a in stats)
            with np.errstate(invalid="ignore", divide="ignore"):
                cov = sxy - sx * sy / n_l
                r = cov / np.sqrt((sxx - sx**2 / n_l) * (syy - sy**2 / n_l))
            r[n_l < 3] = np.nan
            return float(np.nanmean(r))
    draws: dict[str, list[float]] = {k: [] for k in ("mse", "skill_vs_prior", "sample_pearson", "locus_pearson_variable", "skill_vs_prior_variable")}
    for _ in range(n_boot):
        rows = rng.integers(0, n, size=n)
        if len(np.unique(rows)) < 3:
            continue
        draws["mse"].append(sq[rows].sum() / cnt[rows].sum())
        draws["skill_vs_prior"].append(1 - sq[rows].sum() / psq[rows].sum())
        draws["sample_pearson"].append(np.nanmean(tab["pearson"][rows]))
        if var_cols is not None:
            draws["locus_pearson_variable"].append(weighted_locus_r(np.bincount(rows, minlength=n).astype(np.float64)))
            draws["skill_vs_prior_variable"].append(1 - sqv[rows].sum() / psqv[rows].sum())
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in draws.items() if v}
