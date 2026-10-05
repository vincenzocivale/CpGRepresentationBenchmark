"""Registered probe: L2 logistic regression, nested chromosome-blocked C selection, identical for every arm.

Thin wrapper over bioval_v2_launch.chrom_probes.logistic_blocked_cv (imported, never edited): outer folds = frozen
chromosome->fold map; per outer fold the StandardScaler is fitted on the outer-train rows only; C is chosen by leave-one-train-fold-out
inner CV (mean inner AUROC, ties -> smaller C) on the outer-train rows only; refit on the outer-train rows; OOF decision scores.
"""
from __future__ import annotations

import numpy as np

from cpg_repr_benchmark.bioval_v2_launch import chrom_probes as cp

from .design import C_GRID, MAX_ITER


def pair_rows(pairs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """rows layout [positives; backgrounds] -> (cpg_idx, y, fold, chrom)."""
    cpg = np.r_[pairs["pos_cpg_idx"].to_numpy(), pairs["neg_cpg_idx"].to_numpy()].astype(np.int64)
    y = np.r_[np.ones(len(pairs), int), np.zeros(len(pairs), int)]
    fold = np.r_[pairs["fold"].to_numpy(), pairs["fold"].to_numpy()].astype(int)
    chrom = np.r_[pairs["chrom"].to_numpy(), pairs["chrom"].to_numpy()].astype(str)
    return cpg, y, fold, chrom


def fit_oof(X, y, fold, Cs=C_GRID, max_iter=MAX_ITER) -> dict:
    res = cp.logistic_blocked_cv(X, y, fold, Cs=tuple(Cs), max_iter=max_iter)
    if not np.isfinite(res["oof"]).all():
        raise RuntimeError("non-finite OOF scores")
    return res
