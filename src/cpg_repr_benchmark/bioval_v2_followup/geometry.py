"""Part B2 (EXPLORATORY POST-HOC): alternative pair-similarity geometries, all on the same pair set.

Registered definitions:
 * cosine            : frozen reference.
 * centered cosine   : cosine(e - mu); mu = mean embedding over the D2 candidate universe = all 408,399 loci minus the
                       loci that are all-zero in ANY of the 4 stores (849), identical for every arm.
 * PC1-removed cosine: z = e - mu; z' = z - (z.v1) v1 with v1 = top principal axis (eigenvector of the covariance of z over the
                       same universe, exact eigh); cosine(z'_i, z'_j).
 * Euclidean         : s = -||e_i - e_j||_2 on raw embeddings. Any monotone standardisation (z-score, exp(-d^2/median d^2))
                       leaves Spearman unchanged, so the single registered definition is -d.
"""
from __future__ import annotations

import numpy as np

from cpg_repr_benchmark.bioval_v2_launch.embedding_eval import pair_cosine

GEOMETRIES = ("cosine", "centered_cosine", "pc1_removed_cosine", "neg_euclidean")


def universe_mean_pc1(E, chunk=100_000):
    """mu (d,) float64, v1 (d,) float64 (sign: largest |component| positive), explained-variance ratio of PC1."""
    n, d = E.shape
    mu = np.zeros(d)
    for s in range(0, n, chunk):
        mu += E[s:s + chunk].astype(np.float64).sum(0)
    mu /= n
    C = np.zeros((d, d))
    for s in range(0, n, chunk):
        Z = E[s:s + chunk].astype(np.float64) - mu
        C += Z.T @ Z
    C /= n
    lam, Q = np.linalg.eigh(C)
    v1 = Q[:, -1].copy()
    if v1[np.argmax(np.abs(v1))] < 0:
        v1 = -v1
    return mu, v1, float(lam[-1] / max(lam.sum(), 1e-300))


def _checked(x, name):
    if not np.isfinite(x).all():
        raise ValueError(f"{name}: non-finite similarity (zero-norm vector after transformation); explicit failure, no silent drop")
    return x


def centered_cosine(E, ia, ib, mu):
    return _checked(pair_cosine((np.asarray(E, np.float64) - mu).astype(np.float32), ia, ib), "centered_cosine")


def pc1_removed_cosine(E, ia, ib, mu, v1):
    Z = np.asarray(E, np.float64) - mu
    Z = Z - np.outer(Z @ v1, v1)
    return _checked(pair_cosine(Z.astype(np.float32), ia, ib), "pc1_removed_cosine")


def neg_euclidean(E, ia, ib, chunk=200_000):
    E = np.asarray(E)
    out = np.empty(len(ia))
    for s in range(0, len(ia), chunk):
        d = E[ia[s:s + chunk]].astype(np.float64) - E[ib[s:s + chunk]].astype(np.float64)
        out[s:s + chunk] = -np.sqrt((d * d).sum(1))
    return out
