"""Part 5: negative controls and simple covariate baselines on the frozen pair universe (same D1 statistic).
All functions return one similarity score per pair (higher = more similar); NaN = undefined (pair dropped by the statistic)."""
from __future__ import annotations

import numpy as np

from cpg_repr_benchmark.bioval_v2_launch.embedding_eval import pair_cosine

from . import registry as rg


def shuffled_cosine(E_universe, ia_u, ib_u, seed: int):
    """Cosine after a random permutation of the embedding rows among the universe CpGs (CpG -> embedding assignment shuffled)."""
    perm = np.random.default_rng(seed).permutation(E_universe.shape[0])
    return pair_cosine(E_universe, perm[ia_u], perm[ib_u])


def random_gaussian_cosine(n_universe: int, dim: int, seed: int, ia_u, ib_u, chunk: int = 100_000):
    """Cosine of i.i.d. N(0,1) embeddings of dimension ``dim`` assigned to every universe CpG (generated chunk-wise, deterministic)."""
    rng = np.random.default_rng(seed)
    Z = np.empty((n_universe, dim), np.float32)
    for s in range(0, n_universe, chunk):
        Z[s:s + chunk] = rng.standard_normal((min(chunk, n_universe - s), dim), dtype=np.float32)
    return pair_cosine(Z, ia_u, ib_u)


def same_chromosome(chrom_i, chrom_j):
    return (np.asarray(chrom_i) == np.asarray(chrom_j)).astype(float)


def neg_log10_distance(distance_bp):
    """-log10(distance) for intra-chromosomal pairs; NaN for interchromosomal (distance undefined)."""
    d = np.asarray(distance_bp, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -np.log10(d)
    out[~np.isfinite(d) | (d <= 0)] = np.nan
    return out


def neg_abs_diff(a_i, a_j):
    return -np.abs(np.asarray(a_i, float) - np.asarray(a_j, float))


def same_category(c_i, c_j):
    ci, cj = np.asarray(c_i).astype(str), np.asarray(c_j).astype(str)
    return (ci == cj).astype(float)


def null_summary(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"n": 0}
    return {"n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "min": float(v.min()),
            "max": float(v.max()), "q025": float(np.quantile(v, 0.025)), "q975": float(np.quantile(v, rg.NULL_HI_QUANTILE)),
            "q50": float(np.quantile(v, 0.5))}


def null_hi(shuffle_vals, random_vals) -> float:
    """NULL_HI = 97.5th percentile of the pooled primary-style statistic over the 200 null draws (shuffle + Gaussian)."""
    v = np.asarray([x for x in list(shuffle_vals) + list(random_vals) if x is not None and np.isfinite(x)], float)
    return float(np.quantile(v, rg.NULL_HI_QUANTILE)) if len(v) else float("nan")
