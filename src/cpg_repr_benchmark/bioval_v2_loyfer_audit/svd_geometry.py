"""Part 7: SVD / embedding-space geometry audit. Label-free; the covariance/PCs are computed over the 407,550 D2 universe loci
(identical to B2) and never fitted on the target."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from cpg_repr_benchmark.bioval_v2_launch.embedding_eval import pair_cosine

from . import registry as rg


def universe_pca(E, chunk: int = 100_000):
    """(mu, lam (desc), V (d, d) columns = principal axes, sign: largest |component| positive). Exact eigh of the covariance."""
    n, d = E.shape
    mu = np.zeros(d)
    for s in range(0, n, chunk):
        mu += np.asarray(E[s:s + chunk], np.float64).sum(0)
    mu /= n
    C = np.zeros((d, d))
    for s in range(0, n, chunk):
        Z = np.asarray(E[s:s + chunk], np.float64) - mu
        C += Z.T @ Z
    C /= n
    lam, Q = np.linalg.eigh(C)
    lam, Q = lam[::-1].copy(), Q[:, ::-1].copy()
    sgn = np.sign(Q[np.argmax(np.abs(Q), axis=0), np.arange(d)])
    sgn[sgn == 0] = 1
    return mu, lam, Q * sgn[None, :]


def pc_scores(E, mu, V, k: int, chunk: int = 100_000):
    out = np.empty((E.shape[0], k))
    for s in range(0, E.shape[0], chunk):
        out[s:s + chunk] = (np.asarray(E[s:s + chunk], np.float64) - mu) @ V[:, :k]
    return out


def _checked(x, name):
    if not np.isfinite(x).all():
        raise ValueError(f"{name}: non-finite similarity (zero-norm vector after transformation); explicit failure")
    return x


def _cos_rows(Z, ia, ib):
    return pair_cosine(np.asarray(Z, np.float32), ia, ib)


def remove_pcs_cosine(E, ia, ib, mu, V, k: int):
    """Centre with mu, project out the top-k principal axes (V[:, :k]) in embedding space, cosine. k=0 = centred cosine."""
    Z = np.asarray(E, np.float64) - mu
    if k > 0:
        n0 = np.linalg.norm(Z, axis=1)
        Vk = V[:, :k]
        Z = Z - (Z @ Vk) @ Vk.T
        if (np.linalg.norm(Z, axis=1) <= 1e-6 * np.maximum(n0, 1e-300)).any():
            raise ValueError(f"remove_pcs_cosine(k={k}): zero-norm residual vector; explicit failure")
    return _checked(_cos_rows(Z, ia, ib), f"remove_pcs_cosine(k={k})")


def standardized_cosine(E, ia, ib, mu, lam_diag_sd):
    """Cosine of per-coordinate z-scored (centred, divided by universe coordinate sd) embeddings."""
    sd = np.where(lam_diag_sd > 0, lam_diag_sd, 1.0)
    return _checked(_cos_rows((np.asarray(E, np.float64) - mu) / sd, ia, ib), "standardized_cosine")


def whitened_cosine(E, ia, ib, mu, lam, V, eps_rel: float = rg.WHITEN_EPS_REL):
    """ZCA/PCA whitening with W = V diag((lam + eps_rel*lam_max)^-1/2) (V^T optional: cosine is rotation invariant; PCA whitening used)."""
    scale = (lam + eps_rel * lam.max()) ** -0.5
    Z = ((np.asarray(E, np.float64) - mu) @ V) * scale
    return _checked(_cos_rows(Z, ia, ib), "whitened_cosine")


def coord_sd(E, chunk: int = 100_000):
    n, d = E.shape
    s1, s2 = np.zeros(d), np.zeros(d)
    for s in range(0, n, chunk):
        a = np.asarray(E[s:s + chunk], np.float64)
        s1 += a.sum(0)
        s2 += (a * a).sum(0)
    m = s1 / n
    return np.sqrt(np.maximum(s2 / n - m * m, 0.0))


def pc_dot_contribution(S, ia, ib, k_index: int):
    """Contribution of principal axis k to the centred dot product: s_ik * s_jk."""
    return S[ia, k_index] * S[ib, k_index]


def spearman_assoc(scores, covariates: dict):
    """Spearman of every score column with every covariate (pairwise-complete). Returns {cov: [rho_k]}."""
    out = {}
    for name, v in covariates.items():
        v = np.asarray(v, float)
        ok = np.isfinite(v)
        rv = rankdata(v[ok])
        row = []
        for k in range(scores.shape[1]):
            s = scores[ok, k]
            if rv.std() == 0 or np.std(s) == 0:
                row.append(float("nan"))
                continue
            row.append(float(np.corrcoef(rankdata(s), rv)[0, 1]))
        out[name] = row
    return out


def eta_squared(scores, categories):
    """Correlation ratio (share of variance of each score column explained by a categorical covariate)."""
    cat = np.asarray(categories).astype(str)
    out = []
    for k in range(scores.shape[1]):
        x = scores[:, k]
        tot = ((x - x.mean()) ** 2).sum()
        within = 0.0
        for c in np.unique(cat):
            xc = x[cat == c]
            within += ((xc - xc.mean()) ** 2).sum()
        out.append(float(1 - within / tot) if tot > 0 else float("nan"))
    return out


def neg_euclidean_pairs(E, ia, ib, chunk: int = 200_000):
    """s = -||e_i - e_j||_2 on the raw embeddings (as B2)."""
    out = np.empty(len(ia))
    for s in range(0, len(ia), chunk):
        dd = np.asarray(E[ia[s:s + chunk]], np.float64) - np.asarray(E[ib[s:s + chunk]], np.float64)
        out[s:s + chunk] = -np.sqrt((dd * dd).sum(1))
    return out
