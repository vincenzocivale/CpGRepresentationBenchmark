"""Part 6: similarity in the RAW binary regulatory feature space (before SVD). Registered metrics: binary cosine, Jaccard, and the
column-weighted cosine (identical to the binary cosine when the compressor weights are all 1: `replicate_weighting: none`)."""
from __future__ import annotations

import numpy as np
from scipy import sparse


def row_nnz(X):
    return np.asarray(X.getnnz(axis=1)).ravel()


def all_zero_rows(X):
    return row_nnz(X) == 0


def _pair_inter(X, ia, ib, w=None):
    A, B = X[ia], X[ib]
    if w is not None:
        D = sparse.diags(np.asarray(w, np.float64), format="csr")
        A, B = A @ D, B @ D
    return np.asarray(A.multiply(B).sum(1)).ravel()


def pair_binary_metrics(X, ia, ib, chunk: int = 40_000, weights=None):
    """Binary cosine, Jaccard and (optionally) weighted cosine for rows X[ia] vs X[ib] (X binary CSR). Zero vectors -> NaN."""
    X = sparse.csr_matrix(X, dtype=np.float64)
    nnz = row_nnz(X).astype(np.float64)
    wn = None
    if weights is not None:
        wn = np.asarray(X.multiply(np.asarray(weights, np.float64) ** 2).sum(1)).ravel()  # sum_k w_k^2 x_k for binary x
    n = len(ia)
    cos, jac = np.full(n, np.nan), np.full(n, np.nan)
    wcos = np.full(n, np.nan) if weights is not None else None
    for s in range(0, n, chunk):
        a, b = np.asarray(ia[s:s + chunk]), np.asarray(ib[s:s + chunk])
        inter = _pair_inter(X, a, b)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos[s:s + chunk] = inter / np.sqrt(nnz[a] * nnz[b])
            jac[s:s + chunk] = inter / (nnz[a] + nnz[b] - inter)
            if weights is not None:
                wi = _pair_inter(X, a, b, np.asarray(weights, np.float64))
                wcos[s:s + chunk] = wi / np.sqrt(wn[a] * wn[b])
    cos[(nnz[ia] == 0) | (nnz[ib] == 0)] = np.nan
    jac[(nnz[ia] == 0) | (nnz[ib] == 0)] = np.nan
    out = {"binary_cosine": cos, "jaccard": jac}
    if weights is not None:
        wcos[(nnz[ia] == 0) | (nnz[ib] == 0)] = np.nan
        out["weighted_cosine"] = wcos
    return out


def verify_projection(X_rows, projection, E_store_rows, tol: float = 1e-5):
    """Reproduce the stored 256-d embedding as X @ projection.T (weights folded into the projection); returns max |diff|."""
    emb = np.asarray(X_rows @ np.asarray(projection, np.float64).T)
    d = float(np.abs(emb - np.asarray(E_store_rows, np.float64)).max())
    return {"max_abs_diff": d, "ok": d <= tol}
