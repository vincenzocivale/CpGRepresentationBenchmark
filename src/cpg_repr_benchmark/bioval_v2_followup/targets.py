"""Read-only Loyfer WGBS profiles and in-memory profile-similarity re-derivation (NEW; no frozen artifact is written).

The frozen target code (scripts/bioval_v2/loyfer_profile_similarity.py) is NOT imported or run (freeze command). Its
definition is re-implemented vectorised: pairwise-complete groups, >= min_shared, Pearson, undefined if either
profile has (centred) sum of squares <= 1e-12. ``assert_reproduces_frozen`` proves the kernel equals the frozen
``n_shared_groups`` / ``loyfer_pearson`` of the frozen pair table (primary mask) before any variant is trusted.
"""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd

from . import registry as rg


class LoyferProfiles:
    def __init__(self, path):
        with h5py.File(path, "r") as h:
            self.cpg = h["cpg_idx"][:].astype(np.int64)
            self.chrom = h["chrom"][:].astype(str)
            self.beta = h["beta"][:].astype(np.float64)
            self.mask = {"primary": h["mask_primary"][:], "lenient": h["mask_lenient"][:]}
            self.groups = [g.decode() for g in h["groups"][:]]
            self.n_samples = h["n_samples_per_group"][:].astype(int)
        self.order = np.argsort(self.cpg)
        self.sorted = self.cpg[self.order]

    def rows(self, ids) -> np.ndarray:
        q = np.asarray(ids, dtype=np.int64)
        pos = np.searchsorted(self.sorted, q)
        ok = pos < len(self.sorted)
        ok[ok] &= self.sorted[pos[ok]] == q[ok]
        if not ok.all():
            raise ValueError(f"{int((~ok).sum())} CpGs absent from the Loyfer profile table")
        return self.order[pos]

    def single_sample_groups(self) -> list[int]:
        return [i for i, n in enumerate(self.n_samples) if n == 1]

    def matrix(self, mask: str = "primary", drop_groups=()) -> np.ndarray:
        """(n_cpg, n_groups) float64, NaN where the CpG fails ``mask``; dropped groups removed from the columns."""
        P = np.where(self.mask[mask], self.beta, np.nan)
        keep = [g for g in range(P.shape[1]) if g not in set(drop_groups)]
        return P[:, keep]


def pair_pearson(P, ia, ib, min_shared=rg.MIN_SHARED, chunk=50_000):
    """Pairwise-complete Pearson of rows P[ia] vs P[ib]. Returns (n_shared int32, pearson float64 with NaN)."""
    n = len(ia)
    ns = np.zeros(n, np.int32)
    out = np.full(n, np.nan)
    for s in range(0, n, chunk):
        a, b = P[ia[s:s + chunk]], P[ib[s:s + chunk]]
        ok = ~np.isnan(a) & ~np.isnan(b)
        m = ok.sum(1)
        ns[s:s + chunk] = m
        with np.errstate(invalid="ignore", divide="ignore"):
            ma = np.where(ok, a, 0).sum(1) / m
            mb = np.where(ok, b, 0).sum(1) / m
            da = np.where(ok, a - ma[:, None], 0.0)
            db = np.where(ok, b - mb[:, None], 0.0)
            sxx, syy, sxy = (da * da).sum(1), (db * db).sum(1), (da * db).sum(1)
            r = sxy / np.sqrt(sxx * syy)
        out[s:s + chunk] = np.where((m >= min_shared) & (sxx > 1e-12) & (syy > 1e-12), r, np.nan)
    return ns, out


def assert_reproduces_frozen(pairs: pd.DataFrame, prof: LoyferProfiles, tol=1e-9) -> dict:
    ia, ib = prof.rows(pairs.cpg_i.to_numpy()), prof.rows(pairs.cpg_j.to_numpy())
    ns, r = pair_pearson(prof.matrix("primary"), ia, ib)
    if not (ns == pairs.n_shared_groups.to_numpy()).all():
        raise AssertionError("n_shared_groups differs from the frozen pair table")
    fz = pairs.loyfer_pearson.to_numpy()
    if not (np.isnan(r) == np.isnan(fz)).all():
        raise AssertionError("undefined-Pearson pattern differs from the frozen pair table")
    d = float(np.nanmax(np.abs(r - fz)))
    if d > tol:
        raise AssertionError(f"recomputed primary-mask Pearson differs from frozen loyfer_pearson (max |diff| {d})")
    return {"max_abs_diff_vs_frozen_loyfer_pearson": d, "n_pairs": len(pairs)}


def variant_target(pairs: pd.DataFrame, prof: LoyferProfiles, variant: str):
    """(pearson with NaN where undefined, n_shared) for 'mask_lenient' | 'drop_single_sample_groups'."""
    ia, ib = prof.rows(pairs.cpg_i.to_numpy()), prof.rows(pairs.cpg_j.to_numpy())
    if variant == "mask_lenient":
        P = prof.matrix("lenient")
    elif variant == "drop_single_sample_groups":
        P = prof.matrix("primary", drop_groups=prof.single_sample_groups())
    else:
        raise ValueError(variant)
    ns, r = pair_pearson(P, ia, ib)
    return r, ns
