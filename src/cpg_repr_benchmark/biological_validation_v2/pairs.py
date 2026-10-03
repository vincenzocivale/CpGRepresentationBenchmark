"""Deterministic CpG pair sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_BINS = (0, 1e3, 1e4, 1e5, 1e6, 1e7, np.inf)
LONG_RANGE_STRATUM = ">1Mb"  # pre-declared: bins [1e6,1e7) and [1e7,inf)


def sample_pairs(index_df, n, *, kind="intra", distance_bins=DEFAULT_BINS, seed=0, max_tries=50):
    """Sample ``n`` unordered CpG pairs (i<j by cpg_idx), no self-pairs, no duplicates.

    index_df: columns cpg_idx, chrom, pos. Inputs are sorted so output is invariant to row order.
    kind='intra': same chromosome, n split equally across distance bins (feasible bins only), column
    ``bin`` = bin index and ``dist`` = |pos_i-pos_j|. kind='inter': different chromosomes only; dist NaN,
    bin = -1. Returns DataFrame[i, j, chrom_i, chrom_j, pos_i, pos_j, dist, bin, kind].
    """
    if kind not in ("intra", "inter"):
        raise ValueError(kind)
    d = index_df[["cpg_idx", "chrom", "pos"]].sort_values(["cpg_idx"], kind="mergesort").reset_index(drop=True)
    rng = np.random.Generator(np.random.PCG64(seed))
    ids, chr_, pos = d["cpg_idx"].to_numpy(), d["chrom"].to_numpy(), d["pos"].to_numpy()
    N = len(d)
    bins = np.asarray(distance_bins, dtype=float)
    seen = set()
    out = []

    def add(a, b, bi):
        if a == b:
            return False
        if ids[a] > ids[b]:
            a, b = b, a
        key = (ids[a], ids[b])
        if key in seen:
            return False
        seen.add(key)
        dist = abs(int(pos[a]) - int(pos[b])) if chr_[a] == chr_[b] else np.nan
        out.append((ids[a], ids[b], chr_[a], chr_[b], pos[a], pos[b], dist, bi, kind))
        return True

    if kind == "inter":
        got, tries = 0, 0
        while got < n and tries < max_tries * max(n, 1):
            a = rng.integers(0, N, size=n)
            b = rng.integers(0, N, size=n)
            for x, y in zip(a, b, strict=True):
                tries += 1
                if chr_[x] != chr_[y] and add(x, y, -1):
                    got += 1
                    if got >= n:
                        break
    else:
        nb = len(bins) - 1
        per = [n // nb + (1 if k < n % nb else 0) for k in range(nb)]
        # per-chromosome position-sorted views (ties broken by cpg_idx) for window lookups
        order = np.lexsort((ids, pos, chr_))
        sc = chr_[order]
        starts = {c: int(np.searchsorted(sc, c, side="left")) for c in np.unique(sc)}
        ends = {c: int(np.searchsorted(sc, c, side="right")) for c in starts}
        spos = pos[order]
        for k in range(nb):
            got, tries = 0, 0
            lo, hi = bins[k], bins[k + 1]
            while got < per[k] and tries < max_tries * max(per[k], 1):
                tries += 1
                x = int(rng.integers(0, N))
                c = chr_[x]
                s0, e0 = starts[c], ends[c]
                seg = spos[s0:e0]
                sign = 1 if rng.random() < 0.5 else -1
                p = int(pos[x])
                if sign > 0:
                    l_i = np.searchsorted(seg, p + lo, side="left")
                    h_i = len(seg) if np.isinf(hi) else np.searchsorted(seg, p + hi, side="left")
                else:
                    h_i = np.searchsorted(seg, p - lo, side="right")
                    l_i = 0 if np.isinf(hi) else np.searchsorted(seg, p - hi, side="right")
                if h_i <= l_i:
                    continue
                y = int(order[s0 + int(rng.integers(l_i, h_i))])
                if add(x, y, k):
                    got += 1
    cols = ["i", "j", "chrom_i", "chrom_j", "pos_i", "pos_j", "dist", "bin", "kind"]
    return pd.DataFrame(out, columns=cols)
