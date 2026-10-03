# ruff: noqa: SIM115  (one-shot data-prep script; style-only, behaviour unchanged)
"""Cell-type beta-profile similarity of CpG pairs (Loyfer atlas). No embeddings are used or loaded.
API:  profile_similarity(cpg_a, cpg_b, min_shared=20, mask='primary') -> DataFrame
        columns: cpg_a, cpg_b, n_shared, pearson, spearman, distance_bp (NaN inter), stratum
      cpg_a/cpg_b are benchmark cpg_idx (array_cpg_map ids).
Profiles = per-cell-type beta (loyfer_celltype_beta.h5, groups = 39 atlas cell types; 38 present locally), NaN where the CpG fails the
primary coverage mask. Similarity over pairwise-complete cell types, >= min_shared required (else NaN).
CLI test:  python loyfer_profile_similarity.py test   (seed 17; 200k intra + 200k inter pairs)"""
import json
import os
import sys

os.environ.setdefault("NUMBA_NUM_THREADS", "8")
import h5py
import numba as nb
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from loyfer_common import OUT

STRATA = ["<1kb", "1-10kb", "10-100kb", "100kb-1Mb", ">1Mb", "interchromosomal"]
EDGES = [0, 1_000, 10_000, 100_000, 1_000_000]

def stratum_of(dist, inter):
    s = np.full(len(dist), STRATA[4], dtype=object)
    for i in range(4):
        s[(dist >= EDGES[i]) & (dist < (EDGES[i + 1]))] = STRATA[i]
    s[inter] = STRATA[5]
    return s

@nb.njit(parallel=True, cache=True)
def _corr(P, ia, ib, min_shared):
    n = len(ia); G = P.shape[1]
    ns = np.zeros(n, np.int32); pe = np.full(n, np.nan); sp = np.full(n, np.nan)
    for k in nb.prange(n):
        x = np.empty(G); y = np.empty(G); m = 0
        for g in range(G):
            a = P[ia[k], g]; b = P[ib[k], g]
            if not (np.isnan(a) or np.isnan(b)):
                x[m] = a; y[m] = b; m += 1
        ns[k] = m
        if m < min_shared: continue
        x = x[:m]; y = y[:m]
        pe[k] = _pear(x, y)
        sp[k] = _pear(_rank(x), _rank(y))
    return ns, pe, sp

@nb.njit
def _pear(x, y):
    mx = x.mean(); my = y.mean(); sxx = 0.0; syy = 0.0; sxy = 0.0
    for i in range(len(x)):
        a = x[i] - mx; b = y[i] - my; sxx += a * a; syy += b * b; sxy += a * b
    if sxx <= 1e-12 or syy <= 1e-12: return np.nan     # constant profile -> undefined
    return sxy / np.sqrt(sxx * syy)

@nb.njit
def _rank(x):
    o = np.argsort(x); r = np.empty(len(x)); i = 0; n = len(x)
    while i < n:
        j = i
        while j + 1 < n and x[o[j + 1]] == x[o[i]]: j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1): r[o[k]] = avg
        i = j + 1
    return r

class Profiles:
    def __init__(self, mask="primary", path=None):
        with h5py.File(path or OUT / "loyfer_celltype_beta.h5") as h:
            self.cpg = h["cpg_idx"][:]; self.chrom = h["chrom"][:].astype(str); self.pos = h["pos"][:]
            b = h["beta"][:].astype(np.float64); m = h["mask_" + mask][:]
            self.groups = [x.decode() for x in h["groups"][:]]
        self.P = np.where(m, b, np.nan)
        self.row = pd.Series(np.arange(len(self.cpg)), index=self.cpg)

def profile_similarity(cpg_a, cpg_b, min_shared=20, prof=None):
    prof = prof or Profiles()
    ia = prof.row.reindex(np.asarray(cpg_a)).to_numpy(); ib = prof.row.reindex(np.asarray(cpg_b)).to_numpy()
    ok = ~(np.isnan(ia.astype(float)) | np.isnan(ib.astype(float)))
    ia = np.where(ok, ia, 0).astype(np.int64); ib = np.where(ok, ib, 0).astype(np.int64)
    ns, pe, sp = _corr(prof.P, ia, ib, min_shared)
    inter = prof.chrom[ia] != prof.chrom[ib]
    dist = np.where(inter, np.nan, np.abs(prof.pos[ia] - prof.pos[ib]).astype(float))
    st = stratum_of(np.where(inter, 0, dist), inter)
    ns[~ok] = 0; pe[~ok] = np.nan; sp[~ok] = np.nan
    return pd.DataFrame({"cpg_a": np.asarray(cpg_a), "cpg_b": np.asarray(cpg_b), "n_shared": ns, "pearson": pe, "spearman": sp, "distance_bp": dist, "stratum": st})

def sample_pairs(prof, seed=17, n_intra=200_000, n_inter=200_000):
    """Deterministic. Intra: n_intra/5 pairs per distance stratum (a uniform, b uniform among same-chrom CpGs in the distance window,
    so short strata are populated; plain-uniform intra pairs would be >99% '>1Mb'). Inter: uniform a,b on different chromosomes."""
    rng = np.random.default_rng(seed); N = len(prof.cpg)
    order = np.lexsort((prof.pos, prof.chrom)); chrom = prof.chrom[order]; pos = prof.pos[order]
    chroms, start = np.unique(chrom, return_index=True); end = np.append(start[1:], N); cs = dict(zip(chroms, zip(start, end)))
    A, B = [], []
    per = n_intra // 5
    lo_hi = [(1, 1000), (1000, 10_000), (10_000, 100_000), (100_000, 1_000_000), (1_000_000, 10**9)]
    for lo, hi in lo_hi:
        a = rng.integers(0, N, per * 3); got_a, got_b = [], []
        for ai in a:
            s, e = cs[chrom[ai]]; p = pos[ai]
            L = np.searchsorted(pos[s:e], p - hi + 1) + s; Lm = np.searchsorted(pos[s:e], p - lo + 1) + s   # [L,Lm): dist in [lo,hi) left side
            R = np.searchsorted(pos[s:e], p + lo) + s; Rm = np.searchsorted(pos[s:e], p + hi) + s
            nl, nr = Lm - L, Rm - R
            if nl + nr == 0: continue
            k = rng.integers(0, nl + nr); j = L + k if k < nl else R + (k - nl)
            got_a.append(ai); got_b.append(j)
            if len(got_a) == per: break
        A += got_a; B += got_b
    ia = order[np.array(A)]; ib = order[np.array(B)]
    intra = (ia, ib)
    a = rng.integers(0, N, n_inter * 2); b = rng.integers(0, N, n_inter * 2)
    k = prof.chrom[a] != prof.chrom[b]; a = a[k][:n_inter]; b = b[k][:n_inter]
    return intra, (a, b)

def test():
    prof = Profiles(); (ia, ib), (ja, jb) = sample_pairs(prof)
    rows = []
    for tag, (a, b) in [("intra", (ia, ib)), ("inter", (ja, jb))]:
        d = profile_similarity(prof.cpg[a], prof.cpg[b], prof=prof); d["set"] = tag; rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d["surv"] = d.pearson.notna()
    g = d.groupby("stratum").agg(n_pairs=("surv", "size"), n_survive=("surv", "sum"), median_n_shared=("n_shared", "median"),
        pearson_mean=("pearson", "mean"), pearson_p05=("pearson", lambda x: x.quantile(.05)), pearson_p25=("pearson", lambda x: x.quantile(.25)), pearson_median=("pearson", "median"),
        pearson_p75=("pearson", lambda x: x.quantile(.75)), pearson_p95=("pearson", lambda x: x.quantile(.95)),
        spearman_median=("spearman", "median")).reindex(STRATA)
    g["frac_survive"] = g.n_survive / g.n_pairs
    g.to_csv(OUT / "loyfer_profile_similarity_test_by_stratum.tsv", sep="\t")
    # also a plain-uniform intra sample for reference
    rng = np.random.default_rng(17); a = rng.integers(0, len(prof.cpg), 200000); b = rng.integers(0, len(prof.cpg), 200000)
    k = prof.chrom[a] == prof.chrom[b]; u = profile_similarity(prof.cpg[a[k]], prof.cpg[b[k]], prof=prof)
    info = {"uniform_intra_n": int(k.sum()), "uniform_intra_survive": int(u.pearson.notna().sum()), "uniform_intra_stratum_counts": u.stratum.value_counts().to_dict()}
    json.dump(info, open(OUT / "loyfer_profile_similarity_test_uniform_intra.json", "w"), indent=1)
    d.drop(columns="surv").to_parquet(OUT / "loyfer_profile_similarity_test_pairs.parquet", index=False)
    print(g.to_string()); print(info)

if __name__ == "__main__":
    if sys.argv[1:] == ["test"]: test()
