"""Part 1: independent audit of the CpG x 39-group target matrix. Everything is re-implemented here (no import of the frozen
loyfer_* scripts, which are freeze commands). Level A: sample counts -> group beta -> pair Pearson. Level B: raw .pat.gz -> counts.
"""
from __future__ import annotations

import gzip
import re

import numpy as np
import pandas as pd

from . import registry as rg
from .common import float16_ulp

ZRE = re.compile(r"(Z[0-9A-Z]+)$")
GROUP_HARMONISATION = {"Endothelium": "Endothel", "Ovary+Endom-Ep": "Ovary-Ep"}


# ---------------------------------------------------------------------------------------------- sample -> group mapping
def zsuffix(name: str):
    m = ZRE.search(str(name).replace(".hg38.pat.gz", "").strip())
    return m.group(1) if m else None


def verify_mapping(prov: pd.DataFrame, s1: pd.DataFrame) -> dict:
    """Independent re-join of the provenance table with the published Table S1 by the unique Z-suffix of the sample name."""
    s1 = s1.copy()
    s1["z"] = s1["Sample name"].map(zsuffix)
    if s1.z.isna().any() or s1.z.duplicated().any():
        return {"ok": False, "reason": "S1 Z-suffix not unique/complete"}
    s1i = s1.set_index("z")
    used = prov[prov.derived_group.notna()].copy()
    used["z"] = used.file_name.map(zsuffix)
    problems = {"unmatched_in_s1": [], "group_mismatch": [], "patient_mismatch": []}
    for _, r in used.iterrows():
        if r.z not in s1i.index:
            problems["unmatched_in_s1"].append(r.file_name)
            continue
        g = GROUP_HARMONISATION.get(s1i.at[r.z, "Group"], s1i.at[r.z, "Group"])
        if g != r.derived_group:
            problems["group_mismatch"].append((r.file_name, r.derived_group, g))
        if str(s1i.at[r.z, "PatientID"]) != str(r.patient_id):
            problems["patient_mismatch"].append((r.file_name, r.patient_id, s1i.at[r.z, "PatientID"]))
    extra = sorted(set(s1.z) - set(used.z))
    out = {"n_used": len(used), "n_s1": len(s1), "s1_not_in_used": extra, **{k: v for k, v in problems.items()}}
    out["ok"] = (not any(problems.values()) and not extra)
    return out


def group_donor_table(prov: pd.DataFrame) -> pd.DataFrame:
    u = prov[prov.derived_group.notna()][["file_name", "patient_id", "derived_group"]].copy()
    u.columns = ["file", "donor", "group"]
    return u.reset_index(drop=True)


def n_per_group(table: pd.DataFrame) -> pd.DataFrame:
    g = table.groupby("group").agg(n_samples=("file", "size"), n_donors=("donor", "nunique"))
    return g.sort_index()


# ---------------------------------------------------------------------------------------------- Level A kernel
def donor_betas(M, C, files, table: pd.DataFrame, min_cov: int = rg.MIN_COV):
    """Donor-level beta of every (donor, CpG): mean of the valid (cov >= min_cov) sample betas of that donor.

    M, C: (n_files, N) integer counts in the order ``files``. Returns (D (n_donor, N) float64 with 0 where invalid, V bool valid,
    donors DataFrame [group, donor] in a fixed order)."""
    pos = {f: i for i, f in enumerate(files)}
    donors = table[["group", "donor"]].drop_duplicates().sort_values(["group", "donor"]).reset_index(drop=True)
    N = M.shape[1]
    D = np.zeros((len(donors), N))
    V = np.zeros((len(donors), N), bool)
    for di, (g, d) in enumerate(zip(donors.group, donors.donor, strict=True)):
        rows = [pos[f] for f in table.file[(table.group == g) & (table.donor == d)]]
        c = C[rows].astype(np.float64)
        m = M[rows].astype(np.float64)
        ok = c >= min_cov
        k = ok.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            b = np.where(ok, m / np.maximum(c, 1), 0.0).sum(0) / np.maximum(k, 1)
        D[di] = np.where(k > 0, b, 0.0)
        V[di] = k > 0
    return D, V, donors


def group_betas(D, V, donors: pd.DataFrame):
    """Group beta = unweighted mean over donors with a valid donor beta. Returns (beta (N,G) NaN where none, nval (N,G), groups)."""
    groups = sorted(donors.group.unique())
    N = D.shape[1]
    beta = np.full((N, len(groups)), np.nan)
    nval = np.zeros((N, len(groups)), np.int32)
    for gi, g in enumerate(groups):
        ix = np.flatnonzero((donors.group == g).to_numpy())
        k = V[ix].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            beta[:, gi] = np.where(k > 0, (D[ix] * V[ix]).sum(0) / np.maximum(k, 1), np.nan)
        nval[:, gi] = k
    return beta, nval, groups


def primary_mask(nval, n_donors_per_group):
    return nval >= np.minimum(2, np.asarray(n_donors_per_group))[None, :]


def compare_beta(beta64, beta16, mask=None):
    """Compare recomputed float64 group betas to the frozen float16 betas (NaN-aware)."""
    a = np.asarray(beta64, np.float64)
    b = np.asarray(beta16, np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    one_nan = np.isnan(a) ^ np.isnan(b)
    if mask is not None:
        both_nan &= mask
        one_nan &= mask
    ok = ~np.isnan(a) & ~np.isnan(b)
    if mask is not None:
        ok &= mask
    d = np.abs(a - b)[ok]
    ulp = float16_ulp(a[ok])
    rounded = a[ok].astype(np.float16).astype(np.float64)
    mism = int((rounded != b[ok]).sum())
    return {"n_cells": int(ok.sum()), "nan_pattern_mismatch": int(one_nan.sum()),
            "max_abs_diff": float(d.max()) if d.size else 0.0,
            "max_diff_in_ulp": float((d / ulp).max()) if d.size else 0.0,
            "n_beyond_half_ulp": int((d > rg.TOL_BETA_ULP_FRAC * ulp + 1e-6).sum()),
            "n_f16_rounding_mismatch": mism, "frac_f16_rounding_mismatch": mism / max(int(ok.sum()), 1),
            **_mismatch_mechanism(a[ok], rounded, b[ok])}


def _mismatch_mechanism(a, rounded, frozen):
    """Amendment 2: relative distance of the float64 beta to the nearest float16 rounding midpoint, for the cells whose rounding differs
    from the frozen float16; a cell is 'unexplained' if that distance exceeds the float32-precision tolerance."""
    m = rounded != frozen
    if not m.any():
        return {"mismatch_midpoint_rel_max": 0.0, "mismatch_midpoint_ulp_max": 0.0, "n_mismatch_beyond_f32_tol": 0}
    d_ulp, d_rel = f16_midpoint_distance(a[m])
    return {"mismatch_midpoint_rel_max": float(d_rel.max()), "mismatch_midpoint_ulp_max": float(d_ulp.max()),
            "n_mismatch_beyond_f32_tol": int((d_rel > rg.F16_MIDPOINT_REL_TOL).sum())}


def pearson_loop(P, ia, ib, min_shared=rg.MIN_SHARED):
    """Independent per-pair reference: np.corrcoef over pairwise-complete groups (slow; used on the deterministic subset)."""
    out = np.full(len(ia), np.nan)
    ns = np.zeros(len(ia), np.int32)
    for k, (a, b) in enumerate(zip(ia, ib, strict=True)):
        x, y = P[a], P[b]
        ok = ~np.isnan(x) & ~np.isnan(y)
        ns[k] = ok.sum()
        if ns[k] < min_shared:
            continue
        xx, yy = x[ok], y[ok]
        if ((xx - xx.mean()) ** 2).sum() <= 1e-12 or ((yy - yy.mean()) ** 2).sum() <= 1e-12:
            continue
        out[k] = np.corrcoef(xx, yy)[0, 1]
    return ns, out


def subset_indices(pairs: pd.DataFrame, seed=rg.SUBSET_SEED, per_stratum=rg.SUBSET_PAIRS_PER_STRATUM) -> np.ndarray:
    """Deterministic subset: per stratum (in registry order), ``per_stratum`` rows chosen by default_rng(seed) without replacement
    from the row positions of that stratum (ascending); returned sorted."""
    rng = np.random.default_rng(seed)
    idx = []
    strat = pairs.stratum.to_numpy()
    for s in rg.STRATA:
        pos = np.flatnonzero(strat == s)
        k = min(per_stratum, len(pos))
        idx.append(np.sort(rng.choice(pos, size=k, replace=False)))
    return np.sort(np.concatenate(idx))


# ---------------------------------------------------------------------------------------------- structural checks
def structural_checks(pairs: pd.DataFrame, universe: pd.DataFrame, beta16, mask_primary, groups) -> dict:
    """Duplicates, ordering, coordinate uniqueness, beta range, missingness per CpG / per group."""
    out = {}
    out["pairs_n"] = len(pairs)
    out["pairs_cpg_i_lt_cpg_j"] = bool((pairs.cpg_i < pairs.cpg_j).all())
    out["pairs_duplicates"] = int(pairs.duplicated(["cpg_i", "cpg_j"]).sum())
    out["pairs_endpoints_in_universe"] = bool(np.isin(np.r_[pairs.cpg_i, pairs.cpg_j], universe.cpg_idx).all())
    out["universe_cpg_unique"] = bool(universe.cpg_idx.is_unique)
    out["universe_chrom_pos_unique"] = int(universe.duplicated(["chrom", "pos"]).sum()) == 0
    b = np.asarray(beta16, np.float64)
    fin = b[~np.isnan(b)]
    out["beta_min"], out["beta_max"] = float(fin.min()), float(fin.max())
    out["beta_in_0_1"] = bool(fin.min() >= 0 and fin.max() <= 1)
    out["beta_nan_outside_lenient_or_masked"] = None
    ngr = mask_primary.sum(1)
    out["cpgs_with_all_groups_primary"] = int((ngr == len(groups)).sum())
    out["cpgs_with_ge20_groups_primary"] = int((ngr >= rg.MIN_SHARED).sum())
    out["cpgs_with_lt20_groups_primary"] = int((ngr < rg.MIN_SHARED).sum())
    out["missing_primary_frac_per_group"] = {g: float(1 - mask_primary[:, i].mean()) for i, g in enumerate(groups)}
    return out


# ---------------------------------------------------------------------------------------------- fasta -> CpG index (independent)
def read_fai(path):
    d = {}
    with open(path) as fh:
        for line in fh:
            n, ln, off, lw, lb = line.split("\t")[:5]
            d[n] = (int(ln), int(off), int(lw), int(lb))
    return d


def chrom_sequence(fasta, fai: dict, chrom: str) -> np.ndarray:
    ln, off, lw, lb = fai[chrom]
    nlines = -(-ln // lw)
    raw = np.fromfile(fasta, dtype=np.uint8, count=nlines * lb, offset=off)
    if len(raw) < nlines * lb:  # last line may lack the final newline
        raw = np.concatenate([raw, np.zeros(nlines * lb - len(raw), np.uint8)])
    seq = raw.reshape(nlines, lb)[:, :lw].ravel()[:ln]
    return seq & 0xDF  # upper-case


def cg_positions(seq: np.ndarray) -> np.ndarray:
    """1-based coordinate of the C of every CG dinucleotide."""
    return (np.flatnonzero((seq[:-1] == 67) & (seq[1:] == 71)) + 1).astype(np.int64)


def genome_cpg_index(fasta, fai_path, chroms=rg.FASTA_CHROMS):
    """chrom -> (positions, offset) where the wgbstools-style global 1-based index of position p is offset + rank(p) + 1."""
    fai = read_fai(fai_path)
    out, off = {}, 0
    for c in chroms:
        p = cg_positions(chrom_sequence(fasta, fai, c))
        out[c] = (p, off)
        off += len(p)
    return out, off


def universe_global_index(universe: pd.DataFrame, gidx: dict):
    """Global 1-based CpG index per universe row (-1 if the universe position is not the C of a CG in the fasta)."""
    res = np.full(len(universe), -1, np.int64)
    chrom = universe.chrom.to_numpy().astype(str)
    pos = universe.pos.to_numpy().astype(np.int64)
    for c in np.unique(chrom):
        m = chrom == c
        p, off = gidx[c]
        j = np.searchsorted(p, pos[m])
        j = np.minimum(j, len(p) - 1)
        ok = p[j] == pos[m]
        r = np.where(ok, off + j + 1, -1)
        res[np.flatnonzero(m)] = r
    return res


# ---------------------------------------------------------------------------------------------- Level B: pat parser (numpy)
def _parse_ints(buf, start, end, maxlen=12):
    """Vectorised parse of decimal integers buf[start:end] (per line)."""
    ln = end - start
    val = np.zeros(len(start), np.int64)
    for d in range(int(min(ln.max(initial=0), maxlen))):
        m = d < ln
        dig = buf[np.where(m, start + d, 0)].astype(np.int64) - 48
        val = np.where(m, val * 10 + dig, val)
    return val


def parse_pat_chunk(buf: np.ndarray, targets: np.ndarray, cs: np.ndarray, meth: np.ndarray, cov: np.ndarray) -> int:
    """buf: complete lines of a .pat (chrom, 1-based global CpG index of the first CpG, string over {C,T,.}, count).
    targets: sorted unique 1-based global indices wanted; cs[k] = number of wanted indices <= k (len G+1). Accumulates into meth/cov
    (aligned to targets). Returns the number of lines."""
    nl = np.flatnonzero(buf == 10)
    if not len(nl):
        return 0
    tabs = np.flatnonzero(buf == 9)
    if len(tabs) != 3 * len(nl):
        raise ValueError("malformed .pat chunk: expected 3 tabs per line")
    t = tabs.reshape(-1, 3)
    t1, t2, t3 = t[:, 0], t[:, 1], t[:, 2]
    s = _parse_ints(buf, t1 + 1, t2)
    cnt = _parse_ints(buf, t3 + 1, nl)
    L = t3 - t2 - 1
    hit = (cs[np.minimum(s + L - 1, len(cs) - 1)] - cs[s - 1]) > 0
    h = np.flatnonzero(hit)
    if h.size:
        Lh = L[h]
        tot = int(Lh.sum())
        starts = np.cumsum(Lh) - Lh
        li = np.repeat(h, Lh)
        offs = np.arange(tot) - np.repeat(starts, Lh)
        k = s[li] + offs
        ch = buf[t2[li] + 1 + offs]
        pos = np.minimum(np.searchsorted(targets, k), len(targets) - 1)
        ok = targets[pos] == k
        isC = ok & (ch == 67)
        isT = ok & (ch == 84)
        w = cnt[li].astype(np.float64)
        n = len(targets)
        meth += np.bincount(pos[isC], weights=w[isC], minlength=n)
        cov += np.bincount(pos[isC | isT], weights=w[isC | isT], minlength=n)
    return len(nl)


def pat_counts(path, targets: np.ndarray, G: int, chunk_bytes: int = 1 << 26):
    """Stream one .pat.gz -> (meth, cov) float64 arrays aligned to sorted ``targets`` (1-based global indices)."""
    targets = np.asarray(targets, np.int64)
    want = np.zeros(G + 2, np.int32)
    want[targets] = 1
    cs = np.cumsum(want)  # cs[k] = #wanted <= k
    meth = np.zeros(len(targets))
    cov = np.zeros(len(targets))
    nlines = 0
    with gzip.open(path, "rb") as fh:
        carry = b""
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                if carry:
                    if not carry.endswith(b"\n"):
                        carry += b"\n"
                    nlines += parse_pat_chunk(np.frombuffer(carry, np.uint8), targets, cs, meth, cov)
                break
            chunk = carry + chunk
            k = chunk.rfind(b"\n")
            carry = chunk[k + 1:]
            nlines += parse_pat_chunk(np.frombuffer(chunk[:k + 1], np.uint8), targets, cs, meth, cov)
    return meth, cov, nlines


# ---------------------------------------------------------------------------------------------- verdict
def build_verdict(checks: dict, level_b_executed: bool = False) -> dict:
    """checks: name -> bool|None (None = informational). PASS only if every hard check is True. No override.
    Registered verdict = PASS_LEVEL_A (Level A only; raw .pat counts NOT verified)."""
    hard = {k: v for k, v in checks.items() if v is not None}
    failed = sorted(k for k, v in hard.items() if not v)
    if failed:
        verdict = "FAIL"
    else:
        verdict = rg.VERDICT_PASS_WITH_B if level_b_executed else rg.VERDICT_PASS
    return {"verdict": verdict, "failed_checks": failed, "n_hard_checks": len(hard), "level": "A+B" if level_b_executed else rg.VERDICT_LEVEL,
            "raw_level_B_executed": bool(level_b_executed), "caveat": None if level_b_executed else rg.VERDICT_CAVEAT}


# ---------------------------------------------------------------------------------------------- float16 rounding-boundary mechanism (Amendment 2)
def f16_midpoint_distance(x):
    """For float64 values x: distance (in float16 ULP of x, and relative) to the NEAREST float16 rounding midpoint
    (midpoint between round_f16(x) and its neighbour on the side of x). 0 = exactly on a tie; 0.5 = exactly representable."""
    x = np.asarray(x, np.float64)
    r = x.astype(np.float16)
    r64 = r.astype(np.float64)
    nb = np.where(x >= r64, np.nextafter(r, np.float16(np.inf)), np.nextafter(r, np.float16(-np.inf))).astype(np.float64)
    mid = (r64 + nb) / 2.0
    ulp = np.abs(nb - r64)
    d = np.abs(x - mid)
    with np.errstate(invalid="ignore", divide="ignore"):
        return d / ulp, d / np.abs(x)


def frozen_style_group_beta_float32(M, C, files, table: pd.DataFrame, min_cov: int = rg.MIN_COV):
    """Diagnostic re-implementation of the accumulation order/precision of the frozen aggregation (read from, never importing,
    scripts/bioval_v2/loyfer_aggregate.py): per-sample beta and sums in float32; donors in sorted order, samples in file-name order;
    returns float32 group betas (N, G) with NaN where no valid donor."""
    order = np.argsort(np.asarray(files))
    pos = {f: i for i, f in enumerate(files)}
    Mf, Cf = M.astype(np.float32), C.astype(np.float32)
    valid = Cf >= min_cov
    B = np.where(valid, Mf / np.maximum(Cf, 1), np.nan).astype(np.float32)
    groups = sorted(table.group.unique())
    N = M.shape[1]
    out = np.full((N, len(groups)), np.nan, np.float32)
    t = table.assign(i=table.file.map(pos)).sort_values("file")
    for gi, g in enumerate(groups):
        sub = t[t.group == g]
        ix = sub.i.to_numpy()
        pats = sub.donor.astype(str).to_numpy()
        nv = np.zeros(N, np.int32)
        s = np.zeros(N, np.float32)
        for p in sorted(set(pats)):
            jx = ix[pats == p]
            v = valid[jx]
            k = v.sum(0)
            dv = np.where(k > 0, np.where(v, B[jx], 0).sum(0) / np.maximum(k, 1), 0)
            nv += (k > 0)
            s += dv.astype(np.float32)
        out[:, gi] = np.where(nv > 0, s / np.maximum(nv, 1), np.nan)
    del order
    return out


def diagnose_f16_mismatch(beta64, beta16_frozen, beta32_recon):
    """Mechanism of the cells where round_f16(beta_f64) != frozen float16."""
    a = np.asarray(beta64, np.float64)
    b = np.asarray(beta16_frozen, np.float64)
    ok = ~np.isnan(a) & ~np.isnan(b)
    mism = ok & (a.astype(np.float16).astype(np.float64) != b)
    ci, gi = np.nonzero(mism)
    d_ulp, d_rel = f16_midpoint_distance(a[mism])
    f32_f16 = np.asarray(beta32_recon).astype(np.float16).astype(np.float64)
    explained = f32_f16[mism] == b[mism]
    all_equal = (f32_f16[ok] == b[ok])
    return {"n_cells": int(ok.sum()), "n_mismatch": int(mism.sum()),
            "mismatch_distance_to_midpoint_ulp": {"max": float(d_ulp.max()) if len(d_ulp) else 0.0, "median": float(np.median(d_ulp)) if len(d_ulp) else 0.0,
                                                  "q99": float(np.quantile(d_ulp, 0.99)) if len(d_ulp) else 0.0},
            "mismatch_distance_to_midpoint_relative": {"max": float(d_rel.max()) if len(d_rel) else 0.0},
            "float32_accumulation_reproduces_frozen_cells": int(all_equal.sum()), "float32_accumulation_fails_cells": int((~all_equal).sum()),
            "mismatch_cells_explained_by_float32_accumulation": int(explained.sum()),
            "per_group_mismatch": np.bincount(gi, minlength=a.shape[1]).tolist(), "n_distinct_cpgs": len(np.unique(ci)),
            "max_mismatches_in_one_cpg": int(np.bincount(ci).max()) if len(ci) else 0}
