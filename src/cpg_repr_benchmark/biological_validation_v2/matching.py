"""Matched controls from non-embedding covariates only."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

_FORBIDDEN = re.compile(r"(emb|embedding|pca|latent|z_\d+)", re.IGNORECASE)


def _check_covariates(df, extra=()):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("matching functions accept only a covariates DataFrame (no embeddings/arrays)")
    bad = [c for c in df.columns if _FORBIDDEN.search(str(c))]
    if bad:
        raise ValueError(f"embedding-like covariate columns are forbidden: {bad}")


def _tie_hash(ids, seed):
    return np.array([int(hashlib.sha256(f"{i}:{seed}".encode()).hexdigest()[:12], 16) for i in ids])


def fasta_window_stats(chrom, pos, fasta_path, *, window_bp=500):
    """Per-position hg38 window statistics, window = [pos-window_bp, pos+window_bp] (1-based, inclusive,
    clipped to the contig). Returns (gc_content, cpg_hg38) arrays: GC fraction over non-N bases, and the
    number of CG dinucleotides (C at position i, G at i+1) whose C lies in the window, self included.
    One chromosome is read at a time (cumulative sums), no per-row fasta access."""
    from pyfaidx import Fasta

    chrom = np.asarray(chrom, dtype=object)
    pos = np.asarray(pos, dtype=np.int64)
    gc = np.full(len(pos), np.nan)
    cg = np.full(len(pos), np.nan)
    fa = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
    for c in pd.unique(chrom):
        idx = np.where(chrom == c)[0]
        arr = np.frombuffer(str(fa[c][:]).encode("ascii"), dtype=np.uint8)
        n = len(arr)
        is_gc = np.concatenate([[0], np.cumsum((arr == ord("C")) | (arr == ord("G")))])
        is_n = np.concatenate([[0], np.cumsum(arr == ord("N"))])
        cgm = np.zeros(n, dtype=np.int64)
        cgm[:-1] = (arr[:-1] == ord("C")) & (arr[1:] == ord("G"))
        cgc = np.concatenate([[0], np.cumsum(cgm)])
        lo = np.clip(pos[idx] - 1 - window_bp, 0, n)  # 0-based start
        hi = np.clip(pos[idx] + window_bp, 0, n)  # 0-based exclusive end
        valid = (hi - lo) - (is_n[hi] - is_n[lo])
        with np.errstate(divide="ignore", invalid="ignore"):
            gc[idx] = np.where(valid > 0, (is_gc[hi] - is_gc[lo]) / valid, np.nan)
        cg[idx] = cgc[hi] - cgc[lo]
    return gc, cg


def load_probe_type(probe_table, probe_coords, universe, *, fallback_pos_table=None):
    """Infinium design type (I/II) per universe cpg_idx.

    probe_table: DataFrame[ID, Infinium_Design_Type] (GPL21145 EPIC v1 platform table, covers 450k+EPIC);
    probe_coords: DataFrame[probe_id, chr, pos] (GRCh38; ComputAgeBench illumina_probe_grch38.parquet).
    Joined to the universe by exact (chrom, pos). A position with several probes of conflicting design type,
    or without a manifest entry, stays NaN (counted by the caller). ``fallback_pos_table`` (optional
    DataFrame[chrom, pos, Infinium_Design_Type], GRCh38, e.g. the EPIC v2 manifest) fills positions the
    primary route leaves empty. Returns Series indexed by cpg_idx."""
    from .coordinates import normalize_chrom

    t = probe_table[["ID", "Infinium_Design_Type"]].dropna().rename(columns={"ID": "probe_id"})
    pc = probe_coords.assign(chrom=probe_coords["chr"].map(normalize_chrom)).dropna(subset=["chrom"])
    m = pc.merge(t, on="probe_id", how="inner")[["chrom", "pos", "Infinium_Design_Type"]].drop_duplicates()
    nun = m.groupby(["chrom", "pos"])["Infinium_Design_Type"].nunique()
    ok = nun[nun == 1].index
    m = m.set_index(["chrom", "pos"]).loc[ok]["Infinium_Design_Type"]
    key = pd.MultiIndex.from_arrays([universe["chrom"].to_numpy(), universe["pos"].to_numpy()])
    vals = m.reindex(key)
    if fallback_pos_table is not None:
        f = fallback_pos_table[["chrom", "pos", "Infinium_Design_Type"]].dropna().drop_duplicates()
        fn = f.groupby(["chrom", "pos"])["Infinium_Design_Type"].nunique()
        f = f.set_index(["chrom", "pos"]).loc[fn[fn == 1].index]["Infinium_Design_Type"]
        vals = vals.fillna(pd.Series(f.reindex(key).to_numpy(), index=vals.index))
    vals = vals.to_numpy()
    return pd.Series(vals, index=universe["cpg_idx"].to_numpy(), name="probe_type")


def build_covariates(universe, *, context_path=None, window_bp=500, fasta_path=None, tss_bed=None,
                     tss_dist_bp=None, probe_type=None, gc_window=500):
    """Covariate table (no embeddings) for the universe. Returns (DataFrame, notes list).

    Columns: cpg_idx, chrom, pos, plus
      context      : island/shore/shelf/open_sea from the CANONICAL genomic-context table (``context_path``,
                     keyed by universe cpg_idx; see coordinates.map_genomic_context). Never the legacy table.
      cpg_density  : universe CpGs within +-window_bp (self excluded).
      gc_content / cpg_density_hg38 : hg38 window (+-gc_window) GC fraction and genome-wide CG count (fasta).
      tss_dist     : log10(1+bp) to nearest TSS, from ``tss_dist_bp`` (Series by cpg_idx, bp; e.g.
                     dist_to_gencode_tss_bp of regulatory_activity/cpg_enhancer_map.parquet, GENCODE v50) or
                     computed from ``tss_bed`` (DataFrame chrom,pos).
      probe_type   : Series indexed by cpg_idx (I/II), see load_probe_type.
    Missing sources are skipped with a note (never silently NaN).
    """
    notes = []
    cov = universe[["cpg_idx", "chrom", "pos"]].copy().reset_index(drop=True)
    if context_path is not None:
        ctx = pd.read_parquet(context_path)[["cpg_idx", "context"]]
        if ctx["cpg_idx"].duplicated().any():
            raise ValueError("context table has duplicate cpg_idx")
        cov = cov.merge(ctx, on="cpg_idx", how="left")
        n_na = int(cov["context"].isna().sum())
        if n_na == len(cov):
            notes.append("context: no cpg_idx overlap with the context table (id namespaces differ?)")
        elif n_na:
            notes.append(f"context: {n_na} universe CpGs without a label (NaN)")
    else:
        notes.append("context skipped: no context_path")
    cpos = cov["pos"].to_numpy()
    dens = np.zeros(len(cov))
    for g in cov.groupby("chrom").groups.values():
        idx = np.asarray(list(g))
        p = np.sort(cpos[idx])
        n = np.searchsorted(p, cpos[idx] + window_bp, side="right") - np.searchsorted(
            p, cpos[idx] - window_bp, side="left")
        dens[idx] = n - 1
    cov["cpg_density"] = dens
    if fasta_path is not None:
        gc, cg = fasta_window_stats(cov["chrom"].to_numpy(), cpos, fasta_path, window_bp=gc_window)
        cov["gc_content"] = gc
        cov["cpg_density_hg38"] = cg
    else:
        notes.append("gc_content / cpg_density_hg38 skipped: no fasta_path")
    if tss_dist_bp is not None:
        cov["tss_dist"] = np.log10(1 + cov["cpg_idx"].map(tss_dist_bp).to_numpy(dtype=float))
    elif tss_bed is not None:
        td = np.full(len(cov), np.nan)
        for c, g in cov.groupby("chrom").groups.items():
            idx = np.asarray(list(g))
            t = np.sort(tss_bed.loc[tss_bed["chrom"] == c, "pos"].to_numpy())
            if len(t) == 0:
                continue
            p = cpos[idx]
            i = np.searchsorted(t, p)
            d = np.minimum(np.abs(t[np.clip(i - 1, 0, len(t) - 1)] - p), np.abs(t[np.clip(i, 0, len(t) - 1)] - p))
            td[idx] = np.log10(1 + d)
        cov["tss_dist"] = td
    else:
        notes.append("tss_dist skipped: no TSS source supplied (tss_dist_bp or tss_bed)")
    if probe_type is not None:
        cov["probe_type"] = cov["cpg_idx"].map(probe_type)
    else:
        notes.append("probe_type skipped: no probe-type column available")
    _check_covariates(cov)
    return cov, notes


def _smd(a, b):
    sd = np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2)
    return float((np.nanmean(a) - np.nanmean(b)) / sd) if sd > 0 else 0.0


def match_controls(cases, pool, *, exact=("chrom",), numeric=None, n_per_case=1, seed=0, replace=False,
                   id_col="cpg_idx"):
    """Match controls from ``pool`` to ``cases`` (both covariate DataFrames).

    numeric: dict {column: caliper in raw units} (or list -> no caliper). Within each exact stratum, a
    case takes the nearest pool item on standardized numeric covariates (std from pool) within all
    calipers; ties broken by sha256(id:seed). Cases are processed in hash order (order invariant).
    Pool items that are also cases are excluded. Returns (mapping DataFrame[case, control, rank],
    balance DataFrame[covariate, smd_before, smd_after]).
    """
    _check_covariates(cases)
    _check_covariates(pool)
    exact = list(exact)
    numeric = {c: None for c in numeric} if not isinstance(numeric, dict) and numeric else (numeric or {})
    ncols = list(numeric)
    cases = cases.drop_duplicates(id_col).reset_index(drop=True)
    pool = pool[~pool[id_col].isin(cases[id_col])].drop_duplicates(id_col).reset_index(drop=True)
    sd = {c: (pool[c].std() or 1.0) for c in ncols}
    used = set()
    rows = []
    ch = _tie_hash(cases[id_col].to_numpy(), seed)
    order = np.lexsort((cases[id_col].to_numpy(), ch))
    ph = _tie_hash(pool[id_col].to_numpy(), seed)
    pool = pool.assign(_h=ph)
    groups = {(k if isinstance(k, tuple) else (k,)): g for k, g in pool.groupby(exact, sort=True)} if exact \
        else {(): pool}
    for ci in order:
        case = cases.iloc[ci]
        key = tuple(case[e] for e in exact)
        g = groups.get(key)
        if g is None:
            continue
        if not replace:
            g = g[~g[id_col].isin(used)]
        if len(g) == 0:
            continue
        ok = np.ones(len(g), dtype=bool)
        dist = np.zeros(len(g))
        for c in ncols:
            diff = np.abs(g[c].to_numpy(dtype=float) - float(case[c]))
            cal = numeric[c]
            if cal is not None:
                ok &= diff <= cal
            dist += (diff / sd[c]) ** 2
        ok &= np.isfinite(dist)
        if not ok.any():
            continue
        gg = g[ok]
        dd = dist[ok]
        sel = np.lexsort((gg[id_col].to_numpy(), gg["_h"].to_numpy(), dd))[:n_per_case]
        for r, s in enumerate(sel):
            cid = gg.iloc[s][id_col]
            rows.append({"case": case[id_col], "control": cid, "rank": r})
            if not replace:
                used.add(cid)
    mp = pd.DataFrame(rows, columns=["case", "control", "rank"])
    mp = mp.sort_values(["case", "rank"]).reset_index(drop=True)
    matched_cases = cases[cases[id_col].isin(mp["case"])]
    ctrl = mp[["control"]].merge(pool, left_on="control", right_on=id_col, how="left")
    bal = []
    for c in ncols:
        bal.append({"covariate": c, "smd_before": _smd(cases[c].to_numpy(float), pool[c].to_numpy(float)),
                    "smd_after": _smd(matched_cases[c].to_numpy(float), ctrl[c].to_numpy(float))
                    if len(mp) else float("nan")})
    return mp, pd.DataFrame(bal, columns=["covariate", "smd_before", "smd_after"])


def match_pairs(case_pairs, pool_pairs, *, distance_bins=(0, 1e3, 1e4, 1e5, 1e6, 1e7, np.inf), n_per_case=1,
                seed=0, replace=False, same_chrom=True):
    """Match case pairs to pool pairs. Columns required: i, j (ids), chrom_i, chrom_j, dist (bp, NaN for
    inter). Intra (chrom_i==chrom_j) and inter pairs are never mixed: ``same_chrom`` selects which kind of
    case pairs are matched; intra pairs match on (chrom, distance bin), inter on unordered chromosome pair.
    Returns mapping DataFrame[case_row, control_row, rank] (row labels of the given frames).
    """
    for df in (case_pairs, pool_pairs):
        _check_covariates(df)

    def split(df):
        intra = df["chrom_i"] == df["chrom_j"]
        return df[intra] if same_chrom else df[~intra]

    cp, pp = split(case_pairs).copy(), split(pool_pairs).copy()

    def key(df):
        if same_chrom:
            b = np.digitize(df["dist"].to_numpy(float), np.asarray(distance_bins)[1:-1])
            return [f"{c}|{x}" for c, x in zip(df["chrom_i"], b, strict=True)]
        return ["|".join(sorted([a, c])) for a, c in zip(df["chrom_i"], df["chrom_j"], strict=True)]

    cp["_k"], pp["_k"] = key(cp), key(pp)
    cp["_id"] = cp["i"].astype(str) + "_" + cp["j"].astype(str)
    pp["_id"] = pp["i"].astype(str) + "_" + pp["j"].astype(str)
    pp["_h"] = _tie_hash(pp["_id"].to_numpy(), seed)
    cp["_h"] = _tie_hash(cp["_id"].to_numpy(), seed)
    used = set()
    rows = []
    for lab in cp.sort_values(["_h", "_id"]).index:
        c = cp.loc[lab]
        g = pp[pp["_k"] == c["_k"]]
        g = g[g["_id"] != c["_id"]]
        if not replace:
            g = g[~g.index.isin(used)]
        g = g.sort_values(["_h", "_id"]).head(n_per_case)
        for r, gl in enumerate(g.index):
            rows.append({"case_row": lab, "control_row": gl, "rank": r})
            if not replace:
                used.add(gl)
    return pd.DataFrame(rows, columns=["case_row", "control_row", "rank"])
