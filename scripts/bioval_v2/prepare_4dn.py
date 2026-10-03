#!/usr/bin/env python
# ruff: noqa: DTZ005, EXE001, F841, SIM115  (one-shot data-prep script; style-only, behaviour unchanged)
"""3d_genome axis preparation for biological validation v2 (CPU only). Subcommands:
  fetch   <system>   range-extract single-resolution coolers from a 4DN mcool into data/external/bioval_v2/3d_genome/
  compartments <system>  cooltools eigs_cis (GC-phased) -> E1 bins + CpG table
  contacts <system>  intra (10 kb) + inter (1 Mb) contact tables, non-contact pools
  audit    <system>  overlap audit with the 408,399 universe
Pre-declared thresholds are in docs/bioval_v2_prep/3d_genome_PREP.md section 0.
Also exposes project_to_cpg_pairs() to project bin pairs to CpG pairs (never materialized in bulk).
"""
import datetime
import hashlib
import json
import os
import platform
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts/bioval_v2"))
RAW = ROOT / "data/external/bioval_v2/3d_genome"
OUT = ROOT / "data/derived/bioval_v2/3d_genome"
S3 = "https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/"
SYSTEMS = {
    "H1_MicroC": {"acc": "4DNFI9GMP2J8", "expset": "4DNES21D8SP8", "size": 13086652234, "md5": "6b9dd3f9388c06325b713597550ae349",
                      "key": "d13aa1ea-053c-4113-94fe-a1e7ab1dbbab/4DNFI9GMP2J8.mcool", "desc": "H1-hESC Micro-C merged (Tier 1)"},
    "HFFc6_MicroC": {"acc": "4DNFI9FVHJZQ", "expset": "4DNESWST3UBH", "size": 11099585140, "md5": "fc6f3601fc82b8006432fea4c8e513ee",
                         "key": "27bbfb40-0224-427c-be41-398b43d70c48/4DNFI9FVHJZQ.mcool", "desc": "HFFc6 Micro-C merged (Tier 1)"},
    "GM12878_HiC": {"acc": "4DNFIXP4QG5B", "expset": "4DNES3JX38V5", "size": 27408885254, "md5": "a4ff0d005aa78bcd544977b29f152ee6",
                        "key": "d6abea45-b0bb-4154-9854-1d3075b98097/4DNFIXP4QG5B.mcool", "desc": "GM12878 in situ Hi-C merged", "res": [50000, 100000]},
}
FETCH_RES = [10000, 50000, 100000, 1000000]
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]   # chrY/chrM excluded from contact maps (declared)
SEED = 17
P = {  # pre-declared thresholds
    "eig_res": [100000, 50000], "intra_res": 10000, "inter_res": 1000000,
    "intra_min_bins": 2, "intra_max_bins": 200, "intra_min_raw": 8, "intra_min_oe": 2.0,
    "noncontact_max_oe": 0.5, "cov_quantile": 0.25, "pool_mult": 10, "pool_min": 200,
    "inter_min_raw": 50, "inter_min_oe": 2.0, "spearman_flag": 0.2,
}

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

def cool_path(sysn, res): return RAW / f"{SYSTEMS[sysn]['acc']}_{res}.cool"

# ---------------------------------------------------------------- fetch
def fetch(sysn):
    import cooler
    import h5py
    from fourdn_rangefile import RangeMirror
    s = SYSTEMS[sysn]
    m = RangeMirror(S3 + s["key"], str(RAW / f"{s['acc']}.mirror"), s["size"], block=1 << 21, workers=16)
    h = h5py.File(m, "r")
    for res in s.get("res", FETCH_RES):
        dst = cool_path(sysn, res)
        if dst.exists(): continue
        g = h[f"resolutions/{res}"]
        tmp = str(dst) + ".tmp"
        cooler.fileops.cp(f"{RAW / (s['acc'] + '.mirror')}::resolutions/{res}", tmp, overwrite=True) if False else None
        with h5py.File(tmp, "w") as out:
            for k, v in g.attrs.items(): out.attrs[k] = v
            for k in g: h.copy(g[k], out, name=k)
        os.rename(tmp, dst)
        print("wrote", dst, dst.stat().st_size, flush=True)
    m.save(); h.close()
    # keep the mirror's sparse file only if requested; delete (derived coolers hold everything needed)
    (RAW / f"{s['acc']}.mirror").unlink(missing_ok=True); (RAW / f"{s['acc']}.mirror.blocks.json").unlink(missing_ok=True)


# ---------------------------------------------------------------- helpers
def universe():
    from cpg_repr_benchmark.biological_validation_v2 import coordinates as C
    return C.load_benchmark_universe()

def dedup_pixels(i, j, raw, bal):
    """4DN source coolers contain a few duplicated (bin1,bin2) pixel keys; sum them (counts and balanced values are additive)."""
    key = i.astype(np.int64) * (1 << 31) + j
    u, inv = np.unique(key, return_inverse=True)
    if len(u) == len(key): return i, j, raw, bal, 0
    return (u >> 31).astype(i.dtype), (u & ((1 << 31) - 1)).astype(j.dtype), np.bincount(inv, weights=raw), np.bincount(inv, weights=bal), len(key) - len(u)

def gc_track(res):
    g = pd.read_parquet(OUT / "hg38_gc_10kb.parquet")
    f = res // 10000
    g["b"] = g["bin"] // f
    a = g.groupby(["chrom", "b"], sort=False)[["gc", "at", "other"]].sum().reset_index()
    a["GC"] = a["gc"] / (a["gc"] + a["at"]).replace(0, np.nan)
    a["frac_other"] = a["other"] / (a["gc"] + a["at"] + a["other"])
    return a.rename(columns={"b": "bin"})

# ---------------------------------------------------------------- compartments
def compartments(sysn):
    import cooler
    import cooltools
    from scipy.stats import spearmanr
    U = universe(); qc_all = []
    for res in P["eig_res"]:
        clr = cooler.Cooler(str(cool_path(sysn, res)))
        bins = clr.bins()[:]
        chroms = [c for c in CHROMS if c in clr.chromnames]
        gc = gc_track(res)
        bins["bin"] = bins.groupby("chrom", sort=False).cumcount()
        t = bins.merge(gc[["chrom", "bin", "GC", "frac_other"]], on=["chrom", "bin"], how="left")
        # per-chromosome whole-chromosome view
        view = pd.DataFrame({"chrom": chroms, "start": 0, "end": [clr.chromsizes[c] for c in chroms], "name": chroms})
        track = t[["chrom", "start", "end", "GC"]].copy()
        track.loc[t["frac_other"] > 0.5, "GC"] = np.nan   # mostly-N bins carry no GC phasing information
        ev, vecs = cooltools.eigs_cis(clr, phasing_track=track, view_df=view, n_eigs=3, clr_weight_name="weight")
        vecs = vecs.merge(t[["chrom", "start", "bin", "GC"]], on=["chrom", "start"], how="left")
        qc = []
        for c in chroms:
            v = vecs[vecs.chrom == c]; ok = v.E1.notna() & v.GC.notna()
            r = spearmanr(v.E1[ok], v.GC[ok])[0] if ok.sum() > 10 else np.nan
            e = ev[ev.name == c] if "name" in ev else ev[ev.region == c] if "region" in ev else ev
            qc.append({"system": sysn, "res": res, "chrom": c, "n_bins": len(v), "n_valid": int(v.weight.notna().sum()), "n_E1": int(v.E1.notna().sum()),
                           "spearman_E1_GC": float(r), "low_confidence": bool(not np.isfinite(r) or abs(r) < P["spearman_flag"]),
                           "frac_E1_pos": float((v.E1 > 0).sum() / max(1, v.E1.notna().sum())),
                           "eigval1": float(e.eigval1.iloc[0]) if "eigval1" in e and len(e) else np.nan})
        qc = pd.DataFrame(qc); qc_all.append(qc)
        vecs["compartment"] = np.where(vecs.E1 > 0, "A", np.where(vecs.E1 < 0, "B", None))
        binT = vecs[["chrom", "bin", "start", "end", "weight", "GC", "E1", "E2", "E3", "compartment"]].copy() if "end" in vecs else \
            vecs.assign(end=vecs.start + res)[["chrom", "bin", "start", "end", "weight", "GC", "E1", "E2", "E3", "compartment"]]
        binT["bin_valid"] = binT.weight.notna(); binT = binT.drop(columns="weight")
        binT.to_parquet(OUT / f"{sysn}_compartment_bins_{res}.parquet", index=False, compression="zstd")
        u = U.copy(); u["bin"] = (u["pos"] - 1) // res
        u = u.merge(binT[["chrom", "bin", "E1", "compartment", "bin_valid"]], on=["chrom", "bin"], how="left")
        u["bin_valid"] = u["bin_valid"].fillna(False).astype(bool)
        u = u[["cpg_idx", "chrom", "pos", "bin", "E1", "compartment", "bin_valid"]].sort_values("cpg_idx")
        u.to_parquet(OUT / f"{sysn}_cpg_compartment_{res}.parquet", index=False, compression="zstd")
        print(sysn, res, "CpG valid", u.bin_valid.mean(), "E1 defined", u.E1.notna().mean(), flush=True)
    pd.concat(qc_all).to_csv(OUT / f"{sysn}_compartment_qc.csv", index=False)

# ---------------------------------------------------------------- contacts
def _cpg_bin_counts(U, res):
    u = U[U.chrom.isin(CHROMS)]
    return u.assign(bin=(u["pos"] - 1) // res).groupby(["chrom", "bin"]).size()

def intra(sysn):
    import cooler
    import cooltools
    U = universe(); res = P["intra_res"]
    clr = cooler.Cooler(str(cool_path(sysn, res)))
    cnt = _cpg_bin_counts(U, res)
    bins = clr.bins()[:]; bins["bin"] = bins.groupby("chrom", sort=False).cumcount()
    view = pd.DataFrame({"chrom": CHROMS, "start": 0, "end": [clr.chromsizes[c] for c in CHROMS], "name": CHROMS})
    exp = cooltools.expected_cis(clr, view_df=view, clr_weight_name="weight", smooth=False, aggregate_smoothed=False, nproc=8)
    exp = exp[exp.region1 == exp.region2]
    contacts, pools, stats = [], [], []
    for ci, c in enumerate(CHROMS):
        b = bins[bins.chrom == c]; nb = len(b); w = b["weight"].to_numpy(); valid = np.isfinite(w)
        pix = clr.matrix(balance="weight", as_pixels=True, join=False).fetch(c)
        i0 = int(clr.offset(c)); i = pix.bin1_id.to_numpy() - i0; j = pix.bin2_id.to_numpy() - i0
        raw = pix["count"].to_numpy(); bal = pix["balanced"].to_numpy()
        i, j, raw, bal, ndup = dedup_pixels(i, j, raw, bal)
        cov = np.bincount(i, weights=raw, minlength=nb) + np.bincount(j[j != i], weights=raw[j != i], minlength=nb)
        qthr = np.quantile(cov[valid], P["cov_quantile"])
        ncpg = np.array([cnt.get((c, k), 0) for k in range(nb)]); has = ncpg > 0
        elig = valid & (cov >= qthr) & has
        d = j - i
        e = exp[exp.region1 == c].set_index("dist")["balanced.avg"]
        earr = np.full(nb, np.nan); earr[e.index.to_numpy()[e.index.to_numpy() < nb]] = e.to_numpy()[e.index.to_numpy() < nb]
        sel = (d >= P["intra_min_bins"]) & (d <= P["intra_max_bins"]) & valid[i] & valid[j] & np.isfinite(bal)
        i, j, raw, bal, d = i[sel], j[sel], raw[sel], bal[sel], d[sel]
        oe = bal / earr[d]
        isc = (raw >= P["intra_min_raw"]) & (oe >= P["intra_min_oe"])
        ct = pd.DataFrame({"chrom": c, "bin_i": i[isc], "bin_j": j[isc], "dist_bins": d[isc], "dist_bp": d[isc] * res, "raw": raw[isc].astype(np.float32),
                               "balanced": bal[isc].astype(np.float32), "expected": earr[d[isc]].astype(np.float32), "obs_exp": oe[isc].astype(np.float32),
                               "n_cpg_i": ncpg[i[isc]], "n_cpg_j": ncpg[j[isc]]})
        contacts.append(ct)
        # pool: for each distance, eligible pairs not in contacts with raw==0 (absent pixel) or oe<=0.5
        rng_state = []
        key_pix = {}
        low_pix = ~isc & (oe <= P["noncontact_max_oe"])      # observed pixels with low obs/exp
        n_ct_cpg = ct[(ct.n_cpg_i > 0) & (ct.n_cpg_j > 0)].groupby("dist_bins").size()
        for dd in range(P["intra_min_bins"], min(P["intra_max_bins"], nb - 1) + 1):
            cap = max(P["pool_mult"] * int(n_ct_cpg.get(dd, 0)), P["pool_min"])
            ii = np.arange(0, nb - dd); ok = elig[ii] & elig[ii + dd]
            ii = ii[ok]
            if not len(ii): continue
            m = (d == dd)
            present = np.zeros(nb, dtype=bool); present[i[m]] = True
            lowp = np.zeros(nb, dtype=bool); lowp[i[m & low_pix]] = True
            okc = ~present[ii] | lowp[ii]
            ii = ii[okc]
            if not len(ii): continue
            rng = np.random.default_rng([SEED, ci, dd])
            if len(ii) > cap: ii = np.sort(rng.choice(ii, cap, replace=False))
            r_ = np.zeros(nb); o_ = np.zeros(nb); r_[i[m]] = raw[m]; o_[i[m]] = oe[m]
            pools.append(pd.DataFrame({"chrom": c, "bin_i": ii, "bin_j": ii + dd, "dist_bins": dd, "dist_bp": dd * res, "raw": r_[ii].astype(np.float32),
                                           "obs_exp": o_[ii].astype(np.float32), "cov_i": cov[ii].astype(np.float32), "cov_j": cov[ii + dd].astype(np.float32),
                                           "n_cpg_i": ncpg[ii], "n_cpg_j": ncpg[ii + dd]}))
        stats.append({"system": sysn, "chrom": c, "n_bins": nb, "n_valid": int(valid.sum()), "n_elig": int(elig.sum()), "cov_q25": float(qthr),
                          "n_contacts": int(isc.sum()), "n_contacts_cpg_both": int(((ct.n_cpg_i > 0) & (ct.n_cpg_j > 0)).sum()), "n_candidate_pixels": len(i), "n_duplicate_pixel_keys": int(ndup)})
        print(sysn, c, stats[-1], flush=True)
    pd.concat(contacts, ignore_index=True).to_parquet(OUT / f"{sysn}_intra_contacts_{res}.parquet", index=False, compression="zstd")
    pd.concat(pools, ignore_index=True).to_parquet(OUT / f"{sysn}_intra_noncontact_pool_{res}.parquet", index=False, compression="zstd")
    pd.DataFrame(stats).to_csv(OUT / f"{sysn}_intra_stats.csv", index=False)

def inter(sysn):
    import cooler
    U = universe(); res = P["inter_res"]
    clr = cooler.Cooler(str(cool_path(sysn, res)))
    cnt = _cpg_bin_counts(U, res)
    bins = clr.bins()[:]; bins["bin"] = bins.groupby("chrom", sort=False).cumcount()
    off = {c: int(clr.offset(c)) for c in CHROMS}; nbs = {c: int(clr.extent(c)[1] - clr.extent(c)[0]) for c in CHROMS}
    wt = {c: bins[bins.chrom == c]["weight"].to_numpy() for c in CHROMS}
    # trans coverage (raw) per bin: sum over all other chromosomes
    tcov = {c: np.zeros(nbs[c]) for c in CHROMS}; pairs = {}
    for a in range(len(CHROMS)):
        for b in range(a + 1, len(CHROMS)):
            ca, cb = CHROMS[a], CHROMS[b]
            px = clr.matrix(balance="weight", as_pixels=True, join=False).fetch(ca, cb)
            i = px.bin1_id.to_numpy() - off[ca]; j = px.bin2_id.to_numpy() - off[cb]
            raw = px["count"].to_numpy(); bal = px["balanced"].to_numpy()
            i, j, raw, bal, _ = dedup_pixels(i, j, raw, bal)
            tcov[ca] += np.bincount(i, weights=raw, minlength=nbs[ca]); tcov[cb] += np.bincount(j, weights=raw, minlength=nbs[cb])
            pairs[(ca, cb)] = (i, j, raw, bal)
    dec = {}
    for c in CHROMS:
        v = np.isfinite(wt[c]); dec[c] = np.full(nbs[c], -1)
    allc = np.concatenate([tcov[c][np.isfinite(wt[c])] for c in CHROMS]); qs = np.quantile(allc, np.linspace(0, 1, 11)[1:-1])
    for c in CHROMS:
        dec[c] = np.where(np.isfinite(wt[c]), np.searchsorted(qs, tcov[c], side="right"), -1)
    contacts, pools, stats = [], [], []
    for pi, ((ca, cb), (i, j, raw, bal)) in enumerate(pairs.items()):
        va, vb = np.isfinite(wt[ca]), np.isfinite(wt[cb])
        npairs = int(va.sum()) * int(vb.sum())
        ok = va[i] & vb[j] & np.isfinite(bal)
        exp = bal[ok].sum() / max(1, npairs)      # mean balanced contact over valid bin pairs (zeros included)
        i, j, raw, bal = i[ok], j[ok], raw[ok], bal[ok]
        oe = bal / exp
        isc = (raw >= P["inter_min_raw"]) & (oe >= P["inter_min_oe"])
        na = np.array([cnt.get((ca, k), 0) for k in range(nbs[ca])]); nb_ = np.array([cnt.get((cb, k), 0) for k in range(nbs[cb])])
        ct = pd.DataFrame({"chrom_i": ca, "chrom_j": cb, "bin_i": i[isc], "bin_j": j[isc], "raw": raw[isc].astype(np.float32), "balanced": bal[isc].astype(np.float32),
                               "expected": np.float32(exp), "obs_exp": oe[isc].astype(np.float32), "n_cpg_i": na[i[isc]], "n_cpg_j": nb_[j[isc]],
                               "tcov_decile_i": dec[ca][i[isc]], "tcov_decile_j": dec[cb][j[isc]]})
        contacts.append(ct)
        # pool candidates: both bins valid with >=1 CpG, absent or oe<=0.5
        A = np.where(va & (na > 0))[0]; B = np.where(vb & (nb_ > 0))[0]
        lowset = np.zeros((nbs[ca], nbs[cb]), dtype=bool)
        lowset[np.ix_(A, B)] = True
        lowset[i[~(oe <= P["noncontact_max_oe"])], j[~(oe <= P["noncontact_max_oe"])]] = False
        ia, ja = np.nonzero(lowset)
        n_ct_cpg = int(((ct.n_cpg_i > 0) & (ct.n_cpg_j > 0)).sum())
        cap = max(P["pool_mult"] * n_ct_cpg, P["pool_min"])
        rng = np.random.default_rng([SEED, 1000 + pi])
        if len(ia) > cap:
            k = np.sort(rng.choice(len(ia), cap, replace=False)); ia, ja = ia[k], ja[k]
        lookup = {(a_, b_): (r_, o_) for a_, b_, r_, o_ in zip(i, j, raw, oe)}
        rr = np.array([lookup.get((a_, b_), (0, 0.0))[0] for a_, b_ in zip(ia, ja)], dtype=np.float32)
        oo = np.array([lookup.get((a_, b_), (0, 0.0))[1] for a_, b_ in zip(ia, ja)], dtype=np.float32)
        pools.append(pd.DataFrame({"chrom_i": ca, "chrom_j": cb, "bin_i": ia, "bin_j": ja, "raw": rr, "obs_exp": oo, "n_cpg_i": na[ia], "n_cpg_j": nb_[ja],
                                       "tcov_decile_i": dec[ca][ia], "tcov_decile_j": dec[cb][ja]}))
        stats.append({"system": sysn, "chrom_i": ca, "chrom_j": cb, "n_valid_pairs": npairs, "expected": float(exp), "n_contacts": int(isc.sum()),
                          "n_contacts_cpg_both": n_ct_cpg, "n_pool": len(ia)})
    pd.concat(contacts, ignore_index=True).to_parquet(OUT / f"{sysn}_inter_contacts_{res}.parquet", index=False, compression="zstd")
    pd.concat(pools, ignore_index=True).to_parquet(OUT / f"{sysn}_inter_noncontact_pool_{res}.parquet", index=False, compression="zstd")
    pd.DataFrame(stats).to_csv(OUT / f"{sysn}_inter_stats.csv", index=False)
    print(sysn, "inter done", sum(len(x) for x in contacts), "contacts", flush=True)

# ---------------------------------------------------------------- projection to CpG pairs (no bulk materialization)
def project_to_cpg_pairs(bin_pairs, cpg_table, res, K=20, seed=SEED, chrom_cols=("chrom", "chrom")):
    """bin_pairs: DataFrame with chrom (or chrom_i/chrom_j via chrom_cols), bin_i, bin_j. cpg_table: DataFrame[cpg_idx, chrom, pos]
    (benchmark universe). Returns DataFrame[cpg_i, cpg_j, bin_i, bin_j, chrom_i, chrom_j, row] with at most K CpG pairs per bin pair,
    sampled deterministically via default_rng([seed, hash(bin pair)]). Every benchmark CpG in bin i x every CpG in bin j is eligible."""
    t = cpg_table.assign(bin=(cpg_table["pos"] - 1) // res).sort_values(["chrom", "bin", "cpg_idx"])
    grp = {k: v.to_numpy() for k, v in t.groupby(["chrom", "bin"])["cpg_idx"]}
    ci_, cj_ = chrom_cols
    out = []
    for r, (ci, cj, bi, bj) in enumerate(zip(bin_pairs[ci_], bin_pairs[cj_], bin_pairs["bin_i"], bin_pairs["bin_j"])):
        a = grp.get((ci, int(bi))); b = grp.get((cj, int(bj)))
        if a is None or b is None: continue
        n = len(a) * len(b)
        if n <= K: flat = np.arange(n)
        else:
            rng = np.random.default_rng([seed, zlib.crc32(f"{ci}|{cj}".encode()) % 10_007, int(bi), int(bj)])
            flat = np.sort(rng.choice(n, K, replace=False))
        out.append((a[flat // len(b)], b[flat % len(b)], np.full(len(flat), r)))
    if not out: return pd.DataFrame(columns=["i", "j", "chrom_i", "chrom_j", "dist", "bin_i", "bin_j", "row"])
    ii, jj, rr = (np.concatenate(x) for x in zip(*out))
    bp = bin_pairs.iloc[rr]
    ci = bp[ci_].to_numpy(); cj = bp[cj_].to_numpy()
    pos = cpg_table.set_index("cpg_idx")["pos"]
    dist = np.where(ci == cj, np.abs(pos.loc[ii].to_numpy() - pos.loc[jj].to_numpy()).astype(float), np.nan)
    # columns follow biological_validation_v2.matching.match_pairs (i, j, chrom_i, chrom_j, dist)
    return pd.DataFrame({"i": ii, "j": jj, "chrom_i": ci, "chrom_j": cj, "dist": dist, "bin_i": bp["bin_i"].to_numpy(), "bin_j": bp["bin_j"].to_numpy(), "row": rr})

# ---------------------------------------------------------------- audit + manifest
def audit(sysn):
    import cooler
    U = universe(); res_out = {"system": sysn, "universe_n": len(U), "n_universe_chrY_chrM_excluded_from_maps": int((~U.chrom.isin(CHROMS)).sum())}
    for res in P["eig_res"]:
        f = OUT / f"{sysn}_cpg_compartment_{res}.parquet"
        if not f.exists(): continue
        t = pd.read_parquet(f)
        qc = pd.read_csv(OUT / f"{sysn}_compartment_qc.csv"); qc = qc[qc.res == res]
        res_out[f"compartment_{res}"] = {"frac_cpg_in_valid_bin": float(t.bin_valid.mean()), "frac_cpg_E1_defined": float(t.E1.notna().mean()),
            "n_A": int((t.compartment == "A").sum()), "n_B": int((t.compartment == "B").sum()), "n_cpg_no_E1": int(t.E1.isna().sum()),
            "n_low_confidence_chroms": int(qc.low_confidence.sum()), "spearman_E1_GC_min": float(qc.spearman_E1_GC.min()), "spearman_E1_GC_median": float(qc.spearman_E1_GC.median())}
    out = {}
    for kind, res in (("intra", P["intra_res"]), ("inter", P["inter_res"])):
        fc = OUT / f"{sysn}_{kind}_contacts_{res}.parquet"
        if not fc.exists(): continue
        c = pd.read_parquet(fc); pl = pd.read_parquet(OUT / f"{sysn}_{kind}_noncontact_pool_{res}.parquet")
        both = (c.n_cpg_i > 0) & (c.n_cpg_j > 0); pb = (pl.n_cpg_i > 0) & (pl.n_cpg_j > 0)
        out[kind] = {"res": res, "n_contact_bin_pairs": len(c), "n_contact_bin_pairs_cpg_both_sides": int(both.sum()),
                         "est_cpg_pairs_all": int((c.n_cpg_i[both].astype("int64") * c.n_cpg_j[both]).sum()),
                         "est_cpg_pairs_K20": int(np.minimum(c.n_cpg_i[both].astype("int64") * c.n_cpg_j[both], 20).sum()),
                         "n_pool_bin_pairs": len(pl), "n_pool_cpg_both_sides": int(pb.sum()),
                         "est_pool_cpg_pairs_all": int((pl.n_cpg_i[pb].astype("int64") * pl.n_cpg_j[pb]).sum()),
                         "est_pool_cpg_pairs_K20": int(np.minimum(pl.n_cpg_i[pb].astype("int64") * pl.n_cpg_j[pb], 20).sum()),
                         "pool_obs_exp_max": float(pl.obs_exp.max()), "contact_obs_exp_min": float(c.obs_exp.min())}
        if kind == "intra":
            b = np.arange(0, 1)  # distance-stratified counts of contacts with CpG both sides
            out[kind]["contacts_cpg_both_by_dist_bp"] = c[both].groupby(pd.cut(c.dist_bp[both], [0, 1e5, 5e5, 1e6, 2e6], right=True), observed=True).size().astype(int).rename(str).to_dict()
            out[kind]["pool_by_dist_bp"] = pl[pb].groupby(pd.cut(pl.dist_bp[pb], [0, 1e5, 5e5, 1e6, 2e6], right=True), observed=True).size().astype(int).rename(str).to_dict()
    res_out.update(out)
    # build check: cooler chromsizes vs hg38 fasta-derived bin counts
    g = pd.read_parquet(OUT / "hg38_gc_10kb.parquet").groupby("chrom").size()
    clr = cooler.Cooler(str(cool_path(sysn, 10000 if cool_path(sysn, 10000).exists() else 50000).with_name(cool_path(sysn, 10000 if cool_path(sysn, 10000).exists() else 50000).name)))
    bs = clr.binsize
    res_out["build_check"] = {"binsize": int(bs), "chroms_match_hg38_fasta": bool(all(-(-int(clr.chromsizes[c]) // bs) == int(g[c]) * 10000 // bs or -(-int(clr.chromsizes[c]) // bs) == -(-g[c] * 10000 // bs) for c in CHROMS)),
                                  "chr1_len": int(clr.chromsizes["chr1"])}
    for r_, kind_ in ((P["intra_res"], "intra_bins"), (P["inter_res"], "inter_bins")):
        if not cool_path(sysn, r_).exists(): continue
        cl = cooler.Cooler(str(cool_path(sysn, r_))); bb = cl.bins()[:]; bb["bin"] = bb.groupby("chrom", observed=True, sort=False).cumcount()
        bb = bb[bb.chrom.isin(CHROMS)]; vb = bb[np.isfinite(bb.weight)][["chrom", "bin"]].assign(v=True)
        uu = U.assign(bin=(U["pos"] - 1) // r_).merge(vb, on=["chrom", "bin"], how="left")
        res_out[kind_] = {"res": r_, "frac_bins_valid": float(np.isfinite(bb.weight).mean()), "frac_cpg_in_valid_bin": float(uu.v.notna().mean()),
                              "n_bins_with_cpg": int(U.assign(bin=(U["pos"] - 1) // r_).groupby(["chrom", "bin"]).ngroups)}
    res_out["files"] = {f.name: {"rows": len(pd.read_parquet(f)) if f.suffix == ".parquet" else None, "MB": round(f.stat().st_size / 1e6, 2)}
                        for f in sorted(OUT.glob(f"{sysn}_*")) if f.suffix in (".parquet", ".csv")}
    json.dump(res_out, open(OUT / f"{sysn}_audit.json", "w"), indent=2, default=str)
    print(json.dumps(res_out, indent=1, default=str)[:4000])

def manifest():
    import cooler
    import cooltools
    import scipy

    from cpg_repr_benchmark.biological_validation_v2 import provenance as PV
    sources, audits = [], {}
    for sysn, s in SYSTEMS.items():
        if not (OUT / f"{sysn}_audit.json").exists(): continue
        audits[sysn] = json.load(open(OUT / f"{sysn}_audit.json"))
        ext = {f"{cool_path(sysn, r).name}": {"sha256": sha256(cool_path(sysn, r)), "bytes": cool_path(sysn, r).stat().st_size}
               for r in FETCH_RES if cool_path(sysn, r).exists()}
        sources.append({"system": sysn, "description": s["desc"], "accession": s["acc"], "experiment_set": s["expset"], "build": "GRCh38",
                            "url_open_data": S3 + s["key"], "url_portal": f"https://data.4dnucleome.org/files-processed/{s['acc']}/",
                            "full_mcool_bytes": s["size"], "full_mcool_md5_portal": s["md5"], "fetched_by": "HTTP range requests (single resolutions only; full-file md5 not verifiable)",
                            "extracted_single_resolution_coolers": ext, "resolutions_extracted": sorted(int(k.split("_")[1].split(".")[0]) for k in ext),
                            "license": "4DN open data (public; no auth)"})
    derived = {f.name: sha256(f) for f in sorted(OUT.glob("*")) if f.suffix in (".parquet", ".csv", ".json") and f.name != "MANIFEST.json"}
    params = dict(P, seed=SEED, chroms_in_maps=CHROMS, cpg_to_bin="floor((pos-1)/binsize)", fetch_res=FETCH_RES,
                  tools={"python": platform.python_version(), "cooler": cooler.__version__, "cooltools": cooltools.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
                  derived_sha256=derived, created=datetime.datetime.now().isoformat(timespec="seconds"),
                  independence_note="No Hi-C/Micro-C/compartment/TAD/loop track among representation inputs (track_contract.tsv: histone, TF, DNase ChIP/seq only); 201 CTCF + 14 RAD21 + 6 SMC3 ChIP-seq tracks are inputs",
                  qc_notes="4DN source coolers contain a few duplicated (bin1,bin2) pixel keys; summed. ICE weights from the 4DN mcool used unchanged.",
                  genome_assembly_attr_in_cooler="unknown",
                  genome_assembly_verdict=(json.load(open(OUT / "assembly_validation.json"))["verdict"] if (OUT / "assembly_validation.json").exists() else None),
                  genome_assembly_evidence="assembly_validation.json (chrom lengths vs hg38.fa.gz and hg19, 4DN portal genome_assembly, CpG dinucleotide check, E1-GC)",
                  frozen_pairs_and_compartments="FROZEN.json (produced by freeze_4dn_pairs.py before any embedding metric); roles: H1 primary, HFFc6 sensitivity, GM12878 secondary",
                  roles={"H1_MicroC": "primary", "HFFc6_MicroC": "sensitivity", "GM12878_HiC": "secondary"})
    PV.write_manifest(OUT / "MANIFEST.json", "3d_genome", sources, audits, params)
    print("manifest written")

if __name__ == "__main__":
    cmd = sys.argv[1]; sysn = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd == "manifest": manifest()
    else: {"fetch": fetch, "compartments": compartments, "intra": intra, "inter": inter, "audit": audit}[cmd](sysn)
