#!/usr/bin/env python
# ruff: noqa: B023, DTZ005, EXE001, F841, RUF007, SIM115  (one-shot data-prep script; style-only, behaviour unchanged)
"""Freeze 3d_genome contact / non-contact CpG pairs and compartment tables BEFORE any embedding is touched.
CPU only. Uses no embeddings. Subcommands:
  cov <system>            per-bin cis raw coverage + eligibility (same rule as the non-contact pool), parquet
  intra <system>          frozen_pairs_<SYS>_intra10kb_seed17.parquet
  inter <system>          frozen_pairs_<SYS>_inter1Mb_seed17.parquet  (exploratory)
  compartments            frozen_compartments_<SYS>_100kb.parquet
  freeze                  FROZEN.json (sha256, counts, roles) + manifest update
Matching (intra): exact chromosome x fine log-spaced distance bin x quintile of pair min bin cis coverage [x per-CpG covariate tercile bins if a covariates
table exists]; within stratum a deterministic random subset of the larger side is taken to size of the smaller, then the
two sets are paired by distance rank (1:1, no replacement). Positives left without a control are DROPPED and reported.
"""
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts/bioval_v2"))
import prepare_4dn as P4

from cpg_repr_benchmark.biological_validation_v2 import splits as SP

OUT = P4.OUT
SEED = 17; K = 20; N_FOLDS = 5
COVFILE = ROOT / "data/derived/bioval_v2/covariates/universe_covariates.parquet"
ROLE = {"H1_MicroC": "primary", "HFFc6_MicroC": "sensitivity", "GM12878_HiC": "secondary"}
TAG = {"H1_MicroC": "H1", "HFFc6_MicroC": "HFFc6", "GM12878_HiC": "GM12878"}
DIST_EDGES = np.geomspace(1e4, 2.02e6, 81)     # 80 fine log-spaced bins covering all CpG distances of 20 kb-2 Mb bin pairs

def fold_map(U):
    return SP.chromosome_blocked_folds(sorted(U["chrom"].unique()), N_FOLDS, SEED)

# ------------------------------------------------------------------ bin coverage
def _cov_chrom(args):
    sysn, c = args
    import cooler
    clr = cooler.Cooler(str(P4.cool_path(sysn, P4.P["intra_res"])))
    bins = clr.bins()[:]; w = bins[bins.chrom == c]["weight"].to_numpy(); nb = len(w); valid = np.isfinite(w)
    pix = clr.matrix(balance=False, as_pixels=True, join=False).fetch(c)
    i0 = int(clr.offset(c)); i = pix.bin1_id.to_numpy() - i0; j = pix.bin2_id.to_numpy() - i0; raw = pix["count"].to_numpy().astype(float)
    i, j, raw, _, _ = P4.dedup_pixels(i, j, raw, raw)
    cov = np.bincount(i, weights=raw, minlength=nb) + np.bincount(j[j != i], weights=raw[j != i], minlength=nb)
    q = float(np.quantile(cov[valid], P4.P["cov_quantile"]))
    return pd.DataFrame({"chrom": c, "bin": np.arange(nb), "cov": cov, "valid": valid, "cov_q25": q, "cov_ok": valid & (cov >= q)})

def cov(sysn):
    from multiprocessing import Pool
    with Pool(8) as p: parts = p.map(_cov_chrom, [(sysn, c) for c in P4.CHROMS])
    pd.concat(parts, ignore_index=True).to_parquet(OUT / f"{sysn}_intra_bin_cov_10000.parquet", index=False, compression="zstd")

# ------------------------------------------------------------------ pair assembly helpers
def _canon(df):
    """i<j on cpg_idx (swap chrom/bin/pos columns accordingly)."""
    sw = df["cpg_i"].to_numpy() > df["cpg_j"].to_numpy()
    for a, b in (("cpg_i", "cpg_j"), ("chrom_i", "chrom_j"), ("bin_i", "bin_j")):
        x = df[a].to_numpy().copy(); y = df[b].to_numpy().copy()
        df[a] = np.where(sw, y, x); df[b] = np.where(sw, x, y)
    return df

def _project(bp, U, res, chrom_cols):
    pr = P4.project_to_cpg_pairs(bp, U[["cpg_idx", "chrom", "pos"]], res, K=K, seed=SEED, chrom_cols=chrom_cols)
    pr = pr.rename(columns={"i": "cpg_i", "j": "cpg_j", "dist": "distance"})
    bpid = (pr["chrom_i"] + ":" + pr["bin_i"].astype(str) + "-" + pr["chrom_j"] + ":" + pr["bin_j"].astype(str)).to_numpy()
    pr["bin_pair_id"] = bpid
    return pr.drop(columns="row")

def _cov_strata(pairs, covcols):
    """Tercile bins (cut points from the union of positives+pool) of the pair mean of each covariate."""
    if not covcols: return np.zeros(len(pairs), dtype=np.int64), {}
    cov = pd.read_parquet(COVFILE).set_index("cpg_idx")
    key = np.zeros(len(pairs), dtype=np.int64); cuts = {}
    for c in covcols:
        m = (cov[c].reindex(pairs["cpg_i"]).to_numpy(float) + cov[c].reindex(pairs["cpg_j"]).to_numpy(float)) / 2
        q = np.nanquantile(m, [1 / 3, 2 / 3]); cuts[c] = [float(x) for x in q]
        b = np.where(np.isnan(m), 3, np.searchsorted(q, m))
        key = key * 4 + b
    return key, cuts

def _match(df, strat, order_col, seed_tag):
    """df has 'label' (1/0). strat: array same length. Returns (idx_pos_kept, idx_ctrl_kept) aligned 1:1."""
    ip, ic = [], []
    lab = df["label"].to_numpy(); od = df[order_col].to_numpy(float)
    ordr = np.lexsort((np.arange(len(df)), strat))
    s = strat[ordr]; bounds = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = ordr[a:b]; sk = int(s[a])
        P = idx[lab[idx] == 1]; C = idx[lab[idx] == 0]
        n = min(len(P), len(C))
        if n == 0: continue
        rng = np.random.default_rng([SEED, seed_tag, sk & 0x7FFFFFFF, sk >> 31 & 0x7FFFFFFF])
        if len(P) > n: P = P[np.sort(rng.permutation(len(P))[:n])]
        if len(C) > n: C = C[np.sort(rng.permutation(len(C))[:n])]
        P = P[np.lexsort((P, od[P]))]; C = C[np.lexsort((C, od[C]))]
        ip.append(P); ic.append(C)
    return (np.concatenate(ip), np.concatenate(ic)) if ip else (np.array([], int), np.array([], int))

def _smd(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    sd = np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2)
    return float((np.nanmean(a) - np.nanmean(b)) / sd) if sd > 0 else 0.0

def _finish(df, ip, ic, folds, tag):
    pos = df.iloc[ip].reset_index(drop=True); ctl = df.iloc[ic].reset_index(drop=True)
    pid_p = (pos["cpg_i"].astype(str) + "_" + pos["cpg_j"].astype(str)).to_numpy()
    pid_c = (ctl["cpg_i"].astype(str) + "_" + ctl["cpg_j"].astype(str)).to_numpy()
    pos["pair_id"] = "P_" + pid_p; ctl["pair_id"] = "C_" + pid_c
    pos["matched_to"] = ctl["pair_id"].to_numpy(); ctl["matched_to"] = pos["pair_id"].to_numpy()
    pos["match_id"] = np.arange(len(pos)); ctl["match_id"] = np.arange(len(pos))
    out = pd.concat([pos, ctl], ignore_index=True)
    return out

COLS = ["pair_id", "cpg_i", "cpg_j", "chrom_i", "chrom_j", "distance", "label", "bin_pair_id", "bin_i", "bin_j", "fold", "matched_to", "match_id"]

# ------------------------------------------------------------------ intra
def intra(sysn):
    U = P4.universe(); folds = fold_map(U); res = P4.P["intra_res"]
    ct = pd.read_parquet(OUT / f"{sysn}_intra_contacts_{res}.parquet"); pl = pd.read_parquet(OUT / f"{sysn}_intra_noncontact_pool_{res}.parquet")
    bc = pd.read_parquet(OUT / f"{sysn}_intra_bin_cov_{res}.parquet")
    rep = {"system": sysn, "role": ROLE[sysn], "resolution": res, "K": K, "seed": SEED, "n_folds": N_FOLDS, "folds": folds}
    n0 = len(ct); both = ct[(ct.n_cpg_i > 0) & (ct.n_cpg_j > 0)]
    rep["contact_bin_pairs_total"] = int(n0); rep["contact_bin_pairs_cpg_both_sides"] = len(both)
    ok = bc.set_index(["chrom", "bin"])["cov_ok"]
    e_i = ok.reindex(pd.MultiIndex.from_arrays([both.chrom, both.bin_i])).fillna(False).to_numpy()
    e_j = ok.reindex(pd.MultiIndex.from_arrays([both.chrom, both.bin_j])).fillna(False).to_numpy()
    both = both[e_i & e_j]
    rep["contact_bin_pairs_after_pool_equivalent_coverage_eligibility"] = len(both)
    both = both[both.chrom.isin(set(folds))]
    rep["contact_bin_pairs_on_chroms_with_cpgs"] = len(both)
    pl = pl[pl.chrom.isin(set(folds))]
    pos = _project(both, U, res, ("chrom", "chrom")); pos["label"] = 1
    ctl = _project(pl, U, res, ("chrom", "chrom")); ctl["label"] = 0
    rep["positive_cpg_pairs_before_matching"] = len(pos); rep["control_pool_cpg_pairs"] = len(ctl)
    df = _canon(pd.concat([pos, ctl], ignore_index=True))
    covcols = []; diagcols = []
    if COVFILE.exists():
        cc = pd.read_parquet(COVFILE, columns=None).columns
        covcols = [c for c in ("gc_content", "tss_dist") if c in cc]
        diagcols = [c for c in ("gc_content", "cpg_density_hg38", "cpg_density", "tss_dist") if c in cc]
    rep["per_cpg_covariates_used_as_strata"] = covcols; rep["per_cpg_covariates_balance_reported"] = diagcols
    rep["covariates_file_present"] = COVFILE.exists()
    ck, cuts = _cov_strata(df, covcols); rep["covariate_tercile_cuts"] = cuts
    db = np.clip(np.digitize(df["distance"].to_numpy(float), DIST_EDGES) - 1, 0, len(DIST_EDGES) - 2)
    chn = df["chrom_i"].map({c: k for k, c in enumerate(sorted(folds))}).to_numpy()
    cvs = bc.set_index(["chrom", "bin"])["cov"]
    a_ = cvs.reindex(pd.MultiIndex.from_arrays([df.chrom_i, df.bin_i])).to_numpy(); b_ = cvs.reindex(pd.MultiIndex.from_arrays([df.chrom_j, df.bin_j])).to_numpy()
    lcv = np.log10(np.minimum(a_, b_) + 1)
    qcv = np.quantile(lcv, [0.2, 0.4, 0.6, 0.8]); rep["min_bin_cov_quintile_cuts_log10"] = [float(x) for x in qcv]
    cq = np.searchsorted(qcv, lcv)
    strat = ((ck * 8 + cq) * 128 + db) * 32 + chn
    ip, ic = _match(df, strat.astype(np.int64), "distance", 1)
    out = _finish(df, ip, ic, folds, sysn)
    out["fold"] = out["chrom_i"].map(folds).astype(int)
    # balance (pair-level distance and bin-level coverage; bin coverage uses the cov table, not any embedding)
    cv = bc.set_index(["chrom", "bin"])["cov"]
    def lc(d):
        a = cv.reindex(pd.MultiIndex.from_arrays([d.chrom_i, d.bin_i])).to_numpy(); b = cv.reindex(pd.MultiIndex.from_arrays([d.chrom_j, d.bin_j])).to_numpy()
        return np.log10(np.minimum(a, b) + 1)
    P_ = out[out.label == 1]; C_ = out[out.label == 0]
    pre_p, pre_c = df[df.label == 1], df[df.label == 0]
    bal = {"distance_bp": {"smd_before": _smd(pre_p.distance, pre_c.distance), "smd_after": _smd(P_.distance, C_.distance)},
           "log10_distance": {"smd_before": _smd(np.log10(pre_p.distance), np.log10(pre_c.distance)), "smd_after": _smd(np.log10(P_.distance), np.log10(C_.distance))},
           "log10_min_bin_cov": {"smd_before": _smd(lc(pre_p), lc(pre_c)), "smd_after": _smd(lc(P_), lc(C_))}}
    if diagcols:
        covcols = diagcols
        cvv = pd.read_parquet(COVFILE).set_index("cpg_idx")
        for c in covcols:
            f = lambda d: (cvv[c].reindex(d.cpg_i).to_numpy(float) + cvv[c].reindex(d.cpg_j).to_numpy(float)) / 2
            bal[f"pair_mean_{c}"] = {"smd_before": _smd(f(pre_p), f(pre_c)), "smd_after": _smd(f(P_), f(C_))}
    bal["max_abs_dist_diff_within_match_bp"] = float(np.abs(P_.sort_values("match_id").distance.to_numpy() - C_.sort_values("match_id").distance.to_numpy()).max())
    bal["median_abs_dist_diff_within_match_bp"] = float(np.median(np.abs(P_.sort_values("match_id").distance.to_numpy() - C_.sort_values("match_id").distance.to_numpy())))
    rep["balance"] = bal
    npos = len(P_); rep.update(n_positives=npos, n_controls=len(C_), n_positives_unmatched_dropped=int(len(pos) - npos),
                                    drop_rate_vs_projected_positives=float(1 - npos / len(pos)))
    rep["n_distinct_bin_pairs_pos"] = int(P_.bin_pair_id.nunique()); rep["n_distinct_bin_pairs_ctl"] = int(C_.bin_pair_id.nunique())
    rep["per_fold_n_pos"] = {int(k): int(v) for k, v in P_.groupby("fold").size().items()}
    rep["per_chrom_n_pos"] = {k: int(v) for k, v in P_.groupby("chrom_i").size().items()}
    out = out.sort_values(["label", "match_id"], ascending=[False, True]).reset_index(drop=True)
    f = OUT / f"frozen_pairs_{TAG[sysn]}_intra10kb_seed17.parquet"
    out[COLS].to_parquet(f, index=False, compression="zstd")
    json.dump(rep, open(OUT / f"frozen_pairs_{TAG[sysn]}_intra10kb_seed17.report.json", "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("folds", "per_chrom_n_pos")}, indent=1, default=str))

# ------------------------------------------------------------------ inter (exploratory)
def inter(sysn):
    U = P4.universe(); folds = fold_map(U); res = P4.P["inter_res"]
    ct = pd.read_parquet(OUT / f"{sysn}_inter_contacts_{res}.parquet"); pl = pd.read_parquet(OUT / f"{sysn}_inter_noncontact_pool_{res}.parquet")
    rep = {"system": sysn, "role": "exploratory (secondary)", "resolution": res, "K": K, "seed": SEED, "folds": folds,
               "contact_bin_pairs_total": len(ct), "pool_bin_pairs_total": len(pl)}
    ok = set(folds)
    ct = ct[(ct.n_cpg_i > 0) & (ct.n_cpg_j > 0) & ct.chrom_i.isin(ok) & ct.chrom_j.isin(ok)]; pl = pl[pl.chrom_i.isin(ok) & pl.chrom_j.isin(ok)]
    rep["contact_bin_pairs_cpg_both_sides_on_cpg_chroms"] = len(ct); rep["pool_bin_pairs_on_cpg_chroms"] = len(pl)
    pos = _project(ct, U, res, ("chrom_i", "chrom_j")); pos["label"] = 1
    ctl = _project(pl, U, res, ("chrom_i", "chrom_j")); ctl["label"] = 0
    # coverage group from trans-coverage deciles of the two bins (mean decile -> 3 groups)
    def grp(src, d):
        m = (src["tcov_decile_i"].to_numpy() + src["tcov_decile_j"].to_numpy()) / 2
        return pd.Series(np.digitize(m, [3.5, 6.0]), index=src.index)
    dec_pos = ct.set_index(["chrom_i", "chrom_j", "bin_i", "bin_j"]); dec_ctl = pl.set_index(["chrom_i", "chrom_j", "bin_i", "bin_j"])
    def attach(pr, dec):
        a = pr[["chrom_i", "chrom_j", "bin_i", "bin_j"]]
        idx = pd.MultiIndex.from_frame(a)   # project_to_cpg_pairs keeps (chrom_i, bin_i) order of the bin pair
        m = dec.reindex(idx)
        return ((m["tcov_decile_i"].to_numpy() + m["tcov_decile_j"].to_numpy()) / 2)
    pos["tcov_mean_decile"] = attach(pos, dec_pos); ctl["tcov_mean_decile"] = attach(ctl, dec_ctl)
    rep["positive_cpg_pairs_before_matching"] = len(pos); rep["control_pool_cpg_pairs"] = len(ctl)
    df = pd.concat([pos, ctl], ignore_index=True)
    cp = {}
    for a, b in zip(df.chrom_i, df.chrom_j):
        pass
    cpk = (df.chrom_i + "|" + df.chrom_j)          # chrom_i<chrom_j in contact tables (CHROMS order); same for pool
    cpid = pd.factorize(cpk, sort=True)[0]
    g = np.digitize(df.tcov_mean_decile.to_numpy(float), [3.5, 6.0])
    strat = (cpid * 4 + g).astype(np.int64)
    df = _canon(df)
    ip, ic = _match(df, strat, "tcov_mean_decile", 2)
    out = _finish(df, ip, ic, folds, sysn)
    out["distance"] = np.nan
    fi = out["chrom_i"].map(folds).to_numpy(); fj = out["chrom_j"].map(folds).to_numpy()
    out["fold"] = SP.pair_split_by_chrom(out, folds, policy="drop").to_numpy()
    P_ = out[out.label == 1]
    npos = len(P_); proj = len(pos)
    rep.update(n_positives_matched=npos, n_controls_matched=int((out.label == 0).sum()), n_positives_unmatched_dropped=proj - npos,
               drop_rate_unmatched=float(1 - npos / proj),
               n_matched_pairs_straddling_folds_dropped_policy_drop=int((P_.fold == -1).sum()),
               n_positives_analysis_eligible=int((P_.fold >= 0).sum()), n_controls_analysis_eligible=int(((out.label == 0) & (out.fold >= 0)).sum()),
               drop_rate_total_vs_projected=float(1 - (P_.fold >= 0).sum() / proj),
               n_chrom_pairs_with_contacts=int(pos.assign(k=pos.chrom_i + "|" + pos.chrom_j).k.nunique()),
               n_chrom_pairs_with_match=int(P_.assign(k=P_.chrom_i + "|" + P_.chrom_j).k.nunique()),
               per_fold_n_pos_eligible={int(k): int(v) for k, v in P_[P_.fold >= 0].groupby("fold").size().items()},
               balance={"tcov_mean_decile": {"smd_before": _smd(pos.tcov_mean_decile, ctl.tcov_mean_decile),
                                                 "smd_after": _smd(P_.sort_values("match_id").tcov_mean_decile, out[out.label == 0].sort_values("match_id").tcov_mean_decile)}},
               drop_rate_note="unmatched = no non-contact pool CpG pair available in same chromosome pair x coverage group (pool is availability-limited)")
    # bin-pair-level accounting (unit-of-sample view)
    rep["contact_bin_pairs_with_any_matched_pair"] = int(P_.bin_pair_id.nunique())
    out = out.sort_values(["label", "match_id"], ascending=[False, True]).reset_index(drop=True)
    out[COLS].to_parquet(OUT / f"frozen_pairs_{TAG[sysn]}_inter1Mb_seed17.parquet", index=False, compression="zstd")
    json.dump(rep, open(OUT / f"frozen_pairs_{TAG[sysn]}_inter1Mb_seed17.report.json", "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != "folds"}, indent=1, default=str))

# ------------------------------------------------------------------ compartments
def compartments():
    U = P4.universe(); folds = fold_map(U); rep = {}
    for sysn in P4.SYSTEMS:
        t = pd.read_parquet(OUT / f"{sysn}_cpg_compartment_100000.parquet")
        t["fold"] = t["chrom"].map(folds).astype("Int64")
        t.to_parquet(OUT / f"frozen_compartments_{TAG[sysn]}_100kb.parquet", index=False, compression="zstd")
        has = t.E1.notna()
        rep[sysn] = {"role": ROLE[sysn], "n_cpg_universe": len(t), "n_cpg_with_E1": int(has.sum()), "n_cpg_without_E1": int((~has).sum()),
                         "n_A": int((t.compartment == "A").sum()), "n_B": int((t.compartment == "B").sum()),
                         "per_fold_n_with_E1": {int(k): int(v) for k, v in t[has].groupby("fold").size().items()},
                         "E1_oriented_by": "Spearman(E1, hg38 GC)>0 per chromosome (A = positive)"}
    json.dump(rep, open(OUT / "frozen_compartments.report.json", "w"), indent=2)
    print(json.dumps(rep, indent=1))

# ------------------------------------------------------------------ FROZEN.json
def freeze():
    files = sorted(list(OUT.glob("frozen_*.parquet")) + list(OUT.glob("frozen_*.report.json")) + [OUT / "assembly_validation.json"])
    sha = {f.name: P4.sha256(f) for f in files}
    reps = {f.name: json.load(open(f)) for f in OUT.glob("frozen_pairs_*.report.json")}
    cnt = {}
    for k, r in reps.items():
        pos = r.get("n_positives", r.get("n_positives_analysis_eligible")); ctl = r.get("n_controls", r.get("n_controls_analysis_eligible"))
        cnt[k.replace(".report.json", ".parquet")] = {"role": r["role"], "n_positives": pos, "n_controls": ctl, "n_positives_unmatched_dropped": r["n_positives_unmatched_dropped"],
            "drop_rate": r.get("drop_rate_total_vs_projected", r.get("drop_rate_vs_projected_positives")), "per_fold": r.get("per_fold_n_pos", r.get("per_fold_n_pos_eligible"))}
    comp = json.load(open(OUT / "frozen_compartments.report.json"))
    fr = {
        "statement": "Produced before any embedding metric was computed. No embeddings, representation outputs or run results were read; matching used only pair distance, chromosome (pair), bin coverage"
                  " and (when present) non-embedding per-CpG covariates.",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit_of_repo_head": __import__("subprocess").run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "roles": {"primary": "H1 Micro-C 4DNFI9GMP2J8 (intra 10 kb pairs; 100 kb E1)", "sensitivity": "HFFc6 Micro-C 4DNFI9FVHJZQ", "secondary": "GM12878 Hi-C 4DNFIXP4QG5B (compartments only)",
                   "exploratory": "H1 inter 1 Mb pairs"},
        "primary_metric": "AUROC contact vs matched non-contact with cosine similarity as pair score; paired delta-cosine reported as secondary effect size",
        "params": {"seed": SEED, "K_per_bin_pair": K, "n_folds": N_FOLDS, "fold_rule": "splits.chromosome_blocked_folds(chroms with CpGs, 5, 17)", "chrom_folds": fold_map(P4.universe()),
                    "universe_n": 408399, "intra_thresholds": {"dist": "20kb-2Mb", "raw_min": 8, "obs_exp_min": 2.0, "noncontact_obs_exp_max": 0.5, "cov_quantile": 0.25},
                    "inter_thresholds": {"raw_min": 50, "obs_exp_min": 2.0}, "distance_bins": "80 log-spaced edges 1e4..2.02e6 bp", "matching": "1:1 without replacement, exact chrom x distance bin (x covariate terciles), rank pairing",
                    "inter_fold_policy": "splits.pair_split_by_chrom policy='drop' (straddling pairs fold=-1)",
                    "covariates_file_present_at_freeze": COVFILE.exists()},
        "pairs": cnt, "compartments": comp, "sha256": sha}
    json.dump(fr, open(OUT / "FROZEN.json", "w"), indent=2, default=str)
    print(json.dumps(cnt, indent=1, default=str))

if __name__ == "__main__":
    c = sys.argv[1]; a = sys.argv[2] if len(sys.argv) > 2 else None
    {"cov": cov, "intra": intra, "inter": inter, "compartments": lambda s=None: compartments(), "freeze": lambda s=None: freeze()}[c](a)
