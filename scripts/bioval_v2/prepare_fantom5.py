#!/usr/bin/env python
# ruff: noqa: DTZ011, EXE001, F841  (one-shot data-prep script; style-only, behaviour unchanged)
"""Prepare the `regulatory_activity` axis (FANTOM5 enhancer CAGE activity) for biological validation v2.

CPU only; never touches representation embeddings. See docs/bioval_v2_prep/regulatory_activity_PREP.md.

Outputs (data/derived/bioval_v2/regulatory_activity/):
  enhancers.parquet                 enhancer coordinates (hg38 native) + n_samples_expressed
  activity_sample_log2tpm.npy       float16 [n_enh, n_lib]  log2(TPM+1)
  activity_sample_index.parquet     library metadata (cnhs, name, category, group_label, ...)
  activity_collapsed_log2tpm.npy    float32 [n_enh, n_group] mean log2(TPM+1) over libraries
  activity_collapsed_index.parquet  group metadata
  cpg_enhancer_map.parquet          one row per benchmark CpG (408,399)
  audit.json, audit_per_chrom.csv, audit_expressed_thresholds.csv, MANIFEST.json

Importable helper: enhancer_profile_similarity(pairs).
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RAW = ROOT / "data/external/bioval_v2/regulatory_activity"
OUT = ROOT / "data/derived/bioval_v2/regulatory_activity"

F_BED = "F5.hg38.enhancers.bed.gz"
F_TPM = "F5.hg38.enhancers.expression.tpm.matrix.gz"
F_NAMES = "Human.sample_name2library_id.txt"
F_SDRF = "HumanSamples2.0.sdrf.xlsx"
F_CAGE = "hg38_fair+new_CAGE_peaks_phase1and2.bed.gz"
F_GTF = "gencode.v50.annotation.gtf.gz"

# pre-declared (see report section 0)
TPM_EXPRESSED = 1.0
INFORMATIVE_N = 5
N_THRESHOLDS = (1, 3, 5, 10, 20)
WINDOWS = (500, 1000)
BG_DIST = 5000
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

URLS = {
    F_BED: "https://fantom.gsc.riken.jp/5/datafiles/reprocessed/hg38_latest/extra/enhancer/" + F_BED,
    F_TPM: "https://fantom.gsc.riken.jp/5/datafiles/reprocessed/hg38_latest/extra/enhancer/" + F_TPM,
    "enhancer_hg38_00README.txt": "https://fantom.gsc.riken.jp/5/datafiles/reprocessed/hg38_latest/extra/enhancer/00README.txt",
    F_CAGE: "https://fantom.gsc.riken.jp/5/datafiles/reprocessed/hg38_latest/extra/CAGE_peaks/" + F_CAGE,
    F_NAMES: "https://fantom.gsc.riken.jp/5/datafiles/latest/extra/Enhancers/" + F_NAMES,
    F_SDRF: "https://fantom.gsc.riken.jp/5/datafiles/reprocessed/hg38_latest/basic/" + F_SDRF,
    F_GTF: "http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/" + F_GTF,
}


# ----------------------------------------------------------------------------- loading
def load_enhancers() -> pd.DataFrame:
    cols = ["chrom", "start0", "end", "enhancer_id", "score", "strand", "thickStart", "thickEnd",
            "rgb", "nblocks", "blocksizes", "blockstarts"]
    b = pd.read_csv(RAW / F_BED, sep="\t", header=None, names=cols, compression="gzip")
    b = b[b.chrom.isin(CHROMS)].reset_index(drop=True)
    b["mid0"] = b["thickStart"].astype(np.int64)  # 0-based inferred midpoint
    b["width"] = b["end"] - b["start0"]
    return b[["chrom", "start0", "end", "enhancer_id", "mid0", "width", "score"]]


def _strip_rep(name: str, category: str) -> str:
    s = name.strip()
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r",?\s*\((?:fresh|revived|thawed|[^()]*biol_rep\d+[^()]*)\)\s*$", "", s) if category == "cell lines" else s
        s = re.sub(r",?\s*(?:donor\d+|pool\d*|biol_rep\d+|rep\d+|\(donor\d+\))\s*$", "", s, flags=re.IGNORECASE)
    return s


def load_sample_meta(cnhs_cols) -> pd.DataFrame:
    n = pd.read_csv(RAW / F_NAMES, sep="\t", header=None, names=["name", "cnhs"])
    s = pd.read_excel(RAW / F_SDRF)
    s = s.rename(columns={"Charateristics [description]": "name", "Characteristics [Category]": "category",
                          "Charateristics [ff_ontology]": "ff_ontology", "Source Name": "source_name",
                          "Characteristics [Sex]": "sex"})
    s = s[["name", "category", "ff_ontology", "source_name"]].drop_duplicates("name")
    m = pd.DataFrame({"cnhs": list(cnhs_cols)}).merge(n, on="cnhs", how="left").merge(s, on="name", how="left")
    m["category"] = m["category"].fillna("unmapped")
    m["group_label"] = [(_strip_rep(a, c) if isinstance(a, str) else None) for a, c in zip(m["name"], m["category"])]
    return m


# ----------------------------------------------------------------------------- activity
def build_activity(enh: pd.DataFrame):
    mat = pd.read_csv(RAW / F_TPM, sep="\t", index_col=0, compression="gzip", float_precision="round_trip")
    mat = mat.loc[enh["enhancer_id"]]  # keeps BED order, asserts same ID set
    tpm = mat.to_numpy(np.float32)
    meta = load_sample_meta(mat.columns)
    log = np.log2(tpm + 1.0, dtype=np.float32)
    expressed = tpm >= TPM_EXPRESSED
    enh = enh.copy()
    enh["n_samples_expressed"] = expressed.sum(1).astype(np.int32)
    enh["build"] = "GRCh38"
    enh["pos1_start"] = enh["start0"] + 1
    enh["pos1_end"] = enh["end"]
    # collapsed level: categories tissues / primary cells / cell lines only
    keep = meta["category"].isin(["tissues", "primary cells", "cell lines"]) & meta["group_label"].notna()
    meta["collapsed_key"] = np.where(keep, meta["category"] + "|" + meta["group_label"].astype(str), None)
    keys = pd.Index(sorted(set(meta["collapsed_key"].dropna())))
    coll = np.zeros((len(enh), len(keys)), np.float32)
    counts = np.zeros(len(keys), int)
    for j, k in enumerate(keys):
        idx = np.where(meta["collapsed_key"].to_numpy() == k)[0]
        coll[:, j] = log[:, idx].mean(1)
        counts[j] = len(idx)
    cidx = pd.DataFrame({"collapsed_key": keys})
    cidx["category"] = cidx.collapsed_key.str.split("|", n=1).str[0]
    cidx["group_label"] = cidx.collapsed_key.str.split("|", n=1).str[1]
    cidx["n_libraries"] = counts
    return enh, log, meta, coll, cidx, expressed


# ----------------------------------------------------------------------------- CpG mapping
def _nearest_abs(sorted_pts: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(sorted_pts) == 0:
        return np.full(len(x), np.iinfo(np.int64).max // 4, np.int64)
    i = np.searchsorted(sorted_pts, x)
    lo = sorted_pts[np.clip(i - 1, 0, len(sorted_pts) - 1)]
    hi = sorted_pts[np.clip(i, 0, len(sorted_pts) - 1)]
    return np.minimum(np.abs(x - lo), np.abs(hi - x)).astype(np.int64)


def load_tss_gencode() -> pd.DataFrame:
    rows = []
    with gzip.open(RAW / F_GTF, "rt") as f:
        for ln in f:
            if ln[0] == "#":
                continue
            p = ln.split("\t", 8)
            if p[2] != "transcript":
                continue
            tss0 = int(p[3]) - 1 if p[6] == "+" else int(p[4]) - 1
            rows.append((p[0], tss0))
    return pd.DataFrame(rows, columns=["chrom", "tss0"]).drop_duplicates()


def load_tss_cage() -> pd.DataFrame:
    c = pd.read_csv(RAW / F_CAGE, sep="\t", header=None, usecols=[0, 6], names=["chrom", "tss0"],
                    compression="gzip", dtype=str)
    c = c[c.chrom.isin(CHROMS)].copy()
    c["tss0"] = c["tss0"].astype(np.int64)
    return c


def map_cpgs(uni: pd.DataFrame, enh: pd.DataFrame, tss_gc: pd.DataFrame, tss_cage: pd.DataFrame) -> pd.DataFrame:
    out = []
    for chrom in CHROMS:
        u = uni[uni.chrom == chrom]
        if u.empty:
            continue
        p0 = (u["pos"].to_numpy() - 1).astype(np.int64)  # 0-based cytosine
        e = enh[enh.chrom == chrom].sort_values("start0").reset_index(drop=True)
        st, en, mid = e.start0.to_numpy(), e.end.to_numpy(), e.mid0.to_numpy()
        eid = e.enhancer_id.to_numpy()
        n = len(u)
        # nearest enhancer interval via candidate scan over K starts left of p0 + K right
        K = 6
        i = np.searchsorted(st, p0, side="right")  # first start > p0
        best_d = np.full(n, 1 << 60, np.int64)
        best_j = np.full(n, -1, np.int64)
        contained = np.zeros(n, bool)
        for off in range(-K, K):
            j = i + off
            ok = (j >= 0) & (j < len(st))
            jj = np.clip(j, 0, max(len(st) - 1, 0))
            d = np.where(ok, np.maximum(np.maximum(st[jj] - p0, p0 - (en[jj] - 1)), 0), 1 << 60)
            dm = np.where(ok, np.abs(mid[jj] - p0), 1 << 60)
            better = (d < best_d) | ((d == best_d) & (dm < np.where(best_j >= 0, np.abs(mid[np.clip(best_j, 0, None)] - p0), 1 << 60)))
            best_d = np.where(better, d, best_d)
            best_j = np.where(better, j, best_j)
        contained = best_d == 0
        # windows relative to midpoints (independent of nearest-interval choice)
        dmid = _nearest_abs(np.sort(mid), p0)
        bj = np.clip(best_j, 0, max(len(st) - 1, 0))
        res = pd.DataFrame({
            "cpg_idx": u["cpg_idx"].to_numpy(), "chrom": chrom, "pos": u["pos"].to_numpy(),
            "enhancer_id": np.where(best_j >= 0, eid[bj], None) if len(st) else None,
            "dist_to_enhancer_bp": best_d, "dist_to_enhancer_mid_bp": dmid,
            "in_enhancer": contained,
            "in_window500": dmid <= WINDOWS[0], "in_window1000": dmid <= WINDOWS[1],
            "dist_to_gencode_tss_bp": _nearest_abs(np.sort(tss_gc.loc[tss_gc.chrom == chrom, "tss0"].to_numpy()), p0),
            "dist_to_cage_peak_tss_bp": _nearest_abs(np.sort(tss_cage.loc[tss_cage.chrom == chrom, "tss0"].to_numpy()), p0),
        })
        # enhancer_id only meaningful when within 1 kb of the interval/midpoint; else keep nearest-interval id
        out.append(res)
    r = pd.concat(out, ignore_index=True)
    r["bg_enh5k"] = r["dist_to_enhancer_bp"] >= BG_DIST
    r["bg_enh5k_tss5k"] = r["bg_enh5k"] & (r["dist_to_gencode_tss_bp"] >= BG_DIST)
    return r.sort_values("cpg_idx").reset_index(drop=True)


# ----------------------------------------------------------------------------- helper
_CACHE = {}


def enhancer_profile_similarity(pairs, *, level="collapsed", activity_dir=None, min_expressed=INFORMATIVE_N,
                                categories=None, min_groups=3, enhancer_ids=False):
    """Pearson correlation of enhancer log2(TPM+1) tissue-activity profiles for CpG pairs.

    pairs: array-like / DataFrame with two columns. By default entries are benchmark `cpg_idx` values, resolved to
    the enhancer in `cpg_enhancer_map.parquet` for which `in_enhancer` is True. With enhancer_ids=True the entries are
    enhancer IDs (e.g. 'chr1:1000-1300').
    level: 'collapsed' (tissue/primary-cell/cell-line groups; default) or 'sample' (all libraries).
    categories: optional subset of collapsed categories (e.g. ['tissues', 'primary cells']).
    Returns float64 array (n_pairs,), NaN when either CpG is not in an enhancer, an enhancer has
    n_samples_expressed < min_expressed, or a profile has zero variance. Computes nothing from embeddings.
    """
    d = Path(activity_dir) if activity_dir else OUT
    key = (str(d), level, tuple(categories) if categories else None)
    if key not in _CACHE:
        enh = pd.read_parquet(d / "enhancers.parquet")
        if level == "sample":
            A = np.load(d / "activity_sample_log2tpm.npy", mmap_mode="r")
            cols = np.arange(A.shape[1])
        else:
            A = np.load(d / "activity_collapsed_log2tpm.npy", mmap_mode="r")
            ci = pd.read_parquet(d / "activity_collapsed_index.parquet")
            cols = np.where(ci.category.isin(categories))[0] if categories else np.arange(A.shape[1])
        row = pd.Series(np.arange(len(enh)), index=enh["enhancer_id"].to_numpy())
        cm = pd.read_parquet(d / "cpg_enhancer_map.parquet", columns=["cpg_idx", "enhancer_id", "in_enhancer"])
        cm = cm[cm.in_enhancer].set_index("cpg_idx")["enhancer_id"]
        _CACHE[key] = (enh, A, cols, row, cm)
    enh, A, cols, row, cm = _CACHE[key]
    P = np.asarray(pairs)
    ea = P[:, 0] if enhancer_ids else pd.Series(P[:, 0]).map(cm).to_numpy()
    eb = P[:, 1] if enhancer_ids else pd.Series(P[:, 1]).map(cm).to_numpy()
    ia = pd.Series(ea).map(row).to_numpy()
    ib = pd.Series(eb).map(row).to_numpy()
    nexp = enh["n_samples_expressed"].to_numpy()
    out = np.full(len(P), np.nan)
    ok = ~(pd.isna(ia) | pd.isna(ib))
    ia_i = np.where(ok, ia, 0).astype(int)
    ib_i = np.where(ok, ib, 0).astype(int)
    ok &= (nexp[ia_i] >= min_expressed) & (nexp[ib_i] >= min_expressed) & (len(cols) >= min_groups)
    sel = np.where(ok)[0]
    for s in range(0, len(sel), 20000):
        t = sel[s:s + 20000]
        X = np.asarray(A[ia_i[t]][:, cols], dtype=np.float64)
        Y = np.asarray(A[ib_i[t]][:, cols], dtype=np.float64)
        X -= X.mean(1, keepdims=True)
        Y -= Y.mean(1, keepdims=True)
        den = np.sqrt((X * X).sum(1) * (Y * Y).sum(1))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[t] = np.where(den > 0, (X * Y).sum(1) / den, np.nan)
    return out


# ----------------------------------------------------------------------------- audit + main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gtf", action="store_true", help="use only CAGE-peak TSS if GENCODE GTF missing")
    args = ap.parse_args()
    from cpg_repr_benchmark.biological_validation_v2.coordinates import load_benchmark_universe
    from cpg_repr_benchmark.biological_validation_v2.provenance import (
        SourceRecord,
        sha256_file,
        write_manifest,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    uni = load_benchmark_universe()
    enh, log, meta, coll, cidx, expressed = build_activity(load_enhancers())
    enh.to_parquet(OUT / "enhancers.parquet", index=False)
    np.save(OUT / "activity_sample_log2tpm.npy", log.astype(np.float16))
    meta.to_parquet(OUT / "activity_sample_index.parquet", index=False)
    np.save(OUT / "activity_collapsed_log2tpm.npy", coll)
    cidx.to_parquet(OUT / "activity_collapsed_index.parquet", index=False)

    tss_cage = load_tss_cage()
    tss_gc = load_tss_gencode() if not args.skip_gtf else pd.DataFrame({"chrom": [], "tss0": []})
    m = map_cpgs(uni, enh, tss_gc, tss_cage)
    m.to_parquet(OUT / "cpg_enhancer_map.parquet", index=False)

    # audit
    nexp = enh.set_index("enhancer_id")["n_samples_expressed"]
    audit = {"n_universe": len(m), "n_enhancers": len(enh), "enhancer_width": enh.width.describe().to_dict(),
             "n_libraries": int(log.shape[1]), "n_collapsed_groups": int(coll.shape[1]),
             "library_category_counts": meta.category.value_counts().to_dict(),
             "collapsed_category_counts": cidx.category.value_counts().to_dict(),
             "n_libraries_unmapped_to_sdrf": int((meta.category == "unmapped").sum()),
             "tss_gencode_n": len(tss_gc), "tss_cage_n": len(tss_cage),
             "sparsity_fraction_zero_tpm_entries": float((~(log > 0)).mean()),
             "sparsity_fraction_expressed_ge1_entries": float(expressed.mean()),
             "enh_n_samples_expressed_quantiles": nexp.quantile([0, .1, .25, .5, .75, .9, 1]).to_dict(),
             "enh_never_expressed": int((nexp == 0).sum())}
    for name in ["in_enhancer", "in_window500", "in_window1000", "bg_enh5k", "bg_enh5k_tss5k"]:
        audit[f"n_{name}"] = int(m[name].sum())
        audit[f"frac_{name}"] = float(m[name].mean())
    inenh = m[m.in_enhancer]
    audit["n_enhancers_covered_primary"] = int(inenh.enhancer_id.nunique())
    w1 = m[m.in_window500]; w2 = m[m.in_window1000]
    # enhancers with >=1 CpG within window of midpoint: recompute per chrom
    cov500, cov1000 = set(), set()
    for chrom, g in enh.groupby("chrom"):
        pos0 = np.sort(m.loc[m.chrom == chrom, "pos"].to_numpy() - 1)
        mid = g.mid0.to_numpy()
        for W, S in ((500, cov500), (1000, cov1000)):
            lo = np.searchsorted(pos0, mid - W, "left"); hi = np.searchsorted(pos0, mid + W, "right")
            S.update(g.enhancer_id.to_numpy()[hi > lo])
    audit["n_enhancers_covered_w500"] = len(cov500)
    audit["n_enhancers_covered_w1000"] = len(cov1000)
    thr = []
    for N in N_THRESHOLDS:
        inf = m.enhancer_id.map(nexp) >= N
        thr.append({"min_samples_expressed": N, "n_enhancers": int((nexp >= N).sum()),
                    "n_cpgs_in_enhancer_informative": int((m.in_enhancer & inf).sum()),
                    "n_enhancers_covered_informative": int(m.loc[m.in_enhancer & inf, "enhancer_id"].nunique()),
                    "n_cpgs_w500_informative": int((m.in_window500 & inf).sum()),
                    "n_cpgs_w1000_informative": int((m.in_window1000 & inf).sum())})
    thr = pd.DataFrame(thr)
    thr.to_csv(OUT / "audit_expressed_thresholds.csv", index=False)
    per = m.groupby("chrom").agg(n_cpg=("cpg_idx", "size"), n_in_enhancer=("in_enhancer", "sum"),
                                 n_w500=("in_window500", "sum"), n_w1000=("in_window1000", "sum"),
                                 n_bg_enh5k=("bg_enh5k", "sum"), n_bg_enh5k_tss5k=("bg_enh5k_tss5k", "sum"),
                                 n_enh_covered=("enhancer_id", lambda s: 0)).reset_index()
    per["n_enh_covered"] = per.chrom.map(inenh.groupby("chrom").enhancer_id.nunique()).fillna(0).astype(int)
    per["n_enh_total"] = per.chrom.map(enh.groupby("chrom").size()).fillna(0).astype(int)
    per["frac_in_enhancer"] = per.n_in_enhancer / per.n_cpg
    per = per.set_index("chrom").loc[[c for c in CHROMS if c in set(per.chrom)]].reset_index()
    per.to_csv(OUT / "audit_per_chrom.csv", index=False)
    audit["informative_thresholds"] = thr.to_dict(orient="records")
    audit["dist_to_enhancer_quantiles"] = m.dist_to_enhancer_bp.quantile([.01, .1, .5, .9]).to_dict()
    audit["dist_to_gencode_tss_quantiles"] = m.dist_to_gencode_tss_bp.quantile([.1, .5, .9]).to_dict()
    audit["dist_to_cage_peak_quantiles"] = m.dist_to_cage_peak_tss_bp.quantile([.1, .5, .9]).to_dict()
    # in-enhancer vs not: TSS distance medians (covariate imbalance, descriptive only)
    audit["median_gencode_tss_dist_in_enh"] = float(inenh.dist_to_gencode_tss_bp.median()) if len(inenh) else None
    audit["median_gencode_tss_dist_not_in_enh"] = float(m.loc[~m.in_enhancer, "dist_to_gencode_tss_bp"].median())
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    today = date.today().isoformat()
    srcs = []
    for fn, url in URLS.items():
        p = RAW / fn
        if p.exists():
            srcs.append(SourceRecord.from_file(p, name=fn, url=url, retrieval_date=today,
                                               genome_build_in="GRCh38", genome_build_out="GRCh38",
                                               extra={"native_hg38": True, "liftover": None}))
    for fn in ["Enhancers_latest_00_readme.txt", "CAGE_peaks_00_readme.txt", "ff-phase2-170801.obo.txt", "00MD5SUM.txt"]:
        p = RAW / fn
        if p.exists():
            srcs.append(SourceRecord.from_file(p, name=fn, retrieval_date=today, extra={"auxiliary": True}))
    import numpy
    import openpyxl
    import pandas
    params = {"enhancer_set": "FANTOM5 hg38 reprocessed permissive-equivalent (63,285)", "tpm_expressed": TPM_EXPRESSED,
              "informative_min_samples": INFORMATIVE_N, "windows_bp": list(WINDOWS), "bg_dist_bp": BG_DIST,
              "activity": "log2(TPM+1), RLE TPM as provided", "collapse": "mean log2(TPM+1) per replicate-stripped label",
              "derived_files": {f.name: {"sha256": sha256_file(f)} for f in sorted(OUT.glob("*")) if f.name != "MANIFEST.json"},
              "row_counts": {"enhancers": len(enh), "libraries": int(log.shape[1]), "collapsed_groups": int(coll.shape[1]),
                             "cpg_map": len(m), "tss_gencode": len(tss_gc), "tss_cage": len(tss_cage)},
              "tool_versions": {"python": sys.version.split()[0], "pandas": pandas.__version__, "numpy": numpy.__version__,
                                "openpyxl": openpyxl.__version__},
              "liftover": "none (all sources natively GRCh38)"}
    write_manifest(OUT / "MANIFEST.json", "regulatory_activity", srcs,
                   {k: v for k, v in audit.items() if not isinstance(v, (dict, list))}, params)
    print(json.dumps({k: v for k, v in audit.items() if not isinstance(v, (dict, list))}, indent=1))
    print(thr.to_string()); print(per.to_string())


if __name__ == "__main__":
    main()
