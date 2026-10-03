# ruff: noqa: F841, SIM115  (one-shot data-prep script; style-only, behaviour unchanged)
"""Marker reconciliation + provenance strength for the Loyfer atlas.
(1) Reconciles '164 U25 marker CpGs in universe' vs '157 in present groups' (loyfer_markers_universe.parquet).
(2) Compares the GitHub hg38 marker files (nloyfer/UXM_deconv supplemental, commit 8d0bb45) with the PAPER tables (Nature Suppl. Table S4A=top25,
    S4B=top250; hg19 only): hg19 GitHub files vs paper tables (exact), and hg38 GitHub blocks vs paper blocks lifted hg19->hg38
    (UCSC hg19ToHg38 chain via pyliftover; same-target overlapping block).
(3) Builds paper-derived hg38 markers (S4A/S4B single-type rows lifted to hg38, expanded to universe CpGs by position) as a STRONG-provenance alternative.
Outputs: loyfer_markers_reconciled.parquet, marker_reconciliation.json, marker_provenance_verdict.json, loyfer_markers_paper_lifted_blocks.parquet"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from loyfer_common import *
from pyliftover import LiftOver

XLSX = META / "Loyfer2023_Nature_SuppTables_MOESM4.xlsx"
CHAIN = META / "hg19ToHg38.over.chain.gz"
COMMIT = "8d0bb456742ece21802bc8e651e080869c67c944"
GH = "https://github.com/nloyfer/UXM_deconv/blob/" + COMMIT + "/supplemental/"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
MEGA = "Megakaryocytes"

def lift_blocks(df, lo):
    """df: chr,start,end (hg19; start=1-based first C, end=1-based last G). -> lifted lo/hi (first C, last C) or NaN."""
    L = []
    for r in df.itertuples():
        a = lo.convert_coordinate(r.chr, int(r.start) - 1); b = lo.convert_coordinate(r.chr, int(r.end) - 2)
        if not a or not b or a[0][0] != b[0][0] or a[0][2] != "+" or b[0][2] != "+":
            L.append((None, np.nan, np.nan)); continue
        p, q = a[0][1] + 1, b[0][1] + 1
        lo_, hi_ = min(p, q), max(p, q)
        if hi_ - lo_ > 2 * (int(r.end) - int(r.start)) + 100: L.append((None, np.nan, np.nan)); continue
        L.append((a[0][0], lo_, hi_))
    return pd.DataFrame(L, columns=["chr38", "lo38", "hi38"], index=df.index)

def paper_tables():
    out = {}
    for sh, tag in [("Table S4A", "U25"), ("Table S4B", "U250")]:
        d = pd.read_excel(XLSX, sheet_name=sh, header=2)
        d = d.rename(columns={"Type": "target"}); d["table"] = tag
        out[tag] = d[["target", "chr", "start", "end", "startCpG", "endCpG", "Number of CpGs", "Target meth. ", "table"]] if "Target meth. " in d.columns else d
    return out

def main():
    mu = pd.read_parquet(OUT / "loyfer_markers_universe.parquet")
    u = load_universe(); _cid, gpos, chroms = load_genome_index(); ch = np.array(chroms)
    # ---------------- 1. reconcile 164 vs 157 ----------------
    a = mu[mu.in_published_atlas_U25]
    rec = {"U25_rows_in_universe": len(a), "U25_unique_cpg": int(a.cpg_idx.nunique())}
    with_mega = a.groupby("cell_type").size()
    rec["U25_rows_megakaryocytes"] = int(with_mega.get(MEGA, 0))
    rec["U25_rows_present_groups"] = len(a[a.cell_type != MEGA])
    rec["U25_unique_cpg_present_groups"] = int(a[a.cell_type != MEGA].cpg_idx.nunique())
    dup = a.groupby("cpg_idx").cell_type.nunique(); rec["U25_cpgs_in_multiple_celltypes"] = int((dup > 1).sum())
    rec["U25_rows_per_celltype_in_universe"] = {k: int(v) for k, v in with_mega.items()}
    rec["explanation"] = (f"{rec['U25_rows_in_universe']} U25 universe marker rows include {rec['U25_rows_megakaryocytes']} rows of Megakaryocytes (atlas column with no local samples, "
                          f"absent from the 39-group paper table S1/S3); {rec['U25_rows_in_universe']}-{rec['U25_rows_megakaryocytes']}={rec['U25_rows_present_groups']} rows in the 39 present groups.")
    # ---------------- 2. paper vs GitHub ----------------
    pt = paper_tables()
    ver = {}
    g19 = pd.read_csv(META / "hg19_github/Atlas.U25.l4.hg19.full.tsv", sep="\t"); g19 = g19[g19.target != "target"]
    k = lambda d, t: set(zip(d[t], d.chr, d.start.astype(int), d.end.astype(int)))
    s4a = pt["U25"]; s4a_single = s4a[~s4a.target.str.contains(":")]
    A, G = k(s4a_single, "target"), k(g19, "target")
    ver["hg19_github_U25_vs_paper_S4A_single_type"] = {"paper_single_type_rows": len(A), "github_rows": len(G), "identical": len(A & G), "paper_only": len(A - G),
                                                           "github_only": len(G - A), "github_only_types": sorted({t for t, *_ in G - A})}
    s4b = pt["U250"]; s4b_single = s4b[~s4b.target.str.contains(":")]
    m19 = pd.read_csv(META / "hg19_github/Markers.U250.hg19.tsv", sep="\t").rename(columns={"#chr": "chr"})
    B, M = k(s4b_single, "target"), k(m19, "target")
    ver["hg19_github_U250_vs_paper_S4B_single_type"] = {"paper_single_type_rows": len(B), "github_rows": len(M), "identical": len(B & M), "paper_only": len(B - M),
                                                            "github_only": len(M - B), "github_only_types": sorted({t for t, *_ in M - B})}
    # hg38 github vs lifted paper
    lo = LiftOver(str(CHAIN))
    h38_25 = pd.read_csv(META / "Atlas.U25.l4.hg38.full.tsv", sep="\t"); h38_25 = h38_25[h38_25.target != "target"].copy()
    h38_250 = pd.read_csv(META / "Markers.U250.hg38.tsv", sep="\t").rename(columns={"#chr": "chr", "region": "name"})
    for d in (h38_25, h38_250): d["start"] = d.start.astype(int); d["end"] = d.end.astype(int)
    paper_lift = {}
    for tag, src in [("U25", s4a_single), ("U250", s4b_single)]:
        src = src.copy(); src[["start", "end"]] = src[["start", "end"]].astype(int)
        src = src.join(lift_blocks(src, lo)); paper_lift[tag] = src
    def hg38_overlap(h38, pl):
        gp = {t: d.dropna(subset=["lo38"]) for t, d in pl.groupby("target")}; flag = []
        for r in h38.itertuples():
            d = gp.get(r.target)
            flag.append(False if d is None else bool(((d.chr38 == r.chr) & (d.lo38 <= r.end) & (d.hi38 >= r.start)).any()))
        return np.array(flag)
    h38_25["paper_overlap"] = hg38_overlap(h38_25, paper_lift["U25"]); h38_250["paper_overlap"] = hg38_overlap(h38_250, paper_lift["U250"])
    for tag, d, pl in [("U25", h38_25, paper_lift["U25"]), ("U250", h38_250, paper_lift["U250"])]:
        ver[f"hg38_github_{tag}_vs_paper_lifted"] = {
            "hg38_blocks": len(d), "hg38_blocks_overlapping_same_type_lifted_paper_block": int(d.paper_overlap.sum()), "frac": float(d.paper_overlap.mean()),
            "hg38_blocks_without_paper_counterpart": int((~d.paper_overlap).sum()),
            "hg38_non_megakaryocyte_blocks": int((d.target != MEGA).sum()),
            "hg38_non_mega_overlap_frac": float(d[d.target != MEGA].paper_overlap.mean()),
            "paper_blocks": len(pl), "paper_blocks_lift_failed": int(pl.lo38.isna().sum())}
    # per-type counts for hg38 U25 vs 25 / >=4 CpG
    ver["hg38_U25_blocks_per_type_counts"] = {k_: int(v) for k_, v in h38_25.target.value_counts().items() if v != 25}
    ver["hg38_U25_min_lenCpG"] = int((h38_25.endCpG.astype(int) - h38_25.startCpG.astype(int)).min())
    ver["hg38_U250_min_lenCpG"] = int((h38_250.endCpG.astype(int) - h38_250.startCpG.astype(int)).min())
    ver["hg38_U250_blocks_per_type_not_250"] = {k_: int(v) for k_, v in h38_250.target.value_counts().items() if v != 250}
    ver["paper_S4A_total_rows_incl_combination_markers"] = len(s4a); ver["paper_S4A_single_type"] = len(s4a_single)
    ver["paper_S4B_total_rows_incl_combination_markers"] = len(s4b); ver["paper_S4B_single_type"] = len(s4b_single)
    # ---------------- 3. paper-derived hg38 markers ----------------
    gk = gpos.astype(np.int64)  # genome CpG positions; per chrom contiguous & sorted
    uni = u.copy(); uni["chrom"] = uni.chrom.astype(str)
    uby = {c: d.sort_values("pos") for c, d in uni.groupby("chrom")}
    rows, blocks = [], []
    for tag, pl in paper_lift.items():
        pl = pl.dropna(subset=["lo38"])
        for r in pl.itertuples():
            d = uby.get(r.chr38)
            if d is None: continue
            p = d.pos.to_numpy(); i0, i1 = np.searchsorted(p, r.lo38, "left"), np.searchsorted(p, r.hi38, "right")
            blocks.append({"target": r.target, "table": tag, "chr19": r.chr, "start19": r.start, "end19": r.end, "chr38": r.chr38, "lo38": int(r.lo38), "hi38": int(r.hi38), "n_universe_cpgs": int(i1 - i0)})
            for j in range(i0, i1): rows.append((int(d.cpg_idx.iloc[j]), r.chr38, int(p[j]), r.target, tag))
    pm = pd.DataFrame(rows, columns=["cpg_idx", "chrom", "pos", "cell_type", "source_table"]).drop_duplicates()
    pm["marker_source"] = "paper_Suppl_Table_S4A_S4B_hg19_lifted_to_hg38"; pm["provenance_strength"] = "STRONG_paper_table"
    pd.DataFrame(blocks).to_parquet(OUT / "loyfer_markers_paper_lifted_blocks.parquet", index=False)
    # github hg38 rows: strength per block via overlap
    allm = pd.read_parquet(OUT / "loyfer_markers_universe.parquet")
    ov25 = set(h38_25[h38_25.paper_overlap].name); ov250 = set(h38_250[h38_250.paper_overlap].name)
    gm = allm[["cpg_idx", "chrom", "pos", "cell_type", "block", "in_published_atlas_U25"]].copy()
    gm["source_table"] = np.where(gm.in_published_atlas_U25, "U25", "U250")
    ov = np.where(gm.in_published_atlas_U25, gm.block.isin(ov25), gm.block.isin(ov250))
    gm["marker_source"] = "github_UXM_deconv_hg38@" + COMMIT[:7]
    gm["provenance_strength"] = np.where(gm.cell_type == MEGA, "WEAK_github_only_not_in_paper_(Megakaryocytes)",
                                         np.where(ov, "MODERATE_github_hg38_block_overlaps_lifted_paper_block", "WEAK_github_hg38_only_no_paper_counterpart"))
    gm = gm.drop(columns=["block", "in_published_atlas_U25"])
    rm = pd.concat([gm, pm], ignore_index=True)
    rm["in_present_group"] = rm.cell_type != MEGA
    # U250 rows from github with in_published U25 flagged are U25; duplicates are rows of same cpg from different blocks: keep unique per (cpg,cell_type,source_table,marker_source)
    rm = rm.drop_duplicates(["cpg_idx", "cell_type", "source_table", "marker_source"]).reset_index(drop=True)
    rm["cpg_idx"] = rm.cpg_idx.astype(np.int64); rm["pos"] = rm.pos.astype(np.int64)
    rm.to_parquet(OUT / "loyfer_markers_reconciled.parquet", index=False)
    summ = (rm.groupby(["marker_source", "source_table", "in_present_group"]).agg(rows=("cpg_idx", "size"), unique_cpgs=("cpg_idx", "nunique")).reset_index())
    per = (rm[rm.in_present_group].groupby(["marker_source", "source_table", "cell_type"]).size().unstack([0, 1]).fillna(0).astype(int))
    per.to_csv(OUT / "loyfer_markers_reconciled_per_celltype.tsv", sep="\t")
    rec["reconciled_summary"] = summ.to_dict("records")
    rec["paper_lifted_unique_cpgs_by_table"] = {t: int(pm[pm.source_table == t].cpg_idx.nunique()) for t in ["U25", "U250"]}
    rec["paper_lifted_rows_by_table"] = pm.groupby("source_table").size().to_dict()
    rec["paper_lifted_blocks_with_ge1_universe_cpg"] = int((pd.DataFrame(blocks).n_universe_cpgs > 0).sum())
    json.dump({"reconciliation": rec, "verification": ver}, open(OUT / "marker_reconciliation.json", "w"), indent=1, default=str)
    files = ["Atlas.U25.l4.hg38.full.tsv", "Markers.U250.hg38.tsv", "hg19_github/Atlas.U25.l4.hg19.full.tsv", "hg19_github/Markers.U250.hg19.tsv", "Loyfer2023_Nature_SuppTables_MOESM4.xlsx"]
    v = {
        "verdict": "WEAK", "scope": "hg38 GitHub marker files actually used by the benchmark (Atlas.U25.l4.hg38.full.tsv, Markers.U250.hg38.tsv)",
        "reason": ("Paper tables (Nature Suppl. S4A/S4B) are hg19 only. GitHub hg19 files match the paper rows exactly "
                f"(U25: {ver['hg19_github_U25_vs_paper_S4A_single_type']['identical']}/{ver['hg19_github_U25_vs_paper_S4A_single_type']['paper_single_type_rows']}; "
                f"U250: {ver['hg19_github_U250_vs_paper_S4B_single_type']['identical']}/{ver['hg19_github_U250_vs_paper_S4B_single_type']['paper_single_type_rows']}), plus 25/250 Megakaryocyte rows not in the paper. "
                f"The hg38 files are NOT a liftover of the paper tables: only {ver['hg38_github_U25_vs_paper_lifted']['frac']:.1%} of hg38 U25 blocks and {ver['hg38_github_U250_vs_paper_lifted']['frac']:.1%} of hg38 U250 blocks "
                "overlap a same-type paper block lifted to hg38 (atlas values differ from hg19; consistent with markers being recomputed on hg38). Not verifiable as the paper's published tables."),
        "alternative_set": {"name": "paper S4A/S4B single-type markers lifted hg19->hg38 (UCSC chain, pyliftover), expanded to universe CpGs by position",
                             "provenance_strength": "STRONG for marker identity (published table, 953 U25 / 9288 U250 single-type blocks); lift is ours",
                             "file": "loyfer_markers_reconciled.parquet (marker_source == paper_Suppl_Table_S4A_S4B_hg19_lifted_to_hg38)",
                             "unique_cpgs_in_universe": rec["paper_lifted_unique_cpgs_by_table"]},
        "github_repo": "https://github.com/nloyfer/UXM_deconv", "github_commit": COMMIT, "github_commit_date": "2023-01-04T14:40:21Z (HEAD of main when queried 2026-10-02)",
        "files": {f: {"url": GH + f.split("/")[-1], "sha256": sha256 if False else sha(META / f)} for f in files},
        "unverified": ["exact find_markers thresholds beyond 'l4' (>=4 CpGs) and 25/250 per type", "hg38 marker derivation procedure (not documented in repo)"],
        "user_rule": "if not sufficiently strong: WSBS profile similarity PRIMARY, marker-kNN SECONDARY (applied by the doc agent)"}
    json.dump(v, open(OUT / "marker_provenance_verdict.json", "w"), indent=1)
    print(json.dumps(rec, indent=1, default=str)[:3000]); print(json.dumps(ver, indent=1, default=str)); print(per.sum())
if __name__ == "__main__":
    main()
