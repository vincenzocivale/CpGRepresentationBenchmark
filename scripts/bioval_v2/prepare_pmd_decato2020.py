#!/usr/bin/env python
# ruff: noqa: DTZ011, EXE001  (one-shot data-prep script; style-only, behaviour unchanged)
"""PMD annotation (SEPARATE from RT). Source: Decato et al. 2020 Epigenetics Chromatin, MethPipe `pmd` calls, human native
assembly (hg19; VERIFIED empirically, see data/derived/bioval_v2/pmd_decato2020/BUILD_CHECK.json; explicit hg19->GRCh38 liftover), github.com/bdecato/PMD_Paper_Scripts AllPMDs.tar.bz2 (GPL-3.0 repo).
STATUS: SECONDARY/EXPLORATORY endpoint (derived from methylation).
DEPENDENCY FLAG: PMDs are derived from WGBS methylation -> confounded with methylation-prediction tasks (not with ENCODE inputs).
Output: per-sample CpG membership (int8: 1 in PMD, 0 outside called PMDs on a segmented chrom, -1 not assessable: chrX/chrY/chrM),
plus recurrence summaries. Only samples with called PMDs are shipped by the source (>=5% genome PMD, mean seg >=50kb)."""
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from cpg_repr_benchmark.biological_validation_v2 import coordinates as C

RAW = ROOT / "data/external/bioval_v2/pmd_decato2020"; OUT = ROOT / "data/derived/bioval_v2/pmd_decato2020"
OUT.mkdir(parents=True, exist_ok=True)

def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

def category(n):
    if "TCGA" in n:
        return "TCGA_matched_healthy" if "MatchedHealthy" in n else "TCGA_tumor"
    return "non_TCGA"

U = C.load_benchmark_universe()
chrom = U["chrom"].to_numpy(); pos = U["pos"].to_numpy()
files = sorted((RAW / "native_assemblies/Human").glob("*.pmd"))
cols, meta, lift_reps = {}, [], []
naive = np.zeros(len(U), dtype=bool)  # comparison only: coordinates wrongly treated as already GRCh38
tot = {"n_in": 0, "n_unmapped": 0, "n_ambiguous": 0, "n_interval_dropped": 0, "n_dupe": 0, "n_out": 0}
conv = C._make_converter(ROOT / C.DEFAULT_CHAIN)
for f in files:
    d = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "s0", "e0"])
    h, rep = C.harmonize(d, "chrom", "s0", "hg19", pos_base=0, end_col="e0", converter=conv)
    h = h.rename(columns={"pos": "start", "end": "end"})
    for k in tot: tot[k] += rep[k]
    nv = pd.DataFrame({"chrom": d["chrom"].map(C.normalize_chrom), "st": d["s0"] + 1, "en": d["e0"]}).dropna().sort_values("st")
    for c, hh in nv.groupby("chrom"):
        um = np.where(chrom == c)[0]; st = hh["st"].to_numpy(); en = hh["en"].to_numpy()
        i = np.searchsorted(st, pos[um], side="right") - 1
        naive[um[(i >= 0) & (np.maximum.accumulate(en)[np.clip(i, 0, None)] >= pos[um])]] = True
    m = np.full(len(U), -1, dtype=np.int8)
    for c in sorted(set(h["chrom"])):
        um = np.where(chrom == c)[0]; hh = h[h["chrom"] == c].sort_values("start")
        m[um] = 0
        st = hh["start"].to_numpy(); en = hh["end"].to_numpy()
        # merge-free containment: running max of ends handles overlaps
        i = np.searchsorted(st, pos[um], side="right") - 1
        runmax = np.maximum.accumulate(en)
        ok = (i >= 0) & (runmax[np.clip(i, 0, None)] >= pos[um])
        m[um[ok]] = 1
    name = f.stem
    cols[name] = m
    meta.append({"sample": name, "category": category(name), "n_intervals_in": rep["n_in"], "n_intervals_out": rep["n_out"],
                 "n_interval_dropped": rep["n_interval_dropped"] + rep["n_unmapped"] + rep["n_ambiguous"],
                 "n_cpg_in_pmd": int((m == 1).sum()), "sha256": sha(f)})
M = pd.DataFrame(cols)
out = U[["cpg_idx", "chrom", "pos"]].copy()
A = M.to_numpy(); valid = A >= 0
nonT = np.array([category(c) != "TCGA_tumor" for c in M.columns])
def frac(sel):
    v = valid[:, sel]; a = (A[:, sel] == 1)
    n = v.sum(1)
    return np.where(n > 0, a.sum(1) / np.maximum(n, 1), np.nan).astype(np.float32), n
out["pmd_frac_all_samples"], out["n_samples_assessed"] = frac(np.ones(A.shape[1], bool))
out["n_samples_assessed"] = out["n_samples_assessed"].astype(np.int16)
out["pmd_frac_non_tumor_samples"], _ = frac(nonT)
out["pmd_in_any"] = np.where(valid.any(1), (A == 1).any(1).astype(np.int8), -1)
M.insert(0, "cpg_idx", U["cpg_idx"].to_numpy())
out.to_parquet(OUT / "cpg_pmd_summary_universe.parquet", index=False, compression="zstd")
M.to_parquet(OUT / "cpg_pmd_membership_per_sample.parquet", index=False, compression="zstd")
pd.DataFrame(meta).to_csv(OUT / "pmd_samples.csv", index=False)
cov = out.groupby("chrom").agg(n=("cpg_idx", "size"), n_assessed=("n_samples_assessed", lambda s: int((s > 0).sum())),
                               mean_pmd_frac=("pmd_frac_all_samples", "mean"))
cov.to_csv(OUT / "coverage_per_chrom.csv")
man = {"retrieval_date": datetime.date.today().isoformat(),
       "source_url": "https://github.com/bdecato/PMD_Paper_Scripts/raw/master/AllPMDs.tar.bz2",
       "tarball_sha256": sha(RAW / "AllPMDs.tar.bz2"), "license": "GPL-3.0 (repo LICENSE); paper CC-BY",
       "citation": "Decato BE et al. 2020 Epigenetics Chromatin 13:42, doi:10.1186/s13072-020-00363-7",
       "build_in": "hg19 (verified empirically: BUILD_CHECK.json)", "build_out": "GRCh38 (explicit liftover via biological_validation_v2.coordinates.harmonize)",
       "status": "SECONDARY/EXPLORATORY (derived from WGBS methylation; not a primary endpoint)",
       "liftover_totals_all_samples": tot,
       "liftover_drop_fraction": round((tot["n_unmapped"] + tot["n_ambiguous"] + tot["n_interval_dropped"]) / tot["n_in"], 5),
       "n_cpg_in_any_pmd_if_coords_wrongly_treated_as_GRCh38": int(naive.sum()),
       "n_human_samples": len(files), "n_universe": len(out),
       "n_cpg_pmd_in_any": int((out.pmd_in_any == 1).sum()), "n_cpg_assessed": int((out.n_samples_assessed > 0).sum()),
       "categories": pd.Series([m["category"] for m in meta]).value_counts().to_dict(),
       "dependency_flag": "derived from WGBS methylation: confounded with methylation prediction; independent of ENCODE inputs",
       "tool_versions": {"python": platform.python_version(), "pandas": pd.__version__, "liftover": "pyliftover via biological_validation_v2.coordinates.harmonize"},
       "chain_sha256": sha(ROOT / C.DEFAULT_CHAIN)}
(OUT / "MANIFEST_pmd.json").write_text(json.dumps(man, indent=1, default=str))
print(json.dumps(man, indent=1, default=str)); print(cov)
