"""Materialize (1) the canonical genomic-context mapping and (2) the universe covariate table.

Outputs (data/derived/bioval_v2/):
  coordinates/genomic_context_canonical.parquet, coordinates/MANIFEST_genomic_context.json
  covariates/universe_covariates.parquet, covariates/MANIFEST_covariates.json (coverage report)
No embeddings are read. Needs the plain-text hg38 fasta + .fai (data/external/reference/hg38.fa).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from cpg_repr_benchmark.biological_validation_v2 import coordinates as co
from cpg_repr_benchmark.biological_validation_v2.matching import build_covariates, load_probe_type

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/derived/bioval_v2"
TSS_MAP = OUT / "regulatory_activity/cpg_enhancer_map.parquet"
PROBE_TABLE = ROOT / "data/raw/epic_cross_platform/manifests/GPL21145_platform_table.tsv"
V2_MANIFEST = ROOT / "data/raw/epic_cross_platform/illumina_manifest/EPIC-8v2-0_A1.csv"
PROBE_COORDS = ROOT / "data/processed/ComputAgeBench/illumina_probe_grch38.parquet"


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default=str(ROOT / co.PLAIN_FASTA))
    ap.add_argument("--cg-sample", type=int, default=None, help="subsample for the CG check (default: all)")
    a = ap.parse_args()
    (OUT / "coordinates").mkdir(parents=True, exist_ok=True)
    (OUT / "covariates").mkdir(parents=True, exist_ok=True)
    u = co.load_benchmark_universe(ROOT)

    # ---- Task 1: canonical genomic context
    src_p = ROOT / co.GENOMIC_CONTEXT_SOURCE
    ctx = pd.read_parquet(src_p)
    islands = co.load_islands(ROOT / co.ISLAND_TRACK)
    canon, audit = co.map_genomic_context(ctx, u, islands=islands)
    ex = canon[canon["mapping_method"] == "decode_chrom_pos_exact"]
    audit["agreement_recomputed_rule_vs_legacy"] = float(
        (co.classify_island_context(ex["chrom"], ex["pos"], islands) == ex["context"].to_numpy()).mean())
    canon_p = ROOT / co.GENOMIC_CONTEXT_CANONICAL
    canon.to_parquet(canon_p, index=False)
    audit["context_counts"] = canon["context"].value_counts().to_dict()
    audit["cg_check_universe"] = co.cg_consistency(u, a.fasta, sample=a.cg_sample)
    audit["cg_check_context_mapped"] = co.cg_consistency(canon, a.fasta, sample=a.cg_sample)
    # round trip: re-encoding the canonical (chrom,pos) must reproduce the source id
    ex = canon[canon["source_cpg_idx"].notna()]
    audit["roundtrip_source_id_ok"] = bool(
        (co.encode_coord_id(ex["chrom"], ex["pos"]) == ex["source_cpg_idx"].to_numpy(np.int64)).all())
    audit["unmapped_explanation"] = (
        "universe CpGs without an exact (chrom,pos) row in the legacy table (their probes are absent from "
        "illumina_probe_grch38.parquet) get context recomputed from the UCSC cpgIslandExt track "
        "(data/bio_annotations/cpgIslandExt.hg38.txt.gz) with the documented rule; mapping_method marks them")
    man = {"encoding": "cpg_idx = chrom_code*1e9 + pos (GRCh38 1-based CpG cytosine; X=23,Y=24,M=25); "
                       "joined to universe cpg_idx (registry id) via exact (chrom,pos)",
           "source": str(co.GENOMIC_CONTEXT_SOURCE), "source_sha256": _sha(src_p),
           "island_track": co.ISLAND_TRACK, "output": co.GENOMIC_CONTEXT_CANONICAL, "output_sha256": _sha(canon_p),
           "fasta": str(a.fasta), "git_commit": _commit(), "audit": audit}
    (OUT / "coordinates/MANIFEST_genomic_context.json").write_text(json.dumps(man, indent=2, default=str))
    print(json.dumps(audit, indent=2, default=str))

    # ---- Task 2: covariates
    tss = pd.read_parquet(TSS_MAP, columns=["cpg_idx", "dist_to_gencode_tss_bp"]).set_index("cpg_idx")[
        "dist_to_gencode_tss_bp"]
    v2 = pd.read_csv(V2_MANIFEST, skiprows=7, usecols=["Infinium_Design_Type", "CHR", "MAPINFO", "Genome_Build"],
                     low_memory=False, dtype=str)
    v2 = v2[v2["Genome_Build"].eq("GRCh38") & v2["MAPINFO"].notna()]
    v2 = pd.DataFrame({"chrom": v2["CHR"].map(co.normalize_chrom), "pos": v2["MAPINFO"].astype(float).astype("int64"),
                       "Infinium_Design_Type": v2["Infinium_Design_Type"]}).dropna(subset=["chrom"])
    pt = load_probe_type(pd.read_csv(PROBE_TABLE, sep="\t", usecols=["ID", "Infinium_Design_Type"],
                                     low_memory=False), pd.read_parquet(PROBE_COORDS), u, fallback_pos_table=v2)
    cov, notes = build_covariates(u, context_path=canon_p, fasta_path=a.fasta, tss_dist_bp=tss, probe_type=pt)
    cov_p = OUT / "covariates/universe_covariates.parquet"
    cov.to_parquet(cov_p, index=False)
    na = {c: int(cov[c].isna().sum()) for c in cov.columns}
    report = {"n_rows": len(cov), "nan_per_column": na, "notes": notes, "columns": list(cov.columns),
              "sources": {
                  "context": "canonical genomic_context (UCSC cpgIslandExt hg38; island/shore<=2kb/shelf 2-4kb/open_sea)",
                  "cpg_density": "universe CpGs within +-500 bp, self excluded",
                  "cpg_density_hg38": "hg38 CG dinucleotides with C in +-500 bp window, self included",
                  "gc_content": "hg38 fasta, GC fraction over non-N bases in +-500 bp window",
                  "tss_dist": "log10(1+bp) to nearest GENCODE v50 transcript TSS (all transcripts, 388,474 TSS), "
                              "from regulatory_activity/cpg_enhancer_map.parquet:dist_to_gencode_tss_bp",
                  "probe_type": "Infinium design type I/II from GPL21145 (EPIC v1) platform table, joined by "
                                "hg38 (chrom,pos) via illumina_probe_grch38.parquet; positions not found there are filled "
                                "from the EPIC v2 manifest (EPIC-8v2-0_A1.csv, GRCh38 CHR/MAPINFO; agrees with "
                                "the primary route on 99.98% of overlapping CpGs). Remaining NaN = probe on no "
                                "locally available manifest (documented, intentionally not imputed)"},
              "value_counts_probe_type": cov["probe_type"].value_counts(dropna=False).to_dict(),
              "value_counts_context": cov["context"].value_counts(dropna=False).to_dict(),
              "output_sha256": _sha(cov_p), "git_commit": _commit()}
    (OUT / "covariates/MANIFEST_covariates.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
