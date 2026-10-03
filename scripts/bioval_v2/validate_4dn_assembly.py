#!/usr/bin/env python
# ruff: noqa: BLE001, EXE001, SIM115  (one-shot data-prep script; style-only, behaviour unchanged)
"""Task: record evidence that the 4DN coolers (H1/HFFc6/GM12878) are GRCh38 although cooler attr says 'unknown'.
Writes data/derived/bioval_v2/3d_genome/assembly_validation.json. CPU only; no embeddings."""
import gzip
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts/bioval_v2"))
import prepare_4dn as P4

OUT = P4.OUT
HG19 = {"chr1": 249250621, "chr2": 243199373, "chr3": 198022430, "chr4": 191154276, "chr5": 180915260, "chr6": 171115067, "chr7": 159138663,
            "chr8": 146364022, "chr9": 141213431, "chr10": 135534747, "chr11": 135006516, "chr12": 133851895, "chr13": 115169878, "chr14": 107349540,
            "chr15": 102531392, "chr16": 90354753, "chr17": 81195210, "chr18": 78077248, "chr19": 59128983, "chr20": 63025520, "chr21": 48129895,
            "chr22": 51304566, "chrX": 155270560}
CH = P4.CHROMS
seqs = {}; cur = None; buf = []
with gzip.open(ROOT / "data/external/reference/hg38.fa.gz", "rt") as f:
    for line in f:
        if line.startswith(">"):
            if cur in CH: seqs[cur] = np.frombuffer("".join(buf).upper().encode(), dtype=np.uint8)
            cur = line[1:].split()[0]; buf = []
        elif cur in CH: buf.append(line.strip())
    if cur in CH: seqs[cur] = np.frombuffer("".join(buf).upper().encode(), dtype=np.uint8)
fa_len = {c: len(v) for c, v in seqs.items()}
import cooler

res = {"claim": "cooler 'genome-assembly' attribute reads 'unknown'; assembly inferred and validated"}
res["fasta"] = "data/external/reference/hg38.fa.gz"
res["hg38_fasta_sha256"] = P4.sha256(ROOT / "data/external/reference/hg38.fa.gz")
systems = {}
for sysn, s in P4.SYSTEMS.items():
    r = 100000 if sysn == "GM12878_HiC" else 10000
    clr = cooler.Cooler(str(P4.cool_path(sysn, r)))
    cs = {c: int(clr.chromsizes[c]) for c in CH}
    systems[sysn] = {"accession": s["acc"], "cooler_attr_genome_assembly": str(clr.info.get("genome-assembly", clr.info.get("genome_assembly", "<absent>"))),
        "cooler_info_keys": sorted(clr.info.keys()), "chroms_checked": len(CH),
        "n_chroms_length_equal_hg38_fasta": int(sum(cs[c] == fa_len[c] for c in CH)),
        "n_chroms_length_equal_hg19": int(sum(cs[c] == HG19[c] for c in CH)),
        "chr1_len": cs["chr1"], "hg38_chr1_len": fa_len["chr1"], "hg19_chr1_len": HG19["chr1"],
        "max_abs_len_diff_vs_hg38": int(max(abs(cs[c] - fa_len[c]) for c in CH))}
res["systems"] = systems
res["hg38_fasta_vs_hg19_lengths_equal_count"] = int(sum(fa_len[c] == HG19[c] for c in CH))
# 4DN metadata (queried via public REST API)
meta = {}
for sysn, s in P4.SYSTEMS.items():
    url = f"https://data.4dnucleome.org/files-processed/{s['acc']}/?format=json"
    try:
        d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=60).read())
        meta[sysn] = {"url": url, "genome_assembly": d.get("genome_assembly"), "status": d.get("status"), "file_format": d["file_format"]["file_format"],
                          "date_created": d.get("date_created")}
    except Exception as e:
        meta[sysn] = {"url": url, "error": str(e)}
res["4dn_portal_metadata"] = meta
# CpG-site consistency in the universe vs hg38 sequence
U = P4.universe()
ok_c = 0; ok_g = 0; n = 0; per = {}
for c in CH:
    u = U[U.chrom == c]
    if len(u) == 0: per[c] = None; continue
    p = u["pos"].to_numpy() - 1; sq = seqs[c]
    isC = (sq[p] == ord("C")) & (sq[np.minimum(p + 1, len(sq) - 1)] == ord("G"))
    isG = (sq[p] == ord("G")) & (sq[np.maximum(p - 1, 0)] == ord("C"))
    per[c] = float((isC | isG).mean()); ok_c += int(isC.sum()); ok_g += int((isG & ~isC).sum()); n += len(p)
res["chroms_without_universe_cpgs"] = [c for c, v in per.items() if v is None]
res["cpg_site_consistency_hg38"] = {"n_universe": int(n), "frac_cpg_dinucleotide_at_position": float((ok_c + ok_g) / n), "n_C_of_CG": ok_c, "n_G_of_CG_only": ok_g,
    "min_per_chrom": float(min(v for v in per.values() if v is not None)), "note": "universe positions (1-based) examined in hg38.fa; a wrong build would give ~ chance (~1-5%) CpG dinucleotide rate"}
# E1-GC supporting evidence (already computed)
sup = {}
for sysn in P4.SYSTEMS:
    q = pd.read_csv(OUT / f"{sysn}_compartment_qc.csv"); q = q[q.res == 100000]
    sup[sysn] = {"res": 100000, "spearman_E1_GC_median": float(q.spearman_E1_GC.median()), "spearman_E1_GC_min": float(q.spearman_E1_GC.min()),
                     "n_chroms_below_0p2": int((q.spearman_E1_GC.abs() < 0.2).sum()), "note": "GC track from hg38; coherent E1-GC coupling would not be expected under a coordinate-build mismatch"}
res["compartment_E1_vs_hg38_GC_spearman"] = sup
allok = all(v["n_chroms_length_equal_hg38_fasta"] == len(CH) and v["n_chroms_length_equal_hg19"] == 0 for v in systems.values()) \
    and all(m.get("genome_assembly") == "GRCh38" for m in meta.values()) and res["cpg_site_consistency_hg38"]["frac_cpg_dinucleotide_at_position"] > 0.99
res["verdict"] = "GRCh38 (inferred+validated)" if allok else "FLAG: evidence not fully consistent with GRCh38 (inspect fields)"
json.dump(res, open(OUT / "assembly_validation.json", "w"), indent=2)
print(json.dumps({k: res[k] for k in ("verdict", "cpg_site_consistency_hg38", "4dn_portal_metadata", "hg38_fasta_vs_hg19_lengths_equal_count")}, indent=1)); print({k: (v["n_chroms_length_equal_hg38_fasta"], v["n_chroms_length_equal_hg19"], v["cooler_attr_genome_assembly"]) for k, v in systems.items()})
