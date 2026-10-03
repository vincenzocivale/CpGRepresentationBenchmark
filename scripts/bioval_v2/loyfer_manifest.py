# ruff: noqa: DTZ011, SIM115  (one-shot data-prep script; style-only, behaviour unchanged)
"""Write MANIFEST.json using biological_validation_v2.provenance.write_manifest + coordinates.overlap_audit."""
import datetime
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from loyfer_common import *

sys.path.insert(0, str(ROOT / "src"))
import h5py as _h5
import numba
import pandas
import scipy

from cpg_repr_benchmark.biological_validation_v2.coordinates import load_benchmark_universe, overlap_audit
from cpg_repr_benchmark.biological_validation_v2.provenance import SourceRecord, sha256_file, write_manifest

u2 = load_benchmark_universe(); u = load_universe()
same = set(map(tuple, u2[["cpg_idx", "chrom", "pos"]].to_numpy())) == set(map(tuple, u[["cpg_idx", "chrom", "pos"]].to_numpy()))
inv = pd.read_csv(OUT / "sample_inventory.tsv", sep="\t")
mk = pd.read_parquet(OUT / "loyfer_markers_all_cpgs.parquet")
audit_markers = overlap_audit(mk[["chrom", "pos"]].drop_duplicates(), u2)
audit_markers.pop("per_chrom")
cov = json.load(open(OUT / "coverage_summary.json"))
now = datetime.date.today().isoformat()
src = [
  SourceRecord(name="Loyfer2023 WGBS atlas pat.gz (253 files; 207 atlas + 46 cfDNA/WBC)", url="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5652nnn/", accession="GSE186458",
      path=str(PAT_DIR), genome_build_in="GRCh38", n_rows_in=len(inv), n_rows_out=int(inv.kind.eq("atlas").sum()),
      extra={"total_bytes": int(inv.size_bytes.sum()), "sha256": "not computed (93 GB); sizes in sample_inventory.tsv", "only_pat_downloaded": True}),
  SourceRecord.from_file(META / "geo_sample_metadata.tsv", name="GEO sample metadata (per-GSM SOFT brief)", url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", accession="GSE186458", retrieval_date=now, genome_build_in="n/a"),
  SourceRecord.from_file(META / "Loyfer2023_Nature_SuppTables_MOESM4.xlsx", name="Nature 2023 Suppl. tables (MOESM4_ESM.xlsx; S1 sample->group, S4A/S4B hg19 markers)", url="https://doi.org/10.1038/s41586-022-05580-6", retrieval_date=now, genome_build_in="mixed (S1 none; S4 hg19)"),
  SourceRecord.from_file(META / "hg19_github/Atlas.U25.l4.hg19.full.tsv", name="UXM_deconv hg19 U25 (paper-identity check)", url="https://github.com/nloyfer/UXM_deconv/blob/8d0bb456742ece21802bc8e651e080869c67c944/supplemental/Atlas.U25.l4.hg19.full.tsv", retrieval_date=now, genome_build_in="GRCh37"),
  SourceRecord.from_file(META / "hg19_github/Markers.U250.hg19.tsv", name="UXM_deconv hg19 U250 (paper-identity check)", url="https://github.com/nloyfer/UXM_deconv/blob/8d0bb456742ece21802bc8e651e080869c67c944/supplemental/Markers.U250.hg19.tsv", retrieval_date=now, genome_build_in="GRCh37"),
  SourceRecord.from_file(META / "hg19ToHg38.over.chain.gz", name="UCSC hg19ToHg38 chain (paper marker lift)", url="https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz", retrieval_date=now, genome_build_in="GRCh37"),
  SourceRecord.from_file(OUT / "frozen_pairs_seed17.parquet", name="frozen pair list seed 17 (frozen before any embedding seen)", genome_build_in="GRCh38", n_rows_in=len(pd.read_parquet(OUT / "frozen_pairs_seed17.parquet", columns=["cpg_i"]))),
  SourceRecord.from_file(META / "Atlas.U25.l4.hg38.full.tsv", name="Published atlas markers (U25, l4)", url="https://github.com/nloyfer/UXM_deconv/tree/main/supplemental", retrieval_date=now, genome_build_in="GRCh38", n_rows_in=979),
  SourceRecord.from_file(META / "Markers.U250.hg38.tsv", name="find_markers top-250 candidates with stats", url="https://github.com/nloyfer/UXM_deconv/tree/main/supplemental", retrieval_date=now, genome_build_in="GRCh38", n_rows_in=9520),
  SourceRecord.from_file(META / "Atlas.U25.l4.hg38.tsv", name="Published atlas U25 (36-column figure version, unused)", url="https://github.com/nloyfer/UXM_deconv", retrieval_date=now, genome_build_in="GRCh38"),
  SourceRecord.from_file(META / "Atlas.U250.l4.hg38.full.tsv", name="U250 atlas (downloaded, unused)", url="https://github.com/nloyfer/UXM_deconv", retrieval_date=now, genome_build_in="GRCh38"),
  SourceRecord(name="hg38.fa.gz (CpG index source)", path="data/external/reference/hg38.fa.gz", genome_build_in="GRCh38", extra={"sha256": sha256_file(ROOT / "data/external/reference/hg38.fa.gz"), "n_cpg_index": 29401795}),
  SourceRecord.from_file(UNIVERSE, name="benchmark universe array_cpg_map", genome_build_in="GRCh38", n_rows_in=len(u)),
]
params = {"min_cov_per_sample_cpg": 10, "group_rule": "donor=published PatientID; donor beta=mean of valid (cov>=10) sample betas; group beta=mean over donors; mask_primary: n_valid_donors>=min(2,n_donors); mask_lenient: n_valid_donors>=1",
  "sample_to_group_mapping": "PUBLISHED: Nature 2023 Suppl. Table S1 col Group (205 samples; name harmonisation Endothelium->Endothel, Ovary+Endom-Ep->Ovary-Ep); 2 Heart-Cardiomyocyte GEO samples not in S1 excluded; see sample_provenance.csv / manual_decisions.csv",
  "cpg_index": "wgbstools-style global 1-based hg38 CpG index regenerated from hg38.fa.gz (chr1-22,X,Y,M order); validated against 9,520+979 marker blocks (100% start/end match)",
  "markers": "published blocks from UXM_deconv supplemental; expanded to CpGs ours; all direction U (hypomethylated in target)",
  "profile_similarity": {"min_shared": 20, "seed": 17, "n_intra": 200000, "n_inter": 200000}, "frozen_pairs_sha256": json.load(open(OUT / "frozen_pairs_seed17_sha256.json"))["sha256"], "marker_provenance_verdict": json.load(open(OUT / "marker_provenance_verdict.json"))["verdict"],
  "tool_versions": {"numba": numba.__version__, "numpy": np.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__, "h5py": _h5.__version__, "python": sys.version.split()[0]},
  "row_counts": {"universe": len(u), "samples_total": len(inv), "samples_atlas": int((inv.kind == "atlas").sum()), "samples_in_groups": cov["n_atlas_samples_used"], "groups_present": cov["n_groups"],
                  "marker_cpgs_all": len(mk), "marker_cpgs_in_universe": int(mk.in_universe.sum())},
  "universe_same_set_as_package_load_benchmark_universe": bool(same),
  "outputs": {p.name: p.stat().st_size for p in sorted(OUT.iterdir()) if p.name != "MANIFEST.json"}}
write_manifest(OUT / "MANIFEST.json", "external_methylation_programs", src, {"marker_cpg_vs_universe": audit_markers, "coverage": cov}, params)
print(same, audit_markers)
