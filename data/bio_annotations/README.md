# Biological annotation sources

Reference files consumed by `src/cpg_repr_benchmark/bio_validation/annotations.py`. All files
listed below are already present in this directory (built by the scripts referenced, kept for
reproducibility — rerun the script to refresh if the upstream source updates).

## `genomic_context.parquet`

Columns: `cpg_idx` (int64, `grch38_cpg_cytosine_1based_v1` namespace), `context`
(`island`/`shore`/`shelf`/`open_sea`).

- **Source**: UCSC `cpgIslandExt` track for hg38, downloaded from
  `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz`
  (32,038 island intervals; 27,949 kept after restricting to `chr1-22,X,Y`).
- **CpG universe annotated**: the ~1.18M Illumina array probes already mapped to GRCh38 in
  `data/processed/ComputAgeBench/illumina_probe_grch38.parquet` (built for a different
  purpose but reused here as the coordinate source).
- **Classification rule**: `island` = position falls inside a `cpgIslandExt` interval;
  `shore` = within 2kb of an island boundary; `shelf` = 2-4kb; `open_sea` = beyond 4kb
  (standard shore/shelf convention).
- **Not included**: gene-body/promoter/intergenic context — only the island/shore/shelf/
  open_sea axis was built. Add a second table (e.g. `gene_context.parquet`) the same way if
  needed later.

## `known_sets/horvath_clock.npy`

353 `cpg_idx` values (Horvath 2013 pan-tissue epigenetic clock).

- **Source**: `Horvath1.csv` coefficient table from the `bio-learn` library
  (`https://github.com/bio-learn/biolearn`, MIT license), which vendors the original
  Horvath 2013 *Genome Biology* clock coefficients. Verified count matches the published
  353-CpG clock exactly.
- Probe IDs mapped to GRCh38 via the same `illumina_probe_grch38.parquet`; all 353 probes
  mapped with zero misses.
- `horvath_clock_coefficients.parquet` (same directory) keeps the original elastic-net
  coefficients alongside `cpg_idx`, in case an actual Horvath age score (not just set
  membership) is needed later.

## `known_sets/ewas_catalog_*.npy`

Built from the EWAS Catalog bulk download
(`https://www.ewascatalog.org/static/docs/ewascatalog-results.txt.gz` +
`ewascatalog-studies.txt.gz`, ~8.0M association rows across 7,639 studies). For each set,
studies were matched by exact `trait` label, then associations filtered at
`p < 1e-7` (standard EWAS genome-wide significance threshold) and deduplicated by `cpg_idx`:

| set | matched `trait` values | studies | CpGs (p<1e-7) |
|---|---|---|---|
| `ewas_catalog_age` | `Age`, `age`, `Ageing` | 25 | 300,698 |
| `ewas_catalog_rheumatoid_arthritis` | `Rheumatoid arthritis`, `Incident Rheumatoid Arthritis`, `Prevalent Rheumatoid Arthritis (Self-report)` | 8 | 47,956 |
| `ewas_catalog_schizophrenia` | `Schizophrenia`, `schizophrenia` | 10 | 5,460 |

The RA and schizophrenia sets deliberately mirror the existing downstream disease datasets
(`configs/datasets/gse42861.yaml`, `configs/datasets/gse147221.yaml`) so results can be
cross-checked: a representation that predicts disease status well via linear probing should
plausibly also show CpG-locus embedding separation on the matching EWAS set.

**Caveat**: `ewas_catalog_age` and `_rheumatoid_arthritis` are large (25-63% of a full array's
CpGs can land in them) because age and RA have broad, well-documented genome-wide methylation
effects — this is a much noisier/broader signal than the curated 353-CpG Horvath clock, not a
bug. Interpret a high probe score on these sets as "the embedding tracks broad
age/RA-associated methylation variation", not as precise biomarker recovery.

## Rebuilding

Scripts used to build these files are not currently checked into the repo (they were run
ad hoc against the downloaded source files under `/tmp`). If you need to refresh them,
recreate the same steps: download the source, join with
`illumina_probe_grch38.parquet` for coordinates, `p < 1e-7` filter for EWAS Catalog sets,
`encode_many` (from `cpg_repr_benchmark.data.coordinates`) to get `cpg_idx`.

Neither `genomic_context.parquet` nor `known_sets/` is required to run masking/reconstruction
experiments; only `scripts/run_bio_validation.py` needs them.
