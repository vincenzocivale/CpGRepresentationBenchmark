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
  coefficients alongside `cpg_idx`, used by the `clock_coefficients` regression probe (see
  below) as well as set membership.

## `hannum_clock_coefficients.parquet`, `phenoage_clock_coefficients.parquet`

Two more published epigenetic clocks, mapped and stored the same way as Horvath
(`horvath_clock_coefficients.parquet`): columns `cpg_idx` (int64) and `coefficient` (float,
the clock's elastic-net weight for that CpG). Matching `known_sets/hannum_clock.npy` and
`known_sets/phenoage_clock.npy` (membership-only, same `cpg_idx` set) are also provided for
the binary probe, consistent with the Horvath convention.

- **Source**: `Hannum.csv` / `PhenoAge.csv` coefficient tables from the `bio-learn` library
  (`https://github.com/bio-learn/biolearn`, MIT license) — Hannum et al. 2013 (blood-based
  clock, 71 CpGs after dropping the intercept) and Levine et al. 2018 "PhenoAge" (513 CpGs).
- Probe IDs mapped to GRCh38 via `illumina_probe_grch38.parquet`, same as Horvath: 71/71
  Hannum CpGs mapped, 513/514 PhenoAge CpGs mapped (1 miss, likely a retired/renamed probe ID).
- These clocks were chosen specifically because they, like Horvath, have a full public
  per-CpG coefficient table (unlike GrimAge, whose weights are not publicly released) — so
  the stricter regression probe (`clock_coefficients` in `bio_validation_report`, predicting
  the actual weight rather than membership) is meaningful for all three.

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
| `ewas_catalog_bmi` | `BMI`, `body mass index`, `Body mass index` | — | 67,815 |
| `ewas_catalog_smoking` | `smoking`, `Smoking`, `Tobacco smoking` | — | 9,109 |
| `ewas_catalog_cancer` | all cancer traits combined (see per-type sets below) | — | 1,125 |
| `ewas_catalog_breast_cancer` | `Breast cancer`, `breast cancer`, `Incident Breast Cancer`, `Prevalent Breast Cancer (Self-report)` | 7 | 570 |
| `ewas_catalog_lung_cancer` | `Lung cancer`, `lung cancer`, `Incident Lung Cancer`, `Prevalent Lung Cancer (Self-report)` | 8 | 510 |

The RA and schizophrenia sets deliberately mirror the existing downstream disease datasets
(`configs/datasets/gse42861.yaml`, `configs/datasets/gse147221.yaml`) so results can be
cross-checked: a representation that predicts disease status well via linear probing should
plausibly also show CpG-locus embedding separation on the matching EWAS set.

**Cancer is split per tumor type** (rather than only the aggregated `ewas_catalog_cancer`) so
bio-validation can show which tumor types a representation separates well, at the cost of
much smaller per-set CpG counts. `colorectal_cancer` (23 CpGs), `prostate_cancer` (8),
`ovarian_cancer` (14), and `pancreatic_cancer` (0, matched no probes at `p < 1e-7`) all fell
below `bio_validation_report`'s 10-CpG-per-class minimum for a balanced CV probe and were
removed from the repo entirely (not kept for provenance) — see
`scripts/build_ewas_catalog_sets.py` for the excluded trait groups if more EWAS Catalog studies
for these cancers are published later and the set becomes usable.

**Caveat**: `ewas_catalog_age`, `_rheumatoid_arthritis`, and `_bmi` are large (16-63% of a
full array's CpGs can land in them) because these traits have broad, well-documented
genome-wide methylation effects — this is a much noisier/broader signal than the curated
353-CpG Horvath clock or the small cancer sets, not a bug. Interpret a high probe score on
these broad sets as "the embedding tracks broad trait-associated methylation variation", not
as precise biomarker recovery.

## `known_sets/ewas_catalog_sle.npy`, `ewas_catalog_type2_diabetes.npy`, `ewas_catalog_coronary_heart_disease.npy`

Three more disease-status sets, each from a single published study's own reported significant-CpG
list (not the bulk EWAS Catalog download, unlike the sets above) — built ad hoc via PMC/Europe PMC
supplementary files, mapped to `cpg_idx` the same way (`illumina_probe_grch38.parquet` +
`encode_many`):

| set | source study | reported CpGs | CpGs mapped | probes dropped |
|---|---|---|---|---|
| `ewas_catalog_sle` | Lanata et al. 2022, *Arthritis & Rheumatology* (CLUES cohort), Supplementary Table S3 | 309 | 309 | 0 |
| `ewas_catalog_type2_diabetes` | Cardona et al. 2022, *Diabetologia*, meta-analysis of 5 prospective European cohorts (Doetinchem, ESTHER, KORA1, KORA2, EPIC-Norfolk), ESM Table 5 | 76 | 76 | 0 |
| `ewas_catalog_coronary_heart_disease` | Agha et al. 2019, *Circulation*, blood-leukocyte meta-analysis across 9 cohorts (n=11,461), main-text Tables 2 (30 incident-CHD CpGs) + 3 (29 incident-MI CpGs), deduplicated (7 CpGs in both) | 52 | 52 | 0 |

All three probe lists mapped to GRCh38 with zero misses.

## `known_sets/ewas_atlas_colorectal_cancer.npy`, `ewas_atlas_prostate_cancer.npy`, `ewas_atlas_ovarian_cancer.npy`

Built from the **EWAS Atlas** bulk download
(`https://ngdc.cncb.ac.cn/ewas/downloads/batch?file=EWAS_Atlas_associations.tsv`, ~106MB,
804,920 association rows) — a separate curated knowledgebase from the EWAS Catalog above
(`ngdc.cncb.ac.cn/ewas`), used specifically because it has much better coverage for three of
the four cancer trait groups that were dropped from `ewas_catalog_*` for being too small
(23/8/14/0 CpGs after `p < 1e-7`; see the caveat below). EWAS Atlas curates only associations
each source study itself reported significant, with no single uniform p-value cutoff across
studies, so — unlike the EWAS Catalog sets — **no p-value filter is applied on top**: every
row already curated for the matched trait is used. Rows are matched by exact `trait` label,
probe IDs deduplicated, then mapped to `cpg_idx` the same way as `ewas_catalog_*`
(`illumina_probe_grch38.parquet` + `encode_many`):

| set | matched `trait` value | curated probes | CpGs mapped | probes dropped |
|---|---|---|---|---|
| `ewas_atlas_colorectal_cancer` | `colorectal cancer` | 8,895 | 8,894 | 1 |
| `ewas_atlas_prostate_cancer` | `prostate cancer` | 8,170 | 8,170 | 0 |
| `ewas_atlas_ovarian_cancer` | `ovarian cancer` | 72 | 72 | 0 |

Named with an `ewas_atlas_` (not `ewas_catalog_`) prefix to keep the two source databases
distinguishable. Pancreatic cancer was also checked in EWAS Atlas (exact trait `pancreatic
cancer`) and still yielded only 6 unique probes across 2 studies — too few for the
10-CpG-per-class minimum — so, as with the EWAS Catalog, no `pancreatic_cancer` set was built
from either source.

## Rebuilding

`scripts/build_ewas_catalog_sets.py` rebuilds every `ewas_catalog_*` set (pass `--only <names>`
to build a subset, e.g. just the per-tumor-type cancer sets) from the EWAS Catalog bulk
download. `scripts/build_ewas_atlas_sets.py` similarly rebuilds every `ewas_atlas_*` set (same
`--only <names>` convention) from the separate EWAS Atlas bulk download. `genomic_context.parquet`
and the clock coefficient/membership files were built ad hoc the same way (download source, join with `illumina_probe_grch38.parquet` for coordinates,
`encode_many` from `cpg_repr_benchmark.data.coordinates` for `cpg_idx`) but do not yet have a
checked-in script.

Neither `genomic_context.parquet` nor `known_sets/` is required to run masking/reconstruction
experiments; only `scripts/run_bio_validation.py` needs them.
