# External reconstruction confirmation: dataset audit (read-only)

Status: AUDIT, 2026-10-04. Read-only. No model was trained, no GPU was used, **no representation was evaluated on any external data**, no
dataset or parameter was chosen from representation performance, and no frozen artifact (`configs/frozen/*`, `data/cache/representations/regulatory_*`,
`biological_validation_v2` anything, confirmation-matrix spec/results) was modified. TCGA identities were read only as sample-name metadata
(`outputs/encode_atlas_v1/patients.npz`, key `sample_names`); no TCGA beta value was read (the TCGA matrix `tcga_array_official_full.h5` was not opened for beta,
and was not checksummed).
Reproducible numbers: `scripts/audit_external_cohorts.py` (test: `tests/test_audit_external_cohorts.py`) -> `outputs/external_reconstruction_audit/`
(`cohort_audit.json`, `cohort_overlap_summary.csv`, `raw_head_summaries.json`, `universe_zero_loci.npz`, `sha256_inputs.txt`).
Evidence labels: VERIFIED-LOCAL (read in a local file/computed here), VERIFIED-LOCAL-SIBLING (file of a sibling checkout), UNVERIFIED.

## 0. Benchmark locus universe and id systems (facts used everywhere below)

* Universe = `cpg_idx` of `data/cpg/registries/array_cpg_map.parquet` = axis of `data/methylation/tcga_array_official_full.h5` = `outputs/encode_atlas_v1/loci_seen.npz`:
  **408,399 autosomal 450K CpGs** (chr1-19: 388,599; chr20-22: 19,800; no X/Y/M). Build GRCh38, 1-based cytosine (registry `pos` equals the crosswalk `pos`, e.g. cg00000029 = chr16:53434200 in both).
* Id system: the universe and **all representation stores** (regulatory, functional, CpGPT, DeepCpG) are keyed by the TCGA/MethylProphet **global `cpg_idx`** (arbitrary integers, 85..23,125,032).
  The external cohorts prepared by this repo (`data/processed/*`) use the **coordinate-native** `cpg_idx = chrom*1e9 + pos` (`grch38_cpg_cytosine_1based_v1`). The two systems are joined on (chr, pos); the join is exact
  (GSE40279: 408,017 of 408,399 loci match on identical (chr,pos), 0 match with pos+1). The runner requires the matrix `cpg_idx` to be in the store id system, so an external matrix must be re-keyed (section 6).
* All-zero-embedding loci of `regulatory_histone_dnase_v1` (849 of 408,399: 817 on chr1-19, 32 on chr20-22), flagged by chunked read of the store (counts only; list saved in `universe_zero_loci.npz`).

## 1. Prespecified selection criteria (written before the per-dataset table; applied mechanically)

Nothing below refers to any representation's performance. Coverage/N numbers were computed by the audit script in the same session; the thresholds are generic (power and
universe-retention arguments given), not tuned to any outcome.

**Gates (all required; otherwise EXCLUDED with the failing gate named)**

| Gate | Requirement |
| --- | --- |
| G1 | Quantitative array beta values (or exactly invertible M-values) are on local disk (not label-only, not binary/sparse calls, not peaks). |
| G2 | Probe ids resolve to GRCh38 (chr,pos) through a repo crosswalk (the id system is resolvable). |
| G3 | No TCGA/TARGET sample or patient (identifier check + provenance); not a TCGA derivative. |
| G4 | >= 500 unique samples after documented QC exclusions. |
| G5 | >= 95% of the 408,399 universe loci covered (the universe may not shrink by more than 5%). |
| G6 | One study and one platform in the matrix used for training (no pooled-study / mixed-platform matrix). |

**Scored criteria (0/1/2 each; total max 18; no weights)**

| Id | 2 | 1 | 0 |
| --- | --- | --- | --- |
| S1 N samples (power: the test split is 10%; TCGA has 918 test patients) | >= 2000 (test >= 200) | 500-1999 | < 500 |
| S2 universe coverage | >= 99.5% | 95-99.5% | < 95% |
| S3 preprocessing documented for the file actually used | a named normalization/filter method in the GEO record or file header | software named, normalization not stated | not documented / heterogeneous |
| S4 batch structure | single lab AND chip/plate/site identifiers available locally | single lab without identifiers, or multi-site with identifiers | undocumented / pooled studies |
| S5 subject independence (locally verifiable metadata only) | one sample per subject, unique subject ids | replicate samples identified and groupable, or no subject ids but population design without repeated-measures indication | repeated measures/related subjects suspected and not groupable |
| S6 independence of biological domain from TCGA/TARGET | non-tumour tissue | n/a | tumour tissue of TCGA/TARGET cancer types |
| S7 local readiness | prepared `methylation.h5` in the repo contract | raw betas local, needs a conversion script | raw intensities/idat only |
| S8 platform match to the (450K-probe) universe | 450K | EPIC | other/mixed |
| S9 audit status of possible pre-training overlap for the sequence comparators (CpGPT-large corpus) | documented out-of-corpus by a locally available source | UNVERIFIED | documented in-corpus |

Tie-break (fixed now): (i) higher S1; (ii) larger N; (iii) fewer UNVERIFIED items.
Tissue composition (whole blood vs purified cell type vs mixed) is **descriptive only and deliberately not scored**: a homogeneous tissue reduces biological between-sample variance while a diverse one adds
cell-composition effects; there is no a-priori direction for a representation-ranking confirmation, and any direction would have to be justified by performance, which is forbidden. It is reported per cohort and
carried into the confounding-risk list.

## 2. Inventory (everything found locally)

Searched: `configs/datasets/*.yaml`, `docs/*PROTOCOL.md`, `data/{methylation,processed,raw,prepared,derived,external,protocols,cpg}` (metadata only), `data/local_manifest.json`, the stage3 worktree
(`CpGRepresentationBenchmark-stage3`, docs/configs only; it contains the same dataset configs plus EPIC/CNS stage code, no additional methylation matrix), `../data/*` (raw GEO downloads),
`../methylation-fm-benchmark/data/dataset/*`, `../MehylPredictor` (ENCODE WGBS/functional only), `../MethylFM/MethylProphetData` (TCGA/ENCODE derivatives).
`data/local_manifest.json` is only the TCGA bootstrap preflight (status MISSING_REQUIRED_ARTIFACTS: some protocol/reference files not in the tracked location; irrelevant to external cohorts).

### 2a. Candidates (per-dataset facts)

| Cohort | Platform | Samples (subjects) | Loci in file | Overlap with 408,399 | chr1-19 / chr20-22 / zero-emb. | Missingness (NaN) | Local state |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| GSE40279 (Hannum, whole blood) | 450K (GPL13534) | 656 (656) | 472,985 | 408,017 (99.91%) | 388,233 / 19,784 / 846 | 0 (qc: missing_beta_fraction 0.0) | prepared h5 |
| GSE42861 (Liu, RA vs control, PBL) | 450K | 689 (689) | 485,447 | 408,390 (99.998%) | 388,590 / 19,800 / 849 | 0 | prepared h5 |
| GSE147221 (schizophrenia, Dublin blood) | 450K | 679 of 720 (unique arrays; subject ids not in metadata) | 484,004 | 407,909 (99.88%) | 388,126 / 19,783 / 848 | NaN where detection P > 0.01: 5e-5 of values in subsample; max per-CpG 4.1%, max per-sample 0.24% | prepared h5 |
| GSE55763 (Lehne/LOLIPOP, whole blood) | 450K v1.2 (GPL13534) | 2,711 (2,664 individuals per GEO + technical-replicate samples) | 473,864 probes | 408,390 (99.998%) | 388,590 / 19,800 / 849 | NA encoded as `NA`; 0.07% in first 3,000 probes | RAW betas only (`../data/GSE55763/GSE55763_normalized_betas.txt.gz`, 10.4 GB), not prepared |
| GSE87571 (Johansson, whole blood) | 450K | 732 samples (summary says N=421 individuals) | 485,512 probes | 408,390 (99.998%) | same | NA present | RAW matrices in sibling (`methylation-fm-benchmark/.../GSE87571_matrix{1,2}of2.txt.gz`), not prepared; idat also in `../data/GSE87571` |
| GSE56046 (MESA CD14+ monocytes) | 450K | 1,202 (1,128 in ComputAgeBench) | 485,577 probes | 408,390 | same | **M-values** + detection P in file | RAW (sibling), not prepared |
| ComputAgeBench `benchmark` pooled | 450K/EPIC/27K/mixed (6 GPL ids) | 10,410 samples, 65 studies | 900,260 | 408,391 (99.998%) | 388,591 / 19,800 / 849 | 4 studies (27K) have >20% NaN on universe loci; 450K studies ~0 | prepared h5 |
| ComputAgeBench single studies N>=500 (GSE56046 1128, GSE42861 689, GSE117859 608, GSE111629 572, GSE117860 529 [EPIC], GSE72774 508) | single GPL each | as listed | (pooled h5) | same as pooled | same | EPIC studies 7-12% NaN on universe blocks | rows of the pooled h5 |
| `epic_sim_training` (derived from ComputAgeBench EPIC studies) | EPIC, split into observed (439,760) / target (404,095) halves | 1,272 (18 studies) | 843,855 | 382,155 (93.57%) | 363,179 / 18,976 / 799 | none stored | prepared (different contract: obs/target) |
| `paired_450k_epic` GSE86833 / GSE92580 | 450K + EPIC pairs | 15 / 12 pairs | 843,855 | 382,155 (93.57%) | same | ~0.1% | prepared |
| `epic_cross_platform` (GSE229715/240412/240469/286313 + `observed_beta.h5`) | EPIC cell lines/mixtures | 40 | 718,786 | not computed (G4 fails) | | | raw soft + 40-sample h5 |
| GSE90496 (Capper CNS tumours) | 450K | 2,801 | 428,799 probes | 380,987 (93.29%) | 362,085 / 18,902 / 792 | 0 in first 3,000 probes | raw beta (`../data/GSE90496`) |
| GSE75067 (breast tumours) | 450K | 188 | 485,577 | 408,390 | same | | raw (series matrix w/ data) |
| GSE118144 (paediatric SLE sorted cells, EPIC) | EPIC | 145 samples / 29 subjects | 865,918 | 380,818 (93.25%) | 361,937 / 18,881 / 795 | | raw |
| GSE103541 | EPIC (860,400 rows) | 140 | | not computed (G4) | | | raw (sibling), no series matrix |
| GSE69914 / GSE66836 / GSE133556 / GSE56581 | breast / lung / ovarian tumours (407/183/111 samples) ; T cells (199) | | | not computed | | | RAW.tar or series matrix; tumours or N<500 |

**Excluded as not quantitative array beta / not a matrix of subjects (G1) or otherwise unusable**

| Item | Reason |
| --- | --- |
| GSE118922 (`../data/GSE118922`) | 5hmC capture-seq peaks on hg19, 9 pooled samples, no beta matrix |
| GSE103886 | RRBS DMR spreadsheet only (series matrix + xlsx) |
| GSE156699 | RNA expression only |
| `data/raw/sparse_cns` GSE209865 / GSE289246 | nanopore binary 0/1 sparse calls on ~415 CNS tumours (tumour, binary), GSE289246 metadata only |
| `../data/CNSNanoporeData` (136 GB) | long-read nanopore, not array beta |
| `data/external/wgbs_atlas` (Loyfer WGBS .pat) | WGBS of sorted cell types, no per-subject array matrix; used by biological_validation_v2 (frozen) |
| `data/methylation/tcga_array_official_full.h5`, `../MethylFM/MethylProphetData/*tcga*`, `../data/MethylPredictorData/methylation/{tcga_array,epic}_full.h5` | TCGA/TARGET (and TCGA-EPIC) derivatives: fail G3 |
| `../data/MethylPredictorData/methylation/wgbs_full.h5`, ENCODE WGBS | WGBS, ENCODE; not an array cohort |
| CALERIE (`docs/CALERIE_ACCESS.md`) | controlled access, not downloaded |
| `data/bio_annotations`, clock coefficient parquet files | annotations/coefficients, no subject-level beta |

### 2b. Gate and score table (mechanical application)

| Cohort | G1 | G2 | G3 | G4 | G5 | G6 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | **Total** |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **GSE55763** | pass | pass | pass | pass (2,711) | pass | pass | 2 | 2 | 2 | 2 | 1 | 2 | 1 | 2 | 1 | **15** |
| **GSE40279** | pass | pass | pass | pass (656) | pass | pass | 1 | 2 | 1 | 1 | 2 | 2 | 2 | 2 | 2 | **15** |
| GSE42861 | pass | pass | pass | pass (689) | pass | pass | 1 | 2 | 1 | 1 | 2 | 2 | 2 | 2 | 1 | 14 |
| GSE147221 | pass | pass | pass | pass (679) | pass | pass | 1 | 2 | 2 | 1 | 1 | 2 | 2 | 2 | 1 | 14 |
| GSE87571 | pass | pass | pass | pass (732) | pass | pass | 1 | 2 | 2 | 1 | 0 | 2 | 1 | 2 | 1 | 12 |
| GSE56046 | pass (M-values) | pass | pass | pass (1,202) | pass | pass | 1 | 2 | 0 | 1 | 1 | 2 | 1 | 2 | 1 | 11 |
| ComputAgeBench single study N>=500 (best case) | pass | pass | pass | pass | pass | pass | 1 | 2 | 0 | 0 | 0 | 2 | 2 | 2 | 1 | <= 10 |
| ComputAgeBench pooled | pass | pass | pass | pass | pass | **FAIL (65 studies, 6 platforms)** | | | | | | | | | | excluded |
| epic_sim_training | pass | pass | pass | pass | **FAIL (93.57%)** | **FAIL (18 pooled studies)** | | | | | | | | | | excluded |
| GSE90496 | pass | pass | pass | pass | **FAIL (93.29%)** | pass | | | | | | **0 (tumour)** | | | | excluded |
| GSE75067, GSE69914, GSE66836, GSE133556 | pass | pass | pass | G4 (188) / tumour | | | | | | | | 0 | | | | excluded (G4 for GSE75067; tumours) |
| GSE118144, GSE103541, paired_450k_epic, epic_cross_platform | pass | pass | pass | **FAIL (145/140/27/40)** | G5 fails for GSE118144 | | | | | | | | | | | excluded |

Score justifications (evidence):

* S3. GSE55763: "Custom R scripts. Quantile normalisation of signal intensities" with control-probe adjustment (Lehne et al. design; GEO record). GSE147221: methylumi + wateRmelon `pfilter` + `dasen` (GEO record). GSE87571: minfi SWAN (GEO record). GSE40279: "BeadStudio software v3.2" only (no normalization method stated; the local file is the 2012 `average_beta` supplement). GSE42861: "minfi, R" only. GSE56046: file called "methylome_normalized", method not stated locally, values are M-values. ComputAgeBench: per-study preprocessing as deposited, not recorded per study.
* S4. GSE40279: 4 sites (UCSD 304, Utah 178, USC 139, Boston 35), 9 plates, `source`/`plate`/`beadchip_position` in phenotypes (multi-site with identifiers = 1). GSE42861: single case-control study, no chip/plate field locally (1). GSE147221: 60 chip ids derivable from `array_id`, but number of sites/labs not documented locally (1). GSE55763: 249 chips derivable from sample titles, one hybridization campaign per the GEO design (2).
* S5. GSE40279 and GSE42861: unique subject ids (656/689). GSE147221: unique array ids, no subject ids, replicate status UNVERIFIED (1). GSE55763: "population study" 2,639 + "technical replication study" 47 + both 25 = 2,711; 72 samples carry "Technical replicate group k" descriptions (2 groups per the description parser); no individual ids in GEO metadata (1; handled by the forced-train rule of the protocol). GSE87571: 732 samples vs 421 individuals in the series summary; subject ids not in local metadata (0).
* S6. All candidates ranked are peripheral blood / PBL / monocytes (non-tumour). Tumour cohorts (TCGA/TARGET pan-cancer domain) score 0 and fail other gates anyway.
* S9. GSE40279: the sibling doc `methylation-fm-benchmark/docs/CpGPT_analisi_modello.md` (line 201) and `docs/04_DATASETS_SPEC.md` (lines 59-60, 116) state that Hannum GSE40279 is **not** in CpGCorpus (this is a local summary of the CpGPT paper, VERIFIED-LOCAL-SIBLING; the paper's Supplementary Table 1 is not local). For every other cohort the CpGCorpus GSE list is unavailable (`methylation-fm-benchmark/models/CpGPT/data/cpgcorpus/raw` is empty): **UNVERIFIED**. CpGCorpus is a GEO-array corpus (150k samples, >2,000 studies per `docs/BIOLOGICAL_VALIDATION_V2_EXPLORATORY_POSTHOC_PROVENANCE_AUDIT.md`; exact composition UNVERIFIED), so GSE42861 / GSE55763 / GSE147221 / GSE87571 could be in it; DeepCpG-DNA (HCC scRRBS) cannot overlap with any array cohort.

### Result of the mechanical application

Ranking: GSE55763 = GSE40279 = 15 > GSE42861 = GSE147221 = 14 > GSE87571 12 > GSE56046 11 > single ComputAgeBench studies <= 10.
Tie-break (i) S1: GSE55763 (2) beats GSE40279 (1). **Best external by the prespecified criteria: GSE55763; backup (next in the same ordering): GSE40279.**
Fragility, stated honestly: the first two are tied on points; GSE55763 wins on power (S1), GSE40279 wins on readiness (S7), batch-identifier completeness (S4 of GSE55763 relies on one hybridization campaign
and chip ids) and documented out-of-corpus status (S9). A one-point change in S4, S5 or S7 for GSE55763 would put GSE40279 first. The user may prefer to decide between them with that in mind (see the proposal's decision list).

## 3. Per-dataset details for the recommended cohort and the backup

### 3a. GSE55763 (recommended)

* Provenance: GEO GSE55763, "A coherent approach for analysis of the Illumina HumanMethylation450 BeadChip improves data quality and performance in epigenome-wide association studies" (Lehne, Drong, Loh, Zhang, Scott, ... Chambers; submitted Mar 2014, updated Jul 2022); 450K v1.2; GEO series matrix has 2,711 sample columns, peripheral blood.
* Local files (not built by this repo): `../data/GSE55763/GSE55763_normalized_betas.txt.gz` (10,378,167,001 bytes, file mtime Mar 2014 = GEO timestamp; sha256 53941498df0ff37ea5987213845c98340a6b9aa8918d7253dcdbbca651c4bfab), `GSE55763_series_matrix.txt.gz`, `GSE55763_RAW.tar` (idat-level, 192 MB archive), `wget.log`. No repo manifest. File layout: probe rows (473,864 `cg` probes), columns alternate beta and "Detection Pval" per array (`7471147002_R01C01`, `Detection Pval`, ...), 2,711 arrays.
* Preprocessing: quantile normalisation of signal intensities + control-probe adjustment (per GEO record/title); the file name says "normalized". Beta values in [0,1]: outside-[0,1] fraction 0.0 on the first 3,000 probes; NaN 0.07%; fraction of values in [0.2,0.8] = 0.348 (first 3,000 probes; `raw_head_summaries.json`; a head-of-file subsample sorted by probe id, not a random sample).
* Coverage: 473,833 of 473,864 probes map through `data/processed/ComputAgeBench/illumina_probe_grch38.parquet`; **408,390 of 408,399 universe loci covered (99.998%): chr1-19 388,590/388,599, chr20-22 19,800/19,800, zero-embedding loci 849/849**; 65,406 cohort loci lie outside the universe (X/Y and probes not in the TCGA array universe).
* Subjects/metadata (series matrix): 2,711 samples; sex M 1,840 / F 871; age field present; dataset flag: population study 2,639, technical replication study 47, both 25; 249 distinct beadchip ids in titles. No subject id, no site/lab or smoking fields locally.
* Replicates: 72 samples have "Technical replicate group k, sample j" descriptions (GEO overview: "2,664 individuals, and 36 samples measured in duplicate"; the sample-level reconciliation 2,711 vs 2,664+36 is UNVERIFIED).
* TCGA overlap: sample ids are beadchip barcodes (GSM numbers in GEO); 0 names match TCGA/TARGET patterns; 0 intersect with the 9,178 TCGA sample names (8,707 TCGA + 471 TARGET). Provenance is GEO (LOLIPOP, Imperial College; population cohort), disjoint from the TCGA/TARGET portal.
* Pretraining overlap (sequence comparators): CpGCorpus membership UNVERIFIED; DeepCpG-DNA none (scRRBS).
* Preprocessing risk: quantile + control-probe normalisation is heavier than the TCGA beta processing, whose normalization state is **not recorded locally** (TCGA file provenance records only the MethylProphet parquet root `241231-tcga_array`): UNVERIFIED comparability.

### 3b. GSE40279 (backup)

* GEO GSE40279 (Hannum et al.), 656 whole-blood samples, ages 19-101 (mean 64.0), F 338 / M 318, Caucasian-European 426 / Hispanic-Mexican 230, sites UCSD/Utah/USC/Boston, 9 plates.
* Local: `data/processed/GSE40279/{methylation.h5, phenotypes.parquet, sample_mapping.parquet, cpg_mapping.parquet, manifest.json, qc.json}`; built Sep 15 by `scripts/data/prepare_gse40279.py` from `data/raw/GSE40279/GSE40279_average_beta.txt.gz` (2012 supplement) with the CpGPT `illumina_metadata.db` crosswalk (+1 shift 0->1-based), 473,018 of 473,034 probes mapped (16 missing), 33 same-coordinate probes collapsed by nanmean; h5 `beta` (656, 472,985) float32 lzf chunks (1,4096). sha256 in `sha256_inputs.txt` (manifest itself carries no checksums).
* Beta distribution (20,000 random universe columns x all 656 samples): NaN 0; outside [0,1] 0; **exact 0 or 1 fraction 1.6%**; quantiles q01 0.0, q05 0.0092, q25 0.073, q50 0.600, q75 0.870, q95 0.954; fraction <0.2 / >0.8 / mid: 0.383 / 0.374 / 0.243. The point mass at exactly 0 and the long low tail (q05 0.009 vs 0.03-0.04 in GSE42861/GSE147221) are consistent with an un-normalised BeadStudio `average_beta` (normalization method not stated: UNVERIFIED).
* Coverage: 408,017 of 408,399 (99.91%): chr1-19 388,233, chr20-22 19,784, zero-embedding 846. 382 universe loci are not on the file (probes removed upstream).
* No duplicate coordinates in the h5; no duplicate sample rows on the subsample; 656 unique subject ids.
* TCGA overlap: 0 (names are GSM ids).
* Out-of-corpus for CpGPT per the sibling docs (S9 = 2); also, the CpGPT paper uses it as an external test (documented there).

### 3c. Other prepared cohorts (for completeness)

* GSE42861: 689 PBL, RA 354 / control 335, F 492 / M 197, age 18-70 (mean 51.9), 485,447 loci, 408,390 covered; beta in [0,1], exact 0/1 4e-5; q50 0.574; "minfi, R" only. Built by `prepare_gse42861.py` from the 2.7 GB series matrix (which contains the betas); the same GEO study is also inside ComputAgeBench (different beta values: q50 0.689 on a different column subsample; not comparable, and not independent of each other).
* GSE147221: 679 included (SCZ 348 / control 331; 41 excluded: 34 GEO low-quality, 7 technical controls), age 17-70.9, sex M 507 / F 206 (of 720); beta kept only where detection P <= 0.01; q50 0.546; 484,004 loci; sources blood EDTA tube vs buffy coat (collection-type batch covariate).
* ComputAgeBench: 10,410 samples, 65 studies, blood 10,151 / saliva 259; platforms GPL13534 6,245, GPL21145 1,977, GPL16304 1,765, GPL23976 216, GPL8490 (27K) 145, GPL29753 62; cell types whole blood 5,790, CD14+ 1,440, PBL 1,017, CD4+ 645, PBMC 328, others; condition classes include HC 2,836, ISD 3,512, CVD 1,242, NDD 1,606 (no cancer class); 0 TCGA-pattern names. Per-study table: `cohort_audit.json` -> `ComputAgeBench_benchmark.per_study`.

## 4. TCGA non-overlap (identifiers only)

| Check | Result |
| --- | --- |
| TCGA h5 sample axis (read as names only) | 9,178 unique: 8,707 `TCGA-*`, 471 `TARGET-*`; no GEO-type names |
| GSE40279 / GSE42861 / GSE147221 sample names vs TCGA names | 0 / 0 / 0 exact matches, 0 TCGA-pattern names |
| ComputAgeBench 10,410 sample names | 0 matches, 0 TCGA-pattern names |
| Raw cohorts (GSE55763 header ids = beadchip barcodes; GSE90496 "SAMPLE n") | 0 TCGA-pattern ids (checked on the file headers in `raw_head_summaries.json`) |
| Provenance | GEO series (public deposits) vs TCGA/TARGET portal data; a TCGA tumour cannot be in a blood/PBL/monocyte cohort. Formal cross-reference of GEO sample accessions to TCGA barcodes is not possible from local metadata (TCGA samples carry no GSM ids): identifier non-overlap + provenance is the available evidence |

## 5. Preprocessing and batch issues (all candidates)

* No cross-reactive/SNP probe filtering is applied in any prepared cohort (all 450K probes with coordinates are kept; only GSE147221 drops values by detection P). The universe is the TCGA 450K autosomal probe set, so X/Y probes and the ~65-77k probes outside it never enter; no cross-reactive/SNP annotation list was found locally (UNVERIFIED which universe loci are cross-reactive).
* Multi-probe coordinates collapse by nanmean (33 loci in GSE40279/42861).
* Normalization state differs by cohort: BeadStudio average beta (GSE40279), minfi unspecified (GSE42861), dasen + pfilter (GSE147221), quantile + control-probe adjustment (GSE55763), SWAN (GSE87571). TCGA's own state is not recorded locally (UNVERIFIED).
* No platform mixing inside GSE40279 / GSE42861 / GSE147221 / GSE55763; ComputAgeBench mixes 27K/450K/EPIC/EPIC+.
* Probe-design bias: every cohort here is a 450K (or EPIC) array, so the measured loci are probe-design selected exactly like the TCGA universe; this limits generality equally for all arms (see proposal confounders).

## 6. Evidence gaps (UNVERIFIED items and missing artifacts)

1. CpGCorpus GSE list / split (CpGPT pre-training overlap) is not available locally for any cohort except the documented out-of-corpus statement for GSE40279.
2. TCGA array beta normalization state (comparability of preprocessing with each external cohort) is not recorded.
3. GSE55763 has no repo preparer/manifest; the 72 replicate samples' exact grouping and the subject-level reconciliation (2,711 vs 2,664+36) are not resolvable from local metadata.
4. GSE40279/GSE42861 manifests contain no checksums (computed here instead, `sha256_inputs.txt`); the GSE40279 normalization state is not stated by GEO.
5. Relatedness (twins/families) cannot be excluded for any cohort from local metadata (population cohorts assumed; the Utah subset of GSE40279 is of unknown family structure).
6. The sibling `GSE56046` is the MESA monocyte M-value table; conversion would be exact (beta = 2^M/(2^M+1)) but is outside this audit.
