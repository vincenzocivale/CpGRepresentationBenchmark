# Biological task inventory and fairness audit (data collection only)

Scope: every existing result in which a CpG-locus representation was used to predict, classify or retrieve a property of the
LOCUS. Nothing was run, re-run, fixed or frozen to produce this document. All numbers are read from existing result files; the
machine-readable registry is in `paper/registry/` (built by `paper/scripts/build_bio_task_registry.py`, checked by
`paper/scripts/validate_registry.py`). The exploratory Loyfer failure-audit directory was excluded entirely (not read, not hashed,
not in the registry). Problems are listed first (section 4); nothing has been corrected.

Main panel: `regulatory_histone_dnase_v1` (candidate, SVD-256 of 2,492 histone + DNase tracks), `cpgpt_large_locus` (native 512D),
`deepcpg_dna_locus` (native 128D). `functional_annotations_pca` is reported separately and marked LEGACY.

## 1. Headline findings

1. `regulatory_histone_dnase_v1` appears in NO biological probe other than the bioval-v2 family (frozen primaries/secondaries,
   registered sensitivities, exploratory post-hoc Loyfer). It is absent from every EWAS / disease / age / clock / smoking / tissue / context /
   conservation probe. Only weak campaign proxies exist (a histone + 18 context-column arm; raw histone-only feature subsets).
2. Native CpGPT-large 512D and native DeepCpG 128D are present in bioval-v2, the legacy `bio_validation` probes, ENCODE exp5 and exp2.
   They are absent (compact-256 variant only for CpGPT, nothing for DeepCpG) from the single-draw matched-EWAS table and the
   co-methylation test.
3. The only phenotype/disease/EWAS/clock probes with matched negatives, chromosome-blocked folds and CIs (ENCODE exp5) do not include the
   candidate. The legacy probes that do include native CpGPT/DeepCpG use random folds, unmatched negatives, AUROC only, no CIs.
4. No result exists for: promoter / gene body / intergenic classes, cCRE classes, TSS-distance classes, ChromHMM, tissue-associated locus
   sets other than Loyfer markers, or any tissue/cell-type-specific EWAS set. The "regulatory annotation" and "enhancer" coverage is FANTOM5 only.
5. The frozen Loyfer primary is a NEGATIVE result for the candidate (pooled Spearman -0.063) and for legacy functional (-0.038); sequence arms
   are positive (CpGPT 0.106, DeepCpG 0.115). CpGPT-large is methylation-supervised (see `exploratory_posthoc_loyfer/interpretation_ABC.json`:
   `provenance_outcome = DIRECT_METHYLATION_SUPERVISION`).

## 2. (a) Inventory of biological experiments

Status vocabulary: frozen | preregistered | exploratory | discovery-campaign | legacy-unregistered. "In registry" = rows exist in
`bio_task_registry.csv`. Machine-readable version with file paths and sha256: `paper/registry/bio_task_inventory.csv`.

| id | experiment | family | target source | representations tested | metrics | status | in registry |
|---|---|---|---|---|---|---|---|
| v2_frozen | bioval-v2 frozen endpoints: Loyfer profile similarity, Loyfer marker kNN, Micro-C H1 intra 10 kb (+HFFc6, H1 inter 1 Mb), E1 compartments (H1, GM12878), FANTOM5 membership (+win500, tss5k pool), FANTOM5 activity similarity, replication-timing consensus, PMD | methylation program; 3D genome; regulatory activity; replication timing | Loyfer WGBS; 4DN; FANTOM5; UW Repli-seq; Decato 2020 | regulatory_histone_dnase_v1, cpgpt_large_locus 512D, deepcpg_dna_locus 128D, functional_annotations_pca (legacy) | Spearman, AUROC, R2, enrichment, delta-cosine (+22-block bootstrap CIs, paired contrasts) | frozen (tag `bioval-v2-protocol-freeze-v1`) | yes (268 rows; 4 primaries + 9 other endpoints) |
| v2_sensitivity | 12 registered sensitivities (Loyfer lenient mask / single-sample groups; E1 50 kb; FANTOM5 N=1/3/10/20; activity by category/sample level) | same | same | same 4 arms | same | preregistered | yes (144 rows) |
| v2_exploratory_posthoc | Loyfer post-hoc: B3 profile ridge (R2), similarity ranking per stratum, geometry deltas | methylation program | Loyfer WGBS | same 4 arms | R2, Pearson, Spearman | exploratory (registered Part B before running; no claim) | yes (520 rows); geometry deltas not row-ised |
| legacy_bio_validation | legacy random-fold probes: island/shore/shelf/open sea, 16 EWAS/clock sets (membership), 4 coefficient regressions (Horvath, Hannum, PhenoAge, phastCons), locality | genomic locus identity; disease/pathology; aging; environmental exposure | UCSC cpgIslandExt; EWAS Catalog/Atlas; clock tables; phastCons | cpgpt_locus, cpgpt_locus_large 512D, deepcpg_dna_locus 128D, deepcpg_dna_locus_hepg2, functional_annotations_pca (older PCA store), methylgpt_locus, methylgpt_locus_medium, ntv3_pre (all probes errored) | AUROC (mean of folds), accuracy/F1/precision/recall, R2/Pearson/MAE, enrichment | legacy-unregistered | yes (572 rows) |
| umap_neighborhood | UMAP biology neighbourhood enrichment (figure side output) | genomic locus identity; disease/pathology | island/shore/shelf/open sea; colorectal + prostate EWAS Atlas | functional_annotations_pca, deepcpg_dna_locus, cpgpt_locus_large | neighbour-fraction enrichment | legacy-unregistered | yes (270 rows) |
| encode_ewas_matched_single_draw | matched EWAS/clock membership, one matching draw | disease/pathology; aging; exposure | EWAS Catalog/Atlas, clocks | campaign arms: full (=functional), context_only, add/assay/Histone ChIP-seq, fm/cpgpt_locus_large (compact 256) | AUROC, AUPRC (matched sample) | discovery-campaign | yes (104 rows; 3 sets unscored) |
| encode_exp5 | repeated matched EWAS/clock probes, 20 matching draws, base and probe-type variants | same | same | functional_256, cpgpt_large_native512, deepcpg_hcc_native128, cpgpt_large_svd256, deepcpg_hcc_pad256, context_only_256 | AUROC + CIs; paired deltas vs functional_256 | discovery-campaign | yes (156 rows + 130 paired contrasts) |
| encode_exp2 | decodability of training-patient mean beta (11,929 loci) | methylation program | TCGA discovery-patient mean beta | Functional-256/128, CpGPT native 512 / compact 256, DeepCpG native 128 / pad 256, context-only, raw feature subsets | R2, Pearson, Spearman, MSE (chromosome-bootstrap CIs) | discovery-campaign | yes (84 rows) |
| encode_co_methylation | far-neighbour co-methylation (held-out patients) | methylation program | held-out beta correlations | full, context_only, add/assay/Histone, fm/cpgpt_locus_large | mean neighbour vs matched correlation (derived by script) | discovery-campaign | yes (12 rows) |
| encode_biological_probes | feature-group probes of mean/variance beta (3,988 rows) | methylation program | discovery-patient mean/variance | raw feature groups, not embeddings | MSE, R2 | discovery-campaign | no (inventory only) |
| encode_tissue | per-TCGA-project tissue-association probes (60 projects) | disease/pathology | TCGA per-cancer mean beta | raw feature groups, not embeddings | MSE, R2 | discovery-campaign | no (inventory only) |
| encode_exp3 | ridge null / "proxy null" for histone feature subsets | methylation program | mean/variance beta | feature subsets | relative MSE delta % | discovery-campaign | no (inventory only) |
| encode_exp4 | chromosome-transfer (OOD) reconstruction | other | patient reconstruction | campaign arms | MSE / skill | discovery-campaign | no (reconstruction, not a locus property) |
| excluded | patient-level GSE42861/GSE147221/GSE40279 tasks; masking benchmark; regulatory family screen/confirmation/selection; upscaling | other | patient labels / reconstruction | all arms | MSE and patient metrics | n/a | no (not locus-level) |
| excluded | `loyfer_failure_audit` | methylation program | Loyfer | 4 arms | n/a | exploratory, in progress | EXCLUDED by instruction |

A grep for AUROC / AUPRC / roc_auc / average_precision columns across `outputs/` (excluding the 64 GB benchmark payloads and the excluded
audit) found only the bioval-v2 outputs and `outputs/encode_atlas_v1/ewas_matched_membership.csv`; the remaining probe results are in
`summary.json`, exp5/exp2 CSVs and the co-methylation CSVs listed above.

Coverage of the requested categories: CpG context classes (island/shore/shelf/open sea: legacy only, plus UMAP neighbourhood), promoter /
gene-body / intergenic (none), regulatory annotations / cCRE (none; FANTOM5 only), enhancer-associated loci (FANTOM5 frozen), EWAS sets by
trait (smoking, BMI, age, RA, schizophrenia, SLE, T2D, CHD, breast, lung, colorectal, prostate, ovarian: legacy + exp5 + single-draw), age
and clocks (Horvath, Hannum, PhenoAge: legacy membership + coefficient regression + exp5), tissue-associated sets (Loyfer markers only; encode
`tissue/` is a feature-attribution probe), FANTOM5 (frozen), Micro-C / compartments (frozen), replication timing (frozen), Loyfer pair
similarity and marker kNN (frozen), co-methylation (campaign), alcohol (no set exists), sex (no set exists).

## 3. (b) Main-paper-quality list (MAIN_QUALITY)

Rule: frozen/preregistered, matched or stratified design where relevant, chromosome-blocked split, no tuning on test labels, CIs, all three
main-panel representations present. All 22 tasks below come from bioval-v2 (CIs are coarse: 22 chromosome blocks, B=1000, declared in the
frozen protocol).

Frozen: Loyfer profile similarity (primary), Micro-C H1 intra 10 kb (primary), FANTOM5 membership (primary; + win500 and tss5k sensitivities),
replication-timing consensus (primary), E1 compartments H1 and GM12878 (secondary), FANTOM5 activity similarity (secondary), Micro-C HFFc6
intra 10 kb (sensitivity). Preregistered sensitivities: Loyfer lenient mask, Loyfer single-sample-group drop, E1 50 kb (H1, GM12878),
FANTOM5 N=1/3/10/20, FANTOM5 activity by category and sample level.

Frozen primary values (candidate / CpGPT-large / DeepCpG / legacy functional; paired delta candidate minus comparator with 95% CI):

| task (metric) | candidate | CpGPT-512 | DeepCpG-128 | legacy functional | cand - CpGPT | cand - DeepCpG |
|---|---|---|---|---|---|---|
| Loyfer profile (Spearman, pooled >1 Mb + inter) | -0.063 | 0.106 | 0.115 | -0.038 | -0.169 [-0.181,-0.158] | -0.178 [-0.193,-0.163] |
| Micro-C H1 intra 10 kb (AUROC) | 0.568 | 0.531 | 0.502 | 0.568 | 0.037 [0.028,0.047] | 0.066 [0.055,0.077] |
| FANTOM5 membership (AUROC) | 0.833 | 0.701 | 0.623 | 0.850 | 0.132 [0.123,0.141] | 0.210 [0.198,0.223] |
| RT consensus (Spearman) | 0.703 | 0.486 | 0.359 | 0.718 | 0.217 [0.196,0.234] | 0.344 [0.308,0.376] |

Circularity: Loyfer and RT are INDEPENDENT; FANTOM5 and Micro-C are PARTIALLY_RELATED (frozen table in `docs/BIOLOGICAL_VALIDATION_V2.md`
section 3). Candidate - legacy functional is small and mixed (Micro-C +0.001 [-0.000, 0.002]; FANTOM5 -0.017; RT -0.015; Loyfer -0.025).

## 4. (c) Supplementary-only list (SUPPLEMENTARY_ONLY)

- Frozen but secondary/exploratory by protocol: Loyfer marker kNN (marker provenance verdict WEAK, 1,257 markers, not in Holm family),
  Micro-C H1 inter 1 Mb (98.3% drop, weak balance), PMD probe (methylation-derived target, prevalence 0.805, GC/RT confounded).
- Exploratory post-hoc Loyfer (B3 ridge raw/centred x all/variable CpGs; similarity ranking; geometry deltas): no claim; absolute values
  have no CI (paired delta CIs only).
- ENCODE exp5 matched EWAS / clock probes (13 evaluable sets x base/probe variants): sound design but the candidate is not evaluated.
- Single-draw matched EWAS table (campaign arms; no CI; only proxies for candidate and CpGPT; no DeepCpG).
- exp2 mean-beta decodability (label derived from training patients; candidate only via raw histone-only proxy).
- Co-methylation (no CI; proxies only; no DeepCpG, no native CpGPT).
- Legacy locality probe (unsupervised, 2,000 queries, no CI) and UMAP neighbourhood metrics for the two EWAS Atlas sets.

## 5. (d) Problems: fairness, circularity, coverage (specific)

Coverage
1. Candidate missing from all EWAS/clock/disease/age/smoking/context/conservation/locality probes (`legacy_bio_validation`, exp5, UMAP
   neighbourhood); only proxies in single-draw EWAS, co-methylation, exp2 (see section 6).
2. exp5 and legacy contain no histone/DNase arm at all; only `functional_256` (campaign `full`), context-only and compact/pad variants.
3. Two different stores are both called `functional_annotations_pca`: legacy `bio_validation` uses `functional_annotations__pca_native_genomewide.h5`;
   bioval-v2, exp5, exp2 use the campaign `full_a18b869b.h5` store. They are not interchangeable.
4. `ntv3_pre` errored on every probe (chr1-only atlas, stale set names); `methylgpt_locus` and `_medium` errored on 2 probes each
   (`unscored_tasks.csv`). 3 sets (ovarian, CHD, schizophrenia) have no matched support in exp5 (28/49/28 pairs).

Legacy `bio_validation` known-set probes (`outputs/bio_validation/*/native_frozen/seed_17/summary.json`; PROBLEMATIC)
5. Split: `StratifiedKFold(5, shuffle, seed 17)` random loci (`bio_validation/probes.py`): neighbouring CpGs sharing peaks/promoters fall in train and test
   (spatial leakage); no chromosome blocking.
6. Negatives: all other CpGs of the 408,399 universe, unmatched, extreme imbalance (age 0.54%, RA 0.058%, SZ 0.007%, smoking 1.9%, BMI 5.9%);
   AUROC only (mean of fold AUROCs), no AUPRC, no CIs; accuracy/F1/precision/recall are at p=0.5 (F1 NaN or ~0 for most sets).
7. Set size vs universe: README sizes (e.g. age 300,698; RA 47,956; SZ 5,460; BMI 67,815) shrink to 2,194 / 238 / 28 / 24,190 positives inside the
   benchmark universe (set_fraction_in_universe in `per_task_audit.csv`): most EWAS probes lie outside the 408,399-CpG TCGA universe, so the
   age/RA/SZ sets are small, array-specific residues. Six sets have <150 positives (SZ 28, ovarian 28, CHD 49, hannum 64, T2D 71, SLE 139).
8. Clock coefficient regressions: random KFold on n=64/344/498 CpGs; negative R2 for most arms; effectively uninformative.
9. Context targets (island/shore/shelf/open sea): SOURCE_RETENTION for legacy functional (dense columns 0-3 are its inputs; `bio_validation/probes.py`
   states this); for the candidate not an input (frozen table cell I) but biologically coupled to H3K4me3/DNase promoter evidence; island status is
   sequence-composition defined (PARTIALLY_RELATED for sequence arms). Threshold metrics at p=0.5 under imbalance (shelf: recall 0.001).
10. No git commit recorded in legacy summaries; no protocol tag; file mtimes Sep 19-23 only.

ENCODE matched EWAS (exp5, single draw; SUPPLEMENTARY_ONLY)
11. Matching strata = chromosome x CpG context x gene region x core-breadth decile (dense cols 0-7 and 22 of the FUNCTIONAL store): these are inputs of
    the legacy functional/context arms but not of the candidate or sequence arms, so the matching neutralises context information for some arms and
    not others. GC, CpG density and TSS distance are NOT matched (FANTOM5 matches them). Infinium probe type is matched only in the `probe` variant
    (others show imbalance, e.g. BMI type-I fraction 0.142 positives vs 0.186 negatives, age 0.339 vs 0.265; 7.4% of loci have unknown type).
12. Negatives are "unlisted", not proven unassociated (`ewas_protocol.json`: "NOT proven unassociated; original study tested universes unavailable");
    negatives may be members of other EWAS sets.
13. AUPRC not computed (AP exists only in the single-draw file, on the matched sample where prevalence is 0.5 by construction, baseline 0.5).
    `oof_predictions.parquet` exists and would permit AUPRC without new training, but nothing was recomputed here.
14. Probe: fixed C=1 logistic regression (no tuning on evaluation labels), chromosome GroupKFold(5), CIs by seed-mixture bootstrap that mixes matching
    draws with pair resampling (matching draws are not independent biological replicates; the file says so).
15. Results are mixed: functional_256 minus CpGPT-512 > 0 on 7/13 sets (CI excludes 0 for 4 positive, 3 negative); the clock sets favour CpGPT.

Overlaps and duplicates between positive sets (computed on the 408,399 universe; `manifests/input_files_sha256.json`)
16. smoking and lung cancer: 225 of 242 lung-cancer CpGs (93%) are also in the smoking set; BMI and T2D: 65 of 71 T2D CpGs (92%) are in BMI; smoking and
    BMI share 2,082; smoking and colorectal 227; colorectal and prostate 676; age and RA 147. Tasks are therefore not independent evidence.
17. Age EWAS set vs clocks: Horvath 2, PhenoAge 3, Hannum 0 shared CpGs (clocks are essentially disjoint from the age EWAS residue); Horvath-PhenoAge 40;
    Horvath-Hannum 6; Hannum-PhenoAge 6. No duplicate CpGs within any set (verified).

Provenance and leakage
18. No patient-derived data enter any panel representation or any EWAS/clock/context label. EWAS source cohorts are not recorded per CpG; whether EWAS
    Catalog/Atlas studies used TCGA or the benchmark's GEO cohorts is NOT documented. The RA set includes the GSE42861 study and the SZ set mirrors GSE147221
    by design (`data/bio_annotations/README.md`), i.e. it overlaps the benchmark's external disease cohorts (not its training cohort).
19. exp2's label (mean beta over TCGA discovery patients, 11,929 of 408,399 loci) and the co-methylation test (held-out patients' betas) use benchmark
    patient data in the label/evaluation, not in the representation.
20. EWAS probe design (450K/EPIC array content) is biased towards promoters/CpG islands: a covariate shared by positives and (array-based) negatives, flagged,
    not treated as circularity. Probe-type imbalance in exp5 base variant (see 11).

bioval-v2 specifics
21. Frozen CIs: 22 chromosome blocks, B=1000: coarse and approximate for inter-chromosomal pairs (declared). Primary and sensitivity runs recorded
    different repo heads (`f0b00ac3` primary; `7d2153cf` sensitivity/exploratory); protocol commit `27d8bce1` identical.
22. 849 regulatory all-zero rows: excluded from cosine endpoints for all arms (union rule), kept in probes (FANTOM5 positives with zero rows: 23 of 23,150).
23. Circularity: FANTOM5 and compartments PARTIALLY_RELATED (shared enhancer chromatin evidence; GC-oriented E1), Micro-C intra PARTIALLY_RELATED (CTCF/cohesin
    tracks vs loop anchors); cCRE/island/gene-region/TSS targets are SOURCE_RETENTION for legacy functional and were rejected as endpoints. No cCRE, gene-region
    or TSS result exists anywhere (so no cCRE-vs-candidate circularity note is needed in results; if added later, cCRE classes are built from DNase, H3K4me3,
    H3K27ac and CTCF, i.e. from the candidate's own mark families).
24. CpGPT-large is methylation-supervised; Loyfer tasks therefore compare a methylation-trained arm with chromatin-only arms. Micro-C effect sizes are small (AUROC 0.50-0.60).

## 6. (e) Coverage gap matrix (main panel)

HAVE = native/main-panel arm has results; PROXY_VARIANT_ONLY = only a non-panel variant (named); MISSING = no result. Per task: `paper/registry/bio_task_coverage_matrix.csv`
(116 tasks) and `coverage_*` columns of `bio_task_matrix.csv`.

| suite (tasks) | regulatory_histone_dnase_v1 | cpgpt_large_locus (native 512D) | deepcpg_dna_locus (native 128D) | what would be required |
|---|---|---|---|---|
| v2_frozen (13) | HAVE | HAVE | HAVE | nothing |
| v2_sensitivity (12) | HAVE | HAVE | HAVE | nothing |
| v2_exploratory_posthoc (7) | HAVE | HAVE | HAVE | nothing |
| legacy_bio_validation (25: 4 context, 16 sets/clocks, 4 coefficient, locality) | MISSING | HAVE | HAVE | run the candidate on the legacy probes, but only after the design is replaced (blocked, matched) |
| encode_exp5 base and probe (13 + 13 evaluable sets) | MISSING (no histone/DNase arm) | HAVE | HAVE | add `regulatory_histone_dnase_v1` to exp5 (same pairs, folds, bootstrap) |
| encode_ewas_matched_single_draw (13) | PROXY_VARIANT_ONLY: `add/assay/Histone ChIP-seq` (18 context columns + 1,959 histone tracks, SVD256; no DNase; includes context) | PROXY_VARIANT_ONLY: `fm/cpgpt_locus_large` (compact 256) | MISSING | candidate and native arms through exp5; this table is superseded by exp5 |
| encode_exp2 mean-beta decodability (1) | PROXY_VARIANT_ONLY: raw histone-only feature subset (not SVD, no DNase) | HAVE | HAVE | candidate SVD-256 on the same 11,929 loci |
| encode_co_methylation (1) | PROXY_VARIANT_ONLY: histone+context arm | PROXY_VARIANT_ONLY: compact 256 | MISSING | candidate, native CpGPT-512, DeepCpG on the same query/neighbour protocol |
| umap_neighborhood (18) | MISSING | HAVE | HAVE | descriptive only; low priority |

Answers to the two direct questions: (i) `regulatory_histone_dnase_v1` appears in no biological probe besides bioval-v2 (only proxies as above).
(ii) Native CpGPT-512D appears in bioval-v2, legacy `bio_validation`, exp5, exp2; the single-draw EWAS table and co-methylation have only the compact 256D variant.
New experiments would be needed for: the candidate on any EWAS/disease/age/clock/smoking probe (best route: add it to exp5, same matching/folds/bootstrap), the candidate
(and sequence arms) on gene-region/promoter/cCRE/TSS and tissue-specific targets (no results exist; frozen protocol treats context/cCRE/gene-region/TSS as circular),
and an AUPRC for exp5 (derivable from the existing out-of-fold predictions file). Not run.

## 7. (f) Data dictionary

`paper/registry/bio_task_registry.json` holds the column dictionary, enumerations (biological_family, status, circularity, metric, rigor grade, coverage), file
list and matrix-column conventions. Files:

| file | content |
|---|---|
| `bio_task_registry.csv` | long format, one row per task x representation x metric (x stat); 2,130 rows, 41 columns |
| `bio_task_paired_contrasts.csv` | paired deltas exactly as published (candidate vs comparators in bioval-v2; functional_256 vs others in exp5/exp2); 853 rows |
| `bio_task_matrix.csv` | one row per task headline stat (104 rows); AUROC, AUPRC, Spearman, R2, enrichment in separate columns per arm (cand, cpgpt, deepcpg, funcleg = legacy), CIs, N, coverage, proxy, delta columns, circularity, rigor grade |
| `bio_task_coverage_matrix.csv` | task x main-panel coverage (116 tasks) |
| `per_task_audit.csv` | fairness audit and rigor grade per task (122 rows incl. 6 unevaluable exp5 sets) |
| `bio_task_inventory.csv` | experiment-level inventory with file paths and sha256 |
| `unscored_tasks.csv` | probes that errored or lacked support |
| `manifests/input_files_sha256.json` | sha256 of all 144 provenance files, set sizes, overlaps, context counts |
| `manifests/registry_outputs_sha256.json` | sha256 of the registry outputs and scripts |

Conventions: scores are never imputed (blank = not computed); `circularity_level` is the level for the row's own representation, with the three per-reference columns
alongside; delta CIs exist only where the source file contains them (`delta_flag = "no paired CI available"` otherwise); legacy AUROC is the mean of fold AUROCs;
the legacy co-methylation means are computed by the registry script from the per-pair CSVs and flagged as derived.

## 8. (g) Provenance

- Repo HEAD at build: `fc272cacedc27a121ad8c820c62052204a4d9f26` (branch main); protocol tag `bioval-v2-protocol-freeze-v1` = `27d8bce1b2ce391cb4ffb0d89eb15660710c2b33`;
  bioval-v2 run heads recorded in the run manifests: `f0b00ac3` (primary) and `7d2153cf` (sensitivity/exploratory).
- sha256 of the registry outputs (also in `paper/registry/manifests/registry_outputs_sha256.json`):

| file | sha256 |
|---|---|
| paper/registry/bio_task_registry.csv | 6d685a77ed9c09c32d2627f1fd4fce56e2bb502f922bf5827acdb3eb63a2e425 |
| paper/registry/bio_task_registry.json | ac87315652cc93368b4a64578bc58ee0d5f958531332050b83c3f263e266cf6e |
| paper/registry/bio_task_matrix.csv | a7bc37be94d44c65090f9cc3f6299ca8f333078b0a0041146cac44f6ea19a4ac |
| paper/registry/per_task_audit.csv | ccca2b1c0a51e5cd3f0098244ee0b9db255040c77ae7451a8cbfefade3ce1a38 |
| paper/registry/bio_task_paired_contrasts.csv | d665514e40c4f92fbe9ffd6bf679455b727242c7691112c4311a42172347f92e |
| paper/registry/bio_task_coverage_matrix.csv | 4adf7983929e5bfaeacdf559658b87a37be85336f6d69d9c2dc68e9a07d5fdbb |
| paper/registry/bio_task_inventory.csv | ade538b6fdb789a19f1b40398ff5ecb26b6b0050088292af42fe5da342e6c54c |
| paper/registry/unscored_tasks.csv | 5e14e590dd53b2bfea8dd0fed9a221174b028bbf9d7b44e20a45efdd326c73fc |
| paper/registry/manifests/input_files_sha256.json | 184a26f9801cf20be20b74ab73586583fd1eda8172f2d629f1f2145d64a76d84 |

Scripts: `paper/scripts/build_bio_task_registry.py`, `paper/scripts/validate_registry.py` (`--determinism` re-runs the build and compares hashes; passed), and the
synthetic unit test `paper/scripts/test_build_bio_task_registry.py`. Validation result: 0 errors, 0 warnings.

## 9. (h) Open questions for the user

1. Should the candidate be added to exp5 (same pairs/folds/bootstrap) and to a redesigned phenotype probe? Until then no phenotype/EWAS/clock claim can include it.
2. Legacy functional naming: which store is "the" legacy functional arm for the paper (campaign `full` SVD-256, as in bioval-v2/exp5, or the older PCA store used in
   `bio_validation`)? The registry keeps both, labelled by variant.
3. Is the histone+context campaign arm acceptable as a labelled proxy for the candidate in supplementary material, given it contains the 18 context columns and no DNase?
4. EWAS sets: most CpGs of the age/RA/SZ sets fall outside the 408,399-CpG universe. Keep them as-is, or restrict to well-powered sets (age and clocks, smoking, BMI, colorectal,
   prostate, breast, lung)? The lung-cancer set is 93% smoking CpGs.
5. Is exp5's AUPRC (derivable from `oof_predictions.parquet`) wanted? It requires a recompute step, which I did not run.
6. Should gene-region/promoter/cCRE/TSS tasks be built for the candidate despite the frozen CIRCULAR classification (the candidate does not take those as inputs, but cCRE classes derive
   from DNase/H3K4me3/H3K27ac/CTCF)? If yes they must be labelled SOURCE_RETENTION or PARTIALLY_RELATED, not independent validation.
7. Documentation of EWAS source cohort overlap with TCGA/GEO benchmark cohorts is absent; can you provide or authorise a check against the EWAS Catalog study table?
8. Is Loyfer's negative primary to be reported as a headline (candidate -0.063 vs +0.106 / +0.115)? The registry flags it; the exploratory post-hoc analyses (including the
   separate failure audit, which was not read) are not primary evidence.
9. Untracked file `CpGRepresentationBenchmark_plot.ipynb` appeared at the repo root during this task; it was not created by this work.

## 10. Optional: which tasks could feed which figure (data only)

- Candidate-vs-panel summary: v2_frozen rows (4 primaries + secondaries) from `bio_task_matrix.csv`.
- Phenotype/EWAS panel (without candidate): `encode_exp5_base` and `encode_exp5_probe` AUROC with CIs for CpGPT-512, DeepCpG-128, legacy functional.
- Coverage figure: `bio_task_coverage_matrix.csv`.
- Circularity/rigor overview: `per_task_audit.csv` (rigor_grade x circularity columns).
