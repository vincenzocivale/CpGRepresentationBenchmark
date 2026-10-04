# biological_validation_v2: pre-execution audit (phase 1)

Date: 2026-10-04. Scope: audit only. No endpoint metric was computed on any embedding, no freeze artifact was modified or
regenerated, no freeze command was run, the TCGA test set was not read, no GPU, no training, no commit.
Audit tool (read-only, separate from freeze code): `scripts/bioval_v2/preexec_audit.py` (tests: `tests/test_bioval_v2_preexec_audit.py`).

## 1. Freeze audit

| Item | Value |
|---|---|
| HEAD at audit | `29b1e441efb1ddec774736fdbf3c4d985bca9d4d` (working tree clean before this audit) |
| Protocol freeze commit | `27d8bce1b2ce391cb4ffb0d89eb15660710c2b33` ("Freeze biological_validation_v2 protocol v1 (pre-embedding)") |
| Tag | `bioval-v2-protocol-freeze-v1` (annotated; tag object `fa328fcde47c98863e1c745593e2d0ba5cb61cfe`, peels to `27d8bce`) |
| Registration commit | `6338e7b` (doc-only: writes the freeze SHA/tag into section 8 of the protocol doc) |

Differences between the freeze commit and HEAD (`git diff --name-status 27d8bce HEAD`, 59 paths), classified:

- Protocol/config/manifest/pair-list/split/code paths of bioval v2 (`docs/bioval_v2_prep/`, `configs/biological_validation_v2/`,
  `src/cpg_repr_benchmark/biological_validation_v2/`, `scripts/bioval_v2/`, `scripts/build_bioval_v2_covariates.py`,
  `scripts/fetch_cpgislandext_hg38.py`, `tests/test_bioval_v2_*`, all git-tracked `data/derived/bioval_v2/**` manifests):
  **no difference** (empty `git diff`).
- `docs/BIOLOGICAL_VALIDATION_V2.md`: **doc-only**, 2 lines (commit `6338e7b`, section 8 registry of the freeze SHA/tag). Protocol text unchanged.
- Other commits after the freeze (`74cbd75` ... `29b1e44`), none touching bioval v2:
  - data: none under `data/derived/`.
  - config: `configs/frozen/regulatory_histone_dnase_v1.json` (candidate manifest, added), `configs/representations/{regulatory,confirmation_matrix}.yaml`,
    `configs/experiments/regulatory_confirmation_matrix/**` (matrix + 18 run configs, added).
  - code (masking benchmark side only): `scripts/run_masking_benchmark.py`, `src/cpg_repr_benchmark/experiments/{confirmation_matrix,eval_layout,guards}.py`,
    new confirmation scripts (`audit_/analyze_/report_/check_/run_/verify_*`), tests.
  - doc-only: `docs/REGULATORY_*` (confirmation protocol/fairness/results).
  Note the candidate freeze (`74cbd75`) is later than the bioval-v2 freeze; the protocol is therefore independent of the candidate (pre-embedding preserved).
- Untracked, created by this audit: `scripts/bioval_v2/preexec_audit.py`, `tests/test_bioval_v2_preexec_audit.py`, this file.

Manifest verification (read-only):

- `MANIFEST_checksums.json`: **121/121 files** match sha256 and size; 0 missing; 0 extra files on disk.
- Protocol doc section 7 (14 manifests + 17 data files = 31 hashes, including the abbreviated `..._dropped_positives`): **31/31 match**.
- Git-tracked manifests (`FROZEN*.json`, `MANIFEST*.json`, ...) are byte-identical to the freeze commit.
- `scripts/verify_frozen_candidate.py` (read-only mode, no `--write`): **OK** (catalog hash, feature-set hash, fit-loci hash, compressor manifest hash,
  store sha256, 200-locus re-transform max abs diff 0.0).
- `tests/test_bioval_v2_*`: 26 passed, 2 skipped.
- Re-derived consistency checks on frozen lists (counts only): chromosome->fold map recomputed with `chromosome_blocked_folds(22 chrs, 5, seed 17)` equals
  the `fold` column of every H1/HFFc6 intra pair file and the FANTOM5 membership files; each positive's matched control is in the same fold.
- Not re-verifiable here (external, documented in the doc): `data/bio_annotations/cpgIslandExt.hg38.txt.gz` (not versioned; re-fetchable by hash).

Freeze commands that MUST stay unused in phase 2 (never run, never import for side effects):
`scripts/bioval_v2/freeze_4dn_pairs.py`, `freeze_fantom5_matching.py`, `loyfer_freeze_pairs.py`, `loyfer_profile_similarity.py` (writes the test pairs/strata),
`build_checksum_manifest.py` (overwrites `MANIFEST_checksums.json`), all `prepare_*.py` (`prepare_4dn`, `prepare_fantom5`, `prepare_loyfer`,
`prepare_pmd_decato2020`, `prepare_replication_timing`), `loyfer_{aggregate,extract,cpg_index,manifest,markers,markers_reconcile,provenance}.py`,
`validate_4dn_assembly.py`, `fourdn_rangefile.py`, `gc_track_hg38.py`, `scripts/build_bioval_v2_covariates.py`, `scripts/fetch_cpgislandext_hg38.py`,
`scripts/verify_frozen_candidate.py --write`, and `biological_validation_v2.matching.match_controls/match_pairs/build_covariates`, `pairs.sample_pairs`
(they generate lists; phase 2 must only read the frozen parquet files). The planned execution path below reads frozen files only and writes nowhere under
`data/derived/bioval_v2` (the audit script refuses such `--out`).

## 2. What exists / gaps

**Implemented (definitions only, array-in/array-out; none loads an embedding store):**
`metrics.py`: `spearman_profile_vs_embedding_sim`, `pearson_profile_vs_embedding_sim`, `auroc_binary`, `contact_vs_noncontact_auroc`, `average_precision`,
`stratified_by_distance` (default bins 0,1e3,1e4,1e5,1e6,1e7,inf; NOTE the doc strata are <1kb,1-10kb,10-100kb,100kb-1Mb,>1Mb,inter, and the frozen
file already has a `stratum` column), `compartment_eigenvector_corr`, `knn_enrichment` (permutation within chromosome), `bootstrap_ci_by_block` (single
statistic, chromosome blocks, mean/any stat, percentile CI), `PRIMARY_METRICS` (descriptive dict, slightly less specific than the doc).
`splits.py`: `chromosome_blocked_folds`, `pair_split_by_chrom` (drop policy), `assert_no_leak`. `coordinates.py`: `load_benchmark_universe`,
`require_full_universe_coverage`. Frozen-store reading in the benchmark: `representations/hdf5_store.py` (`inspect_representation_h5`, `HDF5RepresentationStore`:
argsort/searchsorted on `/cpg_idx`, rows read as float32; stores' `/cpg_idx` are NOT sorted, so the mapping is mandatory).

**Not implemented (gaps; no CLI/evaluation entry point exists for any endpoint):**
1. Embedding loader for bioval (cpg_idx -> row, float16 -> float32), identical to `hdf5_store`; fail-fast `require_full_universe_coverage`.
2. Cosine score computation for pairs (Loyfer, Micro-C) and paired delta-cosine.
3. Endpoint drivers: Loyfer primary + strata + sensitivities (Pearson vs Spearman, `mask_lenient`, U25, single-sample groups, >=30 shared groups) and marker kNN;
   Micro-C H1/HFFc6/inter AUROC, delta-cosine, distance strata, E1 probe and GM12878 compartments; FANTOM5 membership probe AUROC (+win500, TSS pool as
   sensitivity), activity-similarity Spearman; RT probe (+R2, per-line, residualised, normal vs tumour); PMD.
4. Chromosome-blocked probes: ridge (RT, E1) and linear classifier (FANTOM5), inner CV on train chromosomes only. `bio_validation/probes.py` is legacy
   (random KFold, `RidgeCV` logspace(-2,4,13)) and is not v2-compliant.
5. Paired block bootstrap of the contrast Delta (22 chromosomes, B=1000, seed 17, same resampled chromosomes for both arms), bootstrap p-value with
   (1+k)/(1+B) correction, **Holm over the 4 primary endpoints per contrast**: no Holm implementation exists anywhere in the repo; `bootstrap_ci_by_block` handles one arm only.
6. Output layout, run manifest recording the representation identity (sha256, path, dim, dtype, catalog id, frozen version, protocol tag/commit).
7. Test coverage of the above (none yet).

Pre-registered rules the code applies from frozen files: universe/eligibility are in the frozen lists (386,268 Loyfer pairs after the >=20-shared-groups
and Pearson-defined filters; Micro-C eligibility/matching and `fold`/`matched_to`; FANTOM5 976 dropped positives; RT 54 excluded CpGs = NaN `rt_consensus_z`).
Phase 2 must consume these files as-is (no re-filtering, no per-representation intersection).

**Frozen counts verified (read-only):**

| Item | Expected | Found |
|---|---|---|
| Loyfer pairs | 386,268 (<1kb 37,315; 1-10kb 38,167; 10-100kb 38,785; 100kb-1Mb 38,819; >1Mb 38,914; inter 194,268) | identical; min shared groups 20; 0 NaN Pearson; 0 duplicate pairs |
| Loyfer cell groups | 39 | 39 (`loyfer_celltype_beta.h5`, `groups`) |
| H1 intra 10 kb | 2,689,921 pos = ctrl | 5,379,842 rows, 2,689,921 label 1 / 2,689,921 label 0; per-fold pair rows 1,134,070/1,330,950/912,576/796,590/1,205,656 |
| HFFc6 intra 10 kb | 3,724,445 | 7,448,890 rows, 3,724,445 / 3,724,445 |
| H1 inter 1 Mb (exploratory) | 9,931 after drop | 124,266 rows, of which fold>=0: 9,931 pos + 9,931 ctrl (fold -1 = dropped, must be excluded) |
| FANTOM5 membership | 11,575 pairs | 23,150 rows = 11,575 / 11,575; 976 dropped positives; win500 26,880 pairs; TSS pool 3,349 pairs (sensitivity only) |
| FANTOM5 activity | 171,017 | 171,017 (100,000 inter + 71,017 intra) |
| RT | 408,345 eligible; 15 lines | 408,345 non-NaN `rt_consensus_z` (54 excluded); 15 `rt_<LINE>` columns |

**Bootstrap protocol (doc section 0):** resample the 22 chromosomes (block = chromosome of `cpg_i`; inter pairs approximated), B=1000, seed 17, same chromosomes for the
two arms (paired), 95% percentile CI per metric and per Delta; two-sided p = min(1, 2*min(P*(D<=0), P*(D>=0))) with (1+k)/(1+B); Holm at alpha 0.05 over the 4 primary endpoints per
contrast; claim only if adjusted p < 0.05 and Delta has the expected sign. Secondary/sensitivity/exploratory: descriptive, BH q informative only.

**Designations:** PRIMARY (Holm family): Loyfer profile (pooled >1 Mb + inter, equal stratum weight), H1 intra 10 kb AUROC, FANTOM5 membership AUROC (frozen linear probe),
RT consensus ridge-probe Spearman. SECONDARY: Loyfer marker kNN, GM12878 compartments + H1 E1 probe, FANTOM5 activity similarity, Delta-cosine. SENSITIVITY: HFFc6, win500,
TSS-pool, per-line RT, etc. EXPLORATORY: H1 inter, PMD.

## 3. Needs user decision (not decided, not implemented)

1. **Loyfer primary definition.** The ratified protocol (sections 0 and 2.1) defines the primary as Spearman pooled over the **>1 Mb + interchromosomal strata, equal weight per stratum**
   (38,914 + 194,268 pairs; short strata reported separately as proximity-dominated). The task text reads as all 386,268 pairs. Follow the frozen doc (default) or deviate (then exploratory)?
2. **Zero embedding rows.** `regulatory_histone_dnase_v1` has 849 all-zero rows, so cosine is undefined for pairs touching them (Loyfer 1,593 pairs; H1 intra 15,474; HFFc6 22,205;
   H1 inter 2,775 incl. dropped; FANTOM5 activity 176 pairs) and the protocol forbids per-representation narrowing. Rule needed: cosine := 0 for zero vectors, or NaN and excluded for that arm only
   (breaks pairing), or excluded for all arms. For probe endpoints the zero rows are plain features (FANTOM5 23 rows, RT 849 eligible rows, 32 on chr20-22).
3. **Probe specification** is not frozen in detail: model (logistic regression vs linear SVM for FANTOM5; ridge for RT), feature standardisation, alpha/C grid, inner-CV scheme (the doc says
   "regularisation chosen by inner CV on train chromosomes only", no grid). The task says no tuning unless frozen: the grid itself must be fixed before running.
4. **FANTOM5 primary score.** Doc: frozen linear probe with chromosome-blocked CV (out-of-fold scores), not cosine. Confirm; pair-level or locus-level AUROC (one fold per pair is available).
5. **Comparison list and kNN k** are explicitly "to be registered at launch": confirm contrasts (regulatory_histone_dnase_v1 vs each of the 3 arms; any between the other arms) and k.
6. **Micro-C** primary AUROC pools all 5.38 M rows (positives vs controls, not paired AUROC per match); Delta-cosine is the paired effect size. Confirm distance-strata set for 4DN (doc: per distance bin, `stratified_by_distance`).
7. **Paired delta where the design does not allow it** (e.g. FANTOM5 probe CV folds identical across arms by frozen chromosome folds: allowed; Loyfer/Micro-C: same pairs: allowed; RT: same loci: allowed).
   Confirm that bootstrap block for inter pairs stays `chrom_i` (documented caveat).
8. **R2 definition** for RT (OOF pooled vs per-fold mean) and per-fold reporting (not in protocol).
9. **Output location/layout** (protocol silent): proposal `outputs/biological_validation_v2/<protocol_tag>/<arm>/`, JSON per endpoint + run manifest. Confirm.
10. **Holm / p-value for single-arm** results: Holm is defined for contrasts only; single-arm metrics get CIs only (confirm no null-hypothesis test vs 0).

## 4. Store pre-checks (inputs only; no endpoint metric)

Loader contract: `/cpg_idx` int64 + `/embedding`, canonical keys in all four stores (`inspect_representation_h5` resolves `cpg_idx`/`embedding`), no duplicate ids, `/cpg_idx` **not sorted** in
any store (use argsort + searchsorted like `HDF5RepresentationStore`), float16 cast to float32 on read. Full chunked pass (50,000 rows, niced).

| Arm | sha256 vs matrix.yaml | dim | dtype | rows | NaN / Inf | all-zero rows | max abs |
|---|---|---|---|---|---|---|---|
| regulatory_histone_dnase_v1 (`data/cache/representations/regulatory_histone_dnase__global_svd256__discovery_chr1_19.h5`) | OK `fa3a8163...` (== frozen manifest) | 256 | float32 | 408,399 | 0 / 0 | **849** (as documented) | 31.6 |
| functional_annotations_pca (`outputs/encode_atlas_v1/embeddings/full_a18b869b.h5`) | OK `3d99fe5b...` | 256 | float16 | 408,399 | 0 / 0 | 0 | 110.7 |
| cpgpt_large_locus (`data/cache/representations/cpgpt_locus_large.h5`) | OK `c08138cd...` | 512 | float16 | 408,399 | 0 / 0 | 0 | 28.5 |
| deepcpg_dna_locus (`data/cache/representations/deepcpg_dna_locus.h5`) | OK `e7280d5f...` | 128 | float16 | 408,399 | 0 / 0 | 0 | 5.0 |

Coverage of `/cpg_idx` over the frozen CpG sets (unique CpGs needed / present; identical for all four stores, **0 missing, no blocker**):

| Endpoint CpG set | needed | present (each store) |
|---|---|---|
| Loyfer pairs (386,268) | 343,584 | 343,584 |
| Micro-C H1 intra 10 kb | 394,271 | 394,271 |
| Micro-C HFFc6 intra 10 kb | 394,404 | 394,404 |
| Micro-C H1 inter 1 Mb (exploratory) | 100,816 | 100,816 |
| FANTOM5 membership (primary) / win500 / TSS pool | 23,150 / 53,760 / 6,698 | all |
| FANTOM5 activity pairs | 11,490 | 11,490 |
| RT eligible / RT universe | 408,345 / 408,399 | all |
| Compartments H1 / HFFc6 / GM12878 (with E1) | 407,903 / 408,223 / 408,275 | all |
| Loyfer markers (reconciled rows, unique) / PMD universe | 1,600 / 408,399 | all |

Identity/version to record in every output: arm id, store path, sha256 (re-computed at run time and compared to matrix.yaml), dim, dtype, n rows, catalog entry
(`configs/representations/regulatory.yaml#regulatory_histone_dnase`, `confirmation_matrix.yaml`, `future_models.yaml`), for the candidate the frozen
manifest `configs/frozen/regulatory_histone_dnase_v1.json` (version tag `regulatory_histone_dnase_v1`, `candidate_frozen`, compressor/feature-set hashes), the protocol tag
`bioval-v2-protocol-freeze-v1` + commit `27d8bce`, repo HEAD at run, and `MANIFEST_checksums.json` sha256.
Caveat: regulatory rows of chr20-22 are out-of-fit (transformed only); functional/cpgpt-compact stores are fit transductively on all loci (documented, not an audit failure).

## 5. Planned phase-2 execution (to be written and unit-tested in phase 2 before any run; nothing below exists yet except step 0)

```bash
# 0. (done in phase 1, repeat immediately before running; read-only)
export PYTHONPATH=$PWD/src; PY=~/miniconda3/envs/cpg-repr-benchmark/bin/python
nice -n 10 $PY scripts/bioval_v2/preexec_audit.py verify --out <scratch>/verify.json
nice -n 10 $PY scripts/bioval_v2/preexec_audit.py stores --out <scratch>/stores.json
git rev-parse HEAD; git rev-parse bioval-v2-protocol-freeze-v1^{commit}   # expect 27d8bce
# 1. (after user decisions in section 3 and code review) one evaluation driver, one arm per invocation, CPU only, OMP_NUM_THREADS<=8
nice -n 10 $PY scripts/bioval_v2/evaluate_endpoints.py --arm <id> --store <store.h5> --out outputs/biological_validation_v2/<tag>/<arm> --endpoints all   # NEW, to implement
# 2. contrasts + Holm over the 4 primaries, reading only the per-arm outputs
nice -n 10 $PY scripts/bioval_v2/contrast_holm.py --reference regulatory_histone_dnase --comparators functional_annotations_pca cpgpt_large_locus deepcpg_dna_locus --root outputs/biological_validation_v2/<tag>   # NEW, to implement
```

## 6. Verdict

Data, hashes, frozen lists and stores: **GO** (all verified, 0 missing loci, no non-finite values). Evaluation code: **absent**, and items 1-4 of section 3 are open methodological decisions.
Recommendation: proceed to phase 2 only after the user resolves section 3 (at minimum items 1, 2, 3, 5) and the new driver/contrast code is implemented and unit-tested on synthetic data.
