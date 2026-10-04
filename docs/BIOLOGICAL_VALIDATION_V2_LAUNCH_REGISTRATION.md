# biological_validation_v2: launch registration (phase 2a)

Status: written BEFORE any bioval v2 endpoint metric was computed on any real embedding. To be committed before the first real run; the
pre-run gate refuses to run unless this file is committed and unmodified. The frozen protocol (`docs/BIOLOGICAL_VALIDATION_V2.md`,
freeze tag `bioval-v2-protocol-freeze-v1` = commit `27d8bce1b2ce391cb4ffb0d89eb15660710c2b33`) is NOT amended by this document: it only fixes
what the protocol explicitly left "to be registered at launch" (arm/contrast list, kNN k) and the implementation details the protocol was silent on.
Every such default is marked **[registered at launch]**; items needing the reader's attention are marked **[FLAG]**.
Phase 1 audit: `docs/BIOLOGICAL_VALIDATION_V2_EXECUTION_AUDIT.md`. Code: `src/cpg_repr_benchmark/bioval_v2_launch/{block_bootstrap,embedding_eval,chrom_probes,launch_endpoints,launch_gate}.py` (a new package outside the definitions-only `biological_validation_v2`, whose no-h5py/no-representations test must keep passing), `scripts/bioval_v2/run_evaluation.py`, `tests/test_bioval_v2_launch.py`.

## 1. User decisions D1-D4 (registered exactly)

**D1 Loyfer primary.** Spearman(Pearson similarity of WGBS beta profiles across the 39 groups [frozen column `loyfer_pearson`], cosine similarity of embeddings), POOLED over the
`>1Mb` (38,914) and `interchromosomal` (194,268) strata with EQUAL STRATUM WEIGHT, as ratified. All six strata (`<1kb`, `1-10kb`, `10-100kb`, `100kb-1Mb`, `>1Mb`, `interchromosomal`) are reported separately
(unweighted Spearman, descriptive). The Spearman over all 386,268 pairs is DESCRIPTIVE only. Marker kNN (k = 10) is secondary and never influences the primary.
- **[registered at launch] [FLAG] operational meaning of "equal stratum weight".** The frozen code has no weighted metric. Registered: weighted Spearman = weighted Pearson correlation of the weighted mid-rank fractions of the two variables
  (pooled over the two strata), item weight = 1/n_stratum (after exclusions), so each stratum carries total weight 1/2. With unit weights this equals `scipy.stats.spearmanr` / `metrics.spearman_profile_vs_embedding_sim` (tested; run-time cross-check on every unweighted Loyfer statistic).
  Descriptive alternative reported next to it: arithmetic mean of the two per-stratum Spearman values (`spearman_mean_of_gt1Mb_and_inter`). The primary is the weighted-rank pooled statistic; the alternative never replaces it.
- Loyfer sensitivities implemented: profile similarity = frozen `loyfer_spearman` instead of Pearson; Pearson (instead of Spearman) correlation between profile Pearson and cosine; pairs with < 30 shared groups removed (`n_shared_groups >= 30`, weights recomputed per stratum).
  **[FLAG] NOT run in this launch:** `mask_lenient`, single-sample-group exclusion and U25-only for Loyfer pairs (they require re-deriving the profile-similarity targets from `loyfer_celltype_beta.h5`, which is a target-generation step, not an embedding evaluation; such a re-derivation would be a new freeze artifact). Declared in the protocol, reported here as deferred.
- Marker kNN: k = 10 **[registered at launch]**; exact top-k cosine neighbours over all candidate loci (universe minus the zero-row loci of D2; self excluded); marker label = `cell_type` of reconciled marker rows with `in_present_group`; set `github_U25_U250` (secondary, U25 priority for duplicate CpGs; CpGs with more than one label dropped and counted),
  sensitivities `github_U25_only` and `paper_lifted_U25_U250`; expected same-label fraction per label = (number of marker loci with that label) / (number of candidate loci); enrichment = mean(observed same-label neighbour fraction) / mean(expected) over marker loci (frozen `metrics.knn_enrichment`, n_perm = 0);
  one-sided within-chromosome permutation p-value from 999 permutations, seed 17 (see section 8 for the frozen-function defect that forced a separate permutation routine). CI/contrast: block bootstrap over the chromosome of each marker locus. Secondary, descriptive, outside the Holm family.

**D2 Zero-row rule.** Embeddings with all-zero rows have undefined cosine; only the regulatory store has any (849 loci). For every COSINE-based endpoint (Loyfer, Micro-C H1/HFFc6 intra and H1 inter, FANTOM5 activity-profile similarity, kNN) every pair/row touching a locus that is
all-zero in ANY of the 4 stores is EXCLUDED, so the retained set is IDENTICAL across arms; no per-representation intersection. Exclusion counts are written to every endpoint result and to the run manifest.
Verified read-only in the dry-run (counts only): union of zero loci = 849 (regulatory 849, the other three 0); Loyfer 1,593 excluded; H1 intra 15,474; HFFc6 intra 22,205; FANTOM5 activity 176 (all as expected by the user); H1 inter (fold >= 0 rows, 19,862) 488; kNN candidates = 408,399 - 849.
Linear-probe endpoints (FANTOM5 membership, RT, and the secondary/exploratory probes) do NOT use cosine and keep ALL frozen CpGs/pairs (zero embeddings are valid features). No cosine=0 sensitivity is run.
- **[registered at launch]** Micro-C: AUROC excludes individual rows touching a zero locus; the paired delta-cosine uses only matched (positive, control) pairs in which BOTH members survive (counts reported).
  (AUROC is computed on rows, so the retained row set is identical for all arms; the matched-pair set is identical for all arms.)

**D3 Probes** (identical for every arm; nothing tuned outside the training chromosomes of each fold; code `chrom_probes.py`).
- Outer folds = the FROZEN chromosome-blocked folds (5 folds, seed 17): the `fold` column of the Micro-C/FANTOM5/compartment files; for RT and PMD the frozen `chrom_to_fold` map in `FROZEN_ENDPOINT.json` (asserted equal to `chromosome_blocked_folds(22 chromosomes, 5, 17)`).
- StandardScaler (mean, std with ddof = 0, zero std -> 1) fitted on the outer-train rows only.
- Inner CV **[registered at launch]**: leave-one-frozen-fold-out over the 4 outer-train folds (4 inner folds, chromosome-blocked by construction, consistent with the frozen fold structure); inner CV only ever sees outer-train rows; the outer scaler (fitted on outer-train) is reused inside inner CV (no test-chromosome statistics are involved).
- RT (and every other ridge probe): Ridge, alpha in {1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4}, **[FLAG]** in scikit-learn convention (loss = SSE + alpha*|w|^2, unpenalised intercept, NOT divided by n; with n of order 3e5 the grid is therefore weak regularisation; kept as specified, verified numerically against `StandardScaler + sklearn.Ridge`). Selection: pooled inner out-of-fold SSE (MSE), ties -> larger alpha. Per-target alpha for multi-target fits (the 15 per-line RT targets are selected independently). Solved exactly from Gram matrices.
- FANTOM5 membership: `LogisticRegression(penalty=l2, solver=lbfgs, class_weight=None, max_iter=5000)`, C in {1e-4, 1e-3, 1e-2, 1e-1, 1, 10}; selection = mean over the 4 inner folds of AUROC of the inner out-of-fold decision scores, ties -> smaller C. Convergence (max `n_iter_`, number of non-converged fits) is recorded.
- Primary FANTOM5 = AUROC of the pooled out-of-fold decision scores over the frozen rows (each CpG row scored by the model of its own held-out fold); locus-level (one row per CpG; positives and matched controls are rows of the same file with `fold`).
- RT primary = Spearman of pooled out-of-fold predictions vs `rt_consensus_z` (mean z of the 15 lines, 408,345 eligible CpGs; the 54 NaN rows excluded for all arms); also R2 = 1 - SSE/SST on the pooled out-of-fold predictions (SST around the mean of the pooled target) and per-fold Spearman/R2 (stored in the stat `meta`); 15 per-line Spearman/R2 as sensitivity.
- PMD (exploratory) **[registered at launch]**: ridge regression on the 0/1 label `pmd_in_any` with the same grid/inner CV, AUROC of out-of-fold scores (a ridge-regression probe instead of logistic for cost; 408k rows x 128-512 dims).

**D4 Contrasts.** Reference = `regulatory_histone_dnase_v1` (matrix id `regulatory_histone_dnase`) vs each of `functional_annotations_pca`, `cpgpt_large_locus`, `deepcpg_dna_locus` (3 contrasts; no contrast among the three comparators).
Paired chromosome-block bootstrap: 22 chromosome blocks (`chr1..chr22`, block of a pair = chromosome of `cpg_i`; inter-chromosomal pairs therefore approximated: protocol caveat), B = 1000, seed 17, ONE shared draw table (`rng = Generator(PCG64(17)); draws = rng.integers(0, 22, size=(1000, 22))`, chromosome order = `np.unique`; bit-identical to the block draws of `metrics.bootstrap_ci_by_block(seed=17)`, tested) used for every arm and every statistic, so both arms of every contrast see the same resampled chromosomes.
Resampling is implemented through item weights (multiplicity of the chromosome), with weighted AUROC/Spearman/R2/mean kernels that equal the concatenated-resample statistic exactly (tested). Predictions of probes are NOT refitted in the bootstrap (out-of-fold predictions are fixed; chromosomes are resampled).
Delta = regulatory - comparator, computed replicate by replicate; 95% percentile CI of the Delta replicates; two-sided p = min(1, 2 min((1+#(D*<=0))/(1+B), (1+#(D*>=0))/(1+B))). Holm (alpha 0.05) over the FOUR primary endpoints only, applied per contrast; raw p also reported; Holm is not a selection rule; no aggregate "biological score".
A claim "regulatory supports" requires Holm-adjusted p < 0.05 AND Delta > 0. Non-primary statistics get raw p only (descriptive, no claim, `holm_p_adjusted = null`). Single-arm statistics get the same bootstrap CIs and no null-hypothesis test.
**Sign convention (all registered statistics are higher-is-better): Delta > 0 = regulatory better than the comparator.** Per endpoint: Loyfer Spearman (higher = embedding cosine tracks WGBS-profile similarity better); Micro-C AUROC (higher = contacts have higher cosine than matched non-contacts); delta-cosine (positive minus matched negative, higher = larger separation); FANTOM5 AUROC; RT Spearman/R2; kNN enrichment (higher = markers cluster).

## 2. Endpoint x arm design

Arms (4; identical design; catalog `configs/experiments/regulatory_confirmation_matrix/matrix.yaml`; store sha256 below; every arm in the `native_frozen` track, no re-optimisation):

| CLI arm id | matrix id | store | dim | dtype | sha256 (matrix.yaml) |
|---|---|---|---|---|---|
| `regulatory_histone_dnase_v1` | `regulatory_histone_dnase` | `data/cache/representations/regulatory_histone_dnase__global_svd256__discovery_chr1_19.h5` | 256 | float32 | `fa3a816338ce2c2ce5669b4be6f5334a44470e2b149395901168dbe673a9fe1e` |
| `functional_annotations_pca` | same | `outputs/encode_atlas_v1/embeddings/full_a18b869b.h5` | 256 | float16 | `3d99fe5ba4fb88991100f848e1d2c48ad19037c72b8821421b2eb677996e187f` |
| `cpgpt_large_locus` | same | `data/cache/representations/cpgpt_locus_large.h5` | 512 | float16 | `c08138cd83e08f5015521c2a5d42b88e1834f0670c84906590d9730223d7b71e` |
| `deepcpg_dna_locus` | same | `data/cache/representations/deepcpg_dna_locus.h5` | 128 | float16 | `e7280d5ff14d89a247e4cb3560a952250a3f1fa78daf653b9ec02761389b82b7` |

Loader: `/cpg_idx` is NOT sorted in any store -> argsort + searchsorted; float16 -> float32; finite check; a missing locus raises (no intersection); the regulatory store's candidate manifest is `configs/frozen/regulatory_histone_dnase_v1.json`.

Endpoints (every arm runs all of them; class from the ratified protocol; all statistics have bootstrap CIs, the sole exceptions being listed):

| Endpoint id | Class | Statistics (primary first) | Items |
|---|---|---|---|
| `loyfer_profile` | PRIMARY | `spearman_pooled_gt1Mb_inter_eqw`; descriptive per-stratum Spearman (6), `spearman_all_pairs`, `spearman_mean_of_gt1Mb_and_inter`; sensitivities (profile Spearman, Pearson corr, >= 30 shared groups) | 386,268 frozen pairs minus zero-locus pairs |
| `microc_H1_intra10kb` | PRIMARY | `auroc_contact_vs_noncontact` (score = cosine, all rows pooled, 2,689,921 + 2,689,921); secondary `delta_cosine_paired_mean`; secondary AUROC per distance bin | frozen pairs, fold >= 0 |
| `fantom5_membership` | PRIMARY | `auroc_oof_bg_enh5k` (11,575 + 11,575); sensitivities `auroc_oof__win500` (26,880 pairs) and `auroc_oof__tss5k_pool` (3,349 pairs) | frozen membership files, all rows |
| `rt_consensus` | PRIMARY | `spearman_oof_consensus`; `r2_oof_consensus`; sensitivity per-line Spearman/R2 (15 lines) | 408,345 CpGs |
| `loyfer_marker_knn` | SECONDARY | enrichment (k = 10), 3 marker sets | candidates = universe minus zero loci |
| `compartment_E1_probe_H1_100kb`, `..._GM12878_100kb` | SECONDARY | Spearman(E1, out-of-fold ridge prediction), R2 | CpGs with E1 and fold >= 0 |
| `fantom5_activity_similarity` | SECONDARY | Spearman(profile_sim, cosine) over 171,017 pairs minus zero-locus pairs; descriptive intra/inter | frozen activity pairs |
| `microc_HFFc6_intra10kb` | SENSITIVITY | AUROC, delta-cosine | 3,724,445 pos = ctrl, fold >= 0 |
| `microc_H1_inter1Mb` | EXPLORATORY | AUROC, delta-cosine on rows with fold >= 0 (9,931 + 9,931 pairs before zero exclusion; the 104,404 rows with fold = -1 are removed) | |
| `pmd_probe` | EXPLORATORY | AUROC of out-of-fold ridge scores for PMD membership | 408,399 CpGs |

- **[registered at launch]** Micro-C primary is the pooled AUROC over all frozen rows (as in the protocol), not a per-match AUROC; the paired effect size is the delta-cosine. HFFc6 is a sensitivity (never a Holm member); H1 inter is exploratory.
- **[registered at launch] [FLAG]** Micro-C distance strata only because the protocol names `stratified_by_distance`: its default bins (0, 1e3, 1e4, 1e5, 1e6, 1e7, inf) are used unchanged (intra pairs span 20 kb-2 Mb, so effectively [1e4,1e5), [1e5,1e6), [1e6,1e7)); descriptive. These bins differ from the Loyfer strata labels (which come from the frozen `stratum` column); `stratified_by_distance` is not used for Loyfer.
- **[registered at launch]** GM12878 compartments: the same E1 ridge probe as for H1 (the protocol says only "compartimenti A/B GM12878"); HFFc6 compartments are not evaluated. Frozen `E1` is the target; rows with NaN E1 are removed (identically for all arms).
- **[registered at launch]** FANTOM5 primary pool is `bg_enh5k` (frozen file `frozen_enhancer_membership_seed17.parquet`); the TSS-filtered pool and the 500 bp window are sensitivities. FANTOM5 expression-threshold (N = 1/3/10/20), chromatin-covariate matching and sample-level-vs-collapsed activity variants are **[FLAG] NOT run** (no frozen pair/membership file exists for them; they would need new matching, i.e. a freeze step).
- **[FLAG] NOT run in this launch:** RT residualisation for GC / CpG density / TSS distance and the normal-vs-tumour line split (the covariate file was "to be produced separately" at freeze and the normal/tumour line assignment is not frozen); the 15 per-line RT results are reported so a later, separately registered analysis can group them.
- Stratified secondary results (distance bins, per-fold values, per-line) carry CIs unless marked otherwise; per-fold probe metrics are point estimates in `meta`.

## 3. Output layout

`outputs/biological_validation_v2/bioval-v2-protocol-freeze-v1/<arm_id>/` containing `run_manifest.json`, `results.json`, `results.csv` (flat: arm, endpoint, class, stat, value, ci_lo, ci_hi, n), `endpoints/<endpoint>.json` and `endpoints/<endpoint>.replicates.npz` (1000 bootstrap replicates per statistic, from the shared draw table).
Contrasts under `.../contrasts/<reference>__vs__<comparator>.{json,csv}` (one row per statistic: reference/comparator values, delta, CI, raw p, `in_holm_family`, `holm_p_adjusted`, `supports_regulatory`, sign convention). The output root must lie outside `data/derived/bioval_v2`, `configs`, `src`, `docs/bioval_v2_prep` (enforced by the writer and the gate).
Run manifest: HEAD commit, protocol tag + commit (peeled), registration doc path/sha256/last commit, sha256 of `MANIFEST_checksums.json`, representation identity (path, registered and recomputed sha256, dim, dtype on disk and loaded, rows, id/embedding keys, catalog, frozen manifest, number of all-zero rows), UTC timestamp, seed 17, B, sha256 of the bootstrap draw table, thread count, library versions, exclusion counts (per store zero rows, union, per endpoint), requested endpoints, gate check results, `freeze_modules_imported` (computed from `sys.modules`) and `no_freeze_command_used`, `tcga_test_set_referenced = false`, the command line.

## 4. Threads, resources

CPU only, no GPU, `nice -n 10`, `--threads` <= 8 in total across concurrent processes (BLAS/OpenMP variables are set from `--threads` before numpy is imported); one process per arm.

## 5. Freeze commands that must remain unused (never run, never imported)

`scripts/bioval_v2/freeze_4dn_pairs.py`, `freeze_fantom5_matching.py`, `loyfer_freeze_pairs.py`, `loyfer_profile_similarity.py`, `build_checksum_manifest.py` (would overwrite `MANIFEST_checksums.json`), all `prepare_*.py` (`prepare_4dn`, `prepare_fantom5`, `prepare_loyfer`, `prepare_pmd_decato2020`, `prepare_replication_timing`),
`loyfer_{aggregate,extract,cpg_index,manifest,markers,markers_reconcile,provenance}.py`, `validate_4dn_assembly.py`, `fourdn_rangefile.py`, `gc_track_hg38.py`, `scripts/build_bioval_v2_covariates.py`, `scripts/fetch_cpgislandext_hg38.py`, `scripts/verify_frozen_candidate.py --write`, and the generators `matching.match_controls/match_pairs/build_covariates`, `pairs.sample_pairs`.
The run manifest records `freeze_modules_imported` (must be empty). The launch code only reads the frozen parquet/h5/npy/json files.

## 6. Bugfix-versus-methodology policy

Anything decided in this document is methodology and cannot change after the first real run. A later correction is allowed only if it is a **bugfix**: the code failed to implement the definition registered here or in the frozen protocol (crash, wrong index alignment, wrong sign, mis-counted exclusion, non-determinism, a kernel disagreeing with its frozen reference).
A bugfix must (i) be demonstrated by a failing synthetic test added first, (ii) be listed in an appended, dated "Bugfix log" at the end of this document (commit hash before/after, symptom, which outputs were affected), (iii) re-run all affected endpoints for ALL arms with the same draw table, and (iv) keep the superseded outputs (renamed, never overwritten silently).
Any change of an endpoint definition, statistic, grid, fold structure, exclusion rule, contrast list, bootstrap setting, class (primary/secondary/...) or Holm family is a methodology change: it is NOT allowed as a correction; it can only be run as a new, separately registered, explicitly exploratory analysis alongside the original. No post-hoc replacement of a primary by a better-looking variant; all registered endpoints are reported, including null ones. Frozen modules (`metrics.py` etc.) are never edited for this launch; defects found in them are reported (section 8) and handled by new code with an explicit note.

## 7. Pre-run verification checklist (all enforced by `run_evaluation.py` unless noted)

1. `git rev-parse bioval-v2-protocol-freeze-v1^{commit}` = `27d8bce1b2ce391cb4ffb0d89eb15660710c2b33` (annotated tag object `fa328fcde47c98863e1c745593e2d0ba5cb61cfe`).
2. This document is committed and `git status` is clean for it (gate: `registration_doc_committed_clean`); the launch code files are committed and clean (`launch_code_committed_clean`); record `git rev-parse HEAD`.
3. No pre-existing frozen path (`docs/bioval_v2_prep`, `configs/biological_validation_v2`, tracked manifests under `data/derived/bioval_v2`, the existing bioval v2 modules, `scripts/bioval_v2`, covariate/fetch scripts) differs from the freeze commit; only added files are tolerated (`frozen_paths_unchanged`). `docs/BIOLOGICAL_VALIDATION_V2.md` is doc-only different (commit `6338e7b`, section 8).
4. `MANIFEST_checksums.json` verified: 121/121 files sha256 + size, 0 missing, 0 extra (`scripts/bioval_v2/preexec_audit.py verify`; the gate calls the same function); protocol doc section 7 hashes 31/31 (audit `verify`, run manually before launch).
5. Store sha256 of the 4 arms equal the matrix.yaml values of section 2 (`preexec_audit.py stores`; gate `store_sha256`; re-checked by the loader at load time).
6. Output root outside protected paths; no TCGA / `data/protocols` path among inputs (`no_tcga_or_protocol_input`; every read goes through `assert_readable`).
7. `python scripts/bioval_v2/run_evaluation.py --all --dry-run` shows: zero rows 849 / 0 / 0 / 0, exclusion counts Loyfer 1,593, H1 intra 15,474, HFFc6 intra 22,205, FANTOM5 activity 176, zero coverage gaps.
8. `pytest tests/test_bioval_v2_launch.py` and `ruff check` clean. Launch with `run_evaluation.py --arm <id>` x4 (<= 8 threads in total), then `--contrasts --all`.

## 8. Frozen-module findings (reported, nothing silently worked around)

1. **`metrics.knn_enrichment(n_perm > 0)` is defective.** `is_marker` is computed once from the original labels and reused on the permuted labels: with a per-label `background` dict it raises `KeyError('')` as soon as a permuted label at an original marker position is empty (demonstrated by `test_frozen_knn_enrichment_permutation_bug_is_documented_and_replacement_is_sound`), and with a scalar background the null is evaluated at the wrong loci. The launch uses the frozen function ONLY for the observed enrichment (`n_perm=0`, cross-checked numerically) and a new routine `launch_endpoints.knn_permutation_pvalue` (markers re-identified after every permutation, within-chromosome, one-sided, seed 17, 999 permutations) for the p-value. The kNN p-value is secondary/descriptive; if you prefer, drop it. The frozen file is not edited.
2. `metrics.stratified_by_distance` default bins (1e7 upper decade) do not coincide with the six protocol strata of Loyfer; the frozen `stratum` column is used for Loyfer, the function (default bins) only for Micro-C distance bins (section 2). Not a bug.
3. `metrics.bootstrap_ci_by_block` is single-arm, re-seeds per call and is not paired/not efficient for rank statistics; it is therefore not used for the CIs, but its block-draw stream is reproduced bit-identically by the shared draw table (tested). Not a bug.
4. `PRIMARY_METRICS` (descriptive dict) is less specific than the protocol text (e.g. it says "AUROC FANTOM5 ... vs matched controls" without the pool and pooling rule); it is not used by the launch code.
