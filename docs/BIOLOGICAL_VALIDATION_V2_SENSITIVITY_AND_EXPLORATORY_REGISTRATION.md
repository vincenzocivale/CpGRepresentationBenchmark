# biological_validation_v2: registration of the frozen sensitivities (Part A) and of the EXPLORATORY POST-HOC Loyfer analyses (Part B)

Status: written BEFORE any follow-up statistic was computed on any real embedding. To be committed before the first real follow-up run;
the follow-up gate (`src/cpg_repr_benchmark/bioval_v2_followup/gate.py`) refuses to run unless this file and the follow-up code are committed and unmodified.
The frozen protocol (`docs/BIOLOGICAL_VALIDATION_V2.md`, tag `bioval-v2-protocol-freeze-v1` = `27d8bce1b2ce391cb4ffb0d89eb15660710c2b33`), the launch registration
(`docs/BIOLOGICAL_VALIDATION_V2_LAUNCH_REGISTRATION.md`) and the primary results (`docs/BIOLOGICAL_VALIDATION_V2_RESULTS.md`, HEAD 3e0c115) are NOT amended.
No new primary metric is introduced and no Holm correction is recomputed: everything below is descriptive (raw bootstrap p only).
Code: `src/cpg_repr_benchmark/bioval_v2_followup/{registry,gate,targets,strata,geometry,profile_ridge,compare,interpret,drivers}.py`,
`scripts/bioval_v2/run_followup.py` (subcommands `sensitivity`, `exploratory-loyfer`), `tests/test_bioval_v2_followup.py`.
Provenance audit (B4, documentation only, not gate-checked): `docs/BIOLOGICAL_VALIDATION_V2_EXPLORATORY_POSTHOC_PROVENANCE_AUDIT.md`.

## 0. Conventions common to everything below

Same 4 arms/stores (sha256 of the launch registration), same frozen manifests/pairs/matching, same D1-D4 conventions (launch registration section 1):
cosine endpoints use the D2 zero-row exclusion (union of all-zero loci over the 4 stores = 849 regulatory loci), so every retained pair set is identical across arms
(for follow-up targets that are themselves re-derived, the retained set additionally depends on the target only, never on the arm);
linear probes keep all rows; folds = frozen chromosome-blocked folds (seed 17; `chromosome_blocked_folds(22 chroms, 5, 17)`, asserted equal to the frozen map);
probes = `bioval_v2_launch.chrom_probes` unchanged (scaler on outer-train, ridge alpha grid {1e-2..1e4}, inner leave-one-fold-out CV, per-target alpha);
paired chromosome-block bootstrap = ONE shared draw table (B = 1000, seed 17, 22 blocks), delta = regulatory - comparator, 95% percentile CI, raw two-sided p.
Contrasts are made only against the candidate `regulatory_histone_dnase_v1` vs the three comparators (no contrast among comparators, with one registered exception in B3, see 7).
**No Holm, no claim, no q-value over any follow-up row. Every follow-up table is labelled `descriptive` (Part A) or `EXPLORATORY POST HOC` (Part B).**

## 1. PART A: complete protocol-derived list of sensitivity / secondary analyses and status

Sources re-derived: `docs/BIOLOGICAL_VALIDATION_V2.md` section 0 table + sections 2.1-2.4 "Sensitivity" bullets, `configs/biological_validation_v2/anti_circularity.yaml` (`sensitivities:`), the PREP documents, compared with `outputs/.../<arm>/endpoints/*.json` (all 11 endpoint files, 4 arms).
Classification: DONE (already in the frozen results), RUNNABLE_FROM_FROZEN, REQUIRES_NEW_ARTIFACT, PROTOCOL_AMBIGUOUS.

| # | Axis | Protocol analysis (verbatim) | Status | Note |
|---|---|---|---|---|
| 1 | Loyfer | Pearson vs Spearman | DONE | `spearman_pooled_eqw__profile_spearman`, `pearsoncorr_pooled_eqw__profile_pearson` |
| 2 | Loyfer | per strato di distanza | DONE | 6 per-stratum Spearman |
| 3 | Loyfer | coppie con < 30 gruppi condivisi rimosse | DONE | `spearman_pooled_eqw__min30_shared_groups` |
| 4 | Loyfer | `mask_lenient` | **RUNNABLE_FROM_FROZEN** | item `loyfer_mask_lenient` (section 2.1) |
| 5 | Loyfer | gruppi a campione singolo esclusi | **RUNNABLE_FROM_FROZEN** | item `loyfer_drop_single_sample_groups` (2.1) |
| 6 | Loyfer marker kNN | set marker paper S4A/S4B lifted; solo U25 | DONE | `enrichment_paper_lifted_U25_U250`, `enrichment_github_U25_only` (the yaml lists `U25_only` under `loyfer_marker_knn`, not under the pair endpoint; the earlier "U25-only pairs" reading is therefore moot) |
| 7 | 3D | HFFc6 Micro-C intra | DONE | `microc_HFFc6_intra10kb` |
| 8 | 3D | per bin di distanza | DONE | H1 `auroc_distance_bin_*` (3 bins) |
| 9 | 3D | compartimenti a 50 kb (E1 50 kb "secondario") | **RUNNABLE_FROM_FROZEN** | items `compartment_E1_probe_{H1,GM12878}_50kb` (2.2) |
| 10 | 3D | K per coppia di bin | **REQUIRES_NEW_ARTIFACT** | needs `freeze_4dn_pairs` re-projection with K != 20; the protocol names no alternative K |
| 11 | 3D | confronto con baseline di sola distanza/covariate (dichiarato) | **PROTOCOL_AMBIGUOUS** | score, model and covariate set undefined (pairs are distance-matched, so a distance-only AUROC is ~0.5 by construction) |
| 12 | 3D | E1 probe 100 kb H1 / GM12878; delta-cosine | DONE | secondary endpoints |
| 13 | FANTOM5 | finestra 500 bp; pool TSS-filtrato | DONE | `auroc_oof__win500`, `auroc_oof__tss5k_pool` |
| 14 | FANTOM5 | soglie di espressione N = 1/3/10/20 librerie | **RUNNABLE_FROM_FROZEN (subset reading, FLAG)** | items `fantom5_membership_N{1,3,10,20}` (2.3) |
| 15 | FANTOM5 | matching aggiuntivo su covariate cromatiniche (a posteriori) | **REQUIRES_NEW_ARTIFACT** | new matching (`freeze_fantom5_matching`) with chromatin covariates; the covariates are not specified |
| 16 | FANTOM5 | livello campione vs collassato | **RUNNABLE_FROM_FROZEN** | item `fantom5_activity_sample_level` (2.3) |
| 17 | FANTOM5 | per categoria (tessuti / cellule primarie / linee) | **RUNNABLE_FROM_FROZEN (FLAG: definition from the frozen PREP helper)** | items `fantom5_activity_cat_{cell_lines,primary_cells,tissues}` |
| 18 | FANTOM5 | activity similarity per intra/inter | DONE | descriptive |
| 19 | RT | per singola linea (15) | DONE | `*_line_*` |
| 20 | RT | correlazione parziale/residualizzazione GC / densita' CpG-isola / densita' genica-TSS | **PROTOCOL_AMBIGUOUS** | covariate table frozen and present (`covariates/universe_covariates.parquet`), but residualisation model, which variable (target / prediction / both), fit scope (all vs train chromosomes) and "gene density" (only `tss_dist` exists) are undefined |
| 21 | RT | solo linee normali vs tumorali | **PROTOCOL_AMBIGUOUS** | the normal/tumour assignment of the 15 UW lines is not frozen (hESC BG02ES, LCLs, K562, SK-N-SH, HeLa-S3 ...); also needs refits of the consensus on subsets |
| 22 | RT/PMD | PMD probe (exploratory) | DONE | `pmd_probe` |

Decisions needed from the user (none of them is taken here): **D-A1** accept in-memory re-derivation of the Loyfer profile target (items 4-5) as "read-only from frozen files" (it re-implements, never runs, the frozen target code; no artifact is written to `data/derived`; the kernel must reproduce the frozen `loyfer_pearson`/`n_shared_groups` of all 386,268 pairs bit-for-bit, asserted at run time: dry-run max |diff| = 8.9e-16);
**D-A2** accept the N-threshold subset reading (14); **D-A3** K values + permission to run `freeze_4dn_pairs` (10); **D-A4** chromatin covariates + permission for a new matching (15); **D-A5** definition of the distance/covariate baseline (11);
**D-A6** residualisation specification (20) (a candidate: OLS of the consensus target and of the out-of-fold prediction on [gc_content, cpg_density, island-context dummies, tss_dist] fitted on the training chromosomes of each fold, then Spearman of the residuals; not registered here); **D-A7** frozen normal/tumour assignment (21).

### 2. Runnable items: exact definitions

Pair/row sets, probes, bootstrap: section 0. Per item per arm: `endpoints/<item>.json` (+ `.replicates.npz`), stat class `sensitivity`. Reference statistic for the verdict = the corresponding frozen primary statistic.

**2.1 Loyfer target variants** (profiles from the frozen `loyfer_celltype_beta.h5`, frozen pair list `frozen_pairs_seed17.parquet`; similarity = cosine; D2 exclusion).
Profile Pearson re-implemented exactly as the frozen definition: pairwise-complete groups, `n_shared >= min_shared = 20`, undefined if a profile has centred sum of squares <= 1e-12 (pair then removed, count reported, identical across arms).
 * `loyfer_mask_lenient`: `mask_lenient` (>= 1 valid donor) instead of `mask_primary` (>= 2 donors, 1 for single-sample groups). Pair set = frozen pairs (dry-run: 386,268 defined, 384,675 after D2).
 * `loyfer_drop_single_sample_groups`: the 4 groups with `n_samples_per_group == 1` (Bone-Osteob, Dermal-Fibro, Epid-Kerat, Gallbladder) removed from the profile columns, `mask_primary` otherwise, `min_shared = 20` unchanged (literal reading of "gruppi a campione singolo esclusi"). Dry-run: 385,626 defined (642 pairs fall below 20 shared groups or are constant), 384,038 after D2.
 * Statistics: `spearman_pooled_gt1Mb_inter_eqw` (weights 1/n_stratum after exclusions, launch D1), six per-stratum Spearman, `spearman_all_pairs`; each cross-checked against `metrics.spearman_profile_vs_embedding_sim`.
**2.2 E1 probe at 50 kb**: frozen `H1_MicroC_cpg_compartment_50000.parquet` and `GM12878_HiC_cpg_compartment_50000.parquet` (in `MANIFEST_checksums.json`); rows with E1 (H1 407,826; GM12878 408,073); fold = frozen chrom->fold map; probe/statistics identical to `compartment_E1_probe_*_100kb` (ridge D3; Spearman(E1, OOF), R2).
**2.3 FANTOM5**
 * `fantom5_membership_N{N}`, N in {1,3,10,20}: from the frozen `bg_enh5k` membership file keep the positives whose enhancer is expressed (TPM >= 1) in >= N libraries (`enhancers.parquet:n_samples_expressed`) together with THEIR FROZEN matched controls (`matched_to`); NO re-matching (1:1 exact-stratum pairing preserved by construction; numeric balance not re-verified: **[FLAG]**); the logistic probe (D3) is retrained on the subset with the frozen folds. Pairs: N=1 11,529; 3 11,163; 10 9,089; 20 7,041 (of 11,575). Statistic `auroc_oof__N{N}`. Reading caveat: the PREP defines N as the "informative profile" threshold of the activity secondary; the yaml/protocol list it under the membership primary; the membership reading is used (D-A2).
 * `fantom5_activity_sample_level`: frozen 171,017 activity pairs (`enh_i`, `enh_j`); Pearson over all 1,829 libraries of `activity_sample_log2tpm.npy` instead of the 638 collapsed groups. Precondition (asserted): the collapsed Pearson recomputed from `activity_collapsed_log2tpm.npy` equals the frozen `profile_sim` (|diff| <= 1e-6).
 * `fantom5_activity_cat_*`: Pearson over the collapsed groups of one category (cell lines 249, primary cells 230, tissues 159), minimum 3 groups, undefined if constant (pair removed, identical across arms; dry-run read-only counts of defined pairs: cell lines 170,904, primary cells 171,017, tissues 170,917; sample level 171,017; collapsed profile similarity reproduced from the frozen matrix with max |diff| = 0.0). These semantics are those of `enhancer_profile_similarity(categories=...)` in the frozen PREP code (not of the protocol text): **[FLAG]**.
 * Statistics: `spearman_all_pairs` (cosine vs profile similarity), descriptive intra/inter; D2 exclusion (176 pairs).

### 2.4 Operational verdict "confirms / qualitatively changes the primary" (fixed now)
Per item, with reference = frozen primary statistic of the same family (Loyfer pooled primary; E1 100 kb Spearman; `auroc_oof_bg_enh5k`; activity `spearman_all_pairs`) and its frozen candidate-vs-comparator deltas (3 contrasts):
sign of each sensitivity delta (regulatory - comparator) vs the primary delta: `same_sign` / `sign_flip_CI_includes_0` / `sign_flip_CI_excludes_0` (sensitivity CI).
**CONFIRMS** = all three contrasts `same_sign`; **QUALITATIVELY_CHANGES** = at least one `sign_flip_CI_excludes_0`; otherwise **INCONCLUSIVE_SIGN_FLIP_NOT_RESOLVED**. Also reported: whether the ordering of the 4 arms by value is identical to the primary ordering. Outputs: `sensitivity/SUMMARY.{json,csv}` (all 22 rows above incl. the not-runnable ones with their blocker).

## 3. Part A output layout
`outputs/biological_validation_v2/bioval-v2-protocol-freeze-v1/sensitivity/<arm>/{run_manifest.json, endpoints/<item>.json, endpoints/<item>.replicates.npz}`, `.../sensitivity/contrasts/<reference>__vs__<comparator>.{json,csv}`, `.../sensitivity/SUMMARY.{json,csv}`.

# PART B: EXPLORATORY POST-HOC (`exploratory_posthoc_loyfer`)

**EXPLORATORY POST-HOC.** Conceived AFTER seeing the primary results (candidate clearly worse than CpGPT-large and DeepCpG-DNA and slightly worse than functional PCA on the Loyfer primary, motivating the question). It cannot replace the frozen Loyfer primary, no alternative is designated as a new primary, nothing here enters the Holm family, no data are added, no representation is retrained or created, no feature engineering, no TCGA. Every output file carries `EXPLORATORY_POST_HOC: true` (manifest flag `exploratory_posthoc`).
All definitions below are fixed from FROZEN data and registered before any embedding statistic is computed. Output: `.../exploratory_posthoc_loyfer/`.

## 4. B1 strata of the 386,268 frozen Loyfer pairs (same D2 exclusion as the primary)
Per-CpG profile = frozen beta over the groups observed under `mask_primary`; per-CpG variance V_c (ddof = 0) and mean M_c over those groups. Pair statistics: **V = min(V_i, V_j)** ("variable in BOTH loci"), **M = (M_i + M_j)/2**. Cells (bin edge value belongs to the upper bin; edges = quantiles of the 386,268 frozen pairs, 6 significant digits, asserted at run time to agree within 1e-4 relative, so they cannot be tuned):
 * `V_Q1..V_Q4`: V < 0.000369276 | [0.000369276, 0.00133439) | [0.00133439, 0.00841199) | >= 0.00841199; plus the overlapping cell `V_top10pct`: V >= 0.0266948.
 * `M_Q1..M_Q4`: M < 0.2654 | [0.2654, 0.466704) | [0.466704, 0.698094) | >= 0.698094.
 * distance: the six frozen strata `dist_*`; chromosome: `chrom_intra` (stratum != interchromosomal) / `chrom_inter`.
Per cell and geometry: unweighted Spearman (profile Pearson [frozen `loyfer_pearson`] vs similarity) with bootstrap CI; for `all` and every V/M cell also the primary-style statistic (weighted Spearman pooled over `>1Mb` + `interchromosomal`, weight 1/n_stratum within the cell; cell skipped if a stratum is empty). Pair counts after D2 (dry-run): all 384,675; V_Q1..Q4 96,338 / 96,244 / 95,920 / 96,173; V_top10pct 38,555; M_Q1..Q4 96,535 / 96,263 / 96,016 / 95,861; intra 191,234 / inter 193,441.
"Does the arm ranking change in truly variable loci": ranking = ordering of the 4 arms by the pooled statistic in V_Q4 and V_top10pct, compared with the ordering of the frozen primary pooled statistic (`similarity_ranking_by_cell.csv`: `same_ordering_as_frozen_primary`, candidate rank, candidate-vs-comparator deltas with CI). Descriptive.
Run-time reproduction check: the pooled `cosine` / `all` statistic must reproduce the frozen primary value of the arm to 1e-9 (otherwise the run fails).

## 5. B2 geometry diagnostics (the same pair set as B1; four similarities)
 * `cosine`: frozen reference.
 * `centered_cosine`: cosine(e - mu), **mu = mean embedding over all 408,399 universe loci excluding the 849 D2 union-zero loci (407,550 loci), identical for every arm**.
 * `pc1_removed_cosine`: z = e - mu; z' = z - (z.v1) v1, v1 = top eigenvector (exact `eigh`) of the covariance of z over the same 407,550 loci; cosine(z'_i, z'_j).
 * `neg_euclidean`: s = -||e_i - e_j||_2 on the raw embeddings. Any monotone standardisation (z-scoring of -d, exp(-d^2/median d^2)) leaves Spearman unchanged, so the single registered definition is -d (tested).
A zero-norm vector after transformation raises (explicit failure; no silent drop). mu/PC1 hashes and the PC1 explained-variance ratio are stored. Statistics = those of B1 for every geometry; within-arm deltas (geometry - cosine) with paired bootstrap are written (`geometry_within_arm_deltas.csv`). No alternative is a new primary.

## 6. B3 chromosome-blocked multi-output ridge: profile prediction from the CpG representation
 * **Targets/rows**: the frozen WGBS beta profile (`loyfer_celltype_beta.h5:beta`, float16 -> float64), CpGs observed under `mask_primary` in ALL 39 groups (the primary mask): **382,041 of 408,399** CpGs (per fold 78,611 / 91,624 / 61,816 / 63,322 / 86,668); no imputation; identical rows for all arms; embeddings of zero rows are kept as features (D2 probe convention).
 * Three registered target sets, fitted in ONE multi-output ridge (79 outputs, per-output alpha as in D3): **T1 raw** (39 betas), **T2 centred** (beta minus the per-CpG mean over the 39 groups: the "profile shape" that Pearson similarity measures), **T3 level** (per-CpG mean, 1 output).
 * **Folds**: the frozen chromosome->fold map (seed 17), asserted identical to `chromosome_blocked_folds(22,5,17)`; one map for all arms. **Model**: `chrom_probes.ridge_blocked_cv` unchanged (StandardScaler on outer-train only, alpha in {1e-2,...,1e4} sklearn convention, inner leave-one-frozen-fold-out CV, selection by pooled inner SSE, ties -> larger alpha); no representation-specific hyperparameter.
 * **Subsets**: `all_cpgs` and `variable_cpgs` (per-CpG variance >= 0.00841199, the V_Q4 lower edge; 192,339 CpGs).
 * **Metrics** (out-of-fold; exact functions of per-chromosome sufficient statistics, so the shared bootstrap draw table applies): per-group R2_g = 1 - SSE_g/SST_g (SST around the group mean) for the 39 groups; `r2_global` = 1 - sum SSE_g / sum SST_g (pooled over CpGs x groups, SST around per-group means); `r2_macro` (mean of the 39); `pearson_pooled` (over all CpG x group cells); `pearson_mean_over_cpgs` (mean over CpGs of the per-CpG Pearson across the 39 groups); per fold: global R2/Pearson and the fold x group R2 matrix. **Reference**: the train-fold per-group-mean predictor ("R2 = 0 baseline", stored per target/subset); R2 of the baseline is ~0 by construction.
 * **Contrasts**: candidate vs each comparator for every metric (global and each of the 39 groups), paired chromosome-block bootstrap, raw p, no Holm. One additional registered contrast set: functional vs each of the two sequence arms on `r2_global` (for rule C below).
 * Not done: nearest-neighbour ceiling; any retraining.

## 7. The three interpretations and the operational evidence rules (fixed now; implemented in `interpret.py`; cannot be re-tuned)
A: the regulatory representation lacks long-range methylation-program information. B: the information is present but the cosine geometry does not expose it. C: the result is materially affected by methylation pretraining/provenance of the comparators.
Thresholds: `R2_NEAR_BASELINE = 0.02`; `GEO_IMPROVEMENT = 0.03` (Spearman units). "Significantly below/above" = the 95% paired-bootstrap CI of delta excludes 0. SEQ = {CpGPT-large, DeepCpG-DNA}.
**Primary reading: target T2 (centred), subset `variable_cpgs`, metric `r2_global`**; consistency readings T2/`all_cpgs` and T1/`all_cpgs` (reported; if they disagree with the primary reading the final text must say "subset/target dependent").
 * **RO (ridge outcome)**, from the candidate-minus-SEQ deltas: UNINFORMATIVE (all four arms have R2 <= 0.02) | ABSENT (candidate R2 <= 0.02 and significantly below both SEQ) | LOWER (candidate R2 > 0.02 and significantly below both SEQ) | MIXED (significantly below exactly one SEQ) | COMPARABLE (not significantly below either) | HIGHER (COMPARABLE and significantly above at least one).
 * **GO (geometry outcome)**, on the candidate's pooled primary-style statistic (`all` cell): IMPROVES = some alternative geometry g has within-arm delta (g - cosine) >= 0.03 with CI > 0; CLOSES = IMPROVES for some g and, under that same g for every arm, the candidate is not significantly below either SEQ arm; IMPROVES_ONLY; NONE.
 * **FO**: BELOW_BOTH if the functional arm is significantly below both SEQ arms on the ridge metric, else NOT_BELOW. **PO** (documentary, from the provenance audit): both sequence arms = DIRECT_METHYLATION_SUPERVISION (fixed by the audit; it does not change with the results).
 * **CO** (cosine outcome) is fixed by the frozen primary (candidate below both SEQ).
Mechanical combination table (A | B | C):
| RO | GO | A | B | C |
|---|---|---|---|---|
| UNINFORMATIVE | any | not adjudicable | not adjudicable | not adjudicable |
| ABSENT | any | supported (strong) (GO=CLOSES flagged as anomaly) | not supported | consistent but confounded with A if FO=BELOW_BOTH, else not supported |
| LOWER | CLOSES / IMPROVES_ONLY / NONE | supported (partial) | supported (partial) / weak / not supported | consistent but confounded with A if FO=BELOW_BOTH, else not supported |
| MIXED | CLOSES / IMPROVES_ONLY / NONE | partial / unresolved | partial / weak / not supported | unresolved |
| COMPARABLE or HIGHER | CLOSES | not supported | supported (strong) | not supported by ridge evidence |
| COMPARABLE or HIGHER | IMPROVES_ONLY | not supported | supported (partial) | not supported by ridge evidence |
| COMPARABLE or HIGHER | NONE | not supported | supported as "linearly decodable, not exposed by any registered similarity" | not supported by ridge evidence |
C is never asserted beyond "consistent": the present four arms cannot separate "ENCODE-feature arms lack the information" from "sequence arms were trained on methylation". The provenance audit (B4) never excludes a comparator or invalidates a result.
Outputs: `interpretation_ABC.json` (all three readings, inputs, thresholds) written by `--contrasts`.

## 8. Output layout (Part B)
`.../exploratory_posthoc_loyfer/<arm>/{run_manifest.json, endpoints/exploratory_similarity.{json,replicates.npz}, endpoints/B3_profile_ridge.{json,replicates.npz}}`; `.../contrasts/{similarity,ridge}__<reference>__vs__<comparator>.{json,csv}`; `similarity_ranking_by_cell.{csv,json}`; `geometry_within_arm_deltas.csv`; `ridge_summary_by_arm.csv`; `ridge_functional_vs_sequence.csv`; `interpretation_ABC.json`. Run manifest per arm: HEAD, tags, both registration docs (path/sha256/last commit), manifest-checksums sha256, store sha256, seed 17, B, draw-table sha256, threads, library versions, exclusion counts, gate checks, `freeze_modules_imported`, frozen-results fingerprint before/after, `exploratory_posthoc` flag, command line.

## 9. Bugfix-versus-methodology policy and protected outputs
Same policy as launch registration section 6: anything registered here is methodology; only a bugfix (code fails to implement the registered definition) may change code, demonstrated by a failing synthetic test first, listed in an appended dated bugfix log, all affected outputs re-run for ALL arms, superseded outputs kept. A change of definition (strata, geometry, ridge protocol, thresholds, rules) is a new separately registered analysis.
**Never overwritten** (guarded by `FollowupWriter`, checked by the gate fingerprint `38b9da12700a8fc83dc08f5a01aaa76b61f0d5097bffe395d8eb5225f1f02094` over the frozen result tree): `outputs/biological_validation_v2/bioval-v2-protocol-freeze-v1/{<arm>/endpoints, <arm>/results.json, <arm>/results.csv, <arm>/run_manifest.json, contrasts, report}`; also the protocol doc, `docs/bioval_v2_prep`, `configs/biological_validation_v2`, `data/derived/bioval_v2`, the launch registration, `src/.../biological_validation_v2`, `src/.../bioval_v2_launch`, `scripts/bioval_v2/run_evaluation.py`. Freeze/prepare/build commands (list in launch registration section 5) are never run or imported (recorded in the manifest).

## 10. Pre-run checklist (enforced by `run_followup.py`; real runs refuse unless all PASS)
1. tag `bioval-v2-protocol-freeze-v1` -> `27d8bce1b2ce391cb4ffb0d89eb15660710c2b33`. 2. this document and all follow-up code/tests committed and clean (`followup_registration_committed_clean`, `followup_code_committed_clean`); launch registration and launch code clean; record `git rev-parse HEAD`.
3. `frozen_paths_unchanged` (only added files vs the freeze commit). 4. `MANIFEST_checksums.json` verifies (121/121). 5. store sha256 of the 4 arms equal the matrix values. 6. out root equals the registered root of the subcommand, outside protected and frozen-result paths; no TCGA/protocol input. 7. frozen primary results fingerprint unchanged.
8. `run_followup.py {sensitivity|exploratory-loyfer} --all --dry-run` shows zero rows 849/0/0/0, Loyfer 1,593 excluded, target reproduction diff <= 1e-9, registered edges check PASS. 9. `pytest tests/test_bioval_v2_followup.py` and `ruff check` clean.
10. Launch: one process per arm, <= 8 threads in total, `nice -n 10`; then `--contrasts --all`. Items that need a user decision (D-A1..D-A7) are not run; D-A1/D-A2 must be answered before items 4, 5, 14 are launched (otherwise run with `--item` excluding them).
