# Loyfer primary failure audit: REPORT (EXPLORATORY POST-HOC)

Generated MECHANICALLY by `scripts/bioval_v2/make_loyfer_audit_report.py` from the files under `outputs/biological_validation_v2/loyfer_failure_audit/` (sources and sha256 in `docs/LOYFER_PRIMARY_FAILURE_AUDIT_sources.json`; `--check` verifies them). Registration: `docs/LOYFER_PRIMARY_FAILURE_AUDIT_REGISTRATION.md` (Amendments 1-2, Bugfix log). Classification of section 11 = output of `decision.py` (`report_inputs/REPORT_INPUTS.json`), applied without ad hoc interpretation. Descriptive: raw chromosome-block bootstrap CIs (22 blocks of `cpg_i`, B = 1000, seed 17), no Holm, no claim. It does not replace or amend the frozen primary. 'n/a' = not available in the files.

## 0. Caveats that apply to everything below

1. **Level A only.** Verdict `PASS_LEVEL_A` (level A, raw_level_B_executed = False). It excludes bugs downstream of the per-sample counts (donor/group aggregation, masks, float16 storage, coordinates between artifacts, pairing, Pearson) but does NOT verify the per-sample counts nor the `.pat` parsing / CpG-index mapping against the raw files (Level B not executed, user decision D-L1). Any 'no bug' statement carries this limit.
2. **Attempt 1 of step 1 was FAIL** (`step1_target_audit_attempt1_FAIL/`, failed check: ['group_beta_f16_rounding_mismatch_frac_ok']): the a-priori threshold of 1e-5 on the float16 rounding-mismatch fraction was not met (observed 4.3e-05, 677 of 15756235 cells). `PASS_LEVEL_A` exists only thanks to **Amendment 2** (mechanism-based rule, see section 1). A verdict computed before Bugfix 1 is kept in `step1_target_audit_VERDICT_before_bugfix1.json` (verdict PASS_LEVEL_A; it was bound to the earlier registration hash).
3. **Bugfix 1** (`report-inputs` aggregation of step 5 from the per-arm files) changed no result.
4. Exploratory post hoc: conceived after the primary; thresholds fixed before any result; nothing here selects a new primary or a new representation; linear decodability is not alignment with a 'methylation program' and pretraining effects are not excluded (section 13).

## Summary in plain language

* **What the registered rules return**: classes = ['COMPRESSION_GEOMETRY'] (BUG False, TARGET_LIMITATION False, COMPRESSION_GEOMETRY True, INFORMATION_ABSENCE False, COMPARATOR_ADVANTAGE False).
* **The target is not the problem by the registered criteria**: no Level-A bug (with the Level-A limit), reliability-bounded ceiling for the primary 0.781 (threshold 0.30), reliability in >1Mb / interchromosomal 0.610 / 0.610 (threshold 0.50), the top 1% similarity band contains 0.000 near-constant/small-amplitude pairs.
* **Reliability depends strongly on methylation variance** (not part of the registered rules): Spearman-Brown rho / ceiling by variance quartile of the pair (min variance) Q1 0.036 / 0.189, Q2 0.336 / 0.580, Q3 0.786 / 0.887, Q4 0.939 / 0.969, top 10% 0.969 / 0.984; the lowest-variance quartile is close to noise.
* **Why the primary is -0.063 while the linear probe reads out the profile**: on the same frozen pairs, the cosine of the candidate's 256-d embedding is -0.063 (complete-case pairs), whereas the Pearson similarity of the chromosome-held-out ridge-predicted 39-group profiles of the SAME embedding is +0.277 (T1) / +0.277 (T2 centred); the registered rule COMPRESSION_GEOMETRY is satisfied through these OOF sources (delta vs native +0.340, CI [0.320, 0.359]). The linear probe of the candidate has centred-target R2 (variable CpGs) 0.279. The raw 2,492-track space does NOT recover the target either (raw cosine -0.085, Jaccard -0.076, both below the native 256-d -0.063), so the registered raw-space metrics show no recovery that the SVD would then have removed.
* **The arm ordering reverses under the linear read-out**: OOF-profile similarity (T1) functional PCA (legacy control) > Histone+DNase (candidate) > CpGPT-large > DeepCpG-DNA (Histone+DNase (candidate) +0.277, functional PCA (legacy control) +0.280, CpGPT-large +0.175, DeepCpG-DNA +0.156); ridge R2 (centred target, variable CpGs) Histone+DNase (candidate) > functional PCA (legacy control) > CpGPT-large > DeepCpG-DNA (Histone+DNase (candidate) 0.279, functional PCA (legacy control) 0.278, CpGPT-large 0.031, DeepCpG-DNA 0.009); the native-cosine primary orders them DeepCpG-DNA > CpGPT-large > functional PCA (legacy control) > Histone+DNase (candidate).
* **Geometry operations change the primary-style number a lot for the candidate** (centred cosine +0.174, negative Euclidean +0.202, PC1 removed +0.086); PC1 explains 0.464 of the embedding variance and is associated with the number of overlapped tracks (Spearman +0.915).
* **Is 0.1065 (CpGPT) strong?** It is of the same size as simple covariate baselines on the same pairs (-|dM| 0.1091, -|d CpG density| 0.108) and below the difference-in-peak-count baseline (0.161); null controls are within +-0.005.
* **Comparator advantage is not 'STRONG'**: both sequence comparators lead on the native cosine but the lead disappears under most other registered fair evaluations (CpGPT: PARTIAL, DeepCpG: PARTIAL); the pretraining provenance (section 10) is not excluded as a factor.

### Key numbers (primary-style statistic = weighted Spearman pooled over >1Mb + interchromosomal, equal stratum weight)

| Quantity | Value [95% CI] |
|---|---|
| Reliability ceiling of the primary (sqrt of Spearman-Brown rho), donors>=2 [q2.5, q97.5 over 20 splits] | 0.781 [0.777, 0.792] |
| same, groups with >=4 donors | 0.758 [0.749, 0.767] |
| Spearman-Brown reliability >1Mb / interchromosomal | 0.610 [0.604, 0.628] / 0.610 [0.602, 0.628] |
| Native cosine primary, Histone+DNase (candidate) | -0.0628 [-0.0724, -0.0534] |
| Native cosine primary, functional PCA (legacy control) | -0.0381 [-0.0459, -0.0301] |
| Native cosine primary, CpGPT-large | +0.1065 [+0.1010, +0.1121] |
| Native cosine primary, DeepCpG-DNA | +0.1155 [+0.1079, +0.1220] |
| NULL_HI (97.5% of 200 nulls), Histone+DNase (candidate) | 0.0053 |
| NULL_HI (97.5% of 200 nulls), functional PCA (legacy control) | 0.0052 |
| NULL_HI (97.5% of 200 nulls), CpGPT-large | 0.0050 |
| NULL_HI (97.5% of 200 nulls), DeepCpG-DNA | 0.0053 |
| Baseline neg_abs_delta_peak_count_total | +0.1607 [+0.1515, +0.1707] |
| Baseline neg_abs_delta_methylation_mean | +0.1091 [+0.1031, +0.1160] |
| Baseline neg_abs_delta_cpg_density | +0.1076 [+0.1005, +0.1155] |
| Baseline chromosome_only | -0.0020 [-0.0091, +0.0053] |
| Raw binary cosine (2,492 tracks) | -0.0854 [-0.0954, -0.0752] |
| Raw Jaccard | -0.0759 [-0.0850, -0.0664] |
| Native 256-d cosine (same set) | -0.0628 [-0.0724, -0.0534] |
| OOF-profile similarity T1 raw, candidate (complete-case pairs) | +0.2767 [+0.2651, +0.2889] |
| OOF-profile similarity T2 centred, candidate | +0.2769 [+0.2653, +0.2890] |
| Native cosine on the same complete-case pairs, candidate | -0.0629 [-0.0723, -0.0534] |
| Candidate geometry: centered_cosine | +0.1739 [+0.1646, +0.1834] |
| Candidate geometry: remove_pc1 | +0.0856 [+0.0758, +0.0961] |
| Candidate geometry: remove_pc1_2 | +0.0869 [+0.0765, +0.0971] |
| Candidate geometry: remove_pc1_5 | +0.0490 [+0.0434, +0.0552] |
| Candidate geometry: remove_pc1_10 | +0.0320 [+0.0273, +0.0365] |
| Candidate geometry: standardized_cosine | +0.0410 [+0.0365, +0.0454] |
| Candidate geometry: whitened_cosine | +0.0411 [+0.0364, +0.0454] |
| Candidate geometry: neg_euclidean | +0.2017 [+0.1923, +0.2120] |
| Ridge R2 centred, variable CpGs: original grid / extended grid, Histone+DNase (candidate) | 0.2793 / 0.2793 |
| Ridge R2 centred, variable CpGs: original grid / extended grid, functional PCA (legacy control) | 0.2783 / 0.2783 |
| Ridge R2 centred, variable CpGs: original grid / extended grid, CpGPT-large | 0.0314 / 0.0308 |
| Ridge R2 centred, variable CpGs: original grid / extended grid, DeepCpG-DNA | 0.0090 / 0.0090 |

## 1. Target audit (section 1; Level A)

Verdict `PASS_LEVEL_A`, 26 hard checks, failed: none; registration sha256 `0a6622323fc3d66d...`, code sha256 `1ebf823fe2fd53cb...`, HEAD `2eb34d9a7b`, amendment `1+2 (2026-10-05)`.

| Hard check | Result |
|---|---|
| mapping_matches_published_S1 | PASS |
| n_samples_used_eq_205 | PASS |
| n_donors_summed_over_groups_eq_168 | PASS |
| n_groups_eq_39 | PASS |
| groups_identical_to_frozen | PASS |
| n_samples_per_group_identical | PASS |
| n_donors_per_group_identical | PASS |
| counts_not_clipped | PASS |
| counts_row_order_identical_to_frozen_beta | PASS |
| coordinates_counts_h5_equal_universe_covariates | PASS |
| coordinates_beta_h5_equal_universe_covariates | PASS |
| mask_primary_identical | PASS |
| mask_lenient_identical | PASS |
| n_valid_donors_identical | PASS |
| group_beta_nan_pattern_identical | PASS |
| group_beta_within_half_ulp | PASS |
| group_beta_f16_mismatch_within_float32_precision_of_midpoint | PASS |
| info_f16_mismatch_fraction_vs_superseded_1e-5 | info |
| structural_pairs_cpg_i_lt_cpg_j | PASS |
| structural_pairs_endpoints_in_universe | PASS |
| structural_universe_cpg_unique | PASS |
| structural_universe_chrom_pos_unique | PASS |
| structural_beta_in_0_1 | PASS |
| structural_pairs_no_duplicates | PASS |
| n_shared_groups_identical | PASS |
| pearson_defined_pattern_identical | PASS |
| pearson_reproduces_frozen_float16_path | PASS |

* **Sample-to-group mapping** vs published Table S1 (re-joined by Z-suffix): 205 used samples, 205 S1 rows, group mismatches 0, patient mismatches 0, unmatched 0, S1-not-used 0.
* **Samples/donors per group**: 39 groups, 205 samples, per-group donors sum 168; groups with N = 1: Bone-Osteob, Dermal-Fibro, Epid-Kerat, Gallbladder; groups with >= 2 donors 35, >= 4 donors 21.

| Group | Samples | Donors |
|---|---|---|
| Adipocytes | 3 | 3 |
| Bladder-Ep | 5 | 4 |
| Blood-B | 5 | 5 |
| Blood-Granul | 3 | 3 |
| Blood-Mono+Macro | 11 | 8 |
| Blood-NK | 3 | 3 |
| Blood-T | 22 | 6 |
| Bone-Osteob | 1 | 1 |
| Breast-Basal-Ep | 4 | 4 |
| Breast-Luminal-Ep | 3 | 3 |
| Colon-Ep | 8 | 8 |
| Colon-Fibro | 2 | 2 |
| Dermal-Fibro | 1 | 1 |
| Endothel | 19 | 18 |
| Epid-Kerat | 1 | 1 |
| Eryth-prog | 3 | 3 |
| Fallopian-Ep | 3 | 3 |
| Gallbladder | 1 | 1 |
| Gastric-Ep | 11 | 6 |
| Head-Neck-Ep | 13 | 11 |
| Heart-Cardio | 4 | 4 |
| Heart-Fibro | 4 | 4 |
| Kidney-Ep | 8 | 3 |
| Liver-Hep | 6 | 6 |
| Lung-Ep-Alveo | 4 | 4 |
| Lung-Ep-Bron | 3 | 3 |
| Neuron | 10 | 7 |
| Oligodend | 4 | 4 |
| Ovary-Ep | 4 | 4 |
| Pancreas-Acinar | 4 | 4 |
| Pancreas-Alpha | 3 | 3 |
| Pancreas-Beta | 3 | 3 |
| Pancreas-Delta | 3 | 3 |
| Pancreas-Duct | 4 | 4 |
| Prostate-Ep | 4 | 4 |
| Skeletal-Musc | 2 | 2 |
| Small-Int-Ep | 5 | 4 |
| Smooth-Musc | 5 | 5 |
| Thyroid-Ep | 3 | 3 |

* **Excluded samples**: 48 (manual_decision); e.g. GSM5652214_Heart-Cardiomyocyte-Z0000044N.hg38.pat.gz.
* **Coordinates / duplicates / range**: pairs 386268, cpg_i<cpg_j True, duplicate pairs 0, endpoints in universe True, universe cpg_idx unique True, (chrom,pos) unique True; beta range [0.0, 1.0]. Genome build: counts-h5 / beta-h5 coordinates equal the universe covariates (checks above); the FASTA CG check is part of Level B (not executed).
* **Missingness**: CpGs observed in all 39 groups 382041, >= 20 groups 402638, < 20 groups 5761; per-group missing fraction (primary mask) min 0.0056 / max 0.0473 (Skeletal-Musc).
* **Aggregation rule** (frozen, read not run): cov >= 10; donor = published PatientID; donor beta = mean of valid sample betas; group beta = mean over donors; mask_primary = >= min(2, n_donors) valid donors; stored float16.
* **Target reproduction** (float16-rounded recomputation vs frozen `loyfer_pearson`, all 386268 pairs): max |diff| on pairs not touching a boundary cell 8.88e-16 (tolerance 1e-9); 1280 pairs touch a rounding-boundary cell (max |diff| over all pairs 0.00103).
* **Float16 rounding-boundary cells (attempt 1 FAIL and Amendment 2)**: 677 of 15756235 cells (4.3e-05) have round_f16(beta_f64) != frozen float16 (superseded threshold 1e-5: would have passed = False). Diagnosed mechanism (read-only diagnosis `step1_diag_f16_mismatch/diagnosis_attempt1.json`): all mismatch cells lie within 0.000284 float16 ULP (relative 1.73e-07) of a rounding midpoint, i.e. within float32 precision, consistent with float32 accumulation in the frozen aggregation; 676 distinct CpGs (max 2 per CpG); a float32 re-implementation reproduces 15756227 of 15756235 frozen cells and 673 of 677 mismatch cells (hypothesis supported, not bit-exact). New hard rule: relative distance <= 64 x 2^-24 = 3.8e-06; observed max 1.73e-07; cells beyond tolerance 0. **PASS_LEVEL_A passes only with Amendment 2.** The case is classified NON-BUG.
* **Effect bound of the float16 storage on the target**: unrounded float64 Pearson vs frozen: max |diff| 0.00493, median 0.000104, Spearman 0.9999988, fraction > 0.01: 0.0; primary on the unrounded target changes by at most 1.4e-06 for any arm (ordering changed: False).
* Informational: per-sample mean beta (array universe) vs published S1 'CpG methylation' Spearman 0.719 (n = 205).

## 2. Reliability / noise ceiling (section 2)

Donor-level split-half, 20 repetitions (`default_rng([17000, rep])`), equal halves (odd donor dropped), frozen aggregation rule per half; Spearman-Brown rho = 2r/(1+r); ceiling = sqrt(max(rho,0)) (descriptive attenuation bound). Median [2.5%, 97.5%] over repetitions.

**groups with >= 2 donors**: 35 groups, min shared groups 18, pairs defined per repetition median 384438.

| Cell | n (median) | r (Spearman) | rho (Spearman-Brown) | ceiling |
|---|---|---|---|---|
| all | 384438 | 0.514 [0.507, 0.532] | 0.679 [0.673, 0.694] | 0.824 [0.821, 0.833] |
| dist_<1kb | 37224 | 0.765 [0.761, 0.780] | 0.867 [0.864, 0.876] | 0.931 [0.929, 0.936] |
| dist_1-10kb | 37928 | 0.591 [0.583, 0.611] | 0.743 [0.736, 0.758] | 0.862 [0.858, 0.871] |
| dist_10-100kb | 38606 | 0.538 [0.530, 0.554] | 0.700 [0.693, 0.713] | 0.836 [0.833, 0.845] |
| dist_100kb-1Mb | 38632 | 0.462 [0.454, 0.480] | 0.632 [0.625, 0.649] | 0.795 [0.790, 0.806] |
| dist_>1Mb | 38730 | 0.439 [0.432, 0.458] | 0.610 [0.604, 0.628] | 0.781 [0.777, 0.792] |
| dist_interchromosomal | 193320 | 0.439 [0.431, 0.458] | 0.610 [0.602, 0.628] | 0.781 [0.776, 0.792] |
| chrom_intra | 191119 | 0.579 [0.574, 0.597] | 0.733 [0.730, 0.747] | 0.856 [0.854, 0.865] |
| chrom_inter | 193320 | 0.439 [0.431, 0.458] | 0.610 [0.602, 0.628] | 0.781 [0.776, 0.792] |
| V_Q1 | 96302 | 0.018 [0.013, 0.025] | 0.036 [0.026, 0.048] | 0.189 [0.161, 0.220] |
| V_Q2 | 96144 | 0.202 [0.192, 0.230] | 0.336 [0.323, 0.374] | 0.580 [0.568, 0.612] |
| V_Q3 | 95838 | 0.648 [0.637, 0.676] | 0.786 [0.778, 0.807] | 0.887 [0.882, 0.898] |
| V_Q4 | 96163 | 0.885 [0.879, 0.893] | 0.939 [0.935, 0.944] | 0.969 [0.967, 0.971] |
| V_top10pct | 38551 | 0.940 [0.935, 0.946] | 0.969 [0.966, 0.972] | 0.984 [0.983, 0.986] |
| pooled_gt1Mb_inter_eqw | 232047 | 0.439 [0.432, 0.458] | 0.610 [0.604, 0.628] | 0.781 [0.777, 0.792] |

**groups with >= 4 donors**: 21 groups, min shared groups 11, pairs defined per repetition median 384638.

| Cell | n (median) | r (Spearman) | rho (Spearman-Brown) | ceiling |
|---|---|---|---|---|
| all | 384638 | 0.474 [0.461, 0.487] | 0.643 [0.631, 0.655] | 0.802 [0.795, 0.809] |
| dist_<1kb | 37250 | 0.728 [0.719, 0.736] | 0.843 [0.837, 0.848] | 0.918 [0.915, 0.921] |
| dist_1-10kb | 37946 | 0.542 [0.528, 0.554] | 0.703 [0.691, 0.713] | 0.838 [0.831, 0.844] |
| dist_10-100kb | 38619 | 0.492 [0.475, 0.508] | 0.660 [0.644, 0.674] | 0.812 [0.802, 0.821] |
| dist_100kb-1Mb | 38653 | 0.427 [0.415, 0.441] | 0.598 [0.587, 0.612] | 0.774 [0.766, 0.782] |
| dist_>1Mb | 38749 | 0.405 [0.390, 0.417] | 0.576 [0.561, 0.589] | 0.759 [0.749, 0.767] |
| dist_interchromosomal | 193422 | 0.401 [0.387, 0.416] | 0.573 [0.558, 0.588] | 0.757 [0.747, 0.767] |
| chrom_intra | 191216 | 0.538 [0.527, 0.549] | 0.700 [0.690, 0.709] | 0.837 [0.831, 0.842] |
| chrom_inter | 193422 | 0.401 [0.387, 0.416] | 0.573 [0.558, 0.588] | 0.757 [0.747, 0.767] |
| V_Q1 | 96328 | 0.027 [0.021, 0.029] | 0.052 [0.041, 0.057] | 0.228 [0.202, 0.239] |
| V_Q2 | 96228 | 0.183 [0.172, 0.190] | 0.310 [0.293, 0.319] | 0.557 [0.541, 0.565] |
| V_Q3 | 95909 | 0.592 [0.573, 0.615] | 0.744 [0.729, 0.762] | 0.862 [0.854, 0.873] |
| V_Q4 | 96171 | 0.873 [0.860, 0.886] | 0.932 [0.925, 0.940] | 0.965 [0.962, 0.969] |
| V_top10pct | 38555 | 0.931 [0.922, 0.939] | 0.964 [0.959, 0.968] | 0.982 [0.980, 0.984] |
| pooled_gt1Mb_inter_eqw | 232171 | 0.403 [0.390, 0.417] | 0.574 [0.561, 0.588] | 0.758 [0.749, 0.767] |

Single-sample / single-donor groups (not estimable by split-half): Bone-Osteob, Dermal-Fibro, Epid-Kerat, Gallbladder. N = 1 groups are reported separately; every pair target includes them in all 39 columns.

## 3. Target semantics (section 3)

Thresholds recorded at run time (before examples were drawn): R_small (25th pct of pair min-range) = 0.0845; G_hi (75th pct of per-CpG R2 with the global group level) = 0.1601; V edges [0.000369276, 0.00133439, 0.00841199]; bands: top 1% >= 0.8365, bottom 1% <= -0.4169, median band [0.0696, 0.0752].

| Band | n | NEAR_CONSTANT | SMALL_AMPLITUDE | GLOBAL_LEVEL | SPECIFIC_PROGRAM | median mean beta | median min variance | median min range | median informative groups | median target | median Pearson after global-level removal | fraction persisting |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top1pct | 3847 | 0.000 | 0.000 | 0.144 | 0.856 | 0.520 | 0.0347 | 0.792 | 39 | 0.903 | 0.901 | 1.000 |
| median | 3847 | 0.290 | 0.028 | 0.030 | 0.651 | 0.460 | 0.0009 | 0.141 | 39 | 0.072 | 0.047 | 0.553 |
| bottom1pct | 3847 | 0.158 | 0.017 | 0.009 | 0.816 | 0.462 | 0.0064 | 0.369 | 39 | -0.478 | -0.501 | 0.999 |
| all | 384675 | 0.250 | 0.028 | 0.070 | 0.652 | 0.466 | 0.0013 | 0.176 | 39 | 0.072 | 0.025 | 0.775 |

**Dominance rule R2(c)**: share(near-constant u small-amplitude) in the top-1% band = 0.000, among all pairs = 0.278, enrichment 0.000; thresholds share >= 0.50 and enrichment >= 1.5; dominated = False.

Class shares per stratum:

| Stratum | n | NEAR_CONSTANT | SMALL_AMPLITUDE | GLOBAL_LEVEL | SPECIFIC_PROGRAM |
|---|---|---|---|---|---|
| <1kb | 37253 | 0.269 | 0.027 | 0.099 | 0.606 |
| 1-10kb | 37948 | 0.209 | 0.024 | 0.088 | 0.679 |
| 10-100kb | 38622 | 0.233 | 0.025 | 0.081 | 0.660 |
| 100kb-1Mb | 38658 | 0.254 | 0.028 | 0.066 | 0.651 |
| >1Mb | 38753 | 0.257 | 0.028 | 0.061 | 0.653 |
| interchromosomal | 193441 | 0.256 | 0.029 | 0.062 | 0.653 |

Deterministic examples (6 per band, seeded `default_rng([17, band])`, no manual selection): profile vectors are tabulated in `step3_4_semantics_pairs/semantics_examples.csv`; no figures are produced in this report.

## 4. Pair-list audit per distance stratum (section 4)

Mean per stratum (D2-kept pairs); pair-level value = mean of the two CpGs:

| Variable | <1kb | 1-10kb | 10-100kb | 100kb-1Mb | >1Mb | interchromosomal |
|---|---|---|---|---|---|---|
| pair mean beta | 0.380 | 0.506 | 0.491 | 0.470 | 0.467 | 0.468 |
| pair mean variance | 0.018 | 0.023 | 0.023 | 0.022 | 0.022 | 0.022 |
| pair mean range | 0.396 | 0.478 | 0.474 | 0.463 | 0.460 | 0.459 |
| GC | 0.571 | 0.537 | 0.536 | 0.540 | 0.541 | 0.541 |
| CpG density (hg38) | 49.959 | 37.885 | 38.480 | 40.010 | 40.440 | 40.439 |
| TSS distance (log10) | 2.492 | 2.835 | 2.835 | 2.775 | 2.770 | 2.770 |
| peak count total (regulatory tracks overlapped) | 456.922 | 341.646 | 350.481 | 370.267 | 372.048 | 371.374 |
| peak count histone | 376.515 | 286.634 | 289.682 | 305.255 | 306.866 | 306.333 |
| peak count DNase | 80.407 | 55.012 | 60.798 | 65.012 | 65.182 | 65.041 |
| target Pearson | 0.323 | 0.129 | 0.091 | 0.065 | 0.055 | 0.056 |

Standardised mean difference (Cohen d) / KS statistic of each stratum vs the reference set (pooled >1Mb + interchromosomal):

| Variable | <1kb | 1-10kb | 10-100kb | 100kb-1Mb | >1Mb | interchromosomal |
|---|---|---|---|---|---|---|
| pair mean beta | -0.319 / 0.290 | +0.145 / 0.071 | +0.086 / 0.048 | +0.008 / 0.017 | -0.004 / 0.004 | +0.001 / 0.001 |
| pair mean variance | -0.204 / 0.210 | +0.036 / 0.012 | +0.046 / 0.020 | +0.014 / 0.007 | +0.008 / 0.007 | -0.002 / 0.001 |
| pair mean range | -0.266 / 0.200 | +0.080 / 0.036 | +0.066 / 0.036 | +0.014 / 0.009 | +0.004 / 0.004 | -0.001 / 0.001 |
| GC | +0.433 / 0.187 | -0.062 / 0.033 | -0.071 / 0.049 | -0.019 / 0.029 | -0.002 / 0.008 | +0.000 / 0.002 |
| CpG density (hg38) | +0.442 / 0.162 | -0.124 / 0.060 | -0.094 / 0.047 | -0.021 / 0.020 | +0.000 / 0.004 | -0.000 / 0.001 |
| TSS distance (log10) | -0.368 / 0.227 | +0.089 / 0.051 | +0.088 / 0.037 | +0.007 / 0.012 | +0.000 / 0.004 | -0.000 / 0.001 |
| peak count total (regulatory tracks overlapped) | +0.339 / 0.226 | -0.125 / 0.066 | -0.088 / 0.052 | -0.005 / 0.018 | +0.002 / 0.004 | -0.000 / 0.001 |
| peak count histone | +0.345 / 0.228 | -0.102 / 0.068 | -0.087 / 0.053 | -0.006 / 0.018 | +0.002 / 0.003 | -0.000 / 0.001 |
| peak count DNase | +0.173 / 0.049 | -0.120 / 0.056 | -0.050 / 0.037 | -0.001 / 0.010 | +0.001 / 0.003 | -0.000 / 0.001 |
| |d| mean beta | -0.949 / 0.434 | -0.121 / 0.066 | -0.089 / 0.047 | -0.064 / 0.031 | -0.008 / 0.005 | +0.002 / 0.001 |
| |d| GC | -0.900 / 0.429 | -0.183 / 0.094 | -0.212 / 0.102 | -0.150 / 0.065 | -0.036 / 0.018 | +0.007 / 0.004 |
| |d| CpG density | -0.798 / 0.404 | -0.069 / 0.087 | -0.112 / 0.088 | -0.062 / 0.052 | -0.013 / 0.008 | +0.003 / 0.002 |
| |d| peak count total | -0.935 / 0.418 | -0.257 / 0.134 | -0.147 / 0.071 | -0.069 / 0.032 | -0.008 / 0.005 | +0.002 / 0.001 |
| |d| TSS | -0.852 / 0.384 | -0.323 / 0.124 | -0.106 / 0.046 | -0.056 / 0.026 | -0.014 / 0.008 | +0.003 / 0.002 |

Context and chromosome composition, zero-row exclusion (candidate store: 849 all-zero loci = D2 union):

| Stratum | same-context share | island share (CpG i) | JS distance of chromosome composition vs reference | pairs excluded by D2 (of n) |
|---|---|---|---|---|
| <1kb | 0.709 | 0.425 | 0.228 | 62 of 37315 |
| 1-10kb | 0.396 | 0.269 | 0.235 | 219 of 38167 |
| 10-100kb | 0.343 | 0.292 | 0.232 | 163 of 38785 |
| 100kb-1Mb | 0.316 | 0.309 | 0.233 | 161 of 38819 |
| >1Mb | 0.296 | 0.310 | 0.238 | 161 of 38914 |
| interchromosomal | 0.289 | 0.319 | 0.079 | 827 of 194268 |

**Why the arm ranking differs between short and long range (only what the data show)**: per-stratum Spearman(target, cosine) of each arm and the partial Spearman controlling for covariate closeness (-|dM|, -|dGC|, -|d peak count|):

| Arm | <1kb | 1-10kb | 10-100kb | 100kb-1Mb | >1Mb | interchromosomal |
|---|---|---|---|---|---|---|
| Histone+DNase (candidate) | +0.244 | +0.197 | +0.068 | -0.012 | -0.061 | -0.064 |
| functional PCA (legacy control) | +0.298 | +0.212 | +0.091 | +0.012 | -0.037 | -0.039 |
| CpGPT-large | +0.140 | +0.237 | +0.192 | +0.137 | +0.108 | +0.105 |
| DeepCpG-DNA | +0.203 | +0.239 | +0.188 | +0.126 | +0.114 | +0.117 |

Partial Spearman given [-|dM|, -|dGC|, -|d peak total|]:

| Arm | <1kb | 1-10kb | 10-100kb | 100kb-1Mb | >1Mb | interchromosomal |
|---|---|---|---|---|---|---|
| Histone+DNase (candidate) | +0.091 | +0.012 | -0.031 | -0.064 | -0.095 | -0.098 |
| functional PCA (legacy control) | +0.146 | +0.024 | -0.019 | -0.052 | -0.082 | -0.085 |
| CpGPT-large | +0.060 | +0.055 | +0.070 | +0.052 | +0.047 | +0.041 |
| DeepCpG-DNA | +0.071 | +0.087 | +0.068 | +0.044 | +0.053 | +0.053 |

Ordering of the arms per stratum (by Spearman): <1kb: functional PCA (legacy control) > Histone+DNase (candidate) > DeepCpG-DNA > CpGPT-large; 1-10kb: DeepCpG-DNA > CpGPT-large > functional PCA (legacy control) > Histone+DNase (candidate); 10-100kb: CpGPT-large > DeepCpG-DNA > functional PCA (legacy control) > Histone+DNase (candidate); 100kb-1Mb: CpGPT-large > DeepCpG-DNA > functional PCA (legacy control) > Histone+DNase (candidate); >1Mb: DeepCpG-DNA > CpGPT-large > functional PCA (legacy control) > Histone+DNase (candidate); interchromosomal: DeepCpG-DNA > CpGPT-large > functional PCA (legacy control) > Histone+DNase (candidate).

Spearman of the cosine with -|d peak total| (candidate): <1kb +0.509, 1-10kb +0.515, 10-100kb +0.337, 100kb-1Mb +0.195, >1Mb +0.125, interchromosomal +0.118; of the target with -|d peak total|: <1kb +0.394, 1-10kb +0.333, 10-100kb +0.269, 100kb-1Mb +0.196, >1Mb +0.158, interchromosomal +0.163.

## 5. Baselines on the same frozen pair universe (section 5)

Native cosine, primary-style pooled statistic (>1Mb + inter, equal stratum weight) and per stratum, with CIs; reproduces the frozen primary of every arm (checked at run time to 1e-9):

| Arm | dim | pooled primary-style [CI] | <1kb | 1-10kb | 10-100kb | 100kb-1Mb | >1Mb | interchromosomal |
|---|---|---|---|---|---|---|---|---|
| Histone+DNase (candidate) | 256 | -0.0628 [-0.0724, -0.0534] | +0.244 | +0.197 | +0.068 | -0.012 | -0.061 | -0.064 |
| functional PCA (legacy control) | 256 | -0.0381 [-0.0459, -0.0301] | +0.298 | +0.212 | +0.091 | +0.012 | -0.037 | -0.039 |
| CpGPT-large | 512 | +0.1065 [+0.1010, +0.1121] | +0.140 | +0.237 | +0.192 | +0.137 | +0.108 | +0.105 |
| DeepCpG-DNA | 128 | +0.1155 [+0.1079, +0.1220] | +0.203 | +0.239 | +0.188 | +0.126 | +0.114 | +0.117 |

Negative controls per arm (100 row-permutations of the embedding among CpGs, seeds 1000-1099; 100 random Gaussian embeddings of the arm dimension, seeds 2000-2099); pooled primary-style statistic:

| Arm | shuffled mean (sd) | shuffled [q2.5, q97.5] | Gaussian mean (sd) | Gaussian [q2.5, q97.5] | NULL_HI | native - NULL_HI |
|---|---|---|---|---|---|---|
| Histone+DNase (candidate) | +0.0004 (0.0029) | [-0.0053, 0.0053] | +0.0004 (0.0028) | [-0.0052, 0.0052] | 0.0053 | -0.0681 |
| functional PCA (legacy control) | +0.0005 (0.0028) | [-0.0050, 0.0051] | +0.0004 (0.0028) | [-0.0052, 0.0052] | 0.0052 | -0.0434 |
| CpGPT-large | -0.0002 (0.0026) | [-0.0055, 0.0050] | -0.0003 (0.0027) | [-0.0054, 0.0050] | 0.0050 | +0.1015 |
| DeepCpG-DNA | +0.0003 (0.0025) | [-0.0053, 0.0050] | -0.0002 (0.0029) | [-0.0055, 0.0055] | 0.0053 | +0.1101 |

Chromosome-only, distance and simple covariate baselines (score vs target; pooled primary-style [CI] and per stratum; 'n/a' = undefined by construction: chromosome-only is constant within one stratum, distance is undefined for interchromosomal pairs):

| Baseline | pooled primary-style [CI] | <1kb | 1-10kb | 10-100kb | 100kb-1Mb | >1Mb | interchromosomal | all pairs (unweighted) |
|---|---|---|---|---|---|---|---|---|
| chromosome_only | -0.0020 [-0.0091, +0.0053] | n/a | n/a | n/a | n/a | n/a | n/a | +0.118 |
| neg_log10_distance | n/a | +0.260 | -0.005 | +0.032 | +0.002 | -0.016 | n/a | +0.256 |
| same_context | +0.0401 [+0.0295, +0.0493] | +0.200 | +0.161 | +0.100 | +0.061 | +0.041 | +0.039 | +0.130 |
| neg_abs_delta_methylation_mean | +0.1091 [+0.1031, +0.1160] | -0.193 | +0.259 | +0.177 | +0.132 | +0.107 | +0.112 | +0.172 |
| neg_abs_delta_methylation_variance | +0.0270 [+0.0202, +0.0341] | -0.328 | +0.059 | +0.036 | +0.038 | +0.026 | +0.028 | +0.047 |
| neg_abs_delta_amplitude_range | +0.0600 [+0.0540, +0.0667] | -0.034 | +0.186 | +0.110 | +0.078 | +0.057 | +0.063 | +0.120 |
| neg_abs_delta_gc | +0.0224 [+0.0160, +0.0285] | +0.304 | +0.182 | +0.115 | +0.057 | +0.022 | +0.023 | +0.134 |
| neg_abs_delta_cpg_density | +0.1076 [+0.1005, +0.1155] | +0.293 | +0.201 | +0.174 | +0.123 | +0.107 | +0.109 | +0.187 |
| neg_abs_delta_cpg_density_hg38 | +0.1052 [+0.0987, +0.1118] | +0.354 | +0.263 | +0.209 | +0.131 | +0.103 | +0.107 | +0.206 |
| neg_abs_delta_tss_dist_log10 | +0.0436 [+0.0365, +0.0506] | +0.343 | +0.168 | +0.088 | +0.041 | +0.040 | +0.048 | +0.144 |
| neg_abs_delta_peak_count_total | +0.1607 [+0.1515, +0.1707] | +0.394 | +0.333 | +0.269 | +0.196 | +0.158 | +0.163 | +0.267 |
| neg_abs_delta_peak_count_histone | +0.1488 [+0.1388, +0.1592] | +0.387 | +0.311 | +0.256 | +0.180 | +0.148 | +0.150 | +0.256 |
| neg_abs_delta_peak_count_dnase | +0.1503 [+0.1416, +0.1592] | +0.405 | +0.285 | +0.231 | +0.170 | +0.144 | +0.156 | +0.217 |

Distance baseline over the five intra strata with equal stratum weight: +0.2589 [+0.2486, +0.2692].

**CpGPT 0.1065 vs simple controls**: baselines in the same range or above: neg_abs_delta_peak_count_total +0.161, neg_abs_delta_peak_count_dnase +0.150, neg_abs_delta_peak_count_histone +0.149, neg_abs_delta_methylation_mean +0.109, neg_abs_delta_cpg_density +0.108, neg_abs_delta_cpg_density_hg38 +0.105, neg_abs_delta_amplitude_range +0.060. Null controls reach at most 0.0070.

**Precision of the frozen (float16) target**: primary-style statistic on the UNROUNDED target per arm: Histone+DNase (candidate) -0.062792 (delta -4.04e-08); functional PCA (legacy control) -0.038137 (delta 1.39e-06); CpGPT-large +0.106476 (delta -8.21e-07); DeepCpG-DNA +0.115454 (delta -1.4e-06); ordering frozen: DeepCpG-DNA > CpGPT-large > functional PCA (legacy control) > Histone+DNase (candidate).

Paired contrasts of the native statistic (candidate - comparator, same draws): vs functional PCA (legacy control): -0.0247 [-0.0284, -0.0213]; vs CpGPT-large: -0.1693 [-0.1810, -0.1578]; vs DeepCpG-DNA: -0.1782 [-0.1929, -0.1627].

## 6. Raw regulatory features vs 256-d SVD embedding (section 6)

Raw space: 2492 binary tracks (histone + DNase, resolver `regulatory_histone_dnase`); CSR only. Compressor weights all 1: True (the column-weighted cosine IS the binary cosine). Raw all-zero loci 849; pairs undefined after D2: 0; common pair set 384675 (D2 set 384675). Projection reproduction on 200 rows: max |diff| 1.66e-05, registered tolerance 1e-5, `ok` = False (**tolerance not met; float32 matrix product, informational: no decision rule uses it**).

| Score | pooled primary-style [CI] | all pairs | >1Mb | interchromosomal | V_top10pct pooled | V_top10pct unweighted |
|---|---|---|---|---|---|---|
| Raw binary cosine | -0.0854 [-0.0954, -0.0752] | +0.081 | -0.084 | -0.087 | +0.052 | +0.418 |
| Raw Jaccard | -0.0759 [-0.0850, -0.0664] | +0.088 | -0.075 | -0.077 | +0.045 | +0.412 |
| Native 256-d cosine | -0.0628 [-0.0724, -0.0534] | +0.108 | -0.061 | -0.064 | +0.068 | +0.425 |

Per stratum Spearman: raw_binary_cosine: <1kb +0.152, 1-10kb +0.150, 10-100kb +0.032, 100kb-1Mb -0.039, >1Mb -0.084, interchromosomal -0.087; raw_jaccard: <1kb +0.150, 1-10kb +0.159, 10-100kb +0.042, 100kb-1Mb -0.028, >1Mb -0.075, interchromosomal -0.077; embedding256_cosine_native: <1kb +0.244, 1-10kb +0.197, 10-100kb +0.068, 100kb-1Mb -0.012, >1Mb -0.061, interchromosomal -0.064.

Paired delta (raw - native 256-d, pooled): raw_binary_cosine: -0.0226 [-0.0238, -0.0214]; raw_jaccard: -0.0131 [-0.0142, -0.0119]. Reading under the registered rule: the raw sources do not recover the target (both are below NULL_HI + 0.03 and below the native cosine); the failure is present in the raw feature space.

## 7. SVD / embedding geometry (section 7; descriptive)

Embedding-space PCA of the candidate over 407550 D2 universe loci. First 20 PCs: explained variance, Spearman of the PC score with covariates, eta^2 with CpG context, and the Spearman of the target with the PC's centred dot-product contribution (s_ik s_jk, pooled primary-style).

| PC | EV ratio | peak_total | peak_histone | peak_dnase | gc_content | cpg_density | cpg_density_hg38 | tss_dist_log10 | loyfer_mean_methylation | loyfer_methylation_variance | loyfer_amplitude_range | eta2 context | dot-contribution Spearman |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.4640 | +0.92 | +0.88 | +0.74 | +0.42 | +0.57 | +0.61 | -0.60 | -0.85 | -0.44 | -0.45 | 0.367 | +0.1328 |
| 2 | 0.1036 | +0.27 | +0.48 | -0.17 | +0.16 | +0.14 | +0.20 | -0.09 | -0.21 | -0.03 | -0.03 | 0.040 | +0.0141 |
| 3 | 0.0716 | +0.16 | +0.10 | +0.31 | +0.20 | +0.13 | +0.19 | -0.09 | -0.25 | +0.17 | +0.14 | 0.082 | +0.0365 |
| 4 | 0.0339 | +0.17 | +0.14 | +0.09 | -0.12 | -0.16 | -0.21 | +0.10 | +0.12 | +0.24 | +0.24 | 0.053 | +0.0347 |
| 5 | 0.0274 | +0.36 | +0.38 | +0.17 | +0.45 | +0.37 | +0.52 | -0.29 | -0.21 | -0.43 | -0.42 | 0.063 | +0.0628 |
| 6 | 0.0186 | +0.23 | +0.24 | +0.16 | +0.09 | +0.03 | +0.04 | -0.02 | -0.09 | +0.09 | +0.09 | 0.000 | +0.0140 |
| 7 | 0.0144 | +0.18 | +0.19 | +0.01 | -0.01 | +0.14 | +0.06 | -0.10 | -0.12 | -0.14 | -0.13 | 0.031 | +0.0242 |
| 8 | 0.0113 | -0.09 | -0.11 | +0.03 | -0.23 | -0.18 | -0.17 | +0.13 | +0.03 | +0.01 | +0.01 | 0.010 | +0.0075 |
| 9 | 0.0102 | -0.02 | -0.02 | +0.08 | +0.48 | +0.07 | +0.27 | -0.06 | +0.00 | +0.02 | +0.00 | 0.074 | +0.0059 |
| 10 | 0.0091 | -0.04 | -0.03 | -0.06 | +0.06 | -0.04 | +0.04 | -0.03 | -0.01 | +0.05 | +0.04 | 0.016 | +0.0086 |
| 11 | 0.0079 | +0.02 | +0.01 | +0.04 | +0.25 | +0.08 | +0.16 | -0.03 | -0.04 | -0.05 | -0.05 | 0.023 | +0.0073 |
| 12 | 0.0074 | +0.05 | +0.05 | +0.03 | +0.08 | +0.02 | +0.06 | -0.03 | -0.05 | -0.06 | -0.07 | 0.005 | +0.0134 |
| 13 | 0.0061 | +0.03 | +0.03 | +0.09 | -0.11 | -0.02 | -0.06 | -0.03 | -0.06 | +0.09 | +0.07 | 0.009 | +0.0098 |
| 14 | 0.0053 | -0.04 | -0.03 | -0.11 | +0.08 | +0.02 | +0.00 | -0.00 | +0.04 | +0.01 | +0.02 | 0.007 | +0.0193 |
| 15 | 0.0050 | +0.03 | +0.04 | -0.11 | -0.09 | +0.02 | -0.02 | -0.04 | -0.01 | +0.01 | +0.00 | 0.004 | +0.0070 |
| 16 | 0.0048 | -0.06 | -0.07 | +0.02 | -0.16 | -0.09 | -0.13 | +0.02 | +0.04 | +0.03 | +0.03 | 0.013 | +0.0114 |
| 17 | 0.0042 | -0.01 | -0.02 | +0.06 | -0.16 | -0.12 | -0.15 | +0.07 | +0.03 | +0.11 | +0.11 | 0.009 | +0.0024 |
| 18 | 0.0038 | -0.08 | -0.09 | +0.02 | -0.07 | -0.02 | -0.07 | +0.04 | +0.08 | +0.04 | +0.04 | 0.007 | +0.0123 |
| 19 | 0.0037 | +0.00 | +0.00 | +0.05 | -0.11 | -0.00 | -0.07 | +0.00 | -0.03 | +0.06 | +0.05 | 0.016 | +0.0096 |
| 20 | 0.0035 | +0.01 | +0.01 | -0.01 | -0.12 | -0.01 | -0.05 | -0.05 | -0.03 | -0.04 | -0.03 | 0.001 | +0.0018 |

Raw embedding coordinates 1-5 (the SVD component axes) vs covariates (Spearman): coord 1: peak_total +0.97, gc_content +0.46, loyfer_mean_methylation -0.85; coord 2: peak_total +0.21, gc_content +0.17, loyfer_mean_methylation -0.13; coord 3: peak_total -0.02, gc_content +0.13, loyfer_mean_methylation +0.01; coord 4: peak_total -0.22, gc_content -0.31, loyfer_mean_methylation +0.49; coord 5: peak_total +0.04, gc_content +0.30, loyfer_mean_methylation -0.01.

**Diagnostic curve (primary-style pooled [CI]; V_top10pct pooled point estimate)**; centred = mean removed; PCk removed = project out the top-k axes of the centred embedding; whitening eps 1e-6 lambda_max:

| Geometry | Histone+DNase (candidate) | functional PCA (legacy control) | CpGPT-large | DeepCpG-DNA |
|---|---|---|---|---|
| native_cosine | -0.063 [-0.072, -0.053] | -0.038 [-0.046, -0.030] | +0.106 [+0.101, +0.112] | +0.115 [+0.108, +0.122] |
| centered_cosine | +0.174 [+0.165, +0.183] | +0.180 [+0.170, +0.189] | +0.095 [+0.089, +0.100] | +0.110 [+0.103, +0.115] |
| remove_pc1 | +0.086 [+0.076, +0.096] | +0.093 [+0.083, +0.104] | +0.072 [+0.067, +0.079] | +0.075 [+0.067, +0.082] |
| remove_pc1_2 | +0.087 [+0.077, +0.097] | +0.092 [+0.082, +0.102] | +0.061 [+0.055, +0.067] | +0.014 [+0.008, +0.019] |
| remove_pc1_5 | +0.049 [+0.043, +0.055] | +0.059 [+0.053, +0.066] | +0.058 [+0.054, +0.062] | +0.006 [+0.001, +0.012] |
| remove_pc1_10 | +0.032 [+0.027, +0.037] | +0.032 [+0.027, +0.038] | +0.057 [+0.053, +0.060] | +0.003 [-0.003, +0.008] |
| standardized_cosine | +0.041 [+0.037, +0.045] | +0.044 [+0.039, +0.049] | +0.069 [+0.064, +0.074] | +0.098 [+0.091, +0.104] |
| whitened_cosine | +0.041 [+0.036, +0.045] | +0.039 [+0.035, +0.044] | +0.026 [+0.019, +0.033] | +0.022 [+0.016, +0.027] |
| neg_euclidean | +0.202 [+0.192, +0.212] | +0.209 [+0.199, +0.219] | +0.076 [+0.072, +0.080] | +0.118 [+0.110, +0.125] |

Same curve, V_top10pct pooled (point estimate) and per stratum for the candidate:

| Geometry | V_top10pct (candidate) | <1kb (candidate) | 1-10kb (candidate) | 10-100kb (candidate) | 100kb-1Mb (candidate) | >1Mb (candidate) | interchromosomal (candidate) |
|---|---|---|---|---|---|---|---|
| native_cosine | +0.068 | +0.244 | +0.197 | +0.068 | -0.012 | -0.061 | -0.064 |
| centered_cosine | +0.088 | +0.423 | +0.371 | +0.296 | +0.212 | +0.172 | +0.176 |
| remove_pc1 | +0.141 | +0.518 | +0.352 | +0.238 | +0.134 | +0.087 | +0.084 |
| remove_pc1_2 | +0.148 | +0.547 | +0.352 | +0.239 | +0.134 | +0.089 | +0.084 |
| remove_pc1_5 | +0.160 | +0.561 | +0.344 | +0.219 | +0.107 | +0.048 | +0.050 |
| remove_pc1_10 | +0.128 | +0.561 | +0.326 | +0.211 | +0.096 | +0.032 | +0.032 |
| standardized_cosine | +0.087 | +0.555 | +0.334 | +0.214 | +0.101 | +0.039 | +0.043 |
| whitened_cosine | +0.089 | +0.558 | +0.340 | +0.224 | +0.106 | +0.040 | +0.043 |
| neg_euclidean | +0.080 | +0.512 | +0.388 | +0.321 | +0.234 | +0.199 | +0.204 |

Descriptive attribution (candidate): the centred dot-product contribution of each of the 20 PCs has Spearman with the target between +0.0018 and +0.1328 (all non-negative: True); the largest are PC1 +0.133, PC5 +0.063, PC3 +0.036. The uncentered native cosine is -0.063 whereas the centred cosine is +0.174: subtracting the universe mean embedding alone changes the sign. PC1: EV 0.464, Spearman with peak count +0.915, with Loyfer mean methylation -0.854. No claim is made on mechanism beyond these descriptive numbers.

Consistency with the earlier B2 output: centred 0.173924 vs 0.173924; PC1-removed 0.085636 vs 0.085636.

## 8. Linear decodability <-> pairwise geometry (section 8)

Ridge = `ridge_blocked_cv` unchanged, frozen chromosome folds (seed 17), original alpha grid [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]; 382,041 complete-case CpGs; OOF predictions strictly out of fold (independent closed-form refit check max |diff| 9.15e-12; stored B3 r2_global reproduced, max |diff| 4.9e-10). Pair set: both CpGs complete-case and D2-kept = 346503 pairs (of 384675); same-fold pairs 199974 (interchromosomal 27609). The two CpGs of a pair may be predicted by models of different folds (always the same fold for intra pairs).

| Arm | Score | pooled primary-style [CI] | >1Mb | inter | all pairs (unweighted) | V_top10pct | same-fold only pooled | delta vs native (same pairs) |
|---|---|---|---|---|---|---|---|---|
| Histone+DNase (candidate) | T1 raw | +0.2767 [+0.2651, +0.2889] | +0.278 | +0.275 | +0.389 | +0.426 | +0.277 [+0.264, +0.291] | +0.340 [0.320, 0.359] |
| Histone+DNase (candidate) | T2 centred | +0.2769 [+0.2653, +0.2890] | +0.278 | +0.275 | +0.389 | +0.425 | +0.277 [+0.264, +0.292] | +0.340 [0.320, 0.359] |
| Histone+DNase (candidate) | native cosine (same pairs) | -0.0629 [-0.0723, -0.0534] |  |  |  |  |  |  |
| functional PCA (legacy control) | T1 raw | +0.2796 [+0.2684, +0.2917] | +0.280 | +0.279 | +0.391 | +0.427 | +0.280 [+0.268, +0.295] | +0.315 [0.299, 0.332] |
| functional PCA (legacy control) | T2 centred | +0.2798 [+0.2686, +0.2919] | +0.280 | +0.280 | +0.391 | +0.427 | +0.280 [+0.268, +0.295] | +0.316 [0.299, 0.332] |
| functional PCA (legacy control) | native cosine (same pairs) | -0.0358 [-0.0429, -0.0279] |  |  |  |  |  |  |
| CpGPT-large | T1 raw | +0.1751 [+0.1694, +0.1814] | +0.172 | +0.178 | +0.244 | +0.166 | +0.175 [+0.167, +0.184] | +0.061 [0.053, 0.068] |
| CpGPT-large | T2 centred | +0.1746 [+0.1689, +0.1809] | +0.172 | +0.177 | +0.243 | +0.166 | +0.175 [+0.167, +0.184] | +0.060 [0.053, 0.067] |
| CpGPT-large | native cosine (same pairs) | +0.1143 [+0.1093, +0.1201] |  |  |  |  |  |  |
| DeepCpG-DNA | T1 raw | +0.1559 [+0.1476, +0.1646] | +0.154 | +0.158 | +0.231 | +0.055 | +0.156 [+0.146, +0.166] | +0.038 [0.031, 0.044] |
| DeepCpG-DNA | T2 centred | +0.1557 [+0.1475, +0.1644] | +0.154 | +0.158 | +0.231 | +0.055 | +0.156 [+0.145, +0.166] | +0.037 [0.031, 0.044] |
| DeepCpG-DNA | native cosine (same pairs) | +0.1184 [+0.1113, +0.1251] |  |  |  |  |  |  |

Per stratum OOF T1 Spearman: Histone+DNase (candidate): <1kb +0.626, 1-10kb +0.448, 10-100kb +0.383, 100kb-1Mb +0.300, >1Mb +0.278, interchromosomal +0.275; functional PCA (legacy control): <1kb +0.626, 1-10kb +0.449, 10-100kb +0.383, 100kb-1Mb +0.301, >1Mb +0.280, interchromosomal +0.279; CpGPT-large: <1kb +0.354, 1-10kb +0.266, 10-100kb +0.247, 100kb-1Mb +0.187, >1Mb +0.172, interchromosomal +0.178; DeepCpG-DNA: <1kb +0.451, 1-10kb +0.278, 10-100kb +0.237, 100kb-1Mb +0.166, >1Mb +0.154, interchromosomal +0.158.

Linear probe R2 (original grid; global, pooled over CpGs x groups): Histone+DNase (candidate): raw__all_cpgs 0.776, raw__variable_cpgs 0.472, centered__all_cpgs 0.245, centered__variable_cpgs 0.279, level__all_cpgs 0.862, level__variable_cpgs 0.603; functional PCA (legacy control): raw__all_cpgs 0.785, raw__variable_cpgs 0.484, centered__all_cpgs 0.244, centered__variable_cpgs 0.278, level__all_cpgs 0.873, level__variable_cpgs 0.625; CpGPT-large: raw__all_cpgs 0.544, raw__variable_cpgs 0.100, centered__all_cpgs 0.039, centered__variable_cpgs 0.031, level__all_cpgs 0.625, level__variable_cpgs 0.147; DeepCpG-DNA: raw__all_cpgs 0.504, raw__variable_cpgs 0.105, centered__all_cpgs 0.023, centered__variable_cpgs 0.009, level__all_cpgs 0.581, level__variable_cpgs 0.171.

Paired contrasts on the OOF T1 statistic (candidate - comparator): vs functional PCA (legacy control): -0.0029 [-0.0058, -0.0006]; vs CpGPT-large: +0.1017 [0.0935, 0.1101]; vs DeepCpG-DNA: +0.1208 [0.1117, 0.1302].

## 9. Probe fairness: extended alpha grid (section 9; sensitivity only)

Registered grid 10^-2..10^8 (11 values); edge rule: an arm selects an edge if >= 10% of its (fold x target) alphas sit at the grid min/max; then ALL arms are refitted with the grid extended by 2 decades on that side, at most 2 extensions. Executed: 2 extensions, final grid ['1e-06', '1e-05', '0.0001', '0.001', '0.01', '0.1', '1', '10', '100', '1000', '10000', '100000', '1e+06', '1e+07', '1e+08'].

| Round | n alphas | grid | Histone+DNase (candidate) min / max edge fraction | functional PCA (legacy control) min / max edge fraction | CpGPT-large min / max edge fraction | DeepCpG-DNA min / max edge fraction |
|---|---|---|---|---|---|---|
| 0 | 11 | 0.01 .. 1e+08 | 0.190 / 0.000 | 0.084 / 0.000 | 0.306 / 0.000 | 0.000 / 0.000 |
| 1 | 13 | 0.0001 .. 1e+08 | 0.190 / 0.000 | 0.081 / 0.000 | 0.008 / 0.000 | 0.000 / 0.000 |
| 2 | 15 | 1e-06 .. 1e+08 | 0.190 / 0.000 | 0.078 / 0.000 | 0.008 / 0.000 | 0.000 / 0.000 |

After the maximum number of extensions the candidate still has 0.190 of its selections at the grid minimum (rule threshold 0.10): the edge condition is **not resolved** for that arm; selections at the minimum alpha mean the smallest registered regularisation was preferred.

| Arm | orig min frac | orig max frac | ext min frac | ext max frac | median alpha per fold (orig) | median alpha per fold (ext) |
|---|---|---|---|---|---|---|
| Histone+DNase (candidate) | 0.190 | 0.089 | 0.190 | 0.000 | 100, 100, 100, 100, 100 | 100, 100, 100, 100, 100 |
| functional PCA (legacy control) | 0.084 | 0.078 | 0.078 | 0.000 | 100, 100, 100, 100, 100 | 100, 100, 100, 100, 100 |
| CpGPT-large | 0.306 | 0.453 | 0.008 | 0.000 | 10, 10, 10, 10, 0.01 | 10, 10, 10, 10, 0.01 |
| DeepCpG-DNA | 0.000 | 0.018 | 0.000 | 0.000 | 100, 100, 100, 100, 100 | 100, 100, 100, 100, 100 |

r2_global (T1 raw / T2 centred / T3 level; all CpGs / variable CpGs), original -> extended grid:

| Arm | raw__all_cpgs | raw__variable_cpgs | centered__all_cpgs | centered__variable_cpgs | level__all_cpgs | level__variable_cpgs |
|---|---|---|---|---|---|---|
| Histone+DNase (candidate) | 0.7764 -> 0.7764 | 0.4717 -> 0.4717 | 0.2446 -> 0.2446 | 0.2793 -> 0.2793 | 0.8622 -> 0.8622 | 0.6028 -> 0.6028 |
| functional PCA (legacy control) | 0.7851 -> 0.7851 | 0.4842 -> 0.4842 | 0.2436 -> 0.2436 | 0.2783 -> 0.2783 | 0.8725 -> 0.8725 | 0.6247 -> 0.6247 |
| CpGPT-large | 0.5436 -> 0.5436 | 0.1003 -> 0.1003 | 0.0388 -> 0.0389 | 0.0314 -> 0.0308 | 0.6251 -> 0.6251 | 0.1472 -> 0.1472 |
| DeepCpG-DNA | 0.5036 -> 0.5036 | 0.1054 -> 0.1054 | 0.0230 -> 0.0230 | 0.0090 -> 0.0090 | 0.5811 -> 0.5811 | 0.1713 -> 0.1713 |

Ranking of the arms by r2_global (centred, variable CpGs): original Histone+DNase (candidate) > functional PCA (legacy control) > CpGPT-large > DeepCpG-DNA; extended Histone+DNase (candidate) > functional PCA (legacy control) > CpGPT-large > DeepCpG-DNA. Ranking T1 all CpGs: original functional PCA (legacy control) > Histone+DNase (candidate) > CpGPT-large > DeepCpG-DNA; extended functional PCA (legacy control) > Histone+DNase (candidate) > CpGPT-large > DeepCpG-DNA. Ranking changes: False.

OOF T1 pooled statistic under the extended grid: Histone+DNase (candidate): +0.277 [+0.265, +0.289]; functional PCA (legacy control): +0.280 [+0.268, +0.292]; CpGPT-large: +0.175 [+0.169, +0.181]; DeepCpG-DNA: +0.156 [+0.148, +0.165]; ranking functional PCA (legacy control) > Histone+DNase (candidate) > CpGPT-large > DeepCpG-DNA. The original ridge results are not replaced.

## 10. Pretraining provenance (section 10; documentary, reused)

Source: `docs/BIOLOGICAL_VALIDATION_V2_EXPLORATORY_POSTHOC_PROVENANCE_AUDIT.md` (labels: VERIFIED-IN-REPO, VERIFIED-LOCAL-SIBLING, VERIFIED-EXTERNAL fetched 2026-10-04, UNVERIFIED); no new web search was made for this report. The classification below restates that audit; it does not reinterpret the primary.

| Item | CpGPT-large | DeepCpG-DNA (HCC) |
|---|---|---|
| Pretraining objective | methylation-value reconstruction / uncertainty, Wasserstein on beta distribution, contrastive and sample losses (VERIFIED-LOCAL-SIBLING config `large.yaml`) | prediction of single-cell CpG methylation states from the DNA window (zoo/README statement VERIFIED-IN-REPO; paper text UNVERIFIED) |
| Corpus | CpGCorpus: Illumina arrays (450k/EPIC/EPICv2), >150,000 human bulk samples from >2,000 studies (VERIFIED-EXTERNAL, preprint summary and GitHub README) | Hou et al. 2016 scRRBS, 25 human hepatocellular carcinoma cells (VERIFIED-IN-REPO README; VERIFIED-EXTERNAL zoo page) |
| Direct supervision on methylation | YES (VERIFIED) | YES (VERIFIED as training fact) |
| Overlap with Loyfer individuals/datasets (GSE186458, WGBS of sorted cell types) | direct data overlap UNVERIFIED (arrays vs WGBS; whether Loyfer samples are in CpGCorpus not checked) | no evidence (cancer cell-line scRRBS) |
| Classification | SAME_ASSAY_DOMAIN (methylation objective on array data at the benchmark loci); NO_EVIDENCE_OF_OVERLAP for direct data overlap (UNVERIFIED) | SAME_ASSAY_DOMAIN (methylation states, different tissue/assay); NO_EVIDENCE_OF_OVERLAP |
| Not verifiable here | exact CpGCorpus composition; whether the 512-d locus vector retains methylation information beyond sequence | training CpG set of the HCC DNA module; original paper text |

Documentary outcome (fixed in the registration, independent of any result): both sequence arms = DIRECT_METHYLATION_SUPERVISION. This audit cannot separate 'the ENCODE-feature arms lack the information' from 'the sequence arms were trained on methylation'.

## 11. Decision tree: mechanical application of the registered rules

Constants: M_REC = 0.03; NULL_HI (97.5th percentile of the 200 nulls) of the candidate = 0.00527; 'recovers' = statistic >= NULL_HI + 0.03 = 0.03527.

| Class | Flag | Evidence / margin to the registered threshold |
|---|---|---|
| BUG | False | step-1 verdict PASS_LEVEL_A (failed checks []); scope: no bug downstream of the per-sample counts; per-sample counts / .pat parsing NOT verified |
| TARGET_LIMITATION | False | (a) ceiling 0.781 vs < 0.30 -> False (margin +0.481); (b) reliability >1Mb 0.610 / inter 0.610 vs < 0.50 -> False (margins +0.110 / +0.110); (c) top-band share 0.000 (>= 0.50 needed), enrichment 0.000 (>= 1.5 needed) -> False; (d) max |primary(unrounded) - primary(frozen)| 1.4e-06 vs > 0.02; ordering changed False -> False |
| COMPRESSION_GEOMETRY | True | native -0.0628 (recovers: False); sources recovering with delta-CI > 0: ['oof:centered_ext', 'oof:centered_orig', 'oof:raw_ext', 'oof:raw_orig']; raw sources: raw:raw_binary_cosine -0.085, raw:raw_jaccard -0.076; OOF sources: oof:raw_orig +0.277, oof:centered_orig +0.277, oof:raw_ext +0.277, oof:centered_ext +0.277; supporting PC-removal flag True |
| INFORMATION_ABSENCE | False | requires none of 6 sources >= 0.0353; the OOF sources exceed it (max 0.277) |
| COMPARATOR_ADVANTAGE | False | CpGPT-large: PARTIAL (2/8 fair evaluations won); DeepCpG-DNA: PARTIAL (1/8 fair evaluations won) |

COMPRESSION_GEOMETRY delta (source - native on the same pairs) CIs: oof:raw_orig: [0.320, 0.359]; oof:centered_orig: [0.320, 0.359]; oof:raw_ext: [0.320, 0.359]; oof:centered_ext: [0.320, 0.359]; raw:raw_binary_cosine: [-0.024, -0.021]; raw:raw_jaccard: [-0.014, -0.012].

Supporting PC-removal flags (delta vs native >= 0.03 with CI > 0): centered_cosine True, remove_pc1 True, remove_pc1_2 True, remove_pc1_5 True, remove_pc1_10 True, standardized_cosine True, whitened_cosine True, neg_euclidean True.

COMPARATOR_ADVANTAGE, evaluation by evaluation (win = comparator - candidate CI > 0 and comparator >= its NULL_HI + 0.03):

| Evaluation | CpGPT-large | DeepCpG-DNA |
|---|---|---|
| native_cosine | True | True |
| centered_cosine | False | False |
| remove_pc1 | False | False |
| remove_pc1_10 | True | False |
| whitened_cosine | False | False |
| neg_euclidean | False | False |
| oof_T1_orig | False | False |
| oof_T1_ext | False | False |

**Resulting classes: ['COMPRESSION_GEOMETRY'].** Coexisting classes: none (single class). INFORMATION_ABSENCE and COMPRESSION_GEOMETRY are mutually exclusive by construction. The raw-space evidence does not recover the target; the rule is satisfied through the OOF predicted-profile similarity only; BUG is limited by the Level-A scope.

## 12. Integrity

* Primary fingerprint (registered) `38b9da12700a8fc83dc08f5a01aaa76b61f0d5097bffe395d8eb5225f1f02094` before, `38b9da12700a8fc83dc08f5a01aaa76b61f0d5097bffe395d8eb5225f1f02094` after; unchanged = True.
* Snapshot of ALL files under the frozen tag directory: 263 files before, 263 after; identical = True (added 0, removed 0, changed 0); audit root outside the fingerprinted tree = True.
* Protocol tag `bioval-v2-protocol-freeze-v1` -> `27d8bce1b2ce391cb4ffb0d89eb15660710c2b33`.
| Step | HEAD at run | registration sha256 | code sha256 | freeze modules imported | TCGA referenced | gate checks failing |
|---|---|---|---|---|---|---|
| step1_target_audit | 2eb34d9a7b | 0a6622323fc3 | 1ebf823fe2fd | none | False | none |
| step3_4_semantics_pairs | 305f538275 | ce011169dc9c | 235d08cbf706 | none | False | none |
| step5_baselines | 0d637dcdd6 | ce011169dc9c | 235d08cbf706 | none | False | none |
| step6_raw_vs_svd | 0d637dcdd6 | ce011169dc9c | 235d08cbf706 | none | False | none |
| step7_svd_geometry | 0d637dcdd6 | ce011169dc9c | 235d08cbf706 | none | False | none |
| step8_oof_similarity | 0d637dcdd6 | ce011169dc9c | 235d08cbf706 | none | False | none |
| step9_probe_fairness | 0d637dcdd6 | ce011169dc9c | 235d08cbf706 | none | False | none |
| report_inputs | 2eb34d9a7b | 0a6622323fc3 | 1ebf823fe2fd | none | False | none |

* Amendment 1 (Level A only; Level B disabled, gate refuses `--level-b`), Amendment 2 (float32-precision midpoint rule), Bugfix 1 (report-inputs aggregation): see the registration. Attempt 1 (FAIL) is archived in `step1_target_audit_attempt1_FAIL/`.
* Frozen commands unused (never imported): see the registration section 12; the run manifests record `freeze_modules_imported`. Limit: Level B (raw `.pat` counts and parsing) NOT executed.

## 13. What the results do NOT show

* Linear decodability of the profile from the embedding (ridge R2) is not the same as the embedding's cosine geometry being aligned with cell-type methylation programs; the OOF-profile similarity is a different pairwise object built from a supervised linear map trained on methylation profiles of other chromosomes.
* The Level-A verdict does not verify raw counts or `.pat` parsing; 'BUG absent' is limited to the downstream construction.
* The effect of methylation pretraining of the comparators is not excluded (section 10); the audit cannot attribute the comparators' native-cosine lead to supervision rather than to sequence information.
* The OOF null reuses NULL_HI from the cosine nulls (no refitted nulls); the OOF statistics are on complete-case pairs only (346,503); block bootstrap by the chromosome of `cpg_i` approximates interchromosomal pairs.
* The probe-fairness edge condition remains unresolved for the candidate after the maximum number of extensions (0.190 of selections at the minimum alpha).
* Nothing here designates a new primary or a representation; all statistics are descriptive and exploratory.

