# biological_validation_v2: FINAL REPORT (formal closure)

**STATE: `biological_validation_v2` is CLOSED on 2026-10-04.** This document is documentation only. It recomputes nothing, creates no freeze artifact, and changes no result, protocol, registration or representation. All numbers are copied unchanged from `docs/BIOLOGICAL_VALIDATION_V2_RESULTS.md` and `docs/BIOLOGICAL_VALIDATION_V2_SENSITIVITY_AND_EXPLORATORY_RESULTS.md` (themselves copied from stored outputs). Delta convention everywhere: candidate (`regulatory_histone_dnase_v1`, "Regulatory") MINUS comparator, higher-is-better.

| Item | Value |
|---|---|
| Protocol freeze tag / commit | `bioval-v2-protocol-freeze-v1` -> `27d8bce` (`27d8bce1b2ce391cb4ffb0d89eb15660710c2b33`) |
| Launch registration commit | `4ffa2cd` (`docs/BIOLOGICAL_VALIDATION_V2_LAUNCH_REGISTRATION.md`) |
| Launch code commit | `f0b00ac` |
| Primary results commit | `3e0c115` |
| Follow-up registration commit | `f5036eb` (`docs/BIOLOGICAL_VALIDATION_V2_SENSITIVITY_AND_EXPLORATORY_REGISTRATION.md`) |
| Follow-up code commit | `7d2153c` |
| Follow-up results commit | `67de641` |
| Closure commit | this commit (hash assigned when it is made) |
| Frozen primary results fingerprint | `38b9da12700a8fc83dc08f5a01aaa76b61f0d5097bffe395d8eb5225f1f02094` (`bioval_v2_followup.gate.frozen_results_fingerprint`) |
| Candidate | `regulatory_histone_dnase_v1`, unchanged (frozen as candidate in commit `74cbd75`) |
| Test set | TCGA test set never used (`tcga_test_set_referenced=false` in all 4 run manifests) |
| Seed / bootstrap | 17 / B=1000, 22 chromosome blocks, one shared draw table (sha256 `c403131d04be83df632a413b8949b54d0d5d90e8e5307da707104e575d644a75`) |

The document has five strictly separated sections with different evidential status: (1) frozen primary, (2) preregistered secondary/sensitivity, (3) exploratory post hoc, (4) interpretation, (5) caveats, closure status, provenance, open items.

---

# SECTION 1 - FROZEN PRIMARY RESULTS (registered before any embedding statistic; unchanged)

> BANNER: PRIMARY, FROZEN. Nothing in sections 2-4 amends, replaces, re-weights or softens this section. The Loyfer primary is UNFAVOURABLE to the candidate and stays so.

## 1.1 Four primary endpoints x four arms (value [95% chromosome-block bootstrap CI])

| Primary endpoint (metric) | Regulatory (histone+DNase) | Functional PCA | CpGPT-large | DeepCpG-DNA |
|---|---|---|---|---|
| Loyfer WGBS profile similarity (weighted Spearman, >1 Mb + interchromosomal) | -0.0628 [-0.0724, -0.0534] | -0.0381 [-0.0459, -0.0301] | 0.1065 [0.1010, 0.1121] | 0.1155 [0.1079, 0.1220] |
| H1 Micro-C intra 10 kb (AUROC) | 0.5685 [0.5598, 0.5780] | 0.5676 [0.5598, 0.5766] | 0.5314 [0.5272, 0.5348] | 0.5025 [0.4991, 0.5051] |
| FANTOM5 enhancer membership (blocked-CV linear-probe AUROC) | 0.8330 [0.8255, 0.8396] | 0.8497 [0.8418, 0.8576] | 0.7008 [0.6929, 0.7107] | 0.6229 [0.6107, 0.6348] |
| Replication timing consensus (out-of-fold ridge Spearman) | 0.7031 [0.6837, 0.7169] | 0.7180 [0.6994, 0.7311] | 0.4864 [0.4562, 0.5129] | 0.3594 [0.3196, 0.4022] |

**Loyfer primary, stated without softening:** the candidate is the lowest of the four arms (-0.063) and is worse than Functional PCA (-0.038), CpGPT-large (0.107) and DeepCpG-DNA (0.116). The candidate is worse on the Loyfer long-range primary than both sequence models and than the functional baseline, with all three Holm p = 0.008 (the floor).

## 1.2 Paired contrasts (candidate minus comparator), Holm over the 4 primaries within each contrast (alpha 0.05)

Raw p floor 0.002 (k=0 of B=1000); Holm floor 0.008. "Supports regulatory" requires Holm p < 0.05 AND delta > 0.

| Comparator | Endpoint | delta (reg - comp) | 95% CI | raw p | Holm p | Supports regulatory |
|---|---|---|---|---|---|---|
| Functional PCA | Loyfer WGBS profile similarity | -0.0247 | [-0.0284, -0.0213] | 0.002 | 0.008 | no |
| Functional PCA | H1 Micro-C intra 10 kb | +0.0008 | [-0.0003, +0.0021] | 0.160 | 0.160 | no (no detectable difference) |
| Functional PCA | FANTOM5 enhancer membership | -0.0167 | [-0.0201, -0.0130] | 0.002 | 0.008 | no |
| Functional PCA | Replication timing (consensus z) | -0.0149 | [-0.0176, -0.0119] | 0.002 | 0.008 | no |
| CpGPT-large | Loyfer WGBS profile similarity | -0.1693 | [-0.1810, -0.1578] | 0.002 | 0.008 | no |
| CpGPT-large | H1 Micro-C intra 10 kb | +0.0370 | [+0.0281, +0.0473] | 0.002 | 0.008 | yes |
| CpGPT-large | FANTOM5 enhancer membership | +0.1322 | [+0.1230, +0.1411] | 0.002 | 0.008 | yes |
| CpGPT-large | Replication timing (consensus z) | +0.2167 | [+0.1964, +0.2343] | 0.002 | 0.008 | yes |
| DeepCpG-DNA | Loyfer WGBS profile similarity | -0.1782 | [-0.1929, -0.1627] | 0.002 | 0.008 | no |
| DeepCpG-DNA | H1 Micro-C intra 10 kb | +0.0660 | [+0.0554, +0.0769] | 0.002 | 0.008 | yes |
| DeepCpG-DNA | FANTOM5 enhancer membership | +0.2101 | [+0.1976, +0.2232] | 0.002 | 0.008 | yes |
| DeepCpG-DNA | Replication timing (consensus z) | +0.3438 | [+0.3076, +0.3758] | 0.002 | 0.008 | yes |

Registered pattern reading (from the primary results doc, no equivalence margin exists): overall pattern **D (endpoint-specific / mixed)**. The candidate is not better than the legacy functional baseline on any primary endpoint (only "no detectable difference" on Micro-C AUROC).

---

# SECTION 2 - PREREGISTERED SECONDARY / SENSITIVITY RESULTS (descriptive; raw p only; no Holm; no claim)

> BANNER: PREREGISTERED SENSITIVITIES, DESCRIPTIVE. Variants of the same experiment (same arms, folds, draw table, seed), not independent replications. No Holm family; verdicts are the registered mechanical sign rule.

## 2.1 Summary over the 22 protocol rows

**CONFIRMS 12, QUALITATIVELY_CHANGES 2, INCONCLUSIVE 1, verdict not applicable 2, not executed 5** (total 22). Of the 22 rows, 11 were done at the primary launch, 6 protocol rows (12 items; rows 4, 5, 9, 14, 16, 17) were run in the follow-up (all 12 items CONFIRMS, arm ordering identical to the primary), and 5 were not executed. The three non-CONFIRMS rows classified retrospectively and mechanically are rows 2 (Loyfer per distance stratum; <1 kb stratum), 6 (marker kNN; INCONCLUSIVE) and 8 (H1 per distance bin; 1-10 Mb bin vs Functional, reference delta +0.0008). The full 22-row table is in Part A of `docs/BIOLOGICAL_VALIDATION_V2_SENSITIVITY_AND_EXPLORATORY_RESULTS.md`; it is not duplicated here.

## 2.2 The 5 sensitivities not executed

| Row | Protocol-specified analysis | Classification | Blocker |
|---|---|---|---|
| 10 | 3D: K per bin pair | `NOT_EXECUTED_REQUIRES_NEW_PREREGISTRATION_OR_ARTIFACTS` | D-A3: needs `freeze_4dn_pairs` re-projection with K != 20 (new freeze artifact); protocol names no alternative K; permission not given. |
| 15 | FANTOM5: additional matching on chromatin covariates (post hoc) | `NOT_EXECUTED_REQUIRES_NEW_PREREGISTRATION_OR_ARTIFACTS` | D-A4: needs a new matching (`freeze_fantom5_matching`) with unspecified chromatin covariates; permission not given. |
| 11 | 3D: distance-only / covariate baseline comparison | `NOT_EXECUTED_REQUIRES_NEW_PREREGISTRATION_OR_ARTIFACTS` | D-A5: score, model and covariate set of the baseline are undefined in the protocol. |
| 20 | RT: partial correlation / residualisation for GC, CpG-island density, gene/TSS density | `NOT_EXECUTED_REQUIRES_NEW_PREREGISTRATION_OR_ARTIFACTS` | D-A6: residualisation model, residualised variable, fit scope and "gene density" (only `tss_dist` exists) are undefined. |
| 21 | RT: normal vs tumour lines only | `NOT_EXECUTED_REQUIRES_NEW_PREREGISTRATION_OR_ARTIFACTS` | D-A7: normal/tumour assignment of the 15 UW lines is not frozen; consensus would need refits on subsets. |

**No new freeze artifact was created, and none will be created in this closure.** No freeze/prepare/build/run_evaluation/run_followup command was run in this closure.

## 2.3 Orchestrator decisions retained for review

- **D-A1** (Loyfer `mask_lenient` and single-sample-group exclusion, derived in memory from the frozen files; the kernel reproduced the frozen `loyfer_pearson` / `n_shared_groups` of all 386,268 pairs, max abs diff 8.9e-16) and **D-A2** (FANTOM5 N = 1/3/10/20 library thresholds as a subset of the frozen pairs with their frozen controls, no re-matching) were interpreted by the orchestrator as read-only-from-frozen. **The flag for user review is retained.** The interpretation was not explicitly approved by the user; this closure does not claim approval.

---

# SECTION 3 - EXPLORATORY POST-HOC LOYFER DIAGNOSTICS (`exploratory_posthoc_loyfer`)

> BANNER: EXPLORATORY POST HOC. Conceived AFTER the primary results were seen. It does NOT replace nor reinterpret the frozen primary (Section 1), designates no new primary, enters no Holm family, adds no data, creates no representation, uses no TCGA data. Single seed (17), 22 chromosome blocks, raw p only.

## 3.1 Variance-strata ranking (pooled primary-style statistic, cosine; 384,675 pairs after zero-locus exclusion)

| Cell | Regulatory | Functional PCA | CpGPT-large | DeepCpG-DNA | Ordering (best first) | Candidate rank |
|---|---|---|---|---|---|---|
| all | -0.063 | -0.038 | 0.106 | 0.115 | DeepCpG > CpGPT > Functional > Regulatory | 4 |
| V_Q1 | -0.016 | -0.015 | -0.010 | 0.001 | DeepCpG > CpGPT > Functional > Regulatory | 4 |
| V_Q2 | -0.051 | -0.035 | 0.083 | 0.094 | DeepCpG > CpGPT > Functional > Regulatory | 4 |
| V_Q3 | -0.073 | -0.025 | 0.160 | 0.146 | CpGPT > DeepCpG > Functional > Regulatory | 4 |
| V_Q4 | 0.006 | 0.026 | 0.130 | 0.067 | CpGPT > DeepCpG > Functional > Regulatory | 4 |
| V_top10pct | 0.068 | 0.078 | 0.165 | 0.041 | CpGPT > Functional > Regulatory > DeepCpG | 3 |

The candidate stays numerically below CpGPT-large in every variance cell; the sequence-model advantage on the pooled long-range statistic is not confined to low-variance pairs. DeepCpG-DNA's advantage shrinks in V_Q4 and reverses in V_top10pct (candidate minus DeepCpG +0.026 [+0.007, +0.048]). The ranking depends on the statistic (pooled long-range vs all-pairs unweighted).

## 3.2 Geometry diagnostics (pooled statistic, all pairs; within-arm gain vs cosine in parentheses)

| Geometry | Regulatory | Functional PCA | CpGPT-large | DeepCpG-DNA |
|---|---|---|---|---|
| cosine (primary) | -0.063 | -0.038 | 0.106 | 0.115 |
| centred cosine | 0.174 (+0.237) | 0.180 (+0.218) | 0.095 (-0.012) | 0.110 (-0.006) |
| PC1-removed cosine | 0.086 (+0.148) | 0.093 (+0.132) | 0.072 (-0.034) | 0.075 (-0.040) |
| negative Euclidean | 0.202 (+0.265) | 0.209 (+0.247) | 0.076 (-0.030) | 0.118 (+0.002) |

Within-arm gains are +0.15 to +0.27 for the two ENCODE-feature arms and none (registered criterion: gain >= 0.03 with CI > 0) for CpGPT-large or DeepCpG-DNA. Registered GO outcome: CLOSES (candidate not significantly below either sequence arm; here above both: vs CpGPT-large +0.079 / +0.013 / +0.126 and vs DeepCpG-DNA +0.064 / +0.010 / +0.084 for centred / PC1-removed / Euclidean). In V_top10pct the candidate remains below CpGPT-large under cosine, centred cosine and Euclidean.

## 3.3 Chromosome-blocked multi-output ridge of the 39-group WGBS profile

Global out-of-fold R2, T2 centred profile, variable CpGs (192,339 CpGs): **Regulatory 0.279 [0.273, 0.286]; Functional PCA 0.278 [0.271, 0.286]; CpGPT-large 0.031 [0.030, 0.033]; DeepCpG-DNA 0.009 [0.008, 0.010]** (candidate minus Functional +0.001; candidate minus CpGPT +0.248 [+0.241, +0.255]; candidate minus DeepCpG +0.270 [+0.263, +0.278]). Regulatory and Functional exceed both sequence arms in all six target/subset combinations (T1/T2/T3 x all/variable). Per-cell-group R2 range (T2, variable): Regulatory 0.094 to 0.465 (median 0.267); Functional 0.095 to 0.471 (median 0.264); CpGPT-large 0.003 to 0.122 (median 0.019); DeepCpG-DNA -0.006 to 0.052 (median 0.008). The candidate is significantly above CpGPT-large and DeepCpG-DNA in 39/39 groups and above Functional in 22/39.

## 3.4 Registered A / B / C adjudication (`interpretation_ABC.json`; RO = HIGHER, GO = CLOSES, FO = NOT_BELOW, PO = DIRECT_METHYLATION_SUPERVISION; three readings agree)

- **A (representation lacks long-range methylation-program information): not supported.**
- **B (information present but native cosine geometry does not expose it): supported (strong)**, as the mechanical output of the registered table, not a mechanistic demonstration.
- **C (result materially affected by comparator methylation pretraining): not supported by ridge evidence, but not excluded** (both sequence arms have direct methylation supervision; the design cannot separate the explanations).

## 3.5 Cautions specific to this section

- Linear decodability of cell-type methylation profiles from cell-type-specific chromatin tracks (histone, DNase) is plausible; it is not the primary construct (pair-similarity rank agreement at >1 Mb / interchromosomal distance) and does not show that the embedding "captures long-range methylation programs".
- It does not show that sequence embeddings lack methylation information (nonlinear or other-geometry encoding is possible).
- Ridge alpha for CpGPT-large is at the grid top (1e4) in all five folds for T2 (and at the bottom in some folds for T1/T3); its R2 may be a slight underestimate. The grid was not extended (methodology change).
- The alternative geometries were chosen after the primary result; none is a new primary; centring and PC1 removal are transformations a user may or may not apply.
- The same ridge protocol and alpha grid for all arms may disadvantage the 128-D and 512-D embeddings.
- There is no claim that the candidate is better on Loyfer.

---

# SECTION 4 - INTERPRETATIVE SYNTHESIS (INTERPRETATION, NOT A RESULT)

> BANNER: INTERPRETATION. The statements below are a reading of Sections 1-3, not new measurements, and carry no statistical status of their own.

1. Histone+DNase outperforms CpGPT and DeepCpG on Micro-C, FANTOM5 and replication timing.
2. It does not outperform Functional PCA on the primary endpoints: on Micro-C there is no detectable difference; on FANTOM5 and replication timing Functional PCA is modestly higher; on Loyfer Functional PCA is higher.
3. On the Loyfer long-range primary, CpGPT and DeepCpG are better than Histone+DNase.
4. Nevertheless, a chromosome-blocked linear probe of the Loyfer profile shows strong decodability from the regulatory representation (R2 ~ 0.279) versus CpGPT (~ 0.031) and DeepCpG (~ 0.009).
5. Alternative post hoc geometries strongly improve Histone+DNase, indicating that information content and native cosine geometry do not coincide.
6. These exploratory results neither replace nor reinterpret the frozen primary.

**What the whole picture does NOT license:** any claim of superiority over sequence models on methylation programs, any claim that the candidate is better on Loyfer, or any claim of generality beyond these partially related endpoints (FANTOM5, Micro-C and compartments are partially related to the histone/DNase inputs), one seed, one set of folds and 22-block CIs.

---

# SECTION 5 - CAVEATS, CLOSURE STATUS, PROVENANCE, OPEN ITEMS

## 5.1 Complete caveat list (merged and deduplicated from the two results docs)

1. **Relatedness of endpoints to inputs.** Candidate inputs are 2,492 ENCODE histone ChIP + DNase tracks (no TF/CTCF). Loyfer, RT and H1 inter are independent; H1/HFFc6 intra contacts, compartments, FANTOM5 membership/activity are partially related (not independent validation); PMD is independent of inputs but methylation-derived.
2. **Functional baseline has more inputs** (genomic-context columns, cCRE, TSS distance, 4,165 binary tracks incl. TF/CTCF): candidate vs functional compares different input sets, not only different compressions.
3. **Fit-scope mismatch.** Regulatory SVD fit on chr1-19 (388,599 loci); chr20-22 transformed only. Other arms were fit/pretrained without that restriction.
4. **Dimension/dtype differences** (256 f32 / 256 f16 / 512 f16 / 128 f16); no dimension matching.
5. **Zero-row exclusion.** 849 candidate loci have all-zero rows; cosine endpoints drop every pair touching them for ALL arms; linear-probe endpoints keep them as valid features (plausibly easy negatives for the candidate); no cosine=0 sensitivity.
6. **Methylation pretraining / provenance of comparators.** CpGPT-large and DeepCpG-DNA have direct methylation supervision (provenance audit PO = DIRECT_METHYLATION_SUPERVISION); overlap with Loyfer is unverified directly, indirect overlap plausible; the contrast cannot attribute an advantage to supervision rather than sequence information.
7. **Weighted-Spearman definition** operationalised at launch (not in frozen code); mean-of-two-strata alternative agrees to 4 decimals.
8. **Probe grids.** Ridge alpha 1e-2..1e4 (sklearn convention), logistic C 1e-4..10, nested leave-one-fold-out inner CV; grid edges not extended; PMD uses ridge on the 0/1 label.
9. **Single seed (17); probe predictions not refitted in the bootstrap**: CIs reflect chromosome resampling only.
10. **Block bootstrap**: 22 blocks, B=1000, coarse CIs, raw p floor 0.002, Holm floor 0.008; values at the floor are not measured probabilities; inter-chromosomal pairs are assigned to the block of cpg_i.
11. **Frozen `knn_enrichment` bug workaround** (p-value from a new non-frozen routine, descriptive, saturates at floor); marker hg38 provenance WEAK; low power; very large enrichment values.
12. **Sensitivities not executed** (Section 2.2): in particular RT covariate residualisation, so the RT result may partly reflect GC / gene-density content that histone/DNase signal encodes.
13. **Intra-chromosomal distance confounds**: Loyfer short strata mostly measure proximity; Micro-C controls are distance-matched but not exactly (median difference ~2.1 kb); AUROC pools heterogeneous distances.
14. **No multiplicity control across contrasts** (3 Holm families of 4) nor across secondary/sensitivity/exploratory statistics; Holm is not a selection rule.
15. **Effect size vs significance**; no equivalence margin is defined in the protocol.
16. **Other data caveats**: single cell line per 3D system, different Micro-C depth, RT Rep1 only with hg19->hg38 liftover, FANTOM5 permissive set (median 1 library per group, 90.8% zeros), Loyfer sample imbalance (205 samples = 168 donors), hypomethylated-only markers, H1 inter weakly balanced.
17. **Sensitivities are variants of the same experiment**, and the sign rule is weak when the primary delta is near zero (Micro-C vs Functional).
18. **Post hoc status of Section 3**: geometries chosen after the primary; ridge alpha at the grid edge for CpGPT-large; decodability is not the primary construct (Section 3.5).
19. **D-A1/D-A2** interpreted by the orchestrator, not explicitly approved by the user (Section 2.3). D-A2 covariate balance of subsets was not re-verified.

## 5.2 Closure status of artifacts

| Artifact | Status |
|---|---|
| `docs/BIOLOGICAL_VALIDATION_V2.md` (protocol), `docs/bioval_v2_prep`, `configs/biological_validation_v2`, `data/derived/bioval_v2` | FROZEN. Never modify. |
| `docs/BIOLOGICAL_VALIDATION_V2_LAUNCH_REGISTRATION.md`, `docs/BIOLOGICAL_VALIDATION_V2_SENSITIVITY_AND_EXPLORATORY_REGISTRATION.md` | FROZEN registrations. Never modify. |
| `outputs/biological_validation_v2/bioval-v2-protocol-freeze-v1/<arm>/endpoints`, `contrasts`, `report` (frozen primary results; fingerprint `38b9da12...`) | FROZEN. Never add, remove or modify files in this tree. |
| `.../sensitivity/`, `.../exploratory_posthoc_loyfer/` (including `figures_followup/`) | Closed outputs of the follow-up; excluded from the fingerprint by design; not to be modified either. |
| `src/cpg_repr_benchmark/bioval_v2_launch`, `bioval_v2_followup`, `scripts/bioval_v2/*` (launch/follow-up code) | Closed. Do not edit; only new files may be added. |
| Candidate `regulatory_histone_dnase_v1` and its store | FROZEN, unchanged. |
| Results docs (`..._RESULTS.md`, `..._SENSITIVITY_AND_EXPLORATORY_RESULTS.md`) | Closed; only a one-line pointer to this report was added. They are not hashed by the gate or fingerprint. |
| Guard test `tests/test_bioval_v2_primary_fingerprint.py`, CLI `scripts/bioval_v2/check_primary_fingerprint.py` | Added at closure; read-only. |

## 5.3 Provenance

Commits and tags: see the table at the top. The follow-up figures live under `outputs/biological_validation_v2/bioval-v2-protocol-freeze-v1/exploratory_posthoc_loyfer/figures_followup/` precisely so that the primary fingerprint is unchanged.

Representation store sha256 (registered = actual, from each `run_manifest.json`):

| Arm | store sha256 |
|---|---|
| `regulatory_histone_dnase_v1` | `fa3a816338ce2c2ce5669b4be6f5334a44470e2b149395901168dbe673a9fe1e` |
| `functional_annotations_pca` | `3d99fe5ba4fb88991100f848e1d2c48ad19037c72b8821421b2eb677996e187f` |
| `cpgpt_large_locus` | `c08138cd83e08f5015521c2a5d42b88e1834f0670c84906590d9730223d7b71e` |
| `deepcpg_dna_locus` | `e7280d5ff14d89a247e4cb3560a952250a3f1fa78daf653b9ec02761389b82b7` |

Other manifest hashes: registration doc sha256 `1fdb82209bc049ecbc9920708659adb095f8566f7d3d6ff7f2c0184e53bd270e`; frozen manifest checksums `a3b894cc3e8e635a369ee77aaa39eddcf59535d496d532d5272a11b7486ceac2`; bootstrap draw table `c403131d...5d644a75` (full value at the top).

Read-only verification: `python scripts/bioval_v2/check_primary_fingerprint.py` (exit 1 on mismatch) and `pytest tests/test_bioval_v2_primary_fingerprint.py`.

## 5.4 What remains open

- The 5 sensitivities of Section 2.2 (rows 10, 15, 11, 20, 21; D-A3 to D-A7) remain unexecuted.
- Any follow-up (including those 5, any further Loyfer analysis, any new representation or geometry as a claimed result) requires a NEW preregistration and, where it needs a new frozen artifact, its own freeze; it must not modify anything listed as frozen in Section 5.2.
- The user's review of D-A1/D-A2 is still open (flag retained).
