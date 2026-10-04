# External reconstruction confirmation on GSE40279: PROTOCOL (FROZEN-v1 on commit)

**Status: FROZEN-v1 when the orchestrator commits this document together with `configs/external/gse40279_v1_freeze_manifest.json` and the phase-1 tracked files, and BEFORE any run.**
`freeze_state` in the manifest is `draft` and `test_set_authorized` is `false`; the external TEST split stays blocked until a separate, explicit user authorization after the full freeze (section 14).
**No representation was evaluated, no model forward pass or training was run, and no GPU was used on any external or TCGA data before this freeze.** Phase 1 only prepared data, split, manifests, audits and this protocol.
Supersedes (for GSE40279) the DRAFT `docs/EXTERNAL_RECONSTRUCTION_PROTOCOL_PROPOSAL.md`; dataset facts: `docs/EXTERNAL_RECONSTRUCTION_DATASET_AUDIT.md`.
Never touched: `regulatory_histone_dnase_v1` / `configs/frozen/*`, `data/cache/representations/*`, `outputs/encode_atlas_v1/**`, `biological_validation_v2` anything, the TCGA confirmation matrix spec/results, TCGA protocol/test data (no TCGA beta value was read; `outputs/encode_atlas_v1/patients.npz` was read for the number of train rows only).

## 1. Scientific question

Does the TCGA phase-A validation ordering (`docs/REGULATORY_CONFIRMATION_PHASE_A_RESULTS.md`: `regulatory_histone_dnase_v1` MSE@0.50 0.01478 < `cpgpt_large_locus` 0.01667 < `deepcpg_dna_locus` 0.01962, and near-parity with `functional_annotations_pca` 0.01471) hold on an independent methylation cohort? Descriptive confirmation of an ordering; no decision rule, no equivalence claim.

## 2. Cohort and provenance

* GSE40279 (Hannum et al.), whole blood, Illumina 450K (GPL13534), 656 samples = 656 unique subjects (`subject_id` unique; asserted), BeadStudio average beta, 4 sites (UCSD 304, Utah 178, USC 139, Boston 35), 9 plates; age 19-101; sex F 338 / M 318. Not TCGA/TARGET (no id overlap; audit). Out-of-corpus for CpGPT per the provenance already collected (sibling docs; UNVERIFIED locally beyond that). DeepCpG-DNA is scRRBS-trained.
* GSE55763 is `future_replication_candidate` and is NOT used.
* Source files (read-only): `data/processed/GSE40279/{methylation.h5, phenotypes.parquet, cpg_mapping.parquet}` (sha256 in the mapping manifest). Multi-probe loci were already collapsed upstream by nanmean (9 mapped loci carry >1 probe). Beta values are copied untouched (bitwise verified), no renormalisation, no clipping (the runner's `beta_epsilon` 1e-4 clips only inside the logit, identically for all arms).
* NaN policy: none present (asserted: 0 NaN, all values in [0,1], in the written matrix). The runner would skip NaN in panels and drop loci without a finite train value; neither occurs (checked with `compute_leakage_safe_priors` on the train rows: all 408,017 loci usable, global train mean beta 0.4952).
* Exact 0/1 beta values are 1.6% of values (audit); treated identically for all arms.

## 3. Id system and locus universe

* The benchmark universe and all frozen stores use the TCGA/MethylProphet **global `cpg_idx`**; the cohort file uses coordinate-native `chrom*1e9+pos`. The new matrix `data/derived/external_gse40279_v1/methylation.h5` has `/cpg_idx` = global cpg_idx of the exact-(chr,pos)-joined loci in benchmark universe order (`array_cpg_map.parquet` `raw_cpg_row` order restricted to covered loci), `/beta` float32 `[656, 408,017]` chunks (1, 8192), `/sample_name` (GSM ids). Extra traceability datasets (`chrom`, `pos`, `source_cpg_idx`) are ignored by the runner. Stores are not re-keyed (their sha256 must stay).
* Join: exact (chr,pos), no offset, no tolerance (0 matches with pos+1 in the audit). Benchmark 408,399 loci; **mapped 408,017 (chr1-19 388,233; chr20-22 19,784); unmapped 382** (reason for all: no cohort probe at that exact coordinate, i.e. locus absent from the GSE40279 450K array content). The 382 are listed with cpg_idx/chr/pos in the mapping manifest. Probes: 473,034 source probes (472,952 unique-coordinate, 66 duplicate-coordinate, 16 without crosswalk); 408,026 probes sit on the 408,017 mapped loci.
* **Locus universe = benchmark loci ∩ cohort array content = 408,017 loci**, decided from cohort + benchmark alone, identical for all arms. Locus protocol `data/derived/external_gse40279_v1/loci_seen_external_gse40279_v1.npz` (`loci_seen.npz` format: `train_cpg_idx` = all 408,017 sorted, `heldout_cpg_idx` empty, `seed` 17001, `heldout_fraction` 0.0; genome-wide seen-only per CLAUDE.md). No arm may narrow it; a store missing a locus fails (`RepresentationCoverageError`).
* **Zero-embedding policy:** the 846 universe loci (of the 849 benchmark loci) whose `regulatory_histone_dnase_v1` embedding row is all zeros are KEPT for every arm (valid inputs to the learned adapter; same as TCGA). Excluding them is an exploratory sensitivity only, never primary.

## 4. Split

Unit = individual (656). Seed 20260925 (the TCGA patient-split seed). 80/10/10, stratified by sex x age tercile (6 strata), **524 train / 66 validation / 66 test** (n_val = n_test = floor(0.1*656+0.5) = 66). Age terciles are computed on the full cohort (metadata only): cut points q1 = 58, q2 = 71 (numpy linear quantiles 1/3, 2/3); tercile = (age>q1)+(age>q2); tied ages are never separated, so tercile sizes are 228/212/216. Per-stratum quotas by largest remainder (ties by stratum key), then within a stratum subjects sorted by id and permuted with `default_rng([seed, stratum_rank])`; row-order invariant and deterministic. Strata (total: train/val/test): F_T0 109: 87/11/11; F_T1 119: 95/12/12; F_T2 110: 88/11/11; M_T0 119: 95/12/12; M_T1 93: 75/9/9; M_T2 106: 84/11/11. Balance diagnostics (age mean/sd, sex, site, plate per split, age SMD vs train) are in `configs/external/gse40279_v1_split_manifest.json`; site and plate are NOT stratified and never used to re-draw the split (Boston is 29/1/5, a reported imbalance). Single split, no CV.
Files: patient protocol `data/derived/external_gse40279_v1/patients_external_gse40279_v1.npz` (format of `outputs/encode_atlas_v1/patients.npz`: `train, validation, test, sample_names, seed`; accepted unchanged by `load_patient_protocol`: partition + no group leakage), tracked `configs/external/gse40279_v1_split.csv` (subject ids per split) and `configs/external/gse40279_v1_split_manifest.json`.
The test row ids are in the protocol file because the loader requires a partition; the test betas are not used before authorization (section 14: runs use `evaluation.patient_view: validation` + `require_patient_view: validation`; phase 2 runner enforces and audits it).

## 5. Arms (stores frozen, no feature engineering)

| arm | store | dim / dtype | sha256 (= matrix.yaml) | universe coverage | zero rows | non-finite |
| --- | --- | --- | --- | --- | --- | --- |
| regulatory_histone_dnase_v1 (candidate) | `data/cache/representations/regulatory_histone_dnase__global_svd256__discovery_chr1_19.h5` | 256 / float32 | `fa3a816338ce2c2ce5669b4be6f5334a44470e2b149395901168dbe673a9fe1e` | 408,017/408,017 | 846 | 0 |
| functional_annotations_pca | `outputs/encode_atlas_v1/embeddings/full_a18b869b.h5` | 256 / float16 | `3d99fe5ba4fb88991100f848e1d2c48ad19037c72b8821421b2eb677996e187f` | 408,017/408,017 | 0 | 0 |
| cpgpt_large_locus | `data/cache/representations/cpgpt_locus_large.h5` | 512 / float16 | `c08138cd83e08f5015521c2a5d42b88e1834f0670c84906590d9730223d7b71e` | 408,017/408,017 | 0 | 0 |
| deepcpg_dna_locus | `data/cache/representations/deepcpg_dna_locus.h5` | 128 / float16 | `e7280d5ff14d89a247e4cb3560a952250a3f1fa78daf653b9ec02761389b82b7` | 408,017/408,017 | 0 | 0 |

(Full sha256 of every store is recorded in `configs/external/gse40279_v1_freeze_manifest.json` and in `outputs/external_reconstruction_v1/audit/arm_coverage.{json,md}`; the manifest is authoritative.)

## 6. Training regime (identical to TCGA phase A except the declared adaptations in section 13)

Model `MaskedMethylomeReconstructor` (locus_latent 256, token 256, patient 256, hidden 512), AdamW, lr 1e-4, weight decay 1e-4, constant LR (no scheduler), fp16 autocast, batch 8, panel 2048, `panel_repeats` 1, training mask fractions {0.15, 0.30, 0.50, 0.70, 0.90} (one fraction per batch), `beta_epsilon` 1e-4, `early_stopping: false`, seeds 17 / 42 / 97 (`training.seed`; initialisation only), `num_workers` 8 training / 0 evaluation (throughput only, verified neutral in TCGA; the validation loader uses min(2, num_workers) as before), priors from train patients x train loci only (identical for all arms), `heldout_fraction` 0.0.

## 7. EXACT optimizer-update budget

* TCGA phase A: `MaskFractionBatchSampler` yields `ceil(N/8)` batches per epoch and keeps the last partial batch (no `drop_last`); the engine performs one optimizer update per batch. N_train = 7,342 -> 918 updates/epoch x 120 epochs = **110,160 optimizer updates** per run. Verified on all 12 phase-A runs: `experiment.json` `n_train_patients_rows` = 7,342, resolved config `epochs` = 120, `history.json` has 120 epoch records, all commit `c5061666`; 918 x 120 = 110,160 for every run (the engine does not log a step counter; the count is derived from the sampler contract and verified by the epoch count).
* **Frozen external budget: 110,160 optimizer updates per run (same for every arm and seed)**. N_train = 524 -> `ceil(524/8)` = 66 updates/epoch (65 batches of 8 + one batch of 4). 110,160 = 1,669 x 66 + 6, so each run = **1,669 full external epochs (110,154 updates) + one partial epoch of exactly 6 updates = 110,160 updates, 1,670 epochs started** (epoch indices 0..1669).
* Mask/panel determinism: masks are keyed on (mask_seed 17001, epoch, row, fraction) and the batch order/fractions on (seed, epoch) only (`MaskingDataset.__getitem__`, `MaskFractionBatchSampler.__iter__`), so the partial epoch 1669 is exactly the first 6 batches of the sampler sequence of epoch 1669 (unit-tested). Identical across arms.
* Validation / checkpoint schedule: validation MSE at mask fraction 0.50 (fixed validation masks; the validation dataset is never advanced in epoch, as in TCGA) is computed at the end of every full external epoch (cumulative updates 66, 132, ..., 110,154) and once more at the final update 110,160: **1,670 validation points**. `best.pt` = strict minimum of validation MSE@0.50 over these points (ties keep the earliest); `last.pt` = state at update 110,160; the last state is never selected automatically. No stop, extension or change based on curves; no per-arm change.
* Schedule recorded in the freeze manifest (`budget`); the verifier recomputes it.

## 8. Masking and evaluation

Evaluation mask fractions 0.15 / 0.30 / 0.50 / 0.70 / 0.90; ONE `best.pt` per run (selected at 0.50) evaluated at all five fractions on the validation split (test only after authorization). Mask seed **17001** (`evaluation.mask_seed`, also used for training and validation masks, as in TCGA; locus protocol seed 17001). Primary endpoint: MSE at mask 0.50. Secondary: MAE, MAS-PCC (per patient), MAC-PCC (per locus), prior-only MSE / skill vs prior, all fractions (descriptive).
Output layout `split_dirs`: `<run>/evaluation/validation/seen/mask_<f>/{metrics.json,predictions.npz}` and `<run>/evaluation/validation/summary.json`; `evaluation/test/` only after authorization.

## 9. Statistics, contrasts, reading of outcomes

* Paired bootstrap over patients x 1 Mb genomic blocks (`encode_atlas/statistics.py: paired_bootstrap`, crossed bootstrap, 2,000 replicates, bootstrap seed 17), per seed; pairing check (identical patients, loci, targets, observation counts across arms) is a hard precondition. Contrast = comparator minus candidate on MSE (positive = candidate better).
* Contrasts: (1) functional_annotations_pca - candidate, (2) cpgpt_large_locus - candidate, (3) deepcpg_dna_locus - candidate. MSE@0.50 is the single primary endpoint of each contrast; there is no multiplicity family beyond these 3 contrasts, so **no Holm/BH adjustment**; raw 95% CIs and raw bootstrap p are reported, descriptively. (A 4th descriptive contrast deepcpg - cpgpt expresses the CpGPT > DeepCpG part of the ordering; it is not primary and not part of any family.)
* Functional comparison: report absolute delta MSE, relative delta MSE and the paired 95% CI per seed. **No equivalence / non-inferiority margin is defined (no preregistered external justification); no equivalence or parity claim is made.** The previously proposed +/-1.5% margin is withdrawn.
* Neutral pre-stated reading (per seed, per contrast, no aggregation across arms/seeds/contrasts): for each delta state the sign and whether the CI excludes 0: "delta > 0 with CI above 0" (comparator higher MSE than candidate), "delta < 0 with CI below 0" (comparator lower MSE), "CI includes 0". The across-seed summary is the three per-seed estimates, their min/max and the number of seeds with CI excluding 0 (as in phase A). "Same sign/CI pattern as TCGA" is stated factually, not as a pass/fail. Secondary metrics cannot rescue or overturn an MSE statement. Exploratory (labelled): all-fractions, chr1-19 vs chr20-22 strata, zero-embedding exclusion, sex/age strata, Experiment B.

## 10. Experiment A (primary) and B (secondary; PREPARED, NOT EXECUTED)

**A. `external_within_cohort_reconstruction`**: train / validation / test entirely GSE40279; 4 arms x 3 seeds = 12 runs; this is the confirmation primary.

**B. TCGA -> GSE40279 transfer (secondary, exploratory, reported separately, never primary, never pooled with A)**: no external training of any kind. Uses the 12 phase-A `best.pt` (4 arms x seeds 17/42/97; sha256 in the freeze manifest, from `outputs/regulatory_confirmation_v1/phase_A_status.json`), loaded into `MaskedMethylomeReconstructor(raw_locus_dim=store.dim, 256,256,256,512)`, evaluated on the same external patients/loci/masks (fractions as in section 8, mask seed 17001) on the external universe (408,017 loci, all inside the TCGA universe).
* **B-strict**: original TCGA prior (`prior_logit.npy` of the TCGA phase-A run, re-indexed to external matrix columns by global cpg_idx), no external data in the prior or anywhere.
* **B-recalibrated**: same frozen checkpoint, decoder and adapter unchanged, prior recomputed with `compute_leakage_safe_priors(external matrix, GSE40279 TRAIN rows (524), all 408,017 universe columns, epsilon 1e-4)` using ONLY the 524 train subjects (never validation/test); one prior shared by all arms.
* Both variants pre-declared, reported side by side with no preferred one; B-recalibrated uses external methylation in the prior, so it is not "no external data". Execution is BLOCKED until this protocol is frozen (committed); validation split first, test only per section 14. B needs a new evaluation script (phase 2).

## 11. Pre-run gate (all must be green before any launch; phase 2 audit)

mapping manifest frozen and `scripts/verify_external_freeze.py` exit 0; split frozen (524/66/66, protocol sha); step budget frozen (110,160); every store sha256 = matrix.yaml and 100% universe coverage (`arm_coverage.json` all_green); locus protocol identical for all arms; per-arm resolved configs identical except `representation.*`, `experiment.name`, `training.seed`; `training.epochs`/`max_updates` equal to the frozen values; `early_stopping: false`; `evaluation.require_patient_view: validation`; `test_set_authorized: false` and no `evaluation/test` directory; TCGA matrix and `configs/frozen/*` unchanged.

## 12. Manifests to record per run

Run dir content-addressed as usual plus: sha256 of matrix h5, mapping manifest, locus protocol, patient protocol, split CSV, prior (`prior_logit.npy`, equal across arms), store, `best.pt`, resolved config hash, git commit, universe size (408,017), dropped loci (0 expected), seed, wall-clock, validation curve (1,670 points with updates_done), best update, `confirmation_status.json`. Frozen manifest: `configs/external/gse40279_v1_freeze_manifest.json`.

## 13. Deviations from the TCGA protocol (all declared before any run)

| # | Deviation | Reason |
| --- | --- | --- |
| D1 | Cohort GSE40279 (whole blood, 450K), matrix re-keyed to the global cpg_idx (stores untouched) | independence from TCGA |
| D2 | Universe 408,017 instead of 408,399 (cohort-defined, same for all arms) | array content |
| D3 | N 524/66/66 individuals, stratified sex x age tercile, single split | small N |
| D4 | Budget is a fixed optimizer-update count 110,160 (1,669 epochs of 66 updates + 6) instead of 120 epochs | matches the TCGA number of updates |
| D5 | Validation cadence: every full external epoch + final update (1,670 points) | follows from D4 |
| D6 | Checkpoint pruning: keep `best.pt`, `last.pt` (final state) and full history; do not write per-epoch `epoch_XXXX.pt` | disk (1,670 x ~14 MB = ~23 GB/run) |
| D7 | Engine opt-in keys (default off = unchanged behaviour, regression-tested in phase 2): `training.max_updates` (stop mid-epoch, then validate+save) and `training.save_epoch_checkpoints: false` | needed to realise D4-D6 without changing the default path |
| D8 | External matrix spec and test-authorization parametrisation (the TCGA guards are hard-wired to the TCGA matrix) | TCGA matrix is frozen |
| D9 | No equivalence margin; no multiplicity adjustment beyond 3 primary contrasts | user decision |
| D10 | Experiment B with two pre-declared prior modes | secondary analysis |
Unchanged: model, optimizer, lr, weight decay, constant LR, fp16, batch 8, panel 2048, mask fractions, selection at 0.50, mask seed 17001, seeds, bootstrap (2,000 replicates, seed 17), early stopping off, `num_workers` 8/0.

## 14. Test protection and state machine

`freeze_state: draft | final`; `test_set_authorized: false` in the freeze manifest and any future external matrix spec. The TEST split is evaluated only after `freeze_state: final`, an all-green audit and an explicit user authorization; phase-V runs use `evaluation.patient_view: validation` with `require_patient_view: validation`; the layout keeps `evaluation/validation/` and `evaluation/test/` separate and a split directory that already holds results is never reused (`allow_overwrite_split_dir: false`). Nobody inspects test betas, predictions or metrics before then.

## 15. Bugfix versus methodology policy

A bug fix (code that does not match this document) is allowed before any result is seen, with a recorded commit, and must not change a frozen value. Any change of a methodological choice (cohort, universe, split, budget, cadence, seeds, mask seed, contrasts, reading rules) after a run has started or any curve/metric has been seen requires a new versioned protocol (v2), labelled as such, with both reported. Results of the discarded variants are never dropped.

## 16. Risks and caveats (carried from the audit)

Whole-blood cell-composition and age/sex effects dominate reconstructable variance and shift the MSE scale vs TCGA; 4 sites / 9 plates are unmodelled batch effects (reported, not corrected); BeadStudio average beta with 1.6% exact 0/1, TCGA normalization state unrecorded (UNVERIFIED); probe-design bias shared by all arms (450K-selected loci vs ENCODE track coverage); fit-scope mismatch (candidate SVD fit on chr1-19, functional on all loci, CpGPT/DeepCpG external-pretrained): chr1-19 vs chr20-22 strata are exploratory; adapter widths 128/256/256/512 may change convergence speed (the fixed update budget does not remove this); small N (66 validation / 66 test patients, single split, no CV): CIs wide, the functional contrast may be inconclusive; relatedness cannot be excluded from local metadata (UNVERIFIED); CpGPT pre-training overlap documented out-of-corpus only from a local summary; B inherits prior shift (tumour -> blood) and cannot separate representation quality from decoder tolerance. Boston site (35 subjects) is 29/1/5 across splits.
