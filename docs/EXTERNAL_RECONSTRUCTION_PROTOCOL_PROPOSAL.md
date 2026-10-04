# External reconstruction confirmation: PROTOCOL PROPOSAL (DRAFT, NOT FROZEN)

**Status: DRAFT proposal of 2026-10-04. Nothing here is frozen or authorized until the user approves it in writing. No run, training, GPU use or evaluation of any
representation on any external data has happened.** Companion: `docs/EXTERNAL_RECONSTRUCTION_DATASET_AUDIT.md` (dataset audit, prespecified selection criteria, numbers).
Frozen things that this proposal never touches: `regulatory_histone_dnase_v1` and `configs/frozen/*`, `data/cache/representations/regulatory_*`, the TCGA confirmation matrix
(`configs/experiments/regulatory_confirmation_matrix/*`), `docs/REGULATORY_CONFIRMATION_PROTOCOL.md` (AMENDMENT 2), `outputs/regulatory_confirmation_v1/*`, `biological_validation_v2` anything.
The TCGA TEST split remains unread; this external confirmation is independent of it.

Question: does the TCGA validation ordering (Phase A, `docs/REGULATORY_CONFIRMATION_PHASE_A_RESULTS.md`: `regulatory_histone_dnase_v1` MSE@0.50 0.01478 < CpGPT-large 0.01667 < DeepCpG-DNA 0.01962; `functional_annotations_pca` 0.01471,
i.e. 0.44-0.65% lower MSE than the candidate in 3/3 seeds with CIs excluding 0 = "near parity") hold on a methylation cohort independent of TCGA? Descriptive confirmation, no decision rule.

## 1. Confirmation types

### A. External within-cohort reconstruction (proposed primary)

Train/validation/test split fully inside the external cohort; every arm uses the same reconstructor, hyperparameters, patients, loci, masks, seeds. It tests the locus representation under a
decoder trained on the cohort's own methylation distribution, so decoder distribution shift (TCGA tumours vs blood) is not a confound. Costs: smaller N than TCGA, the decoder learns the cohort (this is a
re-confirmation of the benchmark, not a transfer claim).

Feasibility with the current code (checked by reading, nothing run):

| Component | Status |
| --- | --- |
| Model / training engine / masking dataset / evaluation / split_dirs layout | Work unchanged for any matrix with `/beta [n_samples, n_cpg]`, `/cpg_idx`, `/sample_name` (`data/methylation.py:read_axis`); panel sampling skips NaN (`MaskingDataset._finite_sample`) |
| Dataset registration | There is no central registry: a `dataset:` block in each run config (`name, methylation_h5, cpg_registry, patient_protocol, patient_split_seed`); `configs/datasets/*.yaml` are downstream-task descriptors, not read by the masking runner |
| **Matrix id system** | **Needs new data-prep code.** Stores are keyed by the TCGA global `cpg_idx`; external h5 files use `chrom*1e9+pos`. `HDF5RepresentationStore` matches by id and the runner raises `RepresentationCoverageError` if a locus is missing. Required: a new external matrix whose `/cpg_idx` is the TCGA global idx of the (chr,pos)-joined universe loci, restricted to universe columns in universe order. (`scripts/data/rekey_representation_h5.py` re-keys *stores*; that must not be used here because it would create new store bytes and break the frozen sha256 checks.) |
| GSE55763 preparer | **New**: the file has interleaved beta/"Detection Pval" columns (like `prepare_gse147221.py`) and 473,864 probes; crosswalk `data/processed/ComputAgeBench/illumina_probe_grch38.parquet` (probe_id,chr,pos) exists and maps 473,833 probes |
| Patient protocol | **New small script**: `load_patient_protocol` only validates a partition and TCGA-style `patient_id()` groups (identity for GSM ids), so it cannot enforce replicate grouping or stratification; a generator that writes the same npz format (`train/validation/test/sample_names/seed`) with the forced-train rule is needed. The loader then accepts it unchanged |
| Locus protocol | Automatic: `experiment.locus_split.heldout_fraction: 0.0` writes all candidate columns to `train_columns` (no holdout, as in TCGA). Pre-create it once before launching concurrent runs (avoid a creation race) |
| Priors | `compute_leakage_safe_priors` already fits train patients x train loci only; loci with no finite train value are dropped identically for all arms (depends on the matrix only) |
| Test protection | **New (opt-in) code**: `eval_layout.authorize_test_split` and `confirmation_matrix.py` are hard-wired to the TCGA matrix (`MATRIX_PATH`, `FROZEN_PROTOCOL.max_epochs = 120`, TCGA split paths). The external study needs its own matrix spec and a parametrised audit/gate (opt-in key, default behaviour unchanged) or a forked module; the TCGA matrix must not be edited |
| Checkpoint disk | The engine writes `epoch_XXXX.pt` every epoch (~14 MB); under the budget rule of section 6 this needs an opt-in `training.save_epoch_checkpoints: false` (default true = unchanged) |

### B. TCGA-to-external transfer (proposed secondary / exploratory)

The 12 Phase-A decoders `outputs/regulatory_confirmation_v1/.../checkpoints/best.pt` (4 arms x seeds 17/42/97; arm-specific adapters: raw dim 256/256/512/128) are evaluated on the external cohort with **no external
training**. Each arm's own TCGA-trained decoder is the right one (the `locus_adapter` input width is arm-specific, the rest of the model is shared architecture).
Technical feasibility: the checkpoint is a plain `{"model": state_dict, ...}` loaded into `MaskedMethylomeReconstructor(raw_locus_dim=store.dim, 256,256,256,512)` exactly as `scripts/run_masking_benchmark.py --mode evaluate` does; the runner's
evaluate mode, however, reads `prior_logit.npy`, `locus_split.npz`, `patient_split.npz` from the *run directory* (TCGA objects) and the dataset block of the config, so it cannot be pointed at an external matrix. **Needs a new
evaluation script** (~150 lines: load best.pt, build external `MaskingDataset` on the re-keyed matrix, `evaluate_loader`, `reconstruction_metrics`, write via `eval_layout` under a new `external_transfer` view). No existing code does TCGA-to-external transfer:
`data/transfer_protocols.py` only builds locus-membership sets (used by the removed downstream runners) and `upscaling/` is a separate 450K-to-EPIC task trained on ComputAgeBench.
Verification that nothing needs to be re-trained: all four arms' `prior_logit.npy` are bit-identical across arms (checked for seed 17), 408,399 loci each.

Pitfalls specific to B (reasons it stays secondary):

1. **Prior shift.** Decoder output is `sigmoid(prior_logit + delta)` and the observed tokens carry `logit(beta) - prior`. TCGA priors are pan-cancer train means (global mean beta 0.488); whole blood differs systematically at many loci, so the residual inputs and the delta needed are out of the training distribution. Two variants, both pre-declared and both reported, no preferred one: **B-strict** (TCGA prior from the run directory, no external data at all) and **B-recalibrated** (prior fit on the external *train* patients only; the decoder is still not trained, but external methylation enters the prior, so it is not "no external data").
2. Coverage mismatch: the decoder was trained on the 408,399-locus universe; the cohort has 408,390 (GSE55763) / 408,017 (GSE40279); missing loci are simply absent from the evaluation pools (identical for all arms; no representation is affected).
3. Platform/normalization differences of the beta input (TCGA beta preprocessing not recorded locally; UNVERIFIED) and tissue shift (tumour vs blood) apply to all arms but can differentially affect arms with different adapter dimensions.
4. Mask patterns: unchanged (same `MaskingDataset`, mask seed 17001, panels of 2048), so identical across arms.
5. The transfer result cannot test locus-representation quality independently of the decoder's tolerance to distribution shift; interpret as a robustness descriptor.

### Recommendation

Primary = A (GSE55763; backup GSE40279 if the user prefers a ready, documented out-of-corpus cohort). Secondary exploratory = B-strict and B-recalibrated on the same external test patients (same split file; needs the new eval script). The two types are reported separately, never pooled.

## 2. Fairness contract (identical to the TCGA one)

Same patients, same loci, same masking pattern (`mask_seed 17001`; masks are a function of seed, epoch, row and fraction only), same reconstructor and hyperparameters, same seeds 17/42/97 for every arm. No representation-specific intersection:
a store lacking a universe locus makes the run fail loudly (`RepresentationCoverageError`, already implemented); the universe is fixed from the cohort and the benchmark alone. No per-arm tuning of any kind (epochs, lr, batch, adapter width, early stop, seed). Per-arm overrides are forbidden and
audited (the audit compares each arm's resolved config to the shared template and permits differences only in `representation.*`, `experiment.name`, `training.seed`). Modern sequence FMs are added later by registering a store that covers the same 408,399 TCGA ids (the external matrix is already TCGA-keyed), then running the same 3 seeds on the same split and budget; no protocol parameter is allowed to change when they are added. Frozen store hashes (matrix.yaml): regulatory `fa3a8163...`, functional `3d99fe5b...` (`full_a18b869b.h5`), CpGPT-large `c08138cd...`, DeepCpG-DNA `e7280d5f...`.

## 3. Dataset and split

Recommended: **GSE55763** (2,711 samples; 2,664 individuals; 450K; whole blood; coverage 408,390/408,399). Backup: **GSE40279** (656 subjects; 408,017/408,399; out-of-corpus for CpGPT per sibling docs). Justification: audit sections 1-2.
The decision of primary vs backup is the user's (they tie on points; see audit).

Split unit = individual. Rules (all deterministic, seed 20260925, the TCGA patient-split seed, fixed before any run):

* Leakage control for GSE55763: the 72 samples carrying "Technical replicate group k" descriptions (47 replication-only + 25 population samples) are **forced into the training set** (never validation/test). This removes any possible
  replicate leakage regardless of unresolved subject identity (individual ids do not exist in GEO metadata). Remaining 2,639 samples are assigned 80/10/10.
* Stratification: random assignment within strata of sex x age tercile (6 strata); chip id (249) and site are **not** stratified; their composition per split is reported as a diagnostic only, never used to re-draw the split.
* Resulting sizes: **GSE55763 train 2,183 (2,111 + 72 forced), validation 264, test 264** (total 2,711). **GSE40279 train 524, validation 66, test 66** (656 subjects, unique ids; strata sex x age tercile; site reported).
* Split file: npz in the repo's patient-protocol format (`train, validation, test, sample_names, seed`) stored under `outputs/external_reconstruction_v1/` with sha256 recorded in the new matrix spec; the loader's partition + group check is run before any training.
* The test indices are stored in the file (names only) but the runs are launched with `evaluation.patient_view: validation` and `require_patient_view: validation`, so no test row is read.

## 4. Locus universe

Universe = benchmark universe intersected with the cohort's array content: GSE55763 **408,390 loci** (chr1-19 388,590; chr20-22 19,800; 9 benchmark loci absent), GSE40279 408,017. Decided from cohort and benchmark only, identical for all arms.
Loci absent from the array are simply not in the matrix; no imputation, no fill. All-zero regulatory-embedding loci (849 in the universe; all 849 covered by GSE55763, 846 by GSE40279) are **kept** (identical to TCGA, where they were part of the 408,399);
a sensitivity analysis excluding them is reported as secondary. Loci with no finite training value are dropped by the runner identically for all arms and the count is recorded.
No locus holdout (`heldout_fraction: 0.0`), as in TCGA.

## 5. Panel, masking, evaluation

Panel size 2048 (unchanged): the cohort has ~0 NaN, 408k loci, so a 2048-locus panel is far below the available pool and 1024 targets per patient at mask 0.50 give 264 x 1,024 = 270,336 target values on the test split (TCGA validation: 940,032). Training mask fractions [0.15, 0.30, 0.50, 0.70, 0.90], validation selection fraction 0.50, evaluation at all five fractions, `panel_repeats: 1` (kept; the paired bootstrap requires a one-to-one pairing key `patient x column x block`, so repeats would need extra code), `mask_seed 17001`.
Optimizer: AdamW lr 1e-4, wd 1e-4, fp16, batch 8, constant lr, same model (256/256/256/512), same fractions, no early stopping, checkpoint = lowest validation MSE at mask 0.50 (`best.pt`), the last epoch never used automatically.

## 6. Training budget (rule fixed before runs; user decision between R1 and R2)

TCGA phase A: 120 exact epochs x ceil(7,342/8) = 918 steps = **110,160 optimizer steps** (each patient seen 120 times).

* **R2 (recommended): match optimizer steps.** epochs = ceil(110,160 / ceil(N_train/8)). GSE55763 (N_train 2,183 -> 273 steps/epoch): **404 epochs = 110,292 steps**. GSE40279 (524 -> 66 steps/epoch): 1,669 -> **1,670 epochs**. Rationale: the TCGA curves were still improving at epoch 120 (best epochs 116-119 in all 12 runs), so a rule that gives the external cohort 30% (GSE55763) or 7.2% (GSE40279) of the TCGA steps would leave all arms far from the TCGA training state and could make the ordering depend on convergence speed (adapter dimension 128/256/512) rather than on the locus information. Cost: overfitting risk on smaller N, controlled by the same `best_val_mse` selection for every arm (validation of 264 patients is noisier than TCGA's 918: stated limitation).
* **R1 (alternative, literal TCGA rule): 120 exact epochs.** 120 x 273 = 32,760 steps (30% of TCGA) for GSE55763; 7,920 steps (7.2%) for GSE40279. Cheaper, directly parallel to AMENDMENT 2, but is markedly less trained.
* Both rules: early stopping false, no per-arm change, no stop/extension based on curves; convergence flags (best epoch within the last 10% of the budget) are reported descriptively.
* Only one of R1/R2 will be frozen; running both is not allowed (it would add a researcher degree of freedom). Choice is made now, not after seeing curves.

## 7. Seeds and replicates

Decoder seeds 17/42/97 (`training.seed`), one patient split (single, frozen). Seeds capture initialisation (and GPU non-determinism) only, not data resampling (same as TCGA); there is no cross-validation (a 5-fold CV would cost 5x, 60 runs, and is not proposed). With N_test = 264 the patient-bootstrap CI half-width is roughly sqrt(918/264) = 1.9x wider than the TCGA validation CI for the patient component; the large TCGA effects (CpGPT +12.7%, DeepCpG +32.7% MSE vs the candidate) are expected to remain resolvable, but the near-parity contrast with the functional arm (TCGA rel. delta 0.44-0.65%) may not be: stated in advance.

## 8. Metrics and statistics

Primary: MSE at mask 0.50 (best.pt of each run), per seed. Secondary: MAE, MAS-PCC (per-patient), MAC-PCC (per-locus), prior-only MSE and skill vs prior, all five mask fractions (descriptive).
Paired uncertainty: `encode_atlas.statistics.paired_bootstrap` (crossed patient x 1 Mb genomic-block bootstrap, 2,000 replicates, seed 17), contrasts = **comparator minus frozen candidate** (positive = candidate better), per seed, no aggregation across arms and no multiplicity pooling; across-seed summary = the 3 per-seed estimates, their min/max and the count of seeds whose CI excludes 0 (as in Phase A). Pairing check (identical `sample_index`, `target_matrix_column`, `panel_repeat`, `target`, `prior_prediction` across arms in a seed) is a hard precondition.
Contrasts: (1) functional - candidate, (2) CpGPT-large - candidate, (3) DeepCpG-DNA - candidate (main, as in Phase A); (4) DeepCpG-DNA - CpGPT-large (to express the ordering CpGPT > DeepCpG; secondary).
Stratified, descriptive: target loci chr1-19 vs chr20-22 (candidate fit scope chr1-19 vs functional/CpGPT/DeepCpG all-locus or external), with and without the 849 zero-embedding loci, by sex/age stratum.

## 9. Outcomes that would support / not support the TCGA ordering (fixed now, neutral wording; no decision rule, no aggregation)

All statements are about the per-seed paired contrast at mask 0.50 (relative delta = (comparator - candidate)/candidate on MSE, with the 95% bootstrap CI), and the same sign on MAE:

* Contrast (2) CpGPT-large - candidate and (3) DeepCpG-DNA - candidate: **consistent with the TCGA ordering** if the delta is positive with CI above 0 in each seed; **not consistent** if negative with CI below 0 in any seed; **inconclusive** for a seed whose CI includes 0. Magnitude is reported but not required to match TCGA (+12.7% / +32.7%).
* Contrast (4) DeepCpG-DNA - CpGPT-large: positive with CI above 0 supports "CpGPT-large above DeepCpG-DNA"; negative with CI below 0 does not.
* Contrast (1) functional - candidate: TCGA showed the functional arm 0.44-0.65% *lower* MSE (3/3 seeds, CIs excluding 0). "Near parity" is reported as consistent if each seed's CI lies within +/-1.5% relative (margin fixed now), as "functional lower" if the CI is below 0, as "candidate lower" if above 0 (not consistent with the TCGA sign), and inconclusive if the CI extends beyond +/-1.5%. The sign alone is never turned into a claim about superiority.
* Secondary metrics (MAE, MAS-PCC, MAC-PCC) may not rescue or overturn an MSE contrast; they are reported alongside.
* Exploratory (labelled as such, never primary): all-fraction results, chromosome strata, zero-locus exclusion, B-strict/B-recalibrated transfer, sex/age strata, any modern FM added later.
* Nothing is promoted to the README, catalog or paper until separately approved; a result here is "validation-only, trained, not confirmed" until the authorized test evaluation.

## 10. Test protection, freeze and recording

1. PHASE V (validation only): train all arms x seeds, evaluate `validation` split in `evaluation/validation/` (`output_layout: split_dirs`, `require_patient_view: validation`); statuses limited to `trained_validation_frozen`.
2. Freeze: external matrix spec (new file, e.g. `configs/experiments/external_reconstruction_matrix/matrix.yaml`, never the TCGA one) with `freeze_state: draft|final`, `test_set_authorized: false` until the user sets it after `final` and an all-green audit (same state machine; needs the opt-in parametrisation of `authorize_test_split`).
3. PHASE T (test): `--mode evaluate` on the saved run directories into `evaluation/test/` only; a split directory that already has results is never reused without `allow_overwrite_split_dir`.
Record per run and in the matrix spec: sha256 of the external matrix h5, phenotypes/metadata, patient split npz, locus protocol npz, prior (`prior_logit.npy` equal across arms), each store (must equal the frozen hashes), `best.pt`, resolved config hash, code git commit, universe size, number of dropped loci, forced-train sample list, seed, wall-clock; per-run `confirmation_status.json`.
**Pre-run audit conditions (all must be green before any launch):** store sha256 equal to the frozen values; every store covers 100% of the external universe ids; split file sha, partition and zero overlap (replicate forced-train check, no TCGA/TARGET ids); locus protocol identical for all arms; per-arm resolved configs identical except representation/seed/name; `epochs` equal to the frozen R1/R2 value; `early_stopping: false`; `require_patient_view: validation`; `test_set_authorized: false`; no `evaluation/test` directory anywhere; the TCGA matrix and `configs/frozen/*` unchanged (hash check).

## 11. Explicit deviations from the TCGA protocol (all declared before any run)

| # | Deviation | Reason |
| --- | --- | --- |
| D1 | External cohort instead of TCGA; GSE55763 (or GSE40279) matrix re-keyed to the TCGA global cpg_idx | independence from TCGA; the stores are frozen |
| D2 | Universe 408,390 (GSE55763) instead of 408,399; cohort-defined, same for all arms | array content |
| D3 | Patient split: 80/10/10 of individuals, stratified sex x age tercile, replicate samples forced into train; N 2,183/264/264 | small N, replicate leakage |
| D4 | Budget R2: epochs = ceil(110,160/ceil(N_train/8)) = 404 (instead of the literal 120) | matches optimizer steps; user may choose R1 |
| D5 | `training.save_epoch_checkpoints: false` (opt-in) | disk (404 x 14 MB per run) |
| D6 | New external matrix spec and opt-in test-authorization parametrisation | the TCGA matrix/guards are TCGA-specific and frozen |
| D7 | Single patient split (no CV) and test N=264 | cost; wider CIs than TCGA |
Unchanged: model, optimiser, batch 8, lr, wd, fp16, mask fractions, panel 2048, panel_repeats 1, selection fraction 0.50, mask_seed 17001, seeds, bootstrap (2,000 replicates, seed 17), best_val_mse checkpointing, early stopping off.

## 12. Compute cost estimate

Measured TCGA phase A (12 runs): 5,281-8,545 s per run (1.5-2.4 h; CpGPT 512D slowest ~8.4-8.5k s, DeepCpG 128D ~5.3-5.8k s, regulatory/functional 6.3-7.7k s), ~49-60 s/epoch at batch 8 with 8 loader workers for 7,342 train + 918 validation patients (loader-bound; ~5.9 ms per patient-sample incl. validation), ~1.7 GB/run (14 MB/epoch), 3 concurrent runs scaled fine, shared machine (load average ~30).

| Study | N train / val / test | Epochs (steps) | s/epoch (est.) | per run (est.) | 12 runs sequential | 3 concurrent | Disk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A, GSE55763, R2 | 2,183 / 264 / 264 | 404 (110,292) | ~14 | ~1.6-1.9 h incl. ~5 min setup + eval | ~19-23 h | ~7-9 h | ~5.7 GB/run if per-epoch checkpoints are kept (68 GB); ~0.1 GB/run with `save_epoch_checkpoints: false` |
| A, GSE55763, R1 | same | 120 (32,760) | ~14 | ~0.6 h | ~7 h | ~2.5-3 h | ~1.7 GB/run |
| A, GSE40279, R2 | 524 / 66 / 66 | 1,670 (110,220) | ~3.5 | ~1.6-1.9 h (+ per-epoch overhead) | ~19-23 h | ~7-9 h | 23 GB/run unless checkpoints pruned |
| A, GSE40279, R1 | same | 120 (7,920) | ~3.5 | ~0.2 h (setup dominated) | ~3 h | ~1 h | 1.7 GB/run |
| B transfer, either cohort | inference only (12 checkpoints x 5 fractions x val/test) | none | n/a | minutes per checkpoint (store load + ~2.5k panels) | < 1 h | n/a | negligible (predictions.npz ~ tens of MB) |
| Setup (one-off) | GSE55763 prep: stream 10.4 GB gz, 408,390 x 2,711 float32 = 4.4 GB h5 | | | ~0.5-1 h CPU | | | 4.4 GB |
Estimates scale the measured TCGA throughput linearly with (train + validation) samples; I/O differs (the re-keyed h5 should use chunks (1, 8192), uncompressed, like the TCGA matrix to keep read cost equal). They are not measurements on the new data.
Modern FMs later: +1 arm x 3 seeds per added model at the per-run cost above.

## 13. Main confounding risks

1. Platform/probe overlap: 450K vs EPIC; the benchmark universe is 450K-probe selected, so every cohort probes the same design-biased loci (promoters/islands/genes); ENCODE-track coverage is itself biased toward the same regions (arms inherit it; generality is limited for all).
2. Tissue and cell-type composition: whole blood is a leukocyte mixture; cell composition varies with age/sex/smoking/disease and can dominate the reconstructable variance; TCGA is multi-tumour. Differences in what is reconstructable (blood methylation is low-variance across subjects) change MSE scale and may change the prior-versus-locus contribution.
3. Age/sex effects and disease status (GSE42861/GSE147221) as unmodelled covariates; stratified split by sex x age mitigates distribution mismatch between splits only.
4. Batch effects: GSE55763 249 chips, GSE40279 4 sites/9 plates; random (non-stratified) assignment and reported diagnostics; no batch correction applied.
5. Preprocessing differences vs TCGA beta (quantile + control-probe adjustment for GSE55763; BeadStudio average beta for GSE40279 with 1.6% exact-0/1 values); TCGA normalization state not recorded locally (UNVERIFIED). Beta clipping (`beta_epsilon` 1e-4) treats exact 0/1 identically for all arms.
6. Probe-design bias x ENCODE coverage: the 849 zero-row loci and loci with sparse track overlap; chr20-22 and chr1-19 strata reported.
7. CpG density/GC: sequence models (CpGPT, DeepCpG) encode it directly; the regulatory arm only through peak overlap; any gain cannot be attributed to supervision rather than sequence/annotation content.
8. Zero rows (849 loci all-zero in the candidate store): kept (as in TCGA) with a sensitivity analysis.
9. Fit-scope mismatch: candidate SVD fit on chr1-19 (388,599 loci, chr20-22 transformed only); functional SVD fit on all 408,399 (transductive); CpGPT/DeepCpG external-pretrained. Mitigation: chromosome-level stratified contrasts (chr1-19 vs chr20-22).
10. Dimension/adapter capacity differences (128/256/256/512 raw dims into a shared 256 latent): affects convergence speed (motivates R2) and cannot be removed without violating the fixed-model rule.
11. Small N (test 264 or 66): wide CIs, the near-parity contrast may be inconclusive; single split.
12. Replicate/relatedness leakage: handled for the identified replicate samples (forced-train); relatedness (twins/families) cannot be excluded from local metadata (UNVERIFIED; population cohorts assumed).
13. Pre-training overlap of CpGPT/DeepCpG with the external GEO cohort: CpGCorpus is a GEO array corpus (150k samples, >2,000 studies); membership of GSE55763/GSE42861/GSE147221/GSE87571 is UNVERIFIED (corpus list not local); GSE40279 documented out-of-corpus (sibling docs, local summary of the CpGPT paper); DeepCpG-DNA is scRRBS-trained (no array overlap). A CpGPT advantage on an in-corpus cohort would be a leakage-type confound; this is the reason GSE40279 is recommended as the backup/sensitivity cohort if the user wants a pre-training-clean comparison.
14. Transfer (B): prior shift and tumour-to-blood decoder shift (section 1B).
15. Non-independence between ComputAgeBench GSE42861 and the local GSE42861 (same GEO study), and between GSE56046/GSE56581 (same MESA participants possibly): do not combine cohorts.

## 14. Decisions needed from the user

1. Primary cohort: GSE55763 (mechanical winner on the prespecified criteria, needs a new preparer) versus GSE40279 (tie on points, ready, out-of-corpus for CpGPT, N=656).
2. Budget rule R1 (120 epochs) versus R2 (matched steps, 404 epochs); one only.
3. Approval of the opt-in code changes (test-authorization parametrisation, checkpoint pruning, patient-protocol generator, matrix builder, transfer-evaluation script) - none written yet.
4. Whether B (transfer) is wanted, and the +/-1.5% parity margin for the functional contrast.
5. Whether the 72 forced-train replicate samples rule and the sex x age-tercile stratification are accepted.
