# Regulatory confirmation protocol (frozen before any run)

Status: PHASE 1 (preparation). Arms, training regime, convergence rule and decision rule below are fixed before any confirmation run.
Validation patients only (guard `evaluation.require_patient_view: validation`); TCGA test patients are never read, scored or used.
Nothing here is promoted to the README or to a final representation without separate approval.

## AMENDMENT 2 (2026-10-03): training budget of the FINAL confirmation (frozen)

Recorded before any confirmation run and before any test-set evaluation (no confirmation run has been executed; the test set has not been read).

- **Final confirmation budget:** a fixed MAXIMUM of **120 epochs for every arm** (same value for all arms and seeds), batch size and all other
  training hyperparameters as in the canonical ENCODE campaign recipe (constant-lr AdamW; no scheduler, so `epochs` is only a cap).
- **Checkpoint selection:** `best_val_mse` — the checkpoint with the strictly lowest validation MSE at 50% masking (`best.pt`), restored for
  evaluation, exactly as the runner already does. No other selection criterion (no MAE, PCC, test or external metric) may be used.
- **Historical status of the 80-epoch protocol:** the 80-epoch fixed-budget runs (sections below, including the 3-seed Histone / +DNase / Clean
  comparison) remain the *historical protocol of the feature-selection phase*. They are NOT rerun, and their results are not altered by this
  amendment. Everywhere below, "80 epochs" refers to that phase only.
- **Freeze:** with this amendment the confirmation training budget (120 epochs maximum, `best_val_mse` selection) is FROZEN. It may not be changed,
  extended, shortened or re-tuned — per arm or globally — on the basis of validation or test results of the confirmation.
- **Closure of the last ambiguity (supersedes the earlier early-stopping interpretation):** the final confirmation executes **exactly 120
  epochs for every arm and every seed**. Early stopping is **DISABLED for all confirmation runs** (`training.early_stopping: false`; the engine
  treats a falsy value as off, so the global engine behaviour and the opt-in key are unchanged). `best.pt` is still the checkpoint with the
  lowest `best_val_mse` at 50% masking (`evaluation.selection_mask_fraction: 0.5`); the **last epoch is never used automatically** — it is only
  the evaluated checkpoint if it happens to be that minimum. It is **not allowed to stop or prolong any individual arm or seed** on the basis of
  the observed learning curve (validation or test); every arm gets the same 120 epochs regardless of its curve.

## Question

Does `Histone + DNase` suffice, or does `regulatory_clean` (Histone + TF minus 14 methylation-linked tracks + DNase) carry a reproducible extra
advantage? The family screen (`docs/REGULATORY_FAMILY_SCREEN*.md`) had one decoder seed, batch 32 and under-trained models (best epoch 28-29 of 30).
This run removes both confounders: 3 decoder seeds, canonical batch size, and a fixed convergence rule.

## Arms and runs

| Arm | Tracks | Store (global TruncatedSVD-256, fit on `discovery_chr1_19`, sha256 `995edc58...`) |
| --- | ---: | --- |
| `regulatory_histone` | 1,959 | `data/cache/representations/regulatory_histone__global_svd256__discovery_chr1_19.h5` |
| `regulatory_histone_dnase` | 2,492 | `..._histone_dnase__global_svd256__...` |
| `regulatory_clean` | 4,151 | `..._clean__global_svd256__...` |

Decoder seeds 17, 42, 97 (`training.seed`: model initialisation and global RNG) -> 9 runs, one GPU job at a time, order seed-major
(seed 17: histone, histone_dnase, clean; then 42; then 97) so partial results are usable. Configs: `configs/experiments/regulatory_confirm/`
(generated and checked by `scripts/run_regulatory_confirm.py`), outputs `outputs/regulatory_confirm_v1/`.

Same data protocol as the ENCODE campaign template (`configs/experiments/masking/functional_pca_genomewide.yaml`): frozen patients
(`outputs/encode_atlas_v1/patients.npz`, 7342/918/918), frozen genome-wide seen loci (`loci_seen.npz`, 408,399 CpGs, no held-out loci),
mask seed 17001, panel 2048, `panel_repeats: 1`, `save_predictions: true`, `patient_view: validation`.
Note: `mask_seed` (17001) is shared by all runs, so the three decoder seeds see the same masks, sample order and validation panels;
decoder-seed variance therefore reflects initialisation (and GPU non-determinism) only, not data resampling.

## Training regime: canonical campaign, with three documented deviations (shared by all 9 runs)

Identical to the campaign: batch size 8 (train and evaluation), AdamW lr 1e-4, weight decay 1e-4, fp16, same model (256/256/256/512), same TRAINING mask
fractions [0.15, 0.30, 0.50, 0.70, 0.90], checkpoint selection on 50% validation MSE (`best.pt`, restored for evaluation by the runner). `validate` fails otherwise.
Deviations:

1. `training.epochs` 80 (campaign 30) plus opt-in early stopping (below).
2. `training.num_workers` 8 (campaign resolved configs: 0). Throughput only; see next section.
3. `evaluation.mask_fractions: [0.5]` (only the post-hoc evaluation directories are reduced; training and selection are unchanged).

### LR schedule and max_epochs

The optimiser is plain AdamW at constant learning rate: `training/engine.py` has no scheduler, warmup or decay, so `epochs` is only a cap on run length and
changing it does not change the learning-rate trajectory. The first 30 epochs of an 80-epoch run follow the campaign recipe exactly (verified: epoch-0 train MSE of a throwaway
profile run, 0.020694, matches the campaign's 0.020691; the small remainder is fp16 / GPU non-determinism).

### Workers are scientifically neutral (verified in code and by test)

- Masks and panels: `MaskingDataset.__getitem__` draws from `np.random.default_rng(seed + epoch*1_000_003 + row*97 + fraction_key)`, a function of (mask seed, epoch, patient row, fraction) only.
  No worker id or global RNG is involved. The epoch lives in a shared-memory `Value`, so persistent workers see `set_epoch`.
- Batch composition and the per-batch mask fraction come from `MaskFractionBatchSampler`, which runs in the main process (`default_rng(seed + epoch*1_000_003)`).
- `num_workers` only changes who executes `__getitem__`. Batches are delivered in sampler order. `tests/test_early_stopping.py::test_worker_count_does_not_change_batches` checks bit-identical batches for 0 vs 2 workers across epochs.
- Evaluation uses `num_workers: 0`; validation during training uses 2 workers (runner default), also order-preserving.
- Residual non-determinism (fp16 atomics, cuDNN) exists with any worker count and is not worker-related.

### Convergence rule (identical for all arms and seeds)

Note (post-run wording): the 80-epoch run is a FIXED-BUDGET selection protocol, not a convergence claim. Early stopping never triggered in any of the 9 runs, best epochs were 77-79 and curves were still improving (slope about -0.05%/epoch).

`training.early_stopping: {patience: 10, min_delta_rel: 1.0e-4}` with `training.epochs: 80` (opt-in key; default off for every other caller; implemented in `training/engine.py`).
Validation MSE at 50% masking is evaluated every epoch. An epoch is a significant improvement iff MSE < reference x (1 - 1e-4), the reference being the last significant value. Training stops after 10 consecutive epochs without one.
`best.pt` remains the strict minimum validation MSE (what the runner restores); `early_stopping.json` per run records epochs run, best epoch, whether stopping triggered.
No methylation-based selection beyond this standard validation-MSE criterion, applied equally to every arm.

Justification (read-only campaign histories, `outputs/encode_atlas_v1/benchmark`, 82 batch-8 30-epoch runs):
- 73/82 runs have their best epoch at 29 (the last), so even the campaign was not converged; mean validation MSE was still falling about 1.0% per 5 epochs at epoch 30.
- Epoch-to-epoch validation noise is about 3.2e-5 (0.19% of MSE), larger than the late per-epoch trend (about 0.15%/epoch at epoch 30). Hence min_delta_rel must be far below the noise (1e-4 = 0.01%) and the safeguard is patience, not min_delta; patience 10 spans more than one noise-to-trend ratio.
- A power-law fit of the mean campaign curve (MSE = a + b (e+1)^-k, k about 0.15; a pure exponential fits worse) projects a further 2.2% at epoch 50, 4.1% at 80, 6.4% at 150 relative to epoch 30, with the per-epoch gain at epoch 80 about 0.05%, about 4x below the noise. 80 epochs is therefore the budget at which marginal progress is below noise per epoch while the cost stays bounded (~11 h worst case). 150+ epochs would double the cost for another ~2%.
- It is likely that some arms still improve at epoch 80 (stopping not triggered). That is accepted: the budget is identical for all arms, and the comparison is reported with the convergence diagnostics below. Extending training is outside this protocol.

### Cost estimate (measured with a 3-epoch throwaway run, written to the scratchpad, not to outputs)

Batch 8 with 8 workers: about 48-50 s per training epoch including the 50% validation pass (screen at batch 32: about 43 s; the run is loader bound). Per run: about 7 min setup (priors, split, store load) + epochs x 49 s + about 1 min 50% evaluation.
80 epochs: about 73 min per run, about 11 h for the 9 runs worst case; if early stopping triggers around epoch 60: about 57 min per run, about 8.6 h. Cost is practically independent of the arm. Disk: about 14 MB/epoch checkpoints, about 1.1 GB per run (the engine saves every epoch). The shared machine's load can change these numbers.
The earlier screen's absolute MSE (batch 32, 30 epochs) are NOT comparable with this run; only within-run paired contrasts are used.

## Metrics and analysis plan (per seed, then across seeds)

Per run, from `evaluation/seen/mask_0.50/{metrics.json,predictions.npz}` and `history.json`: MSE, MAE, MAS-PCC, MAC-PCC (`evaluation/metrics.py`), prior-only MSE (from `prior_prediction`, identical across arms), skill vs prior = 1 - MSE/MSE_prior, best epoch, epochs run, stopped_early.
Pairing check before any contrast: `sample_index`, `target_matrix_column`, `panel_repeat`, `target`, `prior_prediction` bit-identical across arms within a seed.

1. Convergence check (per run): best epoch < 80 and a plateau window (best epoch <= epochs_run - 10 when stopped early). A run whose best epoch is within the last 10 epochs of the cap is flagged "not converged". Also report the final learning curve (validation MSE per epoch, last 20 epochs) and the between-arm gap (arm MSE minus reference MSE) averaged over the last 10 epochs; the sign of the gap must be stable over those epochs. Non-convergence does not change how the rule is applied (equal budget) but is reported as a limitation.
2. Paired contrasts per seed: histone -> histone_dnase and histone_dnase -> clean (context: histone -> clean), via `encode_atlas.statistics.paired_bootstrap` (crossed patient x 1 Mb genomic-block bootstrap, 2000 replicates, seed 17): delta MSE (alt - ref), relative delta, 95% CI; delta MAE with CI; delta MAS-PCC and MAC-PCC (point only).
3. Across seeds: seed-mean delta MSE (and relative), min/max and SD of the three per-seed deltas, number of seeds with the correct sign (MSE, MAE) and with CI excluding 0. With n = 3, no inferential statement across seeds beyond these counts and spread.
4. Effect size vs prior: gap_prior = MSE_prior - MSE(ref); report delta MSE as a fraction of gap_prior and the change in skill vs prior.
5. Cost per extra track block (descriptive).

## DECISION RULE (fixed before runs)

Notation: relMSE = (MSE_alt - MSE_ref) / MSE_ref, negative favours alt. "CI excludes 0" refers to the per-seed paired bootstrap 95% CI.

(i) Histone+DNase vs Histone. Histone+DNase is CONFIRMED iff in all 3 seeds relMSE < 0 AND delta MAE < 0 (same direction), with CI excluding 0 for MSE in 3/3 seeds; it is a STRONG confirmation iff additionally the seed-mean relMSE <= -1%. Consistent but seed-mean between 0 and -1%: SUPPORTED-SMALL (reported as such; the screen's 1% bar is not met). Not consistent in sign across MSE/MAE and seeds: NOT CONFIRMED (Histone stays the reference).

(ii) Parsimony. If Clean improves over Histone+DNase by less than 0.5% relative MSE (seed-mean), OR the improvement is not stable (any seed with the wrong sign for MSE or MAE, or CI including 0 in any seed), prefer Histone+DNase (fewer tracks, simpler, interpretable). Clean is never preferred merely because its point estimate is lower.

(iii) Promote Clean over Histone+DNase only if ALL hold: seed-mean relMSE <= -0.5%; per-seed MSE CI excludes 0 with the same sign in 3/3 seeds; delta MAE has the same sign in 3/3 seeds (and the seed-mean MAE CI side supports it); AND the gain is biologically non-negligible, a judgement of the author that is NOT coded. AMENDMENT (after the runs, before any verdict was accepted): this clause originally carried an operational bar (delta MSE >= 1.5% of gap_prior). That bar was chosen post hoc by the assistant and was never preregistered by the user; it is NOT used in the decision. Gain as a percentage of gap_prior (prior-only MSE minus Histone+DNase MSE) is reported only as a descriptive number together with the skill-vs-prior change. When all statistical clauses of (iii) hold, the code verdict is 'Histone+DNase preferred by parsimony pending author judgement on non-negligibility' (rule (i) satisfied) or 'Clean advantage reproducible on MSE; non-negligibility not coded - author decision' (rule (i) not satisfied); Clean is never promoted automatically.

(iv) MAS-PCC and MAC-PCC may NOT break a tie or rescue a contrast when MSE and MAE do not support an advantage; they are reported as secondary.

(v) Out of scope: TCGA test patients, other mask fractions, new feature sets, single marks, block-SVD / equal-group-mass weighting, external biological validation, other seeds, longer training.

(vi) Nothing is promoted to the README, the representation catalog or the paper until separately approved. The result of the rule application is a recommendation only.

## Explicit rule application (to be filled after the runs; table skeleton)

| Seed | relMSE H->H+D [95% CI] | dMAE H->H+D | relMSE H+D->Clean [95% CI] | dMAE H+D->Clean | best epochs (H / H+D / Clean) |
| --- | --- | --- | --- | --- | --- |
| 17 | | | | | |
| 42 | | | | | |
| 97 | | | | | |
| mean (min, max, SD) | | | | | |

Then: rule (i) outcome, rule (ii)/(iii) outcome, convergence flags, delta as fraction of gap_prior, and the final recommendation.

## Reproduce

```bash
python scripts/run_regulatory_confirm.py generate && python scripts/run_regulatory_confirm.py validate
python scripts/run_regulatory_confirm.py run --dry-run
```

## Confirmation matrix (pointer; AMENDMENT 2 above is unchanged)

The final confirmation is specified in `configs/experiments/regulatory_confirmation_matrix/matrix.yaml` (arms, roles, store hashes, seeds 17/42/97,
mask fractions 0.15-0.90, shared protocol = AMENDMENT 2, `freeze_state`/`test_set_authorized` state machine) with the fairness contract in
`docs/REGULATORY_CONFIRMATION_FAIRNESS.md`. Pre-test audit: `python scripts/audit_confirmation_matrix.py`; runner:
`scripts/run_regulatory_confirmation_matrix.py`; analysis scaffold: `scripts/analyze_confirmation_matrix.py`. Nothing had been run when this pointer was written; the TEST split stays locked.

### Phase A and evaluation layout (confirmation-matrix preparation, 2026-10-03)

- Freeze states are `draft | final` (formerly `complete`). `final` requires no pending comparator and an all-green audit; `test_set_authorized: true`
  additionally requires `final`. The two modern sequence FM slots stay `pending`, so the audit still fails overall.
- PHASE A = validation-only training of the 4 fully registered main arms x seeds 17/42/97 (12 runs), launched with
  `run --phase A --split validation` (seed-major; order histone_dnase, functional, cpgpt_large 512D, deepcpg 128D; one job at a time; resumable via
  `.done`). Output root `outputs/regulatory_confirmation_v1/`; per-run `confirmation_status.json` (`trained_validation_frozen`, never `confirmed`,
  `test_read: false`) and aggregate `phase_A_status.json`. Not in Phase A: CpGPT 256D compact, DeepCpG HepG2, modern FMs, test evaluation, biological validation.
- Evaluation output layout: opt-in `evaluation.output_layout: split_dirs` writes `<run>/evaluation/<split>/seen/mask_<f>/{metrics.json,predictions.npz}`
  and `<run>/evaluation/<split>/summary.json` (split = `evaluation.patient_view`); the default `legacy` layout is unchanged. A split directory that
  already holds results is never reused unless `evaluation.allow_overwrite_split_dir: true`; the test split needs `test_set_authorized: true`,
  `freeze_state: final` and a green audit, and can only ever write under `evaluation/test/`.
- Canonical functional store (definitive): `outputs/encode_atlas_v1/embeddings/full_a18b869b.h5`. `functional_annotations__pca_native_genomewide.h5`
  is an unused historical artifact.
