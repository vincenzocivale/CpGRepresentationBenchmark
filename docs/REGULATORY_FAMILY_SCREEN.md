# Regulatory family screen (frozen before any run)

Status: PHASE A (preparation). Criterion, arms and protocol below were fixed before any screening run.
Validation patients only; TCGA test patients are never read, scored or used (guard: `evaluation.require_patient_view: validation`).

## Purpose

Informal screening of which experimental-track FAMILIES carry information about methylation beyond histone ChIP-seq,
to decide which (if any) additions deserve a later confirmation run. This is not a selection of a final representation
and not a formal test.

## Arms (all global TruncatedSVD-256, fit only on `discovery_chr1_19` loci (sha256 `995edc58...`), seed 17, `replicate_weighting: none`, no dense features)

| Arm | Feature set (`configs/feature_sets/`) | Tracks | Store |
| --- | --- | --- | --- |
| `regulatory_histone` | Histone ChIP-seq | 1,959 | `data/cache/representations/regulatory_histone__global_svd256__discovery_chr1_19.h5` |
| `regulatory_histone_dnase` | Histone + DNase-seq | 2,492 (1,959 + 533) | `..._histone_dnase__global_svd256__...` |
| `regulatory_histone_tf` | Histone + TF ChIP-seq (tf_binding + ctcf), minus the 14 methylation-linked tracks (ZBTB33, MBD1, MBD2, DNMT1, DNMT3B) | 3,618 (1,959 + 1,659) | `..._histone_tf__global_svd256__...` |
| `regulatory_clean` | Histone + TF (minus the 14) + DNase | 4,151 | `..._clean__global_svd256__...` |

All four runs are fresh; no number from the ENCODE campaign or the aborted selection campaign is reused.

## Protocol

Identical to the ENCODE campaign template (`configs/experiments/masking/functional_pca_genomewide.yaml`) through the same generator logic as
`scripts/run_regulatory_selection.py`, via `scripts/run_regulatory_family_screen.py` (configs in `configs/experiments/regulatory_family_screen/`,
outputs in `outputs/regulatory_family_screen_v1/`):

- frozen patients (`outputs/encode_atlas_v1/patients.npz`, 7342/918/918), frozen genome-wide seen loci (`loci_seen.npz`, 408,399 CpGs);
- model, optimizer, budget unchanged: 30 epochs, lr 1e-4, wd 1e-4, mixed precision, panel 2048, hidden 512 (enforced by `validate`);
- decoder seed 17, mask seed 17001, `panel_repeats: 1`, `evaluation.num_workers: 0`, `save_predictions: true`, `patient_view: validation`.

### Throughput deviation from the ENCODE campaign (explicit, shared by all four arms)

Profiling on the A100 showed the run is data-loader bound, not GPU bound: with `num_workers=0` the loader delivers about
95 samples/s regardless of batch size (random HDF5 column reads, ~10 ms/sample), while the GPU step itself takes 11-20 ms
for batch 8-64 (peak GPU memory 0.3-1.5 GB; GPU utilisation 5-12 %). Two settings therefore deviate from the campaign:

- `training.batch_size` 32 and `evaluation.batch_size` 32 (campaign: 8). Larger batches do not add throughput once loading is
  the limit, so 32 was chosen over 64/128 to keep more optimizer steps per epoch (about 230 vs 920 in the campaign).
  Learning rate, weight decay and epochs are NOT rescaled. Fewer optimizer steps per epoch may slightly change convergence,
  and absolute MSE is not directly comparable with the campaign numbers (batch 8). All four arms share the setting, so the
  within-screen paired comparison stays valid.
- `training.num_workers` 8 (campaign: 0). Allowed because the stores are precomputed (CLAUDE.md restricts only online
  encoders). Masks are keyed on (seed, epoch, row, fraction), so worker count does not change what is sampled.
  Validation during training uses 2 workers (`min(2, num_workers)` in `run_masking_benchmark.py`) and the training batch size.

`validate` enforces both values.

### Masking fractions: what was found and what was done

In `scripts/run_masking_benchmark.py` the fractions enter in three independent places:
(a) TRAINING: `training.mask_fractions` ([0.15, 0.30, 0.50, 0.70, 0.90]); the batch sampler draws one fraction per batch from this list;
(b) EVALUATION: `evaluation.mask_fractions` only decides which `evaluation/seen/mask_<f>/` directories are produced after training;
(c) CHECKPOINT SELECTION: `evaluation.selection_mask_fraction` (0.50) defines the validation set used for per-epoch validation and `best.pt`.
These are separate config keys, so no ambiguity: the screen keeps (a) and (c) exactly as in the campaign (the model is trained on the
same multi-fraction mixture and the checkpoint is selected at 50%) and sets only (b) to `[0.5]`. The model trained here is therefore
the same model the campaign would have trained; only the 15/30/70/90% post-hoc evaluations are skipped. `validate` fails if (a) or (c) differ from the template.

Outputs per run: `evaluation/seen/mask_0.50/{metrics.json,predictions.npz}` (keys `prediction`, `target`, `prior_prediction`,
`target_matrix_column`, `sample_index`, `panel_repeat`), the format consumed by `encode_atlas/statistics.py::prediction_table` / `paired_bootstrap`;
plus `history.json` (per-epoch validation MSE/MAE/MAS-PCC/MAC-PCC at 50%).

## Screening criterion (fixed before running)

An addition (DNase, TF, or both = Clean) counts as POTENTIALLY USEFUL iff, versus `regulatory_histone`, at 50% masking on validation patients:
1. validation MSE improves by at least about 1% relative (delta = MSE(arm) - MSE(histone); relative = delta / MSE(histone) <= -1%), AND
2. the direction is consistent in the paired analysis (the patient x 1 Mb block bootstrap CI of delta lies entirely below 0, and the sign agrees for MAE).

This is an informal go/no-go for later confirmation runs, not a formal hypothesis test. Out of scope for the screen: other decoder seeds,
other masking fractions, test patients, block-SVD or equal-group-mass weighting, single marks, H3K4me3, external data.

## Analysis plan

- Primary: MSE at 50% (validation), per arm. Secondary: MAE, MAS-PCC and MAC-PCC (`evaluation/metrics.py`), prior-relative skill, best epoch.
- Paired: delta MSE of each of dnase / tf / clean vs `regulatory_histone` (and clean vs the two intermediate arms, descriptively) with
  `paired_bootstrap` (crossed patient x 1 Mb block, 1000 replicates, seed 17). Report point, 95% CI, relative delta.
- Descriptive diagnostics only: MSE by all-zero vs non-zero embedding rows, in-fit (chr1-19) vs out-of-fit (chr20-22).
- Caveat: one decoder seed, so seed variance is not estimated; the bootstrap covers patient/locus sampling only.

## Cost accounting plan

Record per run: wall-clock (log timestamps), epochs until best, GPU hours (sequential single GPU), embedding fit time
(1,959 tracks: see compression report; Clean fit 976 s), embedding size. Compare cost to gain (relative MSE gain per extra track block) in the final write-up.
Prior campaign timing: about 100 s/epoch (30 epochs ~ 50 min) plus loading; expect about 1 h per run, about 4 h total.
