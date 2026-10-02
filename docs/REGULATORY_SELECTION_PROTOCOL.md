# Regulatory representation selection protocol (frozen before any run)

Status: PHASE 1 (preparation). This document and the decision rule were fixed before any selection run was launched.
Selection uses train/discovery and VALIDATION TCGA patients only. Test patients are never read, scored or used.

## Arms and roles (all 256-d, all patient-agnostic, label-free)

| Arm | Role | Store |
| --- | --- | --- |
| `regulatory_clean_global_svd` | candidate (simplest) | `data/cache/representations/regulatory_clean__global_svd256__discovery_chr1_19.h5` |
| `regulatory_clean_block_svd` | candidate | `..._clean__block_svd256__...` |
| `regulatory_clean_block_svd_egm` | candidate (equal-group-mass replicate weighting) | `..._clean__block_svd256_egm__...` |
| `functional_annotations_pca` | control: campaign `full` ENCODE functional SVD-256 | `outputs/encode_atlas_v1/embeddings/full_a18b869b.h5` (read only, as in the campaign) |
| `regulatory_histone_global_svd` | control: histone-only | `..._histone__global_svd256__...` |
| `cpgpt_locus_large` | control: CpGPT-large locus embedding compacted to 256 | `outputs/encode_atlas_v1/embeddings/fm_cpgpt_locus_large_d3474a48.h5` (read only, as in the campaign) |

`regulatory_all_experimental` is excluded. Only the three Clean arms can be selected; controls give context only.

## Protocol (identical to the ENCODE campaign discovery stage; configs in `configs/experiments/regulatory_selection/`)

- Runner: `scripts/run_masking_benchmark.py` (via `scripts/run_regulatory_selection.py`), template `configs/experiments/masking/functional_pca_genomewide.yaml`.
- Patients: frozen `outputs/encode_atlas_v1/patients.npz` (split seed 20260925; 7342 train / 918 validation / 918 test; TCGA aliquots share a partition). Training, priors and checkpoint selection use train (checkpoint selection uses validation, as in the campaign); reporting uses validation only.
- Loci: frozen `outputs/encode_atlas_v1/loci_seen.npz`: all 408,399 array CpGs (genome-wide, no held-out loci), seen view only.
- Masks: mask seed 17001; fractions 15/30/50/70/90%; 2048-locus panels, one panel repeat; 50% primary.
- Model/budget: fixed reconstructor (latent/token/patient 256, hidden 512), 30 epochs, batch 8, lr 1e-4, wd 1e-4, mixed precision. No hyperparameter differs between arms (enforced by `validate`).
- Decoder seeds 17, 42, 97 (patients and masks do not change with seed). Predictions saved (`predictions.npz` per mask fraction) for paired analysis.
- GUARD: every config has `evaluation.require_patient_view: validation`; `cpg_repr_benchmark.experiments.guards.enforce_patient_view` is called by `run_masking_benchmark.py` and the selection runner and raises on `test`, on a missing `patient_view`, or on any other value. Configs without the key behave as before.

## Primary endpoint and analyses

Primary: reconstruction MSE at 50% masking on validation patients (per seed; mean over seeds reported).
Secondary: MAE, and MAS-PCC and MAC-PCC (both provided by `evaluation/metrics.py`), at all five masking fractions; seed stability (per-seed ranking and range).
Paired comparison: each Clean candidate against the other two (and against controls descriptively): paired delta MSE and relative delta with the crossed patient x 1 Mb genomic-block bootstrap (`encode_atlas/statistics.py::paired_bootstrap`, 1000 replicates), per seed and per mask fraction.
Diagnostics (descriptive only, never used for the decision): MSE split by loci with all-zero vs non-zero embedding rows; by track-coverage and peak-density quartiles; chr1-19 (in-fit) vs chr20-22 (out-of-fit transform).

## DECISION RULE (fixed before running)

Definitions, all on validation patients, 50% masking, paired against another Clean candidate B with delta = MSE(A) - MSE(B):
- A "convincingly beats" B iff (i) the 95% patient+block bootstrap CI of delta excludes 0, AND (ii) |relative difference| >= 0.5% of MSE(B), AND (iii) the sign of delta is the same in 3/3 decoder seeds (CI and magnitude checked on every seed; the reported point is the seed mean).
- "Indistinguishable": neither convincingly beats the other.
- A is "best-or-indistinguishable" iff no other Clean candidate convincingly beats A.

Rule:
1. Eligible winners are Clean candidates that are best-or-indistinguishable.
2. If exactly one is eligible, it wins.
3. If several are eligible (including when two differ by <0.5% relative with no convincing advantage), prefer the simplest: `global_svd` > `block_svd` > `block_svd_egm`.
4. A more complex method is chosen over a simpler one only if it convincingly beats the simpler one.
5. No EWAS, test patients, external data or secondary metrics may break ties. Secondary metrics and diagnostics are reported but cannot change the outcome.
6. The result is a candidate selection only; nothing is promoted in the README until test confirmation (a later, separately approved phase).

## Not done (out of scope for this selection)

Test-patient confirmation; external datasets; embedding-dimension sweep; feature pruning; autoencoder or learned compression; new comparators; EWAS/biological follow-ups; chromosome-held-out (OOD) evaluation.

## Known caveats (to be reported with results)

- Regulatory stores were fitted on chr1-19 loci (compression only, no methylation); chr20-22 rows are out-of-fit transforms. The campaign controls (`full`, CpGPT-large SVD) were fitted transductively on all 408,399 loci. The panel includes chr20-22 (19,800 loci), so candidates are slightly disadvantaged relative to controls on those loci; this does not affect candidate-vs-candidate comparisons (same fit scope) and no patient methylation entered any fit.
- Validation patients are also used for checkpoint selection (campaign design), equally for all arms.
- Regulatory stores are float32; controls are float16 (values are cast identically by the model adapter path).
- Zero embedding rows: 733 (Clean) / 1,166 (histone) loci; 0 for controls.
