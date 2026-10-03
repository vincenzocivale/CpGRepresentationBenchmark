# Confirmation matrix: fairness contract

Status: frozen with the matrix spec (`configs/experiments/regulatory_confirmation_matrix/matrix.yaml`, `freeze_state: draft` until the two modern
sequence FMs are registered and materialized). Nothing has been run. Training budget: `REGULATORY_CONFIRMATION_PROTOCOL.md`, AMENDMENT 2.
Fairness issues below are DOCUMENTED, NOT FIXED: no store is rebuilt, re-fit, re-cast or filtered to equalize arms.

## Contract

1. **Native-representation benchmark = primary result.** Every comparator is used at its own native dimension (candidate 256, functional 256,
   CpGPT-large 512, DeepCpG 128, modern FMs: whatever they emit) through the same reconstructor interface: the fixed model's learned adapter
   (`LayerNorm(d) -> Linear(d, 256) -> GELU -> LayerNorm`, `models/model.py`) maps each raw dimension to the common 256-wide latent. No dimension matching
   is required and none is applied in the main result.
2. **Dimension-controlled sensitivity (optional).** Where a native comparator is wider/narrower than 256, a projection to 256D may be reported as a
   SENSITIVITY: fit WITHOUT methylation labels (unsupervised, e.g. TruncatedSVD), the SAME compression procedure (algorithm, seed, fit loci, dtype
   policy) for every comparator it is applied to. Existing sensitivity store: `cpgpt_large_locus_256_compact` (campaign SVD-256, fit on all loci,
   variance retained 0.838). The main result stays native unless a methodological incompatibility is documented here.
3. **Shared everything else** (audit-enforced): patient split, loci, masks and mask seed 17001, reconstructor, optimizer/hyperparameters (batch 8,
   lr 1e-4, wd 1e-4, fp16, constant lr, panel 2048), exactly 120 epochs, no early stopping, `best.pt` by validation MSE @ 0.50, the same `best.pt`
   evaluated at 0.15/0.30/0.50/0.70/0.90. Seeds 17/42/97 (reconstructor init and training only; locus panels and masks are identical across arms).
4. **Sensitivity arms are never main comparators** (`role: sensitivity`, `main_arm` link, reported in a separate table).

## Fairness issues found, and how each will be reported

| # | Issue | Evidence | Reporting |
| --- | --- | --- | --- |
| 1 | Fit-scope mismatch. Candidate SVD is fit on chr1-19 (388,599 loci; chr20-22 transformed only); functional and CpGPT-compact SVD are fit on ALL 408,399 loci (transductive); CpGPT-native, DeepCpG are external (no benchmark fit). | frozen manifest `fit_loci`; `full_a18b869b.json` `n_fit_loci 408399` | The mismatch favors the comparators (more fit loci) and disfavors the candidate; report as a limitation; per-chromosome results (chr1-19 vs chr20-22) as a descriptive check only. Not equalized. |
| 2 | dtype: candidate float32; functional, CpGPT, DeepCpG float16. | store headers | Document only. float16 quantization is a (small) disadvantage of the comparators and is part of their published/campaign stores. |
| 3 | Zero-embedding rows: 849 candidate loci have an all-zero row (no track overlap); controls have none. | measured on the full store | Report n and the share of evaluated pairs on those loci; secondary descriptive MSE split (zero-row loci vs others) without changing the headline. |
| 4 | Adapter parameters grow with native dimension (128D: ~33k, 256D: ~66k, 512D: ~132k in `Linear`; plus LayerNorm). | `LocusAdapter` | Report adapter parameter counts per arm; the 256D compact CpGPT sensitivity separates capacity from information. |
| 5 | The functional store includes dense genomic-context inputs (CpG island context, gene region, cCRE classes, TSS distance, breadth); the candidate has 0 dense columns and no TF/CTCF tracks. | `provenance`, frozen manifest `dense_columns 0` | Framing: genomic context is part of the functional input, not independent validation. Reported as a feature-composition difference; no ablation is added here (the ENCODE campaign `drop_dense_*` arms exist as context). |
| 6 | Methylation-pretrained comparators: CpGPT and DeepCpG encoders were trained on methylation (external, not verifiable locally). Extraction is label-free and patient-independent, but the supervision caveat applies. | `REPRESENTATION_PROVENANCE.md` | Always labelled "pretrained on methylation" in tables; the candidate has no methylation exposure at any stage. Never claimed as supervision-free. |
| 7 | 450K/EPIC-specific or CpG-id-vocabulary FMs (e.g. MethylGPT, 49k-CpG vocabulary) cannot cover the genome-wide 408,399-locus array universe; they are absent by design. | `methylgpt_locus*.h5` coverage | Stated as a coverage limitation. No per-representation locus intersection is ever built (a store that does not cover the panel fails the audit). |
| 8 | Tissue/cell-specific DeepCpG checkpoint (HepG2) vs the canonical HCC checkpoint. | catalog | HepG2 is sensitivity only. |
| 9 | Two functional stores exist (campaign `full_a18b869b.h5` vs masking baseline `functional_annotations__pca_native_genomewide.h5`). | Numerically equivalent (|r| = 1.0000 per column on a 1-in-97 sample, max abs diff 0.05 on std ~32), different bytes. | DECIDED (definitive): `outputs/encode_atlas_v1/embeddings/full_a18b869b.h5` is the only functional store of the matrix; `functional_annotations__pca_native_genomewide.h5` is documented only as an unused historical artifact (matrix.yaml `canonical_store_decision`). Never mixed. |
| 10 | Sequence window and context differ: DeepCpG 1001 bp; CpGPT its own window; modern FMs per registration. The candidate uses no sequence at all. | catalog, registration template | Described per arm in the results table. |
| 11 | Seeds vary only reconstructor initialization (shared masks/panels); n = 3. | protocol | Seed spread descriptive; paired patient x 1 Mb block bootstrap per seed; no inferential claim across seeds beyond counts and spread. |

## Modern sequence foundation models

Two slots (`modern_sequence_fm`, `modern_sequence_fm_alt`) stay `pending`. The final benchmark is NOT authorizable until both are decided and
materialized (audit fails). A slot leaves `pending` only after a COMPLETE registration (`sequence_fm_registration.template.yaml`: model, checkpoint,
version, window, CpG position, strand, pooling, layer, dimension, preprocessing/tokenization, genome build, missing-loci policy, projection). None
of these choices may be optimized using TCGA methylation reconstruction.

## Test-set policy

`freeze_state: draft | final`; `test_set_authorized` may be true only when `freeze_state: final` AND `python scripts/audit_confirmation_matrix.py`
passes. The runner and the analysis script refuse the test split otherwise (`guards.require_test_authorization`).

## Phase A (validation-only training of the fully registered arms)

`run --phase A --split validation` trains the 4 fully registered main arms (histone_dnase, functional, cpgpt_large 512D, deepcpg 128D) x
seeds 17/42/97 (12 runs, validation split only, exact frozen protocol) without waiting for the pending modern FM slots. Those slots still
block `freeze_state: final`, `test_set_authorized: true` and every test evaluation. Phase A results are `trained_validation_frozen`, never
`confirmed`. Evaluation outputs use `evaluation.output_layout: split_dirs` (`<run>/evaluation/validation/...`), so a later test
evaluation (`<run>/evaluation/test/...`) cannot overwrite them (guards in `experiments/eval_layout.py`).
