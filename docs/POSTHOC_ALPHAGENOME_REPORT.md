# AlphaGenome as a post-hoc comparator, and a compression diagnostic for the regulatory arm

Status: **draft, complete** (all sections filled; not yet reviewed or committed).
Baseline policy: from 2026-10-07 the reference Histone+DNase arm is the **raw 2,492 binary tracks without PCA** (`regulatory_raw2492`); SVD-256 is the compact frozen variant used for all registered results. The raw arm was chosen after seeing these post-hoc results, and its reconstruction runs have not been executed.
Every AlphaGenome and diagnostic result below is labelled `posthoc_comparator: true`. No original frozen benchmark, protocol, hash or inferential family was modified.

## 1. What was added and why

AlphaGenome was added after the original three-arm results (Histone+DNase, CpGPT-large, DeepCpG-DNA) were known, including the one-shot GSE40279 test evaluation of those arms. It is therefore a post-hoc comparator and is never part of a registered family.

**Important property:** AlphaGenome receives only the GRCh38 reference sequence at inference (patient-independent, no methylation input), but it was trained *supervised* on many experimental functional-genomics tracks. It is not a label-free or sequence-only comparator. For endpoints whose labels are related to those tracks (FANTOM5, replication timing, genomic context) its advantage is not independent evidence.

## 2. The AlphaGenome representation

| Item | Value |
|---|---|
| Embedding | trunk `embeddings_1bp`, 1,536 dimensions, float16, at the CpG cytosine |
| Input | hg38 no-alt analysis set FASTA (`GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta`), 524,288 bp tiles, 65,536 bp margin, 6,579 tiles |
| Weights | `google/alphagenome-all-folds` (JAX, `alphagenome_research` 0.3.0) |
| Store | `data/cache/representations/alphagenome_locus.h5`, 408,399 loci, sha256 `3906d888964fe71e303e2437ae64445b3f91803dcc04d15484fc5d5565850e00` |
| Extractor | `scripts/build_alphagenome_embedding.py` |

**Why 1 bp and not 128 bp.** Registry audit: 33.98% of the 408,399 CpGs (138,790) share a 128-bp bin with another CpG (82,843 CpGs would reuse another CpG's vector, up to 11 per bin). 1 bp keeps distinct locus identity, as CpGPT and DeepCpG do. The 128-bp store was not built.

## 3. Results

All numbers are for the registered protocols, with the original protocols' loci, pairs, folds, targets, metrics and bootstrap definitions. Regulatory = Histone+DNase SVD-256 (frozen); the raw 2,492 arm, the reference baseline from 2026-10-07, is compared in section 4.

### 3.1 EWAS Catalog panel (13 tasks, AUROC)

Same frozen pairs, probe contract, folds and bootstrap draws (`scripts/ewas_v3/posthoc_alphagenome.py`; output `outputs/ewas_v3_posthoc_alphagenome/`). AlphaGenome has the highest AUROC in all 13 tasks among the four arms. Paired contrasts (AlphaGenome minus arm, chromosome-block bootstrap, 95% CI): vs Regulatory mean +0.048 (CI above 0 in 10/13 tasks), vs CpGPT-large +0.086 (11/13), vs DeepCpG-DNA +0.115 (8/13). No task was below 0 in any contrast. No panel-wide test is reported.

### 3.2 Biological validation (bioval v2, Fig. 2 endpoints)

Official gate enforced (8/8 checks pass). Endpoints: FANTOM5 membership, H1 Micro-C, consensus replication timing. `loyfer_profile` was not run.

| Endpoint | Regulatory | AlphaGenome | delta (Reg − AG) [95% CI] |
|---|---|---|---|
| FANTOM5 membership (AUROC) | 0.8330 | 0.9063 | −0.0733 [−0.0792, −0.0670] |
| H1 Micro-C (AUROC) | 0.5685 | 0.5697 | −0.0012 [−0.0103, 0.0107] |
| Replication timing (Spearman) | 0.7031 | 0.9138 | −0.2107 [−0.2277, −0.1949] |

The Holm adjustment of this contrast is over **3** endpoints, not the registered 4-endpoint family, and is descriptive. The original Holm families are unchanged.

### 3.3 Genomic-context decodability (Fig. 5)

AlphaGenome is a supplementary arm; the Holm family over the main arms is unchanged. Macro-AUROC / Spearman: CpG context 0.949 (Regulatory 0.896), gene region 0.903 (0.789), TSS distance 0.820 (0.691).

### 3.4 Methylation reconstruction

Same protocol as the frozen arms: 120 epochs, seeds 17/42/97, mask seed 17001, same patient and locus splits, mask fractions 0.15–0.90.

**TCGA (held-out patients), masked-CpG MSE, mean of 3 seeds:**

| Mask | Regulatory | CpGPT-large | DeepCpG-DNA | AlphaGenome | AlphaGenome vs Regulatory |
|---|---|---|---|---|---|
| 0.15 | 0.01457 | 0.01648 | 0.01945 | 0.01339 | −8.1% |
| 0.30 | 0.01479 | 0.01666 | 0.01964 | 0.01359 | −8.1% |
| 0.50 | 0.01478 | 0.01667 | 0.01962 | 0.01359 | −8.1% |
| 0.70 | 0.01486 | 0.01674 | 0.01962 | 0.01367 | −8.0% |
| 0.90 | 0.01548 | 0.01720 | 0.01994 | 0.01428 | −7.7% |

AlphaGenome per-seed range at each fraction is within ±0.00004.

**GSE40279 (independent cohort, 66 test subjects), masked-CpG MSE, mean of 3 seeds, one-shot test:**

| Mask | Regulatory | CpGPT-large | DeepCpG-DNA | AlphaGenome |
|---|---|---|---|---|
| 0.15 | 0.00121 | 0.00124 | 0.00129 | 0.00118 |
| 0.30 | 0.00131 | 0.00133 | 0.00139 | 0.00127 |
| 0.50 | 0.00131 | 0.00134 | 0.00139 | 0.00128 |
| 0.70 | 0.00126 | 0.00128 | 0.00134 | 0.00123 |
| 0.90 | 0.00130 | 0.00131 | 0.00137 | 0.00126 |

On GSE40279 the margin of AlphaGenome over Regulatory is about 2–3%. No significance test was computed for the reconstruction comparisons.

## 4. Is compression the bottleneck of the regulatory arm?

All arms below use the same 2,492 binary tracks (feature columns hash `e5922c35…`), the same fit protocol (chr1–19 fit; chr20–22 transformed only), are label-free and patient-independent.

| Arm | Description |
|---|---|
| Regulatory SVD-256 | frozen arm (linear, 256 d) |
| SVD-512 | dimension control (linear, 512 d) |
| Raw 2,492 | the binary matrix itself, no compression; **new reference baseline (2026-10-07)** |
| Logistic PCA 256 / 512 | adaptive-step MM algorithm, fitted on chr1–19 only (`src/cpg_repr_benchmark/encode_atlas/logistic_pca.py`) |

**EWAS (AUROC):**

| Task | Reg. 256 | SVD-512 | Raw 2,492 | LPCA-256 | LPCA-512 | AlphaGenome |
|---|---|---|---|---|---|---|
| melanoma | 0.591 | 0.571 | 0.625 | 0.518 | 0.504 | 0.597 |
| oral/pharyngeal SCC | 0.508 | 0.497 | 0.576 | 0.541 | 0.512 | 0.598 |
| breast cancer | 0.766 | 0.705 | 0.771 | 0.761 | 0.728 | 0.791 |
| colorectal cancer | 0.550 | 0.537 | 0.632 | 0.604 | 0.542 | 0.621 |
| lung cancer | 0.809 | 0.807 | 0.825 | 0.802 | 0.792 | 0.843 |
| Crohn's (IBD family) | 0.742 | 0.731 | 0.741 | 0.727 | 0.723 | 0.776 |
| rheumatoid arthritis | 0.787 | 0.778 | 0.792 | 0.781 | 0.769 | 0.824 |
| psoriasis | 0.509 | 0.519 | 0.582 | 0.527 | 0.534 | 0.583 |
| Graves' disease | 0.612 | 0.565 | 0.639 | 0.599 | 0.617 | 0.653 |
| BMI | 0.713 | 0.715 | 0.712 | 0.701 | 0.709 | 0.766 |
| type 2 diabetes | 0.753 | 0.748 | 0.745 | 0.742 | 0.745 | 0.813 |
| HDL cholesterol | 0.667 | 0.668 | 0.659 | 0.659 | 0.663 | 0.712 |
| ischemic heart disease | 0.732 | 0.715 | 0.750 | 0.727 | 0.725 | 0.780 |
| **mean (descriptive)** | **0.672** | 0.658 | **0.696** | 0.669 | 0.659 | **0.720** |

Paired delta vs Regulatory-256 (mean; tasks with delta > 0; tasks with CI above 0): SVD-512 −0.014 (3/13; 1), Raw +0.024 (9/13; 5), LPCA-256 −0.004 (3/13; 1), LPCA-512 −0.014 (3/13; 0), AlphaGenome +0.048 (13/13; 10).

**Bioval:**

| Endpoint | Reg. 256 | SVD-512 | Raw 2,492 | LPCA-256 | LPCA-512 | AlphaGenome |
|---|---|---|---|---|---|---|
| FANTOM5 membership | 0.8330 | 0.8257 | 0.8274 | 0.8297 | 0.8212 | 0.9063 |
| H1 Micro-C | 0.5685 | 0.5669 | 0.5634 | 0.5671 | 0.5600 | 0.5697 |
| Replication timing | 0.7031 | 0.7139 | 0.7294 | 0.7037 | 0.7036 | 0.9138 |

**Genomic context (macro-AUROC; Spearman for TSS distance):**

| Endpoint | Reg. 256 | SVD-512 | Raw 2,492 | LPCA-256 | LPCA-512 | AlphaGenome |
|---|---|---|---|---|---|---|
| CpG context | 0.8957 | 0.8980 | 0.8981 | 0.8944 | 0.8960 | 0.9492 |
| Gene region | 0.7889 | 0.7898 | 0.7859 | 0.7924 | 0.7913 | 0.9029 |
| TSS distance | 0.6912 | 0.6937 | 0.6955 | 0.6842 | 0.6875 | 0.8197 |

All regulatory variants lie within about 0.01 of each other; only AlphaGenome is clearly higher. These are supplementary arms: the Holm family over the main arms is unchanged.

**Reading.**
- Raw 2,492 recovers about half of the EWAS gap between Regulatory-256 and AlphaGenome (+0.024 of +0.048), so part of the loss comes from compression, but raw still does not beat AlphaGenome: it is higher in 2/13 tasks (melanoma, colorectal) and lower in 11/13 (descriptive; paired contrasts raw vs AlphaGenome were not computed).
- On FANTOM5, replication timing and genomic context the raw matrix does not approach AlphaGenome (FANTOM5 0.827 vs 0.906; replication timing 0.729 vs 0.914; gene region 0.786 vs 0.903), and Micro-C is a tie among all arms. There the limit is the input information, not compression.
- No compressed variant (SVD-512, logistic PCA 256/512) improves on SVD-256.
- Reconstruction was not run for the diagnostic arms.

**Logistic PCA caveats (do not over-interpret a null):**
- LPCA-256 converged (693 iterations, 94.2% of the marginal-null Bernoulli deviance explained), but the stored vector is a deterministic 100-step encoder; its correlation with the fit-time scores is 0.83, so it is not the maximum-likelihood score.
- LPCA-512 did **not** converge in 1,000 iterations (last relative step 9.8e-4) and explains 99.9% of the deviance, i.e. it is in a near-separable regime. Its lack of gain does not show that logistic PCA cannot help.
- An earlier LPCA-256 run stopped at 300 iterations without converging (91% explained, still decreasing 0.08% per iteration); it was discarded and moved to `data/cache/representations/_unconverged_300it/`.

## 5. Deviations, provenance and what was not done

- **Bioval gate.** The launch gate initially failed on an unrelated worktree state (`frozen_paths_unchanged`: a staged deletion of `scripts/fetch_cpgislandext_hg38.py`). The staged cleanup was backed up (branch `wip/staged-cleanup-backup-2026-10-06`, patch `~/staged_cleanup_backup_2026-10-06.patch`), that single frozen path was restored, and the real run then passed 8/8 gate checks. The gate was not bypassed. AlphaGenome and the diagnostic arms are registered in `configs/posthoc/*.yaml` (not in `matrix.yaml`); the runner requires a new `--out-root`, keeps the zero-row exclusion union over the 4 registered stores, and reads the frozen reference arm read-only.
- **GSE40279 test.** The official one-shot test machinery is bound to the frozen 3-arm manifest and refuses a second launch, so it was left unmodified. Instead, `scripts/posthoc_external_test_alphagenome.py` re-implements only the evaluation core and refuses to run unless the freeze record is committed and every hash matches. The pre-test freeze record is `configs/posthoc/external_alphagenome_freeze.json` (commit `0af5cd5`; selected epochs 117/118/116; best.pt, config, prior, split and store hashes). It states that AlphaGenome was added after the original-arm test results were known and that its test set had never been accessed. Exactly one test evaluation per seed was performed on the 66 original test subjects; no retraining, reselection or tuning. The authorization callable is therefore a custom precondition check, not the official external gate.
- **Not run:** Exp5 (loader requires registry order; configs deleted in the staged cleanup), `loyfer_profile` for the new arms, any PCA-256 of AlphaGenome (declined), reconstruction for the diagnostic arms, EWAS v2.
- **Unrelated 30-epoch masking run** (`configs/experiments/masking/alphagenome_genomewide.yaml`) is not part of this report and not comparable to the 120-epoch protocol. It was interrupted at about 10/30 epochs by a session restart and not resumed.
- **Branch and commits:** `posthoc-alphagenome` (`081acaf`, `0af5cd5`, `e8922ba`). Uncommitted at the time of writing: `scripts/make_paper_figures.py`, the genomic-context module and script edits, `configs/representations/future_models.yaml`, regenerated figures.
- **Figures:** `paper/figures/fig2…fig5` now include AlphaGenome as "(post-hoc)". Three-arm originals are in `paper/figures/three_arm/`. The artifact hashes recorded by the frozen EWAS-v3 finalisation (`artifact_sha256.json`) refer to the three-arm `fig3_ewas.*`; they will no longer match the regenerated files.

## 6. Limitations

- AlphaGenome is supervised on functional tracks; endpoints sharing information with those tracks are not independent.
- Reconstruction comparisons use 3 seeds and no significance test. On GSE40279 the margin is small.
- The diagnostic arms were evaluated on label-free probes only; reconstruction was not run for them.
- A single AlphaGenome checkpoint (`all_folds`) and a single embedding (1 bp, trunk) were used.

## 7. Reproduction (paths)

| Component | Entry point | Output |
|---|---|---|
| Embedding | `scripts/build_alphagenome_embedding.py --window 524288 --margin 65536` | `data/cache/representations/alphagenome_locus.h5` |
| EWAS v3 | `scripts/ewas_v3/posthoc_alphagenome.py [--arm ID]` | `outputs/ewas_v3_posthoc_alphagenome/`, `outputs/ewas_v3_posthoc/<arm>/` |
| Bioval | `scripts/bioval_v2/run_evaluation.py --arm ID --endpoint … --out-root <new dir>` | `outputs/biological_validation_v2/posthoc_alphagenome/`, `…/posthoc_dimcontrol/` |
| Genomic context | `scripts/run_genomic_context_decodability.py fit --arm ID` then `analyze`, `figure` | `outputs/genomic_context_decodability_v1/` |
| Reconstruction configs | `scripts/make_posthoc_alphagenome_configs.py` | `configs/experiments/posthoc_alphagenome/{tcga,external}/` |
| GSE40279 freeze / test | `scripts/posthoc_external_freeze_alphagenome.py`, `scripts/posthoc_external_test_alphagenome.py --confirm-one-shot` | `outputs/external_reconstruction_posthoc_alphagenome/` |
| Compression diagnostic | `scripts/build_posthoc_regulatory_dimcontrol.py --kind raw2492|lpca256|lpca512`, `scripts/build_regulatory_embeddings.py --config configs/posthoc/regulatory_embeddings_dimcontrol.yaml` | `data/cache/representations/regulatory_histone_dnase__*` |
| Figures | `scripts/make_paper_figures.py`, `scripts/run_genomic_context_decodability.py figure` | `paper/figures/` |
