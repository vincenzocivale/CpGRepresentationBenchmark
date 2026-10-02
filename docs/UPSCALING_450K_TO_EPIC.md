# 450K → EPIC upscaling task

**Question.** Does a patient-agnostic CpG locus representation help reconstruct the CpGs that only the
Illumina EPIC (v1) array measures (*EPIC-only* probes) from a profile that only contains the probes shared
with the 450K array?

The task is a practical demonstration of the biological usefulness of a locus representation: a
450K profile carries ~440k of the ~845k EPIC loci, and the model has to fill in the remaining ~404k. As in
the rest of the repository, **only the locus representation changes between arms** — model, optimiser
budget, data splits, masking/sampling scheme and evaluation are fixed (`docs/BENCHMARK_V2.md`).

Results: no separate results document is present in this repo (`UPSCALING_RESULTS_2026-09-29.md` was never written). Run outputs live under `outputs/upscale_450k_epic/`; aggregate them with `scripts/summarize_upscaling.py`.

## Design in one picture

```text
             TRAIN (simulated 450K)                         TEST (real pairs)
 EPIC cohort (ComputAgeBench, 18 blood studies)     same sample profiled on both arrays
   ├─ keep probes shared with 450K  (O, 439,760)  ─►  450K array, probes in O  ──► model ──► EPIC-only β̂
   └─ EPIC-only probes           (T, 404,095)  ─►  supervision      EPIC array, probes in T  = ground truth
```

* **Simulated 450K = EPIC restricted to O.** No 450K array is needed for training, so any EPIC cohort can be
  used. The catch: the simulation ignores the technical 450K-vs-EPIC difference on the shared probes; the real
  pairs measure how much that costs (`platform_concordance` in every `summary.json`: on the shared probes the
  two arrays differ by MAE ≈ 0.032, per-sample r ≈ 0.99).
* **Real pairs are test-only.** They are never used for fitting, priors, validation or checkpoint selection.

## Data

| Role | Source | Where | Built by |
|---|---|---|---|
| Locus universe (O, T) | probes of GPL13534 (450K) and GPL21145 (EPICv1), mapped to GRCh38 CpG loci | `data/prepared/paired_450k_epic/loci_v1.{npz,parquet}` | `scripts/data/prepare_paired_450k_epic.py` |
| Simulated training | ComputAgeBench EPICv1 studies (GPL21145, GPL23976) with measured EPIC-only probes | `data/prepared/epic_sim_training/{epic_blood.h5,samples.parquet}` | `scripts/data/prepare_epic_simulated_training.py` |
| Study split | study-disjoint train / validation / test-sim | `data/protocols/paired_450k_epic/study_split_seed18.json` | same |
| Real pairs (test) | GEO **GSE86833** (15 pairs: LNCaP, PrEC, fibroblasts CAF/NAF, Guthrie-card blood) and **GSE92580** (12 pairs: 9 fresh-frozen + 3 FFPE-REPLI-g paediatric brain tumours) | `data/raw/paired_450k_epic/` (series matrices), `data/prepared/paired_450k_epic/<GSE>.h5` | `prepare_paired_450k_epic.py` |
| Representation | functional-annotation PCA, SVD fitted on O and projected on T | `data/cache/representations/functional_annotations__pca_450k_fit_paired_v1.h5` | `scripts/export_functional_feature_store.py` + `scripts/build_functional_pca_projection.py` |

Rules that define the universe (frozen once, identical for every arm; never narrowed per representation):

1. **O = probes present on both arrays**, T = EPIC-only probes. 450K-only probes are unusable because EPIC
   training data do not contain them.
2. Probes are mapped to canonical `grch38_cpg_cytosine_1based_v1` loci with the ComputAgeBench crosswalk;
   loci hit by several probes are averaged; loci hit by both a shared and an EPIC-only probe count as observed (8).
3. Autosomes only (chr1–22): the functional feature store and every existing arm covers only those.
4. Loci that are not reference-genome CpGs in the functional store (SNP-altered/off-strand mappings:
   821 O + 525 T) are dropped **once, here**.
5. Only training studies in which ≥50 % of the EPIC-only probes were measured are kept (studies released
   pre-restricted to 450K probes, and EPICv2, are excluded): 18 studies, 1,272 samples, all blood.
   Split (seed 18, chosen on split sizes only): train 880 / validation 186 / test-sim 206 samples.
6. Missing values (NaN) are never imputed for the model; a missing observed probe is masked out of the set,
   a missing target is excluded from loss and metrics.

### Locus hold-out view

10 % of the EPIC-only loci (seed 17) are *never* supervised, never used to fit priors and never seen at
selection time; they get the global-mean prior. Each evaluation is reported for `seen_loci` (the 90 %) and
`heldout_loci`. This is the repository's *unseen locus* protocol (`docs/BENCHMARK_DESIGN.md`) and the
setting where a representation, unlike a per-locus parameter table, can in principle transfer.

## Model, training, evaluation

* Model: the repository's fixed `MaskedMethylomeReconstructor` (DeepSets patient encoder over
  (locus embedding, β-residual w.r.t. train prior) tokens → residual decoder `sigmoid(prior + Δ)`),
  `src/cpg_repr_benchmark/models/model.py`. Nothing representation-specific.
* Training (`src/cpg_repr_benchmark/upscaling/engine.py`): each step draws 16 training patients, a random
  16k–64k observed CpGs each as context and 4,096 measured EPIC-only CpGs as targets, MSE on β.
  12,000 steps, AdamW lr 3e-4 (warm-up + cosine), bf16, GPU-resident data. Checkpoint = lowest MSE on a
  fixed panel of the validation *studies*.
* Priors: per-locus mean β over **training samples only** (`upscaling/data.py::fit_priors`).
* Evaluation: full reconstruction of all 404k targets per sample from 65,536 random observed CpGs
  (average of 3 random panels); metrics in `upscaling/metrics.py`.

Hyper-parameters were chosen on validation studies only (grid in the results doc). Nothing was tuned on real pairs.

### Arms

| Arm | What differs | Config |
|---|---|---|
| `functional_annotations_pca_450k_fit` | 256-d unsupervised PCA of ENCODE-derived functional annotations (SVD fit on O only) | `functional_pca.yaml` |
| `control_random_stable` | fixed random 256-d vector per locus (identity code, no biology) | `random_stable.yaml` |
| `control_constant` | identical vector for all loci (no locus information) | `constant.yaml` |
| `baseline_prior_mean` | train-mean β per locus, no model | `baseline_prior_mean.yaml` |
| `baseline_ridge_pca` | linear PCA→ridge from observed β to target β; **no locus representation**, locus-specific weights, cannot predict unseen loci | `baseline_ridge_pca.yaml` |

Other catalogued representations (NTv3, CpGPT, DeepCpG, MethylGPT) are not run: their caches cover only the
408k TCGA-array loci, none of the EPIC-only loci. Per repository policy the task universe is not narrowed to
what they cover; adding them requires extracting them on the universe (`loci_v1.parquet`) and registering the
store in `configs/representations/`.

### Metrics (per eval set, per locus view)

| Metric | Meaning |
|---|---|
| `mse`, `mae`, `rmse` | β-scale errors over all measured (sample, locus) pairs |
| `skill_vs_prior` | `1 − MSE / MSE(train-mean prior)`; >0 = beats the mean |
| `sample_pearson` | mean over samples of Pearson r across target loci (dominated by locus means; the prior alone scores high) |
| `locus_pearson_variable` | mean across-sample Pearson r on the 10 % most variable target loci (train-sample std) — the between-sample signal the prior cannot explain |
| `skill_vs_prior_variable` | skill restricted to those variable loci |
| `ci95` (real sets) | percentile bootstrap over samples (200 resamples) |

Eval sets: `sim_test` (held-out EPIC studies, simulated protocol); real `real_GSE86833_blood` (5),
`real_GSE86833_cells` (10), `real_GSE92580_FF` (9), `real_GSE92580_FFPE` (3), `real_primary`
(= all GSE86833 + GSE92580 FF, 24), and the cohorts as a whole. Real pairs are few: treat real-set numbers as
effect sizes with wide intervals, not precise estimates.

### Known confounds (read before interpreting)

* **Tissue.** All training data are blood; real pairs are mostly cell lines and brain tumours. The train-mean
  prior is itself a poor predictor there (MSE 0.053 vs 0.010 on simulated blood), so real-vs-simulated
  differences mix *platform shift* and *tissue shift*. The 5 Guthrie-card blood pairs isolate the
  platform effect.
* Non-blood EPICv1 data already on disk (GSE240412 LNCaP/PrEC/MCF7/tumours) were **not** used for training:
  LNCaP/PrEC there and in GSE86833 are the same cell lines, which would leak the biological state into the test.

## Reproduce

```bash
# 1) data (one time; needs data/raw/paired_450k_epic series matrices, see the download URLs in the script docstring)
python scripts/data/prepare_paired_450k_epic.py
python scripts/data/prepare_epic_simulated_training.py
PYTHONPATH=src python scripts/export_functional_feature_store.py --methylpredictor-root ../MehylPredictor \
  --locus-store data/external/functional/locus_features_v1 \
  --cpg-registry data/prepared/paired_450k_epic/loci_registry_v1.parquet \
  --output data/derived/functional_annotations/reference_functional_features_paired_450k_epic_v1.h5
python scripts/build_functional_pca_projection.py \
  --input data/derived/functional_annotations/reference_functional_features_paired_450k_epic_v1.h5 \
  --fit-protocol data/prepared/paired_450k_epic/loci_v1.npz \
  --output data/cache/representations/functional_annotations__pca_450k_fit_paired_v1.h5

# 2) arms (seed = training.seed in the yaml)
CUDA_VISIBLE_DEVICES=0 python scripts/run_450k_to_epic.py --config configs/experiments/upscale_450k_epic/functional_pca.yaml

# 3) tables + figures
python scripts/summarize_upscaling.py
pytest tests/test_upscaling.py -q
```

The raw series matrices come from
`https://ftp.ncbi.nlm.nih.gov/geo/series/GSE86nnn/GSE86833/matrix/GSE86833-{GPL13534,GPL21145}_series_matrix.txt.gz`
and `.../GSE92nnn/GSE92580/matrix/GSE92580-{GPL13534,GPL21145}_series_matrix.txt.gz`.

## Output layout

```text
outputs/upscale_450k_epic/<dataset>/<arm>/<track>/seed_<seed>/<UTC>-<config hash>/
  resolved_config.yaml  experiment.json  summary.json  history.json  protocol_snapshot.npz
  checkpoints/{best,last}.pt                      (neural arms)
  evaluation/<eval set>/<view>/{metrics.json,per_sample.csv}
  evaluation/real_*/predictions.npz               (float16 prediction + target, per real set)
outputs/upscale_450k_epic/_summary/               (scripts/summarize_upscaling.py)
  runs_long.csv  table_seen_loci.md  table_heldout_loci.md  fig_skill_real.png  fig_scatter_real.png
outputs/_dev_archive/validation_grid/             (hyper-parameter selection runs, validation studies only)
```
