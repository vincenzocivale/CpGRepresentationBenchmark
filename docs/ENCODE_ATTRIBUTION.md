# ENCODE feature attribution campaign

Identify which original annotations make a CpG representation useful, which groups are
redundant, and where their reconstruction benefit occurs. No foundation model is trained
or fine-tuned. Historical benchmark outputs are not confirmation for new hypotheses.
The [completed focused results and paper decisions](ENCODE_RESULTS_2026-09-29.md)
record the final comparisons. The earlier [discovery snapshot](ENCODE_RESULTS_2026-09-26.md)
preserves the provisional state and hypothesis choices.

## Run and resume

From the repository root, using the existing environment:

```bash
conda activate cpg-repr-benchmark
export PYTHONPATH=src
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
python scripts/run_encode_atlas.py prepare
python scripts/run_encode_atlas.py screen --max-jobs 3
python scripts/run_encode_atlas.py run --only full --pilot --max-jobs 1
python scripts/run_encode_atlas.py pipeline
```

The default pipeline runs all biological screening, redundancy/FM controls and tissue
profiling, then freezes a 13-arm paper-focused plan before evaluating test patients.
It runs those discovery and three-seed confirmation decoders, four chromosome-held-out
arms, focused co-methylation, matched EWAS and reporting. The historical exhaustive
pipeline remains in code for reproducibility but is no longer the default. GPU jobs wait for 20 GiB free on
GPU 0, checking every 60 seconds; CPU screening uses four BLAS threads. Other processes
are never stopped. A file lock prevents concurrent writers to one campaign. Completed
groups/jobs are reused; interrupted decoder training restarts rather than checkpoint-resumes.

Commands: `prepare`, `screen`, `augment`, `tissue`, `select`, `run`, `biology`, `ewas`,
`report`, `pipeline`. `run --dry-run` lists tasks without training/materialization.
`run --stage confirm --ood` runs genomic transfer. `--only` takes exact arm names from
`arms.json`, e.g. `drop/assay/DNase-seq`. `--max-jobs` bounds new jobs, not cached jobs.
`report` can run between stages. `pilot` outputs are implementation/cost diagnostics.
`focused_plan.json` freezes the exact arm list; `pipeline_focused/*.json` records stage
completion. The plan includes histone and non-histone control subsets with equal track
counts within each 1%-wide peak-prevalence bin, sampled using annotations only.
This common-support comparison excludes high-prevalence histone tracks for which no
non-histone match exists; the exclusion count is recorded in the plan.
The final focused outputs are `FOCUSED_REPORT.md`, `focused_contrasts.csv`, the generic
paired comparison tables and `context_gains.csv`. Compact panels remain exploratory
and are not part of the focused confirmation set.

`configs/encode_atlas.yaml` is frozen at preparation with content hashes of original
features, metadata, registry, methylation and template. Change the output directory when
changing inputs or protocol. Timing is recorded per group and decoder. The full campaign
is large and cannot be promised to finish in a fixed number of days on a shared GPU.

## Audit and arms

The source contains 408,399 CpGs, 4,165 binary peak-overlap tracks and 23 dense features:
four CpG contexts, four gene regions, nine cCRE classes, TSS distance, and five breadths.
**CSR columns follow catalog row order, not the non-contiguous original `track_index`.**
Preparation verifies the complete dataset/registry axis, bounds and duplicates, and
reconstructs all five breadths. Source namespace attributes are not trusted or edited.

Decoder arms include full, null, genomic-context-only, breadth-only, tracks-only, balanced
scaling, alone/drop/add-context per assay and biosample class, dense-block removal, all
assay pairs, and equal-assay-count controls. Biological screening includes every target,
biosample and functional block: 997 groups with the present catalog and context baseline.
Here `context_only` uses dense columns 0–17: four CpG-context labels, four gene-region
labels, nine cCRE labels and TSS distance. It is an annotation baseline, not a
sequence-only representation. `full` additionally contains five breadth features.

`augment` adds discovery correlation clusters (absolute r >= 0.9), cluster-removal and
representative controls, and existing genome-wide CpGPT small/large and DeepCpG embeddings
alone/combined with ENCODE. Limited-vocabulary FMs are not silently intersected into this
universe. These common-width arms are separate from the historical native-width benchmark.

Track removal recomputes breadth using retained tracks and retained denominators; absent
assays contribute zero. Parallel `drop_preserve_breadth` arms measure identity conditional
on original breadth. cCRE summaries remain in conditional full-model assay ablations;
tracks-only and cCRE-removal controls expose their contribution. One-hot labels are grouped.

All embeddings have width 256. Smaller groups are standardized and zero-padded; larger
groups use TruncatedSVD. Null embeddings are zero. Persisted transforms include scaling,
fit rows, components and original feature selections. Legacy scaling reproduces the ENCODE
builder; balanced scaling additionally normalizes blocks by dimension. Frozen FM coordinates
are standardized as dense covariates. Equal adapter size does not imply equal information rank.

## Discovery and confirmation

Patient identities are partitioned once at seed 20260925. TCGA aliquots share a partition.
Training seeds 17/42/97 do not change patients or masks (mask seed 17001). The new partition
may overlap historical training; it is not a historically untouched external cohort.

Discovery means and variances use training patients only. Screening uniformly samples
12,000 CpGs from chromosomes 1–19 and uses all discovery patients. Loci with fewer than
max(20, 50% of patients) observations are ineligible for property probes. Ridge probes use
outer and inner chromosome GroupKFold, with alpha {0.1, 10, 1000}; scaling is fit within
each training fold. Per-track associations cross-fit context adjustment, with/without breadth;
these descriptive correlations do not receive iid-locus p-values.

Exploratory selection requires the entire screen, including null/unsupported groups. Up to four histone
targets, four TF targets and four biosamples are selected by incremental variance-probe MSE
benefit over context; sufficiency fills slots lacking positive benefit. The default
focused plan keeps H3K4me3 only if this rule selects it. The broad all-pair/FM plan is
recorded during selection but narrowed before any confirmation job. The focused
test-set arm list and chromosome-transfer subset are frozen in `focused_plan.json`.

Compact panels contain 8/16/32/64/128/256 tracks. Discovery variance associations drive
greedy selection penalized by 0.5 times maximum overlap correlation; perfect duplicates are
excluded. This is a heuristic, not proof of optimality. Random controls round-robin across
assay × biosample class at three seeds.

Discovery decoders evaluate validation patients, not test. The default pipeline runs
the focused discovery arms; already completed broader discovery arms remain in the
outputs. Confirmation uses the frozen
test and three model seeds. Primary endpoint: MSE at 50% masking; all five fractions are
saved. Four deterministic panels per patient improve locus support but do not guarantee
dense individual-CpG coverage. Reports retain support counts; increase repeats in a new
campaign if greater locus-level precision is required.

Genomic-transfer confirmation excludes chromosomes 20–22 from compression, decoder training
and locus-specific priors. Held-out loci use the global train-patients × train-loci prior.
No claim is made that external FM pretraining excluded these chromosomes.

## Biological follow-up

Tissue labels come from `cancer_type.parquet` in the HDF5's documented raw source, joined
by exact sample identity. `NA` remains unknown. Cancer contrasts require 30 discovery
samples. Exact organ-name ENCODE tissue tracks are compared with other tissue tracks
matched by assay and prevalence decile; cell-line origins are not guessed.

Co-methylation uses distant neighbors (>=1 Mb or interchromosomal) and context/variance/
distance-matched controls on held-out patients; it is descriptive. EWAS uses available
known sets with chromosome/context/gene-region/breadth-matched backgrounds and five
chromosome folds. Unlisted CpGs are not proven negative. Report AP at matched prevalence,
not population prevalence. This secondary analysis uses label-free transductive embeddings.

## Interfaces, reports and validation

The masking CLI adds optional `dataset.patient_protocol`, `dataset.patient_split_seed`,
`evaluation.mask_seed`, `evaluation.patient_view` (`validation` or `test`) and
`evaluation.panel_repeats`. Defaults preserve prior behavior. Predictions add `panel_repeat`;
sample and locus indices remain joinable through the frozen protocols. The embedding
contract remains `/cpg_idx` + `/embedding`.

Reports include all effects, biological probes, context gains, descriptive interaction
contrasts and PDF/PNG figures. Paired bootstrap independently resamples patient identities
and 1 Mb blocks while retaining observation weights. Identical patients/loci/observations
are required. BH adjustment is per family, stage, view and seed. Seeds are not biological
replicates. A compact panel passes only if every seed's upper 95% relative MSE-loss bound
is below 1%; absent passing panels remain explicit.
`context_gains.csv` also stratifies paired errors by quartiles of discovery-patient
mean beta, variance and missingness on the sampled chr1–19 loci. Quantile edges and
eligible-locus counts are saved in `strata_definitions.json`; these strata localize
effects and are not independent outcomes.

Tests cover column alignment, derived-feature removal, fit-only transformations, redundant
feature selection, signal/null recovery, patient leakage, bootstrap and actual CLI training
with identical panels across seeds. Input-context prediction and predictive ablations do
not establish causal biological mechanisms.
