# GSE40279 age benchmark protocol

## Purpose

GSE40279 is the first external-cohort validation dataset for the coordinate-native
CpG representation benchmark.  Proxy supervision is sourced from TCGA; downstream age
prediction is trained and evaluated only on GSE40279 samples.

The locus analysis is split before representation coverage is considered:

- `shared`: GSE40279 loci also present in the TCGA-array proxy source;
- `external_locus`: GSE40279 loci absent from the TCGA-array proxy source;
- `all`: every mapped GSE40279 locus.

For the current master registry the expected counts are:

```text
all              472985
shared           408017
external_locus    64968
```

These are source-membership definitions, not representation intersections.  Every
multi-representation experiment subsequently creates one common-universe protocol inside
one of these sets.

## 1. Build persistent transfer locus sets

```bash
python scripts/data/build_transfer_locus_protocols.py \
  --dataset-h5 data/processed/GSE40279/methylation.h5 \
  --membership data/cpg/master_cpg_membership.parquet \
  --dataset-source gse40279 \
  --proxy-source tcga_array \
  --output-dir data/protocols/GSE40279/transfer_vs_tcga
```

This creates `all.npz`, `shared.npz`, `external_locus.npz`, and `manifest.json`.

## 2. Materialize chr1 smoke-test representations directly from coordinates

Historical atlases are not re-keyed for primary execution.  Expensive providers are run once
over the explicit locus protocol and their outputs are cached in the canonical coordinate namespace.

NTv3-pre:

```bash
python scripts/representations/materialize.py ntv3-pre \
  --locus-protocol data/protocols/GSE40279/transfer_vs_tcga/shared.npz \
  --chromosomes chr1 \
  --output data/cache/representations/materialized/GSE40279/shared_chr1/ntv3_pre.h5 \
  --batch-size 4 \
  --fasta ../MehylPredictor/MethylPredictionData/reference/hg38/hg38.fa \
  --device cuda
```

Historical Functional checkpoint, smoke-test only:

```bash
python scripts/representations/materialize.py functional-legacy \
  --locus-protocol data/protocols/GSE40279/transfer_vs_tcga/shared.npz \
  --chromosomes chr1 \
  --output data/cache/representations/materialized/GSE40279/shared_chr1/functional_legacy.h5 \
  --batch-size 4096 \
  --legacy-registry data/cpg/registries/array_cpg_map.parquet \
  --methylpredictor-root ../MehylPredictor \
  --checkpoint data/external/functional/checkpoints/functional_encoder.pt \
  --locus-store data/external/functional/locus_features_v1 \
  --device cuda
```

The NTv3 command performs direct inference with the pre-trained checkpoint.  The Functional legacy
command reuses the historical functional checkpoint only to validate the pipeline; it is not the
publication representation.

## 3. Materialize the exact common smoke universe

```bash
python scripts/data/build_common_locus_protocol.py \
  --dataset-h5 data/processed/GSE40279/methylation.h5 \
  --candidate-protocol data/protocols/GSE40279/transfer_vs_tcga/shared.npz \
  --representation functional=data/cache/representations/materialized/GSE40279/shared_chr1/functional_legacy.h5 \
  --representation ntv3=data/cache/representations/materialized/GSE40279/shared_chr1/ntv3_pre.h5 \
  --output data/protocols/GSE40279/smoke_chr1_functional_ntv3_common.npz
```

The adjacent JSON manifest records per-representation coverage and the final common count.

## 4. Run the age smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_classification_benchmark.py \
  --config configs/experiments/age/gse40279_chr1_functional_legacy_smoke.yaml \
  --mode all

CUDA_VISIBLE_DEVICES=0 python scripts/run_classification_benchmark.py \
  --config configs/experiments/age/gse40279_chr1_ntv3_pre_smoke.yaml \
  --mode all
```

Both arms use the same GSE40279 patients (same deterministic seed), same common CpGs,
same model, same optimizer, and the same sparsity sweep.  This smoke comparison validates
the infrastructure only: the inherited functional embedding is `legacy`, so its result is
not a final paper ranking against the native NTv3-pre representation.

## 5. Publication-grade locus-transfer experiments

After genome-wide `proxy_aligned` caches can be exported from the proxy checkpoint to
arbitrary master-registry loci, create three separate common protocols:

```text
shared          -> transfer_vs_tcga/shared.npz
external_locus  -> transfer_vs_tcga/external_locus.npz
all             -> transfer_vs_tcga/all.npz
```

For each setting, `build_common_locus_protocol.py --candidate-protocol ...` must include
all representations in the comparison.  The protocol is then fixed and reused across all
seeds and downstream arms.
