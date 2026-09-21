# GSE42861 disease classification benchmark protocol

> **Note:** `scripts/run_classification_benchmark.py` and the direct representation-conditioned classifier it drove have been removed. Downstream phenotype evaluation now goes through the reconstruction-to-downstream pipeline (sparse methylome -> reconstruction model -> frozen predictor); the dataset-preparation steps above are still current. See CLAUDE.md for the current protocol.

## Purpose

GSE42861 (Liu et al. 2013) is the rheumatoid-arthritis case/control cohort used as an
external-cohort validation dataset for the disease-classification track of the coordinate-native
CpG representation benchmark. TCGA (`tcga_array`) is the masking/reconstruction training source;
downstream disease classification (`disease_label`: 0 = control, 1 = rheumatoid arthritis) is
trained and evaluated only on GSE42861 samples (689 samples: 354 RA, 335 control; whole-blood/PBL,
Illumina 450k, GRCh38).

The locus analysis is split before representation coverage is considered:

- `shared`: GSE42861 loci also present in the TCGA-array training source;
- `external_locus`: GSE42861 loci absent from the TCGA-array training source;
- `all`: every mapped GSE42861 locus.

For the current master registry the expected counts are:

```text
all              485447
shared           408390
external_locus    77057
```

These are source-membership definitions, not representation intersections. Every
multi-representation experiment subsequently creates one common-universe protocol inside
one of these sets.

## 0. Prepare GSE42861 and register it in the master CpG registry

```bash
export CPGPT_ILLUMINA_DB=/path/to/cpgpt/dna_dependencies/illumina_metadata.db
python scripts/data/prepare_gse42861.py \
  --raw-dir data/raw/GSE42861 \
  --output-dir data/processed/GSE42861

python scripts/data/build_master_cpg_registry.py \
  --source tcga_array=data/cpg/registries/array_cpg_map.parquet \
  --source gse40279=data/processed/GSE40279/cpg_mapping.parquet \
  --source gse42861=data/processed/GSE42861/cpg_mapping.parquet
```

The registry rebuild is not incremental: re-list every source that must remain in scope (not
just the new one), or previously registered sources will silently drop out of the membership
table.

## 1. Build persistent transfer locus sets

```bash
python scripts/data/build_transfer_locus_protocols.py \
  --dataset-h5 data/processed/GSE42861/methylation.h5 \
  --membership data/cpg/master_cpg_membership.parquet \
  --dataset-source gse42861 \
  --training-source tcga_array \
  --output-dir data/protocols/GSE42861/transfer_vs_tcga
```

This creates `all.npz`, `shared.npz`, `external_locus.npz`, and `manifest.json`.

## 2. Materialize representations directly from coordinates

Historical atlases are not re-keyed for primary execution. Providers are run once over the
explicit locus protocol and their outputs are cached in the canonical coordinate namespace.

NTv3-pre:

```bash
python scripts/representations/materialize.py ntv3-pre \
  --locus-protocol data/protocols/GSE42861/transfer_vs_tcga/shared.npz \
  --output data/cache/representations/materialized/GSE42861/shared/ntv3_pre.h5 \
  --batch-size 4 \
  --fasta ../MehylPredictor/MethylPredictionData/reference/hg38/hg38.fa \
  --device cuda
```

Functional, native_frozen PCA (primary path; a compact unsupervised PCA/SVD embedding over the
raw ENCODE functional-annotation feature store, see `scripts/build_functional_pca_embedding.py`
and `docs/ADDING_REPRESENTATIONS.md`):

```bash
python scripts/build_functional_pca_embedding.py \
  --input data/derived/functional_annotations/<raw feature store>.h5 \
  --output data/cache/representations/materialized/GSE42861/shared/functional_pca.h5 \
  --n-components 256
```

This runs over the full raw feature store's CpG universe (it does not take a `--locus-protocol`
filter); downstream locus coverage against `shared.npz`/`external_locus.npz` is still enforced
by `build_common_locus_protocol.py` in step 3 below. It reads directly from the raw
functional-annotation feature store rather than going through `scripts/representations/materialize.py`.

Functional, legacy checkpoint (smoke-test only, not a publication arm):

```bash
python scripts/representations/materialize.py functional-legacy \
  --locus-protocol data/protocols/GSE42861/transfer_vs_tcga/shared.npz \
  --output data/cache/representations/materialized/GSE42861/shared/functional_legacy.h5 \
  --batch-size 4096 \
  --legacy-registry data/cpg/registries/array_cpg_map.parquet \
  --methylpredictor-root ../MehylPredictor \
  --checkpoint data/external/functional/checkpoints/functional_encoder.pt \
  --locus-store data/external/functional/locus_features_v1 \
  --device cuda
```

Repeat against `external_locus.npz` or `all.npz` when evaluating that view instead.

## 3. Materialize the exact common locus universe

```bash
python scripts/data/build_common_locus_protocol.py \
  --dataset-h5 data/processed/GSE42861/methylation.h5 \
  --candidate-protocol data/protocols/GSE42861/transfer_vs_tcga/shared.npz \
  --representation functional=data/cache/representations/materialized/GSE42861/shared/functional_pca.h5 \
  --representation ntv3=data/cache/representations/materialized/GSE42861/shared/ntv3_pre.h5 \
  --output data/protocols/GSE42861/disease_shared_functional_ntv3_common.npz
```

The adjacent JSON manifest records per-representation coverage and the final common count. Every
representation entered into a comparison table must be materialized against the same candidate
protocol and folded into the same common-universe call — never build a per-representation
intersection outside this step.

## 4. Register experiment configs and run

Copy `configs/experiments/classification/gse42861_disease_template.yaml` once per representation
arm, set `representation.*` and (if scoping to the common universe from step 3)
`dataset.locus_protocol`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_classification_benchmark.py \
  --config configs/experiments/classification/<your_config>.yaml \
  --mode all
```

All arms compared against each other must share: the same GSE42861 patients (same deterministic
seed), the same common CpGs (same `dataset.locus_protocol`), the same model, optimizer, and
training budget. Only `representation.*` may differ between arms.

## 5. Aggregate results

```bash
python scripts/summarize_runs.py --outputs outputs --csv outputs/gse42861_disease_summary.csv
```
