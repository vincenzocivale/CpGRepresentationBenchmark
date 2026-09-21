# Benchmark V2: representation-first experimental design

## Scientific unit of comparison

The benchmark compares **patient-agnostic CpG locus representations**. It must never pass a
competitor's patient methylation value, RNA state, task token, or final contextualized token as
the locus representation. For methylation foundation models, extract only the component that
identifies/represents the CpG locus before patient-specific value integration.

Every representation is reported under a single track:

- **`native_frozen`** — use the locus embedding as supplied by the published model/checkpoint,
  optionally compacted with an unsupervised transform (e.g. PCA/SVD over the raw ENCODE
  functional-annotation feature store — see `scripts/build_functional_pca_embedding.py`).

No representation is re-optimized against a methylation objective before being benchmarked; a
representation is compared purely on what its source model/feature store already encodes.

### Functional annotations

The old RNA->DNAm model is not retained. Only its CpG branch is migrated:

```text
reference functional track IDs --EmbeddingBag--+
                                                +--> norm -> residual FFN blocks -> 256-D CpG embedding
reference dense locus features -------MLP-------+
```

For the primary functional arm, direct DNA-methylation-derived annotation tracks must be excluded or explicitly audited. A secondary permissive arm may retain them, but it must not be presented as a clean reference-only functional comparison. The compact 256-D embedding above is obtained via unsupervised PCA/SVD (`scripts/build_functional_pca_embedding.py`), not via any methylation-supervised fitting.

## Downstream tasks

All downstream tasks use a fixed methylome backbone and swap only `representation.store_h5`.

### Masked methylation reconstruction

Keep the existing `MaskedMethylomeReconstructor` protocol. Report seen-locus and unseen-locus
views over the masking sweep.

### Age / mortality / disease

The shared patient encoder is:

```text
CpG locus embedding + observed methylation residual
        -> token
        -> DeepSets patient encoder
        -> fixed task head
```

The head is regression, binary, or multiclass according to the phenotype. For a specific task,
architecture, split, optimizer, CpG panel and training budget are identical across representations.

Sparsity is a first-class evaluation axis. Test every trained model at the same masking fractions,
including 0%, 15%, 30%, 50%, 70%, and 90%. Use identical deterministic masks across
representation arms via the shared seed/protocol.

## Canonical cache contracts

Downstream representation cache:

```text
/cpg_idx    int64   [N]
/embedding  float16 [N, D]
```

Self-contained functional source cache:

```text
/cpg_idx       int64 [N]
/track_indices int64 [nnz]
/track_indptr  int64 [N+1]
/dense         float  [N, 23]
```

The one-time migration script `scripts/export_functional_feature_store.py` is the only code path
allowed to depend on the old MehylPredictor repository. Once the source cache exists, all
representation compaction and downstream benchmarking are self-contained here.

## Output hierarchy

```text
outputs/
  masking/<dataset>/<representation>/<track>/seed_<s>/<run>/
  age/<dataset>/<representation>/<track>/seed_<s>/<run>/
  mortality/<dataset>/<representation>/<track>/seed_<s>/<run>/
  disease/<dataset>/<representation>/<track>/seed_<s>/<run>/
```

Each run keeps `resolved_config.yaml`, `representation_manifest.json`, `experiment.json`,
checkpoints, evaluation files and `summary.json`.

## Recommended paper matrix

For each source representation `R`:

| Source | Native frozen | Masking | Age | Mortality | Disease | Sparsity sweep |
|---|---:|---:|---:|---:|---:|---:|
| Functional annotations | yes | yes | yes | yes | yes | yes |
| genomic FM R | yes | yes | yes | yes | yes | yes |
| methylation FM locus component R | yes | yes | yes | yes | yes | yes |

Keep any **native full-model** CpGPT/MethylGPT evaluation in a separate complete-method table; it
answers a different question from locus representation quality.


## Normative coordinate identity

Publication-grade runs use the coordinate-native contract in
`docs/COORDINATE_NATIVE_DATA.md`.  The biological identity of a CpG is the GRCh38
coordinate of its cytosine (1-based), encoded deterministically as
`grch38_cpg_cytosine_1based_v1`.  MethylProphet/TCGA IDs remain supported only for
legacy migration.

External datasets are ingested at their full mappable locus coverage.  A fair multi-arm
comparison then materializes one persistent common-universe protocol (`D ∩ R1 ∩ ... ∩ Rn`)
with `scripts/data/build_common_locus_protocol.py`; no representation may silently shrink
the evaluation universe on its own.
