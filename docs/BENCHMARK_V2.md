# Benchmark V2: representation-first experimental design

## Scientific unit of comparison

The benchmark compares **patient-agnostic CpG locus representations**. It must never pass a
competitor's patient methylation value, RNA state, task token, or final contextualized token as
the locus representation. For methylation foundation models, extract only the component that
identifies/represents the CpG locus before patient-specific value integration.

Every representation is assigned to one of two tracks:

1. **`native_frozen`** — use the locus embedding as supplied by the published model/checkpoint.
2. **`proxy_aligned`** — start from the same patient-agnostic source and optimize an encoder on
   the exact same proxy objective used for the functional representation.

These tracks answer different questions and should be reported in separate main-table blocks.

## Proxy objective

The default proxy is **mean methylation per CpG over training patients**. The proxy encoder may
see only CpGs in the persistent downstream `train_locus` split. A deterministic subset of those
train loci is used for proxy checkpoint selection. Downstream held-out loci are never used for
proxy loss or model selection.

The resulting embedding is materialized once as a canonical HDF5 store and frozen for every
downstream task. This prevents task-specific fine-tuning from contaminating a representation
comparison.

### Functional annotations

The old RNA->DNAm model is not retained. Only its CpG branch is migrated:

```text
reference functional track IDs --EmbeddingBag--+
                                                +--> norm -> residual FFN blocks -> 256-D CpG embedding
reference dense locus features -------MLP-------+
                                                |
                                                +--> scalar mean-beta proxy head (training only)
```

After training, the scalar head is discarded. The 256-D locus embedding is cached.

For the primary functional arm, direct DNA-methylation-derived annotation tracks must be excluded or explicitly audited. A secondary permissive arm may retain them, but it must not be presented as a clean reference-only functional comparison.

### Dense competing embeddings

For sequence FMs and locus components extracted from methylation FMs:

```text
frozen raw CpG embedding -> LayerNorm -> Linear(256) -> residual FFN -> 256-D embedding
                                                           |
                                                           +--> scalar mean-beta head (training only)
```

This is the `proxy_aligned` comparison. The unmodified source embedding remains the separate
`native_frozen` comparison.

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
allowed to depend on the old MehylPredictor repository. Once the source cache exists, all proxy
training and downstream benchmarking are self-contained here.

## Output hierarchy

```text
outputs/
  proxy_mean_methylation/<dataset>/<representation>/<track>/seed_<s>/<run>/
  masking/<dataset>/<representation>/<track>/seed_<s>/<run>/
  age/<dataset>/<representation>/<track>/seed_<s>/<run>/
  mortality/<dataset>/<representation>/<track>/seed_<s>/<run>/
  disease/<dataset>/<representation>/<track>/seed_<s>/<run>/
```

Each run keeps `resolved_config.yaml`, `representation_manifest.json`, `experiment.json`,
checkpoints, evaluation files and `summary.json`.

## Recommended paper matrix

For each source representation `R`:

| Source | Native frozen | Same proxy objective | Masking | Age | Mortality | Disease | Sparsity sweep |
|---|---:|---:|---:|---:|---:|---:|---:|
| Functional annotations | n/a | yes | yes | yes | yes | yes | yes |
| genomic FM R | yes | yes | yes | yes | yes | yes | yes |
| methylation FM locus component R | yes | yes | yes | yes | yes | yes | yes |

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
