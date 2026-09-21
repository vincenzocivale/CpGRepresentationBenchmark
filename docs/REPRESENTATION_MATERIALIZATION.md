# Coordinate-native representation materialization

## Rule

Primary benchmark runs do not re-key historical representation atlases and do not run genomic
foundation models inside downstream training.  A locus protocol is resolved first, then each
representation provider performs inference once and writes a frozen canonical cache.

```text
locus protocol (GRCh38 canonical cpg_idx)
        -> coordinate provider
        -> frozen HDF5
        -> common-universe protocol
        -> downstream model
```

Canonical cache contract:

```text
/cpg_idx    int64 [N]     # grch38_cpg_cytosine_1based_v1
/chrom      bytes [N]
/pos        int64 [N]     # 1-based CpG cytosine
/embedding  float16/32 [N,D]
```

The HDF5 attributes record reference build, coordinate convention, provider, checkpoint/pooling
metadata and a materialization fingerprint.  Interrupted jobs leave `<output>.partial`; rerunning
the same command resumes only incomplete rows.  Changing provider configuration or locus protocol
requires a new output path or `--force`.

## Providers

### NTv3-pre

`ntv3-pre` performs direct reference-genome inference with
`InstaDeepAI/NTv3_650M_pre`.  The default contract is a 32,768-bp centred GRCh38 window,
forward orientation, final decoder representation and mean pooling of the two output bins covering
the central C/G.  This is a `native_frozen` representation.

### Historical Functional checkpoint

`functional-legacy` is retained only for pipeline smoke/regression tests.  It resolves canonical
coordinates through the historical array registry and invokes the old functional checkpoint.  It
cannot support strict external-locus claims because coordinates absent from that legacy registry
are rejected explicitly.

### Native-frozen Functional (PCA-compacted)

`scripts/build_functional_pca_embedding.py` is the publication path. It consumes a
**coordinate-native raw functional feature HDF5** and compacts it with an unsupervised PCA/SVD
transform — no methylation-supervised fitting. The resulting embedding covers every locus in
the raw functional feature store, including GSE40279-only loci.

## GSE40279 chr1 smoke

Materialize the two representations directly over the persistent `shared` protocol restricted to
chr1.  No old NTv3 atlas is used.

```bash
python scripts/representations/materialize.py ntv3-pre \
  --locus-protocol data/protocols/GSE40279/transfer_vs_tcga/shared.npz \
  --chromosomes chr1 \
  --output data/cache/representations/materialized/GSE40279/shared_chr1/ntv3_pre.h5 \
  --batch-size 4 \
  --fasta ../MehylPredictor/MethylPredictionData/reference/hg38/hg38.fa \
  --device cuda

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

Then freeze the exact comparison universe:

```bash
python scripts/data/build_common_locus_protocol.py \
  --dataset-h5 data/processed/GSE40279/methylation.h5 \
  --candidate-protocol data/protocols/GSE40279/transfer_vs_tcga/shared.npz \
  --representation functional=data/cache/representations/materialized/GSE40279/shared_chr1/functional_legacy.h5 \
  --representation ntv3=data/cache/representations/materialized/GSE40279/shared_chr1/ntv3_pre.h5 \
  --output data/protocols/GSE40279/smoke_chr1_functional_ntv3_common.npz
```

The resulting smoke comparison is not a publication ranking because the historical Functional arm
is `legacy`.  Its purpose is to validate the complete coordinate-native pipeline before exporting
the clean `native_frozen` PCA-compacted Functional representation.
