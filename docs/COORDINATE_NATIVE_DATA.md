# Coordinate-native CpG data contract

## Principle

The benchmark's biological identity is a **GRCh38 CpG locus**, not a TCGA row,
MethylProphet ID, Illumina probe ID, or foundation-model token ID.

A locus is defined as:

- reference build: `GRCh38`;
- chromosome: canonical `chr1` ... `chr22`, `chrX`, `chrY`, `chrM`;
- position: **1-based coordinate of the cytosine in the CpG dinucleotide**.

The canonical namespace is `grch38_cpg_cytosine_1based_v1`.

## Deterministic `cpg_idx`

For efficient HDF5 indexing, coordinates are encoded reversibly as

```text
cpg_idx = chromosome_code * 1_000_000_000 + position
```

with chromosome codes 1..22, X=23, Y=24, M=25.  The coordinate remains the
primary scientific identity; `cpg_idx` is only its compact deterministic encoding.
No dataset or model is allowed to mint source-specific IDs in the final benchmark.

## Dataset contract

Processed methylation datasets contain at least:

```text
/beta         float32 [samples, loci]
/cpg_idx      int64   [loci]
/sample_name  string  [samples]
```

Coordinate-native datasets should additionally include `/chrom` and `/pos`, and
HDF5 attributes:

```text
reference_build = GRCh38
coordinate_convention = 1-based position of CpG cytosine
cpg_namespace = grch38_cpg_cytosine_1based_v1
```

Raw source identifiers and mapping decisions belong in auditable Parquet artifacts,
not in the biological identity itself.

## Representation contract

A representation cache must use the same canonical `cpg_idx` namespace.  Its minimal
contract remains:

```text
/cpg_idx
/embedding
```

Representation manifests must record the reference build, coordinate convention,
namespace, source checkpoint, pooling rule and representation track.

## Common-universe rule

Datasets are never destructively reduced to a particular model vocabulary during
preprocessing.  For a comparison among representations R1..Rn on dataset D, create
one persistent protocol

```text
U = D ∩ R1 ∩ ... ∩ Rn
```

and reuse that exact `cpg_idx` list for every arm.  Representation-specific silent
intersections are forbidden.

Use `scripts/data/build_common_locus_protocol.py` to materialize this protocol.

## GSE40279

`prepare_gse40279.py` preserves every probe that can be mapped unambiguously to an
hg38 coordinate.  The CpGPT Illumina dependency database is used only as a published
probe-to-coordinate crosswalk.  Its 0-based position is converted to the benchmark's
1-based cytosine convention by `+1`.

Multiple Illumina probes mapping to the same coordinate are collapsed to one locus
using sample-wise `nanmean`; the source-probe mapping and group size remain in
`cpg_mapping.parquet`.

GSE40279 therefore remains a full external cohort.  Any TCGA/representation
intersection is decided later by the experiment protocol, never during ingestion.

## ComputAgeBench

ComputAgeBench ships one Parquet matrix per GEO study, with probe IDs as rows and GEO
sample IDs as columns. `prepare_computagebench.py` converts a local snapshot to one
coordinate-native HDF5 matrix. It requires an explicit `probe_id,chr,pos` crosswalk in
GRCh38; source probe IDs are retained in `cpg_mapping.parquet`, while only the coordinate
encoding enters `/cpg_idx`.

```bash
python scripts/data/prepare_computagebench.py \
  --snapshot-dir data/raw/ComputAgeBench \
  --probe-crosswalk /path/to/illumina_probe_grch38.parquet \
  --split benchmark \
  --output-dir data/processed/ComputAgeBench/benchmark
```

The benchmark metadata is preserved as `phenotypes.parquet`. It exposes chronological
`age`, `condition`, `condition_class`, and binary `is_healthy_control` /
`is_aging_accelerating_condition`. These are
downstream labels only: they must never be used while materializing or fitting a locus
representation. Cite ComputAgeBench and comply with its CC BY-SA 4.0 terms when using the
derived data. HDF5 sample IDs are made globally unique as `DatasetID:GEO_sample_ID`; the
unmodified GEO ID is retained as `source_sample_name` in the phenotype artifact.

## Legacy MethylProphet artifacts

The existing `array_cpg_map.parquet`, `epic_cpg_map.parquet` and
`wgbs_cpg_map.parquet` remain useful **coordinate crosswalks and provenance sources**.
Their old `cpg_idx` values are not the canonical identity of the new benchmark.
Legacy runs remain reproducible, but publication-grade coordinate-native runs should
materialize methylation and representation axes directly in this namespace. Re-keying is retained only as a legacy migration utility.

## Initial setup commands

After preparing GSE40279:

```bash
export CPGPT_ILLUMINA_DB=/path/to/cpgpt/dna_dependencies/illumina_metadata.db
python scripts/data/prepare_gse40279.py \
  --raw-dir data/raw/GSE40279 \
  --output-dir data/processed/GSE40279
```

Build an auditable benchmark catalog from whichever sources are currently in scope:

```bash
python scripts/data/build_master_cpg_registry.py \
  --source tcga_array=data/cpg/registries/array_cpg_map.parquet \
  --source tcga_epic=data/cpg/registries/epic_cpg_map.parquet \
  --source gse40279=data/processed/GSE40279/cpg_mapping.parquet
```

The old source `cpg_idx` values in TCGA registries are ignored here; only `chr,pos` are used.

Primary runs should infer representations directly from coordinates with
`scripts/representations/materialize.py`; this avoids inheriting source-specific atlas IDs and
materializes only the loci the protocol actually needs.  `rekey_representation_h5.py` remains
available for debugging historical caches, but it is not the primary execution path.

Before comparing several representations on an external dataset, persist the exact intersection once:

```bash
python scripts/data/build_common_locus_protocol.py \
  --dataset-h5 data/processed/GSE40279/methylation.h5 \
  --representation functional=data/cache/representations/functional.coordinate.h5 \
  --representation ntv3=data/cache/representations/ntv3_pre.coordinate.h5 \
  --output data/protocols/GSE40279/age_common_functional_ntv3.npz
```

## External-locus transfer protocols

When a representation's training/fitting source comes from one dataset (for example TCGA Array)
and a downstream cohort contains additional loci (for example GSE40279), first materialize
source-membership sets with `build_transfer_locus_protocols.py`.  This separates the biological
question ("was this locus available to the representation's training source?") from
representation coverage.  Only then intersect a chosen set with every representation using
`build_common_locus_protocol.py --candidate-protocol ...`.

For the current TCGA Array / GSE40279 master registry the split is 408,017 shared loci and
64,968 GSE40279 loci external to TCGA Array, for 472,985 downstream loci in total.
