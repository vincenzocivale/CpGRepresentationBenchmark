# ComputAgeBench downstream protocol

## Scope

This repository uses the `benchmark` partition of [ComputAgeBench](https://huggingface.co/datasets/computage/computage_bench), not its healthy-only clock-training partition. It contains methylation profiles from 65 public GEO studies and labels healthy controls (HC) and aging-accelerating conditions (AACs). The source release is CC BY-SA 4.0; cite the dataset and follow its terms in every derived analysis.

The source stores one Parquet file per study, with Illumina probe IDs in rows and GEO sample IDs in columns. It must not be committed to this repository.

## Local acquisition and conversion

Download the benchmark partition only. This is approximately 20 GB before conversion.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="computage/computage_bench",
    repo_type="dataset",
    local_dir="data/raw/ComputAgeBench",
    allow_patterns=["computage_bench_meta.tsv", "data/benchmark/**"],
)
PY
```

Use a GRCh38 `probe_id,chr,pos` crosswalk. On this host the CpGPT Illumina database at
`/data2/home/vcivale/projects/methylation/methylation-fm-benchmark/data/cpgpt/dna_dependencies/illumina_metadata.db`
was used. Its `homo_sapiens` records are 0-based and are converted to the benchmark's
1-based cytosine coordinate by adding one to the position. The resulting crosswalk is local,
ignored data at `data/processed/ComputAgeBench/illumina_probe_grch38.parquet`.

```bash
PYTHONPATH=src python scripts/data/prepare_computagebench.py \
  --snapshot-dir data/raw/ComputAgeBench \
  --split benchmark \
  --probe-crosswalk data/processed/ComputAgeBench/illumina_probe_grch38.parquet \
  --output-dir data/processed/ComputAgeBench/benchmark
```

The converter retains the union of all mappable source loci. It does not intersect the data
with a representation vocabulary. Repeated source probes at the same coordinate are averaged
per sample; probes missing from the crosswalk are recorded in `cpg_mapping.parquet` and omitted
only because their biological coordinate is unknown.

## Materialized local release

The conversion executed on 2026-09-15 produced:

| Artifact | Result |
| --- | --- |
| studies | 65 |
| samples | 10,410 |
| coordinate-native loci | 900,260 |
| observed source probes | 900,449 |
| mappable-probe fraction | 0.9998589593 |
| age range | 0–99 years |

`methylation.h5` contains `/beta` (`float32`, 10,410 × 900,260), `/cpg_idx`,
`/sample_name`, `/chrom`, and `/pos`, plus the coordinate namespace attributes. The mapping,
phenotypes, QC summary, and provenance are written beside it. The official metadata contains
six GEO IDs that recur in two distinct studies; the canonical HDF5 sample key is therefore
`DatasetID:GEO_sample_ID`, while `source_sample_name` preserves the unmodified GEO ID.

The phenotype table exposes `age`, `condition`, `condition_class`,
`is_healthy_control` and `is_aging_accelerating_condition`. The latter is the positive label
for binary AAC-vs-HC classification.

## Representation-controlled downstream runs

First materialize every candidate locus representation over the selected ComputAgeBench
universe. Then persist the single common universe before any supervised run:

```bash
python scripts/data/build_common_locus_protocol.py \
  --dataset-h5 data/processed/ComputAgeBench/benchmark/methylation.h5 \
  --representation functional=data/cache/representations/materialized/ComputAgeBench/functional.h5 \
  --representation ntv3=data/cache/representations/materialized/ComputAgeBench/ntv3_pre.h5 \
  --output data/protocols/ComputAgeBench/benchmark_common_functional_ntv3.npz
```

Copy `configs/experiments/age/computagebench_template.yaml` once per representation arm and
point `dataset.locus_protocol` to that shared file. Never use a representation-specific
intersection, condition labels, or held-out samples while building a locus representation.
