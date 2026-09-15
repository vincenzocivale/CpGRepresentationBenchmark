# GSE147221 schizophrenia disease-classification protocol

## Purpose and cohort definition

GSE147221 is an external whole-blood/buffy-coat Illumina HumanMethylation450
case-control cohort for schizophrenia. The disease task is binary
`disease_label`: `0 = control`, `1 = schizophrenia`. The GEO series contains
720 profiles: 364 cases, 349 controls and seven fully methylated technical
controls. The benchmark input excludes all technical controls (`status: NA`) and
the 34 samples marked by GEO as excluded for low quality, leaving 679 biological
samples (348 schizophrenia, 331 control).

The series matrix supplies the phenotype metadata. Beta values and detection P
values come from the GEO processed-signals supplement; the series matrix itself
does not contain a beta matrix. The preparer retains a beta only when its paired
detection P value is at most 0.01.

## 1. Download and prepare

```bash
mkdir -p data/raw/GSE147221
curl -L --fail --retry 5 -o data/raw/GSE147221/GSE147221_series_matrix.txt.gz \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE147nnn/GSE147221/matrix/GSE147221_series_matrix.txt.gz
curl -L --fail --retry 5 -o data/raw/GSE147221/GSE147221_Dublin_blood_processed_signals.csv.gz \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE147nnn/GSE147221/suppl/GSE147221_Dublin_blood_processed_signals.csv.gz

python scripts/data/prepare_gse147221.py \
  --raw-dir data/raw/GSE147221 \
  --output-dir data/processed/GSE147221 \
  --illumina-db /path/to/cpgpt/dna_dependencies/illumina_metadata.db
```

The output contract is:

```text
data/processed/GSE147221/
├── methylation.h5       # /beta [sample, CpG], /cpg_idx, /sample_name
├── phenotypes.parquet   # includes array_id, included and disease_label
├── cpg_mapping.parquet  # source probe to GRCh38 coordinate mapping
├── qc.json
└── manifest.json
```

`/sample_name` stores Illumina array barcodes rather than GEO GSM accessions;
this is intentional because the signals supplement is keyed by barcode. The
phenotype table preserves both `sample_name` (GSM accession) and `array_id`, and
the experiment configuration must use `phenotype.sample_column: array_id`.

## 2. Add to the master registry and create transfer sets

Rebuild the registry with every existing source that should remain represented:

```bash
python scripts/data/build_master_cpg_registry.py \
  --source tcga_array=data/cpg/registries/array_cpg_map.parquet \
  --source gse40279=data/processed/GSE40279/cpg_mapping.parquet \
  --source gse42861=data/processed/GSE42861/cpg_mapping.parquet \
  --source gse147221=data/processed/GSE147221/cpg_mapping.parquet

python scripts/data/build_transfer_locus_protocols.py \
  --dataset-h5 data/processed/GSE147221/methylation.h5 \
  --membership data/cpg/master_cpg_membership.parquet \
  --dataset-source gse147221 \
  --proxy-source tcga_array \
  --output-dir data/protocols/GSE147221/transfer_vs_tcga
```

The resulting `shared.npz`, `external_locus.npz` and `all.npz` define source
membership only. They are not representation-specific intersections.

## 3. Materialize a common representation universe and run

Materialize each representation against the same chosen transfer set, then use
`build_common_locus_protocol.py` to create one common locus protocol across all
arms. Set that file as `dataset.locus_protocol` in copies of
`configs/experiments/classification/gse147221_schizophrenia_template.yaml`.
Only the `representation.*` fields may differ between arms.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_classification_benchmark.py \
  --config configs/experiments/classification/<your_gse147221_arm>.yaml \
  --mode all
```

All compared arms must use the same deterministic patient split, common CpG
universe, model/training parameters and mask sweep. The dataset was downloaded
from [NCBI GEO GSE147221](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE147221);
cite the associated study when reporting results.
