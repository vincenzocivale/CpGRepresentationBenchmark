# MethylGPT locus embedding extraction

`extract_cpg_embeddings.py` is a small, single-purpose MethylGPT adapter
(unmodified from the upstream-provided implementation). It looks up static,
pretrained CpG token vectors by Illumina probe ID (`"cg00000109"`, ...) from a
MethylGPT checkpoint's `encoder.embedding.weight` table. **No sample
methylation values are consumed and no forward pass occurs** -- these are the
same probe vectors regardless of sample, which is exactly the patient-agnostic
contract this benchmark requires (see CLAUDE.md). It has no dependency on this
repo's `cpg_repr_benchmark` package.

`scripts/build_methylgpt_embedding.py` (one level up) is the actual
integration point: it reads this repo's CpG registry (`cpg_idx, chr, pos`),
joins it to a probe_id/chr/pos crosswalk to recover each locus's Illumina
probe ID, calls `extract_cpg_embeddings`, and writes the canonical
`/cpg_idx` + `/embedding` HDF5 store that
`configs/representations/future_models.yaml`'s `methylgpt_locus_native`
entry points at.

## Why a coordinate -> probe_id crosswalk is needed

Unlike the sequence-based genomic-FM adapters (CpGPT, DeepCpG), MethylGPT's
vocabulary is keyed by Illumina array probe ID, not genomic coordinate. This
repo's registries are coordinate-native (`docs/COORDINATE_NATIVE_DATA.md`), so
`build_methylgpt_embedding.py` joins the registry's `(chr, pos)` against
`data/processed/ComputAgeBench/illumina_probe_grch38.parquet` (columns
`[probe_id, chr, pos]`, already present in this repo) to recover each locus's
probe ID before calling the extractor. A registry locus whose coordinate has
no probe_id in the crosswalk is dropped, and the script reports/refuses this
by default -- pass `--allow-partial-coverage` to proceed with reduced
coverage instead (see CLAUDE.md: "never build a per-representation
intersection" silently).

## Why no separate environment is required

Unlike CpGPT/DeepCpG, this extractor only needs `torch` and `numpy` to read a
raw state dict -- it never imports the MethylGPT package itself. Any
environment with `torch`, `numpy`, `pandas`, and `h5py` (the `cpg-repr-
benchmark` conda env already has all four) can run
`scripts/build_methylgpt_embedding.py` directly; no pinned sibling
environment or editable install is needed.

## Resources

You need only two files, both from the pinned MethylGPT release (see
upstream links in `extract_cpg_embeddings.py`'s docstring / module
constants):

1. A checkpoint containing `encoder.embedding.weight` (e.g. the published
   tiny checkpoint `tiny-best_model_epoch10.pt`, or a larger release
   checkpoint if available).
2. Its matching ordered probe CSV with an `illumina_probe_id` column (e.g.
   the published `probe_ids_type3.csv`).

Stage them locally, e.g.:

```bash
mkdir -p dependencies/methylgpt
# place/download tiny-best_model_epoch10.pt and probe_ids_type3.csv there,
# or let build_methylgpt_embedding.py --autodownload fetch the published
# tiny checkpoint/CSV to those exact paths.
```

`dependencies/` here is git-ignored the same way `data/cache/` is; it is a
local staging area, not something to commit.

## Building the store

```bash
conda activate cpg-repr-benchmark
python scripts/build_methylgpt_embedding.py \
    --registry data/cpg/registries/array_cpg_map.parquet \
    --probe-crosswalk data/processed/ComputAgeBench/illumina_probe_grch38.parquet \
    --checkpoint-path dependencies/methylgpt/tiny-best_model_epoch10.pt \
    --probe-ids-csv dependencies/methylgpt/probe_ids_type3.csv \
    --output data/cache/representations/methylgpt_locus.h5 \
    --autodownload
```

Smoke-test first with `--limit 200` to confirm the crosswalk join and paths
are wired correctly before running the full genome-wide array universe.
Coverage (`n_loci` vs. `n_loci_dropped_no_probe_id`) is printed and written to
the output's `.json` sidecar; a real (non-tiny/non-test) checkpoint's
production run should be run without `--allow-partial-coverage` so any
coverage gap fails loudly rather than silently narrowing the benchmark's
locus universe.

## Coordinate contract

This repo's registries (e.g. `data/cpg/registries/array_cpg_map.parquet`)
store the 1-based position of the CpG cytosine (see
`docs/COORDINATE_NATIVE_DATA.md`), matching
`data/processed/ComputAgeBench/illumina_probe_grch38.parquet`'s `chr`/`pos`
convention, so the crosswalk join needs no coordinate transform (contrast
with CpGPT's zero-based interval-start keys; see `scripts/cpgpt/README.md`).
