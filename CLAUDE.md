# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this benchmark measures

This repo controls for everything except the **patient-agnostic CpG locus representation**. The downstream
reconstruction/prediction model, patient split, locus split, masking sweep, optimizer budget, and prior policy
are held fixed; only `representation.store_h5` (or online provider) changes between arms. Never let a
representation's extraction pipeline leak patient-specific values, RNA state, or task tokens into the locus
embedding — see `docs/BENCHMARK_V2.md` for the full rationale and `docs/ADDING_REPRESENTATIONS.md` before
wiring in a new representation.

Every representation reports under the `native_frozen` track — embedding exactly as extracted
from the published model/checkpoint, or compacted unsupervised from it (e.g. PCA/SVD over the
raw ENCODE functional-annotation feature store, see `scripts/build_functional_pca_embedding.py`).
There is no proxy-training track: representations are never re-optimized against a methylation
objective before being benchmarked.

## Commands

```bash
# install (env is expected to be named cpg-repr-benchmark)
conda activate cpg-repr-benchmark
python -m pip install -e ".[dev]"

# tests (synthetic invariants only; no biological data required)
pytest -q
pytest tests/test_masking.py::test_name -q   # single test

# lint
ruff check src tests

# one-time local data preflight (validates h5 contracts, CpG coverage, creates symlinks)
python scripts/bootstrap_local_data.py

# run a masking benchmark arm
CUDA_VISIBLE_DEVICES=0 python scripts/run_masking_benchmark.py \
  --config configs/experiments/masking/<config>.yaml --mode all

# compact the raw ENCODE functional-annotation feature store into a PCA embedding
python scripts/build_functional_pca_embedding.py --help

# aggregate finished runs into a summary CSV
python scripts/summarize_runs.py --outputs outputs --csv outputs/masking_summary.csv
```

## Architecture

**Data flow boundary**: `src/cpg_repr_benchmark/representations/` is the only place representation-specific
code may live (`resolver.py` dispatches on `representation.mode`: `precomputed` / `generate_if_missing` /
`regenerate` / `online`; `hdf5_store.py` loads the canonical `/cpg_idx` + `/embedding` contract with alias
auto-detection; `materialization.py` builds caches; `online.py` / `functional_provider.py` implement runtime
encoders with `dim` + `encode(cpg_ids)`). Everything downstream of this boundary — `models/`, `training/`,
`evaluation/` — must stay representation-agnostic and only ever consume the resolved embedding array.

**Locus and patient splits are dataset-defined, not representation-defined.** `data/protocols/*.npz` is
created once by the first run and every subsequent representation is evaluated against that exact same
train/held-out CpG split (`src/cpg_repr_benchmark/data/transfer_protocols.py`,
`src/cpg_repr_benchmark/data/splits.py`). A representation that cannot cover the required loci must fail
explicitly rather than silently narrowing the universe — never build a per-representation intersection.

**Two first-class evaluation views** for masked reconstruction (`src/cpg_repr_benchmark/training/engine.py`,
`data/masking.py`): `seen` (masked targets from train-locus universe) and `unseen_locus` (disjoint held-out
CpG split, excluded from both reconstruction context and the prior fit). Priors for held-out loci are fit
from train patients × train loci only — see `training/priors.py`. Both views must be reported for every
masking fraction.

**Fixed model shape** (`src/cpg_repr_benchmark/models/model.py`): observed locus representation + observed
methylation residual → tokenization → DeepSets patient encoder → patient embedding; target locus
representation + patient embedding → residual decoder → `sigmoid(prior + delta)`. A learned adapter maps
each representation's raw dimension to a common latent width, so this model never needs representation-
specific branches.

**Run outputs** are content-addressed under
`outputs/<task>/<dataset>/<representation>/<track>/seed_<seed>/<timestamp>-<config_hash>/`, driven by
`src/cpg_repr_benchmark/experiments/run_store.py`. Each run directory is self-describing: resolved config,
`representation_manifest.json` (records which HDF5 keys were actually resolved), splits, checkpoints, and
per-mask-fraction metrics under `evaluation/{seen,unseen_locus}/`.

**Representation catalog**: `configs/representations/*.yaml` is the single source of truth for every arm's
family, track, mode, store path, and provenance (patient-specific?, supervision, locus-fit scope). When
adding a representation, register it here rather than hardcoding paths in experiment configs.

**Config layering**: `configs/datasets/`, `configs/representations/`, and
`configs/experiments/{masking,classification,age}/` compose independently — an experiment config references
a representation entry and a dataset, not the other way around.

## Key invariants when modifying code

- Held-out/unseen CpGs must never influence the prior or downstream model selection for that arm.
- Don't change reconstruction architecture, losses, split seed, or training budget for only one
  representation arm — that breaks the controlled comparison that is this repo's entire purpose.
- `online` mode requires `training.num_workers=0` and `evaluation.num_workers=0` (GPU-resident encoder state
  must not be replicated into DataLoader subprocesses).
- New representation caches must use canonical `/cpg_idx` (int64) + `/embedding` (float16/float32) dataset
  names; only pre-existing external atlases may rely on alias auto-detection.
