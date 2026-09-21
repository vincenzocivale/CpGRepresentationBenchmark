# CpGPT locus embedding extraction

`extract_cpg_embeddings.py` is a small, single-purpose CpGPT adapter. It
accepts CpGPT location keys (`"chromosome:position"`, zero-based CpG start)
and returns CpGPT's DNA-sequence-encoder output as a NumPy array. It has no
dependency on this repo's `cpg_repr_benchmark` package.

`scripts/build_cpgpt_embedding.py` (one level up) is the actual integration
point: it reads this repo's CpG registry (`cpg_idx, chr, pos`), converts each
locus to a CpGPT key, calls `extract_cpg_embeddings`, and writes the
canonical `/cpg_idx` + `/embedding` HDF5 store that
`configs/representations/future_models.yaml`'s `cpgpt_locus_native` entry
points at.

## Environment: `.venv-cpgpt` (required for `--generate-missing`)

The shared `cpgpt` conda env (see below) has drifted to `transformers==5.9.0`,
which removed the API (`find_pruneable_heads_and_indices`) that the pinned
Nucleotide Transformer v2 checkpoint's remote code still imports, and its
`torch==2.5.1` is too old for `transformers==5.9.0`'s default `torchao`
integration path either way. Plain cache-hit lookups (no `--generate-missing`)
don't load that HF model, so they work fine directly in `cpgpt`; the
`--generate-missing` path (needed for any CpG not in CpGPT's small precomputed
cache -- most of this repo's ~408k-locus array universe) does not.

Rather than downgrading `transformers` in the shared `cpgpt` env (risk of
breaking the sibling `methylation-fm-benchmark` project that also uses it), a
layered venv pins a compatible `transformers` locally while still reusing the
already-installed `torch`/CUDA stack from `cpgpt` (no multi-GB re-download):

```bash
conda activate cpgpt
cd CpGRepresentationBenchmark
python -m venv --system-site-packages .venv-cpgpt
source .venv-cpgpt/bin/activate
python -m pip install h5py "transformers==4.46.3"
python -m pip uninstall -y torchao   # only needed by newer transformers' quant path; not used here
```

`.venv-cpgpt/` is git-ignored. Always run `scripts/build_cpgpt_embedding.py`
through `source .venv-cpgpt/bin/activate` (not bare `conda activate cpgpt`)
when `--generate-missing` is needed; either works for cache-hit-only runs.

## Why a separate environment

CpGPT pins its own `torch` / `pytorch-lightning` / `hydra-core` /
`omegaconf` / `sqlitedict` / `pyfaidx` stack, which would conflict with the
`cpg-repr-benchmark` conda env's dependencies. Per
`docs/ADDING_REPRESENTATIONS.md`, the "precompute once" path is preferred for
exactly this reason: extraction runs once, offline, in whatever environment
the upstream model needs, and the benchmark only ever reads the resulting
HDF5 file with `h5py` (no CpGPT import required downstream).

## Environment and dependencies: already set up

This machine already has a working CpGPT install from a sibling project, so
there is nothing to newly download for the small checkpoint:

- Conda env: `cpgpt` (already created; has `torch`, `pytorch-lightning`,
  `hydra-core`, `omegaconf`, `pyfaidx`).
- Model checkpoint/config:
  `/data2/home/vcivale/projects/methylation/methylation-fm-benchmark/data/cpgpt/small.ckpt`,
  `.../small.yaml`
- DNA dependency bundle (Ensembl metadata, precomputed DNA-embedding cache,
  reference genome):
  `/data2/home/vcivale/projects/methylation/methylation-fm-benchmark/data/cpgpt/dna_dependencies/`
- The `cpgpt` Python package itself is installed from
  `/data2/home/vcivale/projects/methylation/methylation-fm-benchmark/models/CpGPT`
  (editable install in the `cpgpt` env).

To confirm the env still has the package before running:

```bash
conda activate cpgpt
python -c "import cpgpt; print(cpgpt.__file__)"
```

If that import fails (env was rebuilt), reinstall from the sibling repo:

```bash
conda activate cpgpt
pip install -e /data2/home/vcivale/projects/methylation/methylation-fm-benchmark/models/CpGPT
```

`build_cpgpt_embedding.py`'s `--model-resources-dir` expects
`<dir>/model/weights/<name>.ckpt` and `<dir>/model/config/<name>.yaml`, which
does not match the sibling repo's flat `small.ckpt` / `small.yaml` layout.
Either pass explicit `--checkpoint-path` / `--config-path`, or stage the
expected layout locally with symlinks (no copying of large files needed):

```bash
mkdir -p dependencies/model/weights dependencies/model/config
ln -s /data2/home/vcivale/projects/methylation/methylation-fm-benchmark/data/cpgpt/small.ckpt \
      dependencies/model/weights/small.ckpt
ln -s /data2/home/vcivale/projects/methylation/methylation-fm-benchmark/data/cpgpt/small.yaml \
      dependencies/model/config/small.yaml
```

`dependencies/` here is git-ignored the same way `data/cache/` is; it is a
local staging area, not something to commit.

## Building the store

```bash
source .venv-cpgpt/bin/activate   # or `conda activate cpgpt` if not using --generate-missing
python scripts/build_cpgpt_embedding.py \
    --registry data/cpg/registries/array_cpg_map.parquet \
    --output data/cache/representations/cpgpt_locus.h5 \
    --model-name small \
    --model-resources-dir dependencies \
    --dependencies-dir dependencies/human \
    --generate-missing \
    --device cuda
```

Smoke-test first with `--limit 200` (and `cpu` if no GPU is free) to confirm
the environment, paths, and dependency bundle are wired correctly before
running the full ~400k-locus genome-wide array universe.

If any coordinate in the registry is absent from CpGPT's precomputed DNA
cache, the script raises `KeyError` by default. Pass `--generate-missing` to
compute those vectors on the fly (uses the bundled GRCh38 FASTA under
`dna_dependencies/genomes/`); this does not mutate CpGPT's shared cache.

## Representation choice: `sequence`, not `locus`

`extract_cpg_embeddings` supports two representations. This integration
always uses `representation="sequence"`:

- `sequence`: CpGPT's projection of the local DNA window around each CpG.
  Depends only on that CpG's own genomic context, so it is a stable,
  order-independent per-locus feature — exactly the patient-agnostic,
  reusable-in-any-order feature this benchmark's HDF5 contract assumes.
- `locus`: applies CpGPT's checkpoint-configured positional encoding (RoPE
  for the released small/rotary checkpoint) over the *entire ordered input
  list* as one sequence. The resulting vector for a given CpG is therefore
  not fixed — it changes with the query batch and its ordering — which is
  incompatible with a precomputed, batch-independent `/cpg_idx -> /embedding`
  store. Do not switch to `locus` without re-deriving the whole extraction
  as a single, fixed, benchmark-wide ordered pass.

## Coordinate contract

This repo's registries (e.g. `data/cpg/registries/array_cpg_map.parquet`)
store the 1-based position of the CpG cytosine (see
`docs/COORDINATE_NATIVE_DATA.md`). CpGPT keys use the zero-based start of the
two-base CpG interval, so `build_cpgpt_embedding.py` subtracts 1 and strips
any `chr` prefix (`chr16:53434200` -> `"16:53434199"`). No liftover or strand
conversion is performed; the registry is already GRCh38.
