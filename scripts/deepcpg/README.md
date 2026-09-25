# DeepCpG DNA-module locus embedding extraction

`extract_cpg_embeddings.py` is a small, single-purpose DeepCpG adapter. It
loads DeepCpG's official **DNA module** only (never the CpG module or Joint
model), strips its pretraining output head, and returns the stem's pooled
activation for a reference-genome sequence window around each CpG. It has no
dependency on this repo's `cpg_repr_benchmark` package.

`scripts/build_deepcpg_embedding.py` (one level up) is the actual integration
point: it reads this repo's CpG registry (`cpg_idx, chr, pos`), calls
`extract_cpg_embeddings`, and writes the canonical `/cpg_idx` + `/embedding`
HDF5 store that `configs/representations/future_models.yaml`'s
`deepcpg_dna_locus_native` entry points at.

## Why only the DNA module

DeepCpG (Angermueller et al., *Genome Biology* 2017,
[cangermueller/deepcpg](https://github.com/cangermueller/deepcpg)) has three
components:

- **DNA module**: CNN/ResNet over a sequence window (`--dna_wlen`, default
  1001bp) around a CpG. Depends only on reference-genome sequence -- stable,
  patient-agnostic, exactly the kind of feature this benchmark's HDF5
  contract assumes.
- **CpG module**: bidirectional GRU over the *observed methylation state of
  neighboring CpGs in specific cells*. This is per-patient/per-cell input by
  construction.
- **Joint module**: combines the two.

Using the CpG or Joint module here would leak patient-specific methylation
values into the locus embedding, which this benchmark's controlled-comparison
design forbids (see `CLAUDE.md`). `dcpg_train.py`'s own `build_dna_model()`
already isolates the DNA module and strips its output head
(`remove_outputs`) before recombining it with a CpG module -- this adapter
does the same thing standalone, matching the DNA-only weights published in
DeepCpG's model zoo.

## Environment: pinned legacy stack (separate from `cpg-repr-benchmark`)

DeepCpG's `setup.py` pins standalone `keras>=2.0.2` with a `tensorflow>=1.0.1`
backend (not `tensorflow.keras`), and its saved `model.json` configs use
layer APIs from that era. This is incompatible with the `cpg-repr-benchmark`
conda env's modern stack, so -- following the same "precompute once, in
whatever environment the upstream model needs" pattern used for CpGPT (see
`scripts/cpgpt/README.md`) -- extraction runs once, offline, in a dedicated
environment; the benchmark only ever reads the resulting HDF5 file with
`h5py`.

A `deepcpg-env` conda environment already exists on this machine but is
currently unrelated (a modern PyTorch env with no `deepcpg`/`keras`
installed) -- it needs to be rebuilt for this integration:

```bash
conda create -n deepcpg-env python=3.7 -y
conda activate deepcpg-env
pip install "tensorflow==1.15.5" "keras==2.2.4" "h5py<3" "protobuf<3.20" \
            numpy scipy pandas scikit-learn pyfaidx
pip install git+https://github.com/cangermueller/deepcpg.git
```

`tensorflow==1.15.5` is the last TF1 release and needs `protobuf<3.20`;
`h5py<3` is required because Keras 2.2.4's model-saving code predates h5py's
3.0 string-encoding changes. Confirm the install:

```bash
python -c "import deepcpg, keras; print(deepcpg.__file__, keras.__version__)"
```

## Model zoo checkpoints

DeepCpG's published pretrained DNA modules
(`docs/source/zoo.md` in the DeepCpG repo) include human checkpoints trained
on Hou et al. 2016 scRRBS data:

- **HCC** (25 human hepatocellular carcinoma cells)
- **HepG2** (6 human hepatoplastoma cells)

and mouse checkpoints (Smallwood et al. 2014 scBS-seq: serum/2i mESC) that
are out of scope for a human-genome benchmark arm. Each zoo entry is a
`*.zip` containing the DNA-module-only `model.json` + `model_weights.h5`;
unzip it into its own directory before pointing `--model-dir` at it:

```bash
mkdir -p dependencies/deepcpg/hcc_dna
curl -L -o dependencies/deepcpg/hcc_dna.zip \
    http://www.ebi.ac.uk/~angermue/deepcpg/alias/260e4c19cef65fd36f7e7e3d7edd2c15
unzip dependencies/deepcpg/hcc_dna.zip -d dependencies/deepcpg/hcc_dna
```

`dependencies/` here is git-ignored the same way `data/cache/` is; it is a
local staging area, not something to commit.

## Reference genome

DeepCpG's Hou et al. checkpoints were trained against `hg38`/GRCh38. Point
`--genome-file` at a GRCh38 FASTA whose chromosome names match this repo's
registry convention (`docs/COORDINATE_NATIVE_DATA.md` uses bare `chr1`..
`chrM`); reuse the FASTA already staged for the CpGPT integration
(`scripts/cpgpt/README.md`'s `dna_dependencies/genomes/`) if its naming
matches, rather than re-downloading.

## Coordinate contract

This repo's registries (e.g. `data/cpg/registries/array_cpg_map.parquet`)
store the 1-based position of the CpG cytosine. `extract_cpg_embeddings`
centers the DNA sequence window directly on that coordinate (0-based FASTA
slicing, see `_sequence_window`). No liftover or strand conversion is
performed; the registry is already GRCh38, matching the zoo checkpoints.

## Building the store

Two extraction backends are available via `--backend`:

- `keras` (default): runs the actual DeepCpG/Keras graph
  (`extract_cpg_embeddings.py`), in the `deepcpg-env` conda env above.
  **CPU only** -- TensorFlow 1.15's GPU wheels predate Ampere's compute
  capability (8.0), and NVIDIA's Ampere-patched `nvidia-tensorflow` fork
  is not reachable from this network's package index.
- `torch`: ports the checkpoint's weights into an equivalent PyTorch
  `nn.Sequential`, built dynamically from the checkpoint's own `model.json`
  layer list (`extract_cpg_embeddings_torch.py`). GPU-capable, and verified
  numerically equivalent to the `keras` backend (max abs diff ~2e-3 in
  float16 storage, correlation 0.9999999 on a 50-locus check). Needs only
  `torch`/`h5py`/`pyfaidx` -- no `deepcpg`/legacy-`keras` install -- so it
  runs in any modern CUDA-enabled env; this repo reuses the pre-existing
  `cpgpt` conda env rather than adding a third one.

```bash
# CPU, reference backend:
conda activate deepcpg-env
python scripts/build_deepcpg_embedding.py \
    --registry data/cpg/registries/array_cpg_map.parquet \
    --output data/cache/representations/deepcpg_dna_locus.h5 \
    --model-dir dependencies/deepcpg/hcc_dna \
    --model-name hcc_dna \
    --genome-file dependencies/deepcpg/hg38.fa \
    --dna-wlen 1001 --strip-chr-prefix

# GPU, torch backend (what was actually used to build both zoo checkpoints'
# genome-wide stores registered in future_models.yaml):
conda activate cpgpt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/build_deepcpg_embedding.py --backend torch --device cuda \
    --registry data/cpg/registries/array_cpg_map.parquet \
    --output data/cache/representations/deepcpg_dna_locus.h5 \
    --model-dir dependencies/deepcpg/hcc_dna \
    --model-name hcc_dna \
    --genome-file dependencies/deepcpg/hg38.fa \
    --dna-wlen 1001 --strip-chr-prefix --batch-size 256
```

`--strip-chr-prefix` is required with the staged `hg38.fa` symlink (reused
from the CpGPT dependency bundle): it is an Ensembl toplevel FASTA whose
contigs are named `1`..`22`,`X`,`Y`,`MT`, not `chr1`..`chrM`.

`--batch-size 256` above is deliberately small: this machine's GPU is shared
and was down to ~3GB free at build time, and CUDA OOM'd at the default
larger batch sizes. Raise it if more memory is free (check with
`nvidia-smi`); lower it further (e.g. 64) if it OOMs again.

Both released human zoo checkpoints have been built genome-wide over this
repo's ~408k-locus array registry:

| store | checkpoint | architecture | dim |
|---|---|---|---|
| `data/cache/representations/deepcpg_dna_locus.h5` (`deepcpg_dna_locus_native`) | HCC (25 cells) | CnnL2h128 | 128 |
| `data/cache/representations/deepcpg_dna_locus_hepg2.h5` (`deepcpg_dna_locus_hepg2_native`) | HepG2 (6 cells) | CnnL3h128 | 128 |

Smoke-test a new checkpoint first with `--limit 200` to confirm the
environment, paths, and checkpoint window length are wired correctly before
running the full genome-wide array universe. If `extract_cpg_embeddings`
raises a window-size mismatch, check the checkpoint's actual `dna_wlen`
(Hou et al. checkpoints may differ from the 1001bp default; the input shape
saved in `model.json` is authoritative).

## Architecture coverage of the torch backend

`extract_cpg_embeddings_torch.py` parses `model.json`'s layer list generically
(`Convolution1D`, `Activation('relu')`, `MaxPooling1D`, `Flatten`, `Dense`,
`Dropout`), so it covers any DeepCpG zoo DNA-module checkpoint regardless of
depth -- verified against both the 2-conv-layer HCC checkpoint (`CnnL2h128`)
and the 3-conv-layer HepG2 checkpoint (`CnnL3h128`). It stops parsing at the
checkpoint's first per-cell `cpg/*` output head, same as the `keras` backend's
`remove_outputs`-equivalent logic. It would need extending (not covered by
either released human checkpoint) for `CnnRnn01` (GRU) or `ResNet`/`ResConv`
DNA-module variants.

## Excluded: Methyl-GP

`Methyl-GP` (Xie et al., *NAR* 2025,
[Hao010418/Methyl-GP](https://github.com/Hao010418/Methyl-GP)) was evaluated
for this benchmark and **not integrated**: it predicts 4mC/5hmC/6mA marks
(non-CpG, largely non-human/bacterial-and-plant epigenetic marks), and its
released checkpoints are fine-tuned per species/dataset rather than a single
frozen, general-purpose embedding. Neither property fits this benchmark's
`native_frozen` track, which requires one fixed embedding per human GRCh38
CpG locus, independent of any downstream task.
