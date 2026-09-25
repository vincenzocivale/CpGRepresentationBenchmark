#!/usr/bin/env python3
"""Build the canonical DeepCpG DNA-module locus representation store for this benchmark.

Only DeepCpG's DNA module is used, never its CpG module or Joint model: the DNA
module depends solely on reference-genome sequence around a locus, so it is
patient-agnostic and label-free, matching this benchmark's native_frozen /
locus_only contract (see docs/ADDING_REPRESENTATIONS.md). DeepCpG's CpG module
consumes per-cell neighboring methylation state -- using it here would leak
patient-specific values into the locus embedding, which this benchmark
forbids.

Source contract: this repo's CpG registry parquet, e.g.
data/cpg/registries/array_cpg_map.parquet, columns [cpg_idx, chr, pos] with
1-based cytosine positions (see docs/COORDINATE_NATIVE_DATA.md).

Output contract (canonical representation store):
  /cpg_idx    int64 [N]
  /embedding  float16/float32 [N, D]

This script must run inside a DeepCpG-pinned Python environment (see
scripts/deepcpg/README.md), NOT the cpg-repr-benchmark conda env: DeepCpG
pins a legacy standalone-keras/TensorFlow-1.x stack that would otherwise
conflict with this repo's dependencies. It imports nothing from the
cpg_repr_benchmark package -- the registry parquet's chr/pos columns are
consumed directly, so there is no cross-environment coupling.

Two extraction backends are available (--backend):
  keras  Runs the actual DeepCpG/Keras graph (scripts/deepcpg/extract_cpg_embeddings.py).
         Reference implementation; TF 1.15 has no GPU support on Ampere+ GPUs.
  torch  Ports the checkpoint's weights into an equivalent PyTorch module
         (scripts/deepcpg/extract_cpg_embeddings_torch.py), numerically
         verified against the keras backend. Only needs torch/h5py/pyfaidx,
         so it runs GPU-accelerated in any modern env with CUDA (this repo
         reuses the pre-existing `cpgpt` conda env for that -- see
         scripts/deepcpg/README.md).

Example:
  conda activate deepcpg-env
  python scripts/build_deepcpg_embedding.py \\
      --registry data/cpg/registries/array_cpg_map.parquet \\
      --output data/cache/representations/deepcpg_dna_locus.h5 \\
      --model-dir dependencies/deepcpg/hcc_dna \\
      --genome-file dependencies/deepcpg/hg38.fa \\
      --dna-wlen 1001

  # GPU, in the `cpgpt` env instead:
  conda activate cpgpt
  python scripts/build_deepcpg_embedding.py --backend torch --device cuda \\
      --registry data/cpg/registries/array_cpg_map.parquet \\
      --output data/cache/representations/deepcpg_dna_locus.h5 \\
      --model-dir dependencies/deepcpg/hcc_dna \\
      --genome-file dependencies/deepcpg/hg38.fa
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "deepcpg"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", type=Path, required=True, help="CpG registry parquet with [cpg_idx, chr, pos]")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True,
                    help="Unzipped DeepCpG DNA-module directory (model.json + model_weights.h5)")
    p.add_argument("--model-name", default="dna",
                    help="Free-text label recorded in provenance, e.g. the zoo checkpoint "
                         "identifier (hcc, hepg2, mesc_2i, mesc_serum)")
    p.add_argument("--genome-file", type=Path, required=True,
                    help="Reference FASTA matching the checkpoint's training assembly and "
                         "chromosome-naming convention")
    p.add_argument("--dna-wlen", type=int, default=1001,
                    help="DNA sequence window length in bp; must match the checkpoint's "
                         "saved input shape")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--limit", type=int, default=None, help="Process only the first N loci (smoke test)")
    p.add_argument("--strip-chr-prefix", action="store_true",
                    help="Strip a leading 'chr' from registry chromosome names before querying "
                         "--genome-file, e.g. for Ensembl-style toplevel FASTAs that name "
                         "contigs '1'..'22','X','Y','MT' instead of 'chr1'..'chrM'")
    p.add_argument("--backend", choices=["keras", "torch"], default="keras",
                    help="keras = reference DeepCpG/Keras graph (CPU); "
                         "torch = weight-ported PyTorch module (GPU-capable, see module docstring)")
    p.add_argument("--device", default="cpu", help="--backend torch only: 'cpu' or 'cuda'")
    args = p.parse_args()

    if args.backend == "keras":
        from extract_cpg_embeddings import extract_cpg_embeddings  # noqa: E402
    else:
        from extract_cpg_embeddings_torch import extract_cpg_embeddings  # noqa: E402

    registry = pd.read_parquet(args.registry, columns=["cpg_idx", "chr", "pos"])
    if args.limit is not None:
        registry = registry.iloc[: args.limit]

    cpg_idx = registry["cpg_idx"].to_numpy(dtype=np.int64)
    if len(cpg_idx) != len(np.unique(cpg_idx)):
        raise ValueError(f"{args.registry} contains duplicate cpg_idx values")

    chroms = registry["chr"]
    if args.strip_chr_prefix:
        chroms = chroms.str.replace(r"^chr", "", regex=True).replace({"M": "MT"})
    loci = list(zip(chroms, registry["pos"].astype(int)))

    extract_kwargs = dict(
        model_dir=args.model_dir,
        genome_file=args.genome_file,
        dna_wlen=args.dna_wlen,
        batch_size=args.batch_size,
    )
    if args.backend == "torch":
        extract_kwargs["device"] = args.device
    embeddings = extract_cpg_embeddings(loci, **extract_kwargs).astype(np.float16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as out:
        out.create_dataset("cpg_idx", data=cpg_idx, dtype="int64")
        out.create_dataset("embedding", data=embeddings, dtype="float16")
        out.attrs["representation"] = "deepcpg_dna_stem"
        out.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        out.attrs["source"] = str(args.registry)
        out.attrs["model_name"] = args.model_name
        out.attrs["dna_wlen"] = args.dna_wlen
        out.attrs["patient_specific"] = False
        out.attrs["supervision"] = "none"
        out.attrs["locus_fit_scope"] = "external_pretrained"
        out.attrs["coordinate_convention"] = "1-based position of CpG cytosine"

    sidecar = {
        "n_loci": int(len(cpg_idx)),
        "embedding_dim": int(embeddings.shape[1]),
        "model_name": args.model_name,
        "dna_wlen": args.dna_wlen,
        "source": str(args.registry),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(sidecar, indent=2))
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
