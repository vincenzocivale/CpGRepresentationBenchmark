#!/usr/bin/env python3
"""Build the canonical CpGPT locus representation store for this benchmark.

CpGPT is patient-agnostic and label-free (a DNA-sequence encoder, never fit against
any methylation objective), so it reports under the same native_frozen /
locus_only contract as the other genomic-FM arms (see docs/ADDING_REPRESENTATIONS.md).

Source contract: this repo's CpG registry parquet, e.g.
data/cpg/registries/array_cpg_map.parquet, columns [cpg_idx, chr, pos] with
1-based cytosine positions (see docs/COORDINATE_NATIVE_DATA.md).

Output contract (canonical representation store):
  /cpg_idx    int64 [N]
  /embedding  float16/float32 [N, D]

This script must run inside a CpGPT-pinned Python environment (see
scripts/cpgpt/README.md), NOT the cpg-repr-benchmark conda env: CpGPT pins its
own torch/lightning/hydra stack that would otherwise conflict with this repo's
dependencies. It imports nothing from the cpg_repr_benchmark package -- the
registry parquet's chr/pos columns are consumed directly, so there is no
cross-environment coupling.

We use representation="sequence" (CpGPT's projection of the local DNA window),
not "locus": the "locus" mode's rotary positional encoding is defined over the
complete ordered input list, so a given CpG's "locus" vector is not a stable,
order-independent per-locus feature -- it would change depending on which other
CpGs are queried alongside it. "sequence" gives an ordering-independent,
patient-agnostic feature per CpG, matching what every other arm in this
benchmark provides.

Example:
  conda activate cpgpt
  python scripts/build_cpgpt_embedding.py \\
      --registry data/cpg/registries/array_cpg_map.parquet \\
      --output data/cache/representations/cpgpt_locus.h5 \\
      --model-resources-dir /data2/home/vcivale/projects/methylation/methylation-fm-benchmark/data/cpgpt \\
      --dependencies-dir /data2/home/vcivale/projects/methylation/methylation-fm-benchmark/data/cpgpt/dna_dependencies \\
      --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# This is a shared machine: cap CPU thread pools (torch intra-op, MKL/OMP) before torch is
# imported by extract_cpg_embeddings, since env vars only take effect at library init time.
# The DataLoader used for --generate-missing is already forced to num_workers=0 (no separate
# worker processes), so this is the only knob that controls CPU parallelism here.
_MAX_WORKERS = int(os.environ.get("CPGPT_MAX_WORKERS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(_MAX_WORKERS))
os.environ.setdefault("MKL_NUM_THREADS", str(_MAX_WORKERS))

import h5py
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(_MAX_WORKERS)

sys.path.insert(0, str(Path(__file__).resolve().parent / "cpgpt"))
from extract_cpg_embeddings import extract_cpg_embeddings  # noqa: E402


def _to_cpgpt_location(chrom: str, one_based_pos: int) -> str:
    """This repo's registries are 1-based cytosine positions (docs/COORDINATE_NATIVE_DATA.md);
    CpGPT keys use the zero-based start of the CpG interval, so subtract one."""
    canonical = str(chrom)
    if canonical.lower().startswith("chr"):
        canonical = canonical[3:]
    return f"{canonical}:{int(one_based_pos) - 1}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", type=Path, required=True, help="CpG registry parquet with [cpg_idx, chr, pos]")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-name", default="small", choices=["small", "large"])
    p.add_argument("--model-resources-dir", type=Path, required=True,
                    help="Directory containing model/weights/<name>.ckpt and model/config/<name>.yaml, "
                         "OR (if --checkpoint-path/--config-path are given) any placeholder path")
    p.add_argument("--checkpoint-path", type=Path, default=None)
    p.add_argument("--config-path", type=Path, default=None)
    p.add_argument("--dependencies-dir", type=Path, required=True,
                    help="Directory containing ensembl_metadata.db and dna_embeddings/ (CpGPT's DNA cache)")
    p.add_argument("--species", default="homo_sapiens")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--generate-missing", action="store_true",
                    help="Compute DNA embeddings in memory for coordinates absent from CpGPT's precomputed cache")
    p.add_argument("--genome-file", type=Path, default=None)
    p.add_argument("--generation-batch-size", type=int, default=1,
                    help="Batch size for the upstream DNA-LLM forward pass used by --generate-missing "
                         "(CpGPT's own default of 1 is very slow on GPU)")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N loci (smoke test)")
    args = p.parse_args()

    registry = pd.read_parquet(args.registry, columns=["cpg_idx", "chr", "pos"])
    if args.limit is not None:
        registry = registry.iloc[: args.limit]

    cpg_idx = registry["cpg_idx"].to_numpy(dtype=np.int64)
    if len(cpg_idx) != len(np.unique(cpg_idx)):
        raise ValueError(f"{args.registry} contains duplicate cpg_idx values")

    locations = [
        _to_cpgpt_location(chrom, pos)
        for chrom, pos in zip(registry["chr"], registry["pos"], strict=True)
    ]

    embeddings = extract_cpg_embeddings(
        locations,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        model_resources_dir=args.model_resources_dir,
        dependencies_dir=args.dependencies_dir,
        species=args.species,
        representation="sequence",
        device=args.device,
        batch_size=args.batch_size,
        generate_missing=args.generate_missing,
        genome_file=args.genome_file,
        generation_batch_size=args.generation_batch_size,
    ).astype(np.float16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as out:
        out.create_dataset("cpg_idx", data=cpg_idx, dtype="int64")
        out.create_dataset("embedding", data=embeddings, dtype="float16")
        out.attrs["representation"] = "cpgpt_sequence"
        out.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        out.attrs["source"] = str(args.registry)
        out.attrs["model_name"] = args.model_name
        out.attrs["cpgpt_representation_mode"] = "sequence"
        out.attrs["patient_specific"] = False
        out.attrs["supervision"] = "none"
        out.attrs["locus_fit_scope"] = "external_pretrained"
        out.attrs["coordinate_convention"] = "1-based position of CpG cytosine"
        out.attrs["reference_build"] = "GRCh38"

    sidecar = {
        "n_loci": int(len(cpg_idx)),
        "embedding_dim": int(embeddings.shape[1]),
        "model_name": args.model_name,
        "representation_mode": "sequence",
        "source": str(args.registry),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(sidecar, indent=2))
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
