#!/usr/bin/env python3
"""Build a MethylGPT-scoped diagnostic dataset for a standalone masking-benchmark run.

MethylGPT's checkpoint vocabulary (49,156 Illumina probes, fixed regardless of checkpoint
size -- see scripts/methylgpt/README.md) covers only a small, scattered subset of this
repo's genome-wide array CpG registry. The benchmark's locus-scoping mechanism
(scripts/run_masking_benchmark.py's _scoped_columns) requires dataset.cpg_registry to cover
every CpG in dataset.methylation_h5 exactly -- it has no notion of "restrict to whatever a
representation happens to cover" (CLAUDE.md: locus/patient splits are dataset-defined, never
silently narrowed per representation). So a MethylGPT arm cannot reuse any existing
genome-wide or chr1 protocol/dataset (checked: only ~12% overlap with the existing chr1
protocol) without violating that rule.

This script instead builds a small, self-consistent dataset -- a column subset of the full
array beta matrix restricted to exactly the loci one MethylGPT representation store covers
-- so a fresh, dataset-defined protocol over that reduced universe gives 100% representation
coverage. This is a standalone MethylGPT diagnostic, NOT part of the controlled
representation-comparison table: no other arm shares this reduced locus universe (see
docs/ADDING_REPRESENTATIONS.md's fairness checklist, item 1).

Usage:
  python scripts/build_methylgpt_diagnostic_dataset.py \\
      --source-h5 data/methylation/tcga_array_official_full.h5 \\
      --source-registry data/cpg/registries/array_cpg_map.parquet \\
      --representation-h5 data/cache/representations/methylgpt_locus_medium.h5 \\
      --output-h5 data/cache/datasets/tcga_array_methylgpt_medium_diagnostic.h5 \\
      --output-registry data/cache/datasets/tcga_array_methylgpt_medium_diagnostic_registry.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-h5", type=Path, required=True)
    p.add_argument("--source-registry", type=Path, required=True)
    p.add_argument("--representation-h5", type=Path, required=True,
                    help="Canonical /cpg_idx + /embedding store defining the loci to keep")
    p.add_argument("--output-h5", type=Path, required=True)
    p.add_argument("--output-registry", type=Path, required=True)
    args = p.parse_args()

    with h5py.File(args.representation_h5, "r") as rep:
        rep_cpg_idx = np.asarray(rep["cpg_idx"][:], dtype=np.int64)
    if len(rep_cpg_idx) != len(np.unique(rep_cpg_idx)):
        raise ValueError(f"{args.representation_h5} has duplicate cpg_idx values")

    with h5py.File(args.source_h5, "r") as src:
        full_cpg_idx = np.asarray(src["cpg_idx"][:], dtype=np.int64)
        sample_name = src["sample_name"][:]
        beta = src["beta"]

        keep_mask = np.isin(full_cpg_idx, rep_cpg_idx)
        n_keep = int(keep_mask.sum())
        if n_keep != len(rep_cpg_idx):
            missing = len(rep_cpg_idx) - n_keep
            raise ValueError(
                f"{missing}/{len(rep_cpg_idx)} representation loci are absent from {args.source_h5}"
            )
        keep_columns = np.nonzero(keep_mask)[0]
        subset_cpg_idx = full_cpg_idx[keep_columns]
        subset_beta = beta[:, keep_columns]

        args.output_h5.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.output_h5, "w") as out:
            out.create_dataset("beta", data=subset_beta, dtype="float32")
            out.create_dataset("cpg_idx", data=subset_cpg_idx, dtype="int64")
            out.create_dataset("sample_name", data=sample_name)
            out.attrs["source"] = str(args.source_h5)
            out.attrs["representation_scope"] = str(args.representation_h5)
            out.attrs["note"] = (
                "Column subset of the full array matrix restricted to one representation's "
                "covered loci -- a standalone diagnostic dataset, not part of the controlled "
                "cross-representation comparison."
            )

    registry = pd.read_parquet(args.source_registry, columns=["cpg_idx", "chr", "pos"])
    registry = registry[registry["cpg_idx"].isin(subset_cpg_idx)].reset_index(drop=True)
    if len(registry) != len(subset_cpg_idx):
        raise ValueError(
            f"{args.source_registry} covers only {len(registry)}/{len(subset_cpg_idx)} of the "
            "subset dataset's loci"
        )
    args.output_registry.parent.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(args.output_registry, index=False)

    print(f"n_loci={len(subset_cpg_idx)} n_samples={subset_beta.shape[0]}")
    print(f"chromosomes present: {sorted(registry['chr'].unique().tolist())}")


if __name__ == "__main__":
    main()
