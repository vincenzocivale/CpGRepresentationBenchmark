#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cpg_repr_benchmark.data.methylation import read_axis
from cpg_repr_benchmark.data.transfer_protocols import write_external_locus_protocols


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Build persistent downstream locus sets: all, shared with the training source, "
            "and loci external to the training source"
        )
    )
    p.add_argument("--dataset-h5", type=Path, required=True)
    p.add_argument("--membership", type=Path, required=True, help="master_cpg_membership.parquet")
    p.add_argument("--dataset-source", required=True, help="source label used in master membership")
    p.add_argument(
        "--training-source", required=True, help="representation-training source label used in master membership"
    )
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    dataset_ids, _ = read_axis(args.dataset_h5)
    membership = pd.read_parquet(args.membership, columns=["cpg_idx", "source"])
    if membership.duplicated(["cpg_idx", "source"]).any():
        raise ValueError("master membership contains duplicate (cpg_idx, source) rows")

    dataset_membership = membership.loc[
        membership["source"].astype(str) == str(args.dataset_source), "cpg_idx"
    ].to_numpy(dtype=np.int64)
    training_source_ids = membership.loc[
        membership["source"].astype(str) == str(args.training_source), "cpg_idx"
    ].to_numpy(dtype=np.int64)
    if not len(dataset_membership):
        raise ValueError(f"dataset source {args.dataset_source!r} is absent from master membership")
    if not len(training_source_ids):
        raise ValueError(f"training source {args.training_source!r} is absent from master membership")

    dataset_set = set(dataset_ids.tolist())
    membership_set = set(dataset_membership.tolist())
    if dataset_set != membership_set:
        only_h5 = sorted(dataset_set - membership_set)[:10]
        only_membership = sorted(membership_set - dataset_set)[:10]
        raise ValueError(
            "dataset HDF5 and master membership disagree for the requested dataset source: "
            f"h5={len(dataset_set)}, membership={len(membership_set)}, "
            f"only_h5_examples={only_h5}, only_membership_examples={only_membership}"
        )

    manifest = write_external_locus_protocols(
        dataset_ids,
        training_source_ids,
        args.output_dir,
        dataset_source=str(args.dataset_source),
        training_source=str(args.training_source),
        metadata={
            "dataset_h5": str(args.dataset_h5.resolve()),
            "membership": str(args.membership.resolve()),
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"WROTE={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
