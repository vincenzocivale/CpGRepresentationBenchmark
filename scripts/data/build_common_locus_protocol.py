#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cpg_repr_benchmark.data.methylation import read_axis
from cpg_repr_benchmark.data.universe import read_ids_from_h5, write_common_universe_protocol


def _parse_representation(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("--representation must be NAME=PATH")
    name, value = spec.split("=", 1)
    return name, Path(value).expanduser()


def main() -> None:
    p = argparse.ArgumentParser(description="Persist the exact common CpG universe for a controlled comparison")
    p.add_argument("--dataset-h5", type=Path, required=True)
    p.add_argument("--representation", action="append", required=True, type=_parse_representation)
    p.add_argument(
        "--candidate-protocol",
        type=Path,
        help="optional NPZ containing cpg_idx; intersect representations only within this pre-defined locus set",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    dataset_ids, _ = read_axis(args.dataset_h5)
    candidate_ids = dataset_ids
    candidate_path = None
    if args.candidate_protocol is not None:
        candidate_path = args.candidate_protocol.resolve()
        protocol = np.load(candidate_path)
        if "cpg_idx" not in protocol:
            raise ValueError(f"{candidate_path} must contain cpg_idx")
        candidate_ids = np.asarray(protocol["cpg_idx"], dtype=np.int64)
        if len(candidate_ids) != len(np.unique(candidate_ids)):
            raise ValueError("candidate protocol contains duplicate cpg_idx")
        missing_mask = ~np.isin(candidate_ids, dataset_ids, assume_unique=False)
        if missing_mask.any():
            missing = candidate_ids[missing_mask]
            raise ValueError(
                f"candidate protocol contains {len(missing)} loci absent from dataset; examples={missing[:10].tolist()}"
            )

    representations = {name: read_ids_from_h5(path) for name, path in args.representation}
    manifest = write_common_universe_protocol(
        candidate_ids,
        representations,
        args.output,
        metadata={
            "dataset_h5": str(args.dataset_h5.resolve()),
            "full_dataset_loci": int(len(dataset_ids)),
            "candidate_protocol": str(candidate_path) if candidate_path else None,
            "representation_paths": {n: str(p.resolve()) for n, p in args.representation},
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"WROTE={args.output.resolve()}")


if __name__ == "__main__":
    main()
