#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cpg_repr_benchmark.data.methylation import read_axis
from cpg_repr_benchmark.representations.hdf5_store import HDF5RepresentationStore, inspect_representation_h5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--methylation-h5", type=Path)
    p.add_argument("--id-key", default="auto")
    p.add_argument("--embedding-key", default="auto")
    args = p.parse_args()
    ids, dim, id_key, embedding_key = inspect_representation_h5(
        args.store, id_key=args.id_key, embedding_key=args.embedding_key
    )
    payload = {
        "store": str(args.store),
        "n_loci": int(len(ids)),
        "embedding_dim": int(dim),
        "id_key": id_key,
        "embedding_key": embedding_key,
    }
    if args.methylation_h5:
        matrix_ids, _ = read_axis(args.methylation_h5)
        store = HDF5RepresentationStore(
            args.store, matrix_ids, id_key=id_key, embedding_key=embedding_key
        )
        payload["matrix_loci"] = int(len(matrix_ids))
        payload["covered_matrix_loci"] = int(store.coverage_mask.sum())
        payload["coverage_fraction"] = float(np.mean(store.coverage_mask))
        store.close()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
