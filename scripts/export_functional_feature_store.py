#!/usr/bin/env python3
"""One-time migration helper from the original MehylPredictor functional cache.

After this export, representation compaction (PCA) is self-contained in
CpGRepresentationBenchmark and no RNA branch or MehylPredictor model code is needed for the
benchmark.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from cpg_repr_benchmark.representations.functional_provider import _load_sibling


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methylpredictor-root", type=Path, required=True)
    parser.add_argument("--locus-store", type=Path, required=True)
    parser.add_argument("--cpg-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    _, open_cache = _load_sibling(args.methylpredictor_root)
    cache = open_cache(locus_store=args.locus_store, cpg_registry=args.cpg_registry)
    registry = pd.read_parquet(args.cpg_registry, columns=["cpg_idx"])
    cpg_ids = registry["cpg_idx"].to_numpy(dtype=np.int64)
    all_tracks: list[np.ndarray] = []
    indptr = [0]
    dense_chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(cpg_ids), args.batch_size), desc="functional features"):
        ids = cpg_ids[start : start + args.batch_size]
        batch = cache.get(ids)
        indices = np.asarray(batch["track_indices"], dtype=np.int64)
        offsets = np.asarray(batch["offsets"], dtype=np.int64)
        dense_chunks.append(np.asarray(batch["dense"], dtype=np.float32))
        # Existing cache uses EmbeddingBag include_last_offset=True.
        for i in range(len(ids)):
            chunk = indices[offsets[i] : offsets[i + 1]]
            all_tracks.append(chunk)
            indptr.append(indptr[-1] + len(chunk))
    track_indices = np.concatenate(all_tracks) if all_tracks else np.empty(0, dtype=np.int64)
    dense = np.concatenate(dense_chunks, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.create_dataset("cpg_idx", data=cpg_ids)
        handle.create_dataset("track_indices", data=track_indices, compression="gzip", compression_opts=1)
        handle.create_dataset("track_indptr", data=np.asarray(indptr, dtype=np.int64))
        handle.create_dataset("dense", data=dense.astype(np.float16), compression="gzip", compression_opts=1)
        handle.attrs["patient_specific"] = False
        handle.attrs["source"] = "reference-genome functional annotation cache"
    print(args.output)


if __name__ == "__main__":
    main()
