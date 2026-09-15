#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from cpg_repr_benchmark.data.coordinates import CPG_NAMESPACE, COORDINATE_CONVENTION, REFERENCE_BUILD, encode_many


def _detect(handle: h5py.File, requested: str, candidates: tuple[str, ...]) -> str:
    if requested != "auto":
        if requested not in handle:
            raise KeyError(f"/{requested} not found")
        return requested
    for key in candidates:
        if key in handle:
            return key
    raise KeyError(f"none of {candidates} found")


def main() -> None:
    p = argparse.ArgumentParser(description="Re-key a legacy locus representation into the coordinate-native CpG namespace")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--registry", type=Path, required=True, help="legacy cpg_idx -> chr,pos registry")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--id-key", default="auto")
    p.add_argument("--embedding-key", default="auto")
    p.add_argument("--chunk-rows", type=int, default=4096)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(args.output)
    registry = pd.read_parquet(args.registry, columns=["cpg_idx", "chr", "pos"])
    if registry["cpg_idx"].duplicated().any():
        raise ValueError("legacy registry contains duplicate cpg_idx")
    registry = registry.set_index("cpg_idx")

    with h5py.File(args.input, "r") as src:
        id_key = _detect(src, args.id_key, ("cpg_idx", "cpg_ids", "ids", "id"))
        emb_key = _detect(src, args.embedding_key, ("embedding", "embeddings", "emb", "features"))
        legacy = np.asarray(src[id_key][:], dtype=np.int64)
        aligned = registry.reindex(legacy)
        if aligned[["chr", "pos"]].isna().any().any():
            missing = legacy[aligned["chr"].isna().to_numpy()]
            raise ValueError(f"registry cannot resolve {len(missing)} representation loci; examples={missing[:10].tolist()}")
        canonical = encode_many(aligned["chr"], aligned["pos"])
        if len(canonical) != len(np.unique(canonical)):
            raise ValueError("multiple legacy representation IDs resolve to the same canonical coordinate")
        matrix = src[emb_key]
        if matrix.ndim != 2 or matrix.shape[0] != len(legacy):
            raise ValueError("embedding matrix must be [loci, dim]")
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp.unlink(missing_ok=True)
        with h5py.File(tmp, "w") as dst:
            dst.create_dataset("cpg_idx", data=canonical, dtype="i8")
            dst.create_dataset("chrom", data=np.asarray(aligned["chr"].astype(str), dtype=object), dtype=h5py.string_dtype("utf-8"))
            dst.create_dataset("pos", data=aligned["pos"].to_numpy(dtype=np.int64), dtype="i8")
            out = dst.create_dataset(
                "embedding",
                shape=matrix.shape,
                dtype=matrix.dtype,
                chunks=(min(args.chunk_rows, matrix.shape[0]), matrix.shape[1]),
            )
            for lo in range(0, matrix.shape[0], args.chunk_rows):
                hi = min(matrix.shape[0], lo + args.chunk_rows)
                out[lo:hi] = matrix[lo:hi]
            for key, value in src.attrs.items():
                try:
                    dst.attrs[f"source_{key}"] = value
                except TypeError:
                    pass
            dst.attrs["reference_build"] = REFERENCE_BUILD
            dst.attrs["coordinate_convention"] = COORDINATE_CONVENTION
            dst.attrs["cpg_namespace"] = CPG_NAMESPACE
            dst.attrs["rekeyed_from"] = str(args.input.resolve())
            dst.attrs["legacy_registry"] = str(args.registry.resolve())
        os.replace(tmp, args.output)

    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "n_loci": int(len(canonical)),
        "embedding_dim": int(matrix.shape[1]),
        "cpg_namespace": CPG_NAMESPACE,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
