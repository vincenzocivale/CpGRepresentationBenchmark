#!/usr/bin/env python3
"""Compress the raw reference-genome functional-annotation feature store into a compact,
unsupervised PCA/SVD embedding -- no supervised training against any methylation objective.

Source contract (see data/derived/functional_annotations/*.h5):
  /cpg_idx        int64 [N]                 (legacy array-registry namespace)
  /dense          float16 [N, dense_dim]    continuous per-locus features
  /track_indices  int64 [nnz]               CSR column indices into n_tracks binary tracks
  /track_indptr   int64 [N+1]               CSR row pointer

Output contract (canonical representation store):
  /cpg_idx    int64 [N]
  /embedding  float16 [N, D]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-components", type=int, default=256)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()

    with h5py.File(args.input, "r") as src:
        cpg_idx = np.asarray(src["cpg_idx"][:], dtype=np.int64)
        dense = np.asarray(src["dense"][:], dtype=np.float32)
        track_indices = np.asarray(src["track_indices"][:], dtype=np.int64)
        track_indptr = np.asarray(src["track_indptr"][:], dtype=np.int64)
        n_tracks = int(src.attrs["n_tracks"])

    n_loci = len(cpg_idx)
    track_data = np.ones(len(track_indices), dtype=np.float32)
    tracks = sparse.csr_matrix((track_data, track_indices, track_indptr), shape=(n_loci, n_tracks))

    dense_mean = dense.mean(axis=0, keepdims=True)
    dense_std = dense.std(axis=0, keepdims=True)
    dense_std[dense_std == 0] = 1.0
    dense_standardized = (dense - dense_mean) / dense_std

    # Match the two blocks' per-feature scale (unit column norm on average) so the SVD
    # does not simply spend all its variance budget on whichever block has larger raw
    # magnitude; tracks are 0/1 indicators so their per-column std is already small.
    tracks_col_scale = 1.0 / max(np.sqrt(tracks.multiply(tracks).mean()), 1e-8)
    combined = sparse.hstack(
        [tracks.multiply(tracks_col_scale), sparse.csr_matrix(dense_standardized)],
        format="csr",
    )

    svd = TruncatedSVD(n_components=args.n_components, random_state=args.seed)
    embedding = svd.fit_transform(combined).astype(np.float16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as out:
        out.create_dataset("cpg_idx", data=cpg_idx, dtype="int64")
        out.create_dataset("embedding", data=embedding, dtype="float16")
        out.attrs["representation"] = "functional_annotations_pca"
        out.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        out.attrs["source"] = str(args.input)
        out.attrs["n_components"] = args.n_components
        out.attrs["explained_variance_ratio_sum"] = float(svd.explained_variance_ratio_.sum())
        out.attrs["patient_specific"] = False
        out.attrs["supervision"] = "none"
        out.attrs["coordinate_convention"] = "1-based position of CpG cytosine"
        out.attrs["cpg_namespace"] = "legacy_unspecified"
        out.attrs["reference_build"] = "unspecified"

    sidecar = {
        "n_loci": int(n_loci),
        "n_components": args.n_components,
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "explained_variance_ratio_top10": svd.explained_variance_ratio_[:10].tolist(),
        "source": str(args.input),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(sidecar, indent=2))
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
