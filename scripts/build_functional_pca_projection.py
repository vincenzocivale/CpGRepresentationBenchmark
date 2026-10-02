#!/usr/bin/env python3
"""Fit the unsupervised functional-annotation PCA/SVD on a *fit locus set* and project every locus.

Same recipe as ``build_functional_pca_embedding.py`` (unit-scaled binary track block + standardized
dense block -> TruncatedSVD), but the decomposition is fitted on a chosen subset of loci and then applied to
all loci in the raw store. This lets a representation be defined for loci that the fit never saw, e.g. the
EPIC-only probes when the fit set is the 450K-shared probes used as observations in the 450K->EPIC task.

Reference-genome annotations only: no methylation values, no patient data, no supervision.

Output (canonical representation store): /cpg_idx int64 [N], /embedding float16 [N, D];
plus ``<output>.projection.npz`` with the fitted mean/std/components for exact re-projection.
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="raw functional feature store (.h5)")
    p.add_argument("--fit-protocol", type=Path, required=True, help=".npz holding the fit locus ids")
    p.add_argument("--fit-key", default="observed_cpg_idx")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-components", type=int, default=256)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()

    with h5py.File(args.input, "r") as src:
        cpg_idx = np.asarray(src["cpg_idx"][:], dtype=np.int64)
        dense = np.asarray(src["dense"][:], dtype=np.float32)
        track_indices = np.asarray(src["track_indices"][:], dtype=np.int64)
        track_indptr = np.asarray(src["track_indptr"][:], dtype=np.int64)
        n_tracks = int(src.attrs.get("n_tracks", 4165))
    n_loci = len(cpg_idx)
    fit_ids = np.load(args.fit_protocol)[args.fit_key]
    fit_mask = np.isin(cpg_idx, fit_ids)
    if fit_mask.sum() != len(np.unique(fit_ids)):
        raise RuntimeError(f"raw store misses {len(np.unique(fit_ids)) - int(fit_mask.sum())} fit loci")

    tracks = sparse.csr_matrix(
        (np.ones(len(track_indices), dtype=np.float32), track_indices, track_indptr), shape=(n_loci, n_tracks)
    )
    mean = dense[fit_mask].mean(axis=0, keepdims=True)
    std = dense[fit_mask].std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    scale = 1.0 / max(np.sqrt(tracks[np.flatnonzero(fit_mask)].multiply(tracks[np.flatnonzero(fit_mask)]).mean()), 1e-8)
    combined = sparse.hstack(
        [tracks.multiply(scale), sparse.csr_matrix((dense - mean) / std)], format="csr"
    )
    fit_rows = np.flatnonzero(fit_mask)
    svd = TruncatedSVD(n_components=args.n_components, random_state=args.seed).fit(combined[fit_rows])
    embedding = np.asarray(combined @ svd.components_.T, dtype=np.float32).astype(np.float16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as out:
        out.create_dataset("cpg_idx", data=cpg_idx, dtype="int64")
        out.create_dataset("embedding", data=embedding, dtype="float16")
        out.attrs.update(
            representation="functional_annotations_pca",
            created_utc=datetime.now(timezone.utc).isoformat(),
            source=str(args.input),
            fit_scope=f"{args.fit_protocol.name}:{args.fit_key}",
            n_fit_loci=int(fit_mask.sum()),
            n_components=args.n_components,
            explained_variance_ratio_sum=float(svd.explained_variance_ratio_.sum()),
            patient_specific=False,
            supervision="none",
            coordinate_convention="1-based position of CpG cytosine",
            cpg_namespace="grch38_cpg_cytosine_1based_v1",
            reference_build="GRCh38",
        )
    np.savez_compressed(
        args.output.with_suffix(".projection.npz"),
        dense_mean=mean, dense_std=std, track_scale=np.float32(scale), components=svd.components_.astype(np.float32),
    )
    sidecar = {
        "n_loci": int(n_loci),
        "n_fit_loci": int(fit_mask.sum()),
        "n_components": args.n_components,
        "explained_variance_ratio_sum_on_fit": float(svd.explained_variance_ratio_.sum()),
        "explained_variance_ratio_top10": svd.explained_variance_ratio_[:10].tolist(),
        "source": str(args.input),
        "fit": f"{args.fit_protocol}:{args.fit_key}",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(sidecar, indent=2))
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
