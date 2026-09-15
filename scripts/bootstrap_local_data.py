#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from cpg_repr_benchmark.data.chromosomes import normalize_chromosome
from cpg_repr_benchmark.data.methylation import read_axis
from cpg_repr_benchmark.representations.hdf5_store import inspect_representation_h5


def _safe_symlink(source: Path, target: Path) -> None:
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        current = target.resolve()
        if current == source:
            return
        target.unlink()
    elif target.exists():
        raise FileExistsError(f"refusing to replace existing non-symlink: {target}")
    target.symlink_to(source, target_is_directory=source.is_dir())


def _require(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _file_info(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size)}


def main() -> None:
    repo_root_default = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Prepare and validate the local chr1 benchmark data layout")
    p.add_argument("--repo-root", type=Path, default=repo_root_default)
    p.add_argument("--methylpredictor-root", type=Path)
    p.add_argument("--no-symlinks", action="store_true", help="validate only; do not create functional symlinks")
    args = p.parse_args()

    repo = args.repo_root.resolve()
    mp = (args.methylpredictor_root or (repo.parent / "MehylPredictor")).resolve()
    data = repo / "data"

    methyl = _require(data / "methylation/tcga_array_official_full.h5", "TCGA methylation matrix")
    registry = _require(data / "cpg/registries/array_cpg_map.parquet", "array CpG registry")
    ntv3 = _require(
        data / "derived/ntv3_pre_chr1_atlas/chr1_ntv3_pretrain_atlas_v1.h5",
        "NTv3-pre chr1 atlas",
    )

    source_locus_store = _require(
        mp / "local_methyl_data/derived/locus_features_v1", "MehylPredictor functional locus store"
    )
    source_checkpoint = _require(
        mp
        / "local_methyl_data/runs/encode_wgbs/runs/rna_methylation/genomewide/"
        "encode-wgbs-seed17-b128-c65536/checkpoints/last.pt",
        "functional encoder checkpoint",
    )

    linked_locus_store = data / "external/functional/locus_features_v1"
    linked_checkpoint = data / "external/functional/checkpoints/functional_encoder.pt"
    if not args.no_symlinks:
        _safe_symlink(source_locus_store, linked_locus_store)
        _safe_symlink(source_checkpoint, linked_checkpoint)

    matrix_ids, sample_names = read_axis(methyl)
    reg = pd.read_parquet(registry, columns=["cpg_idx", "chr", "pos"])
    if reg["cpg_idx"].duplicated().any():
        raise ValueError("array_cpg_map.parquet contains duplicate cpg_idx")
    reg_indexed = reg.set_index("cpg_idx")
    aligned = reg_indexed.reindex(matrix_ids)
    if aligned["chr"].isna().any():
        n_missing = int(aligned["chr"].isna().sum())
        raise ValueError(f"array registry is missing {n_missing} CpGs from the methylation matrix")
    chrom = aligned["chr"].map(normalize_chromosome).to_numpy()
    chr1_mask = chrom == "chr1"
    chr1_ids = matrix_ids[chr1_mask]
    if len(chr1_ids) == 0:
        raise ValueError("no chr1 CpGs found after registry alignment")

    ntv3_ids, ntv3_dim, ntv3_id_key, ntv3_embedding_key = inspect_representation_h5(ntv3)
    ntv3_chr1_covered = np.isin(chr1_ids, ntv3_ids, assume_unique=False)
    missing_chr1 = chr1_ids[~ntv3_chr1_covered]

    payload = {
        "schema_version": 1,
        "benchmark_scope": "chr1",
        "repo_root": str(repo),
        "methylpredictor_root": str(mp),
        "methylation": {
            **_file_info(methyl),
            "n_samples": int(len(sample_names)),
            "n_cpg_total": int(len(matrix_ids)),
            "n_cpg_chr1": int(len(chr1_ids)),
        },
        "registry": {**_file_info(registry), "n_rows": int(len(reg))},
        "ntv3_pre": {
            **_file_info(ntv3),
            "n_loci": int(len(ntv3_ids)),
            "embedding_dim": int(ntv3_dim),
            "id_key": ntv3_id_key,
            "embedding_key": ntv3_embedding_key,
            "chr1_matrix_loci_covered": int(ntv3_chr1_covered.sum()),
            "chr1_matrix_loci_missing": int(len(missing_chr1)),
            "chr1_coverage_fraction": float(ntv3_chr1_covered.mean()),
            "missing_examples": missing_chr1[:10].astype(np.int64).tolist(),
        },
        "functional": {
            "source_locus_store": str(source_locus_store),
            "source_checkpoint": str(source_checkpoint),
            "linked_locus_store": str(linked_locus_store),
            "linked_checkpoint": str(linked_checkpoint),
            "links_ready": bool(linked_locus_store.exists() and linked_checkpoint.exists()),
            "cached_chr1_store": str(data / "cache/representations/functional_annotations_chr1.h5"),
        },
    }

    # The benchmark protocol is dataset-defined. We require complete NTv3 coverage of chr1
    # rather than silently shrinking the CpG universe for this representation.
    if len(missing_chr1):
        payload["status"] = "INCOMPLETE_NTV3_CHR1_COVERAGE"
        payload["action"] = (
            "The NTv3 atlas does not cover every TCGA-array chr1 CpG. Do not run the benchmark yet; "
            "either rebuild the atlas for the full chr1 array universe or define one explicit common-universe protocol."
        )
    else:
        payload["status"] = "READY_FOR_FUNCTIONAL_MATERIALIZATION_AND_MASKING"

    out = data / "local_manifest.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nWROTE={out}")
    if len(missing_chr1):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
