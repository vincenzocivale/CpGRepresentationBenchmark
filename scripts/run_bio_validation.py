#!/usr/bin/env python3
"""Validate a CpG-locus representation against known, ENCODE-independent biology.

Unlike `run_embedding_probe.py` (patient embedding vs. phenotype), this script probes the
raw per-CpG representation itself (the `/embedding` array of a representation's HDF5 store)
against external genomic-context and known-CpG-set annotations. See
`bio_validation.annotations` for the annotation file contracts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cpg_repr_benchmark.bio_validation.annotations import load_cpg_annotations
from cpg_repr_benchmark.bio_validation.probes import bio_validation_report
from cpg_repr_benchmark.data.legacy import legacy_ids_to_coordinate_ids
from cpg_repr_benchmark.experiments.result_schema import embedding_summary_payload
from cpg_repr_benchmark.representations.hdf5_store import validate_canonical_h5


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Probe a CpG representation store for known biological signal")
    p.add_argument("--representation-store", type=Path, required=True, help="canonical /cpg_idx + /embedding HDF5")
    p.add_argument("--representation-name", required=True)
    p.add_argument("--track", default="native_frozen")
    p.add_argument("--genomic-context-parquet", type=Path, default=None)
    p.add_argument("--known-sets-dir", type=Path, default=None)
    p.add_argument(
        "--cpg-registry",
        type=Path,
        default=None,
        help="legacy cpg_idx -> chr,pos registry (e.g. data/cpg/registries/array_cpg_map.parquet); "
        "required whenever --representation-store holds legacy (non-coordinate-namespace) CpG "
        "IDs, which is the case for most genome-wide caches produced before coordinate-native "
        "IDs became the default (see docs/ADDING_REPRESENTATIONS.md)",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output-root", type=Path, default=None)
    args = p.parse_args()

    root = _repo_root()
    output_root = args.output_root or (root / "outputs")

    import h5py

    cpg_idx, dim = validate_canonical_h5(args.representation_store)
    if args.cpg_registry is not None:
        cpg_idx = legacy_ids_to_coordinate_ids(cpg_idx, args.cpg_registry)
    with h5py.File(args.representation_store, "r") as handle:
        embedding = np.asarray(handle["embedding"][:], dtype=np.float32)

    annotations = load_cpg_annotations(
        genomic_context_parquet=args.genomic_context_parquet,
        known_sets_dir=args.known_sets_dir,
    )
    report = bio_validation_report(embedding=embedding, cpg_idx=cpg_idx, annotations=annotations, seed=args.seed)

    run_dir = output_root / "bio_validation" / args.representation_name / args.track / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = embedding_summary_payload(
        task="bio_validation",
        dataset="cpg_locus_annotations",
        representation=args.representation_name,
        track=args.track,
        mode="frozen_pretrained",
        embedding_source_type="cpg_locus_embedding",
        embedding_dim=int(dim),
        metrics=report,
        n_patients=None,
        checkpoint=str(args.representation_store),
    )
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
