#!/usr/bin/env python3
"""Validate a CpG-locus representation against known, ENCODE-independent biology.

Probes the raw per-CpG representation itself (the `/embedding` array of a representation's
HDF5 store) against external genomic-context and known-CpG-set annotations. See
`bio_validation.annotations` for the annotation file contracts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from cpg_repr_benchmark.bio_validation.annotations import CpGAnnotations, load_cpg_annotations
from cpg_repr_benchmark.bio_validation.probes import bio_validation_report
from cpg_repr_benchmark.data.legacy import legacy_ids_to_coordinate_ids, legacy_ids_to_coordinate_ids_lenient
from cpg_repr_benchmark.experiments.result_schema import embedding_summary_payload
from cpg_repr_benchmark.representations.hdf5_store import validate_canonical_h5


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def summary_path(output_root: Path, representation_name: str, track: str, seed: int) -> Path:
    return output_root / "bio_validation" / representation_name / track / f"seed_{seed}" / "summary.json"


def _load_existing_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())["metrics"]
    except (json.JSONDecodeError, KeyError):
        return None


def run_bio_validation(
    *,
    representation_store: Path,
    representation_name: str,
    annotations: CpGAnnotations,
    track: str = "native_frozen",
    cpg_registry: Path | None = None,
    cpg_registry_lenient: bool = False,
    seed: int = 17,
    output_root: Path | None = None,
    locality_registry: pd.DataFrame | None = None,
    incremental: bool = False,
) -> dict:
    """Probe one representation store against already-loaded `annotations` and write summary.json.

    Split out from `main()` so an orchestrator (e.g. `run_bio_validation_all.py`) can load the
    annotation sources once and reuse them across every representation in the catalog, instead
    of re-reading the same parquet/npy files from disk on every invocation.

    `cpg_registry_lenient=True` drops legacy CpGs the registry has no chr/pos for instead of
    raising (see `legacy_ids_to_coordinate_ids_lenient`) — acceptable here because bio_validation
    is a read-only diagnostic probe, unlike the masking benchmark's evaluation universe.

    `incremental=True` skips re-running probes an existing summary.json already has a result for.
    Each `known_cpg_sets` entry is individually the most expensive probe (a 5-fold
    LogisticRegressionCV over the full embedding, per set) — when only a few new known_sets were
    added since the last run, recomputing all ~15-20 of them from scratch to add a handful of new
    ones wastes hours. Only the known_sets missing from the old summary are computed; entries for
    sets no longer present in `annotations.known_sets` (e.g. one that was deleted) are dropped
    from the merged result. `genomic_context`/`clock_coefficients`/`locality` are reused verbatim
    from the old summary if present (they're cheap compared to known_sets, but rerunning is still
    pure waste when nothing about those annotation sources changed).
    """
    import h5py

    root = _repo_root()
    output_root = output_root or (root / "outputs")
    run_dir = output_root / "bio_validation" / representation_name / track / f"seed_{seed}"

    old_metrics = _load_existing_metrics(run_dir / "summary.json") if incremental else None
    probe_annotations = annotations
    reuse_locality_registry = locality_registry
    old_known_sets: dict = {}

    if old_metrics is not None:
        old_known_sets = old_metrics.get("known_cpg_sets", {})
        missing_set_names = set(annotations.known_sets) - set(old_known_sets)
        probe_annotations = replace(
            annotations,
            known_sets={name: annotations.known_sets[name] for name in missing_set_names},
            genomic_context=pd.Series(dtype=object) if "genomic_context" in old_metrics else annotations.genomic_context,
            coefficients={} if "clock_coefficients" in old_metrics else annotations.coefficients,
        )
        if old_metrics.get("locality"):
            reuse_locality_registry = None
        known_sets_unchanged = set(old_known_sets) == set(annotations.known_sets)
        if (
            known_sets_unchanged
            and probe_annotations.genomic_context.empty
            and not probe_annotations.coefficients
            and reuse_locality_registry is None
        ):
            # merging would be a genuine no-op: same known_set keys (nothing added *or*
            # removed) and nothing else left to compute either; reuse the old summary untouched
            return json.loads((run_dir / "summary.json").read_text())

    cpg_idx, dim = validate_canonical_h5(representation_store)
    with h5py.File(representation_store, "r") as handle:
        embedding = np.asarray(handle["embedding"][:], dtype=np.float32)
    if cpg_registry is not None:
        if cpg_registry_lenient:
            cpg_idx, keep_mask = legacy_ids_to_coordinate_ids_lenient(cpg_idx, cpg_registry)
            embedding = embedding[keep_mask]
        else:
            cpg_idx = legacy_ids_to_coordinate_ids(cpg_idx, cpg_registry)

    report = bio_validation_report(
        embedding=embedding,
        cpg_idx=cpg_idx,
        annotations=probe_annotations,
        seed=seed,
        locality_registry=reuse_locality_registry,
    )

    if old_metrics is not None:
        merged_known_sets = {
            **{name: result for name, result in old_known_sets.items() if name in annotations.known_sets},
            **report["known_cpg_sets"],
        }
        report["known_cpg_sets"] = merged_known_sets
        if "genomic_context" in old_metrics and not report["genomic_context"]:
            report["genomic_context"] = old_metrics["genomic_context"]
        if "clock_coefficients" in old_metrics and not report["clock_coefficients"]:
            report["clock_coefficients"] = old_metrics["clock_coefficients"]
        if old_metrics.get("locality") and not report.get("locality"):
            report["locality"] = old_metrics["locality"]

    run_dir.mkdir(parents=True, exist_ok=True)

    payload = embedding_summary_payload(
        task="bio_validation",
        dataset="cpg_locus_annotations",
        representation=representation_name,
        track=track,
        mode="frozen_pretrained",
        embedding_source_type="cpg_locus_embedding",
        embedding_dim=int(dim),
        metrics=report,
        n_patients=None,
        checkpoint=str(representation_store),
    )
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Probe a CpG representation store for known biological signal")
    p.add_argument("--representation-store", type=Path, required=True, help="canonical /cpg_idx + /embedding HDF5")
    p.add_argument("--representation-name", required=True)
    p.add_argument("--track", default="native_frozen")
    p.add_argument("--genomic-context-parquet", type=Path, default=None)
    p.add_argument("--known-sets-dir", type=Path, default=None)
    p.add_argument(
        "--coefficients-dir",
        type=Path,
        default=None,
        help="directory of <set_name>_coefficients.parquet files (cpg_idx, coefficient) for "
        "published per-CpG clock weights, e.g. data/bio_annotations/ (horvath/hannum/phenoage)",
    )
    p.add_argument(
        "--cpg-registry",
        type=Path,
        default=None,
        help="legacy cpg_idx -> chr,pos registry (e.g. data/cpg/registries/array_cpg_map.parquet); "
        "required whenever --representation-store holds legacy (non-coordinate-namespace) CpG "
        "IDs, which is the case for most genome-wide caches produced before coordinate-native "
        "IDs became the default (see docs/ADDING_REPRESENTATIONS.md)",
    )
    p.add_argument(
        "--cpg-registry-lenient",
        action="store_true",
        help="drop legacy CpGs missing from --cpg-registry instead of failing (diagnostic-only; "
        "never use for the masking benchmark's evaluation universe)",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument(
        "--locality-registry",
        type=Path,
        default=None,
        help="cpg_idx/chr/pos registry (e.g. data/cpg/master_cpg_registry.parquet) enabling the "
        "unsupervised nearest-neighbor locality probe; omit to skip it",
    )
    p.add_argument(
        "--incremental",
        action="store_true",
        help="only compute known_sets/genomic_context/clock_coefficients/locality probes missing "
        "from an existing summary.json, instead of recomputing everything from scratch",
    )
    args = p.parse_args()

    annotations = load_cpg_annotations(
        genomic_context_parquet=args.genomic_context_parquet,
        known_sets_dir=args.known_sets_dir,
        coefficients_dir=args.coefficients_dir,
    )
    locality_registry = pd.read_parquet(args.locality_registry) if args.locality_registry else None
    payload = run_bio_validation(
        representation_store=args.representation_store,
        representation_name=args.representation_name,
        annotations=annotations,
        track=args.track,
        cpg_registry=args.cpg_registry,
        cpg_registry_lenient=args.cpg_registry_lenient,
        seed=args.seed,
        output_root=args.output_root,
        locality_registry=locality_registry,
        incremental=args.incremental,
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
