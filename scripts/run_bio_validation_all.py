#!/usr/bin/env python3
"""Run `run_bio_validation` for every `component: locus_only` entry in the representation
catalog (`configs/representations/*.yaml`), skipping representations whose summary.json already
exists under `outputs/bio_validation/`.

Loads the annotation sources once and reuses them across all representations, instead of
re-reading the same genomic-context/known-set/coefficient files from disk per representation
(what `run_bio_validation.py` does when invoked once per representation by hand).

Usage:
    python scripts/run_bio_validation_all.py
    python scripts/run_bio_validation_all.py --only cpgpt_locus deepcpg_dna_locus
    python scripts/run_bio_validation_all.py --force   # recompute everything even if summary.json exists
    python scripts/run_bio_validation_all.py --incremental   # only compute NEW probes (e.g. newly
                                                               # added known_sets) missing from an
                                                               # existing summary.json; much cheaper
                                                               # than --force when most probes are
                                                               # already valid and unchanged
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from run_bio_validation import run_bio_validation, summary_path

from cpg_repr_benchmark.bio_validation.annotations import load_cpg_annotations


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_catalog(catalog_path: Path) -> dict:
    return yaml.safe_load(catalog_path.read_text())["representations"]


def main() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, default=root / "configs/representations/future_models.yaml")
    p.add_argument("--genomic-context-parquet", type=Path, default=root / "data/bio_annotations/genomic_context.parquet")
    p.add_argument("--known-sets-dir", type=Path, default=root / "data/bio_annotations/known_sets")
    p.add_argument("--coefficients-dir", type=Path, default=root / "data/bio_annotations")
    p.add_argument("--cpg-registry", type=Path, default=None, help="passed through to every representation unless overridden per-entry via a `bio_validation_cpg_registry` catalog key")
    p.add_argument(
        "--locality-registry",
        type=Path,
        default=root / "data/cpg/master_cpg_registry.parquet",
        help="cpg_idx/chr/pos registry for the unsupervised locality probe; pass an empty/missing path to skip it",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--only", nargs="*", default=None, help="restrict to these representation names")
    p.add_argument("--force", action="store_true", help="fully recompute even if outputs/bio_validation/.../summary.json already exists")
    p.add_argument(
        "--incremental",
        action="store_true",
        help="revisit representations that already have a summary.json too, but only compute "
        "probes missing from it (e.g. newly added known_sets) instead of everything from scratch",
    )
    args = p.parse_args()

    output_root = args.output_root or (root / "outputs")
    catalog = _load_catalog(args.catalog)

    annotations = load_cpg_annotations(
        genomic_context_parquet=args.genomic_context_parquet if args.genomic_context_parquet.exists() else None,
        known_sets_dir=args.known_sets_dir if args.known_sets_dir.exists() else None,
        coefficients_dir=args.coefficients_dir if args.coefficients_dir.exists() else None,
    )
    locality_registry = (
        pd.read_parquet(args.locality_registry)
        if args.locality_registry and args.locality_registry.exists()
        else None
    )

    results = {"ran": [], "skipped": [], "missing_store": [], "failed": []}
    for entry in catalog.values():
        if entry.get("component") != "locus_only":
            continue
        name = entry["name"]
        track = entry.get("track", "native_frozen")
        if args.only and name not in args.only:
            continue

        store = root / entry["store_h5"]
        out_path = summary_path(output_root, name, track, args.seed)

        if not store.exists():
            print(f"[skip] {name}: store_h5 not found at {store}")
            results["missing_store"].append(name)
            continue
        if out_path.exists() and not args.force and not args.incremental:
            print(f"[skip] {name}: already computed at {out_path}")
            results["skipped"].append(name)
            continue

        registry = args.cpg_registry
        registry_key = entry.get("bio_validation_cpg_registry")
        if registry_key is not None:
            registry = root / registry_key
        registry_lenient = bool(entry.get("bio_validation_cpg_registry_lenient", False))

        print(f"[run]  {name}{' (incremental)' if args.incremental else ''}")
        try:
            run_bio_validation(
                representation_store=store,
                representation_name=name,
                annotations=annotations,
                track=track,
                cpg_registry=registry,
                cpg_registry_lenient=registry_lenient,
                seed=args.seed,
                output_root=output_root,
                locality_registry=locality_registry,
                incremental=args.incremental,
            )
            results["ran"].append(name)
        except Exception as exc:  # noqa: BLE001 - keep looping over the rest of the catalog
            print(f"[fail] {name}: {exc}")
            results["failed"].append({"name": name, "error": str(exc)})

    print(json.dumps(results, indent=2))
    if results["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
