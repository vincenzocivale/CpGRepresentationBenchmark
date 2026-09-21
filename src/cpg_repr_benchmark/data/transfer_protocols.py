from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .universe import resolve_ids_to_columns


def split_external_locus_sets(
    dataset_ids: np.ndarray,
    training_source_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Partition a downstream dataset locus axis by training-source membership.

    The returned arrays preserve the downstream dataset order.  ``shared`` loci were
    available in the training-source universe (e.g. the dataset a representation's PCA or
    checkpoint was fit on); ``external_locus`` loci were not.  This definition is
    intentionally independent of representation coverage: a later common-universe protocol
    intersects each set with the representations being compared.
    """
    dataset = np.asarray(dataset_ids, dtype=np.int64)
    training_source = np.asarray(training_source_ids, dtype=np.int64)
    if len(dataset) != len(np.unique(dataset)):
        raise ValueError("dataset_ids must be unique")
    if len(training_source) != len(np.unique(training_source)):
        raise ValueError("training_source_ids must be unique")
    shared_mask = np.isin(dataset, training_source, assume_unique=False)
    return {
        "all": dataset.copy(),
        "shared": dataset[shared_mask],
        "external_locus": dataset[~shared_mask],
    }


def write_external_locus_protocols(
    dataset_ids: np.ndarray,
    training_source_ids: np.ndarray,
    output_dir: Path,
    *,
    dataset_source: str,
    training_source: str,
    metadata: dict | None = None,
) -> dict:
    dataset = np.asarray(dataset_ids, dtype=np.int64)
    sets = split_external_locus_sets(dataset, training_source_ids)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    for name, ids in sets.items():
        columns = resolve_ids_to_columns(dataset, ids)
        path = output_dir / f"{name}.npz"
        np.savez_compressed(path, cpg_idx=ids, matrix_columns=columns)
        paths[name] = str(path)

    n_dataset = len(sets["all"])
    manifest = {
        "schema_version": 1,
        "definition": "training-source locus membership before representation intersection",
        "dataset_source": dataset_source,
        "training_source": training_source,
        "n_dataset_loci": int(n_dataset),
        "n_shared_loci": int(len(sets["shared"])),
        "n_external_locus": int(len(sets["external_locus"])),
        "shared_fraction": float(len(sets["shared"]) / n_dataset) if n_dataset else 0.0,
        "external_locus_fraction": float(len(sets["external_locus"]) / n_dataset) if n_dataset else 0.0,
        "protocols": paths,
        "metadata": metadata or {},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
