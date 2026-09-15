from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def resolve_ids_to_columns(axis_ids: np.ndarray, requested_ids: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis_ids, dtype=np.int64)
    requested = np.asarray(requested_ids, dtype=np.int64)
    if len(axis) != len(np.unique(axis)):
        raise ValueError("axis_ids must be unique")
    order = np.argsort(axis)
    sorted_axis = axis[order]
    slots = np.searchsorted(sorted_axis, requested)
    valid = slots < len(sorted_axis)
    if valid.any():
        valid_idx = np.flatnonzero(valid)
        valid[valid_idx] = sorted_axis[slots[valid_idx]] == requested[valid_idx]
    if not valid.all():
        missing = requested[~valid]
        raise KeyError(f"{len(missing)} requested CpGs are absent from dataset axis; examples={missing[:10].tolist()}")
    return order[slots].astype(np.int64)


def load_locus_protocol(path: Path, axis_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "cpg_idx" not in data:
        raise ValueError(f"{path} must contain cpg_idx")
    ids = np.asarray(data["cpg_idx"], dtype=np.int64)
    if len(ids) != len(np.unique(ids)):
        raise ValueError("locus protocol contains duplicate cpg_idx")
    return ids, resolve_ids_to_columns(axis_ids, ids)


def write_common_universe_protocol(
    dataset_ids: np.ndarray,
    representation_ids: dict[str, np.ndarray],
    output_npz: Path,
    *,
    metadata: dict | None = None,
) -> dict:
    dataset_ids = np.asarray(dataset_ids, dtype=np.int64)
    common = dataset_ids.copy()
    coverage = {}
    dataset_set = set(dataset_ids.tolist())
    for name, values in representation_ids.items():
        ids = np.asarray(values, dtype=np.int64)
        hits = np.isin(dataset_ids, ids, assume_unique=False)
        coverage[name] = {
            "n_representation_loci": int(len(ids)),
            "n_dataset_loci_covered": int(hits.sum()),
            "dataset_coverage_fraction": float(hits.mean()) if len(hits) else 0.0,
        }
        common = common[np.isin(common, ids, assume_unique=False)]
    common_set = set(common.tolist())
    common = np.asarray([x for x in dataset_ids if int(x) in common_set], dtype=np.int64)
    columns = resolve_ids_to_columns(dataset_ids, common)

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, cpg_idx=common, matrix_columns=columns)
    manifest = {
        "schema_version": 1,
        "dataset_loci": int(len(dataset_ids)),
        "common_loci": int(len(common)),
        "common_fraction": float(len(common) / len(dataset_ids)) if len(dataset_ids) else 0.0,
        "representations": coverage,
        "metadata": metadata or {},
    }
    output_npz.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def read_ids_from_h5(path: Path, *, id_key: str = "cpg_idx") -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if id_key not in handle:
            raise KeyError(f"{path} has no /{id_key}")
        return np.asarray(handle[id_key][:], dtype=np.int64)
