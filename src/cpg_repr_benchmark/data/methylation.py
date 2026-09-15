from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def decode_strings(values) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in values]


def read_axis(path: Path) -> tuple[np.ndarray, list[str]]:
    with h5py.File(path, "r") as handle:
        if not {"beta", "cpg_idx", "sample_name"}.issubset(handle):
            raise ValueError(f"{path} must contain /beta, /cpg_idx and /sample_name")
        cpg_ids = np.asarray(handle["cpg_idx"][:], dtype=np.int64)
        sample_names = decode_strings(handle["sample_name"][:])
        if handle["beta"].shape != (len(sample_names), len(cpg_ids)):
            raise ValueError("beta matrix shape does not match sample/CpG axes")
    if len(cpg_ids) != len(np.unique(cpg_ids)):
        raise ValueError("methylation HDF5 contains duplicate cpg_idx values")
    return cpg_ids, sample_names


def read_row_columns(dataset: h5py.Dataset, row: int, columns: np.ndarray) -> np.ndarray:
    columns = np.asarray(columns, dtype=np.int64)
    order = np.argsort(columns)
    values = np.asarray(dataset[row, columns[order]], dtype=np.float32)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return values[inverse]
