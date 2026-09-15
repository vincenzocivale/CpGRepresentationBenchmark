from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def beta_to_logit(beta: np.ndarray, epsilon: float) -> np.ndarray:
    beta = np.clip(np.asarray(beta, dtype=np.float32), epsilon, 1.0 - epsilon)
    return np.log(beta) - np.log1p(-beta)


def _contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open contiguous runs covering exactly ``indices``.

    Using row slices lets h5py apply fancy indexing only on the CpG axis. This matters
    for the unseen-locus benchmark: we can physically read train-patient x train-CpG
    values without ever materialising held-out CpG methylation values while fitting the
    prior.
    """
    values = np.unique(np.asarray(indices, dtype=np.int64))
    if len(values) == 0:
        return []
    if values[0] < 0:
        raise ValueError("row indices must be non-negative")
    boundaries = np.flatnonzero(np.diff(values) != 1) + 1
    chunks = np.split(values, boundaries)
    return [(int(chunk[0]), int(chunk[-1]) + 1) for chunk in chunks]


def compute_leakage_safe_priors(
    matrix_path: Path,
    train_rows: np.ndarray,
    train_locus_columns: np.ndarray,
    n_cpg: int,
    *,
    epsilon: float = 1e-4,
    row_chunk: int = 32,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit priors using *only* train patients x train loci.

    Train loci receive their train-patient empirical mean. Every non-train locus receives
    one scalar global fallback estimated from the same train-patient x train-locus block.
    Therefore a held-out CpG can be queried at evaluation time without any methylation
    label from that CpG having contributed to the prior.

    ``row_chunk`` bounds memory for long contiguous patient runs. It never widens the CpG
    read: HDF5 is sliced on rows and fancy-indexed only by ``train_locus_columns``.
    """
    train_rows = np.asarray(train_rows, dtype=np.int64)
    train_locus_columns = np.asarray(train_locus_columns, dtype=np.int64)
    if len(train_rows) == 0:
        raise ValueError("train_rows cannot be empty")
    if len(train_locus_columns) == 0:
        raise ValueError("train_locus_columns cannot be empty")
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")
    if (train_locus_columns < 0).any() or (train_locus_columns >= n_cpg).any():
        raise ValueError("train_locus_columns are outside the methylation CpG axis")
    if len(np.unique(train_locus_columns)) != len(train_locus_columns):
        raise ValueError("train_locus_columns contains duplicates")

    # h5py requires increasing fancy indices. Keep an inverse permutation so output
    # statistics remain aligned to the caller's train_locus_columns order.
    col_order = np.argsort(train_locus_columns)
    sorted_columns = train_locus_columns[col_order]
    inverse_columns = np.empty_like(col_order)
    inverse_columns[col_order] = np.arange(len(col_order))

    sums_sorted = np.zeros(len(sorted_columns), dtype=np.float64)
    counts_sorted = np.zeros(len(sorted_columns), dtype=np.int64)
    global_sum = 0.0
    global_count = 0

    with h5py.File(matrix_path, "r") as handle:
        beta = handle["beta"]
        if beta.ndim != 2 or beta.shape[1] != n_cpg:
            raise ValueError(f"methylation /beta shape {beta.shape} is incompatible with n_cpg={n_cpg}")
        if train_rows.max(initial=-1) >= beta.shape[0]:
            raise ValueError("train_rows are outside the methylation sample axis")

        for run_start, run_stop in _contiguous_runs(train_rows):
            for r0 in range(run_start, run_stop, row_chunk):
                r1 = min(r0 + row_chunk, run_stop)
                # IMPORTANT: no ':' on the CpG axis. Held-out loci are not read here.
                block = np.asarray(beta[r0:r1, sorted_columns], dtype=np.float32)
                finite = np.isfinite(block)
                sums_sorted += np.nansum(block, axis=0)
                counts_sorted += finite.sum(axis=0)
                global_sum += float(np.nansum(block))
                global_count += int(finite.sum())

    if global_count == 0:
        raise ValueError("no finite methylation values in train patients x train loci")

    sums = sums_sorted[inverse_columns]
    counts = counts_sorted[inverse_columns]
    global_beta = global_sum / global_count
    global_logit = float(beta_to_logit(np.asarray([global_beta]), epsilon)[0])
    prior = np.full(n_cpg, global_logit, dtype=np.float32)
    usable = counts > 0
    if usable.any():
        means = sums[usable] / counts[usable]
        prior[train_locus_columns[usable]] = beta_to_logit(means.astype(np.float32), epsilon)
    return prior, usable, global_beta
