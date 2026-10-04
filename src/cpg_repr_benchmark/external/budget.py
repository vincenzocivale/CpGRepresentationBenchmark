"""Optimizer-update budget arithmetic for the external reconstruction confirmation (torch-free).

The training engine iterates ``ceil(N_train / batch_size)`` batches per epoch (the last partial batch is kept:
``MaskFractionBatchSampler`` has no drop_last) and performs one optimizer update per batch.
"""
from __future__ import annotations

import math


def steps_per_epoch(n_train: int, batch_size: int) -> int:
    if n_train < 1 or batch_size < 1:
        raise ValueError("n_train and batch_size must be positive")
    return math.ceil(n_train / batch_size)


def total_updates(n_train: int, batch_size: int, epochs: int) -> int:
    return steps_per_epoch(n_train, batch_size) * int(epochs)


def external_schedule(total: int, spe: int) -> dict:
    """Map a fixed update budget to full epochs + a final partial epoch.

    Validation points (cumulative optimizer updates): the end of every full epoch (``k * spe``) and the final update
    (``total``; coincides with the last full-epoch end when ``total`` is a multiple of ``spe``).
    """
    if total < 1 or spe < 1:
        raise ValueError("total and spe must be positive")
    full, rem = divmod(int(total), int(spe))
    points = [k * spe for k in range(1, full + 1)]
    if rem:
        points.append(int(total))
    return {
        "total_updates": int(total),
        "steps_per_epoch": int(spe),
        "full_epochs": int(full),
        "partial_epoch_updates": int(rem),
        "n_epochs_started": int(full + (1 if rem else 0)),
        "validation_points": points,
        "n_validation_points": len(points),
    }
