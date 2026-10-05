"""Optimizer-update budget arithmetic for the external reconstruction confirmation (torch-free).

ACTIVE budget (protocol v1.1, AMENDMENT 1, pre-run): epoch-based, ``epoch_budget``. 120 epochs x ceil(524/8) = 66 updates
= 7,920 updates, validation at the end of each epoch (120 points), no partial epoch, no ``max_updates``.
``external_schedule`` below is the SUPERSEDED v1.0 equal-update derivation (110,160 updates = 1,669 epochs + 6 updates,
1,670 validation points); it was NEVER executed and is kept only as documented history (see ``SUPERSEDED_BUDGET``).

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


EXTERNAL_EPOCHS = 120
SUPERSEDED_BUDGET = {"updates": 110160, "epochs_full": 1669, "partial_updates": 6, "validation_points": 1670,
                     "executed": False}


def epoch_budget(n_train: int, batch_size: int, epochs: int) -> dict:
    """Active epoch-based budget: no partial epoch, one validation point per epoch."""
    spe = steps_per_epoch(n_train, batch_size)
    if epochs < 1:
        raise ValueError("epochs must be positive")
    return {"epochs": int(epochs), "max_updates": None, "early_stopping": False, "batch_size": int(batch_size),
            "updates_per_epoch": spe, "total_updates": spe * int(epochs), "validation_points": int(epochs)}


def external_schedule(total: int, spe: int) -> dict:
    """SUPERSEDED (v1.0, never executed; not used by the active budget). Map a fixed update budget to full epochs + a final partial epoch.

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
