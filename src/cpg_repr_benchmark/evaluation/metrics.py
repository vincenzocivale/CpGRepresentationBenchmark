from __future__ import annotations

import numpy as np


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def reconstruction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    prior_prediction: np.ndarray,
) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    prior_prediction = np.asarray(prior_prediction, dtype=np.float32)
    finite = np.isfinite(prediction) & np.isfinite(target) & np.isfinite(prior_prediction)
    if not finite.any():
        raise ValueError("no finite prediction/target pairs")
    squared = (prediction[finite] - target[finite]) ** 2
    prior_squared = (prior_prediction[finite] - target[finite]) ** 2
    mse = float(squared.mean())
    prior_mse = float(prior_squared.mean())
    patient_corr = []
    for p, t in zip(prediction, target):
        m = np.isfinite(p) & np.isfinite(t)
        patient_corr.append(_pearson(p[m], t[m]))
    return {
        "mse": mse,
        "mae": float(np.abs(prediction[finite] - target[finite]).mean()),
        "prior_mse": prior_mse,
        "skill_vs_prior": float(1.0 - mse / prior_mse) if prior_mse > 0 else float("nan"),
        "patient_pearson": float(np.nanmean(patient_corr)),
        "n_pairs": int(finite.sum()),
    }
