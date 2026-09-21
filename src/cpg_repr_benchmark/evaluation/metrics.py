from __future__ import annotations

import numpy as np


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _aligned_mac(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.ndim != 2 or target.shape != prediction.shape:
        return float("nan")
    values = []
    for j in range(prediction.shape[1]):
        m = np.isfinite(prediction[:, j]) & np.isfinite(target[:, j])
        values.append(_pearson(prediction[m, j], target[m, j]))
    return float(np.nanmean(values)) if values else float("nan")


def _grouped_mac(
    prediction: np.ndarray,
    target: np.ndarray,
    target_matrix_column: np.ndarray,
) -> float:
    columns = np.asarray(target_matrix_column)
    if columns.shape != prediction.shape:
        raise ValueError("target_matrix_column must match prediction shape")
    flat_col = columns.reshape(-1)
    flat_pred = prediction.reshape(-1)
    flat_true = target.reshape(-1)
    finite = np.isfinite(flat_pred) & np.isfinite(flat_true)
    flat_col, flat_pred, flat_true = flat_col[finite], flat_pred[finite], flat_true[finite]
    if len(flat_col) == 0:
        return float("nan")
    order = np.argsort(flat_col, kind="stable")
    flat_col, flat_pred, flat_true = flat_col[order], flat_pred[order], flat_true[order]
    cuts = np.flatnonzero(np.diff(flat_col)) + 1
    values = []
    for idx in np.split(np.arange(len(flat_col)), cuts):
        values.append(_pearson(flat_pred[idx], flat_true[idx]))
    return float(np.nanmean(values)) if values else float("nan")


def reconstruction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    prior_prediction: np.ndarray,
    *,
    target_matrix_column: np.ndarray | None = None,
    aligned_loci: bool = False,
) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    prior_prediction = np.asarray(prior_prediction, dtype=np.float32)
    if prediction.shape != target.shape or prediction.shape != prior_prediction.shape:
        raise ValueError("prediction, target and prior_prediction must have identical shapes")
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
    mas_pcc = float(np.nanmean(patient_corr))
    if aligned_loci:
        mac_pcc = _aligned_mac(prediction, target)
    elif target_matrix_column is not None:
        mac_pcc = _grouped_mac(prediction, target, target_matrix_column)
    else:
        mac_pcc = float("nan")
    return {
        "mse": mse,
        "mae": float(np.abs(prediction[finite] - target[finite]).mean()),
        "prior_mse": prior_mse,
        "skill_vs_prior": float(1.0 - mse / prior_mse) if prior_mse > 0 else float("nan"),
        "mas_pcc": mas_pcc,
        "mac_pcc": mac_pcc,
        # Backwards-compatible alias used by old masking summaries.
        "patient_pearson": mas_pcc,
        "n_pairs": int(finite.sum()),
    }
