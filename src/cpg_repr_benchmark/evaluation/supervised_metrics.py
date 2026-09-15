from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def supervised_metrics(task_type: str, prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if task_type == "regression":
        pred = prediction.reshape(-1)
        true = target.astype(np.float32).reshape(-1)
        finite = np.isfinite(pred) & np.isfinite(true)
        return {
            "mse": float(mean_squared_error(true[finite], pred[finite])),
            "mae": float(mean_absolute_error(true[finite], pred[finite])),
            "pearson": _pearson(true[finite], pred[finite]),
            "n_samples": int(finite.sum()),
        }
    if task_type == "binary":
        score = prediction.reshape(-1)
        true = target.astype(np.int64).reshape(-1)
        label = (score >= 0.5).astype(np.int64)
        result = {
            "accuracy": float(accuracy_score(true, label)),
            "balanced_accuracy": float(balanced_accuracy_score(true, label)),
            "f1": float(f1_score(true, label, zero_division=0)),
            "n_samples": int(len(true)),
        }
        if len(np.unique(true)) == 2:
            result["auroc"] = float(roc_auc_score(true, score))
            result["auprc"] = float(average_precision_score(true, score))
        else:
            result["auroc"] = float("nan")
            result["auprc"] = float("nan")
        return result
    if task_type == "multiclass":
        prob = prediction
        true = target.astype(np.int64).reshape(-1)
        label = np.argmax(prob, axis=1)
        result = {
            "accuracy": float(accuracy_score(true, label)),
            "balanced_accuracy": float(balanced_accuracy_score(true, label)),
            "macro_f1": float(f1_score(true, label, average="macro", zero_division=0)),
            "n_samples": int(len(true)),
        }
        if len(np.unique(true)) == prob.shape[1]:
            result["macro_auroc_ovr"] = float(roc_auc_score(true, prob, multi_class="ovr", average="macro"))
        else:
            result["macro_auroc_ovr"] = float("nan")
        return result
    raise ValueError(f"unsupported task_type={task_type!r}")
