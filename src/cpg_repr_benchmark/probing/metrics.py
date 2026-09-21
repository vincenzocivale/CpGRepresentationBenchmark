from __future__ import annotations

import numpy as np


def regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float | int]:
    """Same contract as `downstream.frozen_linear.regression_metrics` — kept here so
    embedding-probing code does not depend on the raw-beta downstream module."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    pearson = float("nan")
    if len(true) >= 2 and np.std(pred) > 0 and np.std(true) > 0:
        pearson = float(np.corrcoef(pred, true)[0, 1])
    return {
        "mae": float(np.mean(np.abs(pred - true))),
        "pearson": pearson,
        "r2": float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot,
        "n": len(true),
    }


def classification_metrics(prob: np.ndarray, true: np.ndarray) -> dict[str, float | int]:
    """Binary classification metrics for `prob` (predicted P(y=1)) against 0/1 `true`."""
    prob = np.asarray(prob, dtype=np.float64)
    true = np.asarray(true, dtype=np.int64)
    pred = (prob >= 0.5).astype(np.int64)
    accuracy = float(np.mean(pred == true))
    tp = int(np.sum((pred == 1) & (true == 1)))
    fp = int(np.sum((pred == 1) & (true == 0)))
    fn = int(np.sum((pred == 0) & (true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if not (np.isnan(precision) or np.isnan(recall)) and (precision + recall) > 0
        else float("nan")
    )
    auc = _roc_auc(prob, true)
    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "n": len(true),
    }


def _roc_auc(prob: np.ndarray, true: np.ndarray) -> float:
    positives = true == 1
    negatives = true == 0
    n_pos, n_neg = int(positives.sum()), int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(prob, kind="stable")
    ranks = np.empty(len(prob), dtype=np.float64)
    ranks[order] = np.arange(1, len(prob) + 1)
    # Average tied ranks.
    sorted_prob = prob[order]
    i = 0
    while i < len(sorted_prob):
        j = i
        while j + 1 < len(sorted_prob) and sorted_prob[j + 1] == sorted_prob[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    sum_ranks_pos = float(ranks[positives].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def probing_metrics(pred: np.ndarray, true: np.ndarray, *, task_type: str) -> dict[str, float | int]:
    if task_type == "regression":
        return regression_metrics(pred, true)
    if task_type == "classification":
        return classification_metrics(pred, true)
    raise ValueError(f"unsupported task_type: {task_type!r}")
