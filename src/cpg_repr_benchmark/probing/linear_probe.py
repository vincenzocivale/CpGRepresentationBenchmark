from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegressionCV, RidgeCV

from cpg_repr_benchmark.probing.metrics import classification_metrics, regression_metrics

_DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


@dataclass
class LinearProbeResult:
    """A frozen linear probe fit on a patient embedding, plus the metrics used to select it.

    Unlike `downstream.frozen_linear.FrozenRidgeRegressor` (which operates on raw beta and
    uses a dual-form solve because the feature dimension is huge), embeddings are low
    dimensional so a plain primal sklearn estimator is used and wrapped here for a uniform
    save/load/predict contract.
    """

    task_type: str
    estimator: Any
    feature_mean: np.ndarray
    feature_std: np.ndarray
    alpha: float
    metrics: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.feature_mean) / self.feature_std

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self._standardize(x)
        if self.task_type == "classification":
            return self.estimator.predict_proba(z)[:, 1]
        return self.estimator.predict(z)

    def save(self, path: Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "task_type": self.task_type,
                "estimator": self.estimator,
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "alpha": self.alpha,
            },
            path,
        )
        metrics_path = path.with_suffix(path.suffix + ".metrics.json")
        metrics_path.write_text(json.dumps(self.metrics, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> LinearProbeResult:
        import joblib

        data = joblib.load(path)
        return cls(
            task_type=data["task_type"],
            estimator=data["estimator"],
            feature_mean=data["feature_mean"],
            feature_std=data["feature_std"],
            alpha=data["alpha"],
        )


def fit_linear_probe(
    *,
    embedding: np.ndarray,
    target: np.ndarray,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    task_type: str,
    alphas: tuple[float, ...] = _DEFAULT_ALPHAS,
) -> LinearProbeResult:
    """Fit a frozen linear probe on a patient embedding: train-only standardization and
    alpha selection, then a single held-out evaluation on `test_rows`.

    `embedding` rows and `target` entries must already be aligned to the same patient order
    (see `scripts/run_embedding_probe.py` for how patient_id alignment is done upstream).
    """
    embedding = np.asarray(embedding, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    train_rows = np.asarray(train_rows, dtype=np.int64)
    test_rows = np.asarray(test_rows, dtype=np.int64)

    feature_mean = embedding[train_rows].mean(axis=0)
    feature_std = embedding[train_rows].std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0

    z_train = (embedding[train_rows] - feature_mean) / feature_std
    z_test = (embedding[test_rows] - feature_mean) / feature_std

    if task_type == "regression":
        estimator = RidgeCV(alphas=alphas)
        estimator.fit(z_train, target[train_rows])
        alpha = float(estimator.alpha_)
        train_pred = estimator.predict(z_train)
        test_pred = estimator.predict(z_test)
        metrics = {
            "train": regression_metrics(train_pred, target[train_rows]),
            "test": regression_metrics(test_pred, target[test_rows]),
        }
    elif task_type == "classification":
        estimator = LogisticRegressionCV(Cs=[1.0 / a for a in alphas], max_iter=2000)
        estimator.fit(z_train, target[train_rows].astype(np.int64))
        alpha = float(1.0 / estimator.C_[0])
        train_pred = estimator.predict_proba(z_train)[:, 1]
        test_pred = estimator.predict_proba(z_test)[:, 1]
        metrics = {
            "train": classification_metrics(train_pred, target[train_rows]),
            "test": classification_metrics(test_pred, target[test_rows]),
        }
    else:
        raise ValueError(f"unsupported task_type: {task_type!r}")

    return LinearProbeResult(
        task_type=task_type,
        estimator=estimator,
        feature_mean=feature_mean,
        feature_std=feature_std,
        alpha=alpha,
        metrics=metrics,
    )
