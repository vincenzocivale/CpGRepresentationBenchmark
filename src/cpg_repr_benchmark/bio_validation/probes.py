from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

from cpg_repr_benchmark.bio_validation.annotations import CpGAnnotations
from cpg_repr_benchmark.probing.metrics import classification_metrics


def _cv_classification(embedding: np.ndarray, label: np.ndarray, *, seed: int, folds: int = 5) -> dict[str, Any]:
    """K-fold cross-validated linear probe of a CpG-locus embedding against one categorical
    label. Cross-validation (rather than a single split, as in `probing.linear_probe`) is used
    here because per-CpG annotation cohorts are typically small and class-imbalanced (e.g. a
    handful of clock CpGs among tens of thousands), so a single held-out split would be noisy.
    """
    embedding = np.asarray(embedding, dtype=np.float64)
    label = np.asarray(label, dtype=np.int64)
    if len(np.unique(label)) < 2:
        return {"error": "label has fewer than two classes", "n": len(label)}
    splitter = StratifiedKFold(n_splits=min(folds, int(np.bincount(label).min())), shuffle=True, random_state=seed)
    fold_metrics = []
    for train_idx, test_idx in splitter.split(embedding, label):
        mean = embedding[train_idx].mean(axis=0)
        std = embedding[train_idx].std(axis=0)
        std[std < 1e-8] = 1.0
        z_train = (embedding[train_idx] - mean) / std
        z_test = (embedding[test_idx] - mean) / std
        estimator = LogisticRegressionCV(max_iter=2000)
        estimator.fit(z_train, label[train_idx])
        prob = estimator.predict_proba(z_test)[:, 1]
        fold_metrics.append(classification_metrics(prob, label[test_idx]))
    aggregated = {
        key: float(np.nanmean([m[key] for m in fold_metrics]))
        for key in fold_metrics[0]
        if key != "n"
    }
    aggregated["n"] = len(label)
    aggregated["n_folds"] = len(fold_metrics)
    return aggregated


def bio_validation_report(
    *,
    embedding: np.ndarray,
    cpg_idx: np.ndarray,
    annotations: CpGAnnotations,
    seed: int = 17,
) -> dict[str, Any]:
    """Evaluate whether a CpG-locus embedding linearly separates biological categories that
    were never part of the ENCODE feature construction: genomic context (island/shore/shelf,
    gene relationship) and literature-curated known CpG sets (e.g. epigenetic-clock CpGs).
    Each annotation is its own classification target; a strong probe score means the
    representation preserves that biological signal even though it was not explicitly given.
    """
    cpg_idx = np.asarray(cpg_idx, dtype=np.int64)
    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.shape[0] != len(cpg_idx):
        raise ValueError("embedding rows must align with cpg_idx")
    index_lookup = {int(c): i for i, c in enumerate(cpg_idx)}

    report: dict[str, Any] = {"genomic_context": {}, "known_cpg_sets": {}}

    if not annotations.genomic_context.empty:
        shared = annotations.genomic_context.index.intersection(cpg_idx)
        if len(shared) >= 10:
            rows = np.array([index_lookup[int(c)] for c in shared])
            categories = annotations.genomic_context.loc[shared]
            for category in categories.unique():
                label = (categories == category).to_numpy(dtype=np.int64)
                report["genomic_context"][str(category)] = _cv_classification(embedding[rows], label, seed=seed)

    for set_name, set_ids in annotations.known_sets.items():
        member = np.isin(cpg_idx, set_ids)
        if 10 <= int(member.sum()) <= len(cpg_idx) - 10:
            report["known_cpg_sets"][set_name] = _cv_classification(embedding, member.astype(np.int64), seed=seed)
        else:
            report["known_cpg_sets"][set_name] = {"error": "membership too small/large for a balanced probe"}

    return report
