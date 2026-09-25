from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from cpg_repr_benchmark.bio_validation.annotations import CpGAnnotations
from cpg_repr_benchmark.bio_validation.metrics import classification_metrics, regression_metrics


def _cv_classification(embedding: np.ndarray, label: np.ndarray, *, seed: int, folds: int = 5) -> dict[str, Any]:
    """K-fold cross-validated linear probe of a CpG-locus embedding against one categorical
    label. Cross-validation (rather than a single held-out split) is used here because per-CpG annotation cohorts are typically small and class-imbalanced (e.g. a
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


def _cv_regression(embedding: np.ndarray, target: np.ndarray, *, seed: int, folds: int = 5) -> dict[str, Any]:
    """K-fold cross-validated ridge probe of a CpG-locus embedding against a continuous
    per-CpG weight (e.g. a published clock's elastic-net coefficient). Stricter than the
    binary membership probe: it asks whether the embedding predicts the *magnitude* a CpG
    matters to the clock, not just whether the CpG is in the clock at all.
    """
    embedding = np.asarray(embedding, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n_folds = min(folds, len(target))
    if n_folds < 2:
        return {"error": "too few CpGs for a cross-validated regression probe", "n": len(target)}
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []
    for train_idx, test_idx in splitter.split(embedding):
        mean = embedding[train_idx].mean(axis=0)
        std = embedding[train_idx].std(axis=0)
        std[std < 1e-8] = 1.0
        z_train = (embedding[train_idx] - mean) / std
        z_test = (embedding[test_idx] - mean) / std
        estimator = RidgeCV(alphas=np.logspace(-2, 4, 13))
        estimator.fit(z_train, target[train_idx])
        pred = estimator.predict(z_test)
        fold_metrics.append(regression_metrics(pred, target[test_idx]))
    aggregated = {
        key: float(np.nanmean([m[key] for m in fold_metrics]))
        for key in fold_metrics[0]
        if key != "n"
    }
    aggregated["n"] = len(target)
    aggregated["n_folds"] = len(fold_metrics)
    return aggregated


def locality_probe(
    embedding: np.ndarray,
    cpg_idx: np.ndarray,
    registry: pd.DataFrame,
    *,
    k: int = 10,
    distance_kb: float = 100.0,
    sample_size: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    """Unsupervised locality probe: no annotation labels or training involved, unlike every
    other probe in this module. For a random sample of CpGs, finds their k nearest neighbors
    in embedding space (cosine distance) and measures how often those neighbors are also
    genomically nearby (same chromosome, within `distance_kb`), against the rate expected from
    random pairing. A representation that encodes genomic locality structure — even though it
    was never told to — should score well above the random-pairing baseline.

    Uses Euclidean distance (each dimension z-scored first) rather than cosine similarity:
    cosine ignores vector magnitude entirely, which would make single-direction/near-collinear
    embeddings (e.g. a monotone function of genomic position) look like ties regardless of how
    far apart the underlying positions are.

    `registry` must have columns `cpg_idx`, `chr`, `pos` (e.g. `data/cpg/master_cpg_registry.parquet`).
    """
    cpg_idx = np.asarray(cpg_idx, dtype=np.int64)
    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.shape[0] != len(cpg_idx):
        raise ValueError("embedding rows must align with cpg_idx")

    coords = registry.set_index("cpg_idx")[["chr", "pos"]]
    shared = coords.index.intersection(cpg_idx)
    if len(shared) < k + 2:
        return {"error": "too few CpGs with registry coordinates for a locality probe", "n": len(shared)}

    index_lookup = {int(c): i for i, c in enumerate(cpg_idx)}
    rows = np.array([index_lookup[int(c)] for c in shared])
    sub_embedding = embedding[rows]
    sub_chr = coords.loc[shared, "chr"].to_numpy()
    sub_pos = coords.loc[shared, "pos"].to_numpy(dtype=np.float64)

    std = sub_embedding.std(axis=0)
    std[std < 1e-8] = 1.0
    sub_embedding = (sub_embedding - sub_embedding.mean(axis=0)) / std

    rng = np.random.default_rng(seed)
    n = len(shared)
    sample_n = min(sample_size, n)
    sample_idx = rng.choice(n, size=sample_n, replace=False)

    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean").fit(sub_embedding)
    _, neighbor_idx = nn.kneighbors(sub_embedding[sample_idx])
    neighbor_idx = neighbor_idx[:, 1:]  # drop self-match

    def _within_distance(query_idx: np.ndarray, other_idx: np.ndarray) -> np.ndarray:
        same_chr = sub_chr[query_idx][:, None] == sub_chr[other_idx]
        dist_kb = np.abs(sub_pos[query_idx][:, None] - sub_pos[other_idx]) / 1000.0
        return same_chr & (dist_kb <= distance_kb)

    neighbor_hits = _within_distance(sample_idx, neighbor_idx)
    neighbor_hit_rate = float(neighbor_hits.mean())

    random_other = rng.integers(0, n, size=(sample_n, neighbor_idx.shape[1]))
    null_hits = _within_distance(sample_idx, random_other)
    null_hit_rate = float(null_hits.mean())

    return {
        "neighbor_hit_rate": neighbor_hit_rate,
        "null_hit_rate": null_hit_rate,
        "enrichment": neighbor_hit_rate / null_hit_rate if null_hit_rate > 0 else float("inf"),
        "k": int(neighbor_idx.shape[1]),
        "distance_kb": distance_kb,
        "n_sampled": sample_n,
        "n_universe": n,
    }


def bio_validation_report(
    *,
    embedding: np.ndarray,
    cpg_idx: np.ndarray,
    annotations: CpGAnnotations,
    seed: int = 17,
    locality_registry: pd.DataFrame | None = None,
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

    report: dict[str, Any] = {
        "genomic_context": {},
        "known_cpg_sets": {},
        "clock_coefficients": {},
        "locality": {},
    }

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

    for set_name, weights in annotations.coefficients.items():
        shared = weights.index.intersection(cpg_idx)
        if len(shared) >= 10:
            rows = np.array([index_lookup[int(c)] for c in shared])
            target = weights.loc[shared].to_numpy(dtype=np.float64)
            report["clock_coefficients"][set_name] = _cv_regression(embedding[rows], target, seed=seed)
        else:
            report["clock_coefficients"][set_name] = {"error": "too few overlapping CpGs for a regression probe"}

    if locality_registry is not None:
        report["locality"] = locality_probe(embedding, cpg_idx, locality_registry, seed=seed)

    return report
