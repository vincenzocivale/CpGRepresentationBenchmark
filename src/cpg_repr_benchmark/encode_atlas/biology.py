from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def methylation_moments(beta, patient_rows, columns, *, batch_size=16):
    """Streaming finite-data moments. Caller supplies discovery patients only."""
    columns = np.asarray(columns)
    count = np.zeros(len(columns), np.int64)
    total = np.zeros(len(columns), np.float64)
    squares = total.copy()
    for start in range(0, len(patient_rows), batch_size):
        rows = np.sort(patient_rows[start:start + batch_size])
        values = np.asarray(beta[rows, :], np.float64)[:, columns]
        finite = np.isfinite(values)
        values[~finite] = 0
        count += finite.sum(axis=0)
        total += values.sum(axis=0)
        squares += (values * values).sum(axis=0)
    mean = np.divide(total, count, out=np.full(len(columns), np.nan), where=count > 0)
    variance = np.divide(squares - total * np.nan_to_num(mean), count - 1,
                         out=np.full(len(columns), np.nan), where=count > 1)
    return mean, np.maximum(variance, 0), count


def blocked_probe(x, y, chromosomes, *, folds=5):
    """Nested chromosome CV; scaling and ridge selection never see outer-test loci."""
    valid = np.isfinite(y)
    x, y, chromosomes = x[valid], y[valid], np.asarray(chromosomes)[valid]
    if len(np.unique(chromosomes)) < 3 or len(y) < 30 or np.std(y) < 1e-12:
        return {'status': 'insufficient_support', 'n': len(y)}
    predictions = np.full(len(y), np.nan)
    fold_results = []
    outer = GroupKFold(n_splits=min(folds, len(np.unique(chromosomes))))
    for train, test in outer.split(x, y, chromosomes):
        inner = GroupKFold(n_splits=min(3, len(np.unique(chromosomes[train]))))
        model = GridSearchCV(make_pipeline(StandardScaler(with_mean=False), Ridge(solver='lsqr')),
                             {'ridge__alpha': [0.1, 10., 1000.]}, cv=inner,
                             scoring='neg_mean_squared_error', n_jobs=1)
        model.fit(x[train], y[train], groups=chromosomes[train])
        predictions[test] = model.predict(x[test])
        fold_results.append({'test_chromosomes': sorted(set(chromosomes[test].tolist())),
                             'mse': float(mean_squared_error(y[test], predictions[test])),
                             'alpha': float(model.best_params_['ridge__alpha'])})
    corr = np.corrcoef(y, predictions)[0, 1] if np.std(predictions) > 1e-12 else 0.
    return {'status': 'complete', 'n': len(y), 'mse': float(mean_squared_error(y, predictions)),
            'r2': float(r2_score(y, predictions)), 'pearson': float(corr), 'folds': fold_results}


def residual_associations(tracks, dense, target, chromosomes):
    """Cross-fitted context-adjusted associations, descriptive (no iid-locus p-values)."""
    valid = np.isfinite(target)
    tracks, dense, target = tracks[valid], dense[valid], target[valid]
    chromosomes = np.asarray(chromosomes)[valid]
    y_res = np.zeros(len(target))
    # Sparse cross-products avoid materializing a full N x tracks matrix.
    numerator = np.zeros(tracks.shape[1])
    x_energy = np.zeros(tracks.shape[1])
    for tr, te in GroupKFold(min(5, len(np.unique(chromosomes)))).split(dense, groups=chromosomes):
        ztr = np.column_stack([np.ones(len(tr)), dense[tr]])
        zte = np.column_stack([np.ones(len(te)), dense[te]])
        inv = np.linalg.pinv(ztr.T @ ztr + np.eye(ztr.shape[1]) * 1e-6)
        yhat = zte @ (inv @ ztr.T @ target[tr])
        residual = target[te] - yhat
        y_res[te] = residual
        coef = inv @ (tracks[tr].T @ ztr).T
        cross = np.asarray(tracks[te].T @ zte)
        numerator += np.asarray(tracks[te].T @ residual).ravel() - coef.T @ (zte.T @ residual)
        x_energy += np.asarray(tracks[te].power(2).sum(axis=0)).ravel()
        x_energy += np.sum(coef * ((zte.T @ zte) @ coef), axis=0) - 2 * np.sum(coef * cross.T, axis=0)
    return numerator / np.maximum(np.sqrt(np.maximum(x_energy, 0) * (y_res @ y_res)), 1e-12)


def co_methylation_neighbors(embedding, beta, chromosomes, positions, context, *, seed=17,
                             queries=500, min_distance=1_000_000, neighbors=5):
    """Far-locus neighbors vs distance/context/variance-matched reference candidates.

    beta is patients x loci from a declared evaluation split. The null does not use
    correlations to choose a match. Rows without enough eligible candidates are omitted.
    """
    rng = np.random.default_rng(seed)
    std = np.nanstd(beta, axis=0)
    bins = pd.qcut(pd.Series(std).rank(method='first'), 5, labels=False).to_numpy()
    norm = np.linalg.norm(embedding, axis=1)
    unit = embedding / np.maximum(norm[:, None], 1e-12)
    result = []
    for i in rng.choice(len(embedding), min(queries, len(embedding)), replace=False):
        distance = np.abs(positions - positions[i])
        inter = chromosomes != chromosomes[i]
        eligible = np.flatnonzero(inter | (distance >= min_distance))
        eligible = eligible[eligible != i]
        if len(eligible) < neighbors * 2:
            continue
        chosen = eligible[np.argsort(-(unit[eligible] @ unit[i]), kind='stable')[:neighbors]]
        for j in chosen:
            same_distance = inter if inter[j] else (~inter & (distance >= distance[j] / 2)
                                                   & (distance <= distance[j] * 2))
            candidates = eligible[(context[eligible] == context[j]) & (bins[eligible] == bins[j])
                                  & same_distance[eligible] & ~np.isin(eligible, chosen)]
            if not len(candidates):
                continue
            k = rng.choice(candidates)
            values = []
            for other in (j, k):
                ok = np.isfinite(beta[:, i]) & np.isfinite(beta[:, other])
                if ok.sum() < 20 or np.std(beta[ok, i]) == 0 or np.std(beta[ok, other]) == 0:
                    values.append(np.nan)
                else:
                    values.append(float(np.corrcoef(beta[ok, i], beta[ok, other])[0, 1]))
            result.append({'query': int(i), 'neighbor': int(j), 'matched': int(k),
                           'neighbor_correlation': values[0], 'matched_correlation': values[1]})
    return pd.DataFrame(result)


def balanced_panels(catalog, sizes, *, seed=17):
    """Round-robin random sampling by assay x biosample class, without replacement."""
    rng = np.random.default_rng(seed)
    pools = [list(rng.permutation(g.feature_column.to_numpy()))
             for _, g in catalog.groupby(['encode_assay', 'encode_biosample_class'], sort=True)]
    order = []
    while any(pools):
        for pool in pools:
            if pool:
                order.append(int(pool.pop()))
    return {int(n): order[:n] for n in sizes if n <= len(order)}


def ranked_panels(tracks, scores, sizes, *, redundancy_weight=0.5):
    """Discovery-only greedy relevance minus maximum absolute binary correlation."""
    n = tracks.shape[0]
    mean = np.asarray(tracks.mean(axis=0)).ravel()
    std = np.sqrt(mean * (1 - mean))
    relevance = rankdata(np.nan_to_num(np.abs(scores))) / len(scores)
    penalty = np.zeros(len(scores))
    selected = []
    for _ in range(min(max(sizes), tracks.shape[1])):
        value = relevance - redundancy_weight * penalty
        value[std == 0] = -np.inf
        value[penalty >= 1 - 1e-6] = -np.inf
        value[selected] = -np.inf
        if not np.isfinite(value).any():
            break
        j = int(np.argmax(value))
        selected.append(j)
        cross = (tracks.T @ tracks[:, j]).toarray().ravel() / n
        corr = (cross - mean * mean[j]) / np.maximum(std * std[j], 1e-12)
        penalty = np.maximum(penalty, np.abs(corr))
    return {int(n): selected[:n] for n in sizes if n <= len(selected)}


def add_context(x, dense):
    return sparse.hstack([sparse.csr_matrix(dense[:, :18]), x], format='csr')
