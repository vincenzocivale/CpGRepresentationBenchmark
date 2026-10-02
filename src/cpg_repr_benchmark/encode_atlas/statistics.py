from __future__ import annotations

import numpy as np
import pandas as pd


def bh_adjust(p):
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        q[order] = np.minimum(1, np.minimum.accumulate((p[order] * len(order)
                                                      / np.arange(1, len(order) + 1))[::-1])[::-1])
    return q


def prediction_table(outputs, registry, patient_names):
    columns = outputs['target_matrix_column']
    samples = np.broadcast_to(outputs['sample_index'][:, None], columns.shape)
    from cpg_repr_benchmark.data.splits import patient_id
    names = np.asarray([patient_id(x) for x in patient_names])
    table = pd.DataFrame({'patient': names[samples.ravel()], 'column': columns.ravel(),
                          'target': outputs['target'].ravel(), 'prediction': outputs['prediction'].ravel()})
    table['block'] = (registry.iloc[table.column.to_numpy()]['chr'].astype(str).to_numpy()
                       + ':' + (registry.iloc[table.column.to_numpy()]['pos'].to_numpy() // 1_000_000).astype(str))
    # Multiple panels can evaluate a patient/locus pair more than once. Average within pair.
    table['squared_error'] = (table.prediction - table.target) ** 2
    return table.groupby(['patient', 'column', 'block'], as_index=False).agg(
        target=('target', 'mean'), squared_error=('squared_error', 'mean'), n=('target', 'size'))


def paired_bootstrap(reference, alternative, *, seed=17, replicates=1000):
    """Crossed patient x 1Mb-block bootstrap of paired MSE; seed is not a biological unit."""
    key = ['patient', 'column', 'block']
    pairs = reference.merge(alternative, on=key, suffixes=('_reference', '_alternative'),
                            how='outer', validate='one_to_one', indicator=True)
    if (not pairs['_merge'].eq('both').all()
            or not np.allclose(pairs.target_reference, pairs.target_alternative, atol=1e-7)
            or not np.array_equal(pairs.n_reference, pairs.n_alternative)):
        raise ValueError('Predictions are not paired on identical patients, loci and observations')
    delta = (pairs.squared_error_alternative - pairs.squared_error_reference).to_numpy()
    baseline = pairs.squared_error_reference.to_numpy()
    observations = pairs.n_reference.to_numpy()
    point = float(np.average(delta, weights=observations))
    baseline_point = float(np.average(baseline, weights=observations))
    patients, pidx = np.unique(pairs.patient, return_inverse=True)
    blocks, bidx = np.unique(pairs.block, return_inverse=True)
    if len(patients) < 2 or len(blocks) < 2:
        raise ValueError('Need multiple patients and genomic blocks for uncertainty')
    # Aggregate once: a crossed bootstrap gives every observation in the same
    # patient/block cell the same weight. This avoids one full prediction-table
    # pass per replicate while preserving the exact resampling unit.
    flat = pidx * len(blocks) + bidx
    shape = (len(patients), len(blocks))
    delta_cells = np.bincount(flat, weights=observations * delta,
                              minlength=np.prod(shape)).reshape(shape)
    baseline_cells = np.bincount(flat, weights=observations * baseline,
                                 minlength=np.prod(shape)).reshape(shape)
    count_cells = np.bincount(flat, weights=observations,
                              minlength=np.prod(shape)).reshape(shape)
    rng = np.random.default_rng(seed)
    patient_weights, block_weights = [], []
    for _ in range(replicates):
        patient_weights.append(rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients))))
        block_weights.append(rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks))))
    pw = np.asarray(patient_weights)
    bw = np.asarray(block_weights)
    n = np.sum((pw @ count_cells) * bw, axis=1)
    weighted_delta = np.sum((pw @ delta_cells) * bw, axis=1)
    weighted_baseline = np.sum((pw @ baseline_cells) * bw, axis=1)
    estimates = weighted_delta[n > 0] / n[n > 0]
    relative = weighted_delta[weighted_baseline > 0] / weighted_baseline[weighted_baseline > 0]
    if not len(estimates) or not len(relative):
        raise ValueError('Degenerate bootstrap')
    # Centered bootstrap null, two-sided; finite Monte Carlo correction.
    p = (1 + np.sum(np.abs(estimates - point) >= abs(point))) / (1 + len(estimates))
    return {'delta_mse': point, 'relative_delta_mse': point / baseline_point,
            'ci95': np.quantile(estimates, [.025, .975]).tolist(),
            'relative_ci95': np.quantile(relative, [.025, .975]).tolist(), 'p': float(p),
            'n_patients': len(patients), 'n_blocks': len(blocks), 'n_pairs': len(pairs),
            'n_observations': int(observations.sum())}
