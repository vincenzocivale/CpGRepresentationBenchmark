"""Discovery-derived redundancy controls and frozen-model complementarity arms."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse.csgraph import connected_components

from cpg_repr_benchmark.encode_atlas.campaign import check_prepared, macro_specs
from cpg_repr_benchmark.encode_atlas.features import FeatureData, write_json


def augment(cfg):
    out = check_prepared(cfg)
    data = FeatureData.read(Path(cfg['features']), Path(cfg['catalog']))
    with np.load(out / 'discovery_targets.npz') as h:
        rows = h['rows']
    x = data.tracks[rows]
    mean = np.asarray(x.mean(axis=0)).ravel()
    std = np.sqrt(mean * (1 - mean))
    cross = (x.T @ x).toarray() / len(rows)
    corr = (cross - mean[:, None] * mean[None, :]) / np.maximum(std[:, None] * std[None, :], 1e-12)
    adjacency = np.abs(corr) >= .9
    np.fill_diagonal(adjacency, False)
    a, b = np.where(np.triu(adjacency, 1))
    pd.DataFrame({'column_a': a, 'column_b': b, 'correlation': corr[a, b]}).to_csv(
        out / 'redundancy_edges.csv', index=False)
    _, labels = connected_components(adjacency, directed=False)
    specs = macro_specs(data.catalog)
    clusters = []
    representative = []
    for label in np.unique(labels):
        columns = np.flatnonzero(labels == label).tolist()
        representative.append(columns[0])
        if len(columns) > 1:
            clusters.append({'cluster': int(label), 'columns': columns})
            # All correlated clusters are retained as candidates; no favorable-result filtering.
            specs[f'drop_cluster/{label}'] = {'tracks': sorted(set(range(x.shape[1])) - set(columns)),
                                             'dense': list(range(23)), 'family': 'redundancy'}
    specs['cluster_representatives'] = {'tracks': representative, 'dense': list(range(23)),
                                         'family': 'redundancy'}
    write_json(out / 'redundancy_clusters.json', {'threshold': .9, 'fit_rows': rows.tolist(),
                                                'clusters': clusters, 'interpretation': 'discovery-sample correlation'})
    root = Path(__file__).resolve().parents[3]
    catalog = yaml.safe_load((root / 'configs/representations/future_models.yaml').read_text())['representations']
    for key in ['cpgpt_locus_native', 'cpgpt_locus_large_native', 'deepcpg_dna_locus_native']:
        model = catalog[key]
        path = str((root / model['store_h5']).resolve())
        if not Path(path).exists():
            continue
        specs['fm/' + model['name']] = {'tracks': [], 'dense': [], 'external_store': path,
                                        'family': 'fm_control'}
        specs['full_plus/' + model['name']] = {'tracks': list(range(x.shape[1])), 'dense': list(range(23)),
                                               'external_store': path, 'family': 'fm_complementarity'}
    write_json(out / 'arms.json', specs)
    return {'arms': len(specs), 'redundancy_clusters': len(clusters), 'correlated_pairs': len(a)}
