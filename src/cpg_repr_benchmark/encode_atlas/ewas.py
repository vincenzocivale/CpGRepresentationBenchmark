from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cpg_repr_benchmark.data.coordinates import encode_many
from cpg_repr_benchmark.encode_atlas.campaign import arm_id, check_prepared
from cpg_repr_benchmark.encode_atlas.features import digest, write_json


def matched_membership(labels, strata, *, seed=17, maximum=2000):
    """One unlisted locus per listed locus in the same predeclared stratum, no replacement."""
    rng = np.random.default_rng(seed)
    pairs = []
    for group in np.unique(strata):
        pos = np.flatnonzero(labels & (strata == group))
        neg = np.flatnonzero(~labels & (strata == group))
        size = min(len(pos), len(neg))
        if size:
            pairs.extend(zip(rng.choice(pos, size, replace=False), rng.choice(neg, size, replace=False)))
    if not pairs:
        return np.empty(0, dtype=int)
    pairs = np.asarray(pairs)
    selected = rng.choice(len(pairs), min(maximum, len(pairs)), replace=False)
    return np.sort(pairs[selected].ravel())


def evaluate_sets(cfg, *, only=None):
    out = check_prepared(cfg)
    root = Path(__file__).resolve().parents[3]
    sets = sorted((root / 'data/bio_annotations/known_sets').glob('*.npy'))
    registry = pd.read_parquet(out / 'loci.parquet')
    canonical = encode_many(registry.chr, registry.pos)
    with h5py.File(cfg['features']) as h:
        dense = h['dense'][:].astype(np.float32)
    strata = (registry.chr.astype(str).to_numpy() + ':' + np.argmax(dense[:, :4], axis=1).astype(str)
              + ':' + np.argmax(dense[:, 4:8], axis=1).astype(str)
              + ':' + np.minimum((dense[:, 22] * 10).astype(int), 9).astype(str))
    if only is None:
        chosen = json.loads((out / 'selection.json').read_text())['selected']
        only = ['full'] + ['add/' + r['group'] for r in chosen]
    results = []
    for source in sets:
        labels = np.isin(canonical, np.load(source, allow_pickle=False))
        rows = matched_membership(labels, strata, seed=cfg['mask_seed'])
        if len(rows) < 100 or len(np.unique(registry.chr.to_numpy()[rows])) < 5:
            results.append({'set': source.stem, 'status': 'insufficient_matched_support', 'n': len(rows)})
            continue
        y = labels[rows]
        chrom = registry.chr.to_numpy()[rows]
        for name in only:
            path = out / 'embeddings' / (arm_id(name) + '.h5')
            if not path.exists():
                results.append({'set': source.stem, 'arm': name, 'status': 'embedding_unavailable'})
                continue
            with h5py.File(path) as h:
                x = h['embedding'][rows].astype(np.float32)
            predictions = np.full(len(rows), np.nan)
            for train, test in GroupKFold(5).split(x, y, chrom):
                if len(np.unique(y[train])) < 2:
                    continue
                model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000))
                model.fit(x[train], y[train])
                predictions[test] = model.predict_proba(x[test])[:, 1]
            valid = np.isfinite(predictions)
            if not valid.all():
                results.append({'set': source.stem, 'arm': name, 'status': 'insufficient_fold_support'})
                continue
            results.append({'set': source.stem, 'arm': name, 'status': 'complete', 'n': len(rows),
                            'auc': float(roc_auc_score(y, predictions)),
                            'average_precision_matched_sample': float(average_precision_score(y, predictions)),
                            'positive_prevalence': float(y.mean()), 'source_sha256': digest(source)})
    pd.DataFrame(results).to_csv(out / 'ewas_matched_membership.csv', index=False)
    write_json(out / 'ewas_protocol.json', {
        'matching': ['chromosome', 'CpG context', 'gene region', 'core breadth decile'],
        'maximum_pairs': 2000, 'classifier': 'fixed C=1 logistic regression, 5 chromosome folds',
        'interpretation': 'exploratory set-membership discrimination against matched unlisted CpGs',
        'negative_status': 'unlisted, NOT proven unassociated; original study tested universes unavailable',
        'compaction_scope': 'label-free transductive cached embeddings; not strict unseen-representation OOD',
        'sets': [str(p) for p in sets],
    })
    return {'status': 'complete', 'rows': len(results)}
