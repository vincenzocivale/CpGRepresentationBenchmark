"""Prespecified, affordable confirmation of the ENCODE attribution story."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cpg_repr_benchmark.encode_atlas.campaign import check_prepared, select
from cpg_repr_benchmark.encode_atlas.features import write_json


PRIMARY_ARMS = (
    'full',
    'context_only',
    'add/assay/Histone ChIP-seq',
    'drop/assay/Histone ChIP-seq',
    'drop_preserve_breadth/assay/Histone ChIP-seq',
    'fm/cpgpt_locus_large',
    'fm/deepcpg_dna_locus',
    'balanced_assays/17',
    'cluster_representatives',
    'control/histone_prevalence_matched',
    'control/non_histone_prevalence_matched',
    'only/target/H3K4me3',
    'add/target/H3K4me3',
)
OOD_ARMS = (
    'full',
    'context_only',
    'add/assay/Histone ChIP-seq',
    'fm/cpgpt_locus_large',
)
BIOLOGY_ARMS = OOD_ARMS


def prevalence_matched_assays(catalog: pd.DataFrame, *, seed=17) -> tuple[dict, dict, dict]:
    """Subsample histone and other tracks within 1%-wide peak-prevalence bins."""
    rng = np.random.default_rng(seed)
    frame = catalog.copy()
    frame['bin'] = np.minimum((frame.prevalence.to_numpy() * 100).astype(int), 99)
    histone = frame.encode_assay.eq('Histone ChIP-seq')
    left, right = [], []
    for _, subset in frame.groupby('bin', sort=True):
        matched = subset.encode_assay.eq('Histone ChIP-seq')
        h = subset.loc[matched, 'feature_column'].to_numpy(dtype=int)
        o = subset.loc[~matched, 'feature_column'].to_numpy(dtype=int)
        count = min(len(h), len(o))
        if count:
            left.extend(rng.choice(h, count, replace=False).tolist())
            right.extend(rng.choice(o, count, replace=False).tolist())
    if not left:
        raise ValueError('No prevalence-overlap tracks for the assay control')
    spec_h = {'tracks': sorted(left), 'dense': list(range(18)), 'family': 'coverage_control'}
    spec_o = {'tracks': sorted(right), 'dense': list(range(18)), 'family': 'coverage_control'}
    audit = {'n_tracks_each': len(left), 'matching': 'equal random count per 1%-wide prevalence bin',
             'seed': seed, 'original_histone_tracks': int(histone.sum()),
             'histone_tracks_without_bin_match': int(histone.sum() - len(left)),
             'interpretation': 'common-support comparison; unmatched high-prevalence histone tracks excluded'}
    return spec_h, spec_o, audit


def focused_selection(cfg):
    """Freeze all confirmation arms before evaluating test patients."""
    out = check_prepared(cfg)
    plan_path = out / 'focused_plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        discovery = json.loads((out / 'arms.json').read_text())
        confirmation = json.loads((out / 'confirmation_arms.json').read_text())
        if (set(plan['primary_arms']) != set(confirmation)
                or any(name not in discovery for name in plan['primary_arms'])):
            raise ValueError('Focused plan differs from persisted arm specifications')
        return plan

    # select() reads only the completed chromosome-blocked discovery screen. It does
    # not inspect decoder validation results or the confirmation/test split.
    select(cfg)
    selected = json.loads((out / 'selection.json').read_text())
    if 'target/H3K4me3' not in {row['group'] for row in selected['selected']}:
        raise ValueError('H3K4me3 did not satisfy the frozen subgroup selection rule')
    catalog = pd.read_csv(out / 'feature_inventory.csv')
    matched_histone, matched_other, audit = prevalence_matched_assays(catalog)
    discovery = json.loads((out / 'arms.json').read_text())
    confirmation = json.loads((out / 'confirmation_arms.json').read_text())
    controls = {'control/histone_prevalence_matched': matched_histone,
                'control/non_histone_prevalence_matched': matched_other}
    discovery.update(controls)
    confirmation.update(controls)
    # Selected target arms are absent from the original discovery arm list.
    for name in ('only/target/H3K4me3', 'add/target/H3K4me3'):
        discovery[name] = confirmation[name]
    missing = set(PRIMARY_ARMS) - set(discovery) | set(PRIMARY_ARMS) - set(confirmation)
    if missing:
        raise ValueError(f'Missing focused arm specifications: {sorted(missing)}')
    write_json(out / 'arms.json', discovery)
    write_json(out / 'confirmation_arms.json', {name: confirmation[name] for name in PRIMARY_ARMS})
    plan = {'status': 'frozen', 'selection_source': 'complete discovery screen only',
            'primary_arms': list(PRIMARY_ARMS), 'ood_arms': list(OOD_ARMS),
            'biology_arms': list(BIOLOGY_ARMS), 'decoder_seeds': cfg['seeds'],
            'primary_endpoint': '50% masking MSE on frozen test patients',
            'prevalence_matched_control': audit,
            'test_split_rule': 'no changes to arm list after confirmation begins'}
    write_json(plan_path, plan)
    return plan
