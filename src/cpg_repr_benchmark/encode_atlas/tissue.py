from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from cpg_repr_benchmark.data.methylation import read_axis
from cpg_repr_benchmark.encode_atlas.biology import blocked_probe, methylation_moments, residual_associations
from cpg_repr_benchmark.encode_atlas.campaign import check_prepared
from cpg_repr_benchmark.encode_atlas.features import FeatureData, digest, write_json
from cpg_repr_benchmark.encode_atlas.protocol import load_patient_protocol

# Deliberately restricted to exact organ names; cell-line origin is not guessed.
ORGANS = {'TCGA-BRCA': 'breast', 'TCGA-LIHC': 'liver', 'TCGA-LUAD': 'lung', 'TCGA-LUSC': 'lung',
          'TCGA-PRAD': 'prostate gland', 'TCGA-STAD': 'stomach', 'TCGA-COAD': 'colon',
          'TCGA-READ': 'colon', 'TCGA-KIRC': 'kidney', 'TCGA-KIRP': 'kidney', 'TCGA-KICH': 'kidney',
          'TCGA-THCA': 'thyroid gland', 'TCGA-PAAD': 'pancreas', 'TCGA-ESCA': 'esophagus',
          'TCGA-OV': 'ovary', 'TCGA-TGCT': 'testis', 'TCGA-ACC': 'adrenal gland', 'TCGA-THYM': 'thymus'}


def sample_labels(cfg):
    """Resolve the explicitly documented source table, join by sample identity (never row order)."""
    out = check_prepared(cfg)
    with h5py.File(cfg['methylation']) as h:
        provenance = json.loads(h['provenance_json'][()])
    root = Path(provenance.get('raw_root', ''))
    source = root / 'cancer_type.parquet'
    if not source.exists():
        write_json(out / 'tissue_status.json', {'status': 'unavailable', 'reason': 'source cancer table absent'})
        return None
    table = pd.read_parquet(source)
    if table.Sample.duplicated().any():
        raise ValueError('Ambiguous sample-to-cancer metadata')
    _, names = read_axis(Path(cfg['methylation']))
    labels = table.set_index('Sample').Cancer.reindex(names)
    if labels.isna().any():
        raise ValueError('Incomplete sample-to-cancer join')
    frame = pd.DataFrame({'sample_index': np.arange(len(names)), 'sample_name': names,
                          'cancer': labels.to_numpy()})
    frame.to_csv(out / 'sample_metadata.csv', index=False)
    write_json(out / 'sample_metadata_provenance.json', {
        'source': str(source), 'join': 'sample_name -> Sample, exact',
        'sha256': {str(p): digest(p) for p in sorted(source.rglob('*.parquet'))},
        'unknown_label': 'NA', 'counts': frame.cancer.value_counts().to_dict()})
    return frame


def tissue_screen(cfg):
    out = check_prepared(cfg)
    labels = sample_labels(cfg)
    if labels is None:
        return {'status': 'unavailable'}
    _, names = read_axis(Path(cfg['methylation']))
    patients = load_patient_protocol(out / 'patients.npz', names)['train']
    with np.load(out / 'discovery_targets.npz') as h:
        rows, global_mean = h['rows'], h['mean']
    data = FeatureData.read(Path(cfg['features']), Path(cfg['catalog']), rows=rows)
    registry = pd.read_parquet(out / 'loci.parquet').iloc[rows]
    x, dense = data.tracks, data.dense
    groups = labels.iloc[patients].groupby('cancer')
    completed = 0
    for cancer, subset in groups:
        if cancer == 'NA' or len(subset) < 30:
            continue
        dest = out / 'tissue' / f'{cancer}.json'
        if dest.exists():
            continue
        with h5py.File(cfg['methylation']) as h:
            mean, _, count = methylation_moments(h['beta'], subset.sample_index.to_numpy(), rows)
        contrast = mean - global_mean
        contrast[count < max(20, len(subset) * .5)] = np.nan
        associations = residual_associations(x, dense[:, :18], contrast, registry.chr.to_numpy())
        dest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'feature_column': np.arange(x.shape[1]), 'context_adjusted_r': associations}).to_csv(
            dest.with_suffix('.csv'), index=False)
        result = {'cancer': cancer, 'n_discovery_samples': len(subset),
                  'endpoint': 'mean_beta_in_cancer_minus_global_discovery_mean',
                  'organ': ORGANS.get(cancer), 'probes': {}}
        organ = ORGANS.get(cancer)
        matched = (data.catalog.encode_biosample.str.lower().eq(organ).to_numpy(copy=True)
                   if organ else np.zeros(x.shape[1], bool))
        matched &= data.catalog.encode_biosample_class.eq('tissue').to_numpy(copy=True)
        if matched.any():
            # Match irrelevant control tracks to the assay and overlap-prevalence decile.
            rng = np.random.default_rng(cfg['mask_seed'])
            prevalence = np.asarray(x.mean(axis=0)).ravel()
            bins = np.minimum((prevalence * 10).astype(int), 9)
            available = set(np.flatnonzero(~matched))
            retained, control = [], []
            for j in np.flatnonzero(matched):
                candidates = [k for k in sorted(available) if bins[k] == bins[j]
                              and data.catalog.iloc[k].encode_assay == data.catalog.iloc[j].encode_assay
                              and data.catalog.iloc[k].encode_biosample_class == 'tissue']
                if candidates:
                    k = int(rng.choice(candidates))
                    available.remove(k)
                    retained.append(int(j))
                    control.append(k)
            if retained:
                for label, columns in [('organ_matched', retained), ('assay_prevalence_matched_other', control)]:
                    result['probes'][label] = blocked_probe(x[:, columns], contrast, registry.chr.to_numpy())
                    result['probes'][label]['feature_columns'] = columns
        write_json(dest, result)
        completed += 1
        print(f'tissue {cancer}: {len(subset)} discovery samples', flush=True)
    return {'status': 'complete', 'new_cancer_profiles': completed}
