from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from cpg_repr_benchmark.config import config_fingerprint
from cpg_repr_benchmark.data.methylation import read_axis
from cpg_repr_benchmark.encode_atlas.biology import (
    add_context,
    balanced_panels,
    blocked_probe,
    methylation_moments,
    ranked_panels,
    residual_associations,
)
from cpg_repr_benchmark.encode_atlas.features import (
    DENSE_NAMES,
    FeatureData,
    digest,
    group_catalog,
    load_catalog,
    materialize,
    write_json,
)
from cpg_repr_benchmark.encode_atlas.protocol import freeze_patient_protocol, load_patient_protocol


def arm_id(name):
    import re
    return re.sub('[^a-z0-9]+', '_', name.lower()).strip('_')[:80] + '_' + hashlib.sha256(
        name.encode()).hexdigest()[:8]


def load_campaign(path):
    cfg = yaml.safe_load(Path(path).read_text())
    root = Path(__file__).resolve().parents[3]
    for key in ('output', 'features', 'catalog', 'registry', 'methylation', 'benchmark_template'):
        cfg[key] = str((root / cfg[key]).resolve())
    return cfg


def registry_for(ids, path):
    frame = pd.read_parquet(path)
    if frame.cpg_idx.duplicated().any():
        raise ValueError('Duplicate registry IDs')
    frame = frame.set_index('cpg_idx').reindex(ids)
    if frame[['chr', 'pos']].isna().any().any() or frame[['chr', 'pos']].duplicated().any():
        raise ValueError('Missing or duplicate registry coordinates')
    return frame.reset_index()


def prepare(cfg):
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / 'campaign.json'
    identity = {k: cfg[k] for k in cfg if k != 'output'}
    hashes = {k: digest(cfg[k]) for k in ('features', 'catalog', 'registry', 'benchmark_template')}
    # Hash methylation source as well: long-lived scientific caches must not rely on mtimes.
    hashes['methylation'] = digest(cfg['methylation'])
    identity['source_sha256'] = hashes
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError('Campaign inputs changed: use a new output directory')
    data = FeatureData.read(Path(cfg['features']), Path(cfg['catalog']))
    ids, names = read_axis(Path(cfg['methylation']))
    if not np.array_equal(data.ids, ids):
        raise ValueError('Feature axis must equal the dataset axis; no implicit intersections')
    registry = registry_for(ids, cfg['registry'])
    freeze_patient_protocol(out / 'patients.npz', names, cfg['patient_split_seed'])
    registry.to_parquet(out / 'loci.parquet', index=False)
    data.catalog.to_parquet(out / 'tracks.parquet', index=False)
    write_json(out / 'groups.json', group_catalog(data.catalog))
    write_json(out / 'dense_features.json', {str(i): name for i, name in enumerate(DENSE_NAMES)})
    write_json(frozen, identity)
    prevalence = np.asarray(data.tracks.mean(axis=0)).ravel()
    catalog = data.catalog.copy()
    catalog['prevalence'] = prevalence
    catalog['n_loci'] = np.asarray(data.tracks.sum(axis=0)).ravel().astype(int)
    catalog.to_csv(out / 'feature_inventory.csv', index=False)
    write_json(out / 'audit.json', {
        'n_loci': len(ids), 'n_tracks': len(catalog), 'n_dense': data.dense.shape[1],
        'mapping': 'CSR column = catalog row ordinal, NOT original track_index',
        'namespace': 'IDs validated against array registry and methylation axis; source attribute overridden',
        'breadth_verified': True, 'source_sha256': hashes,
        'prior_runs': 'New protocol may overlap historical training; not a historically untouched cohort',
        'annotation_independence': {'context_gene_ccre_tss': 'input-derived',
                                    'methylation_properties': 'discovery-patient-derived'},
        'assays': catalog.encode_assay.value_counts().to_dict(),
        'biosample_classes': catalog.encode_biosample_class.value_counts().to_dict(),
    })
    specs = macro_specs(data.catalog)
    write_json(out / 'arms.json', specs)
    return {'status': 'prepared', 'arms': len(specs), 'n_loci': len(ids)}


def macro_specs(catalog):
    all_tracks = list(range(len(catalog)))
    specs = {'full': {'tracks': all_tracks, 'dense': list(range(23)), 'family': 'reference'},
             'null': {'tracks': [], 'dense': [], 'family': 'control'},
             'context_only': {'tracks': [], 'dense': list(range(18)), 'family': 'control'},
             'breadth_only': {'tracks': [], 'dense': list(range(18, 23)),
                              'breadth_policy': 'preserve', 'family': 'control'},
             'tracks_only': {'tracks': all_tracks, 'dense': [], 'family': 'control'},
             'full_balanced': {'tracks': all_tracks, 'dense': list(range(23)),
                               'scaling': 'balanced', 'family': 'scaling'}}
    for name, group in group_catalog(catalog).items():
        if group['family'] not in {'dense', 'assay', 'biosample_class'}:
            continue
        specs['only/' + name] = {**group, 'breadth_policy': 'preserve' if name == 'dense/breadth'
                                 else 'recompute'}
        specs['drop/' + name] = {'tracks': sorted(set(all_tracks) - set(group['tracks'])),
                                 'dense': sorted(set(range(23)) - set(group['dense'])),
                                 'family': group['family']}
        if group['tracks']:
            specs['add/' + name] = {**group, 'dense': list(range(18))}
            specs['drop_preserve_breadth/' + name] = {**specs['drop/' + name], 'breadth_policy': 'preserve'}
    assays = list(catalog.groupby('encode_assay', sort=True))
    for (a, x), (b, y) in itertools.combinations(assays, 2):
        specs[f'pair/{a}+{b}'] = {'tracks': sorted(x.feature_column.tolist() + y.feature_column.tolist()),
                                 'dense': list(range(18)), 'family': 'interaction'}
    # Equal assay budgets distinguish annotation identity from number of experiments.
    minimum = min(len(g) for _, g in assays)
    for seed in (17, 42, 97):
        rng = np.random.default_rng(seed)
        balanced = sorted(np.concatenate([rng.choice(g.feature_column, minimum, replace=False)
                                           for _, g in assays]).astype(int).tolist())
        specs[f'balanced_assays/{seed}'] = {'tracks': balanced, 'dense': list(range(23)),
                                          'family': 'coverage_control'}
    # Remove duplicate experiments, keeping an explicit alias for interpretation/reporting.
    seen = {}
    for name, spec in specs.items():
        key = json.dumps({k: spec.get(k, default) for k, default in
                          [('tracks', []), ('dense', []), ('breadth_policy', 'recompute'),
                           ('scaling', 'legacy')]}, sort_keys=True)
        if key in seen:
            spec['alias'] = seen[key]
        else:
            seen[key] = name
    return specs


def check_prepared(cfg):
    out = Path(cfg['output'])
    frozen = json.loads((out / 'campaign.json').read_text())
    if {k: v for k, v in frozen.items() if k != 'source_sha256'} != {k: v for k, v in cfg.items() if k != 'output'}:
        raise ValueError('Configuration differs from frozen campaign')
    verified_path = out / 'verified_inputs.json'
    verified = json.loads(verified_path.read_text()) if verified_path.exists() else {}
    current = {}
    for key, expected in frozen['source_sha256'].items():
        stat = Path(cfg[key]).stat()
        signature = [stat.st_size, stat.st_mtime_ns]
        if verified.get(key) != signature and digest(cfg[key]) != expected:
            raise ValueError(f'Frozen input content changed: {key}; create a new campaign')
        current[key] = signature
    if current != verified:
        write_json(verified_path, current)
    return out


def screen(cfg, *, max_groups=None):
    out = check_prepared(cfg)
    data = FeatureData.read(Path(cfg['features']), Path(cfg['catalog']))
    registry = pd.read_parquet(out / 'loci.parquet')
    allowed = np.flatnonzero(~registry.chr.isin(cfg['holdout_chromosomes']).to_numpy())
    rng = np.random.default_rng(cfg['mask_seed'])
    rows = np.sort(rng.choice(allowed, min(cfg['screen_loci'], len(allowed)), replace=False))
    _, names = read_axis(Path(cfg['methylation']))
    patients = load_patient_protocol(out / 'patients.npz', names)['train']
    if cfg.get('screen_patients'):
        patients = np.sort(rng.choice(patients, min(len(patients), cfg['screen_patients']), replace=False))
    targets_path = out / 'discovery_targets.npz'
    if targets_path.exists():
        with np.load(targets_path) as h:
            if not np.array_equal(h['rows'], rows) or not np.array_equal(h['patients'], patients):
                raise ValueError('Discovery target provenance differs')
            mean, variance, count = h['mean'], h['variance'], h['count']
    else:
        with h5py.File(cfg['methylation'], 'r') as h:
            mean, variance, count = methylation_moments(h['beta'], patients, rows)
        np.savez_compressed(targets_path, rows=rows, patients=patients, mean=mean, variance=variance, count=count)
    x, dense = data.tracks[rows], data.dense[rows]
    chrom = registry.chr.to_numpy()[rows]
    targets = {'mean_beta': mean, 'variance_beta': variance}
    # Low-observation loci are kept in the inventory but excluded from inferential probes.
    for y in targets.values():
        y[count < max(20, int(len(patients) * .5))] = np.nan
    scores = data.catalog.copy()
    for name, y in targets.items():
        scores[name + '_context_adjusted_r'] = residual_associations(x, dense[:, :18], y, chrom)
        scores[name + '_context_breadth_adjusted_r'] = residual_associations(x, dense, y, chrom)
    scores.to_csv(out / 'track_associations.csv', index=False)
    selected = ranked_panels(x, scores.variance_beta_context_adjusted_r.to_numpy(), cfg['panel_sizes'])
    random = {str(seed): balanced_panels(data.catalog, cfg['panel_sizes'], seed=seed) for seed in cfg['seeds']}
    write_json(out / 'panels.json', {'selected_variance': selected, 'balanced_random': random,
                                    'selection_endpoint': 'discovery context-adjusted variance association',
                                    'method': 'greedy relevance minus 0.5 maximum absolute overlap correlation'})
    groups = group_catalog(data.catalog)
    groups['baseline/context'] = {'tracks': [], 'dense': list(range(18)), 'family': 'baseline'}
    processed = 0
    for name, spec in groups.items():
        dest = out / 'screen' / (arm_id(name) + '.json')
        if dest.exists():
            continue
        started = time.monotonic()
        block = sparse.hstack([x[:, spec['tracks']], sparse.csr_matrix(dense[:, spec['dense']])], format='csr')
        result = {'group': name, 'family': spec['family'], 'n_tracks': len(spec['tracks']), 'endpoints': {}}
        for target, y in targets.items():
            result['endpoints'][target] = {
                'alone': blocked_probe(block, y, chrom),
                'plus_context': blocked_probe(add_context(block, dense), y, chrom)}
        result['seconds'] = time.monotonic() - started
        write_json(dest, result)
        print(f'screen {name}: {result["seconds"]:.1f}s', flush=True)
        processed += 1
        if max_groups is not None and processed >= max_groups:
            break
    return {'status': 'screen_checkpoint', 'new_groups': processed, 'total_groups': len(groups)}


def select(cfg):
    out = check_prepared(cfg)
    groups = json.loads((out / 'groups.json').read_text())
    results = [json.loads(p.read_text()) for p in (out / 'screen').glob('*.json')]
    expected = set(groups) | {'baseline/context'}
    if {r['group'] for r in results} != expected:
        raise ValueError('Selection requires completed screening of every group, including null results')
    baseline = next(r for r in results if r['group'] == 'baseline/context')
    base_mse = baseline['endpoints']['variance_beta']['alone'].get('mse')
    if base_mse is None:
        raise ValueError('Insufficient support for selection endpoint')
    catalog = load_catalog(Path(cfg['catalog']))
    histones = set(catalog.loc[catalog.functional_block.eq('histone'), 'encode_target'])
    tfs = set(catalog.loc[catalog.functional_block.isin(['tf_binding', 'ctcf']), 'encode_target'])
    categories = {'histone': [], 'tf': [], 'biosample': []}
    for result in results:
        group = result['group']
        family, label = group.split('/', 1)
        category = 'biosample' if family == 'biosample' else (
            'histone' if family == 'target' and label in histones else
            'tf' if family == 'target' and label in tfs else None)
        if category:
            endpoint = result['endpoints']['variance_beta']
            if endpoint['plus_context']['status'] != 'complete':
                continue
            gain = base_mse - endpoint['plus_context']['mse']
            categories[category].append((gain, -endpoint['alone']['mse'], group))
    selected = []
    for category, candidates in categories.items():
        positive = sorted([x for x in candidates if x[0] > 0], reverse=True)
        others = sorted([x for x in candidates if x[0] <= 0], key=lambda x: (-x[1], x[2]))
        selected.extend({'category': category, 'group': x[2], 'incremental_gain': x[0]}
                        for x in (positive + others)[:4])
    write_json(out / 'selection.json', {'endpoint': 'variance_beta', 'selected': selected,
                                      'source': 'nested chromosome CV, discovery patients only'})
    specs = json.loads((out / 'arms.json').read_text())
    for chosen in selected:
        name = chosen['group']
        group = groups[name]
        specs['only/' + name] = group
        specs['add/' + name] = {**group, 'dense': list(range(18))}
        specs['drop/' + name] = {'tracks': sorted(set(range(len(catalog))) - set(group['tracks'])),
                                 'dense': list(range(23)), 'family': group['family']}
        for fm_name, fm_spec in list(specs.items()):
            if fm_name.startswith('fm/'):
                specs[f'add_fm/{name}/{fm_name[3:]}'] = {
                    **group, 'external_store': fm_spec['external_store'], 'family': 'fm_complementarity'}
    # Pairwise biological interactions are deliberately limited to the selected subgroups.
    for a, b in itertools.combinations([r['group'] for r in selected], 2):
        specs[f'pair/{a}+{b}'] = {'tracks': sorted(set(groups[a]['tracks']) | set(groups[b]['tracks'])),
                                 'dense': list(range(18)), 'family': 'selected_interaction'}
    panels = json.loads((out / 'panels.json').read_text())
    for n, tracks in panels['selected_variance'].items():
        specs[f'panel/selected/{n}'] = {'tracks': tracks, 'dense': list(range(18)), 'family': 'panel'}
    for seed, panels_for_seed in panels['balanced_random'].items():
        for n, tracks in panels_for_seed.items():
            specs[f'panel/random_{seed}/{n}'] = {'tracks': tracks, 'dense': list(range(18)), 'family': 'panel'}
    write_json(out / 'confirmation_arms.json', specs)
    return {'selected': selected, 'confirmation_arms': len(specs)}


def benchmark_config(cfg, name, spec, seed, stage, *, ood=False, pilot=False):
    out = Path(cfg['output'])
    base = yaml.safe_load(Path(cfg['benchmark_template']).read_text())
    key = arm_id(name) + ('_ood' if ood else '') + ('_pilot' if pilot else '')
    base['experiment']['name'] = f'encode_atlas_{stage}_{key}'
    base['experiment']['output_root'] = str(out / 'benchmark' / stage)
    base['dataset'].update(methylation_h5=cfg['methylation'], cpg_registry=cfg['registry'],
                           patient_protocol=str(out / 'patients.npz'), patient_split_seed=cfg['patient_split_seed'])
    base['representation'].update(name=key, store_h5=str(out / 'embeddings' / f'{key}.h5'),
                                   source='ENCODE atlas controlled feature subset',
                                   provenance={'patient_specific': False, 'supervision': 'none',
                                               'feature_spec': spec, 'transform_fit': 'train_chromosomes' if ood
                                               else 'all_loci_transductive'})
    base['training']['seed'] = seed
    base['training']['num_workers'] = 0
    base['evaluation'].update(save_predictions=True, mask_seed=cfg['mask_seed'], num_workers=0,
                              patient_view='test' if stage == 'confirm' else 'validation',
                              panel_repeats=cfg['confirmation_panel_repeats'] if stage == 'confirm' else 1)
    base['experiment']['locus_split'] = {'heldout_fraction': 0., 'seed': cfg['mask_seed'],
                                        'protocol_path': str(out / 'loci_seen.npz')}
    if ood:
        registry = pd.read_parquet(out / 'loci.parquet')
        held = registry.chr.isin(cfg['holdout_chromosomes']).to_numpy()
        fraction = float(held.mean())
        protocol = out / 'loci_ood.npz'
        if not protocol.exists():
            np.savez_compressed(protocol, train_cpg_idx=np.sort(registry.cpg_idx.to_numpy()[~held]),
                                heldout_cpg_idx=np.sort(registry.cpg_idx.to_numpy()[held]),
                                seed=cfg['mask_seed'], heldout_fraction=fraction)
        base['experiment']['locus_split'].update(heldout_fraction=fraction, protocol_path=str(protocol))
    if pilot:
        base['training'].update(epochs=1, panel_size=128, batch_size=8)
        base['evaluation'].update(panel_size=128, panel_repeats=1, mask_fractions=[.5])
    return base


def execute(cfg, *, stage='discovery', only=None, max_jobs=None, pilot=False, ood=False, dry_run=False):
    out = check_prepared(cfg)
    specs = json.loads((out / ('confirmation_arms.json' if stage == 'confirm' else 'arms.json')).read_text())
    names = list(specs) if not only else only
    if any(name not in specs for name in names):
        raise ValueError('Unknown arm name')
    seeds = cfg['seeds'] if stage == 'confirm' else cfg['seeds'][:1]
    tasks = [(name, seed) for name in names if 'alias' not in specs[name] for seed in seeds]
    if dry_run:
        return {'tasks': [{'arm': n, 'seed': s} for n, s in tasks], 'n_tasks': len(tasks)}
    data = None
    completed = 0
    for name, seed in tasks:
        run_cfg = benchmark_config(cfg, name, specs[name], seed, stage, ood=ood, pilot=pilot)
        fingerprint = config_fingerprint(run_cfg)
        state_path = out / 'jobs' / f'{fingerprint}.json'
        if state_path.exists():
            previous = json.loads(state_path.read_text())
            if previous['status'] == 'complete' and Path(previous['run_dir'], 'summary.json').exists():
                continue
        embedding_path = Path(run_cfg['representation']['store_h5'])
        started = time.monotonic()
        if embedding_path.with_suffix('.json').exists():
            metadata = json.loads(embedding_path.with_suffix('.json').read_text())
            if metadata['spec'] != specs[name] or metadata['width'] != cfg['width']:
                raise ValueError(f'Cached embedding spec differs: {name}')
        if not all(p.exists() for p in [embedding_path, embedding_path.with_suffix('.json'),
                                       embedding_path.with_suffix('.transform.npz')]):
            if data is None:
                data = FeatureData.read(Path(cfg['features']), Path(cfg['catalog']))
            registry = pd.read_parquet(out / 'loci.parquet')
            fit_rows = np.flatnonzero(~registry.chr.isin(cfg['holdout_chromosomes']).to_numpy()) if ood else np.arange(len(data.ids))
            materialize(data, specs[name], embedding_path, fit_rows, width=cfg['width'], seed=cfg['seeds'][0])
        config_path = out / 'job_configs' / f'{fingerprint}.yaml'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(run_cfg, sort_keys=False))
        # Respect other GPU workloads; never kill or occupy a GPU below the configured margin.
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, check=False)
        if gpu.returncode or int(gpu.stdout.strip().splitlines()[0]) < cfg['minimum_free_gpu_gb'] * 1024:
            return {'status': 'waiting_for_gpu', 'completed': completed, 'minimum_free_gb': cfg['minimum_free_gpu_gb']}
        state = {'status': 'running', 'arm': name, 'seed': seed, 'stage': stage, 'ood': ood, 'pilot': pilot,
                 'fingerprint': fingerprint, 'config': str(config_path), 'pid': os.getpid()}
        code_root = Path(__file__).resolve().parents[3]
        state['code_sha256'] = {
            str(p.relative_to(code_root)): digest(p)
            for p in sorted((code_root / 'src/cpg_repr_benchmark').rglob('*.py'))
        }
        state['code_sha256']['scripts/run_masking_benchmark.py'] = digest(
            code_root / 'scripts/run_masking_benchmark.py')
        write_json(state_path, state)
        log_path = out / 'jobs' / f'{fingerprint}.log'
        env = {**os.environ, 'OMP_NUM_THREADS': str(cfg['torch_threads']),
               'OPENBLAS_NUM_THREADS': str(cfg['torch_threads']), 'CUDA_VISIBLE_DEVICES': '0'}
        root = Path(__file__).resolve().parents[3]
        with log_path.open('a') as log:
            process = subprocess.run([sys.executable, str(root / 'scripts/run_masking_benchmark.py'),
                                      '--config', str(config_path), '--mode', 'all'],
                                     cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        run_dirs = [line.removeprefix('RUN_DIR=') for line in log_path.read_text().splitlines()
                    if line.startswith('RUN_DIR=')]
        state.update(seconds=time.monotonic() - started, returncode=process.returncode)
        if process.returncode or not run_dirs:
            state['status'] = 'failed'
            write_json(state_path, state)
            raise RuntimeError(f'Benchmark failed; inspect {log_path}')
        state.update(status='complete', run_dir=run_dirs[-1])
        write_json(state_path, state)
        completed += 1
        print(f'completed {name} seed={seed}: {state["seconds"]:.1f}s', flush=True)
        if max_jobs is not None and completed >= max_jobs:
            break
    return {'status': 'batch_checkpoint', 'completed': completed}
