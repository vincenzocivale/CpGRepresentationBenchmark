from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cpg_repr_benchmark.encode_atlas.biology import blocked_probe, methylation_moments, ranked_panels
from cpg_repr_benchmark.encode_atlas.campaign import macro_specs
from cpg_repr_benchmark.encode_atlas.features import FeatureData, breadth_matrix, make_transform
from cpg_repr_benchmark.encode_atlas.protocol import freeze_patient_protocol, load_patient_protocol
from cpg_repr_benchmark.encode_atlas.statistics import bh_adjust, paired_bootstrap


@pytest.fixture
def feature_fixture(tmp_path):
    frame = pd.DataFrame({
        'track_index': [222, 333, 444, 555, 666, 777],
        'file_accession': [f'ENCFF{i}' for i in range(6)],
        'encode_assay': ['DNase-seq', 'Histone ChIP-seq', 'Histone ChIP-seq', 'TF ChIP-seq',
                         'TF ChIP-seq', 'DNase-seq'],
        'encode_target': ['unknown', 'H3K27ac', 'H3K9me3', 'CTCF', 'POLR2A', 'unknown'],
        'encode_biosample': ['a', 'a', 'b', 'b', 'c', 'c'],
        'encode_biosample_class': ['tissue'] * 6,
        'functional_block': ['accessibility', 'histone', 'histone', 'ctcf', 'tf_binding', 'accessibility'],
        'histone_mark_group': ['unknown', 'H3K27AC', 'H3K9ME3', 'CTCF', 'POLR2A', 'unknown'],
        'assembly': ['GRCh38'] * 6})
    catalog = tmp_path / 'catalog.tsv'
    frame.to_csv(catalog, sep='\t', index=False)
    rng = np.random.default_rng(17)
    tracks = sparse.csr_matrix((rng.random((80, 6)) > .5).astype(np.float32))
    dense = np.zeros((80, 23), np.float32)
    dense[:, 18:] = tracks @ breadth_matrix(frame, np.ones(6, bool))
    source = tmp_path / 'features.h5'
    with h5py.File(source, 'w') as h:
        h['cpg_idx'] = np.arange(80)
        h['track_indices'] = tracks.indices
        h['track_indptr'] = tracks.indptr
        h['dense'] = dense.astype(np.float16)
        h.attrs['n_tracks'] = 6
    return source, catalog, tracks


def test_original_indices_are_not_sparse_columns(feature_fixture):
    source, catalog, tracks = feature_fixture
    data = FeatureData.read(source, catalog)
    np.testing.assert_array_equal(data.catalog.feature_column, np.arange(6))
    np.testing.assert_array_equal(data.tracks.toarray(), tracks.toarray())
    frame = pd.read_csv(catalog, sep='\t')
    frame.loc[[0, 1], 'functional_block'] = ['histone', 'accessibility']
    frame.to_csv(catalog, sep='\t', index=False)
    with pytest.raises(ValueError, match='Breadth mismatch'):
        FeatureData.read(source, catalog)


def test_derived_breadth_does_not_retain_removed_tracks(feature_fixture):
    source, catalog, _ = feature_fixture
    data = FeatureData.read(source, catalog)
    _, dense = data.matrix([0, 5], list(range(18, 23)))
    assert np.all(dense[:, 1:4] == 0)
    np.testing.assert_allclose(dense[:, 0], dense[:, 4])
    _, preserved = data.matrix([0, 5], list(range(18, 23)), breadth_policy='preserve')
    assert np.any(preserved[:, 1:4] > 0)


def test_transform_is_fit_only_on_declared_loci_and_fixed_width():
    rng = np.random.default_rng(3)
    tracks = sparse.csr_matrix(rng.integers(0, 2, (40, 20)).astype(np.float32))
    dense = rng.normal(size=(40, 2))
    a, fit_a, _ = make_transform(tracks, dense, np.arange(30), width=8)
    dense[30:] += 1000
    b, fit_b, _ = make_transform(tracks, dense, np.arange(30), width=8)
    np.testing.assert_allclose(a[:30], b[:30])
    np.testing.assert_allclose(fit_a['components'], fit_b['components'])
    small, _, _ = make_transform(tracks[:, :1], dense[:, :0], np.arange(30), width=8)
    assert small.shape == (40, 8)
    assert np.all(small[:, 1:] == 0)
    null, _, _ = make_transform(tracks[:, :0], dense[:, :0], np.arange(30), width=8)
    assert np.all(null == 0)


def test_patient_protocol_groups_aliquots_and_rejects_changed_axis(tmp_path):
    names = [f'TCGA-AA-{i:04d}-{s}' for i in range(30) for s in ['01A', '11A']]
    path = tmp_path / 'protocol.npz'
    a = freeze_patient_protocol(path, names)
    b = freeze_patient_protocol(path, names, seed=99)
    np.testing.assert_array_equal(a['test'], b['test'])
    with pytest.raises(ValueError, match='axis differs'):
        load_patient_protocol(path, names[::-1])


def test_moments_never_read_test_patients():
    beta = np.array([[.1, .2, np.nan], [.3, .4, .5], [100, 100, 100]])
    mean, var, count = methylation_moments(beta, np.array([0, 1]), np.arange(3))
    np.testing.assert_allclose(mean, [.2, .3, .5])
    np.testing.assert_allclose(var[:2], [.02, .02])
    assert np.isnan(var[2])
    np.testing.assert_array_equal(count, [2, 2, 1])


def test_blocked_probe_recovers_signal_and_panels_avoid_duplicate():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(180, 4))
    y = x[:, 0] * 3 + rng.normal(size=180) * .05
    chromosomes = np.repeat(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6'], 30)
    result = blocked_probe(sparse.csr_matrix(x), y, chromosomes, folds=3)
    assert result['r2'] > .98
    noise = blocked_probe(sparse.csr_matrix(x), rng.normal(size=180), chromosomes, folds=3)
    assert noise['r2'] < .2
    track = rng.integers(0, 2, 180)
    matrix = sparse.csr_matrix(np.column_stack([track, track, rng.integers(0, 2, 180)]), dtype=float)
    panels = ranked_panels(matrix, [1., 1., .9], [2])
    assert 2 in panels[2]


def test_paired_bootstrap_detects_gain_and_refuses_unpaired():
    base = pd.DataFrame({'patient': np.repeat(['a', 'b', 'c'], 3), 'column': list(range(3)) * 3,
                         'block': ['chr1:1', 'chr2:1', 'chr3:1'] * 3,
                         'target': .5, 'squared_error': .2, 'n': 1})
    alt = base.assign(squared_error=.1)
    result = paired_bootstrap(base, alt, replicates=50)
    assert result['ci95'][1] < 0
    assert result['relative_ci95'][1] < -.49
    with pytest.raises(ValueError, match='not paired'):
        paired_bootstrap(base, alt.iloc[:-1], replicates=10)
    np.testing.assert_allclose(bh_adjust([.01, .04, .03]), [.03, .04, .04])


def test_crossed_bootstrap_cell_aggregation_matches_observation_resampling():
    rng = np.random.default_rng(8)
    patients = np.repeat(['a', 'b', 'c', 'd'], 6)
    columns = np.tile(np.arange(6), 4)
    baseline = pd.DataFrame({'patient': patients, 'column': columns,
                             'block': np.tile(['chr1:1', 'chr1:1', 'chr2:2',
                                               'chr2:2', 'chr3:3', 'chr3:3'], 4),
                             'target': .5, 'squared_error': rng.uniform(.02, .3, 24),
                             'n': rng.integers(1, 4, 24)})
    alternative = baseline.copy()
    alternative['squared_error'] += rng.normal(0, .02, 24)
    result = paired_bootstrap(baseline, alternative, seed=27, replicates=80)
    deltas = (alternative.squared_error - baseline.squared_error).to_numpy()
    pidx = np.repeat(np.arange(4), 6)
    bidx = np.tile(np.repeat(np.arange(3), 2), 4)
    rng = np.random.default_rng(27)
    expected = []
    for _ in range(80):
        pw = rng.multinomial(4, np.full(4, .25))
        bw = rng.multinomial(3, np.full(3, 1 / 3))
        weights = pw[pidx] * bw[bidx] * baseline.n.to_numpy()
        if weights.sum():
            expected.append(np.average(deltas, weights=weights))
    np.testing.assert_allclose(result['ci95'], np.quantile(expected, [.025, .975]), rtol=1e-12)


def test_macro_specs_contain_assay_combinations(feature_fixture):
    source, catalog, _ = feature_fixture
    specs = macro_specs(FeatureData.read(source, catalog).catalog)
    assert len([k for k in specs if k.startswith('pair/')]) == 3
    assert specs['breadth_only']['breadth_policy'] == 'preserve'
    assert specs['drop/assay/DNase-seq']['tracks'] == [1, 2, 3, 4]


def test_prevalence_matched_assay_control_has_equal_counts_and_common_support(feature_fixture):
    from cpg_repr_benchmark.encode_atlas.focused import prevalence_matched_assays

    source, catalog, _ = feature_fixture
    frame = FeatureData.read(source, catalog).catalog
    frame['prevalence'] = [.1, .201, .801, .202, .802, .9]
    histone, other, audit = prevalence_matched_assays(frame)
    assert histone['tracks'] == [1, 2]
    assert other['tracks'] == [3, 4]
    assert histone['dense'] == other['dense'] == list(range(18))
    assert audit['n_tracks_each'] == 2


def test_ewas_background_is_matched_without_replacement():
    from cpg_repr_benchmark.encode_atlas.ewas import matched_membership
    labels = np.array([True, False, False, True, True, False, True])
    strata = np.array(['a', 'a', 'a', 'b', 'b', 'b', 'c'])
    rows = matched_membership(labels, strata, maximum=10)
    assert len(rows) == 4
    assert len(np.unique(rows)) == len(rows)
    for label in ['a', 'b']:
        y = labels[rows[strata[rows] == label]]
        assert y.sum() == 1 and len(y) == 2
    assert not np.any(strata[rows] == 'c')


def test_real_benchmark_cli_uses_frozen_patients_and_paired_repeated_masks(feature_fixture, tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    import yaml

    from cpg_repr_benchmark.encode_atlas.campaign import benchmark_config, prepare
    from cpg_repr_benchmark.encode_atlas.features import materialize

    source, catalog, _ = feature_fixture
    matrix = tmp_path / 'methylation.h5'
    rng = np.random.default_rng(4)
    with h5py.File(matrix, 'w') as h:
        h['beta'] = rng.uniform(.1, .9, (30, 80)).astype(np.float32)
        h['cpg_idx'] = np.arange(80)
        h['sample_name'] = np.asarray([f'patient_{i}' for i in range(30)], dtype='S')
    registry = tmp_path / 'registry.parquet'
    pd.DataFrame({'cpg_idx': np.arange(80), 'chr': np.repeat(['chr1', 'chr2', 'chr3', 'chr4'], 20),
                  'pos': np.tile(np.arange(1, 21), 4)}).to_parquet(registry)
    template = tmp_path / 'template.yaml'
    template.write_text(yaml.safe_dump({
        'experiment': {'task': 'masking'}, 'dataset': {'name': 'synthetic'},
        'representation': {'mode': 'precomputed', 'family': 'functional_annotations', 'track': 'native_frozen'},
        'model': {'locus_latent_dim': 8, 'token_dim': 8, 'patient_dim': 8, 'hidden_dim': 16},
        'training': {'device': 'cpu', 'epochs': 1, 'batch_size': 4, 'panel_size': 16,
                     'learning_rate': .001, 'mask_fractions': [.5]},
        'evaluation': {'panel_size': 16, 'mask_fractions': [.5]},
    }))
    cfg = {'features': str(source), 'catalog': str(catalog), 'registry': str(registry),
           'methylation': str(matrix), 'benchmark_template': str(template), 'output': str(tmp_path / 'out'),
           'mask_seed': 17001, 'patient_split_seed': 29, 'seeds': [17, 42], 'width': 16,
           'confirmation_panel_repeats': 2, 'holdout_chromosomes': ['chr4']}
    prepare(cfg)
    data = FeatureData.read(source, catalog)
    spec = macro_specs(data.catalog)['full']
    root = Path(__file__).resolve().parents[1]
    outputs = []
    for seed in [17, 42]:
        config = benchmark_config(cfg, 'full', spec, seed, 'confirm')
        path = Path(config['representation']['store_h5'])
        materialize(data, spec, path, np.arange(80), width=16)
        config_path = tmp_path / f'run{seed}.yaml'
        config_path.write_text(yaml.safe_dump(config))
        process = subprocess.run([sys.executable, str(root / 'scripts/run_masking_benchmark.py'),
                                  '--config', str(config_path), '--mode', 'all'],
                                 env={**os.environ, 'PYTHONPATH': str(root / 'src'),
                                      'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
                                 capture_output=True, text=True, check=False, timeout=120)
        assert process.returncode == 0, process.stdout + process.stderr
        run_dir = Path(next(x.removeprefix('RUN_DIR=') for x in process.stdout.splitlines()
                            if x.startswith('RUN_DIR=')))
        with np.load(run_dir / 'evaluation/seen/mask_0.50/predictions.npz') as h:
            outputs.append({k: h[k] for k in h.files})
    for key in ['sample_index', 'target_matrix_column', 'target', 'panel_repeat']:
        np.testing.assert_array_equal(outputs[0][key], outputs[1][key])
    assert set(outputs[0]['panel_repeat']) == {0, 1}
