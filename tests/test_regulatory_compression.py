from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cpg_repr_benchmark.encode_atlas import compression as comp
from cpg_repr_benchmark.encode_atlas.compression import (
    FitLoci,
    RegulatoryCompressor,
    TrackStore,
    assert_disjoint,
    replicate_groups,
    replicate_weights,
    split_loci,
)
from cpg_repr_benchmark.encode_atlas.feature_sets import read_catalog, resolve_feature_set

N_LOCI, N_TRACKS = 440, 48


def make_catalog() -> pd.DataFrame:
    rows = []
    for i in range(N_TRACKS):
        if i < 20:
            rows.append(('Histone ChIP-seq', f'H3K{i % 4}', f'bs{i % 3}', 'histone'))  # groups of size 1..2
        elif i < 38:
            rows.append(('TF ChIP-seq', ('ZBTB33' if i == 20 else f'TF{i % 7}'), f'bs{i % 4}', 'tf_binding'))
        else:
            rows.append(('DNase-seq', '', f'bs{i % 6}', 'accessibility'))
    return pd.DataFrame({
        'track_index': [3 * i + 2 for i in range(N_TRACKS)], 'track_id': [f'E{i}' for i in range(N_TRACKS)],
        'encode_accession': [f'E{i}' for i in range(N_TRACKS)], 'file_accession': [f'F{i}' for i in range(N_TRACKS)],
        'encode_assay': [r[0] for r in rows], 'encode_target': [r[1] for r in rows],
        'encode_biosample': [r[2] for r in rows], 'functional_block': [r[3] for r in rows]})


def make_matrix(seed: int = 0) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    latent = rng.random((N_LOCI, 6))
    load = rng.random((6, N_TRACKS)) ** 3
    prob = np.clip(latent @ load / 3, 0, 0.9)
    return sparse.csr_matrix((rng.random((N_LOCI, N_TRACKS)) < prob).astype(np.float32))


def write_store(path: Path, matrix: sparse.csr_matrix, ids: np.ndarray) -> Path:
    with h5py.File(path, 'w') as h:
        h.create_dataset('cpg_idx', data=ids.astype(np.int64))
        h.create_dataset('track_indptr', data=matrix.indptr.astype(np.int64))
        h.create_dataset('track_indices', data=matrix.indices.astype(np.int64))
        h.create_dataset('dense', data=np.full((matrix.shape[0], 23), 7, np.float16))
        h.attrs['n_tracks'] = matrix.shape[1]
        h.attrs['patient_specific'] = False
    return path


@pytest.fixture()
def env(tmp_path):
    cat = tmp_path / 'cat.tsv'
    make_catalog().to_csv(cat, sep='\t', index=False)
    ids = np.arange(N_LOCI, dtype=np.int64) * 7 + 100
    chrom = np.where(np.arange(N_LOCI) < 330, 'chr1', 'chr21')
    chrom[:150] = 'chr5'
    reg = tmp_path / 'reg.parquet'
    pd.DataFrame({'cpg_idx': ids, 'chr': chrom, 'pos': np.arange(N_LOCI)}).to_parquet(reg)
    matrix = make_matrix()
    store_path = write_store(tmp_path / 'store.h5', matrix, ids)
    fs = resolve_feature_set('regulatory_clean', cat)
    return {'cat': cat, 'ids': ids, 'reg': reg, 'matrix': matrix, 'store_path': store_path, 'fs': fs, 'tmp': tmp_path}


def fit_args(env, **kw):
    return RegulatoryCompressor(env['fs'], catalog=env['cat'], n_components=8, block_components=6, **kw)


def fit_loci(env):
    fit, held = split_loci('discovery_chr1_19', env['ids'], env['reg'])
    return fit, held


def test_protocol_counts_and_disjoint(env):
    fit, held = fit_loci(env)
    assert fit.n == 330 and held.size == 110 and fit.name == 'discovery_chr1_19'
    assert_disjoint(fit, held)
    with pytest.raises(ValueError, match='inside the fit loci'):
        assert_disjoint(fit, fit.cpg_idx[:3])
    with pytest.raises(ValueError, match='Unknown fit protocol'):
        split_loci('nope', env['ids'], env['reg'])
    with pytest.raises(ValueError, match='in the universe'):
        split_loci('discovery_chr1_19', env['ids'][:100], env['reg'])  # chr20-22 absent


@pytest.mark.parametrize('method', ['global_svd', 'block_svd'])
def test_shapes_finite_determinism_and_signs(env, method):
    store = TrackStore(env['store_path'])
    fit, _ = fit_loci(env)
    a = fit_args(env, method=method).fit(store, fit)
    b = fit_args(env, method=method).fit(store, fit)
    za, zb = a.transform(store), b.transform(store)
    assert za.shape == (N_LOCI, 8) and za.dtype == np.float32 and np.isfinite(za).all()
    np.testing.assert_allclose(za, zb, atol=1e-5)
    comps = a.arrays['components_weighted'] if method == 'global_svd' else a.arrays['final_projection']
    top = np.abs(comps).argmax(axis=1)
    assert (comps[np.arange(len(comps)), top] > 0).all()  # deterministic sign convention
    zc = fit_args(env, method=method, seed=3).fit(store, fit).transform(store)
    assert zc.shape == za.shape


def test_block_svd_without_final_projection_and_block_membership(env):
    store = TrackStore(env['store_path'])
    fit, _ = fit_loci(env)
    c = fit_args(env, method='block_svd', final_projection=False).fit(store, fit)
    assert c.dim == int(c.arrays['block_dims'].sum()) == 18
    cat = read_catalog(env['cat'])
    for b in ('histone', 'tf', 'accessibility'):
        cols = c.columns[c.arrays[f'block_{b}_columns']]
        assert set(cat.encode_assay.iloc[cols].map(comp.ASSAY_BLOCK)) == {b}
    assert c.blocks.tolist().count('tf') == 17  # ZBTB33 track excluded by the clean contract
    # block-diagonal: a block's output dims only load on that block's tracks
    start = 0
    for b, k in zip(c.arrays['block_order'], c.arrays['block_dims']):
        rows = c.projection[start:start + k]
        assert np.all(rows[:, c.blocks != b] == 0)
        start += k
    # normalization: each block's mean squared row norm over fit loci == 1
    for b in c.arrays['block_order']:
        assert c.arrays[f'stat_block_{b}_post_frobenius_sq_per_locus'] == pytest.approx(1.0, rel=1e-4)


def test_train_only_fit_ignores_heldout_data(env):
    fit, held = fit_loci(env)
    store = TrackStore(env['store_path'])
    base = fit_args(env, method='global_svd').fit(store, fit)
    mat = env['matrix'].tolil()
    rng = np.random.default_rng(5)
    rows = np.flatnonzero(np.isin(env['ids'], held))
    for r in rows:
        mat[r, :] = (rng.random(N_TRACKS) < 0.5).astype(np.float32)
    altered = write_store(env['tmp'] / 'store2.h5', mat.tocsr(), env['ids'])
    other = fit_args(env, method='global_svd').fit(TrackStore(altered), fit)
    np.testing.assert_allclose(base.projection, other.projection, atol=1e-6)
    blk = fit_args(env, method='block_svd').fit(store, fit)
    blk2 = fit_args(env, method='block_svd').fit(TrackStore(altered), fit)
    np.testing.assert_allclose(blk.projection, blk2.projection, atol=1e-5)
    # held-out rows do change under transform (they are transformed, not fit)
    assert not np.allclose(base.transform(store, held), base.transform(TrackStore(altered), held))


def test_fit_requires_explicit_loci(env):
    store = TrackStore(env['store_path'])
    with pytest.raises(ValueError, match='explicit fit_loci'):
        fit_args(env).fit(store)
    with pytest.raises(TypeError, match='explicit fit_loci'):
        fit_args(env).fit(store, env['ids'])  # raw arrays are not accepted
    with pytest.raises(KeyError):
        fit_args(env).fit(store, FitLoci('x', np.array([1, 2, 3])))
    explicit = FitLoci('explicit', env['ids'][:200])
    assert fit_args(env).fit(store, explicit).info['fit_loci']['n'] == 200


def test_save_load_identical_and_hash_checks(env):
    store = TrackStore(env['store_path'])
    fit, held = fit_loci(env)
    c = fit_args(env, method='block_svd').fit(store, fit)
    path = c.save(env['tmp'] / 'out' / 'c.npz')
    loaded = RegulatoryCompressor.load(path, feature_set=env['fs'])
    np.testing.assert_array_equal(c.transform(store, held), loaded.transform(store, held))
    m = loaded.manifest()
    assert m['feature_set']['manifest_hash'] == env['fs'].manifest_hash
    assert m['feature_set']['n_features'] == env['fs'].n_selected
    assert m['fit_loci'] == {'protocol': 'discovery_chr1_19', 'n': 330, 'sha256': fit.sha256}
    assert m['assertions']['dense_read'] is False and m['assertions']['methylation_read'] is False
    assert m['scaling']['centering'] is False and m['replicate_weighting'] == 'none'
    assert (path.with_suffix('.json')).is_file()
    other = resolve_feature_set('regulatory_all_experimental', env['cat'])
    with pytest.raises(ValueError, match='hash mismatch'):
        RegulatoryCompressor.load(path, feature_set=other)
    with pytest.raises(ValueError, match='hash mismatch'):
        loaded.transform(store, held, feature_set=other)
    loaded.transform(store, held, feature_set=env['fs'])


def test_replicate_weighting(env):
    cat = read_catalog(env['cat'])
    for name in ('regulatory_clean', 'regulatory_all_experimental', 'regulatory_histone'):
        fs = resolve_feature_set(name, env['cat'])
        groups = replicate_groups(cat, fs.feature_columns)
        w = replicate_weights(groups, 'equal_group_mass')
        assert len(w) == fs.n_selected  # no track dropped
        sums = np.bincount(groups, weights=w.astype(np.float64) ** 2)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-6)  # group squared mass == one track
        assert (np.bincount(groups) > 1).any()
        np.testing.assert_array_equal(replicate_weights(groups, 'none'), np.ones(fs.n_selected))
    with pytest.raises(ValueError):
        replicate_weights(groups, 'bogus')
    # DNase has no target: grouped by (assay, biosample) only
    fs = resolve_feature_set('regulatory_clean', env['cat'])
    g = replicate_groups(cat, fs.feature_columns)
    dn = np.flatnonzero(cat.encode_assay.iloc[list(fs.feature_columns)].to_numpy() == 'DNase-seq')
    bios = cat.encode_biosample.iloc[list(fs.feature_columns)].to_numpy()[dn]
    assert len(set(g[dn])) == len(set(bios))
    store = TrackStore(env['store_path'])
    fit, _ = fit_loci(env)
    none = fit_args(env).fit(store, fit)
    egm = fit_args(env, replicate_weighting='equal_group_mass').fit(store, fit)
    assert (none.weights == 1).all() and egm.weights.min() < 1 and len(egm.weights) == fs.n_selected
    assert not np.allclose(none.projection, egm.projection)
    assert egm.manifest()['replicate_groups']['n_groups_gt1'] > 0


def test_never_reads_dense_and_store_readonly(env, monkeypatch):
    real = h5py.Group.__getitem__

    def guarded(self, name):
        if str(name).strip('/') == 'dense':
            raise AssertionError('/dense accessed')
        return real(self, name)

    monkeypatch.setattr(h5py.Group, '__getitem__', guarded)
    before = hashlib.sha256(env['store_path'].read_bytes()).hexdigest()
    store = TrackStore(env['store_path'])
    fit, held = fit_loci(env)
    c = fit_args(env, method='block_svd').fit(store, fit)
    c.transform(store, held)
    assert 'dense' not in store.datasets_read
    assert sorted(set(store.datasets_read)) == sorted(comp.STORE_DATASETS_READ)
    assert hashlib.sha256(env['store_path'].read_bytes()).hexdigest() == before
    with pytest.raises(RuntimeError, match='Refusing'), h5py.File(env['store_path'], 'r') as h:
        store._read(h, 'dense')
    # store without /dense at all also works
    nodense = env['tmp'] / 'nodense.h5'
    with h5py.File(env['store_path'], 'r') as src, h5py.File(nodense, 'w') as dst:
        for k in comp.STORE_DATASETS_READ:
            dst.create_dataset(k, data=src[k][:])
        dst.attrs['n_tracks'] = N_TRACKS
    TrackStore(nodense)
