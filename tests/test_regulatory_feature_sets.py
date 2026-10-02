from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from cpg_repr_benchmark.encode_atlas.feature_sets import (
    DEFAULT_CATALOG,
    list_feature_sets,
    read_catalog,
    resolve_feature_set,
    verify_expected,
)

ROOT = Path(__file__).resolve().parents[1]
REAL_CATALOG = ROOT / DEFAULT_CATALOG
LINKED = {'ENCSR000BPZ', 'ENCSR000BHC', 'ENCSR542FLV', 'ENCSR156CWW', 'ENCSR260UJI', 'ENCSR396QWK',
          'ENCSR000BNA', 'ENCSR987PBI', 'ENCSR221GAN', 'ENCSR876GXA', 'ENCSR940MHE', 'ENCSR231YFE',
          'ENCSR345YWJ', 'ENCSR516HUP'}
SUBPROCESS_ENV = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(ROOT / 'src'), os.environ.get('PYTHONPATH', '')])}
needs_real = pytest.mark.skipif(not REAL_CATALOG.is_file(), reason=f'real catalog absent: {REAL_CATALOG}')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@needs_real
def test_real_catalog_counts_and_exclusions():
    allx = resolve_feature_set('regulatory_all_experimental', REAL_CATALOG, check_expected=True)
    clean = resolve_feature_set('regulatory_clean', REAL_CATALOG, check_expected=True)
    hist = resolve_feature_set('regulatory_histone', REAL_CATALOG, check_expected=True)
    assert (allx.n_selected, clean.n_selected, hist.n_selected) == (4165, 4151, 1959)
    assert allx.dense_columns == clean.dense_columns == hist.dense_columns == ()
    assert set(clean.excluded_targets) == {'ZBTB33', 'MBD1', 'MBD2', 'DNMT1', 'DNMT3B'}
    removed = set(allx.track_ids) - set(clean.track_ids)
    assert removed == LINKED
    assert {t['track_id'] for t in clean.excluded_tracks} == LINKED
    cat = read_catalog(REAL_CATALOG)
    assert set(cat.track_id[cat.encode_target.isin(clean.excluded_targets)]) == LINKED
    assert clean.counts_per_block == {'accessibility': 533, 'histone': 1959, 'tf': 1659}
    # store column space is catalog row order, not track_index
    assert allx.feature_columns == tuple(range(4165))
    assert allx.track_indices != allx.feature_columns


def make_catalog(order=None) -> pd.DataFrame:
    rows = [('E1', 'Histone ChIP-seq', 'H3K27ac', 'histone'), ('E2', 'Histone ChIP-seq', 'H3K9me3', 'histone'),
            ('E3', 'TF ChIP-seq', 'ZBTB33', 'tf_binding'), ('E4', 'TF ChIP-seq', 'CTCF', 'ctcf'),
            ('E5', 'TF ChIP-seq', 'MBD2', 'tf_binding'), ('E6', 'DNase-seq', 'unknown', 'accessibility'),
            ('E7', 'TF ChIP-seq', 'POLR2A', 'tf_binding')]
    frame = pd.DataFrame({'track_index': [10 * (i + 3) for i in range(len(rows))],
                          'track_id': [r[0] for r in rows], 'encode_accession': [r[0] for r in rows],
                          'file_accession': [f'F{r[0]}' for r in rows], 'encode_assay': [r[1] for r in rows],
                          'encode_target': [r[2] for r in rows], 'encode_biosample': ['x'] * len(rows),
                          'functional_block': [r[3] for r in rows]})
    return frame if order is None else frame.iloc[order].reset_index(drop=True)


def write(frame, tmp_path, name='cat.tsv') -> Path:
    path = tmp_path / name
    frame.to_csv(path, sep='\t', index=False)
    return path


def test_synthetic_rules_and_counts(tmp_path):
    cat = write(make_catalog(), tmp_path)
    allx = resolve_feature_set('regulatory_all_experimental', cat)
    clean = resolve_feature_set('regulatory_clean', cat)
    hist = resolve_feature_set('regulatory_histone', cat)
    assert (allx.n_selected, clean.n_selected, hist.n_selected) == (7, 5, 2)
    assert clean.track_ids == ('E1', 'E2', 'E4', 'E6', 'E7')
    assert clean.feature_columns == (0, 1, 3, 5, 6)
    assert clean.track_indices == (30, 40, 60, 80, 90)
    assert clean.counts_per_assay == {'DNase-seq': 1, 'Histone ChIP-seq': 2, 'TF ChIP-seq': 2}
    assert clean.counts_per_block == {'accessibility': 1, 'histone': 2, 'tf': 2}
    assert clean.counts_per_functional_block == {'accessibility': 1, 'ctcf': 1, 'histone': 2, 'tf_binding': 1}
    assert {t['track_id'] for t in clean.excluded_tracks} == {'E3', 'E5'}
    assert hist.track_ids == ('E1', 'E2') and hist.excluded_tracks == ()
    with pytest.raises(ValueError, match='expected_counts mismatch'):
        verify_expected(clean)  # real-catalog expectations do not hold on the synthetic catalog


def test_shuffle_invariance(tmp_path):
    base = make_catalog()
    order = [4, 2, 6, 0, 5, 1, 3]
    a = resolve_feature_set('regulatory_clean', write(base, tmp_path, 'a.tsv'))
    b = resolve_feature_set('regulatory_clean', write(make_catalog(order), tmp_path, 'b.tsv'))
    assert a.manifest_hash == b.manifest_hash
    assert set(a.track_ids) == set(b.track_ids)
    shuffled = make_catalog(order)
    # indices are remapped: each selected column points at the same track id in the shuffled catalog
    assert [shuffled.track_id[c] for c in b.feature_columns] == list(b.track_ids)
    assert a.feature_columns != b.feature_columns
    assert resolve_feature_set('regulatory_histone', write(base, tmp_path, 'a.tsv')).manifest_hash != a.manifest_hash


def test_errors(tmp_path):
    cat = write(make_catalog(), tmp_path)
    with pytest.raises(ValueError, match='Unknown feature set'):
        resolve_feature_set('regulatory_nope', cat)
    dup = make_catalog()
    dup.loc[1, 'track_id'] = 'E1'
    with pytest.raises(ValueError, match='Duplicate track_id'):
        resolve_feature_set('regulatory_clean', write(dup, tmp_path, 'dup.tsv'))
    with pytest.raises(ValueError, match='Missing catalog columns'):
        resolve_feature_set('regulatory_clean', write(make_catalog().drop(columns='encode_target'), tmp_path, 'm.tsv'))
    assert list_feature_sets() == ['regulatory_all_experimental', 'regulatory_clean', 'regulatory_histone']


def test_source_unchanged_after_resolve_and_audit(tmp_path):
    cat = write(make_catalog(), tmp_path)
    before = (cat.read_bytes(), os.stat(cat).st_mtime_ns)
    resolve_feature_set('regulatory_clean', cat)
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'audit_regulatory_feature_sets.py'), '--catalog',
                    str(cat), '--out-dir', str(tmp_path / 'audit')], check=True, cwd=ROOT, capture_output=True, env=SUBPROCESS_ENV)
    assert (cat.read_bytes(), os.stat(cat).st_mtime_ns) == before
    assert (tmp_path / 'audit' / 'regulatory_clean.json').is_file()
    assert len(pd.read_csv(tmp_path / 'audit' / 'regulatory_clean.tsv', sep='\t')) == 7


@needs_real
def test_real_catalog_untouched(tmp_path):
    before = (sha(REAL_CATALOG), os.stat(REAL_CATALOG).st_mtime_ns)
    resolve_feature_set('regulatory_clean', REAL_CATALOG)
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'audit_regulatory_feature_sets.py'),
                    '--out-dir', str(tmp_path)], check=True, cwd=ROOT, capture_output=True, env=SUBPROCESS_ENV)
    assert (sha(REAL_CATALOG), os.stat(REAL_CATALOG).st_mtime_ns) == before
