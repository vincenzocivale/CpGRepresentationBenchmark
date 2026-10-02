from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

CORE_MARKS = {'H3K27AC', 'H3K27ME3', 'H3K36ME3', 'H3K4ME1', 'H3K4ME3', 'H3K9AC', 'H3K9ME3'}
BLOCKS = ('accessibility', 'histone', 'ctcf', 'tf_binding')
DENSE_GROUPS = {'context': list(range(4)), 'gene_region': list(range(4, 8)),
                'ccre': list(range(8, 17)), 'tss': [17], 'breadth': list(range(18, 23))}
DENSE_NAMES = ([f'context:{x}' for x in ('island', 'shore', 'shelf', 'open_sea')]
               + [f'gene_region:{x}' for x in ('promoter', 'exon', 'intron', 'intergenic')]
               + [f'ccre:{x}' for x in ('none', 'PLS', 'pELS', 'dELS', 'CA', 'TF', 'CA-CTCF',
                                      'CA-H3K4me3', 'CA-TF')]
               + ['log10_tss_distance'] + [f'breadth:{x}' for x in (*BLOCKS, 'core')])


def digest(path: str | Path) -> str:
    result = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    tmp.replace(path)


def load_catalog(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep='\t').fillna('unknown')
    required = {'track_index', 'file_accession', 'encode_assay', 'encode_target', 'encode_biosample',
                'encode_biosample_class', 'functional_block', 'histone_mark_group', 'assembly'}
    if required - set(frame):
        raise ValueError(f'Missing catalog columns: {required - set(frame)}')
    if (frame.track_index.duplicated().any() or not frame.track_index.is_monotonic_increasing
            or frame.file_accession.duplicated().any() or not frame.assembly.eq('GRCh38').all()):
        raise ValueError('Invalid track order, duplicate accession, or non-GRCh38 assembly')
    # Upstream packbits uses enumerate(frame.itertuples()), NOT track_index as the column.
    frame.insert(0, 'feature_column', np.arange(len(frame)))
    return frame


def group_catalog(frame: pd.DataFrame) -> dict:
    groups = {f'dense/{k}': {'tracks': [], 'dense': v, 'family': 'dense'}
              for k, v in DENSE_GROUPS.items()}
    axes = {'assay': 'encode_assay', 'target': 'encode_target', 'biosample': 'encode_biosample',
            'biosample_class': 'encode_biosample_class', 'block': 'functional_block'}
    for family, column in axes.items():
        for label, subset in frame.groupby(column, sort=True):
            groups[f'{family}/{label}'] = {'tracks': subset.feature_column.tolist(), 'dense': [],
                                          'family': family}
    return groups


def breadth_matrix(frame: pd.DataFrame, selected: np.ndarray) -> np.ndarray:
    """Linear mapping from retained track overlaps to breadth; denominator is retained count."""
    blocks = [frame.functional_block.eq(b).to_numpy() for b in BLOCKS]
    core = (frame.functional_block.isin(['accessibility', 'ctcf'])
            | (frame.functional_block.eq('histone') & frame.histone_mark_group.isin(CORE_MARKS))).to_numpy()
    weights = np.column_stack([*blocks, core]).astype(np.float32)
    weights[~selected] = 0
    return weights / np.maximum(weights.sum(axis=0), 1)


@dataclass
class FeatureData:
    ids: np.ndarray
    tracks: sparse.csr_matrix
    dense: np.ndarray
    catalog: pd.DataFrame

    @classmethod
    def read(cls, source: Path, catalog: Path, rows: np.ndarray | None = None):
        frame = load_catalog(catalog)
        with h5py.File(source, 'r') as h:
            ids = h['cpg_idx'][:]
            indptr = h['track_indptr'][:]
            indices = h['track_indices'][:]
            dense = h['dense'][:].astype(np.float32)
            if int(h.attrs['n_tracks']) != len(frame):
                raise ValueError('Track count differs from catalog')
        if (len(np.unique(ids)) != len(ids) or dense.shape != (len(ids), 23)
                or len(indptr) != len(ids) + 1 or indptr[0] != 0 or indptr[-1] != len(indices)
                or np.any(np.diff(indptr) < 0) or np.any(indices < 0) or np.any(indices >= len(frame))):
            raise ValueError('Invalid feature axes or CSR structure')
        tracks = sparse.csr_matrix((np.ones(len(indices), np.float32), indices, indptr),
                                   shape=(len(ids), len(frame)))
        tracks.sum_duplicates()
        if np.any(tracks.data != 1) or not np.isfinite(dense).all():
            raise ValueError('Duplicate track overlaps or nonfinite dense features')
        if rows is not None:
            ids, tracks, dense = ids[rows], tracks[rows], dense[rows]
        expected = tracks @ breadth_matrix(frame, np.ones(len(frame), bool))
        if not np.allclose(dense[:, 18:], expected, atol=5e-4, rtol=1e-3):
            raise ValueError('Breadth mismatch: catalog order/source mapping is not validated')
        return cls(ids, tracks, dense, frame)

    def matrix(self, tracks: list[int], dense: list[int], *, breadth_policy='recompute'):
        if breadth_policy not in {'recompute', 'preserve'}:
            raise ValueError('breadth_policy must be recompute or preserve')
        selected = np.zeros(self.tracks.shape[1], bool)
        selected[tracks] = True
        dense_values = self.dense.copy()
        if breadth_policy == 'recompute':
            dense_values[:, 18:] = self.tracks @ breadth_matrix(self.catalog, selected)
        return self.tracks[:, tracks], dense_values[:, dense]


def make_transform(tracks, dense, fit_rows, *, width=256, seed=17, scaling='legacy'):
    """Fit exclusively on fit_rows. Pad small groups to fixed downstream adapter width."""
    if scaling not in {'legacy', 'balanced'}:
        raise ValueError('Unknown scaling')
    fit_rows = np.asarray(fit_rows, dtype=int)
    if len(fit_rows) < 2:
        raise ValueError('Need at least two fit loci')
    mean = dense[fit_rows].mean(axis=0)
    std = dense[fit_rows].std(axis=0)
    std[std < 1e-8] = 1
    z = (dense - mean) / std
    energy = float(tracks[fit_rows].multiply(tracks[fit_rows]).sum())
    scale = 1.0 / max(np.sqrt(energy / max(len(fit_rows) * tracks.shape[1], 1)), 1e-8)
    dense_scale = 1.0
    if scaling == 'balanced':
        scale /= np.sqrt(max(tracks.shape[1], 1))
        dense_scale /= np.sqrt(max(dense.shape[1], 1))
    x = sparse.hstack([tracks * scale, sparse.csr_matrix(z * dense_scale)], format='csr')
    if x.shape[1] == 0:
        x = sparse.csr_matrix((len(dense), 1), dtype=np.float32)
    rank = min(width, x.shape[1], len(fit_rows) - 1)
    if x.shape[1] <= width:
        components = np.eye(x.shape[1], dtype=np.float32)
        embedding = x.toarray()
        variance = None
    else:
        svd = TruncatedSVD(n_components=rank, random_state=seed)
        svd.fit(x[fit_rows])
        components = svd.components_
        embedding = svd.transform(x)
        variance = float(svd.explained_variance_ratio_.sum())
    embedding = np.pad(embedding, ((0, 0), (0, width - embedding.shape[1]))).astype(np.float32)
    return embedding, {'dense_mean': mean, 'dense_std': std, 'track_scale': np.asarray(scale),
                       'dense_scale': np.asarray(dense_scale), 'components': components,
                       'fit_rows': fit_rows}, variance


def materialize(data: FeatureData, spec: dict, path: Path, fit_rows: np.ndarray, *, width=256, seed=17):
    started = time.monotonic()
    tracks, dense = data.matrix(spec['tracks'], spec['dense'],
                                breadth_policy=spec.get('breadth_policy', 'recompute'))
    if spec.get('external_store'):
        from cpg_repr_benchmark.representations.hdf5_store import HDF5RepresentationStore
        store = HDF5RepresentationStore(Path(spec['external_store']), data.ids)
        if not store.coverage_mask.all():
            raise ValueError('External embedding does not cover the frozen CpG universe')
        external = np.concatenate([store.get_by_matrix_columns(np.arange(i, min(i + 4096, len(data.ids))))
                                   for i in range(0, len(data.ids), 4096)])
        store.close()
        # Treat frozen FM coordinates as a separate standardized dense feature block.
        dense = np.column_stack([dense, external])
    embedding, transform, variance = make_transform(tracks, dense, fit_rows, width=width, seed=seed,
                                                     scaling=spec.get('scaling', 'legacy'))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.h5')
    with h5py.File(temp, 'w') as h:
        h.create_dataset('cpg_idx', data=data.ids)
        h.create_dataset('embedding', data=embedding.astype(np.float16))
        h.attrs.update(patient_specific=False, supervision='none', reference_build='GRCh38',
                       cpg_namespace='array_registry_legacy', transform='TruncatedSVD_or_identity')
    temp.replace(path)
    np.savez_compressed(path.with_suffix('.transform.npz'), **transform,
                        track_columns=spec['tracks'], dense_columns=spec['dense'])
    write_json(path.with_suffix('.json'), {'spec': spec, 'width': width, 'seed': seed,
                                         'variance_retained': variance, 'n_fit_loci': len(fit_rows),
                                         'seconds': time.monotonic() - started,
                                         'external_sha256': digest(spec['external_store'])
                                         if spec.get('external_store') else None})
