"""Label-free linear compression of the regulatory ENCODE track matrix (Step 5).

Inputs are ONLY the binary peak-overlap CSR arrays (`/track_indptr`, `/track_indices`) of the feature store, restricted
to the columns of a resolved `FeatureSet`. `/dense` (handcrafted / breadth columns) and any methylation data are never
read: every dataset access goes through `TrackStore._read`, which refuses `dense`.

Scaling choices (all recorded in the manifest, none implicit):
  * tracks are binary 0/1 (peak overlap); no column scaling is applied (a track's energy is its prevalence, so broad
    tracks carry more energy: this is a property of the representation, reported rather than hidden);
  * NO centering: TruncatedSVD runs on the uncentered sparse matrix (centering would densify it). The first component
    therefore mostly encodes overall peak density. "Explained energy" = sum_i z_i^2 / sum ||x||^2 (uncentered); a
    centered variance ratio of the same orthonormal projection is also stored;
  * `replicate_weighting`: `none` (all column weights 1) or `equal_group_mass` (tracks of an (assay, biosample, target)
    group of size n get column weight 1/sqrt(n), so the group's summed squared weight is 1, like a single track;
    no track is dropped or merged; weights are folded into the stored projection and reversible);
  * `block_svd`: SVD per contract block (histone / tf / accessibility), each block embedding is rescaled so that its
    mean squared row norm over the FIT loci is 1 (Frobenius equalization: block size and prevalence no longer set its
    weight), blocks are concatenated, and an optional label-free SVD projects the concatenation to the final width.
Every map is linear, so the compressor stores one effective `projection` (dim x n_selected_tracks) acting on raw binary
tracks.

Fit loci must be passed explicitly (`FitLoci`); there is no default to "all loci".
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from cpg_repr_benchmark.data.chromosomes import normalize_chromosome
from cpg_repr_benchmark.encode_atlas.feature_sets import (
    ASSAY_BLOCK,
    DEFAULT_CATALOG,
    FeatureSet,
    read_catalog,
)

BLOCKS = ('histone', 'tf', 'accessibility')
METHODS = ('global_svd', 'block_svd')
REPLICATE_WEIGHTINGS = ('none', 'equal_group_mass')
FORBIDDEN_DATASETS = ('dense',)
STORE_DATASETS_READ = ('cpg_idx', 'track_indptr', 'track_indices')
EXPLAINED_AT = (16, 64, 128, 256)
FORMAT_VERSION = 1

# Named fit protocols: the held-out chromosomes are transformed only, never fit.
PROTOCOLS = {
    'discovery_chr1_19': {'fit_chromosomes': tuple(f'chr{i}' for i in range(1, 20)),
                          'heldout_chromosomes': ('chr20', 'chr21', 'chr22')},
}


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_sha256(path: str | Path) -> str:
    result = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


# --------------------------------------------------------------------------------------------- loci / protocols

@dataclass(frozen=True)
class FitLoci:
    """Explicit, sorted, unique set of fit loci (cpg_idx values) with a protocol name."""
    name: str
    cpg_idx: np.ndarray

    def __post_init__(self):
        ids = np.unique(np.asarray(self.cpg_idx, dtype=np.int64))
        if ids.size < 2:
            raise ValueError('Need at least two fit loci')
        object.__setattr__(self, 'cpg_idx', ids)

    @property
    def n(self) -> int:
        return int(self.cpg_idx.size)

    @property
    def sha256(self) -> str:
        return array_sha256(self.cpg_idx)


def chromosomes_from_registry(cpg_idx: np.ndarray, registry: str | Path) -> np.ndarray:
    """Chromosome (chrN) per locus from the repo coordinate registry (columns cpg_idx, chr). No methylation data."""
    reg = pd.read_parquet(registry, columns=['cpg_idx', 'chr'])
    if reg.cpg_idx.duplicated().any():
        raise ValueError('Registry has duplicate cpg_idx')
    chrom = pd.Series(reg.chr.map(normalize_chromosome).to_numpy(), index=reg.cpg_idx.to_numpy())
    ids = np.asarray(cpg_idx, dtype=np.int64)
    missing = ~np.isin(ids, chrom.index.to_numpy())
    if missing.any():
        raise ValueError(f'{int(missing.sum())} loci absent from registry {registry}')
    return chrom.loc[ids].to_numpy()


def split_loci(protocol: str, universe: np.ndarray, registry: str | Path) -> tuple[FitLoci, np.ndarray]:
    """(fit loci, held-out loci) of a named protocol within `universe`; each chromosome group must have loci."""
    if protocol not in PROTOCOLS:
        raise ValueError(f'Unknown fit protocol {protocol!r}; available: {sorted(PROTOCOLS)}')
    spec = PROTOCOLS[protocol]
    universe = np.asarray(universe, dtype=np.int64)
    chrom = chromosomes_from_registry(universe, registry)
    for key in ('fit_chromosomes', 'heldout_chromosomes'):
        if not np.isin(chrom, spec[key]).any():
            raise ValueError(f'Protocol {protocol}: no loci of {key} {list(spec[key])} in the universe')
    fit = FitLoci(protocol, universe[np.isin(chrom, spec['fit_chromosomes'])])
    heldout = np.sort(universe[np.isin(chrom, spec['heldout_chromosomes'])])
    assert_disjoint(fit, heldout)
    return fit, heldout


def assert_disjoint(fit: FitLoci, heldout: np.ndarray) -> None:
    overlap = np.intersect1d(fit.cpg_idx, np.asarray(heldout, dtype=np.int64))
    if overlap.size:
        raise ValueError(f'{overlap.size} held-out loci are inside the fit loci')


# --------------------------------------------------------------------------------------------------- feature store

class TrackStore:
    """Read-only view of the binary track CSR of a feature store. Never reads `/dense`."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.datasets_read: list[str] = []
        with h5py.File(self.path, 'r') as h:
            self.attrs = {k: (v.item() if hasattr(v, 'item') else v) for k, v in h.attrs.items()}
            self.cpg_idx = np.asarray(self._read(h, 'cpg_idx'), dtype=np.int64)
            indptr = np.asarray(self._read(h, 'track_indptr'), dtype=np.int64)
            indices = np.asarray(self._read(h, 'track_indices'))
        self.n_tracks = int(self.attrs['n_tracks'])
        if self.attrs.get('patient_specific', False):
            raise ValueError('Store is patient-specific')
        if (np.unique(self.cpg_idx).size != self.cpg_idx.size or len(indptr) != len(self.cpg_idx) + 1
                or indptr[0] != 0 or indptr[-1] != len(indices) or np.any(np.diff(indptr) < 0)
                or (len(indices) and (indices.min() < 0 or indices.max() >= self.n_tracks))):
            raise ValueError('Invalid loci axis or CSR structure')
        self.identity = {
            'path': str(self.path), 'size_bytes': self.path.stat().st_size,
            'mtime_ns': self.path.stat().st_mtime_ns, 'n_loci': len(self.cpg_idx),
            'n_tracks': self.n_tracks, 'nnz': len(indices),
            'cpg_idx_sha256': array_sha256(self.cpg_idx), 'track_indptr_sha256': array_sha256(indptr),
            'track_indices_sha256': array_sha256(indices),
            'attrs': {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in self.attrs.items()}}
        matrix = sparse.csr_matrix((np.ones(len(indices), np.float32), indices.astype(np.int32, copy=False),
                                    indptr), shape=(len(self.cpg_idx), self.n_tracks))
        matrix.sum_duplicates()
        if matrix.nnz and matrix.data.max() != 1:
            raise ValueError('Duplicate track overlaps in CSR')
        self.matrix = matrix
        self._order = np.argsort(self.cpg_idx, kind='stable')

    def _read(self, handle: h5py.File, name: str):
        if name in FORBIDDEN_DATASETS or name not in STORE_DATASETS_READ:
            raise RuntimeError(f'Refusing to read dataset {name!r}: only {STORE_DATASETS_READ} are allowed')
        self.datasets_read.append(name)
        return handle[name][:]

    def rows_for(self, loci: np.ndarray) -> np.ndarray:
        loci = np.asarray(loci, dtype=np.int64)
        sorted_ids = self.cpg_idx[self._order]
        pos = np.searchsorted(sorted_ids, loci)
        pos = np.minimum(pos, len(sorted_ids) - 1)
        found = sorted_ids[pos] == loci
        if not found.all():
            raise KeyError(f'{int((~found).sum())} requested loci are absent from the store')
        return self._order[pos]


# -------------------------------------------------------------------------------------------------- group weights

def replicate_groups(frame: pd.DataFrame, columns: tuple[int, ...]) -> np.ndarray:
    """Group id per selected track by (assay, biosample, target); DNase (no target) groups by (assay, biosample)."""
    sub = frame.set_index('feature_column').loc[list(columns)]
    target = sub.encode_target.where(sub.encode_assay != 'DNase-seq', '(no target)')
    keys = pd.MultiIndex.from_arrays([sub.encode_assay, sub.encode_biosample, target])
    return pd.factorize(keys)[0].astype(np.int64)


def replicate_weights(groups: np.ndarray, weighting: str) -> np.ndarray:
    if weighting not in REPLICATE_WEIGHTINGS:
        raise ValueError(f'replicate_weighting must be one of {REPLICATE_WEIGHTINGS}')
    if weighting == 'none':
        return np.ones(len(groups), np.float32)
    sizes = np.bincount(groups)[groups]
    return (1.0 / np.sqrt(sizes)).astype(np.float32)


def group_summary(groups: np.ndarray, blocks: np.ndarray | None = None) -> dict:
    sizes = np.bincount(groups)
    out = {'n_tracks': len(groups), 'n_groups': len(sizes),
           'n_groups_gt1': int((sizes > 1).sum()), 'n_tracks_in_groups_gt1': int(sizes[sizes > 1].sum()),
           'max_group_size': int(sizes.max()),
           'size_histogram': {str(int(s)): int(c) for s, c in zip(*np.unique(sizes, return_counts=True))}}
    if blocks is not None:
        out['per_block'] = {b: {'n_tracks': int((blocks == b).sum()),
                                'n_groups': len(np.unique(groups[blocks == b]))} for b in BLOCKS
                            if (blocks == b).any()}
    return out


# --------------------------------------------------------------------------------------------------------- SVD

def _flip_signs(components: np.ndarray) -> np.ndarray:
    """Deterministic sign convention: the largest-|loading| entry of every component is positive."""
    idx = np.argmax(np.abs(components), axis=1)
    sign = np.sign(components[np.arange(len(components)), idx])
    sign[sign == 0] = 1
    return components * sign[:, None]


def _svd_components(x, k: int, seed: int, n_iter: int, n_oversamples: int) -> np.ndarray:
    k = int(min(k, x.shape[1], x.shape[0] - 1))
    if k < 1:
        raise ValueError('Matrix too small for SVD')
    svd = TruncatedSVD(n_components=k, algorithm='randomized', n_iter=n_iter, n_oversamples=n_oversamples,
                       random_state=seed)
    svd.fit(x)
    return _flip_signs(svd.components_.astype(np.float32))


def _chunks(n: int, size: int):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


# ---------------------------------------------------------------------------------------------- compressor

class RegulatoryCompressor:
    """fit(store, fit_loci) / transform(store, loci) / save(path) / load(path) for a label-free linear compression."""

    def __init__(self, feature_set: FeatureSet, *, method: str = 'global_svd', n_components: int = 256,
                 block_components: int = 256, final_projection: bool = True, replicate_weighting: str = 'none',
                 seed: int = 17, n_iter: int = 7, n_oversamples: int = 20, catalog: str | Path = DEFAULT_CATALOG,
                 chunk_rows: int = 50_000):
        if method not in METHODS:
            raise ValueError(f'method must be one of {METHODS}')
        if replicate_weighting not in REPLICATE_WEIGHTINGS:
            raise ValueError(f'replicate_weighting must be one of {REPLICATE_WEIGHTINGS}')
        self.feature_set, self.method, self.seed = feature_set, method, int(seed)
        self.n_components, self.block_components = int(n_components), int(block_components)
        self.final_projection = bool(final_projection)
        self.replicate_weighting = replicate_weighting
        self.n_iter, self.n_oversamples, self.chunk_rows = int(n_iter), int(n_oversamples), int(chunk_rows)
        self.catalog_path = Path(catalog)
        self.columns = np.asarray(feature_set.feature_columns, dtype=np.int64)
        frame = read_catalog(self.catalog_path)
        if tuple(frame.track_id.iloc[self.columns]) != tuple(feature_set.track_ids):
            raise ValueError('Catalog does not match the resolved feature set (track ids differ)')
        self.blocks = frame.encode_assay.map(ASSAY_BLOCK).iloc[self.columns].to_numpy()
        if pd.isna(self.blocks).any():
            raise ValueError('Selected tracks include an assay outside the three contract blocks')
        self.groups = replicate_groups(frame, feature_set.feature_columns)
        self.weights = replicate_weights(self.groups, replicate_weighting)
        self.catalog_sha256 = file_sha256(self.catalog_path)
        self.columns_sha256 = array_sha256(self.columns)
        self.fitted = False
        self.projection: np.ndarray | None = None
        self.arrays: dict[str, np.ndarray] = {}
        self.info: dict = {}

    # ---- helpers
    @property
    def dim(self) -> int:
        if not self.fitted:
            raise RuntimeError('Compressor is not fitted')
        return int(self.projection.shape[0])

    def _x(self, store: TrackStore, rows: np.ndarray):
        if store.n_tracks != self.info.get('n_store_tracks', store.n_tracks):
            raise ValueError('Store track count differs from the one used at fit')
        return store.matrix[rows][:, self.columns]

    def _apply(self, x, weights: np.ndarray):
        return x @ sparse.diags(weights.astype(np.float32), format='csr')

    # ---- fit
    def fit(self, store: TrackStore, fit_loci: FitLoci | None = None) -> RegulatoryCompressor:
        if fit_loci is None:
            raise ValueError('fit() requires explicit fit_loci (a FitLoci); there is no default to all loci')
        if not isinstance(fit_loci, FitLoci):
            raise TypeError('fit() requires explicit fit_loci as a FitLoci, not a raw array')
        if store.n_tracks <= int(self.columns.max()):
            raise ValueError('Store has fewer tracks than the feature set requires')
        started = time.monotonic()
        rows = store.rows_for(fit_loci.cpg_idx)
        self.info['n_store_tracks'] = store.n_tracks
        x = self._x(store, rows)  # binary, n_fit x n_selected (CSR)
        n = x.shape[0]
        p = np.asarray(x.getnnz(axis=0), dtype=np.float64) / n
        w2 = self.weights.astype(np.float64) ** 2
        stats = {'total_energy': float((w2 * p).sum()), 'total_variance': float((w2 * p * (1 - p)).sum())}
        arrays: dict[str, np.ndarray] = {}
        if self.method == 'global_svd':
            comps = _svd_components(self._apply(x, self.weights), self.n_components, self.seed, self.n_iter,
                                    self.n_oversamples)
            arrays['components_weighted'] = comps
            projection = comps * self.weights[None, :]
        else:
            projection, arrays, stats = self._fit_block(x, p, stats)
        self.projection = projection.astype(np.float32)
        self.arrays = arrays
        self.fitted = True
        # final stats from the actual end-to-end embedding of the fit loci
        z = self._embed(x)
        stats['embedding_energy'] = (z.astype(np.float64) ** 2).mean(axis=0)
        stats['embedding_variance'] = z.astype(np.float64).var(axis=0)
        stats['embedding_mean_sq_row_norm'] = float((z.astype(np.float64) ** 2).sum(axis=1).mean())
        for key in ('total_energy', 'total_variance', 'embedding_mean_sq_row_norm'):
            stats[key] = float(stats[key])
        self.arrays.update({f'stat_{k}': np.asarray(v) for k, v in stats.items()})
        self.info.update(
            fit_loci={'protocol': fit_loci.name, 'n': fit_loci.n, 'sha256': fit_loci.sha256},
            source_store=store.identity, fit_seconds=time.monotonic() - started,
            stats_summary=self._summary(stats), datasets_read=sorted(set(store.datasets_read)))
        if 'dense' in store.datasets_read:
            raise AssertionError('/dense was read')
        return self

    def _fit_block(self, x, p, stats):
        n = x.shape[0]
        n_sel = x.shape[1]
        arrays: dict[str, np.ndarray] = {}
        w2 = self.weights.astype(np.float64) ** 2
        parts, loadings_blocks = [], []
        for b in BLOCKS:
            idx = np.flatnonzero(self.blocks == b)
            if idx.size == 0:
                continue
            xb = self._apply(x[:, idx], self.weights[idx])
            comps = _svd_components(xb, self.block_components, self.seed, self.n_iter, self.n_oversamples)
            load = comps * self.weights[idx][None, :]  # acts on raw binary tracks
            z = np.concatenate([np.asarray(x[s][:, idx] @ load.T) for s in _chunks(n, self.chunk_rows)])
            fro = float(np.sqrt((z.astype(np.float64) ** 2).sum()))
            scale = float(np.sqrt(n) / fro)  # mean squared row norm of the block over fit loci becomes 1
            arrays[f'block_{b}_components_weighted'] = comps
            arrays[f'block_{b}_columns'] = idx
            stats[f'block_{b}_total_energy'] = float((w2[idx] * p[idx]).sum())
            stats[f'block_{b}_total_variance'] = float((w2[idx] * p[idx] * (1 - p[idx])).sum())
            stats[f'block_{b}_pre_comp_energy'] = (z.astype(np.float64) ** 2).mean(axis=0)
            stats[f'block_{b}_pre_frobenius_sq_per_locus'] = float(fro ** 2 / n)
            stats[f'block_{b}_scale'] = scale
            stats[f'block_{b}_post_frobenius_sq_per_locus'] = float((fro * scale) ** 2 / n)
            parts.append(z * np.float32(scale))
            full = np.zeros((comps.shape[0], n_sel), np.float32)
            full[:, idx] = load * np.float32(scale)
            loadings_blocks.append(full)
        loadings = np.concatenate(loadings_blocks)
        arrays['block_order'] = np.asarray([b for b in BLOCKS if (self.blocks == b).any()])
        arrays['block_dims'] = np.asarray([a.shape[0] for a in loadings_blocks])
        # energy of the block-normalized concatenation (= n_blocks by construction): denominator of block explained energy
        stats['concat_total_energy'] = float(sum(float((q.astype(np.float64) ** 2).sum(axis=1).mean()) for q in parts))
        if not self.final_projection:
            return loadings, arrays, stats
        cat = np.concatenate(parts, axis=1)
        proj = _svd_components(cat, self.n_components, self.seed, self.n_iter, self.n_oversamples)
        arrays['final_projection'] = proj
        return proj @ loadings, arrays, stats

    def _summary(self, stats: dict) -> dict:
        e = np.asarray(stats['embedding_energy'])
        v = np.asarray(stats['embedding_variance'])
        out = {'dim': int(e.size)}
        if self.method == 'block_svd':
            # Raw-input denominators are meaningless after block rescaling: use the normalized concatenation instead.
            total = stats['concat_total_energy']
            for d in EXPLAINED_AT:
                if d <= e.size:
                    out[f'explained_energy_at_{d}'] = float(e[:d].sum() / total)
            out['explained_energy_total'] = float(e.sum() / total)
            out['explained_energy_denominator'] = 'energy of the block-normalized concatenation of block embeddings'
            return out
        for d in EXPLAINED_AT:
            if d <= e.size:
                out[f'explained_energy_at_{d}'] = float(e[:d].sum() / stats['total_energy'])
                out[f'explained_variance_at_{d}'] = float(v[:d].sum() / stats['total_variance'])
        out['explained_energy_total'] = float(e.sum() / stats['total_energy'])
        out['explained_variance_total'] = float(v.sum() / stats['total_variance'])
        return out

    # ---- transform
    def _embed(self, x) -> np.ndarray:
        out = np.empty((x.shape[0], self.projection.shape[0]), np.float32)
        proj_t = np.ascontiguousarray(self.projection.T)
        for s in _chunks(x.shape[0], self.chunk_rows):
            out[s] = x[s] @ proj_t
        return out

    def transform(self, store: TrackStore, loci: np.ndarray | None = None,
                  feature_set: FeatureSet | None = None) -> np.ndarray:
        """Embed `loci` (default: every store locus) with the fitted projection. Chunked; never densifies."""
        if not self.fitted:
            raise RuntimeError('Compressor is not fitted')
        if feature_set is not None:
            self.check_feature_set(feature_set)
        if store.n_tracks != self.info['n_store_tracks']:
            raise ValueError('Store track count differs from the one used at fit')
        rows = np.arange(len(store.cpg_idx)) if loci is None else store.rows_for(loci)
        out = np.empty((len(rows), self.dim), np.float32)
        proj_t = np.ascontiguousarray(self.projection.T)
        for s in _chunks(len(rows), self.chunk_rows):
            xc = store.matrix[rows[s]][:, self.columns]
            out[s] = xc @ proj_t
        if not np.isfinite(out).all():
            raise FloatingPointError('Non-finite values in embedding')
        return out

    def check_feature_set(self, feature_set: FeatureSet) -> None:
        if feature_set.manifest_hash != self.feature_set.manifest_hash:
            raise ValueError('Feature-set manifest hash mismatch: '
                             f'{feature_set.manifest_hash} != {self.feature_set.manifest_hash}')
        if array_sha256(np.asarray(feature_set.feature_columns, dtype=np.int64)) != self.columns_sha256:
            raise ValueError('Feature-set columns mismatch')

    # ---- manifest / persistence
    def manifest(self) -> dict:
        if not self.fitted:
            raise RuntimeError('Compressor is not fitted')
        fs = self.feature_set
        return {
            'format_version': FORMAT_VERSION, 'kind': 'regulatory_compressor',
            'feature_set': {'name': fs.name, 'manifest_hash': fs.manifest_hash, 'n_features': fs.n_selected,
                            'feature_columns_sha256': self.columns_sha256,
                            'counts_per_block': dict(fs.counts_per_block), 'spec': fs.spec},
            'catalog': {'path': str(self.catalog_path), 'sha256': self.catalog_sha256},
            'method': self.method,
            'params': {'n_components': self.n_components, 'block_components': self.block_components,
                       'final_projection': self.final_projection, 'n_iter': self.n_iter,
                       'n_oversamples': self.n_oversamples, 'svd_algorithm': 'randomized', 'chunk_rows': self.chunk_rows},
            'dim': self.dim, 'seed': self.seed,
            'scaling': {'input': 'binary 0/1 peak overlap', 'centering': False, 'column_scaling': 'none',
                        'sign_convention': 'largest-abs loading of each component positive',
                        'block_normalization': ('per-block mean squared row norm over fit loci = 1 (Frobenius)'
                                                if self.method == 'block_svd' else None),
                        'block_dim_rule': ('min(block_components, n_block_tracks, n_fit-1) per block'
                                           if self.method == 'block_svd' else None),
                        'explained_energy': ('sum_i z_i^2 / sum ||x_w||^2 on fit loci (uncentered)'
                                             if self.method == 'global_svd' else
                                             'sum_i z_i^2 / energy of the block-normalized concatenation (fit loci)'),
                        'explained_variance': ('sum_i var(z_i) / sum_j var(x_wj) on fit loci (centered, same projection)'
                                               if self.method == 'global_svd' else 'not defined for block_svd')},
            'replicate_weighting': self.replicate_weighting,
            'replicate_groups': group_summary(self.groups, self.blocks),
            'assertions': {'dense_read': False, 'methylation_read': False, 'patient_specific': False,
                           'densified_full_matrix': False, 'store_opened_read_only': True,
                           'datasets_read': self.info.get('datasets_read')},
            'fit_loci': self.info['fit_loci'], 'source_store': self.info['source_store'],
            'fit_seconds': self.info['fit_seconds'], 'stats_summary': self.info['stats_summary'],
            'libraries': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                          'scikit-learn': sklearn.__version__, 'h5py': h5py.__version__, 'pandas': pd.__version__},
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = self.manifest()
        payload = {'projection': self.projection, 'weights': self.weights, 'groups': self.groups,
                   'blocks': self.blocks.astype(str), 'columns': self.columns,
                   'track_ids': np.asarray(self.feature_set.track_ids),
                   'manifest_json': np.asarray(json.dumps(manifest, sort_keys=True, allow_nan=False)),
                   'catalog_path': np.asarray(str(self.catalog_path)),
                   **{f'a__{k}': v for k, v in self.arrays.items()}}
        tmp = path.with_name(path.name + '.tmp.npz')
        np.savez_compressed(tmp, **payload)
        tmp.replace(path)
        path.with_suffix('.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
        return path

    @classmethod
    def load(cls, path: str | Path, feature_set: FeatureSet | None = None) -> RegulatoryCompressor:
        with np.load(path, allow_pickle=False) as z:
            manifest = json.loads(str(z['manifest_json']))
            self = cls.__new__(cls)
            fs = manifest['feature_set']
            columns = z['columns'].astype(np.int64)
            if array_sha256(columns) != fs['feature_columns_sha256']:
                raise ValueError('Saved feature columns do not match the manifest hash')
            self.feature_set = _StubFeatureSet(fs['name'], fs['manifest_hash'], tuple(int(c) for c in columns),
                                               tuple(str(t) for t in z['track_ids']), fs['spec'],
                                               fs['counts_per_block'])
            self.columns_sha256 = fs['feature_columns_sha256']
            if feature_set is not None:
                self.check_feature_set(feature_set)
                self.feature_set = feature_set
            self.method, self.seed = manifest['method'], int(manifest['seed'])
            prm = manifest['params']
            self.n_components, self.block_components = prm['n_components'], prm['block_components']
            self.final_projection, self.n_iter = prm['final_projection'], prm['n_iter']
            self.n_oversamples, self.chunk_rows = prm['n_oversamples'], prm['chunk_rows']
            self.replicate_weighting = manifest['replicate_weighting']
            self.catalog_path = Path(manifest['catalog']['path'])
            self.catalog_sha256 = manifest['catalog']['sha256']
            self.columns, self.columns_sha256 = columns, fs['feature_columns_sha256']
            self.blocks, self.groups, self.weights = z['blocks'], z['groups'], z['weights']
            self.projection = z['projection']
            self.arrays = {k[3:]: z[k] for k in z.files if k.startswith('a__')}
            self.info = {'n_store_tracks': manifest['source_store']['n_tracks'], 'fit_loci': manifest['fit_loci'],
                         'source_store': manifest['source_store'], 'fit_seconds': manifest['fit_seconds'],
                         'stats_summary': manifest['stats_summary'],
                         'datasets_read': manifest['assertions']['datasets_read']}
            self.fitted = True
        return self


@dataclass(frozen=True)
class _StubFeatureSet:
    """Minimal FeatureSet stand-in restored from a saved compressor (identity fields only)."""
    name: str
    manifest_hash: str
    feature_columns: tuple
    track_ids: tuple
    spec: dict
    counts_per_block: dict

    @property
    def n_selected(self) -> int:
        return len(self.feature_columns)
