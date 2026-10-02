"""Metadata-driven resolution of regulatory feature sets from the ENCODE track catalog.

A feature set is a declarative rule (assays included, targets excluded) applied to the catalog. Store column
c is catalog ROW c (0-based), not the non-contiguous `track_index`; both are exposed. The manifest hash covers
stable track identity (track_id + file accession) plus the rule spec, so it is independent of row order.
Nothing here opens the feature store or reads methylation data.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_CATALOG = Path('data/external/functional/locus_features_v1/regulatory/track_contract.tsv')
CONFIG_DIR = Path(__file__).resolve().parents[3] / 'configs' / 'feature_sets'
REQUIRED_COLUMNS = ('track_index', 'track_id', 'file_accession', 'encode_accession', 'encode_assay',
                    'encode_target', 'encode_biosample', 'functional_block')
# Contract blocks (docs/REGULATORY_REPRESENTATION.md section 0): `tf` merges tf_binding + ctcf.
ASSAY_BLOCK = {'Histone ChIP-seq': 'histone', 'TF ChIP-seq': 'tf', 'DNase-seq': 'accessibility'}
SPEC_KEYS = ('name', 'assays', 'exclude_targets', 'dense')


@dataclass(frozen=True)
class FeatureSet:
    name: str
    spec: dict
    feature_columns: tuple[int, ...]  # store column space (= catalog row order)
    track_indices: tuple[int, ...]  # catalog `track_index`, aligned with feature_columns
    track_ids: tuple[str, ...]  # aligned with feature_columns
    dense_columns: tuple[int, ...]
    counts_per_assay: dict
    counts_per_block: dict  # contract block (tf = tf_binding + ctcf)
    counts_per_functional_block: dict  # raw catalog functional_block
    excluded_targets: tuple[str, ...]
    excluded_tracks: tuple[dict, ...]  # tracks removed by the target rule, with reasons
    n_catalog: int
    manifest_hash: str

    @property
    def n_selected(self) -> int:
        return len(self.feature_columns)


def list_feature_sets(config_dir: Path = CONFIG_DIR) -> list[str]:
    return sorted(p.stem for p in Path(config_dir).glob('*.yaml'))


def load_spec(name_or_path: str | Path, config_dir: Path = CONFIG_DIR) -> dict:
    path = Path(name_or_path)
    if not (path.suffix in {'.yaml', '.yml'} and path.is_file()):
        path = Path(config_dir) / f'{name_or_path}.yaml'
        if not path.is_file():
            raise ValueError(f'Unknown feature set {str(name_or_path)!r}; available: {list_feature_sets(config_dir)}')
    spec = yaml.safe_load(path.read_text())
    missing = [k for k in SPEC_KEYS if k not in spec]
    if missing:
        raise ValueError(f'Feature-set config {path} lacks keys {missing}')
    if spec['dense'] != 'none':
        raise ValueError(f"{path}: only dense: none is supported, got {spec['dense']!r}")
    return spec


def read_catalog(path: Path) -> pd.DataFrame:
    """Read the catalog preserving row order; adds `feature_column` = row position (the store column)."""
    frame = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame]
    if missing:
        raise ValueError(f'Missing catalog columns: {missing}')
    frame = frame.replace('', 'unknown')
    for column in ('track_id', 'file_accession'):
        dup = frame.loc[frame[column].duplicated(), column].tolist()
        if dup:
            raise ValueError(f'Duplicate {column} in catalog: {dup[:5]}')
    frame['track_index'] = frame['track_index'].astype(int)
    frame.insert(0, 'feature_column', range(len(frame)))
    return frame


def annotate(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Per-track include decision and reason under `spec` (one row per catalog track, catalog order)."""
    assays, excluded = set(spec['assays']), set(spec['exclude_targets'])
    out = frame[['feature_column', 'track_index', 'track_id', 'encode_accession', 'file_accession',
                 'encode_assay', 'encode_target', 'encode_biosample', 'functional_block']].copy()
    out['block'] = out.encode_assay.map(ASSAY_BLOCK).fillna('other')
    in_assay = out.encode_assay.isin(assays)
    bad_target = out.encode_target.isin(excluded)
    out['included'] = in_assay & ~bad_target
    reason = pd.Series(f"included: assay in {sorted(assays)}", index=out.index)
    reason[~in_assay] = 'excluded: assay not in block set'
    reason[in_assay & bad_target] = ('excluded: methylation-linked target ' + out.encode_target)[in_assay & bad_target]
    out['reason'] = reason
    return out


def _canonical(spec: dict, ids: list[tuple[str, str]]) -> bytes:
    payload = {'rule': {'name': spec['name'], 'assays': sorted(spec['assays']),
                        'exclude_targets': sorted(spec['exclude_targets']), 'dense': spec['dense']},
               'tracks': sorted(ids)}
    return json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()


def resolve_feature_set(name_or_path: str | Path, catalog: str | Path = DEFAULT_CATALOG, *,
                        config_dir: Path = CONFIG_DIR, check_expected: bool = False) -> FeatureSet:
    spec = load_spec(name_or_path, config_dir)
    frame = read_catalog(Path(catalog))
    table = annotate(frame, spec)
    chosen = table[table.included]
    if chosen.empty:
        raise ValueError(f"Feature set {spec['name']} selects no tracks")
    ids = list(zip(chosen.track_id, chosen.file_accession))
    # Tracks dropped by the target rule (assay-eligible rows only; assay exclusions are not "targets").
    dropped = table[table.reason.str.startswith('excluded: methylation-linked')]
    result = FeatureSet(
        name=spec['name'], spec=spec,
        feature_columns=tuple(int(c) for c in chosen.feature_column),
        track_indices=tuple(int(c) for c in chosen.track_index),
        track_ids=tuple(chosen.track_id), dense_columns=(),
        counts_per_assay=dict(sorted(chosen.encode_assay.value_counts().items())),
        counts_per_block=dict(sorted(chosen.block.value_counts().items())),
        counts_per_functional_block=dict(sorted(chosen.functional_block.value_counts().items())),
        excluded_targets=tuple(sorted(spec['exclude_targets'])),
        excluded_tracks=tuple(dropped[['feature_column', 'track_index', 'track_id', 'encode_target',
                                       'reason']].to_dict('records')),
        n_catalog=len(frame), manifest_hash=hashlib.sha256(_canonical(spec, ids)).hexdigest())
    if check_expected:
        verify_expected(result)
    return result


def expected_vs_actual(fs: FeatureSet) -> dict:
    expected = fs.spec.get('expected_counts') or {}
    actual = {'n_selected': fs.n_selected, 'n_excluded': fs.n_catalog - fs.n_selected,
              'per_assay': fs.counts_per_assay, 'per_block': fs.counts_per_block}
    return {k: {'expected': v, 'actual': actual[k], 'match': v == actual[k]} for k, v in expected.items()}


def verify_expected(fs: FeatureSet) -> None:
    bad = {k: v for k, v in expected_vs_actual(fs).items() if not v['match']}
    if bad:
        raise ValueError(f'{fs.name}: expected_counts mismatch: {bad}')
