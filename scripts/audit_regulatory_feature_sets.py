#!/usr/bin/env python3
"""Audit regulatory feature sets against the ENCODE catalog (metadata only; no store is opened)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cpg_repr_benchmark.encode_atlas.feature_sets import (
    DEFAULT_CATALOG,
    annotate,
    expected_vs_actual,
    list_feature_sets,
    load_spec,
    read_catalog,
    resolve_feature_set,
)
from cpg_repr_benchmark.encode_atlas.features import digest


def audit(name: str, catalog: Path, out_dir: Path) -> dict:
    fs = resolve_feature_set(name, catalog)
    table = annotate(read_catalog(catalog), load_spec(name))
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f'{fs.name}.tsv', sep='\t', index=False)
    comparison = expected_vs_actual(fs)
    manifest = {
        'name': fs.name, 'rule_spec': fs.spec, 'catalog_path': str(catalog), 'catalog_sha256': digest(catalog),
        'n_catalog': fs.n_catalog, 'n_selected': fs.n_selected, 'n_dense': len(fs.dense_columns),
        'counts_per_assay': fs.counts_per_assay, 'counts_per_block': fs.counts_per_block,
        'counts_per_functional_block': fs.counts_per_functional_block,
        'excluded_targets': list(fs.excluded_targets), 'excluded_tracks': list(fs.excluded_tracks),
        'manifest_hash': fs.manifest_hash, 'expected_vs_actual': comparison,
        'expected_all_match': all(v['match'] for v in comparison.values())}
    (out_dir / f'{fs.name}.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature-set', default='all', help="name, config path, or 'all'")
    p.add_argument('--catalog', type=Path, default=DEFAULT_CATALOG)
    p.add_argument('--out-dir', type=Path, default=Path('outputs/feature_set_audits'))
    args = p.parse_args()
    names = list_feature_sets() if args.feature_set == 'all' else [args.feature_set]
    for name in names:
        m = audit(name, args.catalog, args.out_dir)
        print(f"{m['name']}: selected={m['n_selected']} hash={m['manifest_hash']} "
              f"expected_match={m['expected_all_match']}")


if __name__ == '__main__':
    main()
