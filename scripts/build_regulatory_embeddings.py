#!/usr/bin/env python3
"""Materialize label-free regulatory embeddings (fit on a named protocol, transform ALL store loci).

Reads only the track CSR of the feature store (never /dense, never methylation). For every entry of the config writes
  <output_dir>/<name>.h5                 /cpg_idx int64 + /embedding float32 (canonical store contract)
  <output_dir>/<name>.h5.json            sidecar manifest (full compressor manifest + embedding info)
  <output_dir>/<name>.compressor.npz     saved compressor (+ <name>.compressor.json)
with name = <feature_set>__<method><dim>[_egm]__<protocol>. Rows of held-out chromosomes (chr20-22) are out-of-fit.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml

from cpg_repr_benchmark.encode_atlas.compression import (
    RegulatoryCompressor,
    TrackStore,
    array_sha256,
    assert_disjoint,
    split_loci,
)
from cpg_repr_benchmark.encode_atlas.feature_sets import resolve_feature_set


def embedding_name(entry: dict, protocol: str) -> str:
    suffix = '_egm' if entry.get('replicate_weighting', 'none') == 'equal_group_mass' else ''
    return f"{entry['feature_set']}__{entry['method']}{entry['dim']}{suffix}__{protocol}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=Path('configs/regulatory_embeddings.yaml'))
    p.add_argument('--only', nargs='*', help='embedding names to build (default: all in config)')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out_dir = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    store = TrackStore(cfg['store'])
    fit, heldout = split_loci(cfg['protocol'], store.cpg_idx, cfg['registry'])
    assert_disjoint(fit, heldout)
    print(f'fit {fit.name}: n={fit.n} sha256={fit.sha256}; held-out n={heldout.size}', flush=True)
    for entry in cfg['embeddings']:
        name = embedding_name(entry, cfg['protocol'])
        if args.only and name not in args.only:
            continue
        target = out_dir / f'{name}.h5'
        if target.exists() and not args.overwrite:
            print(f'skip existing {target}', flush=True)
            continue
        fs = resolve_feature_set(entry['feature_set'], cfg['catalog'], check_expected=True)
        comp = RegulatoryCompressor(
            fs, method=entry['method'], n_components=entry['dim'],
            block_components=entry.get('block_components', 256), final_projection=entry.get('final_projection', True),
            replicate_weighting=entry.get('replicate_weighting', 'none'), seed=cfg['seed'], catalog=cfg['catalog'])
        comp.fit(store, fit)
        emb = comp.transform(store)  # every store locus, store order
        assert emb.shape == (len(store.cpg_idx), comp.dim) and np.isfinite(emb).all()
        tmp = target.with_suffix('.tmp.h5')
        with h5py.File(tmp, 'w') as h:
            h.create_dataset('cpg_idx', data=store.cpg_idx, dtype='int64')
            h.create_dataset('embedding', data=emb.astype(cfg['dtype']), dtype=cfg['dtype'])
            h.attrs.update(representation=name, patient_specific=False, supervision='none', reference_build='GRCh38',
                           cpg_namespace='grch38_cpg_cytosine_1based_v1', created_utc=datetime.now(timezone.utc).isoformat(),
                           fit_protocol=fit.name, n_fit_loci=fit.n, fit_loci_sha256=fit.sha256,
                           feature_set=fs.name, feature_set_manifest_hash=fs.manifest_hash,
                           note='rows of chr20-22 are out-of-fit (transformed only)')
        tmp.replace(target)
        comp.save(out_dir / f'{name}.compressor.npz')
        manifest = comp.manifest()
        manifest['embedding'] = {'name': name, 'h5': str(target), 'shape': list(emb.shape), 'dtype': cfg['dtype'],
                                 'size_bytes': target.stat().st_size, 'n_heldout_rows_out_of_fit': int(heldout.size),
                                 'heldout_loci_sha256': array_sha256(heldout),
                                 'registered_as_benchmark_arm': False}
        Path(str(target) + '.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
        print(f"{name}: shape={emb.shape} fit_s={manifest['fit_seconds']:.0f} "
              f"summary={manifest['stats_summary']}", flush=True)


if __name__ == '__main__':
    main()
