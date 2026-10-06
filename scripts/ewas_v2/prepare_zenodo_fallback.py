#!/usr/bin/env python3
"""Prepare a compact Catalog-format snapshot from verified official Zenodo results.

Metadata are curated, sourced rows in data/registry/ewas_catalog_expansion/
zenodo_analysis_metadata.csv. Internal source_analysis_id is not a Catalog StudyID.
All source results are scanned for exact counts; compact TSV keeps all p<1e-4
hits and one coverage witness per analysis. No result-based primary selection.
"""
from __future__ import annotations
import gzip
import hashlib
import io
import json
import re
from pathlib import Path
import sys
import tarfile
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from cpg_repr_benchmark.data.coordinates import encode_many

RAW = ROOT / 'data/raw/ewas_catalog_expansion/zenodo'
OUT = ROOT / 'data/raw/ewas_catalog_expansion/zenodo_prepared'
META = ROOT / 'data/registry/ewas_catalog_expansion/zenodo_analysis_metadata.csv'
THRESHOLD = 1e-4


def main():
    metadata = pd.read_csv(META, keep_default_na=False, dtype=str)
    if metadata.source_analysis_id.duplicated().any():
        raise ValueError('Source analysis keys must be unique')
    probes = pd.read_parquet(ROOT / 'data/processed/ComputAgeBench/illumina_probe_grch38.parquet')
    loci = pd.read_parquet(ROOT / 'outputs/encode_atlas_v1/loci.parquet')
    universe = set(encode_many(loci.chr, loci.pos))
    in_universe = set(probes.loc[np.isin(encode_many(probes.chr, probes.pos), list(universe)), 'probe_id'])
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / 'analysis_cache'
    cache.mkdir(exist_ok=True)
    counts, source_hashes = {}, {}
    result_path = OUT / 'ewascatalog-results.txt.gz'
    with gzip.open(result_path, 'wt') as output:
        output.write('cpg\tp\tstudy_id\n')

        def scan(stream, row):
            sid = row['source_analysis_id']
            cache_name = re.sub(r'[^A-Za-z0-9_.-]', '_', sid)
            cmeta, chits = cache / f'{cache_name}.json', cache / f'{cache_name}.tsv.gz'
            key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
            if cmeta.exists() and chits.exists():
                saved = json.loads(cmeta.read_text())
                if saved['metadata_sha256'] == key and saved['source_sha256'] == source_hashes[row['data_container']]:
                    counts[sid] = saved['counts']
                    with gzip.open(chits, 'rt') as f:
                        while b := f.read(4 << 20):
                            output.write(b)
                    print('cached', sid, flush=True)
                    return
            if row['family'] == 'out_of_scope':
                counts[sid] = dict(n_observed_universe_loci=None, n_tested_probes=None,
                                   n_positive_raw_source=None, n_subjects_per_probe_min=None,
                                   n_subjects_per_probe_max=None, results_scanned=False)
                return
            positive, covered, all_probes = {}, set(), set()
            max_n, min_n = None, None
            wanted = {row.get('cpg_column') or 'cpg', row.get('p_column') or 'p',
                      'totalsamplesize', 'totaln', 'nmiss', 'n'}
            frame = pd.read_csv(stream, sep='\t' if row['separator'] == 'tab' else ',',
                                encoding=row.get('encoding') or 'utf-8-sig', chunksize=150000, dtype=str,
                                usecols=lambda c: c.strip().casefold() in wanted)
            for chunk in frame:
                chunk.columns = [c.strip().casefold() for c in chunk.columns]
                cpcol = row.get('cpg_column') or 'cpg'
                pcol = row.get('p_column') or 'p'
                cp = chunk[cpcol].fillna('')
                pv = pd.to_numeric(chunk[pcol], errors='coerce')
                all_probes.update(cp[cp.str.startswith('cg')])
                covered.update(set(cp) & in_universe)
                hit = cp[(pv < THRESHOLD) & cp.str.startswith('cg')]
                for probe, p in zip(hit, pv.loc[hit.index]):
                    positive[probe] = min(positive.get(probe, 1.), float(p))
                for ncol in ['totalsamplesize', 'totaln', 'nmiss', 'n']:
                    if ncol in chunk:
                        ns = pd.to_numeric(chunk[ncol], errors='coerce').dropna()
                        if len(ns):
                            max_n = max(max_n or 0, float(ns.max()))
                            min_n = min(min_n or float('inf'), float(ns.min()))
                        break
            with gzip.open(chits, 'wt') as f:
                for probe, p in sorted(positive.items()):
                    line = f'{probe}\t{p:.17g}\t{sid}\n'
                    f.write(line)
                    output.write(line)
                if covered:
                    line = f'{min(covered)}\t1\t{sid}\n'
                    f.write(line)
                    output.write(line)
            counts[sid] = dict(n_observed_universe_loci=len(covered), n_tested_probes=len(all_probes),
                               n_positive_raw_source=len(positive), n_subjects_per_probe_min=min_n,
                               n_subjects_per_probe_max=max_n)
            cmeta.write_text(json.dumps(dict(metadata_sha256=key,
                source_sha256=source_hashes[row['data_container']], counts=counts[sid]), indent=2))
            print(sid, len(positive), len(covered), flush=True)

        groups = list(metadata.groupby('data_container', sort=True))
        groups.sort(key=lambda pair: (pair[0].startswith('4672645/'), pair[0]))
        for container, group in groups:
            path = RAW / container
            verified = json.loads(path.with_suffix(path.suffix + '.verified.json').read_text())
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                while b := f.read(4 << 20):
                    h.update(b)
            if h.hexdigest() != verified['sha256']:
                raise ValueError(f'Changed source archive: {path}')
            source_hashes[container] = h.hexdigest()
            if path.name.endswith('.tar.gz'):
                remaining = {r['member_name']: r for r in group.to_dict('records')}
                with tarfile.open(path, 'r|gz') as archive:
                    for member in archive:
                        if member.isfile() and member.name in remaining:
                            scan(io.BytesIO(archive.extractfile(member).read()), remaining.pop(member.name))
                if remaining:
                    raise ValueError(f'Absent archive members: {list(remaining)}')
            elif path.suffix == '.zip':
                with zipfile.ZipFile(path) as archive:
                    for row in group.to_dict('records'):
                        with archive.open(row['member_name']) as f:
                            scan(f, row)
            else:
                for row in group.to_dict('records'):
                    with (gzip.open(path, 'rb') if path.suffix == '.gz' else open(path, 'rb')) as f:
                        scan(f, row)
    metadata['study_id'] = metadata.source_analysis_id
    metadata['catalog_study_id'] = ''
    metadata['study_id_provenance'] = 'Zenodo record ID + original results filename; Catalog StudyID unavailable'
    metadata['n_observed_universe_loci'] = metadata.source_analysis_id.map(lambda s: counts[s]['n_observed_universe_loci'])
    metadata['n_tested_probes'] = metadata.source_analysis_id.map(lambda s: counts[s]['n_tested_probes'])
    metadata['n_positive_raw_source'] = metadata.source_analysis_id.map(lambda s: counts[s]['n_positive_raw_source'])
    for field in ['n_subjects_per_probe_min', 'n_subjects_per_probe_max']:
        metadata[field] = metadata.source_analysis_id.map(lambda s: counts[s][field])
    # When a file reports per-probe N, expose its maximum as analysis N with range retained.
    for i, row in metadata.iterrows():
        nmax = counts[row.source_analysis_id]['n_subjects_per_probe_max']
        if nmax is not None:
            metadata.loc[i, 'n'] = str(int(nmax))
            metadata.loc[i, 'n_provenance'] = 'maximum per-probe N in official summary statistics; range recorded separately'
    metadata.to_csv(OUT / 'ewascatalog-studies.txt.gz', sep='\t', index=False, compression='gzip')
    (OUT / 'preparation_manifest.json').write_text(json.dumps(dict(
        scope='OFFICIAL_ZENODO_FALLBACK; incomplete coverage of the full EWAS Catalog',
        p_threshold=THRESHOLD, compact_results=True, counts=counts, source_sha256=source_hashes,
        metadata_sha256=hashlib.sha256(META.read_bytes()).hexdigest()), indent=2) + '\n')


if __name__ == '__main__':
    main()
