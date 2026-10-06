#!/usr/bin/env python3
"""Catalog-only EWAS task inventory. Never opens representation stores or fits probes.

Run `download`, then `inventory`, review trait_categories.csv, then `build`.
Categories are explicitly assigned per Catalog trait, so no heuristic silently
excludes studies or merges distinct clinical endpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cpg_repr_benchmark.data.coordinates import encode_many
from cpg_repr_benchmark.ewas_v2.design import load_universe

FAMILIES = ['cancer', 'immune/inflammatory', 'cardiometabolic',
            'environmental/behavioral', 'psychiatric']
P_THRESHOLD = 1e-4  # User amendment, 2026-10-06; legacy benchmark remains p<1e-7.
URL = 'https://www.ewascatalog.org/static/docs/ewascatalog-{}.txt.gz'
RAW = ROOT / 'data/raw/ewas_catalog_expansion'
OUT = ROOT / 'data/derived/ewas_catalog_expansion'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(4 << 20):
            h.update(b)
    return h.hexdigest()


def normalize(frame):
    frame.columns = [re.sub(r'[^a-z0-9]+', '_', c.lower()).strip('_')
                     for c in frame.columns]
    return frame.rename(columns={'studyid': 'study_id', 'cpg_id': 'cpg'})


def tier(n):
    return 'main_eligible' if n >= 200 else 'exploratory' if n >= 100 else 'insufficient_support'


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    log = {}
    for kind in ['studies', 'results']:
        dest = RAW / f'ewascatalog-{kind}.txt.gz'
        tmp = dest.with_suffix('.part')
        attempts = []
        for attempt in range(9):
            try:
                req = urllib.request.Request(URL.format(kind), headers={'User-Agent': 'EWAS-benchmark/1.0'})
                with urllib.request.urlopen(req, timeout=60) as response, open(tmp, 'wb') as f:
                    while b := response.read(4 << 20):
                        f.write(b)
                    headers = dict(response.headers)
                with gzip.open(tmp, 'rt') as f:
                    if '\t' not in f.readline():
                        raise ValueError('download is not a gzip TSV')
                    while f.read(4 << 20):
                        pass  # Read through EOF to verify gzip CRC, not just the header.
                tmp.replace(dest)
                log[kind] = dict(url=URL.format(kind), status='downloaded',
                                 sha256=digest(dest), bytes=dest.stat().st_size, headers=headers)
                break
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                attempts.append(dict(attempt=attempt+1, error=str(exc)))
                log[kind] = dict(url=URL.format(kind), status='failed', error=str(exc))
                if attempt < 8:
                    time.sleep(10)
        log[kind]['failed_attempts'] = attempts
        log[kind]['attempt_utc'] = datetime.now(timezone.utc).isoformat()
    (RAW / 'download_manifest.json').write_text(json.dumps(log, indent=2) + '\n')
    if any(v['status'] != 'downloaded' for v in log.values()):
        raise RuntimeError('Catalog download incomplete; see download_manifest.json')


def studies():
    d = normalize(pd.read_csv(RAW / 'ewascatalog-studies.txt.gz', sep='\t', dtype=str,
                              keep_default_na=False))
    for c in ['study_id', 'trait', 'methylation_array', 'tissue', 'n']:
        if c not in d:
            raise ValueError(f'missing metadata column: {c}')
    if d.study_id.duplicated().any():
        raise ValueError('StudyID must uniquely identify metadata rows')
    d['trait_key'] = d.trait.str.strip().str.casefold().str.replace(r'\s+', ' ', regex=True)
    d['array_compatible'] = d.methylation_array.str.contains(r'450\s*k|450000|methylation450|epic|850\s*k|850000', case=False, regex=True)
    return d


def inventory():
    OUT.mkdir(parents=True, exist_ok=True)
    d = studies()
    d.to_csv(OUT / 'all_studies_metadata.csv', index=False)
    traits = d.groupby('trait_key', sort=True).agg(
        trait=('trait', 'first'), n_studies=('study_id', 'size'),
        n_array_compatible=('array_compatible', 'sum')).reset_index()
    traits['family'] = ''
    traits['classification_note'] = ''
    path = OUT / 'trait_categories.csv'
    if path.exists():
        raise FileExistsError(f'will not overwrite reviewed assignments: {path}')
    traits.to_csv(path, index=False)
    print(f'Review all {len(traits)} traits; assign a family or out_of_scope, with a note.')


def build():
    d = studies()
    categories = pd.read_csv(OUT / 'trait_categories.csv', keep_default_na=False)
    if categories.trait_key.duplicated().any():
        raise ValueError('duplicate trait assignments')
    if set(d.trait_key) != set(categories.trait_key):
        raise ValueError('trait assignments must cover exactly every Catalog trait')
    if not categories.family.isin(FAMILIES + ['out_of_scope']).all():
        raise ValueError('classify every trait before building; allowed: five families or out_of_scope')
    d = d.drop(columns=['family', 'classification_note'], errors='ignore').merge(
        categories[['trait_key', 'family', 'classification_note']], on='trait_key', validate='many_to_one')
    u = load_universe(ROOT)
    if len(u) != 408399:
        raise ValueError('unexpected universe size')
    canonical = encode_many(u.chrom, u.pos)
    if len(np.unique(canonical)) != len(u):
        raise ValueError('Universe contains duplicate canonical loci')
    id_to_row = {int(c): i for i, c in enumerate(canonical)}
    probes = pd.read_parquet(ROOT / 'data/processed/ComputAgeBench/illumina_probe_grch38.parquet')
    if probes.probe_id.duplicated().any():
        raise ValueError('ambiguous probe mapping')
    mapped = encode_many(probes.chr, probes.pos)
    probe_map = dict(zip(probes.probe_id, mapped))
    candidates = d[d.array_compatible & d.family.isin(FAMILIES)].copy()
    ids = set(candidates.study_id)
    hits = {sid: set() for sid in ids}
    covered = {sid: set() for sid in ids}
    for chunk in pd.read_csv(RAW / 'ewascatalog-results.txt.gz', sep='\t', dtype=str,
                             chunksize=500000, keep_default_na=False):
        chunk = normalize(chunk)
        chunk = chunk[chunk.study_id.isin(ids)]
        for sid, group in chunk.groupby('study_id', sort=False):
            covered[sid].update(int(probe_map[p]) for p in group.cpg.unique()
                                if p in probe_map and int(probe_map[p]) in id_to_row)
            p = pd.to_numeric(group.p, errors='coerce')
            hits[sid].update(group.loc[p < P_THRESHOLD, 'cpg'])
    candidates['n_catalog_loci_in_universe'] = candidates.study_id.map(lambda s: len(covered[s]))
    if 'n_observed_universe_loci' in candidates:
        candidates['n_catalog_loci_in_universe'] = pd.to_numeric(candidates.n_observed_universe_loci)
    candidates['n_positive_raw'] = candidates.study_id.map(lambda s: len(hits[s]))
    if 'n_positive_raw_source' in candidates:
        expected = pd.to_numeric(candidates.n_positive_raw_source)
        if not (expected == candidates.n_positive_raw).all():
            raise ValueError('Compact results do not reproduce the source-file positive counts')
    candidates['universe_compatible'] = candidates.n_catalog_loci_in_universe > 0
    candidates['whole_blood'] = candidates.tissue.str.strip().str.casefold().isin(['whole blood', 'blood', 'peripheral whole blood'])
    textcols = [c for c in ['analysis', 'further_details', 'consortium'] if c in candidates]
    text = candidates[textcols].agg(' '.join, axis=1)
    candidates['meta_analysis'] = text.str.contains(r'meta[ -]?analys[ie]s', case=False, regex=True)
    if 'is_meta_analysis' in candidates:
        candidates['meta_analysis'] = candidates.is_meta_analysis.str.casefold().eq('true')
    candidates['n_subjects_numeric'] = pd.to_numeric(candidates.n.str.replace(',', '', regex=False), errors='coerce')
    candidates['n_subjects_parseable'] = candidates.n_subjects_numeric.notna()
    # Blood preference first; meta-analysis then largest reported N; lexical StudyID tie-break.
    order = ['trait_key', 'whole_blood', 'meta_analysis', 'n_subjects_numeric', 'study_id']
    eligible = candidates[candidates.universe_compatible].sort_values(
        order, ascending=[True, False, False, False, True], na_position='last')
    primary = eligible.drop_duplicates('trait_key').copy()
    primary['n_candidate_studies'] = primary.trait_key.map(candidates.groupby('trait_key').size())
    primary['n_universe_compatible_studies'] = primary.trait_key.map(eligible.groupby('trait_key').size())
    primary['n_missing_n_candidates'] = primary.trait_key.map(
        candidates.groupby('trait_key').n_subjects_parseable.apply(lambda s: int((~s).sum())))
    primary['primary_selection_scope'] = ('official Zenodo fallback only; full Catalog comparison pending'
                                         if 'catalog_study_id' in primary else 'downloaded Catalog snapshot')
    candidates['selected_primary'] = candidates.study_id.isin(primary.study_id)
    candidates.to_csv(OUT / 'all_candidate_studies.csv', index=False)
    d[~d.array_compatible & d.family.isin(FAMILIES)].to_csv(OUT / 'excluded_arrays.csv', index=False)
    missing = categories[categories.family.isin(FAMILIES) & ~categories.trait_key.isin(primary.trait_key)]
    missing.to_csv(OUT / 'traits_without_compatible_study.csv', index=False)
    sets, rows = {}, {}
    for _, s in primary.iterrows():
        sid = s.study_id
        sets[sid] = np.array(sorted({int(probe_map[p]) for p in hits[sid]
                                    if p in probe_map and int(probe_map[p]) in id_to_row}), dtype=np.int64)
        rows[sid] = np.array(sorted(id_to_row[int(c)] for c in sets[sid]), dtype=np.int64)
    excluded = np.zeros(len(u), bool)
    for r in rows.values():
        excluded[r] = True
    # Conservatively retain the existing benchmark's excluded positives as well.
    for path in sorted((ROOT / 'data/bio_annotations/known_sets').glob('*.npy')):
        excluded |= np.isin(canonical, np.load(path, allow_pickle=False))
    primary['n_positive_universe'] = primary.study_id.map(lambda s: len(sets[s]))
    primary['n_positive_probes_unmapped'] = primary.study_id.map(
        lambda s: sum(p not in probe_map for p in hits[s]))
    primary['n_positive_probes_outside_universe'] = primary.study_id.map(
        lambda s: sum(p in probe_map and int(probe_map[p]) not in id_to_row for p in hits[s]))
    primary['support_tier'] = primary.n_positive_universe.map(tier)
    # Matching feasibility is an exact capacity calculation; no background draw or fit is needed.
    pool = u.loc[~excluded & (u.probe_type > 0)].groupby(['chrom', 'probe_type']).size().to_dict()
    match, strata = {}, []
    for sid, r in rows.items():
        pos = u.iloc[r]
        known = pos[pos.probe_type > 0]
        matched = unmatched = 0
        for (chrom, probe_type), n_pos in known.groupby(['chrom', 'probe_type']).size().items():
            capacity = int(pool.get((chrom, probe_type), 0))
            take = min(int(n_pos), capacity)
            matched += take
            unmatched += int(n_pos) - take
            strata.append(dict(study_id=sid, chrom=chrom, probe_type=probe_type,
                               n_positive=int(n_pos), n_available_background=capacity,
                               n_matchable=take, n_unmatched=int(n_pos)-take))
        match[sid] = dict(n_matched_positives=matched, n_dropped_unknown_probe_type=len(pos)-len(known),
                          n_unmatched=unmatched)
    pd.DataFrame(strata).to_csv(OUT / 'matching_strata.csv', index=False)
    for column in ['n_matched_positives', 'n_dropped_unknown_probe_type', 'n_unmatched']:
        primary[column] = primary.study_id.map(lambda s: match[s][column])
    primary['matched_support_tier'] = primary.n_matched_positives.map(tier)
    primary['matching_feasibility'] = primary.apply(lambda s: 'full' if s.n_matched_positives == s.n_positive_universe
                                                  else 'partial' if s.n_matched_positives else 'none', axis=1)
    overlaps = []
    lookup = primary.set_index('study_id')
    for a, b in itertools.combinations(sorted(sets), 2):
        shared = len(np.intersect1d(sets[a], sets[b]))
        na, nb = len(sets[a]), len(sets[b])
        containment = shared / min(na, nb) if min(na, nb) else 0.
        overlaps.append(dict(study_a=a, trait_a=lookup.loc[a, 'trait'], study_b=b,
                             trait_b=lookup.loc[b, 'trait'], n_a=na, n_b=nb, shared=shared,
                             smaller_set_containment=containment,
                             jaccard=shared / (na + nb - shared) if na + nb - shared else 0.,
                             flagged_gt_50_percent=containment > .5))
    ov = pd.DataFrame(overlaps, columns=['study_a', 'trait_a', 'study_b', 'trait_b', 'n_a', 'n_b', 'shared',
                                       'smaller_set_containment', 'jaccard', 'flagged_gt_50_percent'])
    ov.to_csv(OUT / 'overlap_all_pairs.csv', index=False)
    ov[ov.flagged_gt_50_percent].to_csv(OUT / 'overlap_flags.csv', index=False)
    primary['overlap_flag_partners'] = primary.study_id.map(lambda sid: '; '.join(
        r.trait_b if r.study_a == sid else r.trait_a for r in ov.itertuples()
        if r.flagged_gt_50_percent and sid in [r.study_a, r.study_b]))
    primary['n_overlap_flag_partners'] = primary.study_id.map(lambda sid: int(
        ((ov.study_a == sid) | (ov.study_b == sid))[ov.flagged_gt_50_percent].sum()))
    (OUT / 'positive_sets').mkdir(exist_ok=True)
    for sid, values in sets.items():
        safe = re.sub(r'[^A-Za-z0-9_.-]', '_', sid)
        np.save(OUT / 'positive_sets' / f'{safe}.npy', values)
    primary.sort_values(['family', 'trait_key']).to_csv(OUT / 'primary_traits.csv', index=False)
    report = ['# EWAS Catalog expansion — design only', '',
              'Positive: p < 1e-4; unique probes raw, unique GRCh38 loci after intersection.',
              'Support tiers use intersected positives; matched support is reported separately.', '']
    if 'catalog_study_id' in primary and primary.catalog_study_id.eq('').all():
        report += ['Source: official EWAS Catalog Zenodo community fallback. Coverage of the full Catalog is incomplete.',
                   'Catalog StudyID is unavailable in these deposits. `source_analysis_id` uses Zenodo record ID and original results filename, not an invented Catalog StudyID.',
                   'Incident/prevalent endpoints and developmental stages are separate traits. All selected studies, counts, provenance and pair flags are in the companion CSVs.', '']
    for family in FAMILIES:
        part = primary[primary.family == family].sort_values('trait_key')
        report += [f'## {family}', '', '| Trait | Source analysis | StudyID / PMID | Tissue | N | Array | Raw | Universe | Tier | Matchable | Matching | Flagged partners |',
                   '|---|---|---|---|---|---|---:|---:|---|---:|---|---:|']
        for s in part.itertuples():
            public_id = getattr(s, 'catalog_study_id', s.study_id) or 'Catalog StudyID unavailable'
            values = [s.trait, s.study_id, f'{public_id} / {getattr(s, "pmid", "") or "unavailable"}', s.tissue or 'unavailable', s.n or 'unavailable',
                      s.methylation_array, s.n_positive_raw, s.n_positive_universe, s.support_tier,
                      s.n_matched_positives, s.matching_feasibility, s.n_overlap_flag_partners]
            report.append('| ' + ' | '.join(str(v).replace('|', '/') for v in values) + ' |')
        report.append('')
    (OUT / 'report.md').write_text('\n'.join(report) + '\n')
    inputs = [RAW / 'ewascatalog-studies.txt.gz', RAW / 'ewascatalog-results.txt.gz',
              OUT / 'trait_categories.csv', ROOT / 'outputs/encode_atlas_v1/loci.parquet',
              ROOT / 'data/processed/ComputAgeBench/illumina_probe_grch38.parquet',
              ROOT / 'data/derived/bioval_v2/covariates/universe_covariates.parquet',
              Path(__file__).resolve(),
              *sorted((ROOT / 'data/bio_annotations/known_sets').glob('*.npy'))]
    if (RAW / 'preparation_manifest.json').exists():
        inputs.append(RAW / 'preparation_manifest.json')
    (OUT / 'manifest.json').write_text(json.dumps(dict(
        state='DESIGN_ONLY', n_universe=len(u), p_threshold=P_THRESHOLD,
        selection_order=['whole_blood DESC', 'meta_analysis DESC', 'N DESC, missing last', 'StudyID ASC'],
        matching='Exact capacity for MAIN 1:1 chromosome x Infinium I/II; exclude primary union and existing16 known sets; no backgrounds sampled',
        inputs={str(p.relative_to(ROOT)): digest(p) for p in inputs},
        outputs={str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*'))
                 if p.is_file() and p.name != 'manifest.json'}), indent=2) + '\n')
    print(primary.groupby(['family', 'support_tier']).size().to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['download', 'inventory', 'build'])
    parser.add_argument('--raw-dir', type=Path, default=RAW)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    RAW, OUT = args.raw_dir.resolve(), args.output_dir.resolve()
    action = args.action
    globals()[action]()
