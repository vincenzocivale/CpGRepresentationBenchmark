from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from cpg_repr_benchmark.data.methylation import read_axis
from cpg_repr_benchmark.encode_atlas.biology import co_methylation_neighbors
from cpg_repr_benchmark.encode_atlas.campaign import arm_id, check_prepared
from cpg_repr_benchmark.encode_atlas.features import write_json
from cpg_repr_benchmark.encode_atlas.protocol import load_patient_protocol
from cpg_repr_benchmark.encode_atlas.statistics import bh_adjust, paired_bootstrap, prediction_table


def report(cfg):
    out = check_prepared(cfg)
    registry = pd.read_parquet(out / 'loci.parquet')
    _, names = read_axis(Path(cfg['methylation']))
    jobs = [json.loads(p.read_text()) for p in (out / 'jobs').glob('*.json')]
    completed = [j for j in jobs if j['status'] == 'complete']
    rows, comparisons, strata = [], [], []
    with h5py.File(cfg['features']) as h:
        context = np.argmax(h['dense'][:, :4], axis=1)
        region = np.argmax(h['dense'][:, 4:8], axis=1)
    strata_features = {'context': context, 'gene_region': region}
    strata_metadata = {'context': 'input-derived CpG-island context',
                       'gene_region': 'input-derived gene-region label'}
    with np.load(out / 'discovery_targets.npz') as h:
        sampled_rows = h['rows']
        min_count = max(20, int(len(h['patients']) * .5))
        moments = {'train_mean_beta': h['mean'], 'train_variance_beta': h['variance'],
                   'train_missing_fraction': 1 - h['count'] / len(h['patients'])}
        eligible = h['count'] >= min_count
    for name, values in moments.items():
        finite = eligible & np.isfinite(values)
        if not finite.any():
            continue
        edges = np.quantile(values[finite], [.25, .5, .75])
        labels = np.full(len(registry), -1, dtype=np.int8)
        labels[sampled_rows[finite]] = np.searchsorted(edges, values[finite], side='right')
        strata_features[name] = labels
        strata_metadata[name] = {'source': 'discovery training patients only, chr1-19 screen loci',
                                 'quantile_edges': edges.tolist(), 'n_eligible_loci': int(finite.sum())}
    write_json(out / 'strata_definitions.json', strata_metadata)
    references = {(j['stage'], j['seed'], j['ood'], j['pilot']): j for j in completed if j['arm'] == 'full'}
    for job in completed:
        summary = json.loads((Path(job['run_dir']) / 'summary.json').read_text())
        for view, fractions in summary['evaluation'].items():
            for fraction, metrics in fractions.items():
                rows.append({**{k: job[k] for k in ['arm', 'seed', 'stage', 'ood', 'pilot']},
                             'view': view, 'mask': fraction, **metrics})
        reference = references.get((job['stage'], job['seed'], job['ood'], job['pilot']))
        if reference is None or reference == job:
            continue
        for view in summary['evaluation']:
            path = Path(job['run_dir']) / 'evaluation' / view / 'mask_0.50' / 'predictions.npz'
            ref_path = Path(reference['run_dir']) / 'evaluation' / view / 'mask_0.50' / 'predictions.npz'
            if not path.exists() or not ref_path.exists():
                continue
            with np.load(path) as h:
                alternative = prediction_table(h, registry, names)
            with np.load(ref_path) as h:
                baseline = prediction_table(h, registry, names)
            result = paired_bootstrap(baseline, alternative, seed=cfg['mask_seed'],
                                       replicates=cfg['bootstrap_replicates'])
            spec_path = out / ('confirmation_arms.json' if job['stage'] == 'confirm' else 'arms.json')
            specs = json.loads(spec_path.read_text())
            result.update(arm=job['arm'], family=specs[job['arm']]['family'], seed=job['seed'],
                          stage=job['stage'], ood=job['ood'], pilot=job['pilot'], view=view)
            comparisons.append(result)
            pairs = baseline.merge(alternative, on=['patient', 'column', 'block'], suffixes=('_ref', '_alt'))
            pairs['gain'] = pairs.squared_error_ref - pairs.squared_error_alt
            for label, values in strata_features.items():
                pairs['stratum'] = values[pairs.column.to_numpy()]
                for value, subset in pairs.loc[pairs.stratum.ge(0)].groupby('stratum'):
                    strata.append({'arm': job['arm'], 'seed': job['seed'], 'stage': job['stage'],
                                   'ood': job['ood'], 'pilot': job['pilot'], 'view': view,
                                   'axis': label, 'stratum': int(value), 'n_pairs': len(subset),
                                   'n_loci': int(subset.column.nunique()), 'mean_gain': float(subset.gain.mean())})
    pd.DataFrame(rows).to_csv(out / 'reconstruction_metrics.csv', index=False)
    comp = pd.DataFrame(comparisons)
    if len(comp):
        comp['q'] = comp.groupby(['stage', 'ood', 'pilot', 'view', 'family', 'seed'])['p'].transform(bh_adjust)
        # No pooling of seed-specific significance: stability is reported explicitly.
        comp.to_csv(out / 'paired_comparisons.csv', index=False)
        pd.DataFrame(strata).to_csv(out / 'context_gains.csv', index=False)
        confirmations = comp[(comp.stage == 'confirm') & ~comp.pilot & ~comp.ood & (comp.view == 'seen')]
        panels = []
        for name, subset in confirmations.groupby('arm'):
            if not name.startswith('panel/selected/'):
                continue
            all_seeds = set(subset.seed) == set(cfg['seeds'])
            equivalent = all_seeds and all(ci[1] < .01 for ci in subset.relative_ci95)
            panels.append({'arm': name, 'n_tracks': int(name.rsplit('/', 1)[1]),
                           'all_seeds_complete': all_seeds, 'equivalent': equivalent})
        passed = [p for p in panels if p['equivalent']]
        write_json(out / 'compact_panel_decision.json', {
            'rule': 'upper 95% relative MSE degradation bound < 1% for every prespecified seed',
            'smallest_equivalent': min(passed, key=lambda p: p['n_tracks']) if passed else None,
            'status': 'evaluated' if panels and all(p['all_seeds_complete'] for p in panels) else 'pending',
            'panels': panels})
        # Descriptive difference-in-differences, reported separately from bootstrap tests.
        interaction_rows = []
        metrics_frame = pd.DataFrame(rows)
        for key, subset in metrics_frame.groupby(['stage', 'seed', 'ood', 'pilot', 'view', 'mask']):
            mse = dict(zip(subset.arm, subset.mse))
            baseline_mse = mse.get('context_only')
            if baseline_mse is None:
                continue
            for name, pair_mse in mse.items():
                if not name.startswith('pair/'):
                    continue
                left, right = name[5:].split('+', 1)
                # Macro pairs name assays directly; selected pairs include their group prefix.
                left = left if '/' in left else 'assay/' + left
                right = right if '/' in right else 'assay/' + right
                if 'add/' + left in mse and 'add/' + right in mse:
                    interaction_rows.append({'comparison': name, 'stage': key[0], 'seed': key[1],
                                             'ood': key[2], 'pilot': key[3], 'view': key[4], 'mask': key[5],
                                             'excess_combined_gain': mse['add/' + left] + mse['add/' + right]
                                             - pair_mse - baseline_mse})
        pd.DataFrame(interaction_rows).to_csv(out / 'descriptive_interactions.csv', index=False)
    screens = []
    for path in (out / 'screen').glob('*.json'):
        record = json.loads(path.read_text())
        for endpoint, variants in record['endpoints'].items():
            for variant, metrics in variants.items():
                screens.append({'group': record['group'], 'family': record['family'], 'endpoint': endpoint,
                                'variant': variant, **{k: v for k, v in metrics.items() if k != 'folds'}})
    pd.DataFrame(screens).to_csv(out / 'biological_probes.csv', index=False)
    text = ['# ENCODE attribution campaign', '',
            f'Completed decoder jobs: {len(completed)}. Biological probe rows: {len(screens)}.', '',
            'Discovery results use validation patients; confirmation uses the frozen test split.',
            'Input context labels describe localization of benefit, not independent validation.',
            'Intervals resample patients and 1 Mb genomic blocks. Seeds are reported separately.',
            'Pilot jobs are implementation/cost diagnostics and must not enter paper comparisons.', '',
            '## Artifacts', '',
            '- `feature_inventory.csv`: original track identities, support and metadata.',
            '- `track_associations.csv`: chromosome-cross-fitted context-adjusted discovery associations.',
            '- `biological_probes.csv`: nested chromosome CV; alone and with genomic context.',
            '- `reconstruction_metrics.csv`: all completed runs, including negative results.',
            '- `paired_comparisons.csv`: paired effects, crossed bootstrap intervals and family-wise BH q-values.',
            '- `context_gains.csv`: descriptive localization of reconstruction effects.', '',
            '## Interpretation limits', '',
            'No perturbational causal claims. ENCODE absence is absence of a recorded peak, not proof of inactivity.',
            'Tissue-matched claims require a separately validated sample-to-tissue crosswalk.',
            'EWAS non-members are not established negative associations.',
            'Co-methylation diagnostics are descriptive and use matched distant loci.', '']
    (out / 'REPORT.md').write_text('\n'.join(text))
    if len(comp):
        try:
            render_effects(comp, out)
        except ImportError:
            text.append('Figure rendering unavailable: install the viz extra.')
            (out / 'REPORT.md').write_text('\n'.join(text))
    return {'completed_jobs': len(completed), 'probe_rows': len(screens), 'report': str(out / 'REPORT.md')}


def focused_contrasts(cfg):
    """Direct paper contrasts omitted by the generic full-versus-arm report."""
    out = check_prepared(cfg)
    registry = pd.read_parquet(out / 'loci.parquet')
    _, names = read_axis(Path(cfg['methylation']))
    jobs = [json.loads(p.read_text()) for p in (out / 'jobs').glob('*.json')]
    complete = {(j['seed'], j['ood'], j['arm']): j for j in jobs
                if j['status'] == 'complete' and j['stage'] == 'confirm' and not j['pilot']}
    definitions = [
        ('histone_sufficiency', 'context_only', 'add/assay/Histone ChIP-seq'),
        ('matched_assay_identity', 'control/non_histone_prevalence_matched',
         'control/histone_prevalence_matched'),
        ('h3k4me3_increment', 'context_only', 'add/target/H3K4me3'),
    ]
    rows = []
    for seed in cfg['seeds']:
        for label, baseline_name, alternative_name in definitions:
            baseline_job = complete.get((seed, False, baseline_name))
            alternative_job = complete.get((seed, False, alternative_name))
            if baseline_job is None or alternative_job is None:
                continue
            def predictions(job):
                path = Path(job['run_dir']) / 'evaluation/seen/mask_0.50/predictions.npz'
                with np.load(path) as h:
                    return prediction_table(h, registry, names)
            baseline = predictions(baseline_job)
            alternative = predictions(alternative_job)
            effect = paired_bootstrap(baseline, alternative, seed=cfg['mask_seed'],
                                      replicates=cfg['bootstrap_replicates'])
            rows.append({'contrast': label, 'baseline': baseline_name,
                         'alternative': alternative_name, 'seed': seed, **effect})
    frame = pd.DataFrame(rows)
    if len(frame):
        frame['q'] = frame.groupby('seed')['p'].transform(bh_adjust)
    frame.to_csv(out / 'focused_contrasts.csv', index=False)
    metrics = pd.read_csv(out / 'reconstruction_metrics.csv')
    metrics = metrics.loc[(metrics.stage == 'confirm') & (~metrics.ood) & (~metrics.pilot)
                          & (metrics.view == 'seen') & (metrics['mask'] == 'mask_0.50')]
    lines = ['# Focused ENCODE confirmation', '',
             'Primary endpoint: 50% masked reconstruction MSE on frozen test patients.',
             'All comparisons use identical evaluation panels; lower MSE is better.',
             'Intervals cross-resample patient identities and 1 Mb genomic blocks.',
             'Optimization seeds are stability checks, not biological replicates.', '',
             '## Test-set MSE', '', '| Arm | Seed | MSE |', '| --- | ---: | ---: |']
    for row in metrics.sort_values(['arm', 'seed']).itertuples():
        lines.append(f'| `{row.arm}` | {row.seed} | {row.mse:.6f} |')
    lines += ['', '## Direct paired contrasts', '',
              'Delta is alternative minus baseline; negative favors the alternative.', '',
              '| Contrast | Seed | Relative delta | 95% interval | q |',
              '| --- | ---: | ---: | --- | ---: |']
    for row in frame.itertuples():
        lo, hi = row.relative_ci95
        lines.append(f'| {row.contrast} | {row.seed} | {row.relative_delta_mse:+.2%} | '
                     f'[{lo:+.2%}, {hi:+.2%}] | {row.q:.3g} |')
    lines += ['', 'Other full-versus-arm effects are in `paired_comparisons.csv`.',
              'Per-locus effect strata are in `context_gains.csv`; chromosome-transfer '
              'results are in `reconstruction_metrics.csv`.',
              'Interpret matched assay controls only on their shared prevalence support; '
              'unmatched high-prevalence histone tracks are excluded.',
              'EWAS set membership is exploratory: unlisted loci are not confirmed negatives.', '']
    (out / 'FOCUSED_REPORT.md').write_text('\n'.join(lines))
    return {'contrasts': len(frame), 'report': str(out / 'FOCUSED_REPORT.md')}


def render_effects(frame, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for keys, subset in frame.groupby(['stage', 'ood', 'pilot', 'view', 'seed']):
        subset = subset.sort_values('delta_mse')
        fig, ax = plt.subplots(figsize=(9, max(3, len(subset) * .22)))
        for i, row in enumerate(subset.itertuples()):
            ax.plot(row.ci95, [i, i], color='steelblue')
            ax.plot(row.delta_mse, i, 'o', color='steelblue', markersize=3)
        ax.set_yticks(range(len(subset)), subset.arm)
        ax.axvline(0, color='black', linewidth=.7)
        ax.set_xlabel('MSE alternative − full ENCODE (positive = worse)')
        ax.set_title(' / '.join(map(str, keys)))
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(out / ('effects_' + '_'.join(map(str, keys)) + '.' + extension), dpi=180)
        plt.close(fig)


def biological_neighbors(cfg, *, only=None, stage='discovery'):
    out = check_prepared(cfg)
    registry = pd.read_parquet(out / 'loci.parquet')
    with np.load(out / 'discovery_targets.npz') as h:
        rows = h['rows']
    _, names = read_axis(Path(cfg['methylation']))
    patients = load_patient_protocol(out / 'patients.npz', names)['test' if stage == 'confirm' else 'validation']
    with h5py.File(cfg['methylation']) as h:
        beta = np.stack([h['beta'][int(i), :][rows] for i in patients])
    with h5py.File(cfg['features']) as h:
        context = np.argmax(h['dense'][rows, :4], axis=1)
    specs = json.loads((out / ('confirmation_arms.json' if stage == 'confirm' else 'arms.json')).read_text())
    count = 0
    for name in only or specs:
        path = out / 'embeddings' / (arm_id(name) + '.h5')
        if not path.exists():
            continue
        with h5py.File(path) as h:
            embedding = h['embedding'][rows].astype(np.float32)
        result = co_methylation_neighbors(embedding, beta, registry.chr.to_numpy()[rows],
                                          registry.pos.to_numpy()[rows], context, seed=cfg['mask_seed'])
        dest = out / 'co_methylation' / stage / (arm_id(name) + '.csv')
        dest.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(dest, index=False)
        write_json(dest.with_suffix('.json'), {'arm': name, 'stage': stage, 'patients': patients.tolist(),
                                              'n_pairs': len(result), 'independence': 'held-out patients'})
        count += 1
    return {'completed_neighbor_diagnostics': count}
