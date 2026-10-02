from __future__ import annotations

import time
from pathlib import Path

from cpg_repr_benchmark.encode_atlas.campaign import execute, prepare, screen, select
from cpg_repr_benchmark.encode_atlas.ewas import evaluate_sets
from cpg_repr_benchmark.encode_atlas.extensions import augment
from cpg_repr_benchmark.encode_atlas.features import write_json
from cpg_repr_benchmark.encode_atlas.focused import focused_selection
from cpg_repr_benchmark.encode_atlas.reporting import biological_neighbors, focused_contrasts, report
from cpg_repr_benchmark.encode_atlas.tissue import tissue_screen


def exhaustive_pipeline(cfg):
    """Historical all-arm plan; retained for reproducibility, not the default."""
    out = Path(cfg['output'])
    if not (out / 'campaign.json').exists():
        prepare(cfg)
    stages = [
        ('screen', lambda: screen(cfg)),
        ('augment', lambda: augment(cfg)),
        ('tissue', lambda: tissue_screen(cfg)),
        ('pilot', lambda: execute(cfg, only=['full'], pilot=True, max_jobs=1)),
        ('discovery', lambda: execute(cfg)),
        ('selection', lambda: select(cfg)),
        ('confirmation', lambda: execute(cfg, stage='confirm')),
        ('confirmation_ood', lambda: execute(cfg, stage='confirm', ood=True)),
        ('co_methylation', lambda: biological_neighbors(cfg, stage='confirm')),
        ('ewas', lambda: evaluate_sets(cfg)),
        ('report', lambda: report(cfg)),
    ]
    for name, operation in stages:
        marker = out / 'pipeline' / (name + '.json')
        if marker.exists():
            continue
        print(f'pipeline stage: {name}', flush=True)
        while True:
            result = operation()
            if result.get('status') != 'waiting_for_gpu':
                break
            write_json(out / 'pipeline' / 'waiting.json', {'stage': name, **result})
            print('Waiting 60 seconds for GPU memory.', flush=True)
            time.sleep(60)
        write_json(marker, {'status': 'complete', 'result': result})
    return {'status': 'complete', 'report': str(out / 'REPORT.md')}


def pipeline(cfg):
    """Serial, restartable paper-focused campaign with a frozen test-set plan."""
    out = Path(cfg['output'])
    if not (out / 'campaign.json').exists():
        prepare(cfg)
    prerequisites = [
        ('screen', lambda: screen(cfg)),
        ('augment', lambda: augment(cfg)),
        ('tissue', lambda: tissue_screen(cfg)),
    ]
    for name, operation in prerequisites:
        marker = out / 'pipeline' / (name + '.json')
        if not marker.exists():
            print(f'pipeline prerequisite: {name}', flush=True)
            write_json(marker, {'status': 'complete', 'result': operation()})

    plan = focused_selection(cfg)
    primary = plan['primary_arms']
    ood = plan['ood_arms']
    biology = plan['biology_arms']
    stages = [
        ('discovery', lambda: execute(cfg, only=primary)),
        ('confirmation', lambda: execute(cfg, stage='confirm', only=primary)),
        ('chromosome_transfer', lambda: execute(cfg, stage='confirm', only=ood, ood=True)),
        ('co_methylation', lambda: biological_neighbors(cfg, only=biology, stage='confirm')),
        ('ewas', lambda: evaluate_sets(cfg, only=biology)),
        ('report', lambda: report(cfg)),
        ('focused_contrasts', lambda: focused_contrasts(cfg)),
    ]
    for name, operation in stages:
        marker = out / 'pipeline_focused' / (name + '.json')
        if marker.exists():
            continue
        print(f'focused pipeline stage: {name}', flush=True)
        while True:
            result = operation()
            if result.get('status') != 'waiting_for_gpu':
                break
            write_json(out / 'pipeline_focused' / 'waiting.json', {'stage': name, **result})
            print('Waiting 60 seconds for GPU memory.', flush=True)
            time.sleep(60)
        write_json(marker, {'status': 'complete', 'result': result})
    return {'status': 'complete', 'plan': str(out / 'focused_plan.json'),
            'report': str(out / 'FOCUSED_REPORT.md')}
