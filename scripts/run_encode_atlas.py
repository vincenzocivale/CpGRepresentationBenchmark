#!/usr/bin/env python3
"""Prepare, screen, run and report a frozen, resumable ENCODE attribution campaign."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from cpg_repr_benchmark.encode_atlas.campaign import execute, load_campaign, prepare, screen, select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/encode_atlas.yaml'))
    parser.add_argument('command', choices=['prepare', 'screen', 'augment', 'select', 'run', 'report', 'biology',
                                           'pipeline', 'tissue', 'ewas'])
    parser.add_argument('--stage', choices=['discovery', 'confirm'], default='discovery')
    parser.add_argument('--only', nargs='+', help='Exact arm/group names from arms.json/groups.json')
    parser.add_argument('--max-jobs', type=int)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--ood', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error('--max-jobs must be positive')
    cfg = load_campaign(args.config)
    output = Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'prepare':
            result = prepare(cfg)
        elif args.command == 'screen':
            result = screen(cfg, max_groups=args.max_jobs)
        elif args.command == 'select':
            result = select(cfg)
        elif args.command == 'augment':
            from cpg_repr_benchmark.encode_atlas.extensions import augment
            result = augment(cfg)
        elif args.command == 'tissue':
            from cpg_repr_benchmark.encode_atlas.tissue import tissue_screen
            result = tissue_screen(cfg)
        elif args.command == 'ewas':
            from cpg_repr_benchmark.encode_atlas.ewas import evaluate_sets
            result = evaluate_sets(cfg, only=args.only)
        elif args.command == 'pipeline':
            from cpg_repr_benchmark.encode_atlas.pipeline import pipeline
            result = pipeline(cfg)
        elif args.command == 'report':
            from cpg_repr_benchmark.encode_atlas.reporting import report
            result = report(cfg)
        elif args.command == 'biology':
            from cpg_repr_benchmark.encode_atlas.reporting import biological_neighbors
            result = biological_neighbors(cfg, only=args.only, stage=args.stage)
        else:
            result = execute(cfg, stage=args.stage, only=args.only, max_jobs=args.max_jobs,
                             pilot=args.pilot, ood=args.ood, dry_run=args.dry_run)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
