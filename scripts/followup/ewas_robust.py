"""CLI: robust matched-negative EWAS + clock-set membership probe (follow-up Exp 5 / 6).

Outputs (default outputs/encode_atlas_v1/followup/exp5/):
  oof_predictions.parquet        long pooled-OOF table (variant,set,match_seed,cpg_idx,chromosome,pair_id,y_true,fold,
                                 representation,oof_score); clock sets are ordinary rows of ``set``
  summary_auc_delta.csv          per set/variant AUC [CI] and paired delta [CI], p, BH q (family = ewas | clock)
  matching_summary_per_seed.csv, evaluable_sets.csv, probe_type_balance.csv, probe_type_annotation.parquet, protocol.json
Clock reuse: filter summary_auc_delta.csv on family == 'clock' (sets horvath/hannum/phenoage_clock).
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import yaml  # noqa: E402

from cpg_repr_benchmark.encode_atlas.ewas_robust import run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/encode_atlas.yaml')
p.add_argument('--out', default='outputs/encode_atlas_v1/followup/exp5')
p.add_argument('--seeds', type=int, default=20)
p.add_argument('--boot', type=int, default=10000)
p.add_argument('--workers', type=int, default=32)
p.add_argument('--manifest', default='/data2/fciapi/Giunti/Brain_Tumors/EPICv2.hg38.manifest.tsv')
p.add_argument('--only', nargs='*')
a = p.parse_args()
cfg = yaml.safe_load(open(a.config))
run(cfg, Path(a.out), n_seeds=a.seeds, boot=a.boot, workers=a.workers, manifest=a.manifest, only_sets=a.only)
