#!/usr/bin/env python3
"""Experiment B (SECONDARY, exploratory): TCGA -> GSE40279 transfer of the frozen phase-A best.pt checkpoints (no training).
Protocol v1.2 / AMENDMENT 2: default scope = the 3 MAIN arms x 3 seeds = 9 checkpoints, VALIDATION split; functional_annotations_pca
(legacy_sensitivity_control) is excluded unless --include-legacy (labelled arm_role legacy_sensitivity_control; not planned).

  run_external_transfer.py --mode B_strict|B_recalibrated|both [--split validation|test] [--only ARM...] [--seeds S...] [--dry-run]
                           [--device auto|cpu|cuda] [--include-legacy]

B_strict        original TCGA prior (prior_logit.npy of the checkpoint's own TCGA run), nothing external anywhere.
B_recalibrated  frozen decoder/adapter, prior recomputed from the 524 external TRAIN subjects only.
Outputs: outputs/external_reconstruction_v1/transfer/<mode>/<arm>/seed_<seed>/evaluation/<split>/... (modes never mixed).
The gate (scope B) must be green: freeze verify, hashes, coverage, git tag/commit, implementation committed clean, test locked.
`--split test` is refused unless the manifest is final + test_set_authorized + all-green. Not executed on real data by tooling.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cpg_repr_benchmark.external import transfer as T


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=(*T.MODES, "both"), required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--only", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--include-legacy", action="store_true", help="also transfer the legacy functional checkpoints (labelled; not planned)")
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    os.nice(10)
    rc = 0
    for mode in (T.MODES if a.mode == "both" else (a.mode,)):
        rc = max(rc, T.run_transfer(ROOT, mode, split=a.split, seeds=a.seeds, only=a.only, dry_run=a.dry_run, device=a.device,
                                    include_legacy=a.include_legacy))
    return rc


if __name__ == "__main__":
    sys.exit(main())
