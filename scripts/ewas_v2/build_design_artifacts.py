#!/usr/bin/env python
# ruff: noqa: C408
"""Build the EWAS-v2 design artifacts (NO embeddings, NO betas, NO probes): data/derived/ewas_v2_v1/ + configs/ewas_v2/manifest_v1.json.

`--check` rebuilds into a scratch dir and verifies the hashes equal those of the existing artifacts (determinism)."""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import BlockBootstrap
from cpg_repr_benchmark.ewas_v2 import design as D


def build(root: Path, out_dir: Path) -> dict:
    des = D.build_design(root)
    hashes = D.write_design(des, out_dir)
    inputs = {D.LOCI: D.sha256_file(root / D.LOCI), D.COVARIATES: D.sha256_file(root / D.COVARIATES), D.PINS: D.sha256_file(root / D.PINS),
              D.EXP5_TIERS: D.sha256_file(root / D.EXP5_TIERS)}
    for t in D.TASKS:
        rel = f"{D.KNOWN_SETS_DIR}/{t}.npy"
        inputs[rel] = D.sha256_file(root / rel)
    s = des["summary"].set_index("task")
    pm, ps = des["pairs_main"], des["pairs_s1"]
    tasks = {}
    for t in D.TASKS:
        r = s.loc[t].to_dict()
        r = {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
        for name, df in (("main", pm), ("s1", ps)):
            d = df[df.task == t]
            r[f"{name}_pairs_by_chrom"] = {c: int(n) for c, n in d.groupby("chrom").size().items()}
            r[f"{name}_pairs_by_probe_type"] = {str(k): int(n) for k, n in d.groupby("probe_type").size().items()}
            r[f"{name}_pairs_by_fold"] = {str(k): int(n) for k, n in d.groupby("fold").size().items()}
        tasks[t] = r
    tiers = {k: sorted(s.index[s["tier"] == k]) for k in ("main-quality", "exploratory", "insufficient")}
    tiers_s1 = {k: sorted(s.index[s["s1_tier"] == k]) for k in ("main-quality", "exploratory", "insufficient")}
    bb = BlockBootstrap(np.array(des["blocks"]), D.N_BOOT, D.SEED)
    return dict(
        manifest_version="ewas-v2-design-v1", design_dir=D.DESIGN_DIR, n_universe=des["n_universe"],
        n_background_excluded_union=des["n_excluded_union"], seed=D.SEED, n_boot=D.N_BOOT, n_folds=D.N_FOLDS,
        chrom_to_fold=des["chrom_to_fold"], bootstrap_fingerprint=bb.fingerprint(), s1_caliper=D.S1_CALIPER,
        s1_standardization=des["s1_standardization"], tiers_main=tiers, tiers_s1=tiers_s1,
        non_independent=des["non_independent"], overlap_rows=des["overlap"].to_dict("records"),
        inputs_sha256=inputs, artifacts_sha256=hashes, tasks=tasks)


def _dump(m) -> str:
    return json.dumps(m, indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    root = ROOT
    if a.check:
        with tempfile.TemporaryDirectory(dir=str(root / "data/derived")) as td:
            m = build(root, Path(td) / "chk")
        same = _dump(m) + "\n" == (root / D.MANIFEST).read_text()
        print("rebuild identical to tracked manifest:", same)
        return 0 if same else 1
    m = build(root, root / D.DESIGN_DIR)
    (root / D.MANIFEST).write_text(_dump(m) + "\n")
    print("manifest sha256", hashlib.sha256((root / D.MANIFEST).read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
