#!/usr/bin/env python3
"""Runner of the frozen confirmation matrix (docs/REGULATORY_CONFIRMATION_PROTOCOL.md AMENDMENT 2; spec: matrix.yaml).

  generate   write configs/experiments/regulatory_confirmation_matrix/runs/<arm>__seed<seed>.yaml for every AVAILABLE arm
             (main arms x 3 seeds, plus sensitivity arms, marked role: sensitivity). Pending arms get no config.
  validate   no-training check: configs == frozen builder, equal to the campaign template except allowed keys, stores/cpg_idx
  run        train+evaluate sequentially (one GPU job at a time, seed-major, resumable via .done)
             [--only ARM...] [--seeds S...] [--include-sensitivity] [--dry-run] [--split validation|test]
             [--allow-incomplete-validation-only]

`run` refuses while the audit fails. The only escape hatch, --allow-incomplete-validation-only, tolerates ONLY the pending modern
sequence FM comparators (every other audit failure still blocks), launches only validation-split runs of already-available arms,
and never permits the test split. `--split test` is refused unless test_set_authorized is true AND freeze_state is complete AND
the audit is green; even then this scaffold does not execute test evaluation (see README: it needs a separate output subdirectory
so the validation results are not overwritten). This script never launches anything unless `run` is called without --dry-run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cpg_repr_benchmark.experiments import confirmation_matrix as cm


def _spec(args):
    return cm.load_matrix(args.matrix)


def generate(args) -> int:
    spec = _spec(args)
    out = ROOT / cm.CONFIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    js = cm.jobs(spec, include_sensitivity=True)
    for arm, seed in js:
        (out / cm.config_name(arm["id"], seed)).write_text(yaml.safe_dump(cm.build_config(spec, arm, seed, ROOT), sort_keys=False))
    pend = [a["id"] for a in spec["arms"] if a.get("status") == "pending"]
    print(f"wrote {len(js)} configs to {cm.CONFIG_DIR} (main: {len(cm.jobs(spec))}, sensitivity: {len(js) - len(cm.jobs(spec))}); "
          f"no config for pending arms {pend}")
    return 0


def validate(args) -> int:
    import h5py
    import numpy as np
    spec = _spec(args)
    template = yaml.safe_load((ROOT / spec["protocol"]["template"]).read_text())
    seen = np.load(ROOT / spec["protocol"]["loci_split_path"])
    train_idx = np.sort(seen["train_cpg_idx"])
    if len(seen["heldout_cpg_idx"]) != 0:
        raise ValueError("loci_seen.npz must have no held-out loci")
    with np.load(ROOT / spec["protocol"]["patient_split_path"]) as h:
        sizes = {k: len(h[k]) for k in ("train", "validation", "test")}  # sizes only; no test data is read
    report = {"patients": sizes, "n_loci": len(train_idx), "n_configs": 0, "arms": {}}
    for arm, seed in cm.jobs(spec, include_sensitivity=True):
        path = ROOT / cm.CONFIG_DIR / cm.config_name(arm["id"], seed)
        if not path.exists():
            raise FileNotFoundError(f"{path}; run `generate` first")
        cfg = yaml.safe_load(path.read_text())
        if cfg != cm.build_config(spec, arm, seed, ROOT):
            raise ValueError(f"{path} is stale; re-run generate")
        cm.check_config(cfg, template, spec, arm["id"], seed)
        report["n_configs"] += 1
        if arm["id"] in report["arms"]:
            continue
        with h5py.File(ROOT / arm["store_h5"], "r") as h:
            idx, emb = h["cpg_idx"][:], h["embedding"]
            info = {"role": arm["role"], "shape": list(emb.shape), "dtype": str(emb.dtype),
                    "covers_protocol": bool(np.isin(train_idx, idx).all())}
        if emb.shape[1] != arm["dim"] or str(emb.dtype) != arm["dtype"] or not info["covers_protocol"]:
            raise ValueError(f"{arm['id']}: store violates the contract: {info}")
        report["arms"][arm["id"]] = info
    print(json.dumps(report, indent=2))
    return 0


def run(args) -> int:
    from cpg_repr_benchmark.experiments.guards import enforce_patient_view
    spec = _spec(args)
    os.nice(10)
    results = cm.audit(spec, ROOT, check_hashes=True, check_configs=False)
    ok, why = cm.gate(spec, results, split=args.split, allow_incomplete_validation_only=args.allow_incomplete_validation_only)
    print(f"GATE ({args.split}): {'OPEN' if ok else 'REFUSED'}: {why}")
    for r in cm.failures(results):
        print("  " + r.line())
    if not ok:
        return 2
    if args.split == "test":
        print("REFUSED: test-split execution is intentionally not implemented in this scaffold (needs a separate evaluation output "
              "directory so validation results are never overwritten); implement it before any user authorization.")
        return 2
    template = yaml.safe_load((ROOT / spec["protocol"]["template"]).read_text())
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "PYTHONPATH": str(ROOT / "src")}
    logs = ROOT / cm.OUTPUT_ROOT / "logs"
    for arm, seed in cm.jobs(spec, seeds=args.seeds, include_sensitivity=args.include_sensitivity, only=args.only):
        path = ROOT / cm.CONFIG_DIR / cm.config_name(arm["id"], seed)
        cfg = yaml.safe_load(path.read_text())
        cm.check_config(cfg, template, spec, arm["id"], seed)
        enforce_patient_view(cfg)
        marker = logs / f"{arm['id']}__seed{seed}.done"
        if marker.exists():
            continue
        if args.dry_run:
            print(f"would run [{arm['role']}] {path.relative_to(ROOT)}")
            continue
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / f"{arm['id']}__seed{seed}.log").open("a") as fh:
            rc = subprocess.run([sys.executable, str(ROOT / "scripts/run_masking_benchmark.py"), "--config", str(path), "--mode", "all"],
                                cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False).returncode
        if rc:
            raise RuntimeError(f"run failed ({arm['id']}, seed {seed}); see {logs}")
        marker.write_text("done\n")
        print(f"done {arm['id']} seed {seed}", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, default=ROOT / cm.MATRIX_PATH)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate").set_defaults(fn=generate)
    sub.add_parser("validate").set_defaults(fn=validate)
    r = sub.add_parser("run")
    r.add_argument("--only", nargs="+")
    r.add_argument("--seeds", nargs="+", type=int)
    r.add_argument("--include-sensitivity", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--split", choices=("validation", "test"), default="validation")
    r.add_argument("--allow-incomplete-validation-only", action="store_true")
    r.set_defaults(fn=run)
    args = p.parse_args(argv)
    if args.cmd == "run" and args.seeds and not set(args.seeds) <= set(cm.FROZEN_SEEDS):
        p.error(f"--seeds must be a subset of {cm.FROZEN_SEEDS}")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
