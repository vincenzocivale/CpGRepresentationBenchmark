#!/usr/bin/env python3
"""Runner of the frozen confirmation matrix (docs/REGULATORY_CONFIRMATION_PROTOCOL.md AMENDMENT 2; spec: matrix.yaml).

  generate   write configs/experiments/regulatory_confirmation_matrix/runs/<arm>__seed<seed>.yaml for every AVAILABLE arm
             (main arms x 3 seeds, plus sensitivity arms, marked role: sensitivity). Pending arms get no config. Every config uses
             `evaluation.output_layout: split_dirs` (results under <run>/evaluation/validation/).
  validate   no-training check: configs == frozen builder, equal to the campaign template except allowed keys, stores/cpg_idx
  run        train+evaluate sequentially (one GPU job at a time, seed-major, resumable via .done markers)
             [--phase A] [--only ARM...] [--seeds S...] [--include-sensitivity] [--dry-run] [--split validation|test]
             [--allow-incomplete-validation-only]

PHASE A: `run --phase A --split validation` trains/evaluates ONLY the 4 fully registered main arms (histone_dnase, functional,
cpgpt_large, deepcpg) x seeds 17/42/97 = 12 runs, seed-major, validation split only. It needs no escape-hatch flag: it is permitted
when every Phase A arm passes the audit (only the pending modern sequence FM slots may fail). The pending slots still block
`freeze_state: final`, `test_set_authorized: true` and any test evaluation. On each finished run the runner writes
`<run>/confirmation_status.json` (status `trained_validation_frozen`, never `confirmed`) and refreshes
`outputs/regulatory_confirmation_v1/phase_A_status.json`.

Without --phase, `run` refuses while the audit fails (legacy flag --allow-incomplete-validation-only = equivalent escape hatch for all
available arms). `--split test` is refused unless test_set_authorized is true AND freeze_state is final AND the audit is green; even
then this scaffold does not execute test evaluation. This script never launches anything unless `run` is called without --dry-run.
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
    import time

    from cpg_repr_benchmark.experiments.guards import enforce_patient_view
    spec = _spec(args)
    os.nice(10)
    results = cm.audit(spec, ROOT, check_hashes=True, check_configs=False)
    ok, why = cm.gate(spec, results, split=args.split, allow_incomplete_validation_only=args.allow_incomplete_validation_only,
                      phase=args.phase)
    print(f"GATE ({args.split}{', phase ' + args.phase if args.phase else ''}): {'OPEN' if ok else 'REFUSED'}: {why}")
    for r in cm.failures(results):
        print("  " + r.line())
    if not ok:
        return 2
    if args.split == "test":
        print("REFUSED: test-split execution is intentionally not implemented in this scaffold (the split_dirs layout now separates "
              "evaluation/test from evaluation/validation, but the test run itself needs explicit user authorization).")
        return 2
    template = yaml.safe_load((ROOT / spec["protocol"]["template"]).read_text())
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "PYTHONPATH": str(ROOT / "src")}
    runs_root = ROOT / cm.OUTPUT_ROOT
    logs = runs_root / "logs"
    todo = cm.jobs(spec, seeds=args.seeds, include_sensitivity=args.include_sensitivity, only=args.only, phase=args.phase)
    for arm, seed in todo:   # validate every config first so a stale/invalid one fails before any training starts
        path = ROOT / cm.CONFIG_DIR / cm.config_name(arm["id"], seed)
        cfg = yaml.safe_load(path.read_text())
        if cfg != cm.build_config(spec, arm, seed, ROOT):
            raise ValueError(f"{path} is stale; re-run generate")
        cm.check_config(cfg, template, spec, arm["id"], seed)
        enforce_patient_view(cfg)
    print(f"{len(todo)} job(s), seed-major: " + ", ".join(f"{a['id']}#{s}" for a, s in todo))
    for arm, seed in todo:
        path = ROOT / cm.CONFIG_DIR / cm.config_name(arm["id"], seed)
        marker = logs / f"{arm['id']}__seed{seed}.done"
        if marker.exists():
            continue
        if args.dry_run:
            print(f"would run [{arm['role']}] {path.relative_to(ROOT)}")
            continue
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{arm['id']}__seed{seed}.log"
        t0 = time.time()
        with log.open("a") as fh:
            rc = subprocess.run([sys.executable, str(ROOT / "scripts/run_masking_benchmark.py"), "--config", str(path), "--mode", "all"],
                                cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False).returncode
        if rc:
            raise RuntimeError(f"run failed ({arm['id']}, seed {seed}); see {logs}")
        run_dir = cm.run_dir_from_log(log)
        if run_dir is None:
            raise RuntimeError(f"no RUN_DIR in {log}")
        status = cm.build_status(arm["id"], seed, run_dir, wall_clock_seconds=time.time() - t0, spec=spec)  # raises on protocol failure
        cm.write_status(run_dir, status)
        marker.write_text("done\n")
        cm.write_phase_a_status(spec, runs_root)
        print(f"done {arm['id']} seed {seed}: {status['status']} best_epoch={status['best_epoch_0based']} "
              f"val_mse@0.50={status['best_val_mse_at_0.50']:.6g} wall={status['wall_clock_seconds']}s", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, default=ROOT / cm.MATRIX_PATH)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate").set_defaults(fn=generate)
    sub.add_parser("validate").set_defaults(fn=validate)
    r = sub.add_parser("run")
    r.add_argument("--phase", choices=("A",), default=None,
                   help="A = the 4 fully registered main arms x 3 seeds, validation only (no pending-FM dependency)")
    r.add_argument("--only", nargs="+")
    r.add_argument("--seeds", nargs="+", type=int)
    r.add_argument("--include-sensitivity", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--split", choices=("validation", "test"), default="validation")
    r.add_argument("--allow-incomplete-validation-only", action="store_true")
    r.set_defaults(fn=run)
    args = p.parse_args(argv)
    if args.cmd == "run" and args.phase and args.include_sensitivity:
        p.error("--phase A excludes sensitivity arms")
    if args.cmd == "run" and args.seeds and not set(args.seeds) <= set(cm.FROZEN_SEEDS):
        p.error(f"--seeds must be a subset of {cm.FROZEN_SEEDS}")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
