#!/usr/bin/env python3
"""One-shot TEST evaluation of the frozen post-hoc panel (tcga_posthoc_freeze_v2). Sequential, idempotent.

Default (no flag) = CHECK ONLY: verifies the freeze and prints the state, evaluates nothing. `--execute` runs
`run_masking_benchmark.py --mode evaluate` for each of the 24 runs under `flock` on the GPU lock and `nice`, writing only to
<run>/evaluation/test/. Refuses to start unless: manifest is final + test_set_authorized, all 24 runs are complete and hashed
(no pending), every sha256 matches, and manifest/code are committed and clean. See docs/CONFIRMATORY_FREEZE_V2.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cpg_repr_benchmark.experiments import test_confirmation_v2 as t2
from cpg_repr_benchmark.experiments.guards import require_test_authorization

GPU_LOCK = "/data2/home/vcivale/.gpu_extract.lock"
DONE = ".done_tcga_posthoc_freeze_v2.json"


def preflight(manifest: dict, *, need_authorization: bool) -> list[str]:
    problems = []
    if need_authorization:
        try:
            require_test_authorization(manifest)
        except PermissionError as e:
            problems.append(str(e))
    problems += t2.verify_manifest(manifest, ROOT, stores="all")
    problems += t2.git_problems(ROOT)
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="really evaluate the TEST split (default: check only)")
    ap.add_argument("--manifest", default=t2.MANIFEST_REL)
    ap.add_argument("--only", nargs="*", help="run ids (debug; still needs the full green preflight)")
    args = ap.parse_args()
    manifest = t2.load_manifest(ROOT / args.manifest)
    problems = preflight(manifest, need_authorization=args.execute)
    if problems:
        print("PREFLIGHT NOT GREEN:\n  " + "\n  ".join(problems))
        return 2
    print("preflight green (24 runs hashed, committed and clean)" + (" + test authorized" if args.execute else ""))
    if not args.execute:
        return 0
    for rid, e in manifest["runs"].items():
        if args.only and rid not in args.only:
            continue
        run_dir = ROOT / e["run_dir"]
        marker = run_dir / "evaluation" / "test" / DONE
        if marker.exists():
            print(f"[skip] {rid}: already done"); continue
        test_dir = run_dir / "evaluation" / "test"
        if test_dir.exists() and any(test_dir.iterdir()):
            print(f"[STOP] {rid}: evaluation/test is non-empty without done marker (partial run); inspect manually"); return 3
        # hashes were verified by the preflight; the child re-verifies this run's files via the authorizer before reading any data
        cmd = ["flock", GPU_LOCK, "nice", "-n", "10", sys.executable, str(ROOT / "scripts/run_masking_benchmark.py"),
               "--config", str(ROOT / e["eval_config"]), "--mode", "evaluate", "--run-dir", str(run_dir)]
        env = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "PYTHONPATH": str(ROOT / "src")}
        print(f"[run ] {rid}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
        if r.returncode != 0:
            print(f"[FAIL] {rid}: exit {r.returncode}; stopping"); return r.returncode
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
        marker.write_text(json.dumps({"run_id": rid, "git_head": head, "seconds": time.time() - t0,
                                      "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
    print("all requested runs done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
