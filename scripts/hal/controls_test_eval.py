#!/usr/bin/env python3
"""R15: one-shot TEST evaluation of the 10 finished controls (b1_constant, b1_lookup, b2_gauss_d128, 7 x b2_<arm>_pca128; seed 17).

  freeze [--authorize "<who/when>"]   check best.pt and hash the 10 runs; write configs/experiments/test_controls_v1/{<run>.yaml,freeze_manifest.json}.
                                      Reads no data, no test output. Manifest is DRAFT (test locked) unless --authorize is given (then final).
  run [--execute]                     default = check only (every hash + the authorisation); --execute evaluates the TEST split of each run once
                                      (sequential, `flock` on the GPU lock, nice), writing only <run>/evaluation/test/ (same masks/panels as
                                      the validation evaluation and the main test: mask seed 17001, panels of 2,048, 5 mask fractions).
Evaluation goes through scripts/run_b1b2_arm.py (= run_masking_benchmark.py + the lookup-adapter hook; run_masking_benchmark.py alone
cannot load the lookup checkpoint). The `eval` subcommand (internal, called by `run`) installs the gate of
src/cpg_repr_benchmark/experiments/controls_gate.py for this one process instead of editing eval_layout.py, whose sha256 is pinned by the
GSE90496 zero-shot freeze manifest.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_test_confirmation_v2_freeze import diff_paths  # noqa: E402
from cpg_repr_benchmark.experiments import controls_gate as cg  # noqa: E402
from cpg_repr_benchmark.experiments.test_confirmation_v2 import EVAL_KEYS_ALLOWED_TO_DIFFER, canonical_cfg_sha256, sha256_file  # noqa: E402

CTL = "outputs/b1_b2_baselines/runs/masking/tcga_array_genomewide"
ARMS = (["b1_constant", "b1_lookup", "b2_gauss_d128"]
        + [f"b2_{r}_pca128" for r in ("regulatory_raw2492", "cpgpt_large_locus", "ntv3_pre_latent_native", "alphagenome_locus",
                                      "enformer_latent_native", "ntv3_post_latent_native", "sei_latent_native")])
SEED = 17
OUT = ROOT / "configs/experiments/test_controls_v1"
GPU_LOCK = "/data2/home/vcivale/.gpu_extract.lock"
CODE_FILES = ["scripts/hal/controls_test_eval.py", "scripts/run_b1b2_arm.py", "scripts/run_masking_benchmark.py",
              "src/cpg_repr_benchmark/experiments/controls_gate.py", "src/cpg_repr_benchmark/experiments/eval_layout.py",
              "src/cpg_repr_benchmark/experiments/test_confirmation_v2.py", "src/cpg_repr_benchmark/experiments/guards.py",
              "src/cpg_repr_benchmark/representations/baselines.py", "src/cpg_repr_benchmark/representations/resolver.py",
              "src/cpg_repr_benchmark/representations/hdf5_store.py", "src/cpg_repr_benchmark/training/engine.py",
              "src/cpg_repr_benchmark/data/splits.py", "src/cpg_repr_benchmark/data/masking.py", "src/cpg_repr_benchmark/models/model.py",
              "outputs/encode_atlas_v1/patients.npz", "outputs/encode_atlas_v1/loci_seen.npz"]


def finished_run(arm: str) -> Path:
    """The one complete run dir (best.pt, 120-epoch history, validation metrics); older aborted dirs of the same arm are ignored."""
    ok = [d for d in sorted((ROOT / CTL / arm / f"native_frozen/seed_{SEED}").glob("*"))
          if (d / "checkpoints/best.pt").is_file() and (d / "history.json").is_file()
          and (d / "evaluation/validation/seen/mask_0.50/metrics.json").is_file()]
    if len(ok) != 1:
        sys.exit(f"{arm}: expected exactly one finished run, found {[d.name for d in ok]}")
    return ok[0]


def cmd_freeze(a):
    mp = ROOT / cg.MANIFEST_REL
    if mp.exists() and json.loads(mp.read_text()).get("freeze_state") == "final":
        sys.exit(f"{mp} is final; refusing to overwrite")
    import torch
    runs, issues = {}, []
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        rid, rd = f"{arm}__seed{SEED}", finished_run(arm)
        rel = lambda p: str(Path(p).relative_to(ROOT))
        hist = json.loads((rd / "history.json").read_text())
        best_epoch = min(range(len(hist)), key=lambda i: hist[i]["validation_mse"])
        ck = torch.load(rd / "checkpoints/best.pt", map_location="cpu", weights_only=False)
        if len(hist) != 120 or ck["epoch"] != best_epoch:
            issues.append(f"{rid}: epochs={len(hist)}, checkpoint epoch {ck['epoch']} vs history argmin {best_epoch}")
        if (rd / "evaluation/test").exists():
            issues.append(f"{rid}: evaluation/test already exists")
        resolved = yaml.safe_load((rd / "resolved_config.yaml").read_text())
        cfg = copy.deepcopy(resolved)
        ev = cfg["evaluation"]
        ev.pop("require_patient_view", None)
        ev.update({"patient_view": "test", "output_layout": "split_dirs", "allow_overwrite_split_dir": False, "num_workers": 0,
                   "preserve_run_files": True, "test_authorization": cg.POLICY, "test_freeze_run_id": rid})
        unexpected = [p for p in diff_paths(resolved, cfg) if not (p.startswith("evaluation.") and p.split(".")[1] in EVAL_KEYS_ALLOWED_TO_DIFFER)]
        if unexpected:
            issues.append(f"{rid}: unexpected config differences {unexpected}")
        cp = OUT / f"{rid}.yaml"
        cp.write_text(yaml.safe_dump(cfg, sort_keys=False))
        files = [rel(rd / "checkpoints/best.pt"), rel(rd / "resolved_config.yaml"), rel(rd / "patient_split.npz"), rel(rd / "locus_split.npz"),
                 rel(rd / "prior_logit.npy"), resolved["representation"]["store_h5"], rel(cp)]
        runs[rid] = {"arm": arm, "seed": SEED, "run_dir": rel(rd), "best_epoch": int(ck["epoch"]), "eval_config": rel(cp),
                     "validation_mse_mask050": json.loads((rd / "evaluation/validation/seen/mask_0.50/metrics.json").read_text())["mse"],
                     "eval_config_canonical_sha256": canonical_cfg_sha256(yaml.safe_load(cp.read_text())),
                     "files": {p: sha256_file(ROOT / p) for p in files}}
        print("frozen", rid, flush=True)
    for i in issues:
        print("ISSUE", i)
    if issues:
        sys.exit("freeze not written: fix the issues above")
    final = a.authorize is not None
    m = {"policy": cg.POLICY, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "freeze_state": "final" if final else "draft", "test_set_authorized": final,
         "authorization": a.authorize or "pending: Vincenzo authorised this test evaluation on 2026-11-02 (relayed by the project lead); "
                                          "run `freeze --authorize` to record it and unlock",
         "checkpoint": "best.pt (min validation MSE at mask 0.50, 120 epochs, selected before any test read)",
         "shared_files": {p: sha256_file(ROOT / p) for p in CODE_FILES}, "runs": runs}
    mp.write_text(json.dumps(m, indent=1))
    print(f"wrote {mp.relative_to(ROOT)}: {len(runs)} runs, {m['freeze_state']}")


def cmd_eval(a):
    from cpg_repr_benchmark.experiments import eval_layout as el
    orig = el.authorizer_from_cfg
    el.authorizer_from_cfg = lambda cfg: (cg.authorizer_from_cfg(cfg) if cfg["evaluation"].get("test_authorization") == cg.POLICY else orig(cfg))
    sys.argv = ["run_b1b2_arm.py", "--config", a.config, "--mode", "evaluate", "--run-dir", a.run_dir]
    runpy.run_path(str(ROOT / "scripts/run_b1b2_arm.py"), run_name="__main__")


def cmd_run(a):
    m = json.loads((ROOT / cg.MANIFEST_REL).read_text())
    todo, bad = [], []
    for rid, e in m["runs"].items():
        bad += [f"{rid}: {p}" for p in cg.verify(yaml.safe_load((ROOT / e["eval_config"]).read_text()))]
        tdir = ROOT / e["run_dir"] / "evaluation/test"
        if (tdir / "summary.json").exists():
            print(f"[skip] {rid}: already evaluated")
        elif tdir.exists() and any(tdir.iterdir()):
            sys.exit(f"[STOP] {rid}: evaluation/test is non-empty without summary.json (partial run); inspect manually")
        else:
            todo.append((rid, e))
    if bad:
        print("PREFLIGHT NOT GREEN:\n  " + "\n  ".join(bad))
        return 2
    print(f"preflight green ({len(m['runs'])} runs hashed, test authorised); {len(todo)} to evaluate")
    if not a.execute:
        return 0
    for rid, e in todo:
        cmd = ["flock", GPU_LOCK, "nice", "-n", "10", sys.executable, str(ROOT / "scripts/hal/controls_test_eval.py"),
               "eval", "--config", str(ROOT / e["eval_config"]), "--run-dir", str(ROOT / e["run_dir"])]
        print(f"[run ] {rid}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, check=False)
        if r.returncode:
            print(f"[FAIL] {rid}: exit {r.returncode}; stopping", flush=True)
            return r.returncode
        print(f"[done] {rid} {time.time() - t0:.0f}s", flush=True)
    print("all runs evaluated")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("freeze").add_argument("--authorize", default=None, help="record the authorisation text and make the manifest final")
    sp.add_parser("run").add_argument("--execute", action="store_true", help="really evaluate the TEST split (default: check only)")
    ev = sp.add_parser("eval")
    ev.add_argument("--config", required=True)
    ev.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    sys.exit({"freeze": cmd_freeze, "run": cmd_run, "eval": cmd_eval}[args.cmd](args))
