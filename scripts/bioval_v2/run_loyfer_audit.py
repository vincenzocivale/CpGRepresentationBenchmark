"""Loyfer primary failure audit (EXPLORATORY POST-HOC). Registration: docs/LOYFER_PRIMARY_FAILURE_AUDIT_REGISTRATION.md.

Subcommands (each: --dry-run = gate report + read-only counts + cost, computes NO audit statistic; --threads <= 8):
  target-audit      section 1 LEVEL A only (counts h5 + provenance + S1; verdict PASS_LEVEL_A; raw .pat Level B not executed) (writes VERDICT.json) then section 2 (split-half reliability; only if PASS)
  semantics-pairs   sections 3-4 (target semantics, per-stratum pair-list audit)
  baselines         section 5 (+ precision of the frozen float16 target)     [--arm/--all]
  raw-vs-svd        section 6 (raw binary regulatory space vs 256-d SVD embedding)
  svd-geometry      section 7                                                 [--arm/--all]
  oof-similarity    section 8 (OOF predicted-profile pair similarity, original alpha grid)  [--arm/--all]
  probe-fairness    section 9 (extended alpha grid + edge rule)               [--arm/--all]
  report-inputs     section 11 decision rules over the finished outputs
  final-integrity   --snapshot (before the first real step) | --compare (after the last step)
Every real run refuses unless the audit gate passes; every step except target-audit and final-integrity also needs a PASS verdict of
target-audit (no override flag). Outputs only under outputs/biological_validation_v2/loyfer_failure_audit/ (AuditWriter).
No freeze/prepare/build command is run or imported; no TCGA / data/protocols path is read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _early_threads(argv):
    n = 2
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            n = int(argv[i + 1])
        elif a.startswith("--threads="):
            n = int(a.split("=", 1)[1])
    if not 1 <= n <= 8:
        raise SystemExit("--threads must be in [1, 8] (shared machine)")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(n)
    return n


THREADS = _early_threads(sys.argv)
sys.path.insert(0, str(ROOT / "src"))

from cpg_repr_benchmark.bioval_v2_launch import launch_gate as lg
from cpg_repr_benchmark.bioval_v2_loyfer_audit import drivers as dr
from cpg_repr_benchmark.bioval_v2_loyfer_audit import gate as gt
from cpg_repr_benchmark.bioval_v2_loyfer_audit import registry as rg

ARM_STEPS = ("baselines", "svd-geometry", "oof-similarity", "probe-fairness")
PY = "~/miniconda3/envs/cpg-repr-benchmark/bin/python"


def resolve_arms(args):
    if args.cmd not in ARM_STEPS:
        return list(rg.ARMS)
    if args.all:
        return list(rg.ARMS)
    if not args.arm:
        raise SystemExit("give --arm <id> (repeatable) or --all")
    for a in args.arm:
        if a not in rg.ARMS:
            raise SystemExit(f"unknown arm {a}; known: {rg.ARMS}")
    return args.arm


def launch_lines(cmd):
    """Low-impact launch: ONE heavy process at a time, nice -n 19, ionice idle, 2 threads."""
    base = f"nice -n 19 ionice -c3 {PY} scripts/bioval_v2/run_loyfer_audit.py"
    arms = " ".join(rg.ARMS)
    if cmd == "target-audit":
        return [f"{base} final-integrity --snapshot", f"{base} target-audit --threads 2   # Level A only (amendment 1)"]
    if cmd == "final-integrity":
        return [f"{base} final-integrity --compare"]
    if cmd in ARM_STEPS:
        return [f"for a in {arms}; do {base} {cmd} --arm $a --threads 2; done   # sequential, one arm at a time"] if cmd != "probe-fairness" \
            else [f"{base} probe-fairness --all --threads 2   # fits cached per (grid, arm)"]
    return [f"{base} {cmd} --threads 2"]


def cmd_dry_run(args, arms):
    out_root = Path(args.out_root or gt.audit_root(ROOT))
    print(f"=== Loyfer failure audit: {args.cmd}: DRY RUN (read-only counts; NO audit statistic is computed) ===")
    ctx = dr.Ctx(ROOT, out_root, THREADS)
    print(f"Output root (new, sibling of the frozen tag dir): {out_root}")
    print("\nGate (report only in dry-run):")
    inputs = [ctx.bv / v for v in rg.F.values()] + [ROOT / s.store_h5 for s in ctx.specs.values()]
    checks = gt.run_gate(ROOT, arms, out_root, args.cmd, input_paths=inputs, skip_heavy=args.skip_heavy, level_b=args.level_b)
    print(lg.format_checks(checks))
    if args.cmd != "final-integrity" and not args.no_counts:
        print("\nRead-only counts:")
        cnt = dr.dry_counts(ctx, with_raw=args.with_raw)
        for k, v in cnt.items():
            print(f"  {k}: {json.dumps(v, default=str)}")
    print(f"\nCost / memory estimate ({args.cmd}): {dr.COST_NOTES[args.cmd]}")
    print("\nProposed launch (after the orchestrator commits registration + code and the gate passes):")
    for line in launch_lines(args.cmd):
        print("  " + line)
    failed = [c.name for c in checks if not c.ok]
    print("\nDRY RUN COMPLETE: nothing written under the output root.")
    print("Gate status: " + ("ALL PASS" if not failed else "WOULD REFUSE A REAL RUN (failing: " + ", ".join(failed) + ")"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=rg.STEPS)
    ap.add_argument("--arm", action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-heavy", action="store_true", help="dry-run only: skip the sha256 of manifests/stores")
    ap.add_argument("--threads", type=int, default=2, help="<= 8 hard limit; 2 recommended on the loaded shared machine")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--with-raw", action="store_true", help="dry-run: also count all-zero raw regulatory vectors (reads ~1.5 GB of CSR indices)")
    ap.add_argument("--level-b", action="store_true", help="UNREGISTERED: raw .pat Level B (75 GB scan); the gate refuses it in every run")
    ap.add_argument("--no-counts", action="store_true", help="dry-run: gate + cost only (skip the read-only counts)")
    ap.add_argument("--snapshot", action="store_true", help="final-integrity: record the pre-run snapshot")
    ap.add_argument("--compare", action="store_true", help="final-integrity: compare with the pre-run snapshot")
    ap.add_argument("--sections", default="all", choices=["all", "1"], help="target-audit: '1' = section 1 only")
    args = ap.parse_args()
    arms = resolve_arms(args) if not args.dry_run else (args.arm or list(rg.ARMS) if args.cmd in ARM_STEPS else list(rg.ARMS))
    if args.dry_run:
        return cmd_dry_run(args, arms)
    out_root = Path(args.out_root or gt.audit_root(ROOT))
    ctx = dr.Ctx(ROOT, out_root, THREADS)
    inputs = [ctx.bv / v for v in rg.F.values()] + [ROOT / s.store_h5 for s in ctx.specs.values()]
    checks = gt.run_gate(ROOT, rg.ARMS, out_root, args.cmd, input_paths=inputs, level_b=args.level_b)
    print(lg.format_checks(checks))
    gt.enforce(checks)
    step_dir = rg.STEP_DIRS[args.cmd]
    seeds = {"global": rg.SEED, "subset": rg.SUBSET_SEED, "split_half_base": rg.SPLIT_SEED_BASE, "shuffle_base": rg.SHUFFLE_SEED_BASE,
             "random_base": rg.RANDOM_SEED_BASE, "semantics": rg.SEMANTICS_SEED, "bootstrap": [22, 1000, 17]}
    man = gt.build_manifest(ROOT, args.cmd, threads=THREADS, command=" ".join(sys.argv), checks=checks, seeds=seeds,
                            extra={"arms": arms})
    ctx.w.write_json(f"{step_dir}/run_manifest.json", man)
    if args.cmd == "target-audit":
        dr.step_target_audit(ctx, THREADS, sections=args.sections, with_level_b=args.level_b)
    elif args.cmd == "semantics-pairs":
        dr.step_semantics(ctx)
    elif args.cmd == "baselines":
        dr.step_baselines(ctx, arms)
    elif args.cmd == "raw-vs-svd":
        dr.step_raw_vs_svd(ctx)
    elif args.cmd == "svd-geometry":
        dr.step_svd_geometry(ctx, arms)
    elif args.cmd == "oof-similarity":
        dr.step_oof_similarity(ctx, arms)
    elif args.cmd == "probe-fairness":
        dr.step_probe_fairness(ctx, arms)
    elif args.cmd == "report-inputs":
        dr.step_report_inputs(ctx)
    elif args.cmd == "final-integrity":
        if args.snapshot == args.compare:
            raise SystemExit("final-integrity needs exactly one of --snapshot / --compare")
        res = dr.step_integrity(ctx, "snapshot" if args.snapshot else "compare")
        print(json.dumps(res, indent=1)[:3000])
    man["freeze_modules_imported"] = lg.freeze_modules_imported()
    man["no_freeze_command_used"] = not man["freeze_modules_imported"]
    man["finished_utc"] = gt.datetime.now(gt.timezone.utc).isoformat()
    man["frozen_primary_fingerprint_after"] = gt.fg.frozen_results_fingerprint(ROOT)
    ctx.w.write_json(f"{step_dir}/run_manifest.json", man)


if __name__ == "__main__":
    main()
