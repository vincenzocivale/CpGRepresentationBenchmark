#!/usr/bin/env python3
"""Runner of the FROZEN external reconstruction confirmation on GSE40279 (protocol v1.1, docs/EXTERNAL_RECONSTRUCTION_PROTOCOL.md).

  generate [--keep-last-checkpoint]   write the 12 per-run configs (4 arms x seeds 17/42/97) to configs/experiments/external_confirmation/
  validate                            no-training check: configs == builder, diff vs the TCGA configs only in the declared fields
  audit    [--split validation|test] [--scope A|B] [--json PATH]   run the gate and print every check (exit 1 if any fails)
  run      [--phase A] [--split validation] [--only ARM...] [--seeds S...] [--dry-run] [--test]
                                      gate, then train+evaluate sequentially (seed-major, resumable via .done markers).
                                      --dry-run lists the plan and launches nothing. --split test / --test are REFUSED unless the
                                      manifest is final + test_set_authorized + all-green (and are not part of the validation phase).
  status                              progress of the 12 runs (reads markers / confirmation_status.json only)
  test-eval                           TEST evaluation of the finished runs; refused unless final + authorized + green gate

This script never launches anything unless `run` is called without --dry-run AND the gate is green.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cpg_repr_benchmark.external import gate as G
from cpg_repr_benchmark.external import runner as R


def cmd_generate(a) -> int:
    paths = R.generate(ROOT, keep_last_checkpoint=a.keep_last_checkpoint)
    print(f"wrote {len(paths)} configs to {R.CONFIG_DIR} (save_last_checkpoint={a.keep_last_checkpoint})")
    return 0


def cmd_validate(a) -> int:
    import yaml
    bad = []
    for arm, seed in R.jobs():
        p = ROOT / R.CONFIG_DIR / R.config_name(arm, seed)
        if not p.is_file():
            bad.append(f"{p.name} missing")
            continue
        cfg = yaml.safe_load(p.read_text())
        fresh = R.build_config(arm, seed, ROOT, keep_last_checkpoint=bool(cfg["training"].get("save_last_checkpoint", False)))
        if cfg != fresh:
            bad.append(f"{p.name} stale")
        extra = R.check_diff_vs_tcga(arm, seed, ROOT, cfg)
        if extra:
            bad.append(f"{p.name}: undeclared differences vs TCGA: {extra}")
    print(json.dumps({"n_configs": len(R.jobs()), "problems": bad}, indent=2))
    return 1 if bad else 0


def cmd_audit(a) -> int:
    checks = G.run_gate(ROOT, split=a.split, scope=a.scope)
    for c in checks:
        print(c.line())
    d = G.as_dict(checks)
    print(f"{d['n_checks'] - d['n_failed']}/{d['n_checks']} checks green; all_green={d['all_green']}")
    if a.json:
        out = G.assert_writable(ROOT, a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(d, indent=2))
    return 0 if d["all_green"] else 1


def cmd_run(a) -> int:
    split = "test" if a.test else a.split
    if split == "test":
        manifest = json.loads((ROOT / "configs/external/gse40279_v1_freeze_manifest.json").read_text())
        bad = [c for c in G.check_state(manifest, split="test") if not c.ok]
        for c in bad:
            print("  " + c.line())
        print("REFUSED: the external TEST split is locked (needs freeze_state final + test_set_authorized + green gate)" if bad else
              "REFUSED: `run` is the validation phase only; use `test-eval` for the test evaluation")
        return 2
    return R.run_all(ROOT, seeds=a.seeds, only=a.only, dry_run=a.dry_run, split="validation")


def cmd_status(a) -> int:
    print(json.dumps(R.aggregate_status(ROOT), indent=2))
    return 0


def cmd_test_eval(a) -> int:
    return R.evaluate_test_all(ROOT)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--keep-last-checkpoint", action="store_true", help="write training.save_last_checkpoint: true (audit/debug)")
    g.set_defaults(fn=cmd_generate)
    sub.add_parser("validate").set_defaults(fn=cmd_validate)
    au = sub.add_parser("audit")
    au.add_argument("--split", choices=("validation", "test"), default="validation")
    au.add_argument("--scope", choices=("A", "B"), default="A")
    au.add_argument("--json", type=Path, default=None)
    au.set_defaults(fn=cmd_audit)
    r = sub.add_parser("run")
    r.add_argument("--phase", choices=("A",), default="A")
    r.add_argument("--split", choices=("validation", "test"), default="validation")
    r.add_argument("--test", action="store_true")
    r.add_argument("--only", nargs="+", choices=list(R.ARMS))
    r.add_argument("--seeds", nargs="+", type=int)
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=cmd_run)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("test-eval").set_defaults(fn=cmd_test_eval)
    a = p.parse_args(argv)
    if a.cmd == "run" and a.seeds and not set(a.seeds) <= set(R.SEEDS):
        p.error(f"--seeds must be a subset of {list(R.SEEDS)}")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
