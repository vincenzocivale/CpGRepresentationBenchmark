#!/usr/bin/env python3
"""PRE-TEST audit of the external GSE40279 confirmation (protocol v1.2). Read-only; writes ONLY under outputs/external_reconstruction_v1/audit/.

  report   [--label L] [--skip-heavy] [--skip-fingerprint] [--skip-gate] [--no-write] [--overwrite]
               all checks (checkpoint hashes, frozen split/locus universe, prediction pairing, no evaluation/test dir, primary biological
               fingerprint, gate, tags, manifest state). Exit 1 if any non-pending check fails. Writes pretest_audit_<L>.json.
  snapshot --label L        record sha256 + size + mtime of ALL files under benchmark/ and analysis/A -> A_snapshot_<L>.json
  compare  --label L [--ignore-mtime]
               compare the current tree with that snapshot; exit 1 on any changed / added / removed file (and mtime-only change unless ignored)

`pending` = facts that become true only after the orchestrator's commit / tags (v1.2 tag, authorization tag, committed implementation).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cpg_repr_benchmark.external import pretest_audit as PA


def cmd_report(a) -> int:
    label = a.label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rep = PA.run_audit(ROOT, label=label, heavy=not a.skip_heavy, fingerprint=not a.skip_fingerprint, gate=not a.skip_gate)
    for c in rep["checks"]:
        print(f"{c['status'].upper():7s} {c['name']} {c['detail']}")
    print(f"counts {rep['counts']}; manifest {rep['manifest']}")
    print("expected pending until the orchestrator commits/tags:")
    for x in rep["expected_pending"]:
        print("  -", x)
    if not a.no_write:
        print("report ->", PA.write_report(ROOT, rep, overwrite=a.overwrite))
    return PA.exit_code(rep)


def cmd_snapshot(a) -> int:
    p = PA.write_snapshot(ROOT, a.label)
    print(f"snapshot -> {p} ({json.loads(p.read_text())['n_files']} files)")
    return 0


def cmd_compare(a) -> int:
    res = PA.compare_with_saved(ROOT, a.label, ignore_mtime=a.ignore_mtime)
    print(json.dumps({k: (v if not isinstance(v, list) else v[:20]) for k, v in res.items()}, indent=2))
    print("IDENTICAL" if res["identical"] else "CHANGED")
    return 0 if res["identical"] else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("report")
    r.add_argument("--label")
    r.add_argument("--skip-heavy", action="store_true")
    r.add_argument("--skip-fingerprint", action="store_true")
    r.add_argument("--skip-gate", action="store_true")
    r.add_argument("--no-write", action="store_true")
    r.add_argument("--overwrite", action="store_true")
    r.set_defaults(fn=cmd_report)
    s = sub.add_parser("snapshot")
    s.add_argument("--label", required=True)
    s.set_defaults(fn=cmd_snapshot)
    c = sub.add_parser("compare")
    c.add_argument("--label", required=True)
    c.add_argument("--ignore-mtime", action="store_true")
    c.set_defaults(fn=cmd_compare)
    a = p.parse_args(argv)
    if a.cmd is None:
        a = p.parse_args(["report", *(argv or [])])
    os.nice(10)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
