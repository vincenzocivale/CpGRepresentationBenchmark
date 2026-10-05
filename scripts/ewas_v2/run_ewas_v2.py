#!/usr/bin/env python
"""EWAS-v2 runner (gate | dry-run | run --arm A | contrasts | report-inputs). See docs/EWAS_V2_REGISTRATION.md.

Cap: <= 4 threads per process; launch with `nice -n 10`; one heavy process at a time."""
import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cpg_repr_benchmark.ewas_v2 import gate as G
from cpg_repr_benchmark.ewas_v2 import runner as R


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out-root", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--skip-store-read", action="store_true", help="never allowed for a real run (preparation/dry-run only)")
    sub.add_parser("dry-run").add_argument("--sec-per-unit", type=float, default=5e-9)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, choices=G.ALL_ARMS)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--tasks", nargs="*", default=None)
    r.add_argument("--designs", nargs="*", default=list(R.DESIGNS), choices=list(R.DESIGNS))
    c = sub.add_parser("contrasts")
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--allow-nonconverged", action="store_true")
    sub.add_parser("report-inputs")
    a = ap.parse_args(argv)
    root = Path(a.root)
    out_root = Path(a.out_root) if a.out_root else R.default_out_root(root)
    cmd = " ".join(sys.argv)
    if a.cmd == "gate":
        checks = R.cmd_gate(root, out_root, read_stores=not a.skip_store_read, command="run", arm=None)
        return 0 if all(c.ok for c in checks) else 1
    if a.cmd == "dry-run":
        R.cmd_dry_run(root, out_root, a.sec_per_unit)
        return 0
    if a.cmd == "run":
        print(R.run_arm(root, a.arm, out_root, threads=a.threads, tasks=a.tasks, designs=tuple(a.designs), command=cmd))
    elif a.cmd == "contrasts":
        print(R.run_contrasts(root, out_root, threads=a.threads, allow_nonconverged=a.allow_nonconverged))
    elif a.cmd == "report-inputs":
        print(R.report_inputs(root, out_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
