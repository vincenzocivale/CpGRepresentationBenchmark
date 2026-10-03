#!/usr/bin/env python3
"""Pre-test audit of the confirmation matrix (configs/experiments/regulatory_confirmation_matrix/matrix.yaml).

Exit 0 only if EVERY check passes; otherwise exit 1 with the list of failures. Metadata only: file sha256, HDF5 `/cpg_idx` and the
shape/dtype of `/embedding`, `loci_seen.npz:train_cpg_idx`. No methylation values, predictions or test patients are read.

Checks: store present; sha256 recorded and verified against the file; provenance + catalog entry; no pending comparator
(modern sequence FMs need a COMPLETE registration); seeds / mask fractions frozen; no arm deviates from the shared protocol
(epochs, early stopping, selection fraction, optimizer, panel, mask seed, split paths + their hashes); every store covers the protocol
loci; per-run configs generated and identical to the frozen builder; freeze_state / test_set_authorized state machine.

    python scripts/audit_confirmation_matrix.py [--matrix PATH] [--list] [--skip-rehash] [--no-configs]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cpg_repr_benchmark.experiments import confirmation_matrix as cm


def run_audit(matrix: Path, *, rehash: bool = True, configs: bool = True):
    spec = cm.load_matrix(matrix)
    return spec, cm.audit(spec, ROOT, check_hashes=rehash, check_configs=configs)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, default=ROOT / cm.MATRIX_PATH)
    p.add_argument("--list", action="store_true", help="print every check (PASS and FAIL); default prints failures + summary")
    p.add_argument("--skip-rehash", action="store_true",
                   help="do NOT recompute store sha256 (recorded hash is still required); NOT acceptable for the final audit")
    p.add_argument("--no-configs", action="store_true", help="skip the per-run config comparison")
    args = p.parse_args(argv)
    os.nice(10)
    spec, results = run_audit(args.matrix, rehash=not args.skip_rehash, configs=not args.no_configs)
    fails = cm.failures(results)
    for r in results:
        if args.list or not r.ok:
            print(r.line())
    print(f"\nconfirmation matrix '{spec.get('name')}': freeze_state={spec.get('freeze_state')}, "
          f"test_set_authorized={spec.get('test_set_authorized')}: {len(results) - len(fails)} passed, {len(fails)} FAILED")
    if fails:
        print("AUDIT FAILED: the final benchmark is NOT authorizable.")
        return 1
    print("AUDIT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
