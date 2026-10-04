"""Read-only check: the frozen primary-results fingerprint equals the registered value. Exit 1 on mismatch; never writes."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cpg_repr_benchmark.bioval_v2_followup import gate


def main() -> int:
    base = gate.base_dir(ROOT)
    if not base.is_dir():
        print(f"outputs tree absent: {base}", file=sys.stderr)
        return 1
    got = gate.frozen_results_fingerprint(ROOT)
    ok = got == gate.FROZEN_RESULTS_SHA256
    print(f"current    {got}\nregistered {gate.FROZEN_RESULTS_SHA256}\n{'MATCH' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
