#!/usr/bin/env python3
"""Read-only verifier of the external GSE40279 freeze: recomputes hashes/counts/budget, exits 1 on any mismatch. Never writes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cpg_repr_benchmark.external.freeze import verify

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--skip-stores", action="store_true")
    ap.add_argument("--skip-checkpoints", action="store_true")
    a = ap.parse_args()
    res = verify(a.repo, a.manifest, check_stores=not a.skip_stores, check_checkpoints=not a.skip_checkpoints)
    for name, ok, detail in res:
        print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    bad = [n for n, ok, _ in res if not ok]
    print(f"{len(res) - len(bad)}/{len(res)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
