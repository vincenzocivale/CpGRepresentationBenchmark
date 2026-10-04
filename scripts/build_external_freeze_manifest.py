#!/usr/bin/env python3
"""Write configs/external/gse40279_v1_freeze_manifest.json from the finished phase-1 artifacts (refuses to overwrite)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cpg_repr_benchmark.external.freeze import MANIFEST_REL, build_manifest
from cpg_repr_benchmark.external.io import write_json_new

REPO = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    m = build_manifest(a.repo)
    write_json_new(a.out or a.repo / MANIFEST_REL, m)
    print(json.dumps({k: m[k] for k in ("files_sha256", "budget")}, indent=1))
