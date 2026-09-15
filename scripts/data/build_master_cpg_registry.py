#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cpg_repr_benchmark.data.master_registry import build_master_registry


def _parse_source(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("--source must be NAME=PATH")
    name, value = spec.split("=", 1)
    if not name.strip() or not value.strip():
        raise argparse.ArgumentTypeError("--source must be NAME=PATH")
    return name.strip(), Path(value).expanduser()


def _read(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=["chr", "pos"])
    if suffix in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t" if suffix in {".tsv", ".txt"} else ",", usecols=["chr", "pos"])
    raise ValueError(f"unsupported coordinate source: {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build the coordinate-native GRCh38 CpG benchmark registry")
    p.add_argument("--source", action="append", required=True, type=_parse_source, help="repeatable NAME=PATH")
    p.add_argument("--output", type=Path, default=Path("data/cpg/master_cpg_registry.parquet"))
    p.add_argument("--membership-output", type=Path, default=Path("data/cpg/master_cpg_membership.parquet"))
    args = p.parse_args()

    sources = {}
    for name, path in args.source:
        if name in sources:
            raise ValueError(f"duplicate source name: {name}")
        sources[name] = _read(path)
    registry, membership = build_master_registry(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.membership_output.parent.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(args.output, index=False)
    membership.to_parquet(args.membership_output, index=False)
    payload = {
        "registry": str(args.output),
        "membership": str(args.membership_output),
        "n_loci": int(len(registry)),
        "n_memberships": int(len(membership)),
        "sources": {name: int((membership["source"] == name).sum()) for name in sources},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
