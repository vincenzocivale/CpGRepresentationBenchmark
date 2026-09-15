#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def flatten_summary(path: Path) -> list[dict]:
    summary = json.loads(path.read_text())
    rows = []
    path_track = path.parent.parent.parent.name
    if path_track == summary.get("representation"):
        path_track = None
    for view, masks in summary.get("evaluation", {}).items():
        for mask_name, metrics in masks.items():
            rows.append(
                {
                    "run_dir": str(path.parent),
                    "dataset": summary.get("dataset"),
                    "representation": summary.get("representation"),
                    "representation_dim": summary.get("representation_dim"),
                    "representation_track": summary.get("representation_track") or path_track,
                    "seed": summary.get("seed"),
                    "view": view,
                    "mask": mask_name,
                    "mask_fraction": metrics.get("mask_fraction"),
                    "mse": metrics.get("mse"),
                    "mae": metrics.get("mae"),
                    "skill_vs_prior": metrics.get("skill_vs_prior"),
                    "patient_pearson": metrics.get("patient_pearson"),
                    "n_pairs": metrics.get("n_pairs"),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate masking benchmark summary.json files")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--csv", type=Path, help="optional output CSV")
    args = parser.parse_args()
    # Track is now an explicit path component. Keep a legacy glob so pre-v2 runs
    # remain aggregatable during migration.
    paths = sorted({
        *args.outputs.glob("masking/*/*/*/seed_*/*/summary.json"),
        *args.outputs.glob("masking/*/*/seed_*/*/summary.json"),
    })
    if not paths:
        raise SystemExit(f"no summary.json files found under {args.outputs}")
    rows = [row for path in paths for row in flatten_summary(path)]
    columns = list(rows[0])
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.csv}")
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
