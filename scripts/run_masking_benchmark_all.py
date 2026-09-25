#!/usr/bin/env python3
"""Run `run_masking_benchmark.py --mode all` for every config in `configs/experiments/masking/`,
skipping configs that already have a complete run (a run dir with `summary.json`) under
`outputs/masking/<dataset>/<representation>/<track>/seed_<seed>/`.

Each config previously had to be launched by hand, one at a time, with no way to tell whether a
prior run had already completed without inspecting `outputs/` first. This mirrors
`run_bio_validation_all.py`'s skip-if-done behavior for the masking benchmark.

Usage:
    python scripts/run_masking_benchmark_all.py
    python scripts/run_masking_benchmark_all.py --only ntv3_precomputed
    python scripts/run_masking_benchmark_all.py --force   # recompute even if a summary.json exists
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_run_group_dirs(output_root: Path, cfg: dict) -> list[Path]:
    """Two layouts exist on disk for historical reasons: the current one nests a `<track>/`
    level between representation and seed, but several completed runs predate that and sit
    directly at `<representation>/seed_<seed>/`. Check both so pre-existing completed runs
    aren't mistaken for missing and re-launched.
    """
    task = str(cfg["experiment"].get("task", "masking"))
    dataset_name = str(cfg["dataset"]["name"])
    representation_name = str(cfg["representation"]["name"])
    representation_track = str(cfg["representation"].get("track", "native_frozen"))
    seed = int(cfg["training"]["seed"])
    base = output_root / task / dataset_name / representation_name
    return [
        base / representation_track / f"seed_{seed}",
        base / f"seed_{seed}",
    ]


def _has_complete_run(run_group_dirs: list[Path]) -> bool:
    for run_group_dir in run_group_dirs:
        if not run_group_dir.exists():
            continue
        if any((run_dir / "summary.json").exists() for run_dir in run_group_dir.iterdir() if run_dir.is_dir()):
            return True
    return False


def main() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-dir", type=Path, default=root / "configs/experiments/masking")
    p.add_argument("--output-root", type=Path, default=root / "outputs")
    p.add_argument("--only", nargs="*", default=None, help="restrict to these config stems (filename without .yaml)")
    p.add_argument("--force", action="store_true", help="rerun even if a completed run (summary.json) already exists")
    args = p.parse_args()

    configs = sorted(args.config_dir.glob("*.yaml"))
    results = {"ran": [], "skipped": [], "failed": []}

    for config_path in configs:
        stem = config_path.stem
        if args.only and stem not in args.only:
            continue

        cfg = yaml.safe_load(config_path.read_text())
        run_group_dirs = _candidate_run_group_dirs(args.output_root, cfg)

        if _has_complete_run(run_group_dirs) and not args.force:
            print(f"[skip] {stem}: already complete under {run_group_dirs[0]} (or legacy {run_group_dirs[1]})")
            results["skipped"].append(stem)
            continue

        print(f"[run]  {stem} -> {run_group_dirs[0]}")
        proc = subprocess.run(
            [sys.executable, str(root / "scripts/run_masking_benchmark.py"), "--config", str(config_path), "--mode", "all"],
            cwd=root,
            check=False,
        )
        if proc.returncode != 0:
            print(f"[fail] {stem}: exit code {proc.returncode}")
            results["failed"].append(stem)
        else:
            results["ran"].append(stem)

    print(json.dumps(results, indent=2))
    if results["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
