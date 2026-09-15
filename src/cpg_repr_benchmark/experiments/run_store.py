from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from cpg_repr_benchmark.config import config_fingerprint



def git_revision(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None

def create_run_dir(cfg: dict[str, Any], repo_root: Path) -> Path:
    root = Path(cfg["experiment"].get("output_root", "outputs"))
    if not root.is_absolute():
        root = repo_root / root
    dataset_name = str(cfg["dataset"].get("name", "dataset"))
    representation_name = str(cfg["representation"]["name"])
    seed = int(cfg["training"]["seed"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{config_fingerprint(cfg)}"
    run_dir = root / "masking" / dataset_name / representation_name / f"seed_{seed}" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return run_dir


def write_experiment_manifest(run_dir: Path, payload: dict[str, Any]) -> None:
    payload = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), **payload}
    (Path(run_dir) / "experiment.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_summary(run_dir: Path, payload: dict[str, Any]) -> None:
    (Path(run_dir) / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True))
