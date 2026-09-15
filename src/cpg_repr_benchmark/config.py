from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    required = {"experiment", "dataset", "representation", "model", "training", "evaluation"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"config is missing required top-level keys: {sorted(missing)}")
    return cfg


def config_fingerprint(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:10]
