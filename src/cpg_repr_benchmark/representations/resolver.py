from __future__ import annotations

import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import RepresentationInfo
from .hdf5_store import inspect_representation_h5


def _format_command(command: list[str] | str, variables: dict[str, str]) -> list[str]:
    tokens = shlex.split(command) if isinstance(command, str) else list(command)
    return [token.format(**variables) for token in tokens]


def resolve_representation(
    cfg: dict[str, Any],
    *,
    registry_path: Path,
    run_dir: Path,
    repo_root: Path,
) -> RepresentationInfo:
    """Resolve a representation to the canonical HDF5 contract.

    Modes:
      - precomputed: require an existing store.
      - generate_if_missing: run the configured generator only when absent.
      - regenerate: always run the generator first.

    Generation is intentionally done once at run start, not inside DataLoader workers.
    This gives on-demand feature generation while keeping experiments deterministic and
    avoiding duplicated GPU model loads across workers.
    """
    name = str(cfg["name"])
    mode = str(cfg.get("mode", "precomputed"))
    store_path = Path(cfg["store_h5"]).expanduser()
    if not store_path.is_absolute():
        store_path = (repo_root / store_path).resolve()

    if mode not in {"precomputed", "generate_if_missing", "regenerate"}:
        raise ValueError(f"unsupported representation.mode={mode!r}")

    should_generate = mode == "regenerate" or (mode == "generate_if_missing" and not store_path.exists())
    if should_generate:
        generator = cfg.get("generator")
        if not generator or "command" not in generator:
            raise ValueError(f"representation {name!r} requires generator.command in mode={mode}")
        store_path.parent.mkdir(parents=True, exist_ok=True)
        variables = {
            "output": str(store_path),
            "registry": str(Path(registry_path).resolve()),
            "repo_root": str(repo_root.resolve()),
            "run_dir": str(run_dir.resolve()),
        }
        command = _format_command(generator["command"], variables)
        result = subprocess.run(command, cwd=repo_root, check=False, text=True, capture_output=True)
        log = run_dir / "representation_generation.log"
        log.write_text(
            "$ " + " ".join(shlex.quote(x) for x in command) + "\n\nSTDOUT\n" + result.stdout + "\nSTDERR\n" + result.stderr
        )
        if result.returncode != 0:
            raise RuntimeError(f"representation generator failed with code {result.returncode}; see {log}")

    if not store_path.exists():
        raise FileNotFoundError(f"representation store does not exist: {store_path}")
    cpg_ids, dim, id_key, embedding_key = inspect_representation_h5(
        store_path,
        id_key=cfg.get("id_key", "auto"),
        embedding_key=cfg.get("embedding_key", "auto"),
    )
    info = RepresentationInfo(
        name=name,
        store_path=store_path,
        n_loci=len(cpg_ids),
        dim=dim,
        mode=mode,
        source=str(cfg.get("source", "unspecified")),
        id_key=id_key,
        embedding_key=embedding_key,
        family=str(cfg.get("family", "control")),
        track=str(cfg.get("track", "native_frozen")),
        component=str(cfg.get("component", "locus_only")),
        cpg_namespace=str(cfg.get("cpg_namespace", "legacy_unspecified")),
        reference_build=str(cfg.get("reference_build", "unspecified")),
        coordinate_convention=str(cfg.get("coordinate_convention", "unspecified")),
    )
    manifest = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "name": info.name,
        "family": info.family,
        "track": info.track,
        "component": info.component,
        "cpg_namespace": info.cpg_namespace,
        "reference_build": info.reference_build,
        "coordinate_convention": info.coordinate_convention,
        "mode": info.mode,
        "source": info.source,
        "store_h5": str(info.store_path),
        "n_loci": info.n_loci,
        "embedding_dim": info.dim,
        "id_key": info.id_key,
        "embedding_key": info.embedding_key,
        "generator": cfg.get("generator"),
        "provenance": cfg.get("provenance", {}),
    }
    (run_dir / "representation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return info
