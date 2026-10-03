"""Provenance manifest writer."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def git_commit(root=None):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class SourceRecord:
    name: str
    url: str = ""
    accession: str = ""
    retrieval_date: str = ""
    sha256: str = ""
    path: str = ""
    genome_build_in: str = ""
    genome_build_out: str = "GRCh38"
    liftover_chain: str = ""
    liftover_tool: str = ""
    n_rows_in: int | None = None
    n_rows_out: int | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_file(cls, path, **kw):
        return cls(path=str(path), sha256=sha256_file(path), **kw)


def write_manifest(path, axis, sources, universe_audit, params, code_version=None):
    """Write JSON manifest. ``universe_audit`` may contain DataFrames (converted to records)."""
    def clean(o):
        if hasattr(o, "to_dict"):
            return o.to_dict(orient="records") if hasattr(o, "columns") else o.to_dict()
        if hasattr(o, "item"):
            return o.item()
        return str(o)

    doc = {"axis": axis, "sources": [asdict(s) if isinstance(s, SourceRecord) else s for s in sources],
           "universe_audit": universe_audit, "params": params,
           "code_version": code_version or git_commit(), "git_commit": git_commit()}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, sort_keys=True, default=clean))
    return p
