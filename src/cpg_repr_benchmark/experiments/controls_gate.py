"""Test authorisation for the 10 finished control runs, policy ``tcga_controls_freeze_v1`` (R15; constant, lookup, Gaussian, 7 x PCA-128; seed 17).

Data-free: reads file bytes (sha256) and the freeze manifest only. The TEST split of one control is unlocked only when the manifest is
``freeze_state: final`` with ``test_set_authorized: true``, the evaluation config is exactly the frozen one (canonical sha256), and
best.pt, resolved_config, patient/locus split, prior, store, the protocol files and the code on the evaluation path match their sha256.
Like ``external/zeroshot_gate.py`` (not ``test_confirmation_v2``), code is pinned by sha256 instead of a clean git tree, because the
benchmark working tree carries unrelated uncommitted files. Built by ``scripts/hal/controls_test_eval.py freeze`` and installed per process by its ``eval`` subcommand (eval_layout.py is not edited: its
sha256 is pinned by the GSE90496 freeze manifest).
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cpg_repr_benchmark.experiments.test_confirmation_v2 import canonical_cfg_sha256, sha256_file

POLICY = "tcga_controls_freeze_v1"
MANIFEST_REL = "configs/experiments/test_controls_v1/freeze_manifest.json"


def verify(cfg: dict[str, Any], root: Path | None = None, manifest_path: Path | None = None) -> list[str]:
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    mp = Path(manifest_path or root / MANIFEST_REL)
    if not mp.is_file():
        return [f"freeze manifest {mp} missing"]
    m, ev = json.loads(mp.read_text()), cfg.get("evaluation", {})
    problems = []
    if m.get("policy") != POLICY:
        problems.append(f"manifest policy {m.get('policy')!r} != {POLICY!r}")
    if not (m.get("freeze_state") == "final" and m.get("test_set_authorized") is True):
        problems.append(f"manifest not final/authorized ({m.get('freeze_state')!r}, {m.get('test_set_authorized')!r})")
    if (ev.get("patient_view") != "test" or ev.get("output_layout") != "split_dirs" or "require_patient_view" in ev
            or ev.get("allow_overwrite_split_dir", False)):
        problems.append("config must have patient_view=test, split_dirs, no require_patient_view, no allow_overwrite_split_dir")
    rid = ev.get("test_freeze_run_id")
    entry = m.get("runs", {}).get(rid)
    if entry is None:
        return problems + [f"run id {rid!r} not in the freeze manifest"]
    if canonical_cfg_sha256(cfg) != entry["eval_config_canonical_sha256"]:
        problems.append(f"{rid}: evaluation config differs from the frozen one")
    for rel, digest in {**entry["files"], **m["shared_files"]}.items():
        p = root / rel
        if not p.is_file():
            problems.append(f"missing {rel}")
        elif sha256_file(p) != digest:
            problems.append(f"sha256 mismatch for {rel}")
    return problems


def authorizer_from_cfg(cfg: dict[str, Any]) -> Callable[[], None]:
    def _authorize() -> None:
        bad = verify(cfg)
        if bad:
            raise PermissionError(f"GUARD ({POLICY}): " + "; ".join(bad[:8]) + " -- the TEST split stays locked")
    return _authorize
