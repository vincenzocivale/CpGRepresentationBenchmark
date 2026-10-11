"""Authorization policy ``tcga_posthoc_freeze_v2`` for the one-shot TEST evaluation of the post-hoc panel (8 arms x 3 seeds = 24 runs).

Pure Python, data-free: reads only file bytes (sha256) and the freeze manifest; never opens methylation values, patient rows or
predictions. The TEST split is unlocked ONLY when ALL hold: manifest ``test_set_authorized: true`` and ``freeze_state: final``;
the run's evaluation config is exactly the frozen one (canonical sha256); best.pt / resolved_config / store / patient_split /
locus_split / prior of every run still match the manifest; and the manifest plus the code on the evaluation path are committed
and clean in git. See docs/CONFIRMATORY_FREEZE_V2.md.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from cpg_repr_benchmark.experiments.guards import require_test_authorization

POLICY = "tcga_posthoc_freeze_v2"
MANIFEST_REL = "configs/experiments/test_confirmation_v2/freeze_manifest.yaml"
EVAL_CONFIG_DIR_REL = "configs/experiments/test_confirmation_v2"
PANEL_ARMS = ["regulatory_raw2492", "cpgpt_large_locus", "deepcpg_dna_locus", "alphagenome_locus",
              "enformer_latent_native", "ntv3_post_latent_native", "ntv3_pre_latent_native", "sei_latent_native"]
SEEDS = [17, 42, 97]
# evaluation keys that may differ between a run's resolved_config and its test-evaluation config (nothing else may)
EVAL_KEYS_ALLOWED_TO_DIFFER = {"patient_view", "require_patient_view", "test_authorization", "test_freeze_manifest",
                               "test_freeze_run_id", "preserve_run_files", "allow_overwrite_split_dir", "num_workers"}
# code on the evaluation path: must be committed and clean (no uncommitted diff) when the test is unlocked
REQUIRED_CLEAN = (
    MANIFEST_REL, EVAL_CONFIG_DIR_REL,
    "src/cpg_repr_benchmark/experiments/test_confirmation_v2.py", "src/cpg_repr_benchmark/experiments/eval_layout.py",
    "src/cpg_repr_benchmark/experiments/guards.py", "src/cpg_repr_benchmark/representations/resolver.py",
    "src/cpg_repr_benchmark/representations/hdf5_store.py", "src/cpg_repr_benchmark/training/engine.py",
    "src/cpg_repr_benchmark/data/splits.py", "src/cpg_repr_benchmark/data/chromosomes.py",
    "scripts/run_masking_benchmark.py", "scripts/run_test_confirmation_v2.py",
)


def run_id(arm: str, seed: int) -> str:
    return f"{arm}__seed{seed}"


def sha256_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def canonical_cfg_sha256(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def _resolve(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(root) / p


# file kinds hashed per run (key in manifest entry -> path key); store is hashed separately (large)
RUN_HASHED = (("best_pt", "best_sha256"), ("resolved_config", "resolved_config_sha256"), ("patient_split", "patient_split_sha256"),
              ("locus_split", "locus_split_sha256"), ("prior_logit", "prior_logit_sha256"))


def verify_manifest(manifest: dict[str, Any], root: Path, *, stores: str | list[str] = "all") -> list[str]:
    """Return a list of problems (empty = all hashes match). `stores`: 'all' | 'none' | list of arm ids to re-hash."""
    problems: list[str] = []
    runs = manifest.get("runs") or {}
    expected = {run_id(a, s) for a in PANEL_ARMS for s in SEEDS}
    if set(runs) != expected:
        problems.append(f"panel mismatch: missing={sorted(expected - set(runs))} extra={sorted(set(runs) - expected)}")
    store_cache: dict[str, str] = {}
    for rid, e in runs.items():
        if e.get("status", "complete") != "complete":
            problems.append(f"{rid}: status {e.get('status')!r} (run not complete/hashed yet)")
            continue
        for key, hkey in RUN_HASHED:
            p = _resolve(root, e[key])
            if not p.is_file():
                problems.append(f"{rid}: missing {key} {p}")
            elif sha256_file(p) != e[hkey]:
                problems.append(f"{rid}: sha256 mismatch for {key}")
        want_store = stores == "all" or (isinstance(stores, list) and e["arm"] in stores)
        if want_store:
            sp = _resolve(root, e["store_h5"])
            if not sp.is_file():
                problems.append(f"{rid}: missing store {sp}")
            else:
                digest = store_cache.setdefault(str(sp), sha256_file(sp))
                if digest != e["store_sha256"]:
                    problems.append(f"{rid}: sha256 mismatch for store")
        ev = _resolve(root, e["eval_config"])
        if not ev.is_file() or sha256_file(ev) != e["eval_config_sha256"]:
            problems.append(f"{rid}: eval config missing or modified")
    return problems


def git_problems(root: Path, paths: tuple[str, ...] = REQUIRED_CLEAN) -> list[str]:
    """Paths that are untracked, modified or staged relative to HEAD (freeze must be committed before the test)."""
    out: list[str] = []
    for p in paths:
        r = subprocess.run(["git", "status", "--porcelain", "--", p], cwd=root, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            out.append(f"git status failed for {p}")
        elif r.stdout.strip():
            out.append(f"{p}: not committed/clean")
        elif subprocess.run(["git", "ls-files", "--error-unmatch", "--", p], cwd=root, capture_output=True, check=False).returncode != 0:
            out.append(f"{p}: not tracked")
    return out


def authorize(cfg: dict[str, Any], root: Path, *, manifest_path: Path | None = None,
              git_check: Callable[[Path], list[str]] = git_problems, stores: str | list[str] | None = None) -> None:
    """Raise PermissionError unless the TEST split of this run is authorized by the v2 freeze manifest."""
    root = Path(root)
    ev = cfg.get("evaluation", {})
    manifest = load_manifest(manifest_path or _resolve(root, ev.get("test_freeze_manifest", MANIFEST_REL)))
    require_test_authorization(manifest)  # test_set_authorized is True AND freeze_state == 'final'
    rid = ev.get("test_freeze_run_id")
    entry = (manifest.get("runs") or {}).get(rid)
    if entry is None:
        raise PermissionError(f"GUARD: run id {rid!r} is not in the frozen panel")
    if ev.get("patient_view") != "test" or ev.get("output_layout") != "split_dirs":
        raise PermissionError("GUARD: v2 test evaluation requires patient_view=test and output_layout=split_dirs")
    if "require_patient_view" in ev or ev.get("allow_overwrite_split_dir", False):
        raise PermissionError("GUARD: v2 test config must not carry require_patient_view nor allow_overwrite_split_dir")
    if canonical_cfg_sha256(cfg) != entry["eval_config_canonical_sha256"]:
        raise PermissionError(f"GUARD: evaluation config of {rid} differs from the frozen one")
    problems = verify_manifest(manifest, root, stores=stores if stores is not None else [entry["arm"]])
    problems += git_check(root)
    if problems:
        raise PermissionError("GUARD: freeze verification failed; the TEST split stays locked: " + "; ".join(problems[:8]))


def authorizer_from_cfg_v2(cfg: dict[str, Any]) -> Callable[[], None]:
    root = Path(__file__).resolve().parents[3]
    return lambda: authorize(cfg, root)
