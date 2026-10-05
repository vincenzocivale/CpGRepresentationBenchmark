"""Pre-run gate of the external GSE40279 confirmation (protocol v1.1). Torch-free; reads metadata and file bytes only.

Never reads methylation values or any test patient data. Every check returns a `Check`; `run_gate` composes them and
`require_green` raises `GateRefused` (the CLI exits non-zero). See docs/EXTERNAL_RECONSTRUCTION_RUNNER.md.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .budget import EXTERNAL_EPOCHS, epoch_budget
from .freeze import MANIFEST_REL, PROTOCOL_REL
from .io import sha256_file

TAG_V1 = "external-recon-protocol-freeze-v1"
TAG_V1_1 = "external-recon-protocol-freeze-v1.1"
AMENDMENT_COMMIT_PREFIX = "da02b78"
OUTPUT_ROOT_REL = "outputs/external_reconstruction_v1"
# The output root must be inside OUTPUT_ROOT_REL and never inside / equal to one of these (GuardedWriter-like refusal).
FROZEN_PATHS = (
    "data/derived/external_gse40279_v1", "configs/external", "configs/frozen", "data/cache/representations",
    "outputs/encode_atlas_v1", "outputs/regulatory_confirmation_v1", "outputs/biological_validation_v2",
    "outputs/external_reconstruction_v1/audit", "docs",
)
# Files unchanged since the protocol tag (frozen artifacts that live in git).
FROZEN_TRACKED = (PROTOCOL_REL, MANIFEST_REL, "configs/external", "configs/frozen",
                  "configs/experiments/regulatory_confirmation_matrix")
# Implementation files that must be committed AND clean before launch (the launch gate refuses otherwise).
REQUIRED_COMMITTED = (
    "src/cpg_repr_benchmark/external/gate.py",
    "src/cpg_repr_benchmark/external/runner.py",
    "src/cpg_repr_benchmark/external/transfer.py",
    "src/cpg_repr_benchmark/training/engine.py",
    "src/cpg_repr_benchmark/experiments/eval_layout.py",
    "scripts/run_masking_benchmark.py",
    "scripts/run_external_confirmation.py",
    "scripts/run_external_transfer.py",
    "scripts/analyze_external_confirmation.py",
    "configs/experiments/external_confirmation",
    "tests/test_external_runner.py",
    "tests/test_external_transfer.py",
    "tests/test_external_analysis.py",
    "tests/test_checkpoint_policy.py",
    "docs/EXTERNAL_RECONSTRUCTION_RUNNER.md",
)


class GateRefused(PermissionError):
    pass


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        return f"{'OK  ' if self.ok else 'FAIL'} {self.name} {self.detail}".rstrip()


def _load_manifest(repo: Path, manifest_path: Path | None = None) -> dict:
    return json.loads((Path(manifest_path) if manifest_path else Path(repo) / MANIFEST_REL).read_text())


# ----------------------------------------------------------------------------- manifest state / authorization
def check_state(manifest: dict, *, split: str) -> list[Check]:
    """freeze_state / test_set_authorized consistency and the test request rule."""
    state, auth = manifest.get("freeze_state"), manifest.get("test_set_authorized")
    out = [Check("freeze_state_valid", state in ("draft", "final"), str(state)),
           Check("test_authorized_is_bool", isinstance(auth, bool), str(auth)),
           # authorized without a final freeze is an inconsistent (unsafe) manifest, whatever split is requested
           Check("state_consistent", not (auth is True and state != "final"),
                 f"freeze_state={state!r} test_set_authorized={auth!r}")]
    if split == "test":
        out.append(Check("test_authorized_and_final", auth is True and state == "final",
                         f"test requested: needs test_set_authorized true and freeze_state final (got {auth!r}, {state!r})"))
    elif split == "validation":
        out.append(Check("validation_phase_test_locked", auth is False,
                         f"validation phase expects test_set_authorized false (got {auth!r})"))
    else:
        out.append(Check("split_known", False, f"unknown split {split!r}"))
    return out


def authorize_external_test(repo_root: Path | None = None, manifest_path: Path | None = None) -> None:
    """Hook for `evaluation.test_authorization: external_gse40279_v1`: raises unless authorized + final + all-green gate."""
    repo = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    manifest = _load_manifest(repo, manifest_path)
    bad = [c for c in check_state(manifest, split="test") if not c.ok]
    if bad:
        raise PermissionError("GUARD: " + "; ".join(c.line() for c in bad))
    bad = [c for c in run_gate(repo, split="test", scope="A", manifest_path=manifest_path) if not c.ok]
    if bad:
        raise PermissionError(f"GUARD: external gate has {len(bad)} failure(s); the TEST split stays locked")


# ----------------------------------------------------------------------------- hashes / coverage
def check_file_hashes(repo: Path, manifest: dict) -> list[Check]:
    repo = Path(repo)
    out = []
    pdoc = repo / manifest.get("protocol_doc", PROTOCOL_REL)
    got = sha256_file(pdoc) if pdoc.exists() else "missing"
    out.append(Check("protocol_doc_sha256", got == manifest.get("protocol_doc_sha256"), got[:16]))
    for key in ("patient_protocol_npz", "locus_protocol_npz", "dataset_h5", "mapping_manifest", "split_csv"):
        rel = manifest.get("files", {}).get(key)
        p = repo / rel if rel else None
        got = sha256_file(p) if p is not None and p.exists() else "missing"
        out.append(Check(f"sha256:{key}", got == manifest.get("files_sha256", {}).get(key), got[:16]))
    return out


def check_store_hashes(repo: Path, manifest: dict) -> list[Check]:
    out = []
    for arm, s in manifest.get("arms", {}).items():
        p = Path(repo) / s["store_h5"]
        got = sha256_file(p) if p.exists() else "missing"
        out.append(Check(f"store_sha256:{arm}", got == s["store_sha256"] == s["matrix_yaml_sha256"], got[:16]))
    return out


def check_checkpoint_hashes(repo: Path, manifest: dict) -> list[Check]:
    out = []
    for c in manifest.get("phase_a_checkpoints_experiment_B", []):
        p = Path(repo) / c["best_pt"]
        got = sha256_file(p) if p.exists() else "missing"
        out.append(Check(f"phaseA_best_pt:{c['arm']}:{c['seed']}", got == c["best_pt_sha256"], got[:16]))
    return out


def universe_ids(repo: Path, manifest: dict) -> np.ndarray:
    with np.load(Path(repo) / manifest["files"]["locus_protocol_npz"]) as z:
        return np.sort(np.concatenate([z["train_cpg_idx"], z["heldout_cpg_idx"]])).astype(np.int64)


def check_store_coverage(repo: Path, manifest: dict) -> list[Check]:
    """Every arm must cover 100% of the frozen locus universe (reads /cpg_idx only)."""
    uni = universe_ids(repo, manifest)
    out = []
    for arm, s in manifest.get("arms", {}).items():
        with h5py.File(Path(repo) / s["store_h5"], "r") as h:
            ids = np.asarray(h["cpg_idx"][:], dtype=np.int64)
            dim = int(h["embedding"].shape[1])
        cov = int(np.isin(uni, ids).sum())
        out.append(Check(f"coverage_100pct:{arm}", cov == len(uni) and dim == s["dim"],
                         f"{cov}/{len(uni)} dim={dim}"))
    return out


def check_universe(repo: Path, manifest: dict) -> list[Check]:
    uni = universe_ids(repo, manifest)
    with h5py.File(Path(repo) / manifest["files"]["dataset_h5"], "r") as h:
        ids = np.sort(np.asarray(h["cpg_idx"][:], dtype=np.int64))
    return [Check("universe_size", len(uni) == manifest["universe_size"], str(len(uni))),
            Check("universe_equals_matrix_axis", np.array_equal(uni, ids))]


def n_train_from_protocol(repo: Path, manifest: dict) -> int:
    with np.load(Path(repo) / manifest["files"]["patient_protocol_npz"]) as z:
        return len(z["train"])


# ----------------------------------------------------------------------------- configs
def check_config_budget(cfg: dict, n_train: int, where: str = "cfg") -> list[Check]:
    tr, ev = cfg["training"], cfg["evaluation"]
    spe = epoch_budget(n_train, int(tr["batch_size"]), int(tr["epochs"]))
    return [
        Check(f"{where}:epochs_120", tr.get("epochs") == EXTERNAL_EPOCHS, str(tr.get("epochs"))),
        Check(f"{where}:early_stopping_false", tr.get("early_stopping") is False, str(tr.get("early_stopping"))),
        Check(f"{where}:max_updates_absent_or_null", tr.get("max_updates", None) is None, str(tr.get("max_updates"))),
        Check(f"{where}:batch_size_8", tr.get("batch_size") == 8 and ev.get("batch_size") == 8, str(tr.get("batch_size"))),
        Check(f"{where}:updates_per_epoch_66", spe["updates_per_epoch"] == 66, f"ceil({n_train}/8)={spe['updates_per_epoch']}"),
        Check(f"{where}:total_updates_7920", spe["total_updates"] == 7920, str(spe["total_updates"])),
        Check(f"{where}:save_epoch_checkpoints_false", tr.get("save_epoch_checkpoints") is False,
              str(tr.get("save_epoch_checkpoints"))),
        Check(f"{where}:record_update_counts", tr.get("record_update_counts") is True, str(tr.get("record_update_counts"))),
    ]


def check_configs(repo: Path, manifest: dict, *, split: str = "validation") -> list[Check]:
    """12 per-run configs: exist, equal the freshly built config, satisfy the frozen budget, validation guards present."""
    import yaml

    from . import runner
    repo = Path(repo)
    n_train = n_train_from_protocol(repo, manifest)
    out, bad = [], []
    for arm, seed in runner.jobs():
        path = repo / runner.CONFIG_DIR / runner.config_name(arm, seed)
        if not path.is_file():
            bad.append(f"{path.name} missing")
            continue
        cfg = yaml.safe_load(path.read_text())
        keep_last = bool(cfg.get("training", {}).get("save_last_checkpoint", False))
        try:
            fresh = runner.build_config(arm, seed, repo, keep_last_checkpoint=keep_last)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{path.name}: cannot rebuild ({e!r})")
            continue
        if cfg != fresh:
            bad.append(f"{path.name} stale (differs from the generated config)")
            continue
        fails = [c for c in check_config_budget(cfg, n_train, path.name) if not c.ok]
        bad += [c.line() for c in fails]
        ev = cfg["evaluation"]
        if split == "validation" and not (ev.get("patient_view") == "validation" and ev.get("require_patient_view") == "validation"):
            bad.append(f"{path.name}: validation-phase config lacks the validation-only guard")
        if ev.get("output_layout") != "split_dirs" or ev.get("allow_overwrite_split_dir") is not False:
            bad.append(f"{path.name}: output layout must be split_dirs with allow_overwrite_split_dir false")
        out_bad = check_output_root(repo, cfg["experiment"]["output_root"])
        if not out_bad.ok:
            bad.append(f"{path.name}: {out_bad.detail}")
    out.append(Check("configs_12", not bad and len(runner.jobs()) == 12, "12 configs match the frozen protocol" if not bad else "; ".join(bad)))
    return out


def check_output_root(repo: Path, output_root: str | Path) -> Check:
    """The output root must lie inside outputs/external_reconstruction_v1 and outside every frozen path."""
    repo = Path(repo).resolve()
    p = Path(output_root)
    p = (p if p.is_absolute() else repo / p).resolve()
    allowed = (repo / OUTPUT_ROOT_REL).resolve()
    if p != allowed and allowed not in p.parents:
        return Check("output_root_safe", False, f"{p} is outside {OUTPUT_ROOT_REL}")
    for f in FROZEN_PATHS:
        fp = (repo / f).resolve()
        if p == fp or fp in p.parents:
            return Check("output_root_safe", False, f"{p} is inside the frozen path {f}")
    return Check("output_root_safe", True, str(p.relative_to(repo)))


def assert_writable(repo: Path, path: str | Path) -> Path:
    """GuardedWriter-style check for any path a tool is about to write (raises PermissionError inside frozen paths)."""
    repo = Path(repo).resolve()
    p = Path(path)
    p = (p if p.is_absolute() else repo / p).resolve()
    for f in FROZEN_PATHS:
        fp = (repo / f).resolve()
        if p == fp or fp in p.parents:
            raise PermissionError(f"GUARD: refusing to write inside the frozen path {f}: {p}")
    return p


# ----------------------------------------------------------------------------- git
def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)


def check_git(repo: Path, *, tag: str = TAG_V1_1, amendment_prefix: str = AMENDMENT_COMMIT_PREFIX,
              required: Iterable[str] = REQUIRED_COMMITTED, frozen_tracked: Iterable[str] = FROZEN_TRACKED,
              require_clean_code: bool = True) -> list[Check]:
    repo = Path(repo)
    out = []
    r = _git(repo, "rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}")
    tag_commit = r.stdout.strip() if r.returncode == 0 else None
    out.append(Check("protocol_tag_present", bool(tag_commit), f"{tag} -> {(tag_commit or 'missing')[:10]}"))
    out.append(Check("amendment_commit_is_tag", bool(tag_commit) and tag_commit.startswith(amendment_prefix),
                     f"tag commit {(tag_commit or '-')[:10]} vs amendment {amendment_prefix}"))
    anc = bool(tag_commit) and _git(repo, "merge-base", "--is-ancestor", tag_commit, "HEAD").returncode == 0
    out.append(Check("amendment_commit_ancestor_of_HEAD", anc))
    if tag_commit:
        changed = [p for p in frozen_tracked if _git(repo, "diff", "--quiet", tag_commit, "--", p).returncode != 0]
        dirty = [p for p in frozen_tracked if _git(repo, "status", "--porcelain", "--", p).stdout.strip()]
        out.append(Check("frozen_files_unchanged_since_tag", not changed and not dirty,
                         f"changed={changed} dirty={dirty}" if (changed or dirty) else "unchanged"))
    else:
        out.append(Check("frozen_files_unchanged_since_tag", False, "no tag"))
    if require_clean_code:
        untracked, dirty = [], []
        for p in required:
            tracked = _git(repo, "ls-files", "--", p).stdout.strip()
            if not tracked:
                untracked.append(p)
            elif _git(repo, "status", "--porcelain", "--", p).stdout.strip():
                dirty.append(p)
        out.append(Check("implementation_committed_clean", not untracked and not dirty,
                         "all implementation files committed and clean" if not (untracked or dirty)
                         else f"not committed: {untracked}; modified: {dirty}"))
    return out


# ----------------------------------------------------------------------------- validation-phase test protection
def check_no_test_artifacts(repo: Path, split: str = "validation") -> list[Check]:
    """Validation phase: no evaluation/test directory anywhere under the external output root."""
    root = Path(repo) / OUTPUT_ROOT_REL
    found = []
    if split == "validation" and root.is_dir():
        found = [str(p.relative_to(root)) for p in root.rglob("evaluation") if (p / "test").exists()]
    return [Check("no_evaluation_test_dir", not found, "none" if not found else f"found: {found[:5]}")]


# ----------------------------------------------------------------------------- composition
def run_gate(repo: Path, *, split: str = "validation", scope: str = "A", manifest_path: Path | None = None,
             verify_fn: Callable | None = None, git: bool = True, heavy: bool = True,
             amendment_prefix: str = AMENDMENT_COMMIT_PREFIX, required: Iterable[str] = REQUIRED_COMMITTED) -> list[Check]:
    """All gate checks. scope 'A' = Experiment A (adds the config checks); scope 'B' = transfer (no per-run configs)."""
    repo = Path(repo)
    try:
        manifest = _load_manifest(repo, manifest_path)
    except Exception as e:  # noqa: BLE001
        return [Check("manifest_readable", False, repr(e))]
    out = [Check("manifest_readable", True)]
    if scope not in ("A", "B"):
        return out + [Check("scope_known", False, scope)]
    out += check_state(manifest, split=split)
    out.append(Check("protocol_version_v1_1", manifest.get("protocol_version") == "v1.1", str(manifest.get("protocol_version"))))
    if verify_fn is None:
        from .freeze import verify as verify_fn
    res = verify_fn(repo, Path(manifest_path) if manifest_path else None)
    nbad = [n for n, ok, _ in res if not ok]
    out.append(Check("freeze_verify", not nbad, f"{len(res) - len(nbad)}/{len(res)} checks" + (f" FAILED: {nbad}" if nbad else "")))
    try:
        if heavy:
            out += check_file_hashes(repo, manifest)
            out += check_store_hashes(repo, manifest)
        out += check_universe(repo, manifest)
        out += check_store_coverage(repo, manifest)
        if scope == "B":
            out += check_checkpoint_hashes(repo, manifest)
        else:
            out += check_configs(repo, manifest, split=split)
    except Exception as e:  # noqa: BLE001
        out.append(Check("gate_inputs_readable", False, repr(e)))
    out += check_no_test_artifacts(repo, split)
    if git:
        out += check_git(repo, amendment_prefix=amendment_prefix, required=required)
    return out


def failures(checks: Iterable[Check]) -> list[Check]:
    return [c for c in checks if not c.ok]


def require_green(checks: Iterable[Check]) -> None:
    bad = failures(checks)
    if bad:
        raise GateRefused(f"{len(bad)} gate check(s) failed: " + "; ".join(c.name for c in bad))


def as_dict(checks: Iterable[Check]) -> dict[str, Any]:
    cs = list(checks)
    return {"all_green": all(c.ok for c in cs), "n_checks": len(cs), "n_failed": sum(not c.ok for c in cs),
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in cs]}
