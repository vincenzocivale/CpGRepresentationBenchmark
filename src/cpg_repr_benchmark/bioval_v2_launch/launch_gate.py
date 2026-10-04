"""Hard pre-run gate, guarded output writer and run-manifest for the bioval v2 launch (NEW module).

The gate FAILS (SystemExit in real runs) unless:
  1. git tag bioval-v2-protocol-freeze-v1 exists and peels to commit 27d8bce...
  2. docs/BIOLOGICAL_VALIDATION_V2_LAUNCH_REGISTRATION.md is committed and has a clean git state (tracked, unmodified);
  3. the implementation files of this launch are committed and clean (the recorded HEAD describes the code that runs);
  4. no pre-existing frozen file (protocol/config/prep docs/data manifests/modules/freeze scripts) differs from the freeze
     commit (only ADDED files are tolerated);
  5. MANIFEST_checksums.json verifies (sha256 + size of every file, no extra file) via scripts/bioval_v2/preexec_audit.py;
  6. the store sha256 of every requested arm matches matrix.yaml;
  7. the output root is outside every protected path (data/derived/bioval_v2, configs, src, docs/bioval_v2_prep);
  8. no TCGA test-set / masking-protocol path is among the input paths.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROTOCOL_TAG = "bioval-v2-protocol-freeze-v1"
PROTOCOL_COMMIT_PREFIX = "27d8bce"
PROTOCOL_COMMIT = "27d8bce1b2ce391cb4ffb0d89eb15660710c2b33"
REGISTRATION_DOC = "docs/BIOLOGICAL_VALIDATION_V2_LAUNCH_REGISTRATION.md"
PROTOCOL_DOC = "docs/BIOLOGICAL_VALIDATION_V2.md"
BV = "data/derived/bioval_v2"
CODE_FILES = (
    "src/cpg_repr_benchmark/bioval_v2_launch/block_bootstrap.py",
    "src/cpg_repr_benchmark/bioval_v2_launch/embedding_eval.py",
    "src/cpg_repr_benchmark/bioval_v2_launch/chrom_probes.py",
    "src/cpg_repr_benchmark/bioval_v2_launch/launch_endpoints.py",
    "src/cpg_repr_benchmark/bioval_v2_launch/launch_gate.py",
    "src/cpg_repr_benchmark/bioval_v2_launch/__init__.py",
    "scripts/bioval_v2/run_evaluation.py",
    "tests/test_bioval_v2_launch.py",
    REGISTRATION_DOC,
)
FROZEN_DIRS = (
    "docs/bioval_v2_prep", "configs/biological_validation_v2", BV,
    "src/cpg_repr_benchmark/biological_validation_v2", "scripts/bioval_v2",
    "scripts/build_bioval_v2_covariates.py", "scripts/fetch_cpgislandext_hg38.py",
)
PROTECTED_REL = (BV, "configs", "src", "docs/bioval_v2_prep", "docs/BIOLOGICAL_VALIDATION_V2.md")
FREEZE_COMMAND_SCRIPTS = (
    "scripts/bioval_v2/freeze_4dn_pairs.py", "scripts/bioval_v2/freeze_fantom5_matching.py",
    "scripts/bioval_v2/loyfer_freeze_pairs.py", "scripts/bioval_v2/loyfer_profile_similarity.py",
    "scripts/bioval_v2/build_checksum_manifest.py", "scripts/bioval_v2/prepare_4dn.py",
    "scripts/bioval_v2/prepare_fantom5.py", "scripts/bioval_v2/prepare_loyfer.py",
    "scripts/bioval_v2/prepare_pmd_decato2020.py", "scripts/bioval_v2/prepare_replication_timing.py",
    "scripts/build_bioval_v2_covariates.py", "scripts/fetch_cpgislandext_hg38.py",
)
FREEZE_MODULE_NAMES = tuple(Path(p).stem for p in FREEZE_COMMAND_SCRIPTS)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def _git(root, *args, check=False):
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.returncode, r.stdout.strip()


def head_commit(root) -> str:
    return _git(root, "rev-parse", "HEAD")[1]


def check_tag(root) -> Check:
    rc, out = _git(root, "rev-parse", "--verify", "--quiet", f"refs/tags/{PROTOCOL_TAG}^{{commit}}")
    if rc:
        return Check("protocol_tag", False, f"tag {PROTOCOL_TAG} not found")
    ok = out.startswith(PROTOCOL_COMMIT_PREFIX) and out == PROTOCOL_COMMIT
    return Check("protocol_tag", ok, f"{PROTOCOL_TAG} -> {out}" + ("" if ok else f" (expected {PROTOCOL_COMMIT})"))


def _clean_committed(root, rels) -> list[str]:
    bad = []
    for rel in rels:
        if not (Path(root) / rel).exists():
            bad.append(f"{rel}: missing")
            continue
        if _git(root, "ls-files", "--error-unmatch", rel)[0]:
            bad.append(f"{rel}: not tracked / not committed")
            continue
        if _git(root, "status", "--porcelain", "--", rel)[1]:
            bad.append(f"{rel}: modified relative to HEAD")
    return bad


def check_registration_committed(root) -> Check:
    bad = _clean_committed(root, [REGISTRATION_DOC])
    if bad:
        return Check("registration_doc_committed_clean", False, "; ".join(bad))
    _, c = _git(root, "log", "-1", "--format=%H", "--", REGISTRATION_DOC)
    return Check("registration_doc_committed_clean", True, f"last commit touching it: {c}")


def check_code_committed(root) -> Check:
    bad = _clean_committed(root, [p for p in CODE_FILES if p != REGISTRATION_DOC])
    return Check("launch_code_committed_clean", not bad, "; ".join(bad) or "all launch files tracked and unmodified")


def check_frozen_unchanged(root) -> Check:
    problems = []
    rc, out = _git(root, "diff", "--name-status", PROTOCOL_COMMIT, "HEAD", "--", *FROZEN_DIRS)
    if rc:
        return Check("frozen_paths_unchanged", False, f"cannot diff against freeze commit {PROTOCOL_COMMIT}")
    for line in out.splitlines():
        st, *paths = line.split("\t")
        if st[0] != "A":
            problems.append(f"{st} {' '.join(paths)} (vs freeze commit)")
    rc, out = _git(root, "status", "--porcelain", "--", *FROZEN_DIRS)
    for line in out.splitlines():
        if not line.startswith("??"):
            problems.append(f"worktree: {line}")
    return Check("frozen_paths_unchanged", not problems, "; ".join(problems) or "no frozen file differs from freeze commit (added files only)")


def _load_preexec(root):
    spec = importlib.util.spec_from_file_location("bioval_preexec_audit", Path(root) / "scripts/bioval_v2/preexec_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_manifest(root) -> tuple[Check, str | None]:
    pa = _load_preexec(root)
    mpath = Path(root) / BV / "MANIFEST_checksums.json"
    man = json.loads(mpath.read_text())
    r = pa.verify_checksum_manifest(Path(root) / BV, man)
    ok = r["ok"] == r["n"] and not r["missing"] and not r["size_mismatch"] and not r["sha_mismatch"] and not r["extra_on_disk"]
    sha = hashlib.sha256(mpath.read_bytes()).hexdigest()
    return Check("frozen_manifest_checksums", ok, f"{r['ok']}/{r['n']} files ok; missing={len(r['missing'])} "
                 f"size_mismatch={len(r['size_mismatch'])} sha_mismatch={len(r['sha_mismatch'])} extra={len(r['extra_on_disk'])}"), sha


def check_stores(root, arms, specs=None) -> Check:
    from .embedding_eval import load_arm_specs, sha256_file
    specs = specs or load_arm_specs(Path(root), arms)
    bad, ok_ = [], []
    for a in arms:
        p = Path(root) / specs[a].store_h5
        if not p.is_file():
            bad.append(f"{a}: store missing {p}")
            continue
        got = sha256_file(p)
        (ok_ if got == specs[a].sha256 else bad).append(a if got == specs[a].sha256 else f"{a}: sha256 {got} != {specs[a].sha256}")
    return Check("store_sha256", not bad, "; ".join(bad) or f"{len(ok_)} stores match matrix.yaml")


def is_protected(root, path) -> bool:
    p = Path(path).resolve()
    for rel in PROTECTED_REL:
        q = (Path(root) / rel).resolve()
        if p == q or q in p.parents:
            return True
    return False


def check_out_root(root, out_root) -> Check:
    if is_protected(root, out_root):
        return Check("out_root_not_protected", False, f"{out_root} is inside a protected path")
    return Check("out_root_not_protected", True, str(Path(out_root).resolve()))


def check_inputs_not_tcga(paths) -> Check:
    from .embedding_eval import FORBIDDEN_READ_SUBSTRINGS
    bad = [str(p) for p in paths if any(s in str(p).replace("\\", "/").lower() for s in FORBIDDEN_READ_SUBSTRINGS)]
    return Check("no_tcga_or_protocol_input", not bad, "; ".join(bad) or "no input path matches a forbidden pattern")


def freeze_modules_imported() -> list[str]:
    return sorted(m for m in sys.modules if m.split(".")[-1] in FREEZE_MODULE_NAMES or m in FREEZE_MODULE_NAMES)


def run_gate(root, arms, out_root, *, input_paths=(), skip_heavy=False) -> list[Check]:
    root = Path(root)
    checks = [check_tag(root), check_registration_committed(root), check_code_committed(root),
              check_frozen_unchanged(root), check_out_root(root, out_root), check_inputs_not_tcga(input_paths)]
    if skip_heavy:
        checks.append(Check("frozen_manifest_checksums", False, "skipped (--skip-heavy; never allowed for a real run)"))
        checks.append(Check("store_sha256", False, "skipped (--skip-heavy; never allowed for a real run)"))
    else:
        checks.append(check_manifest(root)[0])
        checks.append(check_stores(root, arms))
    return checks


def format_checks(checks) -> str:
    return "\n".join(f"  [{'PASS' if c.ok else 'FAIL'}] {c.name}: {c.detail}" for c in checks)


def enforce(checks) -> None:
    failed = [c for c in checks if not c.ok]
    if failed:
        raise SystemExit("PRE-RUN GATE FAILED:\n" + format_checks(failed))


class GuardedWriter:
    """All outputs go through here: only below ``out_root``, never below a protected path."""

    def __init__(self, root, out_root):
        self.root = Path(root).resolve()
        self.out_root = Path(out_root).resolve()
        if is_protected(self.root, self.out_root):
            raise PermissionError(f"output root {self.out_root} is inside a protected path")

    def path(self, *parts) -> Path:
        p = self.out_root.joinpath(*parts).resolve()
        if self.out_root != p and self.out_root not in p.parents:
            raise PermissionError(f"{p} escapes the output root")
        if is_protected(self.root, p):
            raise PermissionError(f"refusing to write inside protected path: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_text(self, rel, text):
        self.path(rel).write_text(text)

    def write_json(self, rel, obj):
        self.write_text(rel, json.dumps(obj, indent=1, default=_json_default))

    def save_npz(self, rel, **arrays):
        import numpy as np
        np.savez_compressed(self.path(rel), **arrays)


def _json_default(o):
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def library_versions() -> dict:
    out = {"python": platform.python_version()}
    for m in ("numpy", "scipy", "pandas", "sklearn", "h5py", "pyarrow", "yaml"):
        try:
            out[m] = __import__(m).__version__
        except (ImportError, AttributeError):  # pragma: no cover
            out[m] = None
    return out


def build_run_manifest(root, arm_id, identity, *, seed, n_boot, boot_fingerprint, threads, exclusion_counts,
                       checks, endpoints, command, manifest_checksums_sha256=None, out_root=None) -> dict:
    root = Path(root)
    reg = root / REGISTRATION_DOC
    return {
        "arm_id": arm_id,
        "head_commit": head_commit(root),
        "protocol_tag": PROTOCOL_TAG,
        "protocol_commit": _git(root, "rev-parse", f"refs/tags/{PROTOCOL_TAG}^{{commit}}")[1],
        "registration_doc": {"path": REGISTRATION_DOC,
                             "sha256": hashlib.sha256(reg.read_bytes()).hexdigest() if reg.exists() else None,
                             "last_commit": _git(root, "log", "-1", "--format=%H", "--", REGISTRATION_DOC)[1]},
        "frozen_manifest_checksums_sha256": manifest_checksums_sha256,
        "representation": identity,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "n_bootstrap": n_boot, "bootstrap_draw_table_sha256": boot_fingerprint,
        "threads": threads, "library_versions": library_versions(),
        "exclusion_counts": exclusion_counts,
        "endpoints_requested": endpoints,
        "gate_checks": [c.__dict__ for c in checks],
        "freeze_modules_imported": freeze_modules_imported(),
        "no_freeze_command_used": not freeze_modules_imported(),
        "tcga_test_set_referenced": False,
        "command": command,
        "out_root": str(out_root) if out_root else None,
    }
