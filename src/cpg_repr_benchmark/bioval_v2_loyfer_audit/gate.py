"""Pre-run gate, guarded writer, integrity snapshots and the step-1 dependency of the Loyfer failure audit.
Imports (never edits) the launch and follow-up gates."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from cpg_repr_benchmark.bioval_v2_followup import gate as fg
from cpg_repr_benchmark.bioval_v2_followup import registry as frg
from cpg_repr_benchmark.bioval_v2_launch import launch_gate as lg
from cpg_repr_benchmark.bioval_v2_launch.launch_gate import Check

from . import registry as rg


def audit_root(root) -> Path:
    return (Path(root) / rg.AUDIT_OUT).resolve()


def tag_dir(root) -> Path:
    return (Path(root) / rg.TAG_DIR).resolve()


def _under(p: Path, base: Path) -> bool:
    return p == base or base in p.parents


def is_frozen_audit_path(root, path) -> bool:
    """True for anything the audit must never write: the whole frozen result tree (including its sensitivity/ and
    exploratory_posthoc_loyfer/ sub-trees, which are READ-ONLY inputs here), the frozen docs, and every lg-protected path."""
    p = Path(path).resolve()
    if lg.is_protected(root, p) or _under(p, tag_dir(root)):
        return True
    docs = Path(root).resolve() / "docs"
    return _under(p, docs) and p.name.startswith("BIOLOGICAL_VALIDATION_V2")


class AuditWriter(lg.GuardedWriter):
    """Writes only below outputs/biological_validation_v2/loyfer_failure_audit; refuses every frozen/protected path."""

    def __init__(self, root, out_root):
        super().__init__(root, out_root)
        if self.out_root != audit_root(root) and audit_root(root) not in self.out_root.parents:
            raise PermissionError(f"output root {self.out_root} is not the registered audit root {audit_root(root)}")
        if is_frozen_audit_path(self.root, self.out_root):
            raise PermissionError(f"output root {self.out_root} is a frozen/protected path")

    def path(self, *parts) -> Path:
        p = self.out_root.joinpath(*parts).resolve()
        if is_frozen_audit_path(self.root, p):
            raise PermissionError(f"refusing to write into a frozen/protected path: {p}")
        return super().path(*parts)

    def write_csv(self, rel, df):
        df.to_csv(self.path(rel), index=False)

    def save_npy(self, rel, arr):
        import numpy as np
        np.save(self.path(rel), arr)


# ------------------------------------------------------------------------------------------------ snapshots
def snapshot_tree(root) -> dict:
    """sha256 of EVERY file under the frozen tag directory (arms: endpoints/results/manifests, contrasts, report, sensitivity,
    exploratory_posthoc_loyfer)."""
    base = tag_dir(root)
    files = {}
    for f in sorted(base.rglob("*")):
        if f.is_file():
            files[f.relative_to(base).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
    h = hashlib.sha256()
    for k, v in files.items():
        h.update(f"{k}\0{v}\n".encode())
    return {"n_files": len(files), "tree_sha256": h.hexdigest(), "files": files,
            "primary_fingerprint": fg.frozen_results_fingerprint(root), "timestamp_utc": datetime.now(timezone.utc).isoformat()}


def compare_snapshots(before: dict, after: dict) -> dict:
    a, b = before["files"], after["files"]
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    return {"identical": not (added or removed or changed), "added": added, "removed": removed, "changed": changed,
            "primary_fingerprint_before": before["primary_fingerprint"], "primary_fingerprint_after": after["primary_fingerprint"],
            "primary_fingerprint_unchanged": before["primary_fingerprint"] == after["primary_fingerprint"] == rg.FROZEN_RESULTS_SHA256}


def fingerprint_excludes_audit(root) -> bool:
    """The follow-up fingerprint hashes only the tag directory: the audit root is a sibling and must not be inside it."""
    return not _under(audit_root(root), tag_dir(root))


# ------------------------------------------------------------------------------------------------ gate
def _clean(root, rels, name):
    bad = lg._clean_committed(root, rels)
    return Check(name, not bad, "; ".join(bad) or "tracked and unmodified")


def check_out_root(root, out_root) -> Check:
    got = Path(out_root).resolve()
    ok = (got == audit_root(root) or audit_root(root) in got.parents) and not is_frozen_audit_path(root, got)
    return Check("out_root_is_audit_root", ok, f"{got}" + ("" if ok else f" (registered: {audit_root(root)})"))


def check_snapshot_before(root) -> Check:
    p = audit_root(root) / rg.SNAPSHOT_BEFORE
    if not p.is_file():
        return Check("snapshot_before_recorded", False, f"{p} missing: run `final-integrity --snapshot` before the first real step")
    try:
        before = json.loads(p.read_text())
        ok = before["primary_fingerprint"] == rg.FROZEN_RESULTS_SHA256
    except Exception as e:  # noqa: BLE001
        return Check("snapshot_before_recorded", False, f"unreadable: {e}")
    return Check("snapshot_before_recorded", ok, f"{before['n_files']} files, tree {before['tree_sha256'][:12]}")


def current_code_sha256(root) -> str:
    h = hashlib.sha256()
    for rel in rg.CODE_FILES:
        f = Path(root) / rel
        h.update(rel.encode() + b"\0" + (f.read_bytes() if f.exists() else b"") + b"\n")
    return h.hexdigest()


def check_step1_pass(root) -> Check:
    p = audit_root(root) / rg.VERDICT_FILE
    if not p.is_file():
        return Check("step1_target_audit_PASS", False, f"{p} missing: run `target-audit` first (blocks every other step)")
    v = json.loads(p.read_text())
    if v.get("verdict") not in rg.ACCEPTED_VERDICTS:
        return Check("step1_target_audit_PASS", False, f"verdict {v.get('verdict')} (accepted: {rg.ACCEPTED_VERDICTS}); failed {v.get('failed_checks')}")
    if v.get("level") != rg.VERDICT_LEVEL or v.get("raw_level_B_executed") is not False:
        return Check("step1_target_audit_PASS", False, "verdict is not a registered Level-A-only verdict (level/raw_level_B_executed mismatch)")
    reg = Path(root) / rg.REGISTRATION_DOC
    if v.get("registration_sha256") != hashlib.sha256(reg.read_bytes()).hexdigest():
        return Check("step1_target_audit_PASS", False, "registration changed since the verdict")
    if v.get("code_sha256") != current_code_sha256(root):
        return Check("step1_target_audit_PASS", False, "audit code changed since the verdict (re-run target-audit)")
    return Check("step1_target_audit_PASS", True, f"verdict {v['verdict']} (Level A only; raw .pat counts NOT verified) written {v.get('timestamp_utc')}")


def run_gate(root, arms, out_root, step, *, input_paths=(), skip_heavy=False, level_b=False):
    root = Path(root)
    checks = [
        lg.check_tag(root),
        _clean(root, [rg.REGISTRATION_DOC], "audit_registration_committed_clean"),
        _clean(root, list(rg.CODE_FILES), "audit_code_committed_clean"),
        lg.check_registration_committed(root),
        lg.check_code_committed(root),
        _clean(root, [frg.FOLLOWUP_REGISTRATION_DOC], "followup_registration_committed_clean"),
        _clean(root, list(frg.FOLLOWUP_CODE_FILES), "followup_code_committed_clean"),
        lg.check_frozen_unchanged(root),
        check_out_root(root, out_root),
        lg.check_inputs_not_tcga(input_paths),
        fg.check_frozen_results(root),
        Check("level_b_not_requested", not level_b, "registered runs are Level A only (amendment 1); the raw .pat Level B flag is refused" if level_b else "Level B not requested"),
        Check("fingerprint_does_not_cover_audit_root", fingerprint_excludes_audit(root), str(audit_root(root))),
    ]
    if skip_heavy:
        for n in ("frozen_manifest_checksums", "store_sha256"):
            checks.append(Check(n, False, "skipped (--skip-heavy; never allowed for a real run)"))
    else:
        checks.append(lg.check_manifest(root)[0])
        checks.append(lg.check_stores(root, list(arms)))
    if step not in ("final-integrity",):
        checks.append(check_snapshot_before(root))
    if step in rg.GATED_AFTER_STEP1:
        checks.append(check_step1_pass(root))
    return checks


def enforce(checks):
    lg.enforce(checks)


def head_commit(root) -> str:
    return lg.head_commit(root)


def build_manifest(root, step, *, threads, command, checks, seeds: dict, extra: dict | None = None) -> dict:
    root = Path(root)
    reg = root / rg.REGISTRATION_DOC
    m = {
        "step": step, "exploratory_posthoc": True,
        "label": "EXPLORATORY POST-HOC (Loyfer primary failure audit); does not replace or amend the frozen primary",
        "head_commit": head_commit(root), "protocol_tag": rg.PROTOCOL_TAG,
        "protocol_commit": lg._git(root, "rev-parse", f"refs/tags/{rg.PROTOCOL_TAG}^{{commit}}")[1],
        "registration_doc": {"path": rg.REGISTRATION_DOC, "sha256": hashlib.sha256(reg.read_bytes()).hexdigest() if reg.exists() else None,
                             "last_commit": lg._git(root, "log", "-1", "--format=%H", "--", rg.REGISTRATION_DOC)[1]},
        "code_sha256": current_code_sha256(root),
        "frozen_manifest_checksums_sha256": lg.hashlib.sha256((root / lg.BV / "MANIFEST_checksums.json").read_bytes()).hexdigest(),
        "frozen_primary_fingerprint_registered": rg.FROZEN_RESULTS_SHA256,
        "frozen_primary_fingerprint_now": fg.frozen_results_fingerprint(root),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "threads": threads,
        "library_versions": lg.library_versions(), "seeds": seeds,
        "gate_checks": [c.__dict__ for c in checks],
        "freeze_modules_imported": lg.freeze_modules_imported(),
        "no_freeze_command_used": not lg.freeze_modules_imported(),
        "tcga_test_set_referenced": False, "command": command, "out_root": str(audit_root(root)),
    }
    if extra:
        m.update(extra)
    return m
