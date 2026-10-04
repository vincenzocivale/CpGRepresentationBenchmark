"""Pre-run gate + guarded writer for the follow-up. Imports (never edits) bioval_v2_launch.launch_gate."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from cpg_repr_benchmark.bioval_v2_launch import launch_gate as lg
from cpg_repr_benchmark.bioval_v2_launch.launch_gate import Check

from . import registry as rg

# Aggregate sha256 over the frozen primary result tree (see frozen_results_fingerprint), registered before any follow-up run.
FROZEN_RESULTS_SHA256 = "38b9da12700a8fc83dc08f5a01aaa76b61f0d5097bffe395d8eb5225f1f02094"


def base_dir(root) -> Path:
    return (Path(root) / rg.BASE_OUT).resolve()


def is_frozen_result_path(root, path) -> bool:
    """True if ``path`` is inside the frozen primary-result tree (<base>/<arm>/..., <base>/contrasts, <base>/report, or
    <base> itself). Only <base>/sensitivity and <base>/exploratory_posthoc_loyfer are writable follow-up areas."""
    base = base_dir(root)
    p = Path(path).resolve()
    if base != p and base not in p.parents:
        return False
    rel = p.relative_to(base).parts
    return not rel or rel[0] not in ("sensitivity", "exploratory_posthoc_loyfer")


class FollowupWriter(lg.GuardedWriter):
    """GuardedWriter that additionally refuses the frozen primary result directories."""

    def __init__(self, root, out_root):
        super().__init__(root, out_root)
        if is_frozen_result_path(self.root, self.out_root):
            raise PermissionError(f"output root {self.out_root} is a frozen primary result path")

    def path(self, *parts) -> Path:
        p = self.out_root.joinpath(*parts).resolve()
        if is_frozen_result_path(self.root, p):
            raise PermissionError(f"refusing to write into frozen primary results: {p}")
        return super().path(*parts)


def frozen_results_fingerprint(root) -> str:
    """sha256 over sorted (relative path, file sha256) of the frozen primary results (arm dirs minus follow-up dirs,
    contrasts, report)."""
    base = base_dir(root)
    items = []
    for f in sorted(base.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(base).as_posix()
        if rel.split("/")[0] in ("sensitivity", "exploratory_posthoc_loyfer"):
            continue
        items.append((rel, hashlib.sha256(f.read_bytes()).hexdigest()))
    h = hashlib.sha256()
    for rel, s in items:
        h.update(f"{rel}\0{s}\n".encode())
    return h.hexdigest()


def check_frozen_results(root) -> Check:
    got = frozen_results_fingerprint(root)
    return Check("frozen_primary_results_unchanged", got == FROZEN_RESULTS_SHA256,
                 f"fingerprint {got}" + ("" if got == FROZEN_RESULTS_SHA256 else f" != registered {FROZEN_RESULTS_SHA256}"))


def _clean(root, rels, name):
    bad = lg._clean_committed(root, rels)
    return Check(name, not bad, "; ".join(bad) or "tracked and unmodified")


def check_out_root(root, out_root, subcommand) -> Check:
    want = (Path(root) / (rg.SENS_OUT if subcommand == "sensitivity" else rg.EXPL_OUT)).resolve()
    got = Path(out_root).resolve()
    if lg.is_protected(root, got):
        return Check("out_root_registered", False, f"{got} inside a protected path")
    if is_frozen_result_path(root, got):
        return Check("out_root_registered", False, f"{got} is a frozen primary result path")
    return Check("out_root_registered", got == want, f"{got}" + ("" if got == want else f" (registered: {want})"))


def run_gate(root, arms, out_root, subcommand, *, input_paths=(), skip_heavy=False):
    root = Path(root)
    checks = [
        lg.check_tag(root),
        _clean(root, [rg.FOLLOWUP_REGISTRATION_DOC], "followup_registration_committed_clean"),
        _clean(root, list(rg.FOLLOWUP_CODE_FILES), "followup_code_committed_clean"),
        lg.check_registration_committed(root),
        lg.check_code_committed(root),
        lg.check_frozen_unchanged(root),
        check_out_root(root, out_root, subcommand),
        lg.check_inputs_not_tcga(input_paths),
        check_frozen_results(root),
    ]
    if skip_heavy:
        for n in ("frozen_manifest_checksums", "store_sha256"):
            checks.append(Check(n, False, "skipped (--skip-heavy; never allowed for a real run)"))
    else:
        checks.append(lg.check_manifest(root)[0])
        checks.append(lg.check_stores(root, arms))
    return checks


def head_commit(root) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
