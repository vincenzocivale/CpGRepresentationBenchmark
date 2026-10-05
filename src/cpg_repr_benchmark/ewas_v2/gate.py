"""Pre-run gate, guarded writer and state machine for EWAS-v2 (imports, never edits, bioval_v2_launch.launch_gate).

State machine: DRAFT -> DESIGN_ARTIFACTS_BUILT -> FROZEN -> EXECUTED -> REPORTED. FROZEN is *derived*: registration doc committed
clean + tag exists + design files identical to the tag. EXECUTED/REPORTED live in <out_root>/state.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from cpg_repr_benchmark.bioval_v2_launch import launch_gate as lg
from cpg_repr_benchmark.bioval_v2_launch.embedding_eval import (
    FORBIDDEN_READ_SUBSTRINGS,
    load_arm_specs,
    sha256_file,
)
from cpg_repr_benchmark.bioval_v2_launch.launch_gate import Check, _git

from . import design as D

TAG = "ewas-v2-protocol-freeze-v1"
REGISTRATION_DOC = "docs/EWAS_V2_REGISTRATION.md"
SPEC = "configs/ewas_v2/ewas_v2_spec_v1.yaml"
OUT_BASE = "outputs/ewas_v2"
FORBIDDEN_PATH_PARTS = FORBIDDEN_READ_SUBSTRINGS + ("loyfer_failure_audit",)
MAIN_ARMS = ("regulatory_histone_dnase_v1", "cpgpt_large_locus", "deepcpg_dna_locus")
SUPP_ARMS = ("functional_annotations_pca",)
ALL_ARMS = MAIN_ARMS + SUPP_ARMS
# design-defining files: must be byte-identical between the freeze tag and HEAD (a change = declared amendment, never silent)
DESIGN_FILES = (REGISTRATION_DOC, SPEC, D.MANIFEST)
# code files: committed and clean (code changes after the tag are listed in the run manifest, see code_changes_since_tag)
CODE_FILES = (
    "src/cpg_repr_benchmark/ewas_v2/__init__.py", "src/cpg_repr_benchmark/ewas_v2/design.py",
    "src/cpg_repr_benchmark/ewas_v2/metrics.py", "src/cpg_repr_benchmark/ewas_v2/probe.py",
    "src/cpg_repr_benchmark/ewas_v2/gate.py", "src/cpg_repr_benchmark/ewas_v2/runner.py",
    "scripts/ewas_v2/run_ewas_v2.py", "scripts/ewas_v2/build_design_artifacts.py",
    "tests/test_ewas_v2_package.py",
)


def _ok(root, rels):
    bad = lg._clean_committed(root, list(rels))
    return bad


def check_registration(root) -> Check:
    bad = _ok(root, [REGISTRATION_DOC])
    return Check("registration_committed_clean", not bad, "; ".join(bad) or "tracked and unmodified")


def check_code(root) -> Check:
    bad = _ok(root, CODE_FILES)
    return Check("code_committed_clean", not bad, "; ".join(bad) or "all EWAS-v2 code files tracked and unmodified")


def check_tag(root) -> Check:
    rc, out = _git(root, "rev-parse", "--verify", "--quiet", f"refs/tags/{TAG}^{{commit}}")
    if rc:
        return Check("protocol_tag", False, f"tag {TAG} not found")
    anc = _git(root, "merge-base", "--is-ancestor", out, "HEAD")[0] == 0
    return Check("protocol_tag", anc, f"{TAG} -> {out}" + ("" if anc else " (not an ancestor of HEAD)"))


def check_design_files_unchanged(root) -> Check:
    rc, out = _git(root, "rev-parse", "--verify", "--quiet", f"refs/tags/{TAG}^{{commit}}")
    if rc:
        return Check("design_files_unchanged_since_tag", False, "tag missing")
    bad = []
    for rel in DESIGN_FILES:
        if _git(root, "diff", "--name-only", out, "--", rel)[1]:
            bad.append(f"{rel}: differs from tag")
        if _git(root, "cat-file", "-e", f"{out}:{rel}")[0]:
            bad.append(f"{rel}: absent at tag")
    return Check("design_files_unchanged_since_tag", not bad, "; ".join(bad) or "registration, spec and manifest identical to the tag")


def code_changes_since_tag(root) -> list[str]:
    rc, out = _git(root, "rev-parse", "--verify", "--quiet", f"refs/tags/{TAG}^{{commit}}")
    if rc:
        return ["tag missing"]
    return _git(root, "diff", "--name-status", out, "HEAD", "--", *CODE_FILES)[1].splitlines()


def check_design_artifacts(root) -> Check:
    """Design artifacts on disk hash-match the tracked manifest, set files match the pins, draw table fingerprint matches."""
    root = Path(root)
    mp = root / D.MANIFEST
    if not mp.is_file():
        return Check("design_artifacts_hashes", False, f"{D.MANIFEST} missing")
    man = json.loads(mp.read_text())
    bad = []
    dd = root / D.DESIGN_DIR
    for name, sha in man["artifacts_sha256"].items():
        f = dd / name
        if not f.is_file():
            bad.append(f"{name}: missing")
        elif sha256_file(f) != sha:
            bad.append(f"{name}: sha256 mismatch")
    extra = sorted(p.name for p in dd.iterdir() if p.is_file() and p.name not in man["artifacts_sha256"]) if dd.is_dir() else []
    if extra:
        bad.append(f"extra files: {extra}")
    pins = json.loads((root / D.PINS).read_text())["known_sets_sha256"]
    for rel, sha in man["inputs_sha256"].items():
        f = root / rel
        if not f.is_file() or sha256_file(f) != sha:
            bad.append(f"input {rel}: missing or changed")
        if rel in pins and pins[rel] != sha:
            bad.append(f"input {rel}: differs from exp5 pins")
    from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import BlockBootstrap
    blocks = json.loads((dd / "draw_table_blocks.json").read_text()) if (dd / "draw_table_blocks.json").is_file() else []
    if blocks:
        bb = BlockBootstrap(np.array(sorted(blocks)), D.N_BOOT, D.SEED)
        if bb.fingerprint() != man["bootstrap_fingerprint"] or not np.array_equal(bb.draws, np.load(dd / "draw_table.npy")):
            bad.append("draw table differs from BlockBootstrap(seed 17, B 1000)")
    return Check("design_artifacts_hashes", not bad, "; ".join(bad) or f"{len(man['artifacts_sha256'])} artifacts + {len(man['inputs_sha256'])} inputs match manifest")


def check_registered_stores(root, arms=ALL_ARMS, read_stores: bool = True) -> list[Check]:
    """Registered identities (matrix.yaml vs the spec/frozen manifest) always; actual store sha256 and cpg_idx coverage only when
    ``read_stores`` (real runs). The dry-run in the preparation task does NOT read stores."""
    import yaml
    root = Path(root)
    spec = yaml.safe_load((root / SPEC).read_text())
    specs = load_arm_specs(root, arms)
    out = []
    bad = []
    for a in arms:
        reg = spec["arms"][a]
        s = specs[a]
        if s.sha256 != reg["sha256"] or s.store_h5 != reg["store"] or s.dim != reg["dim"]:
            bad.append(f"{a}: matrix.yaml entry differs from the registered spec")
    fz = json.loads((root / "configs/frozen/regulatory_histone_dnase_v1.json").read_text())
    if fz.get("status") != "candidate_frozen" or fz["file_sha256"]["store_h5"] != specs["regulatory_histone_dnase_v1"].sha256:
        bad.append("candidate frozen manifest: status/sha256 mismatch")
    out.append(Check("registered_store_identities", not bad, "; ".join(bad) or "matrix.yaml == spec == candidate frozen manifest"))
    if not read_stores:
        out.append(Check("store_sha256_and_coverage", False, "NOT EXECUTED (stores are not read in preparation/dry-run); mandatory in a real run"))
        return out
    bad = []
    import pandas as pd

    from cpg_repr_benchmark.representations.hdf5_store import inspect_representation_h5
    pm = pd.read_csv(root / D.DESIGN_DIR / "pairs_main.csv")
    ps = pd.read_csv(root / D.DESIGN_DIR / "pairs_s1.csv")
    need = np.unique(np.r_[pm.pos_cpg_idx, pm.neg_cpg_idx, ps.pos_cpg_idx, ps.neg_cpg_idx])
    for a in arms:
        p = root / specs[a].store_h5
        if not p.is_file():
            bad.append(f"{a}: store missing")
            continue
        if sha256_file(p) != specs[a].sha256:
            bad.append(f"{a}: store sha256 mismatch")
            continue
        ids = inspect_representation_h5(p)[0]
        miss = np.setdiff1d(need, ids)
        if len(miss):
            bad.append(f"{a}: {len(miss)} design CpGs absent from the store (coverage must be 100%)")
    out.append(Check("store_sha256_and_coverage", not bad, "; ".join(bad) or f"{len(arms)} stores match and cover 100% of {len(need)} design CpGs"))
    return out


def check_out_root(root, out_root) -> Check:
    root = Path(root).resolve()
    p = Path(out_root).resolve()
    base = (root / OUT_BASE).resolve()
    if base != p and base not in p.parents:
        return Check("out_root", False, f"{p} is not inside {OUT_BASE}/")
    if any(s in str(p).lower() for s in FORBIDDEN_PATH_PARTS) or lg.is_protected(root, p):
        return Check("out_root", False, f"{p} is protected/forbidden")
    return Check("out_root", True, str(p))


def check_inputs(paths) -> Check:
    bad = [str(p) for p in paths if any(s in str(p).replace("\\", "/").lower() for s in FORBIDDEN_PATH_PARTS)]
    return Check("no_tcga_protocol_or_loyfer_audit_input", not bad, "; ".join(bad) or "no input path matches a forbidden pattern")


def read_state(out_root) -> dict:
    p = Path(out_root) / "state.json"
    st = json.loads(p.read_text()) if p.is_file() else {"state": "FROZEN", "arms_done": [], "contrasts_done": False, "reported": False}
    # arms_done is also derived from the per-arm manifests so that two concurrently running arms cannot lose each other's entry
    on_disk = [a for a in ALL_ARMS if (Path(out_root) / "arms" / a / "arm_manifest.json").is_file()]
    st["arms_done"] = sorted(set(st["arms_done"]) | set(on_disk))
    return st


def check_state(out_root, command: str, arm: str | None = None) -> Check:
    st = read_state(out_root)
    done = set(st["arms_done"])
    if command == "run":
        ok = st["state"] in ("FROZEN", "ARMS_RUNNING") and not st.get("contrasts_done")
        return Check("state_machine", ok, f"state={st['state']} arms_done={sorted(done)}; run {arm} " + ("allowed" if ok else "refused (contrasts already computed)"))
    if command == "contrasts":
        miss = [a for a in MAIN_ARMS if a not in done]
        return Check("state_machine", not miss, f"state={st['state']}; missing main arms: {miss}" if miss else "all main arms done")
    if command == "report-inputs":
        return Check("state_machine", bool(st.get("contrasts_done")), f"contrasts_done={st.get('contrasts_done')}")
    return Check("state_machine", True, f"state={st['state']}")


def run_gate(root, out_root, *, command="run", arm=None, input_paths=(), read_stores=True) -> list[Check]:
    root = Path(root)
    checks = [check_registration(root), check_code(root), check_tag(root), check_design_files_unchanged(root),
              check_design_artifacts(root)]
    checks += check_registered_stores(root, ALL_ARMS, read_stores=read_stores)
    checks += [check_out_root(root, out_root), check_inputs(input_paths), check_state(out_root, command, arm)]
    return checks


class EwasWriter(lg.GuardedWriter):
    """Writes only below <repo>/outputs/ewas_v2/...; never into other experiments' outputs, design dirs, configs, src or the loyfer audit."""

    def __init__(self, root, out_root):
        super().__init__(root, out_root)
        base = (self.root / OUT_BASE).resolve()
        if self.out_root != base and base not in self.out_root.parents:
            raise PermissionError(f"output root {self.out_root} is outside {OUT_BASE}/")
        self._check_forbidden(self.out_root)

    @staticmethod
    def _check_forbidden(p):
        if any(s in str(p).lower() for s in FORBIDDEN_PATH_PARTS):
            raise PermissionError(f"forbidden path: {p}")

    def path(self, *parts) -> Path:
        p = self.out_root.joinpath(*parts).resolve()
        self._check_forbidden(p)
        if (self.root / D.DESIGN_DIR).resolve() in [p, *p.parents]:
            raise PermissionError(f"refusing to write into the frozen design artifacts: {p}")
        return super().path(*parts)

    def write_df(self, rel, df):
        df.to_csv(self.path(rel), index=False)

    def set_state(self, **kw):
        st = read_state(self.out_root)
        st.update(kw)
        self.write_json("state.json", st)


def run_manifest(root, arm, identity, *, checks, threads, command, extra=None) -> dict:
    import subprocess  # noqa: F401
    root = Path(root)
    reg = root / REGISTRATION_DOC
    man = root / D.MANIFEST
    tag = _git(root, "rev-parse", f"refs/tags/{TAG}^{{commit}}")[1]
    d = {"arm": arm, "head_commit": lg.head_commit(root), "protocol_tag": TAG, "protocol_commit": tag,
         "registration_sha256": hashlib.sha256(reg.read_bytes()).hexdigest() if reg.exists() else None,
         "design_manifest_sha256": hashlib.sha256(man.read_bytes()).hexdigest() if man.exists() else None,
         "representation": identity, "seed": D.SEED, "n_bootstrap": D.N_BOOT, "folds": D.N_FOLDS, "C_grid": list(D.C_GRID),
         "max_iter": D.MAX_ITER, "threads": threads, "library_versions": lg.library_versions(),
         "gate_checks": [c.__dict__ for c in checks], "code_changes_since_tag": code_changes_since_tag(root),
         "tcga_test_set_referenced": False, "command": command}
    d.update(extra or {})
    return d
