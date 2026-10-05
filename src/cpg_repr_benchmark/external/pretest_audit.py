"""PRE-TEST audit of the external GSE40279 confirmation (protocol v1.2). READ-ONLY with one exception: the report / snapshot JSON files it
writes under `outputs/external_reconstruction_v1/audit/` (`pretest_audit_<label>.json`, `A_snapshot_<label>.json`). Reads no methylation
beta value of any split and never creates `evaluation/test`.

Each check has a status: ``ok`` | ``fail`` | ``pending``. ``pending`` is used ONLY for facts that become true when the orchestrator commits
and tags (protocol v1.2 tag, test-authorization tag, implementation committed clean, frozen files unchanged since the v1.2 tag); every other
mismatch is a ``fail``. `exit_code(report)` is non-zero iff a non-pending check failed.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import gate as G
from .freeze import ARMS, LEGACY_ARMS, MAIN_ARMS, MANIFEST_REL, SEEDS
from .io import sha256_file

AUDIT_DIR_REL = Path(G.OUTPUT_ROOT_REL) / "audit"
SNAPSHOT_TREES = ("benchmark", "analysis/A")           # relative to outputs/external_reconstruction_v1
PRIMARY_FINGERPRINT = "38b9da12700a8fc83dc08f5a01aaa76b61f0d5097bffe395d8eb5225f1f02094"
PAIR_KEYS = ("sample_index", "target_matrix_column", "panel_repeat", "target", "prior_prediction")
FRACTIONS = (0.15, 0.30, 0.50, 0.70, 0.90)
TCGA_STATUS_REL = "outputs/regulatory_confirmation_v1/phase_A_status.json"
TCGA_OUTPUT_ROOT_REL = "outputs/regulatory_confirmation_v1"
# gate checks that can only turn green after the orchestrator's commit / tag
GATE_PENDING_NAMES = frozenset({"protocol_tag_present", "amendment_commit_ancestor_of_HEAD", "frozen_files_unchanged_since_tag",
                                "implementation_committed_clean", "protocol_v1_2_tag_present", "authorization_tag_present"})
EXPECTED_PENDING = (
    "tag:external-recon-protocol-freeze-v1.2 (created by the orchestrator with the amendment-2 commit)",
    "tag:external-recon-test-authorization-v1 (created by the separate dated promotion commit, BEFORE the first test read)",
    "gate:implementation_committed_clean (new/changed implementation files are not committed yet)",
    "gate:frozen_files_unchanged_since_tag (needs the v1.2 tag)",
    "gate:amendment_commit_ancestor_of_HEAD / protocol_tag_present (v1.2 tag)",
)


@dataclass
class AuditCheck:
    name: str
    status: str            # ok | fail | pending
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        return f"{self.status.upper():7s} {self.name} {self.detail}".rstrip()


def _ok(name, ok, detail="", **data) -> AuditCheck:
    return AuditCheck(name, "ok" if ok else "fail", str(detail), data)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)


def _manifest(repo: Path, manifest_path: Path | None = None) -> dict:
    return json.loads((Path(manifest_path) if manifest_path else Path(repo) / MANIFEST_REL).read_text())


# ----------------------------------------------------------------------------- guarded writes (the ONLY writes of this module)
def audit_path(repo: Path, name: str) -> Path:
    """`<repo>/outputs/external_reconstruction_v1/audit/<name>`; only `pretest_audit_*.json` / `A_snapshot_*.json` are accepted."""
    if not re.fullmatch(r"(pretest_audit|A_snapshot)_[A-Za-z0-9_.\-]+\.json", name):
        raise PermissionError(f"GUARD: the audit may only write pretest_audit_<label>.json / A_snapshot_<label>.json, not {name!r}")
    return Path(repo) / AUDIT_DIR_REL / name


def _write_new(path: Path, obj: Any, *, overwrite: bool = False) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; refusing to overwrite (pick another --label)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    return path


def _label(label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", label or ""):
        raise ValueError(f"invalid label {label!r}")
    return label


# ----------------------------------------------------------------------------- snapshot / compare of the Experiment-A artifacts
def _tree_files(root: Path) -> list[Path]:
    out = []
    for dirpath, _dirs, files in os.walk(root):
        out += [Path(dirpath) / f for f in files]
    return sorted(out)


def take_snapshot(repo: Path) -> dict:
    """sha256 + size + mtime of ALL files under outputs/external_reconstruction_v1/{benchmark, analysis/A}."""
    base = Path(repo) / G.OUTPUT_ROOT_REL
    files: dict[str, dict] = {}
    for tree in SNAPSHOT_TREES:
        for f in _tree_files(base / tree):
            st = f.stat()
            files[f.relative_to(base).as_posix()] = {"sha256": sha256_file(f), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    return {"schema": "external_A_snapshot/1", "trees": list(SNAPSHOT_TREES), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_files": len(files), "files": files}


def write_snapshot(repo: Path, label: str) -> Path:
    return _write_new(audit_path(repo, f"A_snapshot_{_label(label)}.json"), take_snapshot(repo))


def compare_snapshots(old: dict, new: dict, *, ignore_mtime: bool = False) -> dict:
    a, b = old["files"], new["files"]
    added, removed = sorted(set(b) - set(a)), sorted(set(a) - set(b))
    changed = sorted(k for k in set(a) & set(b) if a[k]["sha256"] != b[k]["sha256"] or a[k]["size"] != b[k]["size"])
    mtime_only = sorted(k for k in set(a) & set(b) if k not in changed and a[k]["mtime_ns"] != b[k]["mtime_ns"])
    ok = not (added or removed or changed or (mtime_only and not ignore_mtime))
    return {"identical": ok, "n_files_old": len(a), "n_files_new": len(b), "added": added, "removed": removed, "changed": changed,
            "mtime_only_changed": mtime_only, "ignore_mtime": ignore_mtime}


def compare_with_saved(repo: Path, label: str, *, ignore_mtime: bool = False) -> dict:
    p = audit_path(repo, f"A_snapshot_{_label(label)}.json")
    return compare_snapshots(json.loads(p.read_text()), take_snapshot(repo), ignore_mtime=ignore_mtime)


# ----------------------------------------------------------------------------- run discovery (markers; read-only)
def a_runs(repo: Path) -> dict[tuple[str, int], Path]:
    out = {}
    for arm in ARMS:
        for seed in SEEDS:
            mk = Path(repo) / G.OUTPUT_ROOT_REL / "logs" / f"{arm}__seed{seed}.done"
            if mk.is_file():
                out[(arm, seed)] = Path(json.loads(mk.read_text())["run_dir"])
    return out


# ----------------------------------------------------------------------------- checks
def check_manifest_state(manifest: dict) -> list[AuditCheck]:
    state, auth = manifest.get("freeze_state"), manifest.get("test_set_authorized")
    consistent = (state, auth) in (("draft", False), ("final", True))
    return [_ok("manifest_state_consistent", consistent, f"freeze_state={state!r} test_set_authorized={auth!r}"),
            AuditCheck("manifest_state_report", "ok", f"freeze_state={state} test_set_authorized={auth} "
                       f"({'pre-test: draft/false as expected' if (state, auth) == ('draft', False) else 'PROMOTED state'})"),
            _ok("manifest_protocol_v1_2", manifest.get("protocol_version") == "v1.2", str(manifest.get("protocol_version"))),
            _ok("manifest_disclosure_flag", (manifest.get("amendment_2") or {}).get("decided_after_validation_A_results") is True,
                "amendment_2.decided_after_validation_A_results")]


def check_a_checkpoints(repo: Path, runs: dict | None = None) -> list[AuditCheck]:
    """12 A best.pt: sha256 vs confirmation_status.json and phase_A_status.json; not modified after the status file; best epoch."""
    repo = Path(repo)
    runs = a_runs(repo) if runs is None else runs
    agg_p = repo / G.OUTPUT_ROOT_REL / "phase_A_status.json"
    agg = {(r["arm"], int(r["seed"])): r for r in json.loads(agg_p.read_text())["runs"]} if agg_p.is_file() else {}
    bad, rows = [], {}
    for arm in ARMS:
        for seed in SEEDS:
            key = f"{arm}__seed{seed}"
            rd = runs.get((arm, seed))
            if rd is None:
                bad.append(f"{key}: no finished run")
                continue
            try:
                st = json.loads((rd / "confirmation_status.json").read_text())
                best = rd / "checkpoints" / "best.pt"
                got = sha256_file(best)
                hist = json.loads((rd / "history.json").read_text())
                mse = [float(r["validation_mse"]) for r in hist]
                row = {"sha256": got, "status_sha256": st["best_pt_sha256"], "aggregate_sha256": (agg.get((arm, seed)) or {}).get("best_pt_sha256")}
                probs = []
                if got != st["best_pt_sha256"]:
                    probs.append("sha != confirmation_status.json")
                if row["aggregate_sha256"] != got:
                    probs.append("sha != phase_A_status.json")
                if int(np.argmin(mse)) != int(st["best_epoch_0based"]) or len(hist) != 120:
                    probs.append("history argmin/epochs != status")
                if best.stat().st_mtime_ns > (rd / "confirmation_status.json").stat().st_mtime_ns:
                    probs.append("best.pt modified after confirmation_status.json")
                if st.get("test_read") is not False or (rd / "evaluation" / "test").exists():
                    probs.append("test read/dir")
                rows[key] = {**row, "problems": probs}
                bad += [f"{key}: {p}" for p in probs]
            except Exception as e:  # noqa: BLE001
                bad.append(f"{key}: {e!r}")
    return [_ok("A_best_pt_sha256_12", not bad and len(rows) == 12, "12/12 best.pt consistent (status, aggregate, history, mtime)"
                if not bad else "; ".join(bad), runs=rows)]


def check_tcga_checkpoints(repo: Path, manifest: dict) -> list[AuditCheck]:
    """The 9 main (+3 legacy) TCGA checkpoints used by B: sha256 vs the freeze manifest and the TCGA phase_A_status.json."""
    repo = Path(repo)
    st_p = repo / TCGA_STATUS_REL
    tcga = {(r["arm"], int(r["seed"])): r.get("best_pt_sha256") for r in json.loads(st_p.read_text())["runs"]} if st_p.is_file() else {}
    out = []
    for role, arms, n in (("main", MAIN_ARMS, 9), ("legacy", LEGACY_ARMS, 3)):
        bad, seen = [], 0
        for c in manifest.get("phase_a_checkpoints_experiment_B", []):
            if c["arm"] not in arms:
                continue
            seen += 1
            p = repo / c["best_pt"]
            got = sha256_file(p) if p.is_file() else "missing"
            if got != c["best_pt_sha256"]:
                bad.append(f"{c['arm']}#{c['seed']}: sha != manifest")
            if tcga.get((c["arm"], int(c["seed"]))) != c["best_pt_sha256"]:
                bad.append(f"{c['arm']}#{c['seed']}: manifest != TCGA phase_A_status.json")
        out.append(_ok(f"tcga_best_pt_{role}_{n}", not bad and seen == n, f"{seen}/{n} checkpoints verified" if not bad else "; ".join(bad)))
    return out


def check_frozen_files(repo: Path, manifest: dict, runs: dict, *, heavy: bool = True) -> list[AuditCheck]:
    repo = Path(repo)
    bad = []
    for k, rel in manifest["files"].items():
        if not heavy and k == "dataset_h5":
            continue
        p = repo / rel
        got = sha256_file(p) if p.is_file() else "missing"
        if got != manifest["files_sha256"][k]:
            bad.append(f"{k}")
    res = [_ok("frozen_files_sha256", not bad, "all manifest file hashes match" if not bad else f"mismatch: {bad}")]
    inconsistent = []
    for (arm, seed), rd in runs.items():
        try:
            st = json.loads((rd / "confirmation_status.json").read_text())
        except Exception as e:  # noqa: BLE001
            inconsistent.append(f"{arm}#{seed}: {e!r}")
            continue
        for key, mk in (("patient_protocol_sha256", "patient_protocol_npz"), ("locus_protocol_sha256", "locus_protocol_npz"),
                        ("matrix_h5_sha256", "dataset_h5")):
            if st.get(key) != manifest["files_sha256"][mk]:
                inconsistent.append(f"{arm}#{seed}:{key}")
        if st.get("store_sha256") != manifest["arms"][arm]["store_sha256"]:
            inconsistent.append(f"{arm}#{seed}:store_sha256")
    res.append(_ok("runs_trained_on_frozen_inputs", not inconsistent, "split/locus/matrix/store hashes at training time = manifest"
                   if not inconsistent else "; ".join(inconsistent)))
    return res


def check_split(repo: Path, manifest: dict, runs: dict) -> list[AuditCheck]:
    """Counts, partition, the 66 test ids equal the split CSV's test subjects (ids only, no beta), runs' own split = protocol."""
    import csv
    repo = Path(repo)
    with np.load(repo / manifest["files"]["patient_protocol_npz"]) as z:
        parts = {k: np.asarray(z[k]) for k in ("train", "validation", "test")}
        names = [str(x) for x in z["sample_names"]]
    counts = {k: len(v) for k, v in parts.items()}
    joined = np.sort(np.concatenate(list(parts.values())))
    out = [_ok("split_counts_524_66_66", counts == manifest["counts"] == {"train": 524, "validation": 66, "test": 66}, str(counts)),
           _ok("split_partition", np.array_equal(joined, np.arange(len(names))))]
    with open(repo / manifest["files"]["split_csv"], newline="") as f:
        rows = list(csv.DictReader(f))
    for split in ("validation", "test"):
        csv_ids = sorted(r["sample_name"] for r in rows if r["split"] == split)
        proto_ids = sorted(names[i] for i in parts[split])
        out.append(_ok(f"{split}_subject_ids_equal_split_csv_66", csv_ids == proto_ids and len(proto_ids) == 66,
                       f"{len(proto_ids)} ids (names only; no beta read)"))
    bad = []
    for (arm, seed), rd in runs.items():
        with np.load(rd / "patient_split.npz") as z:
            if not all(np.array_equal(np.asarray(z[k]), parts[k]) for k in parts):
                bad.append(f"{arm}#{seed}")
    out.append(_ok("runs_patient_split_equals_protocol", not bad and bool(runs), f"{len(runs)} runs" if not bad else f"differs: {bad}"))
    return out


def check_universe(repo: Path, manifest: dict, runs: dict) -> list[AuditCheck]:
    repo = Path(repo)
    out = [_ok(c.name, c.ok, c.detail) for c in G.check_universe(repo, manifest) + G.check_store_coverage(repo, manifest)]
    uni = G.universe_ids(repo, manifest)
    from cpg_repr_benchmark.data.methylation import read_axis
    ids, _ = read_axis(repo / manifest["files"]["dataset_h5"])
    bad = []
    for (arm, seed), rd in runs.items():
        with np.load(rd / "locus_split.npz") as z:
            cols, ho = np.asarray(z["train_matrix_columns"]), np.asarray(z["heldout_matrix_columns"])
        if len(ho) or not np.array_equal(np.sort(ids[cols]), uni):
            bad.append(f"{arm}#{seed}")
    out.append(_ok("runs_locus_columns_equal_universe", not bad and bool(runs), f"{len(runs)} runs, universe {len(uni)}" if not bad else f"differs: {bad}"))
    cov_p = repo / G.OUTPUT_ROOT_REL / "audit" / "arm_coverage.json"
    if cov_p.is_file():
        out.append(_ok("arm_coverage_report_all_green", json.loads(cov_p.read_text()).get("all_green") is True, "audit/arm_coverage.json"))
    return out


def check_prediction_pairing(repo: Path, runs: dict, *, split: str = "validation") -> list[AuditCheck]:
    """For every seed x fraction the validation predictions of all arms share sample_index, target_matrix_column, panel_repeat, target and
    prior_prediction exactly (same logic as `analyze_regulatory_confirm.pairing_check`)."""
    bad, n = [], 0
    for seed in SEEDS:
        arms = [a for a in ARMS if (a, seed) in runs]
        if len(arms) < 2:
            bad.append(f"seed {seed}: fewer than 2 arms")
            continue
        for f in FRACTIONS:
            raw = {}
            for a in arms:
                with np.load(runs[(a, seed)] / "evaluation" / split / "seen" / f"mask_{f:.2f}" / "predictions.npz") as h:
                    raw[a] = {k: h[k] for k in PAIR_KEYS}
            ref = arms[0]
            for a in arms[1:]:
                for k in PAIR_KEYS:
                    if not np.array_equal(raw[a][k], raw[ref][k]):
                        bad.append(f"seed {seed} f {f:.2f}: {a} {k} != {ref}")
            n += 1
    return [_ok("prediction_pairing_across_arms", not bad and n == len(SEEDS) * len(FRACTIONS),
                f"{n} seed x fraction cells identical across arms" if not bad else "; ".join(bad[:6]))]


def find_test_dirs(root: Path) -> list[str]:
    found = []
    for dirpath, dirs, _files in os.walk(root):
        if os.path.basename(dirpath) == "evaluation" and "test" in dirs:
            found.append(str((Path(dirpath) / "test").relative_to(root)))
    return found


def check_no_test_dirs(repo: Path) -> list[AuditCheck]:
    repo = Path(repo)
    ext = find_test_dirs(repo / G.OUTPUT_ROOT_REL) if (repo / G.OUTPUT_ROOT_REL).is_dir() else []
    tc = find_test_dirs(repo / TCGA_OUTPUT_ROOT_REL) if (repo / TCGA_OUTPUT_ROOT_REL).is_dir() else []
    return [_ok("no_evaluation_test_dir_external", not ext, "none" if not ext else f"FOUND {ext[:5]}"),
            _ok("no_evaluation_test_dir_tcga_confirmation", not tc, "none" if not tc else f"FOUND {tc[:5]}")]


def check_fingerprint(repo: Path) -> list[AuditCheck]:
    from cpg_repr_benchmark.bioval_v2_followup import gate as BG
    base = BG.base_dir(repo)
    if not Path(base).is_dir():
        return [_ok("primary_biological_fingerprint", False, f"outputs tree absent: {base}")]
    got = BG.frozen_results_fingerprint(repo)
    return [_ok("primary_biological_fingerprint", got == PRIMARY_FINGERPRINT == BG.FROZEN_RESULTS_SHA256, got)]


def check_tags(repo: Path) -> list[AuditCheck]:
    out = []
    for tag, prefix in (("external-recon-protocol-freeze-v1", G.V1_COMMIT_PREFIX), (G.TAG_V1_1, G.V1_1_COMMIT_PREFIX)):
        r = _git(repo, "rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}")
        c = r.stdout.strip() if r.returncode == 0 else ""
        out.append(_ok(f"tag:{tag}", c.startswith(prefix), f"{c[:10] or 'missing'} (expected {prefix}...)"))
    for tag, why in ((G.TAG_V1_2, "created with the amendment-2 commit"), (G.TAG_AUTH, "created by the dated promotion commit before the test read")):
        r = _git(repo, "rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}")
        if r.returncode == 0:
            out.append(AuditCheck(f"tag:{tag}", "ok", r.stdout.strip()[:10]))
        else:
            out.append(AuditCheck(f"tag:{tag}", "pending", f"missing ({why})"))
    return out


def check_protected_unmodified(repo: Path) -> list[AuditCheck]:
    paths = ("configs/frozen", "configs/experiments/regulatory_confirmation_matrix", "configs/external/gse40279_v1_split.csv",
             "configs/external/gse40279_v1_split_manifest.json", "configs/external/gse40279_v1_mapping_manifest.json")
    dirty = [p for p in paths if (Path(repo) / p).exists() and _git(repo, "status", "--porcelain", "--", p).stdout.strip()]
    return [_ok("protected_tracked_files_unmodified", not dirty, "clean" if not dirty else f"modified: {dirty}")]


def check_test_never_read(repo: Path, runs: dict) -> list[AuditCheck]:
    flags = []
    for rd in runs.values():
        st = json.loads((rd / "confirmation_status.json").read_text())
        flags.append(st.get("test_read") is False)
    agg = Path(repo) / G.OUTPUT_ROOT_REL / "phase_A_status.json"
    agg_ok = agg.is_file() and json.loads(agg.read_text()).get("test_read") is False
    return [_ok("status_files_record_test_read_false", all(flags) and bool(flags) and agg_ok, f"{sum(flags)}/{len(flags)} runs, aggregate {agg_ok}")]


def gate_status(repo: Path, *, scope: str = "A", heavy: bool = True) -> list[AuditCheck]:
    """The runner gate (`run_external_confirmation.py audit`): green checks are ok; git-only failures are pending; others fail."""
    out = []
    checks = G.run_gate(Path(repo), split="validation", scope=scope, heavy=heavy)
    for c in checks:
        if c.ok:
            continue
        out.append(AuditCheck(f"gate_{scope}:{c.name}", "pending" if c.name in GATE_PENDING_NAMES else "fail", c.detail))
    n_bad = len(out)
    out.insert(0, AuditCheck(f"gate_{scope}_summary", "ok" if n_bad == 0 else ("pending" if all(o.status == "pending" for o in out) else "fail"),
                             f"{len(checks) - n_bad}/{len(checks)} gate checks green (validation phase, scope {scope})"))
    return out


# ----------------------------------------------------------------------------- composition
def run_audit(repo: Path, *, label: str, heavy: bool = True, fingerprint: bool = True, gate: bool = True, manifest_path: Path | None = None) -> dict:
    repo = Path(repo)
    manifest = _manifest(repo, manifest_path)
    runs = a_runs(repo)
    checks: list[AuditCheck] = []

    def group(name, fn):
        try:
            checks.extend(fn())
        except Exception as e:  # noqa: BLE001
            checks.append(AuditCheck(f"{name}:error", "fail", repr(e)))
    group("manifest_state", lambda: check_manifest_state(manifest))
    group("a_checkpoints", lambda: check_a_checkpoints(repo, runs))
    group("tcga_checkpoints", lambda: check_tcga_checkpoints(repo, manifest))
    group("frozen_files", lambda: check_frozen_files(repo, manifest, runs, heavy=heavy))
    group("split", lambda: check_split(repo, manifest, runs))
    group("universe", lambda: check_universe(repo, manifest, runs))
    group("pairing", lambda: check_prediction_pairing(repo, runs))
    group("test_dirs", lambda: check_no_test_dirs(repo))
    group("test_never_read", lambda: check_test_never_read(repo, runs))
    group("tags", lambda: check_tags(repo))
    group("protected", lambda: check_protected_unmodified(repo))
    if fingerprint:
        group("fingerprint", lambda: check_fingerprint(repo))
    if gate:
        group("gate_A", lambda: gate_status(repo, scope="A", heavy=heavy))
        group("gate_B", lambda: gate_status(repo, scope="B", heavy=False))
    n = {s: sum(c.status == s for c in checks) for s in ("ok", "fail", "pending")}
    return {"schema": "external_pretest_audit/1", "label": label, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manifest": {"freeze_state": manifest.get("freeze_state"), "test_set_authorized": manifest.get("test_set_authorized"),
                         "protocol_version": manifest.get("protocol_version")},
            "counts": n, "all_non_pending_green": n["fail"] == 0, "expected_pending": list(EXPECTED_PENDING),
            "pending": [c.name for c in checks if c.status == "pending"],
            "checks": [{"name": c.name, "status": c.status, "detail": c.detail, "data": c.data} for c in checks]}


def exit_code(report: dict) -> int:
    return 0 if report["counts"]["fail"] == 0 else 1


def write_report(repo: Path, report: dict, *, overwrite: bool = False) -> Path:
    return _write_new(audit_path(repo, f"pretest_audit_{_label(report['label'])}.json"), report, overwrite=overwrite)
