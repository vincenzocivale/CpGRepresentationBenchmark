"""Pre-test audit, A-artifact snapshot/compare and their write guards (synthetic trees; the real tree is only read by the CLI)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from cpg_repr_benchmark.external import gate as G
from cpg_repr_benchmark.external import pretest_audit as PA

ROOT = Path(__file__).resolve().parents[1]
ARMS, SEEDS, FRACS = PA.ARMS, PA.SEEDS, PA.FRACTIONS


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _make_tree(repo: Path):
    """12 finished A runs (4 arms x 3 seeds) with consistent status files, histories, predictions and markers."""
    base = repo / G.OUTPUT_ROOT_REL
    rng = np.random.default_rng(0)
    runs, agg = {}, []
    for seed in SEEDS:
        shared = {"sample_index": np.arange(6), "target_matrix_column": rng.integers(0, 50, (6, 8)), "panel_repeat": np.zeros(6, np.int64),
                  "target": rng.uniform(0, 1, (6, 8)).astype(np.float32), "prior_prediction": rng.uniform(0, 1, (6, 8)).astype(np.float32)}
        for arm in ARMS:
            rd = base / "benchmark" / "masking" / arm / f"seed_{seed}" / "run"
            (rd / "checkpoints").mkdir(parents=True)
            (rd / "checkpoints/best.pt").write_bytes(f"{arm}{seed}".encode())
            hist = [{"epoch": e, "validation_mse": 1.0 - 0.001 * e} for e in range(120)]
            (rd / "history.json").write_text(json.dumps(hist))
            st = {"arm": arm, "seed": seed, "status": "trained_validation_frozen", "best_pt_sha256": _sha(rd / "checkpoints/best.pt"),
                  "best_epoch_0based": 119, "test_read": False}
            (rd / "confirmation_status.json").write_text(json.dumps(st))
            for f in FRACS:
                d = rd / "evaluation/validation/seen" / f"mask_{f:.2f}"
                d.mkdir(parents=True)
                np.savez(d / "predictions.npz", prediction=rng.uniform(0, 1, (6, 8)).astype(np.float32), **shared)
            (base / "logs").mkdir(exist_ok=True)
            (base / "logs" / f"{arm}__seed{seed}.done").write_text(json.dumps({"run_dir": str(rd)}))
            runs[(arm, seed)] = rd
            agg.append({"arm": arm, "seed": seed, "best_pt_sha256": st["best_pt_sha256"]})
    (base / "phase_A_status.json").write_text(json.dumps({"test_read": False, "runs": agg}))
    (base / "analysis/A/validation").mkdir(parents=True)
    (base / "analysis/A/validation/report.md").write_text("x")
    return runs


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"
    _make_tree(r)
    return r


# ------------------------------------------------------------------ snapshot / compare
def test_snapshot_compare_identical_then_each_kind_of_change(repo):
    p = PA.write_snapshot(repo, "pre")
    assert p == repo / G.OUTPUT_ROOT_REL / "audit/A_snapshot_pre.json" and json.loads(p.read_text())["n_files"] > 90
    assert PA.compare_with_saved(repo, "pre")["identical"]
    base = repo / G.OUTPUT_ROOT_REL
    best = next(base.rglob("best.pt"))
    # mtime-only change: fails unless ignored
    st = best.stat()
    os.utime(best, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    r = PA.compare_with_saved(repo, "pre")
    assert not r["identical"] and r["mtime_only_changed"] and not r["changed"]
    assert PA.compare_with_saved(repo, "pre", ignore_mtime=True)["identical"]
    # content change (same size): changed
    data = best.read_bytes()
    best.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    r = PA.compare_with_saved(repo, "pre", ignore_mtime=True)
    assert not r["identical"] and r["changed"] == [best.relative_to(base).as_posix()]
    best.write_bytes(data)
    # added / removed file in benchmark/ and in analysis/A
    (base / "benchmark/new.txt").write_text("n")
    (base / "analysis/A/validation/report.md").unlink()
    r = PA.compare_with_saved(repo, "pre", ignore_mtime=True)
    assert r["added"] == ["benchmark/new.txt"] and r["removed"] == ["analysis/A/validation/report.md"] and not r["identical"]
    # files outside the two trees are not part of the snapshot (e.g. B outputs, audit reports)
    (base / "transfer/B_strict").mkdir(parents=True)
    (base / "transfer/B_strict/x.npz").write_text("b")
    assert "transfer/B_strict/x.npz" not in PA.take_snapshot(repo)["files"]


def test_snapshot_records_sha_size_mtime_and_refuses_overwrite_and_bad_labels(repo):
    p = PA.write_snapshot(repo, "L1")
    ent = next(iter(json.loads(p.read_text())["files"].values()))
    assert set(ent) == {"sha256", "size", "mtime_ns"} and len(ent["sha256"]) == 64
    with pytest.raises(FileExistsError):
        PA.write_snapshot(repo, "L1")
    for bad in ("../x", "a/b", "", "x y"):
        with pytest.raises((ValueError, PermissionError)):
            PA.write_snapshot(repo, bad)


def test_audit_writes_only_report_names_under_audit_dir(repo):
    for bad in ("evil.json", "../x.json", "pretest_audit_.json/../../x.json", "A_snapshot_x.txt", "arm_coverage.json"):
        with pytest.raises(PermissionError):
            PA.audit_path(repo, bad)
    assert PA.audit_path(repo, "pretest_audit_ok-1.json").parent == repo / G.OUTPUT_ROOT_REL / "audit"
    before = {p: _sha(p) for p in (repo / G.OUTPUT_ROOT_REL / "benchmark").rglob("*") if p.is_file()}
    PA.write_snapshot(repo, "x")
    assert before == {p: _sha(p) for p in (repo / G.OUTPUT_ROOT_REL / "benchmark").rglob("*") if p.is_file()}
    assert sorted(os.listdir(repo / G.OUTPUT_ROOT_REL / "audit")) == ["A_snapshot_x.json"]


# ------------------------------------------------------------------ individual checks
def test_a_checkpoint_hash_checks(repo):
    runs = PA.a_runs(repo)
    assert len(runs) == 12
    assert PA.check_a_checkpoints(repo)[0].status == "ok"
    rd = runs[("cpgpt_large_locus", 42)]
    (rd / "checkpoints/best.pt").write_bytes(b"tampered")
    c = PA.check_a_checkpoints(repo)[0]
    assert c.status == "fail" and "cpgpt_large_locus__seed42" in c.detail
    # best.pt newer than the status file is flagged too
    (rd / "checkpoints/best.pt").write_bytes(b"cpgpt_large_locus42")
    st = rd / "confirmation_status.json"
    os.utime(rd / "checkpoints/best.pt", ns=(0, st.stat().st_mtime_ns + 10_000_000_000))
    assert "modified after confirmation_status.json" in PA.check_a_checkpoints(repo)[0].detail
    # aggregate disagreement
    agg = repo / G.OUTPUT_ROOT_REL / "phase_A_status.json"
    d = json.loads(agg.read_text())
    d["runs"][0]["best_pt_sha256"] = "0" * 64
    agg.write_text(json.dumps(d))
    assert PA.check_a_checkpoints(repo)[0].status == "fail"


def test_prediction_pairing_ok_and_each_key_mismatch_fails(repo):
    runs = PA.a_runs(repo)
    assert PA.check_prediction_pairing(repo, runs)[0].status == "ok"
    for key in PA.PAIR_KEYS:
        p = runs[("deepcpg_dna_locus", 97)] / "evaluation/validation/seen/mask_0.50/predictions.npz"
        orig = dict(np.load(p))
        mod = dict(orig)
        mod[key] = np.asarray(orig[key]).copy()
        mod[key].flat[0] = mod[key].flat[0] + 1
        np.savez(p, **mod)
        c = PA.check_prediction_pairing(repo, runs)[0]
        assert c.status == "fail" and key in c.detail, key
        np.savez(p, **orig)
    assert PA.check_prediction_pairing(repo, runs)[0].status == "ok"


def test_no_test_dir_checks(repo):
    assert all(c.status == "ok" for c in PA.check_no_test_dirs(repo))
    (repo / G.OUTPUT_ROOT_REL / "benchmark/masking/x/evaluation/test").mkdir(parents=True)
    assert PA.check_no_test_dirs(repo)[0].status == "fail"
    (repo / PA.TCGA_OUTPUT_ROOT_REL / "benchmark/y/evaluation/test").mkdir(parents=True)
    assert PA.check_no_test_dirs(repo)[1].status == "fail"
    assert PA.find_test_dirs(repo / G.OUTPUT_ROOT_REL) == ["benchmark/masking/x/evaluation/test"]


def test_test_never_read_flags(repo):
    runs = PA.a_runs(repo)
    assert PA.check_test_never_read(repo, runs)[0].status == "ok"
    st = runs[("regulatory_histone_dnase", 17)] / "confirmation_status.json"
    d = json.loads(st.read_text())
    d["test_read"] = True
    st.write_text(json.dumps(d))
    assert PA.check_test_never_read(repo, runs)[0].status == "fail"


def test_manifest_state_report():
    ok = PA.check_manifest_state({"freeze_state": "draft", "test_set_authorized": False, "protocol_version": "v1.2",
                                  "amendment_2": {"decided_after_validation_A_results": True}})
    assert all(c.status == "ok" for c in ok)
    assert "pre-test: draft/false" in next(c for c in ok if c.name == "manifest_state_report").detail
    bad = PA.check_manifest_state({"freeze_state": "draft", "test_set_authorized": True, "protocol_version": "v1.2",
                                   "amendment_2": {"decided_after_validation_A_results": False}})
    assert {c.name for c in bad if c.status == "fail"} == {"manifest_state_consistent", "manifest_disclosure_flag"}


def test_split_checks_with_synthetic_protocol(repo):
    runs = PA.a_runs(repo)
    rng = np.random.default_rng(1)
    perm = rng.permutation(30)
    parts = {"train": np.sort(perm[:18]), "validation": np.sort(perm[18:24]), "test": np.sort(perm[24:])}
    names = np.array([f"GSM{i}" for i in range(30)])
    (repo / "data").mkdir()
    np.savez(repo / "data/pat.npz", sample_names=names, seed=1, **parts)
    import csv
    with open(repo / "data/split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "sample_name", "split"])
        for k, v in parts.items():
            for i in v:
                w.writerow([i, names[i], k])
    man = {"files": {"patient_protocol_npz": "data/pat.npz", "split_csv": "data/split.csv"}, "counts": {"train": 524, "validation": 66, "test": 66}}
    for rd in runs.values():
        np.savez(rd / "patient_split.npz", **parts)
    res = {c.name: c for c in PA.check_split(repo, man, runs)}
    assert res["split_counts_524_66_66"].status == "fail"            # synthetic counts are 18/6/6: the 524/66/66 contract is enforced
    assert res["split_partition"].status == "ok" and res["runs_patient_split_equals_protocol"].status == "ok"
    assert res["validation_subject_ids_equal_split_csv_66"].status == "fail"     # 6 ids != the expected 66
    # a run whose own patient_split differs from the protocol is flagged
    np.savez(runs[("cpgpt_large_locus", 17)] / "patient_split.npz", train=parts["test"], validation=parts["validation"], test=parts["train"])
    res = {c.name: c for c in PA.check_split(repo, man, runs)}
    assert res["runs_patient_split_equals_protocol"].status == "fail"


def test_universe_checks_flag_run_with_wrong_locus_columns(repo, monkeypatch):
    runs = PA.a_runs(repo)
    ids = np.arange(100, 130, dtype=np.int64)
    (repo / "data").mkdir()
    with h5py.File(repo / "data/m.h5", "w") as h:
        h["cpg_idx"], h["beta"], h["sample_name"] = ids, np.zeros((2, 30), np.float32), np.array([b"a", b"b"])
    np.savez(repo / "data/loci.npz", train_cpg_idx=ids, heldout_cpg_idx=np.array([], np.int64))
    man = {"files": {"dataset_h5": "data/m.h5", "locus_protocol_npz": "data/loci.npz"}, "universe_size": 30, "arms": {}}
    for rd in runs.values():
        np.savez(rd / "locus_split.npz", train_matrix_columns=np.arange(30), heldout_matrix_columns=np.array([], np.int64))
    assert all(c.status == "ok" for c in PA.check_universe(repo, man, runs))
    np.savez(runs[("deepcpg_dna_locus", 42)] / "locus_split.npz", train_matrix_columns=np.arange(29), heldout_matrix_columns=np.array([], np.int64))
    res = {c.name: c for c in PA.check_universe(repo, man, runs)}
    assert res["runs_locus_columns_equal_universe"].status == "fail" and "deepcpg_dna_locus#42" in res["runs_locus_columns_equal_universe"].detail


def _git(repo, *a):
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def test_tag_checks_pending_vs_fail(tmp_path):
    g = tmp_path / "g"
    g.mkdir()
    _git(g, "init", "-q", "-b", "main")
    (g / "a").write_text("a")
    _git(g, "add", "-A")
    _git(g, "commit", "-qm", "c")
    res = {c.name: c.status for c in PA.check_tags(g)}
    assert res["tag:external-recon-protocol-freeze-v1"] == "fail" and res["tag:external-recon-protocol-freeze-v1.1"] == "fail"   # historic: must exist
    assert res["tag:external-recon-protocol-freeze-v1.2"] == "pending" and res["tag:external-recon-test-authorization-v1"] == "pending"
    _git(g, "tag", "external-recon-protocol-freeze-v1.2")
    assert {c.name: c.status for c in PA.check_tags(g)}["tag:external-recon-protocol-freeze-v1.2"] == "ok"


def test_real_repo_historic_tags_are_intact():
    res = {c.name: c for c in PA.check_tags(ROOT)}
    assert res["tag:external-recon-protocol-freeze-v1"].status == "ok" and res["tag:external-recon-protocol-freeze-v1.1"].status == "ok"
    assert res["tag:external-recon-protocol-freeze-v1.2"].status in ("pending", "ok")
    assert res["tag:external-recon-test-authorization-v1"].status in ("pending", "ok")


def test_gate_failures_are_classified_pending_only_for_git_facts(monkeypatch):
    fake = [G.Check("freeze_verify", True), G.Check("protocol_tag_present", False, "missing"), G.Check("implementation_committed_clean", False, "x"),
            G.Check("frozen_files_unchanged_since_tag", False, "no tag")]
    monkeypatch.setattr(G, "run_gate", lambda *a, **k: fake)
    res = PA.gate_status(ROOT, scope="A")
    assert res[0].status == "pending" and all(c.status == "pending" for c in res[1:]) and len(res) == 4
    fake.append(G.Check("sha256:dataset_h5", False, "mismatch"))
    res = PA.gate_status(ROOT, scope="A")
    assert res[0].status == "fail" and [c.name for c in res if c.status == "fail"] == ["gate_A_summary", "gate_A:sha256:dataset_h5"]


def test_exit_code_pending_only_is_zero_and_any_fail_is_nonzero():
    def rep(*statuses):
        return {"counts": {s: statuses.count(s) for s in ("ok", "fail", "pending")}}
    assert PA.exit_code(rep("ok", "pending", "pending")) == 0
    assert PA.exit_code(rep("ok", "fail", "pending")) == 1


def test_run_audit_never_raises_on_a_broken_tree_and_reports_failure(tmp_path):
    (tmp_path / "configs/external").mkdir(parents=True)
    (tmp_path / "configs/external/gse40279_v1_freeze_manifest.json").write_text(json.dumps(
        {"freeze_state": "draft", "test_set_authorized": False, "protocol_version": "v1.2", "files": {}, "arms": {}, "counts": {},
         "amendment_2": {"decided_after_validation_A_results": True}, "phase_a_checkpoints_experiment_B": []}))
    rep = PA.run_audit(tmp_path, label="t", fingerprint=False, gate=False)
    assert PA.exit_code(rep) == 1 and rep["counts"]["fail"] > 0 and rep["expected_pending"]
    assert not (tmp_path / G.OUTPUT_ROOT_REL / "audit").exists()                     # run_audit itself writes nothing
    PA.write_report(tmp_path, rep)
    assert (tmp_path / G.OUTPUT_ROOT_REL / "audit/pretest_audit_t.json").is_file()


def test_cli_wiring():
    spec = importlib.util.spec_from_file_location("_audit_cli", ROOT / "scripts/audit_external_pretest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    doc = mod.__doc__
    assert "snapshot" in doc and "compare" in doc and "pending" in doc and "benchmark/" in doc and "analysis/A" in doc
