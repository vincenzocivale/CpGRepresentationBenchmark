"""External confirmation runner, gate failure modes, configs and status (synthetic; no training, no real data read)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from cpg_repr_benchmark.experiments import eval_layout as el
from cpg_repr_benchmark.experiments.guards import enforce_patient_view
from cpg_repr_benchmark.external import gate as G
from cpg_repr_benchmark.external import runner as R

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["regulatory_histone_dnase", "functional_annotations_pca", "cpgpt_large_locus", "deepcpg_dna_locus"]


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ configs / order / diff
def test_run_order_is_seed_major_in_frozen_arm_order():
    js = R.jobs()
    assert len(js) == 12
    assert js == [(a, s) for s in (17, 42, 97) for a in ARMS]
    assert R.jobs(seeds=[42], only=["deepcpg_dna_locus", "regulatory_histone_dnase"]) == [("regulatory_histone_dnase", 42), ("deepcpg_dna_locus", 42)]


def test_12_tracked_configs_equal_builder_and_only_declared_fields_differ_from_tcga():
    seen_diff = set()
    for arm, seed in R.jobs():
        p = ROOT / R.CONFIG_DIR / R.config_name(arm, seed)
        cfg = yaml.safe_load(p.read_text())
        assert cfg == R.build_config(arm, seed, ROOT)
        assert R.check_diff_vs_tcga(arm, seed, ROOT, cfg) == []
        seen_diff |= set(R.diff_vs_tcga(arm, seed, ROOT, cfg))
        tr, ev = cfg["training"], cfg["evaluation"]
        assert (tr["epochs"], tr["early_stopping"], tr["batch_size"], tr["num_workers"], tr["learning_rate"], tr["weight_decay"]) == (120, False, 8, 8, 1e-4, 1e-4)
        assert tr["save_epoch_checkpoints"] is False and tr["save_last_checkpoint"] is False and "max_updates" not in tr
        assert ev["mask_seed"] == 17001 and ev["selection_mask_fraction"] == 0.5 and ev["mask_fractions"] == [0.15, 0.3, 0.5, 0.7, 0.9]
        assert ev["patient_view"] == ev["require_patient_view"] == "validation" and ev["output_layout"] == "split_dirs"
        assert ev["allow_overwrite_split_dir"] is False and ev["num_workers"] == 0 and ev["panel_size"] == 2048
        assert cfg["dataset"]["name"] == "external_gse40279_v1" and cfg["training"]["seed"] == seed
        assert cfg["experiment"]["output_root"] == "outputs/external_reconstruction_v1/benchmark"
    assert seen_diff == set(R.ALLOWED_DIFF_VS_TCGA)


def test_undeclared_difference_is_detected():
    cfg = R.build_config(ARMS[0], 17, ROOT)
    cfg["training"]["learning_rate"] = 3e-4
    cfg["model"]["hidden_dim"] = 1024
    assert set(R.check_diff_vs_tcga(ARMS[0], 17, ROOT, cfg)) == {"training.learning_rate", "model.hidden_dim"}


def test_test_configs_are_never_built_by_build_config():
    with pytest.raises(PermissionError):
        R.build_config(ARMS[0], 17, ROOT, split="test")


def test_keep_last_checkpoint_flag_is_explicit():
    assert R.build_config(ARMS[0], 17, ROOT, keep_last_checkpoint=True)["training"]["save_last_checkpoint"] is True


# ------------------------------------------------------------------ dry run / refusals
def test_dry_run_lists_12_in_order_and_launches_nothing(capsys):
    def boom(*a, **k):
        raise AssertionError("dry-run must not launch")
    lines = []
    rc = R.run_all(ROOT, dry_run=True, gate_checks=[], popen=boom, out=lines.append)
    assert rc == 0
    runs = [l for l in lines if l.startswith(("would run", "skip"))]
    assert len(runs) == 12
    names = [Path(l.split()[-1]).name for l in runs]
    assert names == [R.config_name(a, s) for a, s in R.jobs()]


def test_real_run_refused_when_gate_fails_and_test_split_refused():
    def boom(*a, **k):
        raise AssertionError("must not launch")
    bad = [G.Check("implementation_committed_clean", False, "x")]
    assert R.run_all(ROOT, gate_checks=bad, popen=boom, out=lambda *_: None) == 2
    assert R.run_all(ROOT, split="test", gate_checks=[], popen=boom, out=lambda *_: None) == 2
    assert R.run_all(ROOT, dry_run=True, gate_checks=bad, popen=boom, out=lambda *_: None) == 3   # lists, gate closed


def test_cli_run_test_refused_exit_2():
    spec = importlib.util.spec_from_file_location("_run_ext", ROOT / "scripts/run_external_confirmation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["run", "--test"]) == 2
    assert mod.main(["run", "--split", "test"]) == 2


def test_real_gate_refuses_test_while_unauthorized():
    checks = G.run_gate(ROOT, split="test", scope="A", git=False, heavy=False)
    failed = {c.name for c in checks if not c.ok}
    assert "test_authorized_and_final" in failed
    with pytest.raises(PermissionError):
        G.authorize_external_test(ROOT)


# ------------------------------------------------------------------ gate failure modes (synthetic inputs)
def test_state_checks():
    m = {"freeze_state": "draft", "test_set_authorized": False}
    assert all(c.ok for c in G.check_state(m, split="validation"))
    assert {c.name for c in G.check_state(m, split="test") if not c.ok} == {"test_authorized_and_final"}
    inc = {"freeze_state": "draft", "test_set_authorized": True}     # authorized without a final freeze
    assert "state_consistent" in {c.name for c in G.check_state(inc, split="validation") if not c.ok}
    assert "state_consistent" in {c.name for c in G.check_state(inc, split="test") if not c.ok}
    fin = {"freeze_state": "final", "test_set_authorized": True}
    assert all(c.ok for c in G.check_state(fin, split="test"))
    assert "validation_phase_test_locked" in {c.name for c in G.check_state(fin, split="validation") if not c.ok}
    assert not all(c.ok for c in G.check_state({"freeze_state": "weird", "test_set_authorized": False}, split="validation"))


def _mini_repo(tmp_path, n_universe=6, n_store=6):
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "configs/external").mkdir(parents=True)
    (repo / "data").mkdir()
    (repo / "docs/P.md").write_text("protocol")
    uni = np.arange(100, 100 + n_universe, dtype=np.int64)
    np.savez(repo / "data/loci.npz", train_cpg_idx=uni, heldout_cpg_idx=np.array([], np.int64))
    np.savez(repo / "data/pat.npz", train=np.arange(4), validation=np.array([4]), test=np.array([5]))
    (repo / "data/split.csv").write_text("a")
    (repo / "configs/external/map.json").write_text("{}")
    with h5py.File(repo / "data/m.h5", "w") as h:
        h["cpg_idx"] = uni
        h["beta"] = np.zeros((6, n_universe), np.float32)
    with h5py.File(repo / "data/store.h5", "w") as h:
        h["cpg_idx"] = np.arange(100, 100 + n_store, dtype=np.int64)
        h["embedding"] = np.zeros((n_store, 3), np.float32)
    ckpt = repo / "data/best.pt"
    torch.save({"epoch": 1}, ckpt)
    man = {"protocol_doc": "docs/P.md", "protocol_doc_sha256": _sha(repo / "docs/P.md"), "universe_size": n_universe,
           "files": {"patient_protocol_npz": "data/pat.npz", "locus_protocol_npz": "data/loci.npz", "dataset_h5": "data/m.h5",
                     "mapping_manifest": "configs/external/map.json", "split_csv": "data/split.csv"},
           "arms": {"a1": {"store_h5": "data/store.h5", "dim": 3, "store_sha256": _sha(repo / "data/store.h5"),
                           "matrix_yaml_sha256": _sha(repo / "data/store.h5")}},
           "phase_a_checkpoints_experiment_B": [{"arm": "a1", "seed": 17, "best_pt": "data/best.pt", "best_pt_sha256": _sha(ckpt)}]}
    man["files_sha256"] = {k: _sha(repo / v) for k, v in man["files"].items()}
    return repo, man


def _failed(checks):
    return {c.name for c in checks if not c.ok}


def test_hash_mismatch_failures(tmp_path):
    repo, man = _mini_repo(tmp_path)
    assert not _failed(G.check_file_hashes(repo, man)) and not _failed(G.check_store_hashes(repo, man))
    assert not _failed(G.check_checkpoint_hashes(repo, man))
    (repo / "docs/P.md").write_text("protocol edited")
    assert _failed(G.check_file_hashes(repo, man)) == {"protocol_doc_sha256"}
    np.savez(repo / "data/pat.npz", train=np.arange(3), validation=np.array([3, 4]), test=np.array([5]))
    np.savez(repo / "data/loci.npz", train_cpg_idx=np.arange(100, 105), heldout_cpg_idx=np.array([], np.int64))
    assert {"sha256:patient_protocol_npz", "sha256:locus_protocol_npz"} <= _failed(G.check_file_hashes(repo, man))
    with h5py.File(repo / "data/store.h5", "a") as h:
        h["embedding"][0, 0] = 1.0
    assert _failed(G.check_store_hashes(repo, man)) == {"store_sha256:a1"}
    torch.save({"epoch": 2}, repo / "data/best.pt")
    assert _failed(G.check_checkpoint_hashes(repo, man)) == {"phaseA_best_pt:a1:17"}


def test_store_must_cover_100pct_of_universe(tmp_path):
    repo, man = _mini_repo(tmp_path)
    assert not _failed(G.check_store_coverage(repo, man))
    repo2, man2 = _mini_repo(tmp_path / "x", n_universe=6, n_store=5)
    assert _failed(G.check_store_coverage(repo2, man2)) == {"coverage_100pct:a1"}


def test_universe_checks(tmp_path):
    repo, man = _mini_repo(tmp_path)
    assert not _failed(G.check_universe(repo, man))
    man["universe_size"] = 7
    assert "universe_size" in _failed(G.check_universe(repo, man))


def _cfg(**over):
    cfg = R.build_config(ARMS[0], 17, ROOT)
    for k, v in over.items():
        sec, key = k.split("__")
        cfg[sec][key] = v
    return cfg


@pytest.mark.parametrize("over,name", [
    ({"training__epochs": 119}, "epochs_120"),
    ({"training__early_stopping": {"patience": 5}}, "early_stopping_false"),
    ({"training__max_updates": 7920}, "max_updates_absent_or_null"),
    ({"training__batch_size": 16}, "batch_size_8"),
    ({"training__save_epoch_checkpoints": True}, "save_epoch_checkpoints_false"),
    ({"training__record_update_counts": False}, "record_update_counts"),
])
def test_config_budget_failure_modes(over, name):
    assert not _failed(G.check_config_budget(_cfg(), 524))
    assert name in {n.split(":", 1)[1] for n in _failed(G.check_config_budget(_cfg(**over), 524, "c"))}


def test_updates_per_epoch_is_computed_from_n_train():
    assert not _failed(G.check_config_budget(_cfg(), 524))
    bad = {n.split(":", 1)[1] for n in _failed(G.check_config_budget(_cfg(), 500, "c"))}
    assert {"updates_per_epoch_66", "total_updates_7920"} <= bad          # ceil(500/8)=63
    assert not _failed(G.check_config_budget(_cfg(), 521))               # ceil(521/8)=66 still


def test_runner_budget_numbers_from_n_train_524():
    from cpg_repr_benchmark.external.budget import epoch_budget
    b = epoch_budget(524, 8, 120)
    assert (b["updates_per_epoch"], b["total_updates"], b["validation_points"], b["max_updates"], b["early_stopping"]) == (66, 7920, 120, None, False)


def _git(repo, *a):
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _git_repo(tmp_path):
    repo = tmp_path / "g"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "frozen.txt").write_text("f")
    (repo / "impl.py").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c1")
    return repo


def test_git_checks_pass_then_each_failure(tmp_path):
    repo = _git_repo(tmp_path)
    _git(repo, "tag", "T1")
    head = _git(repo, "rev-parse", "HEAD")[:10]
    kw = {"tag": "T1", "amendment_prefix": head, "required": ["impl.py"], "frozen_tracked": ["frozen.txt"]}
    assert not _failed(G.check_git(repo, **kw))
    # tag missing
    assert {"protocol_tag_present", "amendment_commit_is_tag", "amendment_commit_ancestor_of_HEAD"} <= _failed(G.check_git(repo, **{**kw, "tag": "NOPE"}))
    # implementation file modified / untracked
    (repo / "impl.py").write_text("changed")
    assert _failed(G.check_git(repo, **kw)) == {"implementation_committed_clean"}
    _git(repo, "checkout", "--", "impl.py")
    (repo / "new_impl.py").write_text("n")
    assert _failed(G.check_git(repo, **{**kw, "required": ["impl.py", "new_impl.py"]})) == {"implementation_committed_clean"}
    # frozen file changed since the tag
    (repo / "frozen.txt").write_text("edited")
    _git(repo, "commit", "-qam", "edit frozen")
    assert "frozen_files_unchanged_since_tag" in _failed(G.check_git(repo, **kw))
    # amendment commit not an ancestor of HEAD (tag on a side branch)
    _git(repo, "checkout", "-q", "-b", "side", "T1")
    (repo / "s.txt").write_text("s")
    _git(repo, "add", "s.txt")
    _git(repo, "commit", "-qm", "side")
    _git(repo, "tag", "T2")
    _git(repo, "checkout", "-q", "main")
    side = _git(repo, "rev-parse", "T2")[:10]
    assert "amendment_commit_ancestor_of_HEAD" in _failed(G.check_git(repo, **{**kw, "tag": "T2", "amendment_prefix": side}))


def test_real_repo_has_the_historic_protocol_tags_and_ancestors():
    # v1.1 (AMENDMENT 1) is historic and must stay where it was; the v1.2 tag is created by the orchestrator (tolerated as absent here)
    out = {c.name: c.ok for c in G.check_git(ROOT, tag=G.TAG_V1_1, amendment_prefix=G.V1_1_COMMIT_PREFIX, require_clean_code=False,
                                             frozen_tracked=())}
    assert out["protocol_tag_present"] and out["amendment_commit_is_tag"] and out["amendment_commit_ancestor_of_HEAD"]
    assert G.TAG_V1_2 == "external-recon-protocol-freeze-v1.2" and G.TAG_AUTH == "external-recon-test-authorization-v1"


def test_output_root_guard(tmp_path):
    assert G.check_output_root(ROOT, "outputs/external_reconstruction_v1/benchmark").ok
    for bad in ("outputs/regulatory_confirmation_v1/benchmark", "data/derived/external_gse40279_v1/out", "configs/external/x",
                "configs/frozen", "outputs", "/tmp/elsewhere", "outputs/external_reconstruction_v1/audit/x", "data/cache/representations/y",
                "outputs/external_reconstruction_v1/../encode_atlas_v1"):
        assert not G.check_output_root(ROOT, bad).ok, bad
    for bad in ("data/derived/external_gse40279_v1/methylation.h5", "configs/frozen/regulatory_histone_dnase_v1.json",
                "outputs/external_reconstruction_v1/audit/arm_coverage.json", "data/cache/representations/s.h5"):
        with pytest.raises(PermissionError):
            G.assert_writable(ROOT, bad)
    assert G.assert_writable(ROOT, "outputs/external_reconstruction_v1/logs/x.log")


def test_required_committed_list_is_complete_and_self_consistent():
    for needed in ("src/cpg_repr_benchmark/external/gate.py", "src/cpg_repr_benchmark/training/engine.py",
                   "scripts/run_external_confirmation.py", "configs/experiments/external_confirmation"):
        assert needed in G.REQUIRED_COMMITTED
    for p in G.REQUIRED_COMMITTED:
        assert (ROOT / p).exists(), p


# ------------------------------------------------------------------ validation phase never touches test
def test_validation_phase_cannot_evaluate_test_or_overwrite(tmp_path):
    cfg = R.build_config(ARMS[0], 17, ROOT)
    enforce_patient_view(cfg)
    bad = yaml.safe_load(yaml.safe_dump(cfg))
    bad["evaluation"]["patient_view"] = "test"
    with pytest.raises(PermissionError):
        enforce_patient_view(bad)
    miss = yaml.safe_load(yaml.safe_dump(cfg))
    del miss["evaluation"]["patient_view"]
    with pytest.raises(PermissionError):
        enforce_patient_view(miss)
    run = tmp_path / "run"
    fake = lambda v, f: ({"mse": 1.0}, {"prediction": np.zeros((2, 3), np.float32)})
    el.evaluate_split(run, cfg, {"seen": "p"}, [0.5], fake)
    before = {p: _sha(p) for p in run.rglob("*") if p.is_file()}
    # unauthorized test request (TCGA hook default) is refused and writes nothing
    tcfg = R.make_test_config(ARMS[0], 17, ROOT, _write_cfg(run, cfg))
    with pytest.raises(PermissionError):
        el.evaluate_split(run, tcfg, {"seen": "p"}, [0.5], fake, authorize=el.authorizer_from_cfg(tcfg))
    assert not (run / "evaluation/test").exists()
    # even when authorized, test writes only into evaluation/test and the validation tree is bit-identical
    el.evaluate_split(run, tcfg, {"seen": "p"}, [0.5], fake, authorize=lambda: None)
    assert (run / "evaluation/test/seen/mask_0.50/metrics.json").is_file()
    assert before == {p: _sha(p) for p in (run / "evaluation/validation").rglob("*") if p.is_file()}
    # and a second test write is refused (no overwrite)
    with pytest.raises(FileExistsError):
        el.evaluate_split(run, tcfg, {"seen": "p"}, [0.5], fake, authorize=lambda: None)


def _write_cfg(run, cfg):
    run.mkdir(parents=True, exist_ok=True)
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(cfg))
    return run


def test_no_test_dir_check(tmp_path, monkeypatch):
    repo = tmp_path / "r"
    (repo / G.OUTPUT_ROOT_REL / "benchmark/x/evaluation/validation").mkdir(parents=True)
    assert not _failed(G.check_no_test_artifacts(repo, "validation"))
    (repo / G.OUTPUT_ROOT_REL / "benchmark/x/evaluation/test").mkdir()
    assert _failed(G.check_no_test_artifacts(repo, "validation")) == {"no_evaluation_test_dir"}


# ------------------------------------------------------------------ status file
def _fake_run(tmp_path, *, epochs=120, upe=66, epoch_ck=False, test_dir=False, best_epoch_in_file=1, n_train=524):
    repo = tmp_path / "repo"
    (repo / "configs/external").mkdir(parents=True, exist_ok=True)
    man = {"protocol_doc_sha256": "abc", "mask_fractions": [0.15, 0.3, 0.5, 0.7, 0.9], "universe_size": 408017,
           "files_sha256": {"patient_protocol_npz": "p", "locus_protocol_npz": "l", "dataset_h5": "d"},
           "arms": {"a1": {"store_sha256": "s"}}}
    (repo / "configs/external/gse40279_v1_freeze_manifest.json").write_text(json.dumps(man))
    rd = tmp_path / "run"
    (rd / "checkpoints").mkdir(parents=True)
    cfg = R.build_config(ARMS[0], 17, ROOT)
    (rd / "resolved_config.yaml").write_text(yaml.safe_dump(cfg))
    mse = [1.0, 0.5] + [0.5] * (epochs - 2)       # tie plateau: best epoch 1
    hist = [{"epoch": e, "train_mse": 1.0, "validation_mse": mse[e], "updates_in_epoch": upe, "updates_done": upe * (e + 1)} for e in range(epochs)]
    (rd / "history.json").write_text(json.dumps(hist))
    (rd / "experiment.json").write_text(json.dumps({"n_train_patients_rows": n_train, "n_train_loci": 408017, "code_git_commit": "abc"}))
    torch.save({"epoch": best_epoch_in_file}, rd / "checkpoints/best.pt")
    np.save(rd / "prior_logit.npy", np.zeros(3, np.float32))
    for f in man["mask_fractions"]:
        d = rd / "evaluation/validation/seen" / f"mask_{f:.2f}"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text("{}")
    if epoch_ck:
        torch.save({}, rd / "checkpoints/epoch_0000.pt")
    if test_dir:
        (rd / "evaluation/test").mkdir()
    return repo, rd


def test_status_file_content(tmp_path):
    repo, rd = _fake_run(tmp_path)
    s = R.build_status("a1", 17, rd, wall_clock_seconds=12.34, repo=repo)
    assert s["status"] == "trained_validation_frozen" and s["test_read"] is False
    assert (s["arm"], s["seed"], s["best_epoch_0based"], s["epochs_run"], s["updates_done"], s["updates_per_epoch"]) == ("a1", 17, 1, 120, 7920, 66)
    assert s["best_val_mse_at_0.50"] == 0.5 and s["best_update"] == 132 and s["wall_clock_seconds"] == 12.3
    assert s["protocol_tag"] == "external-recon-protocol-freeze-v1.1" and s["protocol_version"] == "v1.1"
    assert s["best_pt_sha256"] == _sha(rd / "checkpoints/best.pt") and s["best_pt_path"].endswith("best.pt")
    assert s["config_hash"] and s["protocol_check"]["ok"] and s["last_pt_present"] is False and s["universe_size"] == 408017
    R.write_status(rd, s)
    assert json.loads((rd / R.STATUS_FILE).read_text())["best_epoch_0based"] == 1


@pytest.mark.parametrize("kw", [{"epochs": 119}, {"upe": 65}, {"epoch_ck": True}, {"test_dir": True}, {"best_epoch_in_file": 2}, {"n_train": 523}])
def test_status_refuses_protocol_violations(tmp_path, kw):
    repo, rd = _fake_run(tmp_path, **kw)
    with pytest.raises(ValueError):
        R.build_status("a1", 17, rd, wall_clock_seconds=1.0, repo=repo)


def test_marker_and_aggregate_status(tmp_path):
    repo = tmp_path / "r"
    assert R.aggregate_status(repo)["n_trained_validation_frozen"] == 0
    assert R.plan(repo)[0][3] is False
    assert len(R.aggregate_status(repo)["runs"]) == 12
