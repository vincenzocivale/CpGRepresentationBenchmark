"""One-shot test evaluation + rehearsal (synthetic, CPU, tiny): layout, byte-identical validation, guards, refusals, authorization tag."""
from __future__ import annotations

import functools
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

from cpg_repr_benchmark.external import gate as G
from cpg_repr_benchmark.external import test_eval as TE
from cpg_repr_benchmark.external import transfer as T

ROOT = Path(__file__).resolve().parents[1]
N_SAMPLES, N_LOCI, DIM = 30, 40, 5
MODEL = {"locus_latent_dim": 8, "token_dim": 8, "patient_dim": 8, "hidden_dim": 8}
FRACS = [0.15, 0.30, 0.50, 0.70, 0.90]
EXPECTED = {"panel_size": 20, "batch_size": 8, "panel_repeats": 1, "mask_seed": 17001, "save_predictions": True}
ARMS = ("regulatory_histone_dnase", "functional_annotations_pca", "cpgpt_large_locus", "deepcpg_dna_locus")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _tree(root: Path) -> dict:
    return {str(p.relative_to(root)): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture()
def world(tmp_path):
    repo = tmp_path / "repo"
    rng = np.random.default_rng(0)
    ids = np.array(rng.permutation(np.arange(1000, 1000 + N_LOCI)), dtype=np.int64)
    (repo / "data").mkdir(parents=True)
    perm = rng.permutation(N_SAMPLES)
    train, val, test = np.sort(perm[:18]), np.sort(perm[18:24]), np.sort(perm[24:])
    with h5py.File(repo / "data/m.h5", "w") as h:
        h["cpg_idx"], h["beta"] = ids, rng.uniform(0.1, 0.9, (N_SAMPLES, N_LOCI)).astype(np.float32)
        h["sample_name"] = np.array([f"GSM{i}".encode() for i in range(N_SAMPLES)])
    np.savez(repo / "data/pat.npz", train=train, validation=val, test=test, sample_names=np.array([f"GSM{i}" for i in range(N_SAMPLES)]), seed=1)
    np.savez(repo / "data/loci.npz", train_cpg_idx=np.sort(ids), heldout_cpg_idx=np.array([], np.int64), seed=17001, heldout_fraction=0.0)
    arms, entries, runs = {}, [], {}
    for a_i, arm in enumerate(ARMS):
        with h5py.File(repo / f"data/store_{arm}.h5", "w") as h:
            h["cpg_idx"], h["embedding"] = ids, rng.normal(size=(N_LOCI, DIM)).astype(np.float32)
        arms[arm] = {"store_h5": f"data/store_{arm}.h5", "store_sha256": _sha(repo / f"data/store_{arm}.h5"), "dim": DIM}
        for seed in (17, 42):
            rd = repo / G.OUTPUT_ROOT_REL / "benchmark" / arm / f"seed_{seed}" / "run"
            (rd / "checkpoints").mkdir(parents=True)
            torch.manual_seed(100 * a_i + seed)
            model = T.build_model(DIM, MODEL)
            for p in model.parameters():
                torch.nn.init.normal_(p, std=0.2)
            torch.save({"model": model.state_dict(), "epoch": 3}, rd / "checkpoints/best.pt")
            prior = rng.normal(size=N_LOCI).astype(np.float32)
            np.save(rd / "prior_logit.npy", prior)
            np.savez(rd / "patient_split.npz", train=train, validation=val, test=test)
            np.savez(rd / "locus_split.npz", train_matrix_columns=np.arange(N_LOCI), heldout_matrix_columns=np.array([], np.int64))
            cfg = {"model": MODEL, "training": {"beta_epsilon": 1e-4, "seed": seed},
                   "evaluation": {**EXPECTED, "mask_fractions": FRACS, "patient_view": "validation", "output_layout": "split_dirs"}}
            (rd / "resolved_config.yaml").write_text(yaml.safe_dump(cfg))
            status = {"status": "trained_validation_frozen", "arm": arm, "seed": seed, "best_pt_sha256": _sha(rd / "checkpoints/best.pt"),
                      "best_epoch_0based": 3, "prior_logit_sha256": _sha(rd / "prior_logit.npy"),
                      "patient_protocol_sha256": _sha(repo / "data/pat.npz"), "locus_protocol_sha256": _sha(repo / "data/loci.npz"),
                      "matrix_h5_sha256": _sha(repo / "data/m.h5"), "store_sha256": arms[arm]["store_sha256"], "test_read": False}
            (rd / "confirmation_status.json").write_text(json.dumps(status))
            logs = repo / G.OUTPUT_ROOT_REL / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            (logs / f"{arm}__seed{seed}.done").write_text(json.dumps({"run_dir": str(rd)}))
            runs[(arm, seed)] = rd
            entries.append({"arm": arm, "seed": seed})
    man = {"files": {"patient_protocol_npz": "data/pat.npz", "locus_protocol_npz": "data/loci.npz", "dataset_h5": "data/m.h5"},
           "files_sha256": {"patient_protocol_npz": _sha(repo / "data/pat.npz"), "locus_protocol_npz": _sha(repo / "data/loci.npz"),
                            "dataset_h5": _sha(repo / "data/m.h5")},
           "counts": {"train": 18, "validation": 6, "test": 6}, "universe_size": N_LOCI, "mask_fractions": FRACS,
           "arms": arms, "freeze_state": "final", "test_set_authorized": True, "protocol_version": "v1.2"}
    (repo / "configs/external").mkdir(parents=True)
    (repo / "configs/external/gse40279_v1_freeze_manifest.json").write_text(json.dumps(man))
    # a pre-existing validation tree in every run (must stay byte-identical)
    for rd in runs.values():
        d = rd / "evaluation/validation/seen/mask_0.50"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text('{"mse": 0.1}')
        (d.parent.parent / "summary.json").write_text("{}")
    return {"repo": repo, "runs": runs, "test": test, "val": val, "man": man}


def _run(world, **kw):
    evaluate_fn = functools.partial(TE.evaluate_frozen_run, expected_eval=EXPECTED)
    return TE.evaluate_test_all(world["repo"], gate_checks=[], authorize=lambda: None, evaluate_fn=evaluate_fn, device="cpu",
                                seeds=(17,), out=lambda *_: None, confirm_one_shot=kw.pop("confirm_one_shot", True), **kw)


def test_jobs_main_9_and_legacy_labelled():
    assert len(TE.test_jobs()) == 9 and all(a != "functional_annotations_pca" for a, _ in TE.test_jobs())
    leg = TE.test_jobs(include_legacy=True)
    assert len(leg) == 12 and [x for x in leg if x[0] == "functional_annotations_pca"] == [("functional_annotations_pca", s) for s in (17, 42, 97)]
    assert TE.role_of("functional_annotations_pca") == "legacy_sensitivity_control" and TE.role_of("cpgpt_large_locus") == "main"


def test_e2e_layout_validation_byte_identical_and_only_test_dir_written(world):
    before_val = {rd: _tree(rd / "evaluation/validation") for rd in world["runs"].values()}
    all_before = {rd: _tree(rd) for rd in world["runs"].values()}
    assert _run(world) == 0
    for (arm, seed), rd in world["runs"].items():
        if seed != 17:
            assert not (rd / "evaluation/test").exists()
            continue
        if arm == "functional_annotations_pca":                       # legacy runs are NOT evaluated by default
            assert not (rd / "evaluation/test").exists()
            continue
        t = rd / "evaluation/test"
        for f in FRACS:
            m = json.loads((t / f"seen/mask_{f:.2f}/metrics.json").read_text())
            assert m["patient_view"] == "test" and m["mask_seed"] == 17001 and m["mask_fraction"] == f
            p = np.load(t / f"seen/mask_{f:.2f}/predictions.npz")
            assert set(np.unique(p["sample_index"])) <= set(world["test"].tolist())          # test rows only
        assert (t / "summary.json").is_file()
        rec = json.loads((t / "eval_manifest.json").read_text())
        assert rec["kind"] == "test_evaluation" and rec["one_shot"] is True and rec["role"] == "main" and rec["split"] == "test"
        assert rec["checkpoint_sha256"] == _sha(rd / "checkpoints/best.pt") and rec["weights_updated"] is False
        assert rec["checkpoint_reselected"] is False and rec["freeze_state"] == "final" and rec["test_set_authorized"] is True
        for k in ("patient_protocol_sha256", "locus_protocol_sha256", "store_sha256", "timestamp_utc", "protocol_tag", "authorization_tag"):
            assert rec[k]
        assert rec["universe_size"] == N_LOCI and rec["n_eval_patients"] == 6
        # validation byte-identical; nothing new anywhere in the run dir except evaluation/test/
        assert _tree(rd / "evaluation/validation") == before_val[rd]
        new = set(_tree(rd)) - set(all_before[rd])
        assert new and all(n.startswith("evaluation/test/") for n in new)
        assert {n: h for n, h in _tree(rd).items() if n in all_before[rd]} == all_before[rd]      # old files unchanged
    st = json.loads((world["repo"] / TE.TEST_STATUS).read_text())
    assert st["one_shot"] is True and st["include_legacy"] is False and st["n_expected"] == 3 and all(r["complete"] for r in st["runs"])
    assert (world["repo"] / TE.STARTED_LOCK).is_file()


def test_legacy_runs_only_with_flag_and_labelled(world):
    assert _run(world, include_legacy=True) == 0
    rd = world["runs"][("functional_annotations_pca", 17)]
    rec = json.loads((rd / "evaluation/test/eval_manifest.json").read_text())
    assert rec["role"] == "legacy_sensitivity_control" and rec["arm"] == "functional_annotations_pca"
    assert json.loads((world["repo"] / TE.TEST_STATUS).read_text())["n_expected"] == 4


def test_second_test_evaluation_is_refused_and_changes_nothing(world):
    assert _run(world) == 0
    snap = {rd: _tree(rd) for rd in world["runs"].values()}
    with pytest.raises(FileExistsError):
        _run(world)                                                   # evaluation/test exists
    with pytest.raises(FileExistsError):
        _run(world, include_legacy=True)                              # even for a bigger set: no overwrite of the existing ones
    assert {rd: _tree(rd) for rd in world["runs"].values()} == snap
    # resume only skips complete runs, never overwrites
    assert _run(world, resume_completed=True) == 0
    assert {rd: _tree(rd) for rd in world["runs"].values()} == snap
    # the lock alone also blocks a fresh attempt
    for rd in world["runs"].values():
        if (rd / "evaluation/test").exists():
            import shutil
            shutil.rmtree(rd / "evaluation/test")
    with pytest.raises(FileExistsError):
        _run(world)


def test_refusals_leave_no_test_dir_and_no_lock(world):
    repo = world["repo"]
    bad = [G.Check("test_authorized_and_final", False, "x")]
    assert TE.evaluate_test_all(repo, gate_checks=bad, confirm_one_shot=True, out=lambda *_: None) == 2
    assert TE.evaluate_test_all(repo, gate_checks=[], confirm_one_shot=False, authorize=lambda: None, out=lambda *_: None) == 2
    def deny():
        raise PermissionError("locked")
    with pytest.raises(PermissionError):
        TE.evaluate_test_all(repo, gate_checks=[], confirm_one_shot=True, authorize=deny, seeds=(17,), out=lambda *_: None)
    # a tampered checkpoint is caught in the PREFLIGHT, before any evaluation of any run, no lock written
    rd = world["runs"][("cpgpt_large_locus", 17)]
    with open(rd / "checkpoints/best.pt", "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="best.pt sha256"):
        _run(world)
    assert not list((repo / G.OUTPUT_ROOT_REL).rglob("test")) and not (repo / TE.STARTED_LOCK).exists()


def test_default_authorizer_is_the_real_external_gate_and_is_locked_while_draft(monkeypatch):
    with pytest.raises(PermissionError):
        G.authorize_external_test(ROOT)
    spec = importlib.util.spec_from_file_location("_run_ext2", ROOT / "scripts/run_external_confirmation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = {}
    monkeypatch.setattr(mod.R, "evaluate_test_all", lambda repo, **kw: calls.update(kw) or 2)
    assert mod.main(["test-eval", "--include-legacy", "--confirm-one-shot", "--resume-completed", "--device", "cpu"]) == 2
    assert calls == {"include_legacy": True, "confirm_one_shot": True, "resume_completed": True, "device": "cpu"}
    calls.clear()
    mod.main(["test-eval"])
    assert calls["include_legacy"] is False and calls["confirm_one_shot"] is False


def test_real_test_eval_refused_while_draft(monkeypatch):
    checks = [G.Check("test_authorized_and_final", False, "draft")]
    assert TE.evaluate_test_all(ROOT, gate_checks=checks, confirm_one_shot=True, out=lambda *_: None) == 2
    assert not list((ROOT / G.OUTPUT_ROOT_REL).rglob("test"))


# ------------------------------------------------------------------ rehearsal (same function, validation split, scratch dir)
def test_rehearsal_guards_and_reproducibility(world, tmp_path):
    repo, rd = world["repo"], world["runs"][("regulatory_histone_dnase", 17)]
    kw = {"expected_eval": EXPECTED, "device": "cpu"}
    before = _tree(rd)
    with pytest.raises(PermissionError):                                           # validation re-evaluation needs rehearsal=True
        TE.evaluate_frozen_run(repo, "regulatory_histone_dnase", 17, rd, eval_split="validation", out_run_dir=tmp_path / "s", **kw)
    with pytest.raises(PermissionError):                                           # never inside the external output root
        TE.evaluate_frozen_run(repo, "regulatory_histone_dnase", 17, rd, eval_split="validation", rehearsal=True,
                               out_run_dir=repo / G.OUTPUT_ROOT_REL / "scratch", **kw)
    with pytest.raises(PermissionError):                                           # nor the run itself
        TE.evaluate_frozen_run(repo, "regulatory_histone_dnase", 17, rd, eval_split="validation", rehearsal=True, out_run_dir=rd, **kw)
    with pytest.raises(PermissionError):                                           # a rehearsal never touches test
        TE.evaluate_frozen_run(repo, "regulatory_histone_dnase", 17, rd, eval_split="test", rehearsal=True, out_run_dir=tmp_path / "s", **kw)
    with pytest.raises(PermissionError):                                           # test without the authorization callable
        TE.evaluate_frozen_run(repo, "regulatory_histone_dnase", 17, rd, eval_split="test", **kw)
    assert _tree(rd) == before and not (rd / "evaluation/test").exists()
    r1 = TE.evaluate_frozen_run(repo, "regulatory_histone_dnase", 17, rd, eval_split="validation", rehearsal=True, out_run_dir=tmp_path / "s1", **kw)
    TE.evaluate_frozen_run(repo, "regulatory_histone_dnase", 17, rd, eval_split="validation", rehearsal=True, out_run_dir=tmp_path / "s2", **kw)
    assert r1["manifest"]["kind"] == "rehearsal_validation" and r1["manifest"]["one_shot"] is False and r1["manifest"]["test_rows_read"] is False
    for f in FRACS:
        a = json.loads((tmp_path / f"s1/evaluation/validation/seen/mask_{f:.2f}/metrics.json").read_text())
        b = json.loads((tmp_path / f"s2/evaluation/validation/seen/mask_{f:.2f}/metrics.json").read_text())
        assert a["mse"] == b["mse"] and a["patient_view"] == "validation"
        p = np.load(tmp_path / f"s1/evaluation/validation/seen/mask_{f:.2f}/predictions.npz")
        assert set(np.unique(p["sample_index"])) <= set(world["val"].tolist())      # validation rows only
    assert _tree(rd) == before                                                       # the real run is untouched
    with pytest.raises(FileExistsError):                                             # scratch dir is not reused either
        TE.evaluate_frozen_run(repo, "regulatory_histone_dnase", 17, rd, eval_split="validation", rehearsal=True, out_run_dir=tmp_path / "s1", **kw)


# ------------------------------------------------------------------ authorization tag (git)
def _git(repo, *a):
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _auth_repo(tmp_path):
    repo = tmp_path / "g"
    (repo / "configs").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    m, d = repo / "configs/manifest.json", repo / "protocol.md"
    m.write_text(json.dumps({"freeze_state": "draft", "test_set_authorized": False}))
    d.write_text("doc")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "v1.2")
    _git(repo, "tag", "P")
    return repo, m, d


def _chk(repo):
    return {c.name: c.ok for c in G.check_authorization_tag(repo, manifest_rel="configs/manifest.json", protocol_rel="protocol.md",
                                                           protocol_tag="P", auth_tag="A")}


def test_authorization_tag_checks(tmp_path):
    repo, m, _d = _auth_repo(tmp_path)
    r = _chk(repo)
    assert r["protocol_v1_2_tag_present"] and not r["authorization_tag_present"]
    m.write_text(json.dumps({"freeze_state": "final", "test_set_authorized": True}))
    _git(repo, "commit", "-qam", "promote")
    _git(repo, "tag", "A")
    r = _chk(repo)
    assert all(r.values()), r
    m.write_text(json.dumps({"freeze_state": "final", "test_set_authorized": True, "x": 1}))          # dirty after the tag
    assert not _chk(repo)["authorized_state_unchanged_since_authorization_tag"]
    _git(repo, "commit", "-qam", "edit after auth")                                                 # committed change after the tag
    assert not _chk(repo)["authorized_state_unchanged_since_authorization_tag"]


def test_authorization_tag_must_hold_a_final_authorized_manifest_and_follow_protocol_tag(tmp_path):
    repo, _m, _d = _auth_repo(tmp_path)
    (repo / "z.txt").write_text("z")
    _git(repo, "add", "z.txt")
    _git(repo, "commit", "-qm", "later")
    _git(repo, "tag", "A")                                                                           # manifest at A still draft/false
    assert not _chk(repo)["authorization_tag_manifest_final_and_authorized"]
    _git(repo, "tag", "-d", "A")
    _git(repo, "tag", "A", "P")                                                                      # same commit as the protocol tag
    r = _chk(repo)
    assert not r["authorization_tag_descends_from_protocol_tag"]


def test_test_phase_frozen_check_excludes_manifest_but_not_other_frozen_files(tmp_path):
    repo = tmp_path / "g2"
    (repo / "configs/external").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    (repo / "configs/external/manifest.json").write_text("{}")
    (repo / "configs/external/other.json").write_text("{}")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c")
    _git(repo, "tag", "T")
    spec = (("configs/external", ":(exclude)configs/external/manifest.json"),)
    kw = {"tag": "T", "required": [], "frozen_tracked": spec, "require_clean_code": False}
    (repo / "configs/external/manifest.json").write_text('{"freeze_state": "final"}')
    _git(repo, "commit", "-qam", "promote")
    assert all(c.ok for c in G.check_git(repo, **kw) if c.name == "frozen_files_unchanged_since_tag")
    (repo / "configs/external/other.json").write_text('{"x": 1}')
    _git(repo, "commit", "-qam", "tamper")
    assert not next(c for c in G.check_git(repo, **kw) if c.name == "frozen_files_unchanged_since_tag").ok
    assert G.FROZEN_TRACKED_TEST_PHASE and any(isinstance(x, tuple) for x in G.FROZEN_TRACKED_TEST_PHASE)


def test_test_gate_requires_authorization_tag_checks_in_real_repo_while_draft():
    checks = G.run_gate(ROOT, split="test", scope="A", heavy=False)
    failed = {c.name for c in checks if not c.ok}
    assert "test_authorized_and_final" in failed
    assert "authorization_tag_present" in failed or "authorization_tag_ancestor_of_HEAD" in failed
