"""Synthetic invariants of the tcga_posthoc_freeze_v2 TEST lock. No real data is read."""
from __future__ import annotations

import copy
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from cpg_repr_benchmark.experiments import eval_layout as el
from cpg_repr_benchmark.experiments import test_confirmation_v2 as t2
from cpg_repr_benchmark.experiments.guards import enforce_patient_view


def _eval_cfg(rid: str) -> dict:
    return {"evaluation": {"patient_view": "test", "output_layout": "split_dirs", "num_workers": 0, "allow_overwrite_split_dir": False,
                           "test_authorization": t2.POLICY, "test_freeze_manifest": t2.MANIFEST_REL, "test_freeze_run_id": rid,
                           "preserve_run_files": True, "mask_fractions": [0.5]}}


@pytest.fixture()
def freeze(tmp_path: Path):
    root = tmp_path
    (root / "store.h5").write_bytes(b"store")
    runs = {}
    for arm in t2.PANEL_ARMS:
        for seed in t2.SEEDS:
            rid = t2.run_id(arm, seed)
            d = root / "runs" / rid
            d.mkdir(parents=True)
            e = {"arm": arm, "seed": seed, "status": "complete", "store_h5": "store.h5", "store_sha256": t2.sha256_file(root / "store.h5")}
            for key, fn, hk in (("best_pt", "best.pt", "best_sha256"), ("resolved_config", "resolved_config.yaml", "resolved_config_sha256"),
                                ("patient_split", "patient_split.npz", "patient_split_sha256"),
                                ("locus_split", "locus_split.npz", "locus_split_sha256"), ("prior_logit", "prior_logit.npy", "prior_logit_sha256")):
                (d / fn).write_bytes(f"{rid}{fn}".encode())
                e[key] = str((d / fn).relative_to(root)); e[hk] = t2.sha256_file(d / fn)
            cfg = _eval_cfg(rid)
            ep = root / f"{rid}.yaml"
            ep.write_text(yaml.safe_dump(cfg))
            e.update(eval_config=ep.name, eval_config_sha256=t2.sha256_file(ep), eval_config_canonical_sha256=t2.canonical_cfg_sha256(cfg))
            runs[rid] = e
    man = {"freeze_state": "draft", "test_set_authorized": False, "runs": runs}
    mp = root / "manifest.yaml"
    mp.write_text(yaml.safe_dump(man))
    return root, mp, man


def _auth(root, mp, rid, git=lambda r: []):
    t2.authorize(_eval_cfg(rid), root, manifest_path=mp, git_check=git)


def _set(mp, man, **kw):
    man = {**man, **kw}
    mp.write_text(yaml.safe_dump(man))
    return man


RID = t2.run_id(t2.PANEL_ARMS[0], 17)


def test_a_draft_or_unauthorized_blocks_test(freeze):
    root, mp, man = freeze
    with pytest.raises(PermissionError):
        _auth(root, mp, RID)  # draft + false
    _set(mp, man, freeze_state="final")  # final but not authorized
    with pytest.raises(PermissionError, match="test_set_authorized"):
        _auth(root, mp, RID)
    _set(mp, man, test_set_authorized=True)  # authorized but draft
    with pytest.raises(PermissionError, match="freeze_state"):
        _auth(root, mp, RID)
    _set(mp, man, freeze_state="final", test_set_authorized=True)
    _auth(root, mp, RID)  # all green -> passes


def test_a_uncommitted_code_blocks(freeze):
    root, mp, man = freeze
    _set(mp, man, freeze_state="final", test_set_authorized=True)
    with pytest.raises(PermissionError, match="not committed"):
        _auth(root, mp, RID, git=lambda r: ["scripts/run_masking_benchmark.py: not committed/clean"])


def test_b_hash_mismatch_blocks(freeze):
    root, mp, man = freeze
    _set(mp, man, freeze_state="final", test_set_authorized=True)
    # any run's best.pt changed (not only the evaluated one)
    other = root / man["runs"][t2.run_id(t2.PANEL_ARMS[-1], 97)]["best_pt"]
    other.write_bytes(b"tampered")
    with pytest.raises(PermissionError, match="sha256 mismatch for best_pt"):
        _auth(root, mp, RID)
    other.write_bytes(f"{t2.run_id(t2.PANEL_ARMS[-1], 97)}best.pt".encode())
    _auth(root, mp, RID)
    (root / "store.h5").write_bytes(b"other")
    with pytest.raises(PermissionError, match="store"):
        _auth(root, mp, RID)


def test_b_pending_or_modified_eval_config_blocks(freeze):
    root, mp, man = freeze
    man = _set(mp, man, freeze_state="final", test_set_authorized=True)
    bad = copy.deepcopy(man)
    bad["runs"][t2.run_id(t2.PANEL_ARMS[-1], 42)] = {"arm": t2.PANEL_ARMS[-1], "seed": 42, "status": "pending"}
    mp.write_text(yaml.safe_dump(bad))
    with pytest.raises(PermissionError, match="pending"):
        _auth(root, mp, RID)
    mp.write_text(yaml.safe_dump(man))
    cfg = _eval_cfg(RID)
    cfg["evaluation"]["mask_fractions"] = [0.3]  # config differs from the frozen one
    with pytest.raises(PermissionError, match="differs from the frozen"):
        t2.authorize(cfg, root, manifest_path=mp, git_check=lambda r: [])


def test_b_config_with_validation_guard_or_overwrite_rejected(freeze):
    root, mp, man = freeze
    _set(mp, man, freeze_state="final", test_set_authorized=True)
    for k, v in (("require_patient_view", "validation"), ("allow_overwrite_split_dir", True)):
        cfg = _eval_cfg(RID)
        cfg["evaluation"][k] = v
        with pytest.raises(PermissionError):
            t2.authorize(cfg, root, manifest_path=mp, git_check=lambda r: [])


def test_c_test_output_cannot_enter_validation_dir(tmp_path):
    run = tmp_path / "run"
    (run / "evaluation" / "validation" / "seen" / "mask_0.50").mkdir(parents=True)
    keep = run / "evaluation" / "validation" / "seen" / "mask_0.50" / "metrics.json"
    keep.write_text("orig")
    with pytest.raises(PermissionError):
        el.split_output_dir(run, "test", "../../validation/seen", "mask_0.50")
    cfg = {"evaluation": {"output_layout": "split_dirs", "patient_view": "test"}}
    with pytest.raises(PermissionError, match="payload patient_view"):
        el.write_view_outputs(run, cfg, "test", "seen", "mask_0.50", {"patient_view": "validation"}, None)
    out = el.write_view_outputs(run, cfg, "test", "seen", "mask_0.50", {"patient_view": "test"}, None)
    assert out == (run / "evaluation" / "test" / "seen" / "mask_0.50").resolve()
    assert keep.read_text() == "orig"


def test_c_nonempty_test_dir_not_overwritten(tmp_path):
    run = tmp_path / "run"
    (run / "evaluation" / "test").mkdir(parents=True)
    (run / "evaluation" / "test" / "x").write_text("1")
    cfg = _eval_cfg(RID)
    with pytest.raises(FileExistsError):
        el.prepare_split_dir(run, cfg, "test", authorize=lambda: None)


def test_d_validation_only_runs_cannot_evaluate_test():
    cfg = {"evaluation": {"patient_view": "test", "require_patient_view": "validation"}}
    with pytest.raises(PermissionError):
        enforce_patient_view(cfg)
    with pytest.raises(PermissionError):
        enforce_patient_view({"evaluation": {"require_patient_view": "validation"}})  # omitted view fails too


def test_policy_is_wired_and_locked_by_default():
    cfg = _eval_cfg(RID)
    guard = el.authorizer_from_cfg(cfg)
    assert callable(guard)
    with pytest.raises(PermissionError):
        guard()  # real repo manifest is draft/false


def test_resolver_preserve_run_files(tmp_path):
    from cpg_repr_benchmark.representations.resolver import resolve_representation
    store = tmp_path / "s.h5"
    with h5py.File(store, "w") as f:
        f["cpg_idx"] = np.arange(5, dtype=np.int64)
        f["embedding"] = np.zeros((5, 3), dtype=np.float16)
    run = tmp_path / "run"
    run.mkdir()
    cfg = {"name": "x", "mode": "precomputed", "store_h5": str(store)}
    resolve_representation(cfg, registry_path=tmp_path, run_dir=run, repo_root=tmp_path, write_manifest=False)
    assert not (run / "representation_manifest.json").exists()
    resolve_representation(cfg, registry_path=tmp_path, run_dir=run, repo_root=tmp_path)
    assert (run / "representation_manifest.json").exists()
