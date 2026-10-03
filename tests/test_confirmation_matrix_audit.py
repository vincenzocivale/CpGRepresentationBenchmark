"""Synthetic tests of the confirmation-matrix audit, run gate, guards and per-run configs (no real data, no training)."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from cpg_repr_benchmark.experiments import confirmation_matrix as cm
from cpg_repr_benchmark.experiments.guards import require_test_authorization

REPO = Path(__file__).resolve().parents[1]
N = 50


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _store(path: Path, dim: int, dtype: str, idx=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = np.arange(N, dtype=np.int64) if idx is None else idx
    with h5py.File(path, "w") as h:
        h["cpg_idx"] = idx
        h["embedding"] = np.zeros((len(idx), dim), dtype=dtype)


def _arm(root: Path, aid: str, role: str, dim: int, dtype: str, **kw) -> dict:
    rel = f"stores/{aid}.h5"
    _store(root / rel, dim, dtype)
    return {"id": aid, "role": role, "status": "not_run", "family": "x", "catalog": "cat.yaml#" + aid, "store_h5": rel,
            "store_present": True, "dim": dim, "dtype": dtype, "sha256": _sha(root / rel),
            "provenance": {"patient_specific": False, "supervision": "none", "locus_fit_scope": "u", "genome_build": "GRCh38",
                           "source": "synthetic"}, **kw}


@pytest.fixture
def make_green(tmp_path):
    counter = iter(range(1000))

    def factory():
        return _green(tmp_path / f"g{next(counter)}")
    return factory


@pytest.fixture
def green(make_green):
    return make_green()


def _green(root: Path):
    root.mkdir(parents=True)
    (root / "outputs/encode_atlas_v1").mkdir(parents=True)
    np.savez(root / cm.FROZEN_PROTOCOL["loci_split_path"], train_cpg_idx=np.arange(N), heldout_cpg_idx=np.array([], dtype=int))
    np.savez(root / cm.FROZEN_PROTOCOL["patient_split_path"], train=np.arange(6), validation=np.arange(2), test=np.arange(2))
    arms = []
    for aid, dim, dt in (("regulatory_histone_dnase", 256, "float32"), ("functional_annotations_pca", 256, "float16"),
                         ("cpgpt_large_locus", 512, "float16"), ("deepcpg_dna_locus", 128, "float16")):
        arms.append(_arm(root, aid, "main", dim, dt))
    for aid in ("modern_sequence_fm", "modern_sequence_fm_alt"):
        reg = {"registration": {**{k: "filled" for k in cm.REGISTRATION_REQUIRED}, cm.REGISTRATION_ACK: True}}
        (root / f"reg_{aid}.yaml").write_text(yaml.safe_dump(reg))
        arms.append(_arm(root, aid, "main", 64, "float16", requires_registration=True, registration=f"reg_{aid}.yaml"))
    arms.append(_arm(root, "cpgpt_large_locus_256_compact", "sensitivity", 256, "float16", main_arm="cpgpt_large_locus"))
    cat = {"representations": {a["id"]: {} for a in arms}}
    (root / "cat.yaml").write_text(yaml.safe_dump(cat))
    spec = {"name": "t", "freeze_state": "draft", "test_set_authorized": False, "seeds": [17, 42, 97],
            "mask_fractions": [0.15, 0.30, 0.50, 0.70, 0.90], "protocol": dict(cm.FROZEN_PROTOCOL), "arms": arms,
            "splits": {"patient_split": {"path": cm.FROZEN_PROTOCOL["patient_split_path"],
                                         "sha256": _sha(root / cm.FROZEN_PROTOCOL["patient_split_path"])},
                       "loci_split": {"path": cm.FROZEN_PROTOCOL["loci_split_path"],
                                      "sha256": _sha(root / cm.FROZEN_PROTOCOL["loci_split_path"])}}}
    return spec, root


def _codes(spec, root):
    return {r.code for r in cm.failures(cm.audit(spec, root))}


def _arm_of(spec, aid):
    return next(a for a in spec["arms"] if a["id"] == aid)


def test_all_green_path(green):
    spec, root = green
    assert cm.failures(cm.audit(spec, root)) == []
    spec["freeze_state"], spec["test_set_authorized"] = "final", True  # complete + green -> authorization is legal
    assert cm.failures(cm.audit(spec, root)) == []


def test_missing_store_file_and_store_present_false(green, make_green):
    spec, root = green
    (root / _arm_of(spec, "deepcpg_dna_locus")["store_h5"]).unlink()
    assert "STORE_MISSING" in _codes(spec, root)
    spec2, root2 = make_green()
    _arm_of(spec2, "functional_annotations_pca")["store_present"] = False
    assert "STORE_MISSING" in _codes(spec2, root2)


def test_hash_missing_and_mismatch(green):
    spec, root = green
    _arm_of(spec, "cpgpt_large_locus")["sha256"] = None
    assert "HASH_MISSING" in _codes(spec, root)
    _arm_of(spec, "cpgpt_large_locus")["sha256"] = "0" * 64
    assert "HASH_MISMATCH" in _codes(spec, root)


def test_provenance_and_catalog_missing(green, make_green):
    spec, root = green
    del _arm_of(spec, "deepcpg_dna_locus")["provenance"]["supervision"]
    assert "PROVENANCE_MISSING" in _codes(spec, root)
    spec, root = make_green()
    _arm_of(spec, "deepcpg_dna_locus")["provenance"]["patient_specific"] = True
    assert "PROVENANCE_MISSING" in _codes(spec, root)
    spec, root = make_green()
    _arm_of(spec, "deepcpg_dna_locus")["catalog"] = "cat.yaml#nonexistent"
    assert "CATALOG_MISSING" in _codes(spec, root)


def test_pending_comparator_and_incomplete_registration(green, make_green):
    spec, root = green
    _arm_of(spec, "modern_sequence_fm")["status"] = "pending"
    assert {"PENDING_COMPARATOR"} <= _codes(spec, root)
    spec, root = make_green()
    reg = root / "reg_modern_sequence_fm.yaml"
    d = yaml.safe_load(reg.read_text())
    d["registration"]["pooling_rule"] = "<TODO>"
    reg.write_text(yaml.safe_dump(d))
    fails = [r for r in cm.failures(cm.audit(spec, root)) if r.code == "REGISTRATION_INCOMPLETE"]
    assert fails and "pooling_rule" in fails[0].message
    d["registration"]["pooling_rule"] = "ok"
    d["registration"][cm.REGISTRATION_ACK] = False
    reg.write_text(yaml.safe_dump(d))
    assert "REGISTRATION_INCOMPLETE" in _codes(spec, root)
    reg.unlink()
    assert "REGISTRATION_INCOMPLETE" in _codes(spec, root)


def test_registration_template_is_incomplete_by_design():
    t = yaml.safe_load((REPO / "configs/experiments/regulatory_confirmation_matrix/sequence_fm_registration.template.yaml").read_text())
    probs = cm.registration_problems(t)
    assert set(cm.REGISTRATION_REQUIRED) <= set(probs) and cm.REGISTRATION_ACK in probs


@pytest.mark.parametrize("key,value", [("seeds", [17, 42]), ("seeds", [1, 42, 97]), ("mask_fractions", [0.5]),
                                       ("mask_fractions", [0.15, 0.3, 0.5, 0.7, 0.95])])
def test_seeds_and_fractions_must_be_frozen(green, key, value):
    spec, root = green
    spec[key] = value
    assert ("SEEDS" if key == "seeds" else "FRACTIONS") in _codes(spec, root)


@pytest.mark.parametrize("key,value", [("max_epochs", 80), ("early_stopping", True), ("checkpoint_selection_mask_fraction", 0.3),
                                       ("learning_rate", 3e-4), ("batch_size", 32), ("panel_size", 1024), ("mask_seed", 1),
                                       ("patient_split_path", "other.npz"), ("loci_split_path", "other.npz"),
                                       ("checkpoint_selection", "last_epoch")])
def test_shared_protocol_deviation_fails(green, key, value):
    spec, root = green
    spec["protocol"][key] = value
    assert "PROTOCOL" in _codes(spec, root)


def test_per_arm_protocol_override_fails(green):
    spec, root = green
    _arm_of(spec, "deepcpg_dna_locus")["protocol_overrides"] = {"max_epochs": 60}
    fails = [r for r in cm.failures(cm.audit(spec, root)) if r.code == "PROTOCOL"]
    assert any("deepcpg_dna_locus" in r.scope for r in fails)


def test_split_hash_mismatch_fails(green):
    spec, root = green
    spec["splits"]["loci_split"]["sha256"] = "1" * 64
    assert "PROTOCOL" in _codes(spec, root)


def test_store_must_cover_panel_and_match_dim_dtype(green, make_green):
    spec, root = green
    arm = _arm_of(spec, "deepcpg_dna_locus")
    _store(root / arm["store_h5"], 128, "float16", idx=np.arange(N - 5, dtype=np.int64))  # 5 protocol loci missing
    arm["sha256"] = _sha(root / arm["store_h5"])
    fails = [r for r in cm.failures(cm.audit(spec, root)) if r.code == "COVERAGE"]
    assert fails and "5 of 50" in fails[0].message
    spec, root = make_green()
    arm = _arm_of(spec, "deepcpg_dna_locus")
    arm["dim"] = 64
    assert "COVERAGE" in _codes(spec, root)


def test_role_structure_enforced(green, make_green):
    spec, root = green
    spec["arms"] = [a for a in spec["arms"] if a["id"] != "deepcpg_dna_locus"]
    assert "ROLES" in _codes(spec, root)
    spec, root = make_green()
    _arm_of(spec, "cpgpt_large_locus_256_compact")["main_arm"] = "nope"
    assert "ROLES" in _codes(spec, root)


def test_test_authorization_state_machine(green):
    spec, root = green
    spec["test_set_authorized"] = True  # draft + authorized -> fail
    assert "TEST_AUTH" in _codes(spec, root)
    spec["freeze_state"] = "final"
    _arm_of(spec, "modern_sequence_fm")["status"] = "pending"  # complete but not green -> fail
    codes = _codes(spec, root)
    assert "TEST_AUTH" in codes and "FREEZE_STATE" in codes
    spec["freeze_state"] = "bogus"
    assert "FREEZE_STATE" in _codes(spec, root)
    spec["freeze_state"], spec["test_set_authorized"] = "draft", "yes"
    assert "TEST_AUTH" in _codes(spec, root)


def test_guard_require_test_authorization():
    with pytest.raises(PermissionError):
        require_test_authorization({"test_set_authorized": False, "freeze_state": "final"})
    with pytest.raises(PermissionError):
        require_test_authorization({"test_set_authorized": True, "freeze_state": "draft"})
    require_test_authorization({"test_set_authorized": True, "freeze_state": "final"})


def test_gate_semantics(green, make_green):
    spec, root = green
    ok_results = cm.audit(spec, root)
    assert cm.gate(spec, ok_results, split="validation", allow_incomplete_validation_only=False)[0]
    assert not cm.gate(spec, ok_results, split="test", allow_incomplete_validation_only=False)[0]  # not authorized
    _arm_of(spec, "modern_sequence_fm")["status"] = "pending"
    res = cm.audit(spec, root)
    assert not cm.gate(spec, res, split="validation", allow_incomplete_validation_only=False)[0]
    assert cm.gate(spec, res, split="validation", allow_incomplete_validation_only=True)[0]
    assert not cm.gate(spec, res, split="test", allow_incomplete_validation_only=True)[0]
    # the hatch never tolerates anything but pending comparators
    _arm_of(spec, "deepcpg_dna_locus")["sha256"] = "0" * 64
    res = cm.audit(spec, root)
    assert not cm.gate(spec, res, split="validation", allow_incomplete_validation_only=True)[0]
    # test can only open with authorization + complete + green
    spec2, root2 = make_green()
    spec2["freeze_state"], spec2["test_set_authorized"] = "final", True
    assert cm.gate(spec2, cm.audit(spec2, root2), split="test", allow_incomplete_validation_only=False)[0]


# --------------------------------------------------------------------------------------------- the real matrix
REAL = cm.load_matrix(REPO / cm.MATRIX_PATH)


def test_real_matrix_state_is_draft_and_test_locked():
    assert REAL["freeze_state"] == "draft" and REAL["test_set_authorized"] is False
    assert REAL["seeds"] == [17, 42, 97] and REAL["mask_fractions"] == cm.FROZEN_FRACTIONS
    pend = [a["id"] for a in REAL["arms"] if a["status"] == "pending"]
    assert pend == ["modern_sequence_fm", "modern_sequence_fm_alt"]
    assert {a["id"] for a in REAL["arms"] if a["role"] == "sensitivity"} == {"cpgpt_large_locus_256_compact", "deepcpg_dna_locus_hepg2"}


def test_real_matrix_audit_structure_without_data():
    res = cm.audit(REAL, REPO, check_hashes=False, check_coverage=False, check_configs=True)
    codes = sorted({(r.code, r.scope) for r in cm.failures(res)})
    assert codes == [("PENDING_COMPARATOR", "arm:modern_sequence_fm"), ("PENDING_COMPARATOR", "arm:modern_sequence_fm_alt"),
                     ("REGISTRATION_INCOMPLETE", "arm:modern_sequence_fm"), ("REGISTRATION_INCOMPLETE", "arm:modern_sequence_fm_alt")]


def test_per_run_configs_match_campaign_template():
    template = yaml.safe_load((REPO / REAL["protocol"]["template"]).read_text())
    js = cm.jobs(REAL, include_sensitivity=True)
    assert len(cm.jobs(REAL)) == 12 and len(js) == 18
    assert [s for _, s in cm.jobs(REAL)] == [17] * 4 + [42] * 4 + [97] * 4
    for arm, seed in js:
        cfg = cm.build_config(REAL, arm, seed, REPO)
        cm.check_config(cfg, template, REAL, arm["id"], seed)
        assert cfg["training"]["epochs"] == 120 and cfg["training"]["early_stopping"] is False
        assert cfg["evaluation"]["mask_fractions"] == cm.FROZEN_FRACTIONS and cfg["evaluation"]["selection_mask_fraction"] == 0.5
        assert cfg["representation"]["provenance"]["role"] == arm["role"]
    assert {a["role"] for a, _ in cm.jobs(REAL)} == {"main"}


@pytest.mark.parametrize("section,key,value", [
    ("training", "batch_size", 32), ("training", "learning_rate", 3e-4), ("training", "epochs", 80),
    ("training", "early_stopping", {"patience": 10, "min_delta_rel": 1e-4}), ("model", "hidden_dim", 256),
    ("evaluation", "selection_mask_fraction", 0.3), ("evaluation", "mask_fractions", [0.5]),
    ("evaluation", "patient_view", "test"), ("evaluation", "require_patient_view", "test"), ("evaluation", "save_predictions", False),
    ("evaluation", "mask_seed", 5), ("evaluation", "panel_size", 512),
])
def test_check_config_rejects_deviations(section, key, value):
    template = yaml.safe_load((REPO / REAL["protocol"]["template"]).read_text())
    arm = cm.job_arms(REAL)[0]
    cfg = cm.build_config(REAL, arm, 17, REPO)
    cfg[section][key] = value
    with pytest.raises((ValueError, PermissionError, KeyError)):
        cm.check_config(cfg, template, REAL, arm["id"], 17)


def test_build_config_refuses_test_split():
    with pytest.raises(PermissionError):
        cm.build_config(REAL, cm.job_arms(REAL)[0], 17, REPO, split="test")


def _runner():
    spec = importlib.util.spec_from_file_location("runner_cm", REPO / "scripts/run_regulatory_confirmation_matrix.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_refuses_test_and_incomplete_and_never_launches(monkeypatch, capsys):
    mod = _runner()
    launched = []
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: launched.append(a))
    # test split: refused (not authorized), also with the escape hatch
    assert mod.main(["run", "--split", "test"]) == 2
    assert mod.main(["run", "--split", "test", "--allow-incomplete-validation-only"]) == 2
    # validation with failing audit (pending FMs): refused by default
    assert mod.main(["run", "--dry-run"]) == 2
    # escape hatch: dry-run lists available arms only
    assert mod.main(["run", "--dry-run", "--allow-incomplete-validation-only"]) == 0
    out = capsys.readouterr().out
    assert out.count("would run [main]") == 12 and "modern_sequence" not in out.split("would run")[1]
    assert launched == []
    # PHASE A: no escape-hatch flag needed; exactly the 12 runs of the 4 fully registered arms, seed-major, nothing launched
    capsys.readouterr()
    assert mod.main(["run", "--phase", "A", "--dry-run"]) == 0
    out = capsys.readouterr().out
    lines = [x for x in out.splitlines() if x.startswith("would run")]
    names = [Path(x.split()[-1]).stem for x in lines]
    arms = ["regulatory_histone_dnase", "functional_annotations_pca", "cpgpt_large_locus", "deepcpg_dna_locus"]
    assert names == [f"{a}__seed{s}" for s in (17, 42, 97) for a in arms]
    assert mod.main(["run", "--phase", "A", "--split", "test", "--dry-run"]) == 2
    assert launched == []
