"""Evaluation output layouts, split guards, Phase A gate/status and a 512D/128D adapter smoke test (synthetic only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cpg_repr_benchmark.experiments import confirmation_matrix as cm
from cpg_repr_benchmark.experiments import eval_layout as el
from cpg_repr_benchmark.experiments.guards import enforce_patient_view
from cpg_repr_benchmark.models.model import MaskedMethylomeReconstructor

FRACS = [0.5, 0.9]
VIEWS = {"seen": "prior"}


def _cfg(layout=None, view="validation", **ev):
    e = {"patient_view": view, "mask_seed": 5, "panel_repeats": 1, "save_predictions": True, **ev}
    if layout:
        e["output_layout"] = layout
    return {"evaluation": e, "training": {"seed": 1}}


def _fake(value):
    def fn(view, fraction):
        return ({"mse": value * fraction}, {"prediction": np.full((2, 3), value, np.float32), "target": np.zeros((2, 3), np.float32)})
    return fn


def _tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def _authorize_ok():
    return None


def test_split_layouts_are_separated(tmp_path):
    el.evaluate_split(tmp_path, _cfg("split_dirs"), VIEWS, FRACS, _fake(1.0))
    assert (tmp_path / "evaluation/validation/seen/mask_0.50/metrics.json").is_file()
    assert (tmp_path / "evaluation/validation/seen/mask_0.90/predictions.npz").is_file()
    assert not (tmp_path / "evaluation/seen").exists() and not (tmp_path / "evaluation/test").exists()
    m = json.loads((tmp_path / "evaluation/validation/seen/mask_0.50/metrics.json").read_text())
    assert m["patient_view"] == "validation" and m["mask_seed"] == 5
    # every written file is under evaluation/validation/
    assert all(k.startswith("evaluation/validation/") for k in _tree(tmp_path))
    assert el.metrics_dir(tmp_path, _cfg("split_dirs"), "validation", "seen", "mask_0.50") == tmp_path / "evaluation/validation/seen/mask_0.50"
    s = el.write_split_summary(tmp_path, _cfg("split_dirs"), "validation", {"a": 1})
    assert s == tmp_path / "evaluation/validation/summary.json" and not (tmp_path / "summary.json").exists()


def test_test_write_never_touches_validation(tmp_path):
    cfg_v = _cfg("split_dirs")
    el.evaluate_split(tmp_path, cfg_v, VIEWS, FRACS, _fake(1.0))
    el.write_split_summary(tmp_path, cfg_v, "validation", {"a": 1})
    before = {k: v for k, v in _tree(tmp_path).items()}
    # authorization monkeypatched true INSIDE this test only: proves the write lands in evaluation/test and nowhere else
    cfg_t = _cfg("split_dirs", view="test")
    el.evaluate_split(tmp_path, cfg_t, VIEWS, FRACS, _fake(7.0), authorize=_authorize_ok)
    after = _tree(tmp_path)
    assert {k: v for k, v in after.items() if k.startswith("evaluation/validation/")} == before
    new = set(after) - set(before)
    assert new and all(k.startswith("evaluation/test/") for k in new)
    # same-split rewrite fails unless explicitly allowed; validation bytes still untouched
    with pytest.raises(FileExistsError):
        el.evaluate_split(tmp_path, cfg_v, VIEWS, FRACS, _fake(2.0))
    with pytest.raises(FileExistsError):
        el.evaluate_split(tmp_path, cfg_t, VIEWS, FRACS, _fake(2.0), authorize=_authorize_ok)
    assert {k: v for k, v in _tree(tmp_path).items() if k.startswith("evaluation/validation/")} == before
    el.evaluate_split(tmp_path, _cfg("split_dirs", allow_overwrite_split_dir=True), VIEWS, FRACS, _fake(2.0))
    assert {k: v for k, v in _tree(tmp_path).items() if k.startswith("evaluation/test/")} == {k: v for k, v in after.items() if k.startswith("evaluation/test/")}


def test_cross_split_writes_are_impossible(tmp_path):
    with pytest.raises(PermissionError):  # payload of one split into the other split's dir
        el.write_view_outputs(tmp_path, _cfg("split_dirs"), "test", "seen", "mask_0.50", {"patient_view": "validation"}, None)
    with pytest.raises(PermissionError):  # path traversal into the other split
        el.split_output_dir(tmp_path, "test", "..", "validation")
    with pytest.raises(PermissionError):
        el.split_output_dir(tmp_path, "validation", "..", "test")
    with pytest.raises(PermissionError):  # config says validation, caller asks for test
        el.prepare_split_dir(tmp_path, _cfg("split_dirs"), "test", authorize=_authorize_ok)
    assert not (tmp_path / "evaluation").exists() or not any((tmp_path / "evaluation").rglob("*.json"))


def test_legacy_layout_is_default_and_unchanged(tmp_path):
    for cfg in (_cfg(None), _cfg("legacy")):
        d = tmp_path / str(id(cfg))
        res = el.evaluate_split(d, cfg, VIEWS, FRACS, _fake(1.0))
        assert (d / "evaluation/seen/mask_0.50/metrics.json").is_file() and (d / "evaluation/seen/mask_0.90/predictions.npz").is_file()
        assert not (d / "evaluation/validation").exists()
        assert el.write_split_summary(d, cfg, "validation", {"x": 1}) == d / "summary.json" and (d / "summary.json").is_file()
        assert set(res["seen"]["mask_0.50"]) == {"mse", "mask_fraction", "view", "prior_policy", "patient_view", "panel_repeats", "mask_seed"}
    # legacy never refuses an existing directory (historical behaviour) and a legacy test run needs no matrix authorization
    el.evaluate_split(tmp_path / "t", _cfg(None, view="test"), VIEWS, FRACS, _fake(1.0))
    el.evaluate_split(tmp_path / "t", _cfg(None, view="test"), VIEWS, FRACS, _fake(1.0))
    assert el.get_layout({"evaluation": {}}) == "legacy"
    with pytest.raises(ValueError):
        el.get_layout({"evaluation": {"output_layout": "nope"}})


def test_split_dirs_test_refused_when_unauthorized(tmp_path):
    cfg_t = _cfg("split_dirs", view="test")
    called = []
    with pytest.raises(PermissionError):  # real authorization path: matrix.yaml has test_set_authorized false / draft / failing audit
        el.evaluate_split(tmp_path, cfg_t, VIEWS, FRACS, lambda v, f: called.append(1))
    assert called == [] and not (tmp_path / "evaluation").exists()
    with pytest.raises(PermissionError):
        el.authorize_test_split()
    spec = cm.load_matrix(Path(__file__).resolve().parents[1] / cm.MATRIX_PATH)
    assert spec["test_set_authorized"] is False and spec["freeze_state"] == "draft"


def test_patient_view_validation_guard_still_enforced():
    enforce_patient_view({"evaluation": {"patient_view": "validation", "require_patient_view": "validation"}})
    for ev in ({"patient_view": "test", "require_patient_view": "validation"}, {"require_patient_view": "validation"},
               {"patient_view": "test", "require_patient_view": "test"}):
        with pytest.raises((PermissionError, ValueError)):
            enforce_patient_view({"evaluation": ev})
    REPO = Path(__file__).resolve().parents[1]
    for p in (REPO / cm.CONFIG_DIR).glob("*.yaml"):
        ev = yaml.safe_load(p.read_text())["evaluation"]
        assert ev["output_layout"] == "split_dirs" and ev["patient_view"] == "validation" and ev["allow_overwrite_split_dir"] is False
        enforce_patient_view(yaml.safe_load(p.read_text()))


@pytest.mark.parametrize("dim", [512, 128])
def test_reconstructor_forward_backward_native_dim(dim):
    torch.manual_seed(0)
    model = MaskedMethylomeReconstructor(raw_locus_dim=dim, locus_latent_dim=256, token_dim=256, patient_dim=256, hidden_dim=512)
    b, n, m = 2, 16, 8
    obs = torch.randn(b, n, dim).half().float()  # stores are float16
    tgt = torch.randn(b, m, dim).half().float()
    pred, _ = model(obs, torch.randn(b, n), torch.ones(b, n, dtype=torch.bool), tgt, torch.randn(b, m))
    assert pred.shape == (b, m) and torch.isfinite(pred).all()
    loss = ((pred - torch.rand(b, m)) ** 2).mean()
    loss.backward()
    grads = [p.grad for p in model.locus_adapter.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert model.locus_adapter[1].weight.shape == (256, dim)


def _mini_run(root: Path, spec: dict, arm: str, seed: int, epochs: int = 120, with_test_dir=False):
    d = root / "benchmark" / f"{arm}_{seed}" / "run"
    (d / "checkpoints").mkdir(parents=True)
    (d / "checkpoints/best.pt").write_bytes(b"ckpt-" + arm.encode())
    cfg = {"evaluation": {"patient_view": "validation", "require_patient_view": "validation", "output_layout": "split_dirs"},
           "training": {"seed": seed, "epochs": 120, "early_stopping": False}}
    (d / "resolved_config.yaml").write_text(yaml.safe_dump(cfg))
    (d / "history.json").write_text(json.dumps([{"epoch": i, "validation_mse": 1.0 + abs(i - 77)} for i in range(epochs)]))
    for f in spec["mask_fractions"]:
        k = d / "evaluation/validation/seen" / f"mask_{f:.2f}"
        k.mkdir(parents=True)
        (k / "metrics.json").write_text("{}")
    if with_test_dir:
        (d / "evaluation/test").mkdir()
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / f"{arm}__seed{seed}.log").write_text(f"noise\nRUN_DIR={d}\n")
    return d


def _spec():
    return {"seeds": [17, 42, 97], "mask_fractions": cm.FROZEN_FRACTIONS, "protocol": dict(cm.FROZEN_PROTOCOL),
            "arms": [{"id": i, "role": "main", "status": "not_run"} for i in cm.PHASE_A_ARMS]
                    + [{"id": "modern_sequence_fm", "role": "main", "status": "pending"}]}


def test_status_file_and_phase_a_aggregate(tmp_path):
    spec = _spec()
    d = _mini_run(tmp_path, spec, "regulatory_histone_dnase", 17)
    st = cm.build_status("regulatory_histone_dnase", 17, d, wall_clock_seconds=12.34, spec=spec)
    assert st["status"] == "trained_validation_frozen" and st["status"] != "confirmed" and st["test_read"] is False
    assert st["best_epoch_0based"] == 77 and st["epochs_run"] == 120 and st["best_val_mse_at_0.50"] == 1.0
    assert st["best_pt_sha256"] == hashlib.sha256(b"ckpt-regulatory_histone_dnase").hexdigest()
    assert st["protocol_check"]["ok"] and st["protocol_check"]["early_stopping_false"]
    cm.write_status(d, st)
    (tmp_path / "logs/regulatory_histone_dnase__seed17.done").write_text("done\n")
    agg = json.loads(cm.write_phase_a_status(spec, tmp_path).read_text())
    assert agg["n_expected"] == 12 and agg["n_trained_validation_frozen"] == 1 and agg["test_read"] is False
    assert [(r["arm"], r["seed"]) for r in agg["runs"][:5]] == [(a, s) for s in (17,) for a in cm.PHASE_A_ARMS] + [("regulatory_histone_dnase", 42)]
    assert agg["runs"][0]["status"] == "trained_validation_frozen" and agg["runs"][1]["status"] == "not_run"


@pytest.mark.parametrize("kw", [{"epochs": 60}, {"with_test_dir": True}])
def test_status_rejects_protocol_violations(tmp_path, kw):
    spec = _spec()
    d = _mini_run(tmp_path, spec, "deepcpg_dna_locus", 17, **kw)
    with pytest.raises(ValueError):
        cm.build_status("deepcpg_dna_locus", 17, d, wall_clock_seconds=1.0, spec=spec)


def test_phase_a_gate_tolerates_only_pending_fm_slots():
    spec = _spec()
    pend = [cm.Result("PENDING_COMPARATOR", "arm:modern_sequence_fm", False, "pending")]
    assert cm.gate(spec, pend, split="validation", phase="A")[0]
    assert not cm.gate(spec, pend, split="validation")[0]
    assert not cm.gate(spec, pend, split="test", phase="A")[0]
    bad = pend + [cm.Result("HASH_MISMATCH", "arm:cpgpt_large_locus", False, "x")]
    assert not cm.gate(spec, bad, split="validation", phase="A")[0]
    spec["arms"][0]["status"] = "pending"
    assert not cm.gate(spec, pend, split="validation", phase="A")[0]
    assert [a["id"] for a, _ in cm.jobs(_spec(), phase="A")][:4] == cm.PHASE_A_ARMS
