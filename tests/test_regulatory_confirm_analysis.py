import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

_spec = importlib.util.spec_from_file_location(
    "analyze_confirm", Path(__file__).resolve().parents[1] / "scripts" / "analyze_regulatory_confirm.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

H, HD, CLEAN = mod.ARMS
N_PAT, N_LOC = 12, 40


def _registry():
    return pd.DataFrame({"chr": ["chr1"] * N_LOC, "pos": np.arange(N_LOC) * 400_000})


def _names():
    return [f"TCGA-AA-{i:04d}-01" for i in range(N_PAT)]


def make_run(root: Path, arm, seed, *, mse_scale=1.0, epochs=30, best=None, stopped=False, view="validation",
             done=True):
    d = root / "benchmark" / f"{arm}_{seed}" / "run"
    (d / "evaluation/seen/mask_0.50").mkdir(parents=True)
    rng = np.random.default_rng(0)  # identical targets/prior/panels for every run
    target = rng.random((N_PAT, N_LOC)).astype(np.float32)
    prior = np.clip(target + rng.normal(0, .2, target.shape), 0, 1).astype(np.float32)
    noise = np.random.default_rng(seed + ARMS_IDX[arm]).normal(0, 1, target.shape)
    pred = np.clip(target + noise * .1 * mse_scale, 0, 1).astype(np.float32)
    np.savez(d / "evaluation/seen/mask_0.50/predictions.npz", sample_index=np.arange(N_PAT),
             target_matrix_column=np.tile(np.arange(N_LOC), (N_PAT, 1)), panel_repeat=np.zeros((N_PAT, N_LOC), int),
             target=target, prior_prediction=prior, prediction=pred)
    mse = float(np.mean((pred - target) ** 2))
    (d / "evaluation/seen/mask_0.50/metrics.json").write_text(json.dumps({"mse": mse, "mae": float(np.mean(np.abs(pred - target))), "mas_pcc": .9, "mac_pcc": .3, "n_pairs": target.size,
               "patient_view": view, "mask_fraction": 0.5}))
    (d / "resolved_config.yaml").write_text(yaml.safe_dump({"evaluation": {"patient_view": view, "require_patient_view": view, "mask_seed": mod.MASK_SEED},
                    "training": {"seed": seed}}))
    curve = mse * (1 + 1.0 / (np.arange(epochs) + 1))
    if best is not None:  # force best epoch by making the curve rise afterwards
        curve = np.where(np.arange(epochs) > best, curve[best] + 1e-3 + np.arange(epochs) * 1e-6, curve)
        curve[best] = curve.min() - 1e-4
    hist = [{"epoch": i, "train_mse": float(c), "validation_mse": float(c), "validation_mae": float(c) * 5,
             "validation_mas_pcc": .9, "validation_mac_pcc": .3} for i, c in enumerate(curve)]
    (d / "history.json").write_text(json.dumps(hist))
    (d / "early_stopping.json").write_text(json.dumps({"patience": 10, "min_delta_rel": 1e-4, "max_epochs": 80, "epochs_run": epochs,
               "best_epoch": int(np.argmin(curve)), "stopped_early": stopped,
               "stopped_epoch": epochs - 1 if stopped else None}))
    (d / "experiment.json").write_text("{}")
    (d / "summary.json").write_text("{}")
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / f"{arm}__seed{seed}.log").write_text(f"x\nRUN_DIR={d}\n")
    if done:
        (root / "logs" / f"{arm}__seed{seed}.done").write_text("done\n")
    return d


ARMS_IDX = {H: 0, HD: 1, CLEAN: 2}


def test_partial_results_are_provisional_and_idempotent(tmp_path):
    root = tmp_path / "runs"
    for arm in (H, HD):
        make_run(root, arm, 17, mse_scale={H: 1.0, HD: 0.9}[arm])
    make_run(root, H, 42)
    make_run(root, CLEAN, 97, done=False)  # not finished -> must be ignored
    out = tmp_path / "out"
    res = mod.analyze(root, out, tmp_path / "report.md", _registry(), _names(), replicates=50, seed=3)
    assert res["provisional"] and res["n_runs_done"] == 3
    assert (H, 42) not in [(a, s) for a, s in []] and "regulatory_clean__seed97" in res["missing_runs"]
    pm = pd.read_csv(out / "per_run_metrics.csv")
    assert len(pm) == 3 and set(pm.seed) == {17, 42}
    assert res["rule_ii_iii"] == {} and res["rule_i"]["n_seeds"] == 1
    assert "PROVISIONAL" in (tmp_path / "report.md").read_text()
    first = {p.name: p.read_bytes() for p in out.iterdir() if p.suffix in (".csv", ".json")}
    mod.analyze(root, out, tmp_path / "report2.md", _registry(), _names(), replicates=50, seed=3)
    assert first == {p.name: p.read_bytes() for p in out.iterdir() if p.suffix in (".csv", ".json")}
    assert (tmp_path / "report.md").read_text() == (tmp_path / "report2.md").read_text().replace("report2", "report")


def test_convergence_flag_and_curve_stats():
    assert mod.convergence_flag(79, 80) and mod.convergence_flag(70, 80) and not mod.convergence_flag(69, 80)
    s = mod.curve_stats(np.linspace(1, 0.9, 80))
    assert s["val_mse_epoch30"] == pytest.approx(np.linspace(1, 0.9, 80)[29]) and s["slope_last10_pct_per_epoch"] < 0
    assert np.isnan(mod.curve_stats(np.ones(20))["val_mse_epoch30"])


def test_not_converged_run_marks_provisional_and_sensitivity(tmp_path):
    root = tmp_path / "runs"
    for arm, sc in ((H, 1.0), (HD, .9), (CLEAN, .9)):
        for seed in mod.SEEDS:
            make_run(root, arm, seed, mse_scale=sc, epochs=20, best=19 if arm == H and seed == 17 else 5)
    res = mod.analyze(root, tmp_path / "o", tmp_path / "r.md", _registry(), _names(), replicates=50, seed=1)
    pm = pd.read_csv(tmp_path / "o/per_run_metrics.csv")
    assert pm[(pm.arm == H) & (pm.seed == 17)].not_converged.iloc[0]
    assert res["n_runs_done"] == 9 and res["provisional"] and "not converged" in res["provisional_reasons"][0]
    assert "sensitivity" in res


def test_all_converged_and_complete_is_final(tmp_path):
    root = tmp_path / "runs"
    for arm, sc in ((H, 1.0), (HD, .9), (CLEAN, .9)):
        for seed in mod.SEEDS:
            make_run(root, arm, seed, mse_scale=sc, epochs=40, best=10)
    res = mod.analyze(root, tmp_path / "o", None, _registry(), _names(), replicates=50, seed=1)
    assert not res["provisional"] and "pairing" not in res
    assert all(v["all_identical"] for v in json.loads((tmp_path / "o/pairing_checks.json").read_text()).values())


def test_validation_only_assertion_fails_on_test_view(tmp_path):
    root = tmp_path / "runs"
    make_run(root, H, 17, view="test")
    with pytest.raises(PermissionError):
        mod.load_run(H, 17, mod.discover(root)[0][(H, 17)])


def seed_rec(rel, dmae=None, hi=None, gap=0.01, dmse=None):
    return {"rel_mse": rel, "d_mse": rel * 0.02 if dmse is None else dmse, "d_mae": rel * .1 if dmae is None else dmae,
            "mse_ci_hi": (rel * .02 + .0001 if rel < 0 else abs(rel)) if hi is None else hi, "mse_ci_lo": -1,
            "mae_ci_hi": rel * .1 + 1e-5, "gap_prior": gap}


def rule(si, siii, **kw):
    return mod.apply_decision_rule(si, siii, n_runs_done=9, **kw)


def test_verdict_branches():
    hd_ok = [seed_rec(-.02)] * 3
    hd_zero = [seed_rec(.001, dmae=.001, hi=.002)] * 3
    hd_mixed = [seed_rec(-.02), seed_rec(.01, dmae=.01, hi=.02), seed_rec(-.02)]
    cl_none = [seed_rec(-.001)] * 3
    # gap_prior 0.007, dMSE -0.0002 -> 2.9% of gap -> biologically non-negligible
    cl_good = [seed_rec(-.012, dmse=-.0002, gap=.007)] * 3
    cl_tiny = [seed_rec(-.006, dmse=-.00002, gap=.007)] * 3
    r = rule(hd_ok, cl_none)
    assert r["verdict"] == "Histone+DNase preferred" and r["rule_i"]["ge_1pct_flag"] and r["rule_i"]["PASS"]
    assert rule([seed_rec(-.005, hi=-1e-5)] * 3, cl_none)["rule_i"]["label"].startswith("SUPPORTED-SMALL")
    assert rule(hd_ok, cl_good)["verdict"] == "Clean promoted" and not rule(hd_ok, cl_good)["provisional"]
    t = rule(hd_ok, cl_tiny)
    assert t["verdict"] == "Histone+DNase preferred" and not t["rule_ii_iii"]["biologically_non_negligible"]
    unstable = [seed_rec(-.012, dmse=-.0002, gap=.007), seed_rec(-.012, dmse=-.0002, gap=.007),
                seed_rec(.001, dmae=.001, hi=.002, dmse=.00002, gap=.007)]
    assert rule(hd_ok, unstable)["verdict"] == "Histone+DNase preferred" and rule(hd_ok, unstable)["rule_ii_iii"]["prefer_HD_by_rule_ii"]
    assert rule(hd_zero, cl_none)["verdict"] == "Histone sufficient"
    assert rule(hd_mixed, cl_none)["verdict"] == "inconclusive"
    ci_fail = [seed_rec(-.02, hi=.001)] * 3  # consistent sign but CI includes 0
    assert rule(ci_fail, cl_none)["verdict"] == "inconclusive"
    # MAE disagreeing blocks Clean promotion
    mae_bad = [dict(seed_rec(-.012, dmse=-.0002, gap=.007), d_mae=.001)] * 3
    assert rule(hd_ok, mae_bad)["verdict"] != "Clean promoted"
    assert rule([], [])["verdict"] == "not evaluable"
    assert mod.apply_decision_rule(hd_ok[:2], cl_none[:2], n_runs_done=6)["provisional"]
    assert rule(hd_ok, cl_none, any_not_converged=True)["provisional"]


def test_pcc_cannot_enter_verdict():
    params = inspect.signature(mod.apply_decision_rule).parameters
    assert not any("pcc" in p.lower() for p in params)
    src = inspect.getsource(mod.apply_decision_rule).lower()
    assert "mas_pcc" not in src.replace("mas-pcc / mac-pcc", "") and "mac_pcc" not in src
    si, sc = [seed_rec(.001, dmae=.001, hi=.002)] * 3, [seed_rec(-.001)] * 3
    a = rule(si, sc)
    for s in si + sc:
        s["d_mas_pcc"], s["d_mac_pcc"] = 0.5, 0.5  # extra PCC-like fields must be ignored
    assert rule(si, sc)["verdict"] == a["verdict"]
