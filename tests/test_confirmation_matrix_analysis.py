"""Synthetic-only tests of scripts/analyze_confirmation_matrix.py (no real runs, no test patients)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cpg_repr_benchmark.experiments import confirmation_matrix as cm

REPO = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("analyze_cm", REPO / "scripts/analyze_confirmation_matrix.py")
mod = importlib.util.module_from_spec(_s)
_s.loader.exec_module(mod)

N_PAT, N_LOC, EPOCHS = 12, 40, 12
CAND, A, B = "regulatory_histone_dnase", "functional_annotations_pca", "deepcpg_dna_locus"
SCALE = {CAND: 0.8, A: 1.0, B: 1.3}  # candidate has the lowest error


def spec_(seeds=(17, 42)):
    arms = [{"id": i, "role": "main", "status": "not_run"} for i in (CAND, A, B)]
    return {"seeds": list(seeds), "mask_fractions": cm.FROZEN_FRACTIONS, "arms": arms,
            "protocol": {**cm.FROZEN_PROTOCOL, "max_epochs": EPOCHS}, "test_set_authorized": False, "freeze_state": "draft"}


def registry():
    return pd.DataFrame({"chr": ["chr1"] * N_LOC, "pos": np.arange(N_LOC) * 400_000})


def names():
    return [f"TCGA-AA-{i:04d}-01" for i in range(N_PAT)]


def make_run(root: Path, arm, seed, *, view="validation", epochs=EPOCHS, done=True, layout="legacy"):
    d = root / "benchmark" / f"{arm}_{seed}" / "run"
    rng = np.random.default_rng(0)
    target = rng.random((N_PAT, N_LOC)).astype(np.float32)
    prior = np.clip(target + rng.normal(0, .2, target.shape), 0, 1).astype(np.float32)
    for f in cm.FROZEN_FRACTIONS:
        k = d / ("evaluation/validation/seen" if layout == "split_dirs" else "evaluation/seen") / f"mask_{f:.2f}"
        k.mkdir(parents=True)
        noise = np.random.default_rng(seed + int(f * 100)).normal(0, 1, target.shape) * (1 + f)
        pred = np.clip(target + noise * .1 * SCALE[arm], 0, 1).astype(np.float32)
        np.savez(k / "predictions.npz", sample_index=np.arange(N_PAT), target_matrix_column=np.tile(np.arange(N_LOC), (N_PAT, 1)),
                 panel_repeat=np.zeros((N_PAT, N_LOC), int), target=target, prior_prediction=prior, prediction=pred)
        (k / "metrics.json").write_text(json.dumps({"mse": float(np.mean((pred - target) ** 2)), "mae": float(np.mean(np.abs(pred - target))),
                                                    "mas_pcc": .9, "mac_pcc": .3, "patient_view": view, "mask_fraction": f}))
    (d / "resolved_config.yaml").write_text(yaml.safe_dump({
        "evaluation": {"patient_view": view, "require_patient_view": view, "mask_seed": cm.FROZEN_PROTOCOL["mask_seed"],
                       **({"output_layout": "split_dirs"} if layout == "split_dirs" else {})},
        "training": {"seed": seed, "epochs": EPOCHS, "early_stopping": False}}))
    curve = 1.0 / (np.arange(epochs) + 1)
    (d / "history.json").write_text(json.dumps([{"epoch": i, "validation_mse": float(c)} for i, c in enumerate(curve)]))
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / f"{arm}__seed{seed}.log").write_text(f"RUN_DIR={d}\n")
    if done:
        (root / "logs" / f"{arm}__seed{seed}.done").write_text("done\n")


def test_full_synthetic_analysis(tmp_path):
    root = tmp_path / "runs"
    for arm in (CAND, A, B):
        for seed in (17, 42):
            make_run(root, arm, seed)
    res = mod.analyze(spec_(), root, tmp_path / "o", registry(), names(), replicates=40, seed=3)
    assert res["n_runs_done"] == 6 and not res["provisional"]
    pc = pd.read_csv(tmp_path / "o/paired_contrasts.csv")
    # 2 seeds x 5 fractions x 2 comparators x 2 metrics
    assert len(pc) == 40 and set(pc.mask_fraction) == set(cm.FROZEN_FRACTIONS) and set(pc.reference) == {CAND}
    assert (pc[pc.metric == "mse"].delta > 0).all()  # comparators worse than the candidate -> positive delta
    assert pc[pc.primary].mask_fraction.eq(0.5).all() and len(pc[pc.primary]) == 4
    sm = pd.read_csv(tmp_path / "o/across_seeds_summary.csv")
    assert (sm.k_candidate_better == 2).all() and sm.n_seeds.eq(2).all()
    pairing = json.loads((tmp_path / "o/pairing_checks.json").read_text())
    assert all(v["all_identical"] for v in pairing.values())


def test_partial_results_provisional_and_unfinished_ignored(tmp_path):
    root = tmp_path / "runs"
    make_run(root, CAND, 17)
    make_run(root, A, 17)
    make_run(root, B, 17, done=False)
    res = mod.analyze(spec_(), root, tmp_path / "o", registry(), names(), replicates=20, seed=1)
    assert res["provisional"] and res["n_runs_done"] == 2 and f"{B}__seed17" in res["missing_runs"]


def test_non_validation_or_wrong_epochs_rejected(tmp_path):
    root = tmp_path / "runs"
    make_run(root, CAND, 17, view="test")
    with pytest.raises(PermissionError):
        mod.load_run(CAND, 17, mod.discover(root, [CAND], [17])[0][(CAND, 17)], spec_(), "validation")
    root2 = tmp_path / "runs2"
    make_run(root2, CAND, 17, epochs=EPOCHS - 2)
    with pytest.raises(ValueError):
        mod.load_run(CAND, 17, mod.discover(root2, [CAND], [17])[0][(CAND, 17)], spec_(), "validation")


def test_split_test_refused_unless_authorized_complete_green():
    s = spec_()
    mod.check_split_allowed(s, "validation")
    with pytest.raises(PermissionError):
        mod.check_split_allowed(s, "test", [])
    s["test_set_authorized"] = True
    with pytest.raises(PermissionError):
        mod.check_split_allowed(s, "test", [])  # still draft
    s["freeze_state"] = "final"
    with pytest.raises(PermissionError):  # green audit but no test layout in this scaffold
        mod.check_split_allowed(s, "test", [])
    with pytest.raises(PermissionError):
        mod.check_split_allowed(s, "test", [cm.Result("X", "s", False, "m")])


def test_cli_test_split_refused_exit_code():
    assert mod.main(["--split", "test"]) == 2


def test_analysis_reads_split_dirs_layout(tmp_path):
    """New confirmation analysis reads evaluation/validation/...; identical numbers to the legacy layout."""
    r_split, r_legacy = tmp_path / "split", tmp_path / "legacy"
    for arm in (CAND, A, B):
        for seed in (17, 42):
            make_run(r_split, arm, seed, layout="split_dirs")
            make_run(r_legacy, arm, seed)
    res = mod.analyze(spec_(), r_split, tmp_path / "o1", registry(), names(), replicates=20, seed=3)
    mod.analyze(spec_(), r_legacy, tmp_path / "o2", registry(), names(), replicates=20, seed=3)
    assert res["n_runs_done"] == 6 and not res["provisional"]
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "o1/paired_contrasts.csv"), pd.read_csv(tmp_path / "o2/paired_contrasts.csv"))
