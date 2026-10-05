"""External confirmation analysis scaffold: contrasts, sign, descriptive block separation, wording, partial results (synthetic)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_ana_ext", ROOT / "scripts/analyze_external_confirmation.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

NOISE = {"regulatory_histone_dnase": 0.05, "functional_annotations_pca": 0.06, "cpgpt_large_locus": 0.08, "deepcpg_dna_locus": 0.10}
N_PAT, T = 8, 25
FORBIDDEN = ("equivalen", "non-inferior", "noninferior", "parity", "within the margin", "margin")


def _registry():
    cols = np.arange(400)
    return pd.DataFrame({"chr": "chr1", "pos": cols * 10_000})        # 4 Mb -> 4 blocks


def _make_run(root: Path, arm: str, seed: int, split="validation", epochs=120, best_epoch=None):
    rng = np.random.default_rng(1000 + seed)                           # same targets/columns for every arm of a seed
    rd = root / f"{arm}__seed{seed}"
    mse = np.linspace(1, 0.2, epochs) if best_epoch is None else np.where(np.arange(epochs) == best_epoch, 0.1, 0.5)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "history.json").write_text(json.dumps([{"epoch": e, "validation_mse": float(mse[e])} for e in range(epochs)]))
    cols = np.stack([rng.choice(400, T, replace=False) for _ in range(N_PAT)])
    target = rng.uniform(0.2, 0.8, (N_PAT, T)).astype(np.float32)
    prior = np.full_like(target, 0.5)
    noise = np.random.default_rng(hash((arm, seed)) % 2**32).normal(0, NOISE[arm], target.shape).astype(np.float32)
    for f in A.FRACTIONS:
        pred = np.clip(target + noise * (0.5 + f), 0.01, 0.99).astype(np.float32)
        d = rd / "evaluation" / split / "seen" / f"mask_{f:.2f}"
        d.mkdir(parents=True)
        np.savez(d / "predictions.npz", prediction=pred, target=target, prior_prediction=prior, target_matrix_column=cols,
                 sample_index=np.arange(N_PAT), panel_repeat=np.zeros(N_PAT, np.int64))
        (d / "metrics.json").write_text(json.dumps({"mse": float(((pred - target) ** 2).mean()), "mae": float(np.abs(pred - target).mean()),
                                                    "mas_pcc": 0.5, "mac_pcc": 0.4, "patient_view": split, "n_pairs": int(target.size)}))
    return rd


def _loaded(tmp_path, arms=A.ARMS, seeds=A.SEEDS, **kw):
    return {(a, s): A.load_run(_make_run(tmp_path, a, s, **kw), "A", a, s) for a in arms for s in seeds}


def _run_analysis(tmp_path, loaded, **kw):
    names = [f"GSM{i}" for i in range(N_PAT)]
    return A.analyze(loaded, _registry(), names, tmp_path / "out", replicates=40, seed=17, **kw)


def test_family_structure_constants():
    assert A.MAIN_ARMS == ("regulatory_histone_dnase", "cpgpt_large_locus", "deepcpg_dna_locus")
    assert A.PRIMARY_COMPARATOR == "cpgpt_large_locus" and A.SECONDARY_COMPARATOR == "deepcpg_dna_locus"
    assert A.LEGACY_ARM == "functional_annotations_pca" and A.LEGACY_ARM not in A.MAIN_ARMS
    assert A.BLOCKS == {"primary": "cpgpt_large_locus", "secondary": "deepcpg_dna_locus", "legacy_sensitivity_control": "functional_annotations_pca"}
    assert A.DESCRIPTIVE_PAIR == {"reference": "cpgpt_large_locus", "alternative": "deepcpg_dna_locus"}


def test_primary_secondary_legacy_blocks_sign_and_separation(tmp_path):
    res = _run_analysis(tmp_path, _loaded(tmp_path, seeds=(17, 42)))
    out = tmp_path / "out"
    pri = pd.read_csv(out / "primary_contrast_per_seed.csv")
    sec = pd.read_csv(out / "secondary_contrast_per_seed.csv")
    leg = pd.read_csv(out / "legacy_sensitivity_control_contrast_per_seed.csv")
    assert set(pri.comparator) == {"cpgpt_large_locus"} and set(sec.comparator) == {"deepcpg_dna_locus"}
    assert set(leg.comparator) == {"functional_annotations_pca"}
    for d in (pri, sec, leg):
        assert set(d.candidate) == {A.CANDIDATE}
        assert set(d.mask_fraction.round(2)) == set(A.FRACTIONS) and set(d.metric) == {"mse", "mae"}
        assert {"ci_lo", "ci_hi", "rel_ci_lo", "rel_ci_hi", "d_mas_pcc", "d_mac_pcc", "reading", "block"} <= set(d.columns)
        assert np.allclose(d[d.metric == "mse"].delta, d[d.metric == "mse"].alt_value - d[d.metric == "mse"].ref_value)
    assert (pri.block == "primary").all() and (sec.block == "secondary").all() and (leg.block == "legacy_sensitivity_control").all()
    prim = pri[pri.primary]
    assert len(prim) == 2 and (prim.metric == "mse").all() and np.allclose(prim.mask_fraction, 0.5)      # one contrast x 2 seeds
    assert not sec.primary.any() and not leg.primary.any()                                                   # only the primary block is primary
    assert (prim.delta > 0).all()                                    # comparator - candidate > 0: candidate has the lower error
    assert np.allclose(prim.relative, prim.delta / prim.ref_value)
    assert res["primary"]["n_contrasts"] == 1 and "single contrast" in res["primary"]["multiplicity_adjustment"]
    assert res["primary"]["contrast"] == "cpgpt_large_locus - regulatory_histone_dnase"
    assert res["secondary"]["label"] == "secondary" and res["secondary"]["contrast"] == "deepcpg_dna_locus - regulatory_histone_dnase"
    assert res["legacy_sensitivity_control"]["in_main_inferential_comparison"] is False
    assert res["legacy_sensitivity_control"]["may_select_representation"] is False and res["main_panel"] == list(A.MAIN_ARMS)
    assert "comparator - candidate" in res["sign_convention"] and res["secondary_metrics"] == ["mae", "mas_pcc", "mac_pcc"]
    across = pd.read_csv(out / "primary_contrast_across_seeds.csv")
    row = across[across.primary & (across.metric == "mse")].iloc[0]
    assert row.n_seeds == 2 and len(json.loads(row.per_seed_delta)) == 2
    assert row.n_seeds_ci_above_0 + row.n_seeds_ci_below_0 + row.n_seeds_ci_includes_0 == 2
    assert {"mean_d_mas_pcc", "mean_d_mac_pcc"} <= set(across.columns)
    assert len(across) == 5 * 2                                      # five fractions x (mse, mae)
    assert not (out / "primary_contrasts_per_seed.csv").exists()     # old 3-contrast family files are gone
    rep = (out / "report.md").read_text()
    i1, i2, i3, i4 = (rep.index(t) for t in ("PRIMARY comparison", "SECONDARY comparison", "Descriptive block", "LEGACY SENSITIVITY CONTROL"))
    assert i1 < i2 < i3 < i4
    assert "functional_annotations_pca" not in rep[:i4].replace("`functional_annotations_pca`", "") or "Per-arm metrics" in rep[:i4]
    assert "functional" not in rep[i1:i3].split("## Per-arm metrics")[0] and "functional" not in rep[i1:i2]   # not in primary/secondary sections
    assert "not part of the main inferential comparison" in rep and "never changes the main claim" in rep


def test_descriptive_deepcpg_minus_cpgpt_is_separate(tmp_path):
    res = _run_analysis(tmp_path, _loaded(tmp_path, seeds=(17,)))
    out = tmp_path / "out"
    pdesc = pd.read_csv(out / "descriptive_deepcpg_minus_cpgpt_per_seed.csv")
    for f in ("primary", "secondary", "legacy_sensitivity_control"):
        d = pd.read_csv(out / f"{f}_contrast_per_seed.csv")
        assert not ((d.comparator == "deepcpg_dna_locus") & (d.candidate == "cpgpt_large_locus")).any()
        assert "cpgpt_large_locus" not in set(d.candidate)
    assert (pdesc.reference == "cpgpt_large_locus").all() and (pdesc.alternative == "deepcpg_dna_locus").all()
    assert (pdesc.block == "descriptive_not_inferential").all() and (pdesc.delta > 0).all()
    assert res["descriptive"]["in_inferential_family"] is False
    rep = (out / "report.md").read_text()
    assert "outside every inferential comparison" in rep and "Not part of any inferential comparison" in rep


def test_no_equivalence_wording_anywhere(tmp_path):
    _run_analysis(tmp_path, _loaded(tmp_path, seeds=(17,)))
    blob = "".join(p.read_text().lower() for p in (tmp_path / "out").glob("*") if p.suffix in (".md", ".json"))
    for w in FORBIDDEN:
        assert w not in blob, w
    assert "no decision threshold is applied" in blob
    # no wording that makes the functional arm part of the primary/main comparison
    assert "primary family" not in blob and "three primary contrasts" not in blob and "functional comparison" not in blob


def test_script_text_has_no_equivalence_or_functional_primary_wording():
    txt = (ROOT / "scripts/analyze_external_confirmation.py").read_text().lower()
    for w in ("non-inferior", "noninferior", "parity", "within the margin", "functional_annotations_pca - candidate, cpgpt"):
        assert w not in txt, w
    assert "legacy sensitivity control" in txt and "primary comparison" in txt


def test_partial_results_are_provisional_and_missing_listed(tmp_path):
    loaded = _loaded(tmp_path, seeds=(17,))
    del loaded[("cpgpt_large_locus", 17)]
    res = _run_analysis(tmp_path, loaded, missing=[("cpgpt_large_locus", 17)])
    assert res["provisional"] and "cpgpt_large_locus__seed17" in res["missing_runs"] and res["n_runs_done"] == 3
    out = tmp_path / "out"
    assert (out / "primary_contrast_per_seed.csv").read_text().strip() == ""          # primary needs cpgpt
    assert set(pd.read_csv(out / "secondary_contrast_per_seed.csv").comparator) == {"deepcpg_dna_locus"}
    assert (out / "descriptive_deepcpg_minus_cpgpt_per_seed.csv").read_text().strip() == ""   # needs both arms
    assert "PROVISIONAL" in (out / "report.md").read_text()


def test_legacy_block_is_optional_and_never_makes_a_result_provisional(tmp_path):
    loaded = _loaded(tmp_path, arms=A.MAIN_ARMS, seeds=(17,))
    res = _run_analysis(tmp_path, loaded, missing=[])
    out = tmp_path / "out"
    assert not res["provisional"] and res["legacy_runs_present"] == []
    assert (out / "legacy_sensitivity_control_contrast_per_seed.csv").read_text().strip() == ""
    assert "LEGACY SENSITIVITY CONTROL:" not in (out / "report.md").read_text()
    # discover: a missing functional run is not reported as missing
    _, missing = A.discover("A", ROOT, tmp_path / "nothing")
    assert all(arm != "functional_annotations_pca" for arm, _ in missing) and len(missing) == 9


def test_convergence_descriptive(tmp_path):
    rd = _make_run(tmp_path, "regulatory_histone_dnase", 17, best_epoch=115)
    row = A.convergence_row(A.load_run(rd, "A", "regulatory_histone_dnase", 17))
    assert row["best_epoch_0based"] == 115 and row["best_epoch_in_last_10"] is True and row["epochs_run"] == 120
    rd2 = _make_run(tmp_path, "functional_annotations_pca", 17, best_epoch=50)
    assert A.convergence_row(A.load_run(rd2, "A", "functional_annotations_pca", 17))["best_epoch_in_last_10"] is False
    assert A.convergence_row(A.load_run(rd2, "A", "functional_annotations_pca", 17)) ["slope_last10_pct_per_epoch"] == pytest.approx(0.0, abs=1e-9)
    # B runs carry no history -> no convergence row
    d = tmp_path / "b"
    _make_run(d, "regulatory_histone_dnase", 17)
    assert A.convergence_row(A.load_run(d / "regulatory_histone_dnase__seed17", "B_strict", "regulatory_histone_dnase", 17)) is None


def test_reading_rule():
    assert "CI above 0" in A.reading(0.1, 0.3, 0.2) and "CI below 0" in A.reading(-0.3, -0.1, -0.2) and A.reading(-0.1, 0.1, 0.0) == "CI includes 0"


def test_test_split_refused_unless_authorized_and_final():
    draft = {"freeze_state": "draft", "test_set_authorized": False}
    A.check_split_allowed(draft, "validation")
    with pytest.raises(PermissionError):
        A.check_split_allowed(draft, "test")
    with pytest.raises(PermissionError):
        A.check_split_allowed({"freeze_state": "draft", "test_set_authorized": True}, "test")
    with pytest.raises(PermissionError):
        A.check_split_allowed({"freeze_state": "final", "test_set_authorized": False}, "test")
    assert A.main(["--split", "test"]) == 2           # real manifest: locked


def test_load_run_refuses_wrong_split(tmp_path):
    rd = _make_run(tmp_path, "regulatory_histone_dnase", 17, split="validation")
    mfile = rd / "evaluation/validation/seen/mask_0.50/metrics.json"
    m = json.loads(mfile.read_text())
    m["patient_view"] = "test"
    mfile.write_text(json.dumps(m))
    with pytest.raises(PermissionError):
        A.load_run(rd, "A", "regulatory_histone_dnase", 17)


def test_pairing_mismatch_is_fatal(tmp_path):
    loaded = _loaded(tmp_path, arms=("regulatory_histone_dnase", "functional_annotations_pca"), seeds=(17,))
    p = loaded[("functional_annotations_pca", 17)]["paths"][0.5]
    d = dict(np.load(p))
    d["target"] = d["target"] + 0.01
    np.savez(p, **d)
    with pytest.raises((SystemExit, ValueError)):
        _run_analysis(tmp_path, loaded)


def test_discover_uses_markers_and_transfer_dirs(tmp_path):
    (tmp_path / "logs").mkdir()
    rd = _make_run(tmp_path / "runs", "regulatory_histone_dnase", 17)
    (tmp_path / "logs/regulatory_histone_dnase__seed17.done").write_text(json.dumps({"run_dir": str(rd)}))
    done, missing = A.discover("A", ROOT, tmp_path)
    assert list(done) == [("regulatory_histone_dnase", 17)] and len(missing) == 8      # 9 main runs expected; legacy never "missing"
    base = tmp_path / "tr" / "B_strict" / "cpgpt_large_locus" / "seed_42"
    base.mkdir(parents=True)
    (base / "transfer.done").write_text("done")
    done, _ = A.discover("B_strict", ROOT, tmp_path / "tr")
    assert list(done) == [("cpgpt_large_locus", 42)]
