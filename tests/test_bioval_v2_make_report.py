# ruff: noqa: C408
"""Cheap synthetic tests for scripts/bioval_v2/make_report.py (no real outputs needed)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location("make_report", Path(__file__).resolve().parents[1] / "scripts" / "bioval_v2" / "make_report.py")
mr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mr)


def test_classify_and_reading_signs():
    assert mr.classify(0.1, 0.05, 0.2) == "regulatory higher"
    assert mr.classify(-0.1, -0.2, -0.05) == "regulatory lower"
    assert mr.classify(0.001, -0.002, 0.004) == "CI includes 0"
    c = "deepcpg_dna_locus"
    assert "DeepCpG-DNA better than regulatory" in mr.reading("x", c, -0.178, -0.19, -0.16, 0.008)
    assert "Regulatory better than DeepCpG-DNA" in mr.reading("x", c, 0.2, 0.19, 0.21, 0.008)
    assert "No detectable" in mr.reading("x", c, 0.0, -0.1, 0.1, None)


def test_assert_writable(tmp_path):
    root, doc = tmp_path / "root", tmp_path / "doc.md"
    mr.assert_writable(root / "report" / "figures" / "a.png", root, doc, repo=tmp_path)
    mr.assert_writable(doc, root, doc, repo=tmp_path)
    with pytest.raises(PermissionError):
        mr.assert_writable(root / "other.csv", root, doc, repo=tmp_path)
    with pytest.raises(PermissionError):
        mr.assert_writable(tmp_path / "docs" / "bioval_v2_prep" / "x.md", root, doc, repo=tmp_path)
    with pytest.raises(PermissionError):
        mr.assert_writable(tmp_path / "data" / "derived" / "bioval_v2" / "x", root, doc, repo=tmp_path)


def test_sign_check_synthetic(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    ep, st = "ep", "s"
    monkeypatch.setattr(mr, "PRIMARY", [(ep, st, "lab", "d")])
    monkeypatch.setattr(mr, "COMPARATORS", ["cmp"])
    reps = {"reg": rng.normal(0.1, 0.01, 1000), "cmp": rng.normal(0.3, 0.01, 1000)}
    arms = {}
    for name, key in [(mr.REF, "reg"), ("cmp", "cmp")]:
        d = tmp_path / name / "endpoints"
        d.mkdir(parents=True)
        np.savez(d / f"{ep}.replicates.npz", **{st: reps[key]})
        arms[name] = {"endpoints": {ep: {"stats": {st: {"value": float(reps[key].mean())}}}}}
    d = reps["reg"] - reps["cmp"]
    lo, hi = np.percentile(d, [2.5, 97.5])
    c = pd.DataFrame([dict(endpoint=ep, stat=st, delta=arms[mr.REF]["endpoints"][ep]["stats"][st]["value"] - arms["cmp"]["endpoints"][ep]["stats"][st]["value"],
                           ci_lo=lo, ci_hi=hi, k_le0=int((d <= 0).sum()))])
    out = mr.sign_check(tmp_path, arms, {"cmp": c})
    assert out[0]["sign_agrees"] and out[0]["delta_from_values"] < 0 and out[0]["max_abs_diff"] < 1e-12
    c.loc[0, "delta"] = -c.loc[0, "delta"]  # inverted sign must be detected
    assert not mr.sign_check(tmp_path, arms, {"cmp": c})[0]["sign_agrees"]
