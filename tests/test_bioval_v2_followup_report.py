# ruff: noqa: C408
"""Cheap synthetic tests for scripts/bioval_v2/make_followup_report.py (no real outputs needed)."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "make_followup_report", Path(__file__).resolve().parents[1] / "scripts" / "bioval_v2" / "make_followup_report.py")
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)


def test_sign_class_and_verdict():
    d = dict(delta=-0.02, ci_lo=-0.03, ci_hi=-0.01)
    assert mf.sign_class(-0.1, d) == "same_sign"
    assert mf.sign_class(0.1, d) == "sign_flip_CI_excludes_0"
    assert mf.sign_class(0.1, dict(delta=-0.02, ci_lo=-0.05, ci_hi=0.01)) == "sign_flip_CI_includes_0"
    assert mf.verdict_of(["same_sign"] * 3) == "CONFIRMS"
    assert mf.verdict_of(["same_sign", "sign_flip_CI_excludes_0", "same_sign"]) == "QUALITATIVELY_CHANGES"
    assert mf.verdict_of(["same_sign", "sign_flip_CI_includes_0", "same_sign"]) == "INCONCLUSIVE"


def test_assert_writable(tmp_path):
    root, doc = tmp_path / "root", tmp_path / "doc.md"
    mf.assert_writable(root / "exploratory_posthoc_loyfer" / "figures_followup" / "a.png", root, doc, repo=tmp_path)
    mf.assert_writable(doc, root, doc, repo=tmp_path)
    for bad in [root / "report" / "summary_primary.csv", root / "report" / "figures" / "a.png", root / "contrasts" / "x.csv",
                root / "sensitivity" / "SUMMARY.csv", tmp_path / "docs" / "BIOLOGICAL_VALIDATION_V2_RESULTS.md",
                tmp_path / "data" / "derived" / "bioval_v2" / "x"]:
        with pytest.raises(PermissionError):
            mf.assert_writable(bad, root, doc, repo=tmp_path)


def test_formatting():
    assert mf.fv(0.12345, 0.1, 0.2) == "0.123 [0.100, 0.200]"
    assert mf.fv(float("nan")) == "NA"
    assert mf.fd(dict(delta=0.1, ci_lo=0.05, ci_hi=0.2, p=0.002)) == "+0.100 [+0.050, +0.200] p=0.002"
    assert mf.order_by({"a": 1, "b": 3, "c": 2}) == ["b", "c", "a"]
