"""Cheap synthetic checks for scripts/make_external_final_report.py (no real results needed)."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("mfr", Path(__file__).resolve().parents[1] / "scripts" / "make_external_final_report.py")
mfr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mfr)


def test_guard_refuses_outside(tmp_path):
    doc, rep = tmp_path / "doc.md", tmp_path / "report"
    mfr.guarded_write(doc, "x", doc=doc, report_dir=rep)
    mfr.guarded_write(rep / "sub" / "a.json", "{}", doc=doc, report_dir=rep)
    assert doc.read_text() == "x"
    with pytest.raises(PermissionError):
        mfr.guarded_write(tmp_path / "other.md", "x", doc=doc, report_dir=rep)
    with pytest.raises(PermissionError):
        mfr.guarded_write(rep / ".." / "escape.md", "x", doc=doc, report_dir=rep)


def test_snapshot_diff_detects_changes(tmp_path):
    (tmp_path / "benchmark" / "r").mkdir(parents=True)
    a, b = tmp_path / "benchmark" / "r" / "a.txt", tmp_path / "benchmark" / "r" / "b.txt"
    a.write_text("a"); b.write_text("b")
    snap = {"trees": ["benchmark"], "files": {f"benchmark/r/{n}.txt": {"sha256": hashlib.sha256(n.encode()).hexdigest()} for n in "ab"}}
    d = mfr.snapshot_diff(tmp_path, snap)
    assert d["changed"] == [] and d["removed"] == [] and d["n_added"] == 0
    b.write_text("changed"); (tmp_path / "benchmark" / "r" / "evaluation" / "test").mkdir(parents=True)
    (tmp_path / "benchmark" / "r" / "evaluation" / "test" / "summary.json").write_text("{}")
    a.unlink()
    d = mfr.snapshot_diff(tmp_path, snap)
    assert d["changed"] == ["benchmark/r/b.txt"] and d["removed"] == ["benchmark/r/a.txt"] and d["n_added"] == 1
    assert d["added_outside_evaluation_test"] == []


def test_pcc_tag_and_formatting():
    assert "cand higher" in mfr.pcc_tag(-0.1, -0.2, -0.05)
    assert "comp higher (seeds mixed)" in mfr.pcc_tag(0.1, -0.1, 0.2)
    assert mfr.pct(0.0182) == "+1.82%"
