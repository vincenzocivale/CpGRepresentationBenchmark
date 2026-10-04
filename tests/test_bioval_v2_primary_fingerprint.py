"""Guard for the closed biological_validation_v2: frozen primary results fingerprint and closure-doc consistency."""
from __future__ import annotations

from pathlib import Path

import pytest

from cpg_repr_benchmark.bioval_v2_followup import gate, registry

ROOT = Path(__file__).resolve().parents[1]
REGISTERED = "38b9da12700a8fc83dc08f5a01aaa76b61f0d5097bffe395d8eb5225f1f02094"
LABEL = "NOT_EXECUTED_REQUIRES_NEW_PREREGISTRATION_OR_ARTIFACTS"
REPORT = ROOT / "docs" / "BIOLOGICAL_VALIDATION_V2_FINAL_REPORT.md"


def _tree(root: Path) -> Path:
    base = gate.base_dir(root)
    base.mkdir(parents=True)
    for rel in ("reg/endpoints/a.json", "reg/contrasts/b.csv", "report/c.md", "reg/run_manifest.json"):
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).write_text(rel)
    return base


def test_registered_constant_matches_closure_value():
    assert gate.FROZEN_RESULTS_SHA256 == REGISTERED


def test_real_primary_fingerprint_unchanged():
    base = gate.base_dir(ROOT)
    if not base.is_dir():
        pytest.skip(f"outputs tree absent on this machine: {base}")
    assert gate.frozen_results_fingerprint(ROOT) == REGISTERED, "FROZEN PRIMARY RESULTS TREE CHANGED"


def test_fingerprint_excludes_followup_dirs_and_tracks_primary(tmp_path):
    base = _tree(tmp_path)
    h0 = gate.frozen_results_fingerprint(tmp_path)
    for d in ("sensitivity", "exploratory_posthoc_loyfer/figures_followup"):
        (base / d).mkdir(parents=True, exist_ok=True)
        (base / d / "x.png").write_text("new")
    assert gate.frozen_results_fingerprint(tmp_path) == h0
    for rel in ("reg/endpoints/a.json", "reg/contrasts/b.csv", "report/c.md"):
        p = base / rel
        old = p.read_text()
        p.write_text(old + "x")
        assert gate.frozen_results_fingerprint(tmp_path) != h0, rel
        p.write_text(old)
    (base / "report" / "new.txt").write_text("n")
    assert gate.frozen_results_fingerprint(tmp_path) != h0


def test_final_report_blocked_label_exactly_five_times():
    text = REPORT.read_text()
    assert text.count(LABEL) == 5
    rows = [ln for ln in text.splitlines() if LABEL in ln]
    assert len(rows) == 5 and all(ln.startswith("| ") for ln in rows)
    for row_id in ("10", "15", "11", "20", "21"):
        assert any(ln.startswith(f"| {row_id} |") for ln in rows), row_id
    assert "38b9da12700a8fc83dc08f5a01aaa76b61f0d5097bffe395d8eb5225f1f02094" in text


def test_followup_figures_under_exploratory_dir():
    figs = ("fig1_loyfer_variance_strata.png", "fig2_geometry_diagnostics.png",
            "fig3_ridge_r2_and_per_group.png", "fig4_cosine_vs_ridge_dissociation.png")
    base = gate.base_dir(ROOT)
    if not base.is_dir():
        pytest.skip(f"outputs tree absent on this machine: {base}")
    for f in figs:
        assert (base / "exploratory_posthoc_loyfer" / "figures_followup" / f).is_file()
    assert not (base / "report" / "figures" / figs[0]).exists()
    assert registry.EXPL_OUT.endswith("exploratory_posthoc_loyfer")
