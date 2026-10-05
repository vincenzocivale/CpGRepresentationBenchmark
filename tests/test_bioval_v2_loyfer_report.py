"""Consistency check of the Loyfer audit report: it must equal a fresh mechanical generation from the audit outputs, and the sha256 manifest of its sources must be unchanged."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/biological_validation_v2/loyfer_failure_audit/report_inputs/REPORT_INPUTS.json"


@pytest.mark.skipif(not OUT.is_file() or not (ROOT / "docs/LOYFER_PRIMARY_FAILURE_AUDIT_sources.json").is_file(), reason="audit outputs/report not present")
def test_report_is_mechanically_reproducible_from_sources():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/bioval_v2/make_loyfer_audit_report.py"), "--check"], capture_output=True, text=True, cwd=ROOT, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
