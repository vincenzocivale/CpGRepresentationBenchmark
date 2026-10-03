"""Cheap check of the frozen `regulatory_histone_dnase_v1` candidate (skips when the local data are absent)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/frozen/regulatory_histone_dnase_v1.json"
STORE = ROOT / "data/cache/representations/regulatory_histone_dnase__global_svd256__discovery_chr1_19.h5"


def test_manifest_contract_fields():
    m = json.loads(MANIFEST.read_text())
    assert m["name"] == "regulatory_histone_dnase" and m["version"] == "regulatory_histone_dnase_v1"
    assert m["status"] == "candidate_frozen"  # NOT final_confirmed
    assert m["feature_set"]["n_tracks"] == 2492 and m["dense_columns"] == 0 and m["patient_independent"]
    assert m["feature_set"]["tracks_per_assay"] == {"DNase-seq": 533, "Histone ChIP-seq": 1959}
    assert m["fit_loci"]["n"] == 388599 and m["embedding"]["shape"] == [408399, 256] and m["seed"] == 17


@pytest.mark.skipif(not STORE.exists(), reason="local regulatory store not available")
def test_verify_script_passes_without_store_sha():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/verify_frozen_candidate.py"), "--skip-store-sha"],
                       capture_output=True, text=True, cwd=ROOT, timeout=900, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
