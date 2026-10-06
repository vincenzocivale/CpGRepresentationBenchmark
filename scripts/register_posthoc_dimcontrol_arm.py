#!/usr/bin/env python3
"""Append a built dimension-control store to configs/posthoc/dimcontrol_arms.yaml (id, store path, sha256, dim, dtype, provenance).
  python scripts/register_posthoc_dimcontrol_arm.py <arm_id> <store.h5>"""
import hashlib, sys
from pathlib import Path
import h5py, yaml
ROOT = Path(__file__).resolve().parents[1]
arm, store = sys.argv[1], Path(sys.argv[2]).resolve()
Y = ROOT / "configs/posthoc/dimcontrol_arms.yaml"
doc = yaml.safe_load(Y.read_text()) if Y.exists() else {"arms": []}
assert arm not in {a["id"] for a in doc["arms"]}, f"{arm} already registered"
h = hashlib.sha256()
with open(store, "rb") as f:
    while b := f.read(1 << 24):
        h.update(b)
with h5py.File(store) as f:
    shape, dtype = f["embedding"].shape, str(f["embedding"].dtype)
    assert shape[0] == 408399
doc["arms"].append({"id": arm, "posthoc_comparator": True, "role": "posthoc_comparator", "family": "regulatory_dimcontrol",
    "store_h5": str(store.relative_to(ROOT)), "sha256": h.hexdigest(), "dim": int(shape[1]), "dtype": dtype,
    "provenance": {"source": "regulatory_histone_dnase (2,492 binary ENCODE tracks), label-free, patient-independent; post-hoc compression diagnostic",
                   "supervision": "none", "locus_fit_scope": "chr1-19 fit, chr20-22 transform-only", "patient_specific": False, "genome_build": "GRCh38"}})
header = "# POST-HOC dimension/compression-control arms (diagnostic: is compression the bottleneck of the SVD-256 regulatory arm?)\n"
Y.write_text(header + yaml.safe_dump(doc, sort_keys=False))
print(arm, h.hexdigest()[:16], shape, dtype)
