"""Cheap synthetic tests for the read-only pre-execution audit helpers."""
import hashlib
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "preexec_audit", Path(__file__).resolve().parents[1] / "scripts/bioval_v2/preexec_audit.py")
pa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pa)


def test_verify_checksum_manifest(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"hello")
    (tmp_path / "b.txt").write_bytes(b"world")
    (tmp_path / "extra.txt").write_bytes(b"x")
    man = {"files": {"a.txt": {"sha256": hashlib.sha256(b"hello").hexdigest(), "bytes": 5},
                     "b.txt": {"sha256": "0" * 64, "bytes": 5},
                     "gone.txt": {"sha256": "0" * 64, "bytes": 1}}}
    r = pa.verify_checksum_manifest(tmp_path, man)
    assert r["ok"] == 1 and r["sha_mismatch"] == ["b.txt"] and r["missing"] == ["gone.txt"]
    assert r["extra_on_disk"] == ["extra.txt"]


def test_parse_doc_hashes():
    sha = "ab" * 32
    txt = f"| `x/y.json` | `{sha}` |\n| `d/f.parquet` (3 pairs) | `{sha}` |\n| `regulatory_activity/..._seed17_dropped_positives.parquet` | `{sha}` |\n"
    d = pa.parse_doc_hashes(txt)
    assert set(d) == {"x/y.json", "d/f.parquet",
                      "regulatory_activity/frozen_enhancer_membership_seed17_dropped_positives.parquet"}


def test_coverage_counts_only():
    c = pa.coverage(np.array([1, 2, 3]), np.array([2, 3, 3, 9]))
    assert (c["n_needed"], c["n_present"], c["n_missing"]) == (3, 2, 1) and c["missing_examples"] == [9]


def test_scan_finite(tmp_path):
    a = np.ones((7, 3), dtype=np.float16)
    a[2] = 0
    a[4, 1] = np.nan
    a[5, 0] = np.inf
    with h5py.File(tmp_path / "s.h5", "w") as h:
        h["embedding"] = a
        r = pa.scan_finite(h["embedding"], chunk_rows=3)
    assert (r["n_nan"], r["n_inf"], r["n_all_zero_rows"], r["n_rows_nonfinite"]) == (1, 1, 1, 2)


def test_guard_out_refuses_protected():
    with pytest.raises(SystemExit):
        pa._guard_out(pa.ROOT / "data/derived/bioval_v2/x.json")
    pa._guard_out(None)
