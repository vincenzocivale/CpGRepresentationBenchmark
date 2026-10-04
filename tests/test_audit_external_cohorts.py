"""Cheap synthetic checks for scripts/audit_external_cohorts.py helpers (no biological data)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("audit_external_cohorts", Path(__file__).resolve().parents[1] / "scripts/audit_external_cohorts.py")
mod = importlib.util.module_from_spec(_spec)
sys.modules["audit_external_cohorts"] = mod
_spec.loader.exec_module(mod)


def test_coord_key_matches_external_cpg_idx_convention():
    k = mod.coord_key(np.array(["chr1", "chr22", "chrX", "chrUn"]), np.array([15865, 5, 7, 9]))
    assert k.tolist() == [1_000_015_865, 22_000_000_005, 23_000_000_007, -1]


def test_overlap_counts_chromosome_groups_and_zero_loci():
    uni_chr = np.array([1, 5, 19, 20, 22])
    uni_keys = uni_chr * 1_000_000_000 + np.array([10, 20, 30, 40, 50])
    zero = np.array([False, True, False, True, False])
    cohort = np.array([uni_keys[0], uni_keys[1], uni_keys[3], 7_000_000_001, -1, uni_keys[1]])
    o = mod.overlap_counts(cohort, uni_keys, uni_chr, zero)
    assert (o["covered"], o["covered_chr1_19"], o["covered_chr20_22"]) == (3, 2, 1)
    assert (o["covered_zero_embedding"], o["cohort_loci_not_in_universe"]) == (2, 1)


def test_beta_summary_nan_and_range():
    b = np.array([[0.0, 0.1, 0.9, np.nan], [1.0, 1.2, 0.5, 0.95]])
    s = mod.beta_summary(b)
    assert abs(s["nan_fraction"] - 1 / 8) < 1e-12
    assert abs(s["outside_0_1_fraction"] - 1 / 7) < 1e-12
    assert s["per_cpg_nan"]["n_cpg_all_nan"] == 0 and s["per_sample_nan"]["max"] == 0.25


def test_tcga_id_check():
    r = mod.tcga_id_check(["GSM1", "TCGA-AB-1234", "x"], {"TCGA-AB-1234"})
    assert r["n_tcga_pattern"] == 1 and r["n_exact_overlap_with_tcga_names"] == 1
