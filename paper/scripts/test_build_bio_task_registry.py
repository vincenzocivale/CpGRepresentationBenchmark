#!/usr/bin/env python
"""Cheap synthetic unit test for the registry builder (no real results needed).  Run: python paper/scripts/test_build_bio_task_registry.py"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("build_reg", ROOT / "paper/scripts/build_bio_task_registry.py")
B = importlib.util.module_from_spec(spec)
sys.modules["build_reg"] = B
spec.loader.exec_module(B)


def row(task, arm_raw, metric="AUROC", score=0.6, stat="s", headline=True, **kw):
    na = B.norm_arm(arm_raw)
    r = {c: "" for c in B.REG_COLUMNS}
    r.update(suite="synthetic", task_id=task, task_name=task, stat_name=stat, is_headline=headline, biological_family="other",
             representation_raw=arm_raw, representation_id=na["id"], representation_variant=na["variant"], representation_dim=na["dim"],
             is_main_panel=na["main"], proxy_for=na["proxy_for"], metric=metric, score=score, status="frozen", endpoint_class="primary",
             circularity_vs_candidate="INDEPENDENT", circularity_vs_functional_legacy="INDEPENDENT", circularity_vs_sequence="INDEPENDENT",
             circularity_level="INDEPENDENT", positive_n=10, negative_n=10, n_items=20)
    r.update(kw)
    return r


def test_norm_arm():
    assert B.norm_arm("cpgpt_locus_large")["id"] == "cpgpt_large_locus" and B.norm_arm("cpgpt_locus_large")["main"]
    assert not B.norm_arm("fm/cpgpt_locus_large")["main"] and B.norm_arm("fm/cpgpt_locus_large")["proxy_for"] == "cpgpt_large_locus"
    assert B.norm_arm("deepcpg_hcc_native128")["dim"] == 128
    assert B.norm_arm("functional_annotations_pca", "legacy_bio_validation")["variant"].startswith("legacy_pca")
    try:
        B.norm_arm("nonsense")
    except KeyError:
        return
    raise AssertionError("unmapped arm must fail")


def test_metric_of():
    assert B.metric_of("auroc_oof__win500") == "AUROC" and B.metric_of("spearman_<1kb") == "Spearman"
    assert B.metric_of("enrichment_github_U25_only") == "enrichment" and B.metric_of("n_pairs") is None


def test_coverage_and_matrix():
    rows = [row("t1", "regulatory_histone_dnase_v1", score=0.8, ci_lo=0.7, ci_hi=0.9, ci_definition="x"),
            row("t1", "cpgpt_large_locus", score=0.7), row("t1", "deepcpg_dna_locus", score=0.6),
            row("t1", "functional_annotations_pca", score=0.85),
            row("t2", "cpgpt_large_locus", score=0.7), row("t2", "add/assay/Histone ChIP-seq", score=0.75),
            row("t2", "fm/deepcpg_dna_locus", score=0.5)]
    cov = B.coverage_table(rows).set_index("task_id")
    assert cov.loc["t1", "regulatory_histone_dnase_v1"] == "HAVE"
    assert cov.loc["t2", "regulatory_histone_dnase_v1"] == "PROXY_VARIANT_ONLY"
    assert cov.loc["t2", "cpgpt_large_locus"] == "HAVE" and cov.loc["t2", "deepcpg_dna_locus"] == "PROXY_VARIANT_ONLY"
    assert cov.loc["t1", "functional_annotations_pca_legacy"] == "HAVE" and cov.loc["t2", "functional_annotations_pca_legacy"] == "MISSING"
    reg = pd.DataFrame(rows, columns=B.REG_COLUMNS)
    con = pd.DataFrame([{**{c: "" for c in B.CONTRAST_COLUMNS}, "task_id": "t1", "stat_name": "s", "reference_id": "regulatory_histone_dnase_v1",
                         "comparator_id": "cpgpt_large_locus", "delta": 0.1, "ci_lo": 0.05, "ci_hi": 0.15, "ci_definition": "d"}],
                       columns=B.CONTRAST_COLUMNS).replace("", float("nan"))
    aud = pd.DataFrame({"task_id": ["t1", "t2"], "rigor_grade": ["MAIN_QUALITY", "SUPPLEMENTARY_ONLY"]})
    cov2 = cov.reset_index()
    mat = B.build_matrix(reg, cov2, con, aud).set_index("task_id")
    assert abs(mat.loc["t1", "cand__auroc"] - 0.8) < 1e-12 and abs(mat.loc["t1", "candidate_minus_cpgpt__auroc"] - 0.1) < 1e-12
    assert mat.loc["t1", "delta_flag"].startswith("paired") and mat.loc["t2", "delta_flag"] == "no paired CI available"
    assert mat.loc["t2", "cand__auroc"] == "" and abs(mat.loc["t2", "proxy_value_cand"] - 0.75) < 1e-12  # proxy only, never promoted
    assert mat.loc["t2", "coverage_deepcpg"] == "PROXY_VARIANT_ONLY"


def test_overlap():
    ov = B.overlaps({"a": {1, 2, 3}, "b": {3, 4}, "c": {9}})
    assert ov["a"]["partner"] == "b" and ov["a"]["shared"] == 1 and abs(ov["a"]["jaccard"] - 0.25) < 1e-12
    assert ov["c"]["shared"] == 0


def test_guard():
    try:
        B.guard("outputs/x/loyfer_failure_audit/y.csv")
    except RuntimeError:
        return
    raise AssertionError("guard must refuse the excluded directory")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
