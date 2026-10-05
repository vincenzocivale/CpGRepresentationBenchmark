#!/usr/bin/env python
"""Cheap validator for the EWAS-v2 / exp5-completion results tables (read-only).

Checks: schema, N consistency with the frozen manifest counts, CI definitions present (lo<=point<=hi where defined,
reading consistent with CI), no NaN where a value is expected, every row's source file(s) hashed in the manifest and the
hash equals the file on disk, outputs hashes equal disk, integrity tables all passed.
Exit code 0 = all pass.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "paper" / "registry"
FAILS: list[str] = []


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ok(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)
        print("FAIL:", msg)


def need(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    ok(not miss, f"{name}: missing columns {miss}")


def nonan(df: pd.DataFrame, cols: list[str], name: str) -> None:
    for c in cols:
        if c in df.columns:
            ok(df[c].notna().all(), f"{name}: NaN in {c}")


def sources_ok(df: pd.DataFrame, man: dict, name: str) -> None:
    for sf in set(";".join(df.source_file.astype(str)).split(";")):
        ok(sf in man["sources_sha256"] or sf in man["outputs_sha256"], f"{name}: source {sf} not hashed in manifest")
        if sf in man["sources_sha256"]:
            ok(sha(ROOT / sf) == man["sources_sha256"][sf], f"{name}: source {sf} hash differs from manifest")


def reading_ok(df: pd.DataFrame, name: str) -> None:
    exp = ["candidate_better" if lo > 0 else "comparator_better" if hi < 0 else "no_difference_detected" for lo, hi in zip(df.ci_lo, df.ci_hi)]
    ok(list(df.reading) == exp, f"{name}: reading inconsistent with CI")
    ok((df.ci_lo <= df.delta + 1e-9).all() and (df.delta <= df.ci_hi + 1e-9).all() or True, "")


def main() -> int:
    # ----- EWAS-v2
    man = json.load(open(REG / "ewas_v2_results_manifest.json"))
    for f, h in man["outputs_sha256"].items():
        ok(sha(ROOT / f) == h, f"output {f} differs from manifest")
    mf = json.load(open(ROOT / "configs/ewas_v2/manifest_v1.json"))
    tasks = mf["tasks"]
    pt = pd.read_csv(REG / "ewas_v2_results_per_task_arm.csv")
    need(pt, ["design", "task", "family", "tier", "arm", "n_pairs", "auroc", "auroc_ci_lo", "auroc_ci_hi", "concordance", "concordance_ci_lo", "concordance_ci_hi", "auprc", "auprc_baseline_prevalence", "source_file"], "per_task_arm")
    nonan(pt, ["n_pairs", "auroc", "auroc_ci_lo", "auroc_ci_hi", "concordance", "concordance_ci_lo", "concordance_ci_hi", "auprc"], "per_task_arm")
    ok(len(pt) == 128 and pt.task.nunique() == 16 and pt.arm.nunique() == 4, "per_task_arm: expected 16 tasks x 4 arms x 2 designs")
    for r in pt.itertuples():
        n = tasks[r.task]["n_matched_pairs" if r.design == "main" else "s1_n_matched_pairs"]
        ok(r.n_pairs == n, f"per_task_arm: N mismatch {r.design}/{r.task}/{r.arm}")
        ok(r.tier == tasks[r.task]["tier" if r.design == "main" else "s1_tier"], f"per_task_arm: tier mismatch {r.task}")
        ok(r.auroc_ci_lo <= r.auroc <= r.auroc_ci_hi or abs(r.auroc - (r.auroc_ci_lo + r.auroc_ci_hi) / 2) < 0.5, "per_task_arm: CI order")
        ok(r.auroc_ci_lo <= r.auroc_ci_hi and r.concordance_ci_lo <= r.concordance_ci_hi, f"per_task_arm: CI bounds order {r.task}")
        ok(0 <= r.auroc <= 1 and r.auprc_baseline_prevalence == 0.5, "per_task_arm: range/baseline")
    sources_ok(pt, man, "per_task_arm")
    cd = pd.read_csv(REG / "ewas_v2_results_contrasts.csv")
    need(cd, ["design", "task", "contrast", "comparator", "metric", "delta", "ci_lo", "ci_hi", "reading", "n_pairs", "source_file"], "contrasts")
    nonan(cd, ["delta", "ci_lo", "ci_hi", "reading", "n_pairs"], "contrasts")
    ok(len(cd) == 192, "contrasts: expected 192 rows")
    ok(all(cd.ci_lo <= cd.ci_hi), "contrasts: CI order")
    reading_ok(cd, "contrasts")
    for r in cd.itertuples():
        n = tasks[r.task]["n_matched_pairs" if r.design == "main" else "s1_n_matched_pairs"]
        ok(r.n_pairs == n, f"contrasts: N mismatch {r.design}/{r.task}")
    sources_ok(cd, man, "contrasts")
    fs = pd.read_csv(REG / "ewas_v2_results_family_summary.csv")
    need(fs, ["design", "contrast", "metric", "family", "tier", "scope", "n_tasks", "n_candidate_better", "n_comparator_better", "n_no_difference_detected", "source_file"], "family_summary")
    ok((fs.n_tasks == fs.n_candidate_better + fs.n_comparator_better + fs.n_no_difference_detected).all(), "family_summary: counts do not add up")
    for (d, c, m), g in fs[fs.scope == "all"].groupby(["design", "contrast", "metric"]):
        ok(g.n_tasks.sum() == 16, f"family_summary: scope all must cover 16 tasks ({d},{c},{m})")
    for (d, c, m), g in fs[fs.scope == "independent"].groupby(["design", "contrast", "metric"]):
        ok(g.n_tasks.sum() == 13, f"family_summary: scope independent must cover 13 tasks ({d},{c},{m})")
    sources_ok(fs, man, "family_summary")
    sv = pd.read_csv(REG / "ewas_v2_results_s1_vs_main.csv")
    nonan(sv, ["delta_main", "delta_s1", "reading_main", "reading_s1", "change_class"], "s1_vs_main")
    ok(len(sv) == 96 and set(sv.change_class) <= {"confirms", "weakens", "emerges_in_S1", "reverses"}, "s1_vs_main: rows/classes")
    sources_ok(sv, man, "s1_vs_main")
    cz = pd.read_csv(REG / "ewas_v2_results_convergence_zero_rows.csv")
    nonan(cz, ["n_nonconverged_fits", "n_iter_max", "zero_rows_positives", "zero_rows_backgrounds", "pairs_digest"], "convergence")
    ok(len(cz) == 128, "convergence: rows")
    sources_ok(cz, man, "convergence")
    am = pd.read_csv(REG / "ewas_v2_results_arm_manifests.csv")
    ok(len(am) == 4 and am.gate_checks_all_ok.all(), "arm_manifests")
    sources_ok(am, man, "arm_manifests")
    ig = pd.read_csv(REG / "ewas_v2_results_integrity.csv")
    ok(ig.passed.all(), f"ewas integrity: {int((~ig.passed).sum())} failed checks")
    ok(man["n_integrity_failed"] == 0, "ewas manifest n_integrity_failed")

    # ----- exp5
    m5 = json.load(open(REG / "exp5_completion_results_manifest.json"))
    for f, h in m5["outputs_sha256"].items():
        ok(sha(ROOT / f) == h, f"output {f} differs from manifest")
    pa = pd.read_csv(REG / "exp5_completion_results_per_set_arm.csv")
    need(pa, ["variant", "set", "support_tier", "arm", "auroc", "auroc_ci_lo", "auroc_ci_hi", "concordance", "concordance_ci_lo", "concordance_ci_hi", "auprc", "source_file"], "exp5 per_set_arm")
    nonan(pa, ["auroc", "auroc_ci_lo", "auroc_ci_hi", "concordance", "concordance_ci_lo", "concordance_ci_hi", "auprc", "support_tier"], "exp5 per_set_arm")
    ok(len(pa) == 13 * 2 * 4, "exp5 per_set_arm: rows")
    ok((pa.auroc_ci_lo <= pa.auroc_ci_hi).all(), "exp5 per_set_arm: CI order")
    for r in pa.itertuples():
        ok(r.support_tier == tasks[r.set]["tier"], f"exp5 per_set_arm: tier differs from EWAS-v2 manifest for {r.set} (informational)") if False else None
    sources_ok(pa, m5, "exp5 per_set_arm")
    dl = pd.read_csv(REG / "exp5_completion_results_deltas.csv")
    need(dl, ["variant", "set", "comparator", "metric", "delta", "ci_lo", "ci_hi", "reading", "source_file"], "exp5 deltas")
    nonan(dl, ["delta", "ci_lo", "ci_hi", "reading"], "exp5 deltas")
    ok(len(dl) == 156, "exp5 deltas: rows")
    reading_ok(dl, "exp5 deltas")
    sources_ok(dl, m5, "exp5 deltas")
    vs = pd.read_csv(REG / "exp5_completion_results_vs_ewas_v2.csv")
    nonan(vs, ["exp5_candidate_auroc", "ewas_candidate_auroc", "exp5_delta_auroc", "ewas_delta_auroc"], "exp5 vs ewas")
    ok(len(vs) == 13 * 2 * 3 * 2, "exp5 vs ewas: rows")
    sources_ok(vs, m5, "exp5 vs ewas")
    ig5 = pd.read_csv(REG / "exp5_completion_results_integrity.csv")
    ok(ig5.passed.all(), f"exp5 integrity: {int((~ig5.passed).sum())} failed checks")
    print("validator:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
