"""Part 11: operational evidence rules (thresholds in registry.py; fixed before any result). Pure functions of the audit outputs."""
from __future__ import annotations

import numpy as np

from . import registry as rg


def rule_bug(step1: dict) -> dict:
    ok = step1.get("verdict") in rg.ACCEPTED_VERDICTS
    return {"flag": not ok, "evidence": {"step1_verdict": step1.get("verdict"), "failed_checks": step1.get("failed_checks", [])},
            "caveat": rg.VERDICT_CAVEAT, "raw_level_B_executed": bool(step1.get("raw_level_B_executed", False)),
            "bug_absent_statement_scope": "no bug downstream of the per-sample counts; per-sample counts / .pat parsing NOT verified"}


def rule_target_limitation(ceiling_primary, rel_gt1mb, rel_inter, topband: dict, precision_deltas: dict, precision_order_changed: bool) -> dict:
    a = bool(np.isfinite(ceiling_primary) and ceiling_primary < rg.CEILING_MIN)
    b = bool((np.isfinite(rel_gt1mb) and rel_gt1mb < rg.REL_STRATUM_MIN) or (np.isfinite(rel_inter) and rel_inter < rg.REL_STRATUM_MIN))
    c = bool(topband.get("dominated", False))
    d = bool(any(abs(v) > rg.PRECISION_DELTA for v in precision_deltas.values() if np.isfinite(v)) or precision_order_changed)
    return {"flag": a or b or c or d, "sub_rules": {"a_ceiling_lt_0.30": a, "b_stratum_reliability_lt_0.5": b,
                                                   "c_topband_dominated_by_near_constant_or_small_amplitude": c,
                                                   "d_float16_precision_changes_primary": d},
            "evidence": {"ceiling_primary": ceiling_primary, "rel_gt1Mb": rel_gt1mb, "rel_inter": rel_inter, "topband": topband,
                         "precision_deltas": precision_deltas, "precision_order_changed": precision_order_changed}}


def recovers(value, null_hi) -> bool:
    return bool(np.isfinite(value) and np.isfinite(null_hi) and value >= null_hi + rg.M_REC)


def rule_compression_geometry(native, null_hi, raw: dict, oof: dict, delta_ci: dict, pc_support: dict | None = None) -> dict:
    """raw / oof: name -> primary-style value for the candidate; delta_ci: name -> (lo, hi) of paired block-bootstrap CI of (value - native)."""
    cand = {**{f"raw:{k}": v for k, v in raw.items()}, **{f"oof:{k}": v for k, v in oof.items()}}
    rec = {k: recovers(v, null_hi) and (delta_ci.get(k) is not None and delta_ci[k][0] > 0) for k, v in cand.items()}
    native_fails = not recovers(native, null_hi)
    flag = bool(native_fails and any(rec.values()))
    return {"flag": flag, "sources_recovering": sorted(k for k, v in rec.items() if v), "native_recovers": not native_fails,
            "support_pc_removal": bool(pc_support and any(pc_support.values())),
            "evidence": {"native": native, "null_hi": null_hi, "candidates": cand, "delta_ci": delta_ci, "pc_removal_support": pc_support}}


def rule_information_absence(null_hi, candidate_values: dict) -> dict:
    none_recover = all(not recovers(v, null_hi) for v in candidate_values.values())
    return {"flag": bool(none_recover), "evidence": {"null_hi": null_hi, "candidate_values": candidate_values, "margin": rg.M_REC}}


def rule_comparator_advantage(comp_minus_cand: dict, null_hi_by_arm: dict, comp_values: dict) -> dict:
    """comp_minus_cand[comparator][evaluation] = (delta, ci_lo, ci_hi) of (comparator - candidate); comp_values[comparator][evaluation]
    = comparator's statistic. STRONG: comparator above candidate with CI > 0 AND above NULL_HI+M_REC in ALL fair evaluations;
    PARTIAL: native cosine plus at least one but not all; NONE otherwise."""
    out = {}
    for comp, evals in comp_minus_cand.items():
        wins = {}
        for ev, (delta, lo, hi) in evals.items():
            wins[ev] = bool(np.isfinite(lo) and lo > 0 and recovers(comp_values[comp][ev], null_hi_by_arm[comp]))
        n = sum(wins.values())
        level = "STRONG" if n == len(wins) and n > 0 else ("PARTIAL" if n >= 1 else "NONE")
        out[comp] = {"level": level, "wins": wins}
    flag = any(v["level"] == "STRONG" for v in out.values())
    return {"flag": flag, "per_comparator": out}


def classify(flags: dict) -> dict:
    """Multiple classes can coexist; INFORMATION_ABSENCE and COMPRESSION_GEOMETRY are mutually exclusive by construction."""
    names = {"BUG": "BUG", "TARGET_LIMITATION": "TARGET_LIMITATION", "COMPRESSION_GEOMETRY": "COMPRESSION_GEOMETRY",
             "INFORMATION_ABSENCE": "INFORMATION_ABSENCE", "COMPARATOR_ADVANTAGE": "COMPARATOR_ADVANTAGE"}
    active = sorted(k for k in names if flags.get(k, {}).get("flag"))
    if "COMPRESSION_GEOMETRY" in active and "INFORMATION_ABSENCE" in active:
        raise AssertionError("COMPRESSION_GEOMETRY and INFORMATION_ABSENCE cannot both hold (inconsistent inputs)")
    return {"classes": active, "flags": flags}
