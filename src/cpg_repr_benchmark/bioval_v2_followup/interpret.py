"""EXPLORATORY POST-HOC: mechanical A/B/C reading rules, fixed in the registration BEFORE any real computation.

Inputs are produced by the paired chromosome-block bootstrap (95% percentile CI); every threshold below is registered.
 A: regulatory representation lacks long-range methylation-program information.
 B: information present but the cosine geometry does not expose it.
 C: comparator advantage materially due to methylation pretraining/provenance.
"""
from __future__ import annotations

R2_NEAR_BASELINE = 0.02     # ridge R2 <= this = "near the per-group-mean baseline"
GEO_IMPROVEMENT = 0.03      # minimum within-arm gain (alternative geometry minus cosine) in the pooled primary Spearman
SEQ = ("cpgpt_large_locus", "deepcpg_dna_locus")
REF = "regulatory_histone_dnase_v1"
FUNC = "functional_annotations_pca"


def _below(row):   # Delta (ref - other) negative with CI excluding 0
    return row["ci_hi"] < 0


def _above(row):
    return row["ci_lo"] > 0


def ridge_outcome(values: dict, cand_vs_seq: dict) -> str:
    """values {arm: M_ridge point}; cand_vs_seq {seq_arm: {delta, ci_lo, ci_hi}} (candidate - seq)."""
    cand = values[REF]
    if all(v <= R2_NEAR_BASELINE for v in values.values()):
        return "UNINFORMATIVE"
    n_below = sum(_below(cand_vs_seq[s]) for s in SEQ)
    n_above = sum(_above(cand_vs_seq[s]) for s in SEQ)
    if n_below == 2:
        return "ABSENT" if cand <= R2_NEAR_BASELINE else "LOWER"
    if n_below == 1:
        return "MIXED"
    return "HIGHER" if n_above >= 1 else "COMPARABLE"


def geometry_outcome(within: dict, gap: dict) -> str:
    """within {g: {delta, ci_lo, ci_hi}} = candidate(g) - candidate(cosine) on the pooled primary statistic;
    gap {g: {seq_arm: {delta, ci_lo, ci_hi}}} = candidate(g) - seq(g). CLOSES if for some g the candidate is not significantly
    below either sequence arm under g AND g improves the candidate by >= GEO_IMPROVEMENT (CI > 0); IMPROVES_ONLY if some g
    improves the candidate meaningfully but the gap remains; else NONE."""
    improves = [g for g, r in within.items() if r["delta"] >= GEO_IMPROVEMENT and _above(r)]
    closes = [g for g in improves if not any(_below(gap[g][s]) for s in SEQ)]
    return "CLOSES" if closes else ("IMPROVES_ONLY" if improves else "NONE")


def functional_outcome(func_vs_seq: dict) -> str:
    """func_vs_seq {seq_arm: {delta, ci_lo, ci_hi}} = functional - seq on the ridge metric."""
    return "BELOW_BOTH" if all(_below(func_vs_seq[s]) for s in SEQ) else "NOT_BELOW"


def interpret(ro: str, go: str, fo: str, po: str = "DIRECT_METHYLATION_SUPERVISION") -> dict:
    """Mechanical combination table (registered). PO is documentary (provenance audit): both sequence arms = DIRECT."""
    A = B = C = "not supported"
    note = []
    if ro == "UNINFORMATIVE":
        A = B = C = "not adjudicable (no representation exceeds the near-baseline ridge R2)"
    elif ro == "ABSENT":
        A = "supported (strong)"
        if go == "CLOSES":
            note.append("ANOMALY: geometry closes the gap although ridge is near baseline; re-inspect")
        C = "consistent, confounded with A" if (fo == "BELOW_BOTH" and po.startswith("DIRECT")) else "not supported"
    elif ro == "LOWER":
        A = "supported (partial: information weaker than in sequence arms)"
        B = {"CLOSES": "supported (partial)", "IMPROVES_ONLY": "weak", "NONE": "not supported"}[go]
        C = "consistent, confounded with A" if (fo == "BELOW_BOTH" and po.startswith("DIRECT")) else "not supported"
    elif ro == "MIXED":
        A = "partial / unresolved"
        B = {"CLOSES": "partial", "IMPROVES_ONLY": "weak", "NONE": "not supported"}[go]
        C = "unresolved"
        note.append("candidate significantly below exactly one sequence arm: no single reading")
    else:  # COMPARABLE / HIGHER
        A = "not supported"
        B = {"CLOSES": "supported (strong)", "IMPROVES_ONLY": "supported (partial)",
             "NONE": "supported as 'linearly decodable, not exposed by any registered similarity'"}[go]
        C = "not supported by ridge evidence (comparator advantage disappears under supervised readout)"
    return {"ridge_outcome": ro, "geometry_outcome": go, "functional_outcome": fo, "provenance_outcome": po,
            "A": A, "B": B, "C": C, "notes": note}
