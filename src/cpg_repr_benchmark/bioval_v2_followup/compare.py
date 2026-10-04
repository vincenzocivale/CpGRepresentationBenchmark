"""Descriptive paired contrasts (no Holm) and the operational 'confirms / qualitatively changes the primary' verdict."""
from __future__ import annotations

import numpy as np

from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import contrast

SAME, FLIP_UNRESOLVED, FLIP_RESOLVED = "same_sign", "sign_flip_CI_includes_0", "sign_flip_CI_excludes_0"
CONFIRMS, CHANGES, WEAKENS = "CONFIRMS", "QUALITATIVELY_CHANGES", "INCONCLUSIVE_SIGN_FLIP_NOT_RESOLVED"


def paired_contrast(point_a, rep_a, point_b, rep_b) -> dict:
    """delta = reference(regulatory) - comparator; raw two-sided bootstrap p; descriptive only (no Holm, no claim)."""
    c = contrast(point_a, rep_a, point_b, rep_b)
    c["descriptive_only_no_holm"] = True
    return c


def contrast_category(delta_primary: float, delta_sens: float, ci_lo: float, ci_hi: float) -> str:
    if np.sign(delta_primary) == np.sign(delta_sens) or delta_sens == 0:
        return SAME
    return FLIP_RESOLVED if (ci_lo > 0 or ci_hi < 0) else FLIP_UNRESOLVED


def ordering(values: dict) -> list[str]:
    return [k for k, _ in sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))]


def item_verdict(primary_deltas: dict, sens_rows: dict, primary_values: dict, sens_values: dict) -> dict:
    """primary_deltas {comp: delta}; sens_rows {comp: contrast dict}; *_values {arm: value} (all 4 arms).

    CONFIRMS = all comparator contrasts have the same sign as in the primary; QUALITATIVELY_CHANGES = at least one contrast
    flips sign with a CI that excludes 0; otherwise INCONCLUSIVE. ``ordering_identical`` is reported alongside."""
    cats = {c: contrast_category(primary_deltas[c], r["delta"], r["ci_lo"], r["ci_hi"]) for c, r in sens_rows.items()}
    if any(v == FLIP_RESOLVED for v in cats.values()):
        v = CHANGES
    elif any(v == FLIP_UNRESOLVED for v in cats.values()):
        v = WEAKENS
    else:
        v = CONFIRMS
    return {"verdict": v, "per_contrast": cats, "ordering_primary": ordering(primary_values),
            "ordering_sensitivity": ordering(sens_values),
            "ordering_identical": ordering(primary_values) == ordering(sens_values)}
