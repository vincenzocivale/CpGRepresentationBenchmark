"""Part 2: split-half reliability / noise ceiling of the Loyfer pair-similarity target.

Unit of splitting = DONOR (the unit the frozen target aggregates over). For every group with >= 2 donors the donors are randomly
split into two equal halves (odd donor dropped); each half yields an independent group-beta profile per CpG with the SAME rule as the
frozen target (donor beta = mean of valid sample betas; group beta = unweighted mean over donors with a valid beta; a half-group is
observed iff >= 1 valid donor). Pair similarity = Pearson over the pairwise-complete groups, >= ceil(G_used * 20/39) shared, non-constant.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cpg_repr_benchmark.bioval_v2_followup import strata as st
from cpg_repr_benchmark.bioval_v2_followup import targets as tg

from . import registry as rg
from .common import spearman_fast, stat_pooled


def spearman_brown(r, k: float = 2.0):
    r = np.asarray(r, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(r > -1, k * r / (1 + (k - 1) * r), np.nan)


def ceiling_from_reliability(rho):
    """Reliability-bounded maximum plausible correlation with the target = sqrt(max(rho, 0)) (classical attenuation bound)."""
    return np.sqrt(np.maximum(np.asarray(rho, float), 0.0))


def min_shared_for(n_groups: int) -> int:
    return math.ceil(n_groups * rg.MIN_SHARED_FRACTION)


def make_halves(donors: pd.DataFrame, rep: int, min_donors: int = 2):
    """Return (half_of_donor array in {0 (A), 1 (B), -1 (dropped/unused)}, groups_used list). Deterministic: rng = default_rng([seed, rep])."""
    rng = np.random.default_rng([rg.SPLIT_SEED_BASE, rep])
    half = np.full(len(donors), -1, np.int8)
    used = []
    for g in sorted(donors.group.unique()):
        ix = np.flatnonzero((donors.group == g).to_numpy())
        if len(ix) < min_donors:
            continue
        p = rng.permutation(ix)
        h = len(ix) // 2
        half[p[:h]] = 0
        half[p[h:2 * h]] = 1
        used.append(g)
    return half, used


def half_group_betas(D, V, donors: pd.DataFrame, half, groups_used):
    """Two (N, G_used) arrays (NaN where a half-group has no valid donor)."""
    N = D.shape[1]
    out = []
    for h in (0, 1):
        B = np.full((N, len(groups_used)), np.nan)
        for gi, g in enumerate(groups_used):
            ix = np.flatnonzero(((donors.group == g).to_numpy()) & (half == h))
            k = V[ix].sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                B[:, gi] = np.where(k > 0, (D[ix] * V[ix]).sum(0) / np.maximum(k, 1), np.nan)
        out.append(B)
    return out


def half_similarities(DA, DB, ia, ib, n_groups_used):
    ms = min_shared_for(n_groups_used)
    _, ta = tg.pair_pearson(DA, ia, ib, min_shared=ms)
    _, tb = tg.pair_pearson(DB, ia, ib, min_shared=ms)
    return ta, tb


REL_CELLS = ("all", "dist_<1kb", "dist_1-10kb", "dist_10-100kb", "dist_100kb-1Mb", "dist_>1Mb", "dist_interchromosomal",
             "chrom_intra", "chrom_inter", "V_Q1", "V_Q2", "V_Q3", "V_Q4", "V_top10pct")


def reliability_cells(strat, vmin, mmean):
    cl = st.cells(strat, vmin, mmean)
    return {k: cl[k] for k in REL_CELLS}


def reliability_table(ta, tb, strat, vmin, mmean, keep=None) -> dict:
    """Half-vs-half Spearman (and Pearson), Spearman-Brown corrected, per cell; plus the primary-style pooled statistic."""
    ok = np.isfinite(ta) & np.isfinite(tb)
    if keep is not None:
        ok &= keep
    cells = reliability_cells(strat, vmin, mmean)
    out = {}
    for name, m in cells.items():
        mm = m & ok
        if mm.sum() < 10:
            out[name] = {"n": int(mm.sum()), "r_spearman": float("nan"), "r_pearson": float("nan"),
                         "rho_sb": float("nan"), "ceiling": float("nan")}
            continue
        r = spearman_fast(ta[mm], tb[mm])
        rp = float(np.corrcoef(ta[mm], tb[mm])[0, 1])
        rho = float(spearman_brown(r))
        out[name] = {"n": int(mm.sum()), "r_spearman": r, "r_pearson": rp, "rho_sb": rho, "ceiling": float(ceiling_from_reliability(rho))}
    # primary-style (weighted Spearman, >1Mb + inter equal weight) of half A vs half B
    s = stat_pooled(None, "rel_pooled", ta, tb, strat, np.zeros(len(ta), np.int16), mask=ok, ci=False)
    if s is not None:
        rho = float(spearman_brown(s.value))
        out["pooled_gt1Mb_inter_eqw"] = {"n": s.n, "r_spearman": s.value, "rho_sb": rho, "ceiling": float(ceiling_from_reliability(rho))}
    return out


def summarize_reps(per_rep: list[dict]) -> dict:
    """Median and 2.5/97.5 percentile (over split repetitions) of every number."""
    keys = per_rep[0].keys()
    out = {}
    for k in keys:
        out[k] = {}
        for f in ("r_spearman", "rho_sb", "ceiling", "n"):
            vals = np.array([r[k].get(f, np.nan) for r in per_rep], float)
            vals = vals[np.isfinite(vals)]
            out[k][f] = ({"median": float(np.median(vals)), "q025": float(np.quantile(vals, 0.025)),
                          "q975": float(np.quantile(vals, 0.975))} if len(vals) else None)
    return out
