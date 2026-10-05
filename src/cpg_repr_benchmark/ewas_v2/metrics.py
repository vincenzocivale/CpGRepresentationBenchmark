# ruff: noqa: C408
"""EWAS-v2 metrics, paired chromosome-block bootstrap on matched pairs, contrasts, family summaries.

Row layout of every task: rows [0, P) = EWAS positives, rows [P, 2P) = matched backgrounds; pair i = rows (i, P+i); both members
share the chromosome, hence the fold and the bootstrap block. Resampling a chromosome block with multiplicity m keeps every pair
whose EWAS CpG lies in the block with weight m (pairs are never split). The draw table is the shared bioval-v2 BlockBootstrap
(B=1000, seed 17, 22 blocks), identical for every arm, task and metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cpg_repr_benchmark.biological_validation_v2.metrics import auroc_binary, average_precision
from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import (
    BlockBootstrap,
    RankPlan,
    block_index,
    contrast,
    percentile_ci,
    weighted_auroc,
    weighted_mean,
)


def concordance_per_pair(pos, neg) -> np.ndarray:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return (pos > neg).astype(float) + 0.5 * (pos == neg)


def matched_pair_concordance(pos, neg) -> float:
    """Share of pairs with score(EWAS CpG) > score(matched background); ties 0.5."""
    return float(concordance_per_pair(pos, neg).mean())


def point_metrics(pos, neg) -> dict:
    y = np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]
    s = np.r_[np.asarray(pos, float), np.asarray(neg, float)]
    return dict(auroc=auroc_binary(y, s), concordance=matched_pair_concordance(pos, neg), auprc=average_precision(y, s))


def bootstrap_arms(scores: dict, pair_chrom, bb: BlockBootstrap, contrasts: list[tuple[str, str, str]]) -> dict:
    """scores[arm] = (pos, neg) OOF scores per pair; pair_chrom[i] = chromosome of pair i. contrasts = [(name, a, b)], delta = a - b.
    Returns dict(point[arm][metric], ci[arm][metric], contrast[(name, metric)] -> {delta, ci_lo, ci_hi, p, ...})."""
    pair_chrom = np.asarray(pair_chrom).astype(str)
    P = len(pair_chrom)
    blk = block_index(pair_chrom, bb.names)
    blk_rows = np.r_[blk, blk]
    y = np.r_[np.ones(P, bool), np.zeros(P, bool)]
    point, rep = {}, {}
    for a, (pos, neg) in scores.items():
        if len(pos) != P or len(neg) != P:
            raise ValueError("scores must have one entry per pair")
        s = np.r_[np.asarray(pos, float), np.asarray(neg, float)]
        plan = RankPlan(s)
        cpp = concordance_per_pair(pos, neg)
        point[a] = point_metrics(pos, neg)
        rep[a] = dict(
            auroc=bb.replicates(lambda w, plan=plan: weighted_auroc(plan, y, w), blk_rows),
            concordance=bb.replicates(lambda w, c=cpp: weighted_mean(c, w), blk))
    ci = {a: {m: percentile_ci(rep[a][m]) for m in ("auroc", "concordance")} for a in scores}
    out = {}
    for name, a, b in contrasts:
        for m in ("auroc", "concordance"):
            out[(name, m)] = contrast(point[a][m], rep[a][m], point[b][m], rep[b][m])
    return dict(point=point, ci=ci, contrast=out)


def direction(ci_lo: float, ci_hi: float) -> str:
    """'candidate_better' if CI > 0, 'candidate_worse' if CI < 0, else 'spans_zero'."""
    if ci_lo > 0:
        return "candidate_better"
    if ci_hi < 0:
        return "candidate_worse"
    return "spans_zero"


def family_summary(contrasts: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Descriptive counts, NO pooling, NO multiplicity adjustment, NO aggregate score.

    contrasts: columns task, design, contrast, metric, ci_lo, ci_hi. meta: columns task, family, tier, non_independent_of ('' if none).
    Groups: design x contrast x metric x family x tier x scope; scope 'independent' excludes non_independent tasks (each is
    counted once, through its partner, which sits in its own family), 'all' includes them."""
    d = contrasts.merge(meta[["task", "family", "tier", "non_independent_of"]], on="task", how="left")
    d["direction"] = [direction(a, b) for a, b in zip(d["ci_lo"], d["ci_hi"], strict=True)]
    d["indep"] = d["non_independent_of"].fillna("") == ""
    rows = []
    for scope, sub in (("independent", d[d["indep"]]), ("all", d)):
        for key, g in sub.groupby(["design", "contrast", "metric", "family", "tier"], sort=True):
            rows.append(dict(zip(["design", "contrast", "metric", "family", "tier"], key, strict=True), scope=scope,
                             n_tasks=len(g), n_candidate_better=int((g.direction == "candidate_better").sum()),
                             n_candidate_worse=int((g.direction == "candidate_worse").sum()),
                             n_spans_zero=int((g.direction == "spans_zero").sum()),
                             tasks_better=";".join(sorted(g.task[g.direction == "candidate_better"])),
                             tasks_worse=";".join(sorted(g.task[g.direction == "candidate_worse"])),
                             tasks_spanning=";".join(sorted(g.task[g.direction == "spans_zero"]))))
    return pd.DataFrame(rows)
