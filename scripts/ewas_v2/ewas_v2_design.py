"""EWAS-v2 design primitives (DRAFT SCAFFOLD, NOT EXECUTED; see docs/EWAS_V2_PROTOCOL_DRAFT.md).

Pure, embedding-free functions that implement the design rules of the draft protocol so they can be unit-tested on synthetic
data and, once the protocol is frozen, used to build the frozen artifacts (sets, shared backgrounds, folds, tiers, draw table).
Nothing here reads an embedding, a beta value, TCGA data or the Loyfer audit directory.

Implemented: support tier rule, set-overlap (containment) rule, shared technical background matching, matched-pair concordance,
shared chromosome-block draw table and paired block bootstrap for AUROC / concordance.
NOT implemented (spec only): nested chromosome-blocked C selection and the probe fit, covariate-caliper sensitivity matching, I/O.
"""
from __future__ import annotations

import hashlib

import numpy as np

SUPPORT_MAIN, SUPPORT_EXPLORATORY = 200, 100
CONTAINMENT_THRESHOLD = 0.5
DRAW_SEED, N_BOOT = 17, 1000


def support_tier(n_matched_positives: int) -> str:
    """Decision 4 (N only): >=200 main-quality; 100-199 exploratory; <100 insufficient."""
    if n_matched_positives >= SUPPORT_MAIN:
        return "main-quality"
    if n_matched_positives >= SUPPORT_EXPLORATORY:
        return "exploratory"
    return "insufficient"


def containment_flags(sets: dict[str, np.ndarray], threshold: float = CONTAINMENT_THRESHOLD) -> list[dict]:
    """A set is `non_independent` when >= threshold of its members are in a LARGER set (ties: lexicographic).

    Returns rows (small, large, shared, containment, n_set_difference). The set-difference variant is analysed as a
    sensitivity only if its size reaches the exploratory threshold (rule applied by the caller via support_tier).
    """
    sets = {k: np.unique(np.asarray(v)) for k, v in sets.items()}
    rows = []
    for a in sorted(sets):
        for b in sorted(sets):
            if a == b or (len(sets[a]), a) > (len(sets[b]), b):
                continue
            shared = int(np.intersect1d(sets[a], sets[b]).size)
            c = shared / len(sets[a]) if len(sets[a]) else 0.0
            if c >= threshold:
                rows.append(dict(small=a, large=b, shared=shared, containment=c, n_set_difference=len(sets[a]) - shared))
    return rows


def technical_background(positive_idx: np.ndarray, chrom: np.ndarray, probe_type: np.ndarray, excluded: np.ndarray,
                         seed: int, k: int = 1) -> dict:
    """ONE shared background per task: exact strata = chromosome x Infinium type; independent of every embedding.

    positive_idx : row indices of the task's positives (universe rows).
    chrom, probe_type : per-universe-row arrays (probe_type: 1/2, 0 or NaN = unknown).
    excluded : boolean per universe row; True = member of ANY set of the panel (never a background candidate).
    Positives with unknown probe type are dropped (counted). Within a stratum positives are processed in sorted row order, background
    drawn without replacement from sorted eligible rows with default_rng(seed); strata processed in sorted order, so the result is a
    deterministic function of (inputs, seed). Strata with too few backgrounds yield fewer than k per positive (counted as `unmatched`).
    """
    probe_type = np.nan_to_num(np.asarray(probe_type, float), nan=0).astype(int)
    chrom = np.asarray(chrom).astype(str)
    pos = np.sort(np.asarray(positive_idx))
    pos_known = pos[probe_type[pos] > 0]
    is_pos = np.zeros(len(chrom), bool)
    is_pos[pos] = True
    eligible = ~excluded & ~is_pos & (probe_type > 0)
    rng = np.random.default_rng(seed)
    pairs, unmatched = [], 0
    keys = sorted({(chrom[i], probe_type[i]) for i in pos_known})
    for c, t in keys:
        p = pos_known[(chrom[pos_known] == c) & (probe_type[pos_known] == t)]
        pool = np.flatnonzero(eligible & (chrom == c) & (probe_type == t))
        take = min(len(p) * k, len(pool))
        neg = rng.choice(pool, take, replace=False) if take else np.empty(0, int)
        n_pos_matched = take // k
        unmatched += len(p) - n_pos_matched
        for j in range(n_pos_matched):
            pairs.append((int(p[j]), [int(x) for x in neg[j * k:(j + 1) * k]]))
    return dict(pairs=pairs, n_dropped_unknown_probe_type=int(len(pos) - len(pos_known)), n_unmatched=int(unmatched),
                n_matched_positives=len(pairs))


def pairs_digest(pairs) -> str:
    h = hashlib.sha256()
    for p, ns in pairs:
        h.update(f"{p}:{','.join(map(str, ns))};".encode())
    return h.hexdigest()


def matched_pair_concordance(pos, neg) -> float:
    """Share of pairs with score(EWAS CpG) > score(matched background); ties 0.5."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / len(pos))


def draw_table(n_blocks: int, B: int = N_BOOT, seed: int = DRAW_SEED) -> np.ndarray:
    """ONE shared table of chromosome-block resamples (B x n_blocks), reused by every task and arm (paired bootstrap)."""
    return np.random.default_rng(seed).integers(0, n_blocks, size=(B, n_blocks))


def paired_block_bootstrap(blocks: list[str], pair_block: np.ndarray, scores: dict[str, tuple[np.ndarray, np.ndarray]],
                           comparisons: list[tuple[str, str]], draws: np.ndarray) -> dict:
    """Paired chromosome-block bootstrap. blocks = sorted chromosome names (draw indices refer to them); pair_block[i] = chromosome
    of pair i (both members share it); scores[arm] = (pos, neg) OOF scores per pair. Blocks without pairs for this task contribute
    nothing (the draw table is not altered). Returns point estimates, 95% percentile CIs, deltas (first - second) for AUROC and concordance.
    """
    from scipy.stats import rankdata
    idx = {b: np.flatnonzero(pair_block == b) for b in blocks}

    def auc(p, n):
        r = rankdata(np.concatenate([p, n]))
        return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))

    point = {a: dict(auroc=auc(*s), concordance=matched_pair_concordance(*s)) for a, s in scores.items()}
    boot = {a: dict(auroc=np.empty(len(draws)), concordance=np.empty(len(draws))) for a in scores}
    for b, d in enumerate(draws):
        sel = np.concatenate([idx[blocks[j]] for j in d])
        for a, (p, n) in scores.items():
            boot[a]["auroc"][b] = auc(p[sel], n[sel])
            boot[a]["concordance"][b] = matched_pair_concordance(p[sel], n[sel])
    out = dict(point=point, ci={}, delta={})
    for a in scores:
        out["ci"][a] = {m: tuple(np.percentile(boot[a][m], [2.5, 97.5])) for m in ("auroc", "concordance")}
    for a, c in comparisons:
        for m in ("auroc", "concordance"):
            dd = boot[a][m] - boot[c][m]
            out["delta"][(a, c, m)] = dict(estimate=point[a][m] - point[c][m], ci=tuple(np.percentile(dd, [2.5, 97.5])),
                                           p_raw=min(1.0, 2 * (min((dd <= 0).mean(), (dd >= 0).mean()) + 1 / (len(dd) + 1))))
    return out
