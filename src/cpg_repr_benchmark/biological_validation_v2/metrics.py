"""Metric DEFINITIONS only. Every function takes plain arrays supplied by the caller (similarity or score
vectors, labels, neighbor indices). Nothing here loads or touches embeddings."""

from __future__ import annotations

import itertools

import numpy as np
from scipy import stats

# To be preregistered. axis -> primary metric, unit of analysis, direction (higher/lower is better).
PRIMARY_METRICS = {
    "external_methylation_programs": {
        "metric": "spearman(methylation-profile similarity [cross-cell-type Pearson of CpG beta profiles], "
                  "embedding cosine similarity) on sampled pairs; secondary: marker kNN enrichment",
        "unit": "CpG pair (CI by chromosome block)", "direction": "higher"},
    "3d_genome": {
        "metric": "AUROC contact vs matched non-contact (distance-matched intra; inter reported separately); "
                  "secondary: A/B eigenvector Spearman (chromosome-blocked CV linear probe)",
        "unit": "CpG pair / CpG locus (CI by chromosome block)", "direction": "higher"},
    "regulatory_activity": {
        "metric": "AUROC FANTOM5 enhancer membership vs matched controls; secondary: Spearman of "
                  "tissue-activity-profile similarity vs embedding similarity",
        "unit": "CpG locus (CI by chromosome block)", "direction": "higher"},
    "replication_domains": {
        "metric": "Spearman / R2 of continuous replication timing (chromosome-blocked ridge probe)",
        "unit": "CpG locus (CI by chromosome block)", "direction": "higher"},
}


def spearman_profile_vs_embedding_sim(profile_sim, emb_sim):
    r = stats.spearmanr(profile_sim, emb_sim)
    return float(r.statistic)


def pearson_profile_vs_embedding_sim(profile_sim, emb_sim):
    return float(stats.pearsonr(profile_sim, emb_sim).statistic)


def knn_enrichment(labels, neighbor_index, k, background, *, chroms=None, n_perm=0, seed=0):
    """Marker kNN enrichment.

    labels: array (n,) of cell-type label per locus, '' / None / -1 for non-marker. neighbor_index: (n, >=k)
    int array of neighbors (caller-computed). background: expected fraction of same-label neighbors, either a
    scalar or per-label dict (e.g. label frequency among candidate neighbors). Enrichment =
    observed / expected over marker loci. If n_perm>0, labels are permuted within chromosome (strata =
    ``chroms``) and the one-sided p-value is returned.
    """
    labels = np.asarray(labels, dtype=object)
    nn = np.asarray(neighbor_index)[:, :k]
    is_marker = np.array([(x is not None) and (x != "") and (x != -1) for x in labels])

    def stat(lab):
        obs, exp = [], []
        for i in np.where(is_marker)[0]:
            same = np.mean([lab[j] == lab[i] for j in nn[i]])
            obs.append(same)
            exp.append(background[lab[i]] if isinstance(background, dict) else background)
        return float(np.mean(obs) / np.mean(exp)) if obs else float("nan")

    observed = stat(labels)
    res = {"enrichment": observed, "k": k}
    if n_perm:
        rng = np.random.Generator(np.random.PCG64(seed))
        strata = np.asarray(chroms) if chroms is not None else np.zeros(len(labels))
        null = []
        for _ in range(n_perm):
            lab = labels.copy()
            for s in np.unique(strata):
                m = np.where(strata == s)[0]
                lab[m] = labels[m][rng.permutation(len(m))]
            null.append(stat(lab))
        null = np.asarray(null)
        res["null_mean"] = float(np.mean(null))
        res["p_value"] = float((1 + np.sum(null >= observed)) / (1 + n_perm))
    return res


def auroc_binary(y, score):
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, score):
    y = np.asarray(y).astype(bool)
    o = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    yy = y[o]
    if yy.sum() == 0:
        return float("nan")
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float(prec[yy].mean())


def compartment_eigenvector_corr(eigenvector, prediction):
    """Spearman between A/B eigenvector and (cross-validated, chromosome-blocked) probe prediction."""
    m = np.isfinite(eigenvector) & np.isfinite(prediction)
    return float(stats.spearmanr(np.asarray(eigenvector)[m], np.asarray(prediction)[m]).statistic)


def contact_vs_noncontact_auroc(contact_scores, noncontact_scores):
    s = np.concatenate([contact_scores, noncontact_scores])
    y = np.r_[np.ones(len(contact_scores)), np.zeros(len(noncontact_scores))]
    return auroc_binary(y, s)


def stratified_by_distance(y, score, dist, bins=(0, 1e3, 1e4, 1e5, 1e6, 1e7, np.inf), metric=auroc_binary):
    """metric(y, score) within each distance bin; returns {bin_label: value} (NaN if degenerate)."""
    dist = np.asarray(dist, dtype=float)
    out = {}
    for lo, hi in itertools.pairwise(bins):
        m = (dist >= lo) & (dist < hi)
        out[f"[{lo:g},{hi:g})"] = metric(np.asarray(y)[m], np.asarray(score)[m]) if m.any() else float("nan")
    return out


def bootstrap_ci_by_block(values, blocks, stat=np.mean, n_boot=1000, alpha=0.05, seed=0):
    """Block bootstrap CI: resample blocks (chromosomes) with replacement, recompute ``stat`` on the
    concatenated rows. Returns (point, lo, hi)."""
    values = np.asarray(values)
    blocks = np.asarray(blocks)
    ub = np.unique(blocks)
    groups = [values[blocks == b] for b in ub]
    rng = np.random.Generator(np.random.PCG64(seed))
    est = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        est.append(stat(np.concatenate([groups[p] for p in pick])))
    lo, hi = np.quantile(est, [alpha / 2, 1 - alpha / 2])
    return float(stat(values)), float(lo), float(hi)
