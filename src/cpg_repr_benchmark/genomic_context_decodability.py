"""Genomic-context decodability (``genomic_context_decodability_v1``): representation characterization, NOT validation.

Question: how much genomic context (CpG-island class, gene region, TSS distance) can be decoded linearly from a
patient-independent locus embedding that never received those labels as input? Histone/DNase tracks do not contain
island/gene-region/TSS labels but are biologically correlated with them, so this is a characterization of the
representation, not an independent biological validation. The functional-annotation baseline *does* contain these
labels as dense input columns and is therefore reported in Supplementary only.

Protocol (identical for every arm; see docs/GENOMIC_CONTEXT_DECODABILITY.md):
 * same locus universe (408,399 bioval-v2 array loci, zero-vector loci KEPT), same frozen 5-fold chromosome-blocked folds;
 * per outer fold: StandardScaler fitted on training chromosomes only;
 * multiclass endpoints (context, gene region): LogisticRegression(C=1, class_weight="balanced"), no tuning;
   primary = macro one-vs-rest AUROC, secondary = macro-F1 + confusion matrix;
 * TSS distance: y = log10(1 + bp), Ridge with alpha chosen ONLY by inner chromosome-blocked CV;
   primary = pooled out-of-fold Spearman rho, secondary = R^2, MAE;
 * statistics: pooled OOF predictions, chromosome-block bootstrap (B=2000, one shared draw table), paired contrasts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cpg_repr_benchmark.bioval_v2_launch import chrom_probes as cp
from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import (
    RankPlan,
    weighted_auroc,
    weighted_mean,
    weighted_r2,
    weighted_spearman,
)

BV = "data/derived/bioval_v2"
COVARIATES = f"{BV}/covariates/universe_covariates.parquet"
FUNCTIONAL_DENSE = "data/derived/functional_annotations/reference_functional_features_tcga_array_all_v1_cpgidx_fixed.h5"
CONTEXT_CLASSES = ("island", "shore", "shelf", "open_sea")
GENE_CLASSES = ("promoter", "exon", "intron", "intergenic")  # dense cols 4-7 of the GENCODE v50 annotation
GENE_DENSE_COLS = slice(4, 8)
RIDGE_ALPHAS = (1.0, 10.0, 1e2, 1e3, 1e4)
LOGREG_C = 1.0
LOGREG_MAX_ITER = 1000
N_BOOT = 2000
SEED = 17
MAIN_ARMS = ("regulatory_histone_dnase_v1", "cpgpt_large_locus", "deepcpg_dna_locus")
SUPPLEMENTARY_ARMS = ("functional_annotations_pca", "alphagenome_locus", "regulatory_raw2492", "regulatory_lpca256", "regulatory_lpca512", "regulatory_svd512")  # alphagenome: post-hoc, outside Holm family
ENDPOINTS = {  # id -> (primary stat, secondary stats)
    "cpg_context": ("macro_auroc", ("macro_f1",)),
    "gene_region": ("macro_auroc", ("macro_f1",)),
    "tss_distance": ("spearman", ("r2", "mae")),
}


# ----------------------------------------------------------------------------- labels
def load_labels(root: Path) -> pd.DataFrame:
    """cpg_idx-sorted frame: chrom, context (int), gene_region (int), tss (log10(1+bp)). Labels never enter any input."""
    import h5py

    cov = pd.read_parquet(Path(root) / COVARIATES, columns=["cpg_idx", "chrom", "context", "tss_dist"])
    cov = cov.sort_values("cpg_idx").reset_index(drop=True)
    with h5py.File(Path(root) / FUNCTIONAL_DENSE, "r") as h:
        ids = h["cpg_idx"][:]
        dense = h["dense"][:, GENE_DENSE_COLS].astype(np.float32)
    if not (np.isin(cov.cpg_idx.to_numpy(), ids).all() and (dense.sum(1) == 1).all()):
        raise ValueError("gene-region annotation must be one-hot and cover the whole universe")
    order = np.argsort(ids)
    pos = order[np.searchsorted(ids[order], cov.cpg_idx.to_numpy())]
    out = pd.DataFrame({
        "cpg_idx": cov.cpg_idx.to_numpy(),
        "chrom": cov.chrom.astype(str).to_numpy(),
        "context": pd.Categorical(cov.context, categories=CONTEXT_CLASSES).codes.astype(np.int8),
        "gene_region": dense[pos].argmax(1).astype(np.int8),
        "tss": cov.tss_dist.to_numpy(dtype=np.float64),
    })
    if (out.context < 0).any() or not np.isfinite(out.tss).all():
        raise ValueError("unmapped context class or non-finite TSS distance")
    return out


# ----------------------------------------------------------------------------- probes
def multiclass_oof(X, y, fold, n_classes, C=LOGREG_C, max_iter=LOGREG_MAX_ITER):
    """Chromosome-blocked multinomial logistic probe. Returns dict(proba=(n,K) pooled OOF, n_nonconverged_folds)."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    y, fold = np.asarray(y).astype(int), np.asarray(fold)
    proba = np.full((len(y), n_classes), np.nan, dtype=np.float32)
    nonconv = 0
    for f in np.unique(fold):
        te, tr = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
        cp.assert_blocked(np.unique(fold[tr]), [f])
        mu, sd = cp._std_stats(X, tr)  # scaler on training chromosomes only
        m = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=max_iter)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            m.fit(cp._scaled(X, tr, mu, sd), y[tr])
        nonconv += int(any(issubclass(x.category, ConvergenceWarning) for x in w))
        if not np.array_equal(m.classes_, np.arange(n_classes)):
            raise ValueError("a training fold is missing a class")
        proba[te] = m.predict_proba(cp._scaled(X, te, mu, sd))
    return {"proba": proba, "n_nonconverged_folds": nonconv}


# ----------------------------------------------------------------------------- weighted metrics (w = bootstrap multiplicity)
def weighted_confusion(y, pred, w, k):
    return np.bincount(np.asarray(y) * k + np.asarray(pred), weights=w, minlength=k * k).reshape(k, k)


def f1_from_confusion(c):
    tp = np.diag(c)
    denom = 2 * tp + (c.sum(0) - tp) + (c.sum(1) - tp)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmean(np.where(denom > 0, 2 * tp / denom, np.nan)))


class MulticlassScorer:
    """Macro one-vs-rest AUROC and macro-F1 of pooled OOF probabilities, evaluable under any weight vector."""

    def __init__(self, y, proba):
        self.y, self.k = np.asarray(y).astype(int), proba.shape[1]
        self.pred = proba.argmax(1)
        self.plans = [RankPlan(proba[:, j]) for j in range(self.k)]
        self.pos = [self.y == j for j in range(self.k)]

    def macro_auroc(self, w):
        return float(np.nanmean([weighted_auroc(p, m, w) for p, m in zip(self.plans, self.pos, strict=True)]))

    def macro_f1(self, w):
        return f1_from_confusion(weighted_confusion(self.y, self.pred, w, self.k))

    def confusion(self, w=None):
        return weighted_confusion(self.y, self.pred, np.ones(len(self.y)) if w is None else w, self.k)


class RegressionScorer:
    def __init__(self, y, pred):
        self.y, self.pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
        self.py, self.pp = RankPlan(self.y), RankPlan(self.pred)

    def spearman(self, w):
        return weighted_spearman(self.py, self.pp, w)

    def r2(self, w):
        return weighted_r2(self.y, self.pred, w)

    def mae(self, w):
        return weighted_mean(np.abs(self.y - self.pred), w)


def endpoint_scorers(labels: pd.DataFrame, oof: dict) -> dict:
    """{endpoint: (scorer, {stat: fn(w)})} from the pooled OOF predictions of ONE arm."""
    ctx = MulticlassScorer(labels.context.to_numpy(), oof["proba_context"])
    gen = MulticlassScorer(labels.gene_region.to_numpy(), oof["proba_gene"])
    tss = RegressionScorer(labels.tss.to_numpy(), oof["pred_tss"])
    return {"cpg_context": (ctx, {"macro_auroc": ctx.macro_auroc, "macro_f1": ctx.macro_f1}),
            "gene_region": (gen, {"macro_auroc": gen.macro_auroc, "macro_f1": gen.macro_f1}),
            "tss_distance": (tss, {"spearman": tss.spearman, "r2": tss.r2, "mae": tss.mae})}


def fit_arm(X, labels: pd.DataFrame, fold) -> dict:
    """All three endpoints for one embedding matrix X (n, d), aligned to ``labels`` rows. No label touches X."""
    ctx = multiclass_oof(X, labels.context.to_numpy(), fold, len(CONTEXT_CLASSES))
    gen = multiclass_oof(X, labels.gene_region.to_numpy(), fold, len(GENE_CLASSES))
    rid = cp.ridge_blocked_cv(X, labels.tss.to_numpy(), fold, alphas=RIDGE_ALPHAS)
    return {"proba_context": ctx["proba"], "proba_gene": gen["proba"], "pred_tss": rid["oof"][:, 0],
            "ridge_alpha": rid["alpha"][:, 0], "folds": rid["folds"],
            "n_nonconverged_folds": np.array([ctx["n_nonconverged_folds"], gen["n_nonconverged_folds"]])}
