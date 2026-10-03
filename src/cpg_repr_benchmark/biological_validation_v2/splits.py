"""Chromosome-blocked splits (no chromosome ever appears in two folds)."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def chromosome_blocked_folds(chroms, n_folds, seed=0):
    """Dict chrom -> fold id. Deterministic; independent of input order/duplicates. Chromosomes are
    ordered by sha256(chrom:seed) then dealt round-robin."""
    uniq = sorted(set(map(str, chroms)))
    if n_folds < 2 or n_folds > len(uniq):
        raise ValueError("n_folds must be in [2, n_chroms]")
    order = sorted(uniq, key=lambda c: hashlib.sha256(f"{c}:{seed}".encode()).hexdigest())
    return {c: i % n_folds for i, c in enumerate(order)}


def leave_chromosome_out(chroms):
    """Yield (held_out_chrom, train_chroms) for each chromosome, sorted."""
    uniq = sorted(set(map(str, chroms)))
    for c in uniq:
        yield c, [x for x in uniq if x != c]


def pair_split_by_chrom(pair_df, folds, policy="both_in_test"):
    """Assign pairs (cols chrom_i, chrom_j) to a test fold.

    Intra pairs: fold of the chromosome. Inter pairs: both chromosomes in the same fold -> that fold;
    straddling pairs (chromosomes in different folds) are handled per policy:
      'both_in_test'  : a straddling pair is TEST for fold f if either chromosome is in f (train for fold f
                        therefore never contains a pair touching a test chromosome);
      'drop'          : straddling pairs are dropped (fold = -1).
    Returns Series ``fold`` (test fold id; -1 = dropped). With 'both_in_test' a straddling pair gets
    fold = min(fold_i, fold_j) as primary and appears in the other fold's test via ``pair_test_mask``.
    Inter pairs are keyed by the unordered chromosome pair, so (a,b) and (b,a) always agree.
    """
    if policy not in ("both_in_test", "drop"):
        raise ValueError(policy)
    fi = pair_df["chrom_i"].map(folds).to_numpy()
    fj = pair_df["chrom_j"].map(folds).to_numpy()
    straddle = fi != fj
    fold = np.minimum(fi, fj)
    if policy == "drop":
        fold = np.where(straddle, -1, fold)
    return pd.Series(fold, index=pair_df.index, name="fold")


def pair_test_mask(pair_df, folds, fold, policy="both_in_test"):
    fi = pair_df["chrom_i"].map(folds).to_numpy()
    fj = pair_df["chrom_j"].map(folds).to_numpy()
    if policy == "both_in_test":
        return (fi == fold) | (fj == fold)
    return (fi == fold) & (fj == fold)


def pair_train_mask(pair_df, folds, fold):
    """Train pairs: neither chromosome in the test fold."""
    fi = pair_df["chrom_i"].map(folds).to_numpy()
    fj = pair_df["chrom_j"].map(folds).to_numpy()
    return (fi != fold) & (fj != fold)


def assert_no_leak(train_chroms, test_chroms, pair_df_train=None, pair_df_test=None):
    """Raise if any chromosome is shared between train and test (rows, or either end of pairs)."""
    tr = set(map(str, train_chroms))
    if pair_df_train is not None:
        tr |= set(pair_df_train["chrom_i"]) | set(pair_df_train["chrom_j"])
    te = set(map(str, test_chroms))
    if pair_df_test is not None:
        te |= set(pair_df_test["chrom_i"]) | set(pair_df_test["chrom_j"])
    shared = tr & te
    if shared:
        raise AssertionError(f"chromosome leakage between train and test: {sorted(shared)}")
    return True
