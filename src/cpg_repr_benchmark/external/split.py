"""Deterministic stratified individual-level split (torch-free)."""
from __future__ import annotations

import numpy as np


def age_tercile_thresholds(ages: np.ndarray) -> tuple[float, float]:
    """Cut points at the 1/3 and 2/3 quantiles (numpy linear interpolation) of the FULL cohort.

    Tercile = (age > q1) + (age > q2): tied ages are never separated, so tercile sizes are only approximately equal.
    """
    q1, q2 = np.quantile(np.asarray(ages, dtype=float), [1 / 3, 2 / 3])
    return float(q1), float(q2)


def age_terciles(ages: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    a = np.asarray(ages, dtype=float)
    return ((a > thresholds[0]).astype(int) + (a > thresholds[1]).astype(int)).astype(np.int64)


def _largest_remainder(sizes: dict, target: int, fraction: float) -> dict:
    """Per-stratum quotas summing exactly to ``target``: floor(fraction*n_s), then +1 by largest remainder (ties: key order)."""
    keys = sorted(sizes)
    exact = {k: fraction * sizes[k] for k in keys}
    quota = {k: int(np.floor(exact[k])) for k in keys}
    missing = target - sum(quota.values())
    if missing < 0 or missing > len(keys):
        raise ValueError("inconsistent quota target")
    order = sorted(keys, key=lambda k: (-(exact[k] - quota[k]), k))
    for k in order[:missing]:
        quota[k] += 1
    return quota


def stratified_split(
    subject_ids: list[str],
    strata: list[str],
    seed: int,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> np.ndarray:
    """Return an array of 'train'/'validation'/'test' labels aligned to ``subject_ids`` (one row per individual).

    n_test = n_validation = floor(f*N + 0.5); train = remainder. Quotas per stratum by largest remainder (test first,
    validation second, both from the full stratum size); inside a stratum subjects are sorted by id, permuted with
    ``default_rng([seed, stratum_rank])`` and assigned test, validation, train in that order. Deterministic and
    independent of input row order.
    """
    if len(subject_ids) != len(strata):
        raise ValueError("subject_ids and strata must have the same length")
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("subject ids must be unique (split unit = individual)")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to one")
    n = len(subject_ids)
    n_val = int(np.floor(fractions[1] * n + 0.5))
    n_test = int(np.floor(fractions[2] * n + 0.5))
    members: dict[str, list[int]] = {}
    for i, s in enumerate(strata):
        members.setdefault(str(s), []).append(i)
    sizes = {k: len(v) for k, v in members.items()}
    q_test = _largest_remainder(sizes, n_test, fractions[2])
    q_val = _largest_remainder(sizes, n_val, fractions[1])
    labels = np.full(n, "train", dtype=object)
    for rank, key in enumerate(sorted(members)):
        if q_test[key] + q_val[key] >= sizes[key]:
            raise ValueError(f"stratum {key!r} (n={sizes[key]}) cannot keep a training subject")
        idx = sorted(members[key], key=lambda i: str(subject_ids[i]))
        perm = np.random.default_rng([int(seed), rank]).permutation(len(idx))
        ordered = [idx[j] for j in perm]
        for i in ordered[: q_test[key]]:
            labels[i] = "test"
        for i in ordered[q_test[key] : q_test[key] + q_val[key]]:
            labels[i] = "validation"
    return labels.astype(str)
