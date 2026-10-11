from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def patient_id(sample_name: str) -> str:
    parts = sample_name.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 and parts[0].upper() == "TCGA" else sample_name


def patient_disjoint_split(
    sample_names: list[str],
    seed: int,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, np.ndarray]:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("patient split fractions must sum to one")
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(sample_names):
        groups.setdefault(patient_id(name), []).append(i)
    patients = np.asarray(sorted(groups), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    a = int(len(patients) * fractions[0])
    b = int(len(patients) * (fractions[0] + fractions[1]))
    result = {}
    for name, selected in zip(("train", "validation", "test"), (patients[:a], patients[a:b], patients[b:])):
        result[name] = np.asarray([i for patient in selected for i in groups[str(patient)]], dtype=np.int64)
    return result


@dataclass(frozen=True)
class LocusSplit:
    train_columns: np.ndarray
    heldout_columns: np.ndarray


def locus_disjoint_split(
    candidate_columns: np.ndarray,
    cpg_ids: np.ndarray,
    *,
    seed: int,
    heldout_fraction: float,
) -> LocusSplit:
    """Split candidate loci into train/heldout for the unseen_locus OOD view.

    `heldout_fraction == 0.0` is a valid opt-out: it means "no locus holdout" and
    returns every candidate locus as `train_columns` with an empty `heldout_columns`.
    Use this path (via `load_or_create_locus_protocol`) for the genome-wide
    seen-only masking protocol; any `heldout_fraction` in (0, 1) still produces a
    genuine disjoint OOD split for future unseen_locus work.
    """
    if not 0.0 <= heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be between 0 (inclusive) and 1 (exclusive)")
    candidate_columns = np.asarray(candidate_columns, dtype=np.int64)
    if heldout_fraction == 0.0:
        if len(candidate_columns) < 1:
            raise ValueError("need at least one candidate CpG for a locus split")
        canonical_order = candidate_columns[np.argsort(cpg_ids[candidate_columns], kind="stable")]
        return LocusSplit(
            train_columns=np.sort(canonical_order),
            heldout_columns=np.asarray([], dtype=np.int64),
        )
    if len(candidate_columns) < 2:
        raise ValueError("need at least two candidate CpGs for a locus split")
    # Sort by canonical ID first so the split is invariant to matrix column ordering.
    canonical_order = candidate_columns[np.argsort(cpg_ids[candidate_columns], kind="stable")]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(canonical_order))
    n_heldout = max(1, int(round(len(canonical_order) * heldout_fraction)))
    heldout = np.sort(canonical_order[perm[:n_heldout]])
    train = np.sort(canonical_order[perm[n_heldout:]])
    if np.intersect1d(train, heldout).size:
        raise AssertionError("locus split overlap")
    return LocusSplit(train_columns=train, heldout_columns=heldout)


def _norm_chrom(value) -> str:
    from cpg_repr_benchmark.data.chromosomes import normalize_chromosome

    return normalize_chromosome(value)


def chromosome_locus_split(
    candidate_columns: np.ndarray,
    cpg_ids: np.ndarray,
    chromosome_labels,
    heldout_chromosomes,
) -> LocusSplit:
    """Strict chromosome holdout: every candidate CpG on `heldout_chromosomes` is held out, all others train.

    `chromosome_labels[j]` is the chromosome of matrix column j (any label style; normalized here).
    """
    candidate_columns = np.asarray(candidate_columns, dtype=np.int64)
    held = {_norm_chrom(c) for c in heldout_chromosomes}
    if not held:
        raise ValueError("heldout_chromosomes is empty")
    labels = np.asarray([_norm_chrom(x) for x in np.asarray(chromosome_labels)[candidate_columns]])
    is_held = np.isin(labels, sorted(held))
    heldout, train = candidate_columns[is_held], candidate_columns[~is_held]
    if len(heldout) < 1 or len(train) < 2:
        raise ValueError(f"chromosome holdout {sorted(held)} leaves {len(train)} train / {len(heldout)} heldout loci")
    return LocusSplit(train_columns=np.sort(train), heldout_columns=np.sort(heldout))


def assert_no_heldout_chromosomes(columns_by_name: dict, chromosome_labels, heldout_chromosomes) -> None:
    """Raise if any column of any named array (train targets/contexts, prior-fit loci, ...) lies on a held-out chromosome."""
    held = {_norm_chrom(c) for c in heldout_chromosomes}
    labels = np.asarray(chromosome_labels)
    for name, cols in columns_by_name.items():
        got = {_norm_chrom(x) for x in np.unique(labels[np.asarray(cols, dtype=np.int64)])}
        if got & held:
            raise AssertionError(f"LEAK: {name} contains held-out chromosomes {sorted(got & held)}")


def load_or_create_locus_protocol(
    path,
    candidate_columns: np.ndarray,
    cpg_ids: np.ndarray,
    *,
    seed: int,
    heldout_fraction: float,
    heldout_chromosomes=None,
    chromosome_labels=None,
) -> LocusSplit:
    """Create once, then reuse the identical canonical CpG split across representations.

    The protocol stores CpG IDs rather than matrix positions. On reload they are mapped back
    to the current matrix order and the candidate universe must match exactly. This prevents
    a representation's coverage from silently changing the benchmark locus population.

    Opt-in strict chromosome holdout: if `heldout_chromosomes` is given (with `chromosome_labels`, a
    normalized chrN label per matrix column), `heldout_fraction` must be 0 and the split is
    chromosome-disjoint (see `chromosome_locus_split`). The protocol additionally stores the chromosome
    list and refuses a mismatching reuse. Without it, behaviour and file format are unchanged.
    """
    from pathlib import Path

    path = Path(path)
    candidate_columns = np.asarray(candidate_columns, dtype=np.int64)
    cpg_ids = np.asarray(cpg_ids, dtype=np.int64)
    candidate_ids = np.sort(cpg_ids[candidate_columns])
    held_chroms = None
    if heldout_chromosomes:
        if chromosome_labels is None:
            raise ValueError("heldout_chromosomes requires chromosome_labels")
        if float(heldout_fraction) != 0.0:
            raise ValueError("heldout_chromosomes is exclusive with a nonzero heldout_fraction")
        held_chroms = sorted({_norm_chrom(c) for c in heldout_chromosomes})

    if path.exists():
        with np.load(path) as saved:
            train_ids = np.asarray(saved["train_cpg_idx"], dtype=np.int64)
            heldout_ids = np.asarray(saved["heldout_cpg_idx"], dtype=np.int64)
            saved_seed = int(saved["seed"])
            saved_fraction = float(saved["heldout_fraction"])
            saved_chroms = [str(x) for x in saved["heldout_chromosomes"]] if "heldout_chromosomes" in saved.files else None
        saved_universe = np.sort(np.concatenate([train_ids, heldout_ids]))
        if not np.array_equal(saved_universe, candidate_ids):
            raise ValueError(
                f"locus protocol {path} was built for a different CpG universe "
                f"({len(saved_universe)} vs {len(candidate_ids)} loci)"
            )
        if saved_chroms != held_chroms:
            raise ValueError(f"locus protocol {path} heldout_chromosomes {saved_chroms} != config {held_chroms}")
        if saved_seed != int(seed) or not np.isclose(saved_fraction, heldout_fraction):
            raise ValueError(
                f"locus protocol {path} seed/fraction ({saved_seed}, {saved_fraction}) "
                f"does not match config ({seed}, {heldout_fraction})"
            )
        order = np.argsort(cpg_ids)
        sorted_ids = cpg_ids[order]

        def map_ids(ids: np.ndarray) -> np.ndarray:
            loc = np.searchsorted(sorted_ids, ids)
            if (loc >= len(sorted_ids)).any() or not np.array_equal(sorted_ids[loc], ids):
                raise ValueError(f"locus protocol {path} contains CpGs absent from methylation matrix")
            return np.sort(order[loc])

        return LocusSplit(map_ids(train_ids), map_ids(heldout_ids))

    if held_chroms:
        split = chromosome_locus_split(candidate_columns, cpg_ids, chromosome_labels, held_chroms)
    else:
        split = locus_disjoint_split(
            candidate_columns,
            cpg_ids,
            seed=seed,
            heldout_fraction=heldout_fraction,
        )
    extra = {"heldout_chromosomes": np.asarray(held_chroms)} if held_chroms else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        train_cpg_idx=np.sort(cpg_ids[split.train_columns]),
        heldout_cpg_idx=np.sort(cpg_ids[split.heldout_columns]),
        seed=np.asarray(seed, dtype=np.int64),
        heldout_fraction=np.asarray(heldout_fraction, dtype=np.float64),
        **extra,
    )
    temporary.replace(path)
    return split
