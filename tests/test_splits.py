from pathlib import Path

import numpy as np

from cpg_repr_benchmark.data.splits import locus_disjoint_split, load_or_create_locus_protocol


def test_locus_split_is_canonical_id_invariant():
    cpg_a = np.array([40, 10, 30, 20, 50, 60], dtype=np.int64)
    cols = np.arange(len(cpg_a))
    a = locus_disjoint_split(cols, cpg_a, seed=17, heldout_fraction=1 / 3)
    train_ids_a = set(cpg_a[a.train_columns])
    heldout_ids_a = set(cpg_a[a.heldout_columns])

    permutation = np.array([2, 5, 0, 4, 1, 3])
    cpg_b = cpg_a[permutation]
    b = locus_disjoint_split(np.arange(len(cpg_b)), cpg_b, seed=17, heldout_fraction=1 / 3)
    assert set(cpg_b[b.train_columns]) == train_ids_a
    assert set(cpg_b[b.heldout_columns]) == heldout_ids_a


def test_shared_protocol_reuses_same_cpg_ids(tmp_path: Path):
    path = tmp_path / "protocol.npz"
    ids = np.array([7, 2, 9, 1, 4, 8], dtype=np.int64)
    first = load_or_create_locus_protocol(path, np.arange(6), ids, seed=3, heldout_fraction=0.33)
    first_train = set(ids[first.train_columns])
    first_test = set(ids[first.heldout_columns])

    order = np.array([3, 1, 5, 0, 4, 2])
    reordered = ids[order]
    second = load_or_create_locus_protocol(path, np.arange(6), reordered, seed=3, heldout_fraction=0.33)
    assert set(reordered[second.train_columns]) == first_train
    assert set(reordered[second.heldout_columns]) == first_test
    assert first_train.isdisjoint(first_test)
