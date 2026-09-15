from pathlib import Path

import h5py
import numpy as np

from cpg_repr_benchmark.data.masking import MaskingDataset, counts_for_mask_fraction


class ToyStore:
    dim = 3

    def __init__(self, n):
        self.coverage_mask = np.ones(n, dtype=bool)

    def get_by_matrix_columns(self, columns):
        c = np.asarray(columns, dtype=np.float32)
        return np.stack([c, c + 1, c + 2], axis=1)

    def close(self):
        pass


def _matrix(path: Path):
    beta = np.linspace(0.05, 0.95, 40, dtype=np.float32).reshape(2, 20)
    with h5py.File(path, "w") as h:
        h.create_dataset("beta", data=beta)


def test_counts_for_fraction():
    assert counts_for_mask_fraction(10, 0.3) == (7, 3)
    assert counts_for_mask_fraction(10, 0.95) == (1, 9)


def test_unseen_targets_are_strictly_heldout(tmp_path: Path):
    path = tmp_path / "methyl.h5"
    _matrix(path)
    train = np.arange(0, 12, dtype=np.int64)
    heldout = np.arange(12, 20, dtype=np.int64)
    ds = MaskingDataset(
        methylation_h5=path,
        representation_store=ToyStore(20),
        prior_logit_full=np.zeros(20, dtype=np.float32),
        sample_indices=np.array([0]),
        context_columns=train,
        target_columns=heldout,
        panel_size=8,
        mask_fractions=[0.5],
        seed=17,
    )
    item = ds[(0, 0.5)]
    targets = item["target_matrix_column"].numpy()
    assert set(targets).issubset(set(heldout))
    assert not set(targets).intersection(set(train))
    assert item["observed_locus"].shape == (4, 3)
    assert item["target_locus"].shape == (4, 3)
