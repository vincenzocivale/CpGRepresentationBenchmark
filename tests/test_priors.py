from pathlib import Path

import h5py
import numpy as np

from cpg_repr_benchmark.training.priors import beta_to_logit, compute_leakage_safe_priors


def _write(path: Path, beta: np.ndarray):
    with h5py.File(path, "w") as h:
        h.create_dataset("beta", data=beta.astype(np.float32))


def test_heldout_locus_values_cannot_change_unseen_prior(tmp_path: Path):
    beta = np.array(
        [
            [0.10, 0.91, 0.30, 0.87, 0.50, 0.77],
            [0.20, 0.82, 0.40, 0.73, 0.60, 0.66],
            [0.70, 0.11, 0.80, 0.22, 0.90, 0.33],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "methyl.h5"
    _write(path, beta)
    train_rows = np.array([0, 1])
    train_cols = np.array([0, 2, 4])
    prior_a, usable_a, global_a = compute_leakage_safe_priors(path, train_rows, train_cols, 6)

    changed = beta.copy()
    changed[np.ix_(train_rows, np.array([1, 3, 5]))] = np.array([[0.001, 0.999, 0.001], [0.999, 0.001, 0.999]])
    _write(path, changed)
    prior_b, usable_b, global_b = compute_leakage_safe_priors(path, train_rows, train_cols, 6)

    np.testing.assert_allclose(prior_a, prior_b)
    np.testing.assert_array_equal(usable_a, usable_b)
    assert np.isclose(global_a, global_b)
    expected_global = beta[np.ix_(train_rows, train_cols)].mean()
    assert np.isclose(global_a, expected_global)
    heldout = np.array([1, 3, 5])
    np.testing.assert_allclose(prior_a[heldout], beta_to_logit(np.full(3, expected_global), 1e-4))


def test_train_locus_without_values_is_not_usable(tmp_path: Path):
    beta = np.array([[0.2, np.nan, 0.4], [0.3, np.nan, 0.5]], dtype=np.float32)
    path = tmp_path / "methyl.h5"
    _write(path, beta)
    prior, usable, _ = compute_leakage_safe_priors(path, np.array([0, 1]), np.array([0, 1]), 3)
    assert usable.tolist() == [True, False]
    assert np.isfinite(prior).all()
