"""Synthetic checks of the R14 blend fit, the entry-by-entry pairing check and the crossed bootstrap cells (scripts/hal/knn_blend.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/hal"))
import knn_blend as kb  # noqa: E402


def _h(rng, P=6, E=8):
    t = rng.random((P, E)).astype(np.float32)
    return {"sample_index": np.arange(P) * 3, "target_matrix_column": rng.integers(0, 50, (P, E)), "target": t,
            "prior_prediction": rng.random((P, E)).astype(np.float32), "prediction": rng.random((P, E)).astype(np.float32)}


def test_fit_blend_recovers_weights():
    rng = np.random.default_rng(0)
    arm, knn, pri = rng.random((3, 20, 30))
    y = pri + 0.3 * (arm - pri) + 0.5 * (knn - pri)
    wa, wk, wp = kb.fit_blend(arm, knn, pri, y)
    assert np.allclose([wa, wk, wp], [0.3, 0.5, 0.2])


def test_aligned_reorders_and_rejects_unpaired():
    rng = np.random.default_rng(1)
    ref = _h(rng)
    perm = rng.permutation(6)
    shuf = {k: v[perm] for k, v in ref.items()}
    assert np.array_equal(kb.aligned(shuf, ref, "x"), ref["prediction"])
    bad = {**shuf, "target_matrix_column": shuf["target_matrix_column"].copy()}
    bad["target_matrix_column"][0, 0] += 1
    with pytest.raises(AssertionError):
        kb.aligned(bad, ref, "x")


def test_cells_point_and_resampling():
    rng = np.random.default_rng(2)
    P, E = 12, 10
    block = rng.integers(0, 4, (P, E)).astype(str)
    c = kb.Cells(np.array([f"p{i}" for i in range(P)]), block, reps=300, seed=17)
    se = rng.random((P, E))
    point, boot = c.mse(se)
    assert np.isclose(point, se.mean()) and boot.shape == (300,) and boot.min() < point < boot.max()
    assert np.allclose(c.mse(np.full((P, E), 0.2))[1], 0.2)  # constant error: every replicate equals it
