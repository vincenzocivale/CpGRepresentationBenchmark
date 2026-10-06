import numpy as np
import pytest

torch = pytest.importorskip("torch")
from cpg_repr_benchmark.encode_atlas import logistic_pca as lp


def test_monotone_deviance_and_consistent_encoding():
    rng = np.random.default_rng(0)
    n, d, k = 3000, 80, 4
    theta = rng.normal(size=(n, k)) @ rng.normal(size=(k, d)) * 0.8 - 1.0
    x = (rng.random((n, d)) < 1 / (1 + np.exp(-theta))).astype(np.float32)
    m = lp.fit(x, k, max_iter=80, device="cpu", log=lambda *a: None)
    assert np.all(np.diff(m["deviance"]) <= 0), "accepted deviance trace must be monotone non-increasing"
    assert np.allclose(m["V"].T @ m["V"], np.eye(k), atol=1e-4)
    a = lp.encode(x, m["mu"], m["V"], steps=200, device="cpu")
    null = -(x * np.log(x.mean(0) + 1e-9) + (1 - x) * np.log(1 - x.mean(0) + 1e-9)).sum()
    assert m["deviance"][-1] < 0.9 * null  # the low-rank model explains structure beyond the marginals
    assert a.shape == (n, k) and np.isfinite(a).all()
