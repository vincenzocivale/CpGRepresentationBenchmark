import numpy as np

from cpg_repr_benchmark.probing.linear_probe import fit_linear_probe
from cpg_repr_benchmark.probing.metrics import classification_metrics, regression_metrics


def test_fit_linear_probe_regression_recovers_linear_signal():
    rng = np.random.default_rng(17)
    embedding = rng.normal(size=(120, 16))
    w = rng.normal(size=16)
    target = embedding @ w + rng.normal(scale=0.01, size=120)
    train_rows = np.arange(0, 100)
    test_rows = np.arange(100, 120)
    result = fit_linear_probe(
        embedding=embedding, target=target, train_rows=train_rows, test_rows=test_rows, task_type="regression"
    )
    assert result.metrics["test"]["r2"] > 0.95
    pred = result.predict(embedding[test_rows])
    assert regression_metrics(pred, target[test_rows])["r2"] > 0.95


def test_fit_linear_probe_classification_recovers_separable_signal():
    rng = np.random.default_rng(17)
    embedding = rng.normal(size=(200, 8))
    w = rng.normal(size=8)
    logits = embedding @ w
    target = (logits > np.median(logits)).astype(np.float64)
    train_rows = np.arange(0, 160)
    test_rows = np.arange(160, 200)
    result = fit_linear_probe(
        embedding=embedding, target=target, train_rows=train_rows, test_rows=test_rows, task_type="classification"
    )
    assert result.metrics["test"]["auc"] > 0.9
    prob = result.predict(embedding[test_rows])
    assert classification_metrics(prob, target[test_rows])["auc"] > 0.9


def test_linear_probe_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    embedding = rng.normal(size=(50, 4))
    target = embedding[:, 0] * 2.0 + rng.normal(scale=0.01, size=50)
    result = fit_linear_probe(
        embedding=embedding,
        target=target,
        train_rows=np.arange(0, 40),
        test_rows=np.arange(40, 50),
        task_type="regression",
    )
    path = tmp_path / "probe.joblib"
    result.save(path)
    from cpg_repr_benchmark.probing.linear_probe import LinearProbeResult

    loaded = LinearProbeResult.load(path)
    assert np.allclose(loaded.predict(embedding[40:]), result.predict(embedding[40:]))
