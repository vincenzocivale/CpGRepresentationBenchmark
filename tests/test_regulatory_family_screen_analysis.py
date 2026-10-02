import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from cpg_repr_benchmark.encode_atlas.statistics import paired_bootstrap

_spec = importlib.util.spec_from_file_location(
    "analyze_screen", Path(__file__).resolve().parents[1] / "scripts" / "analyze_regulatory_family_screen.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _table(rng, shift):
    n = 600
    patient = np.repeat([f"p{i}" for i in range(20)], 30)
    column = np.tile(np.arange(30), 20)
    base = np.random.default_rng(0).random(n)
    return pd.DataFrame({"patient": patient, "column": column, "block": [f"c:{c // 5}" for c in column],
                         "target": base, "squared_error": rng.random(n) * 0.1 + shift, "n": 1})


def test_replicate_estimates_match_paired_bootstrap_ci():
    ref, alt = _table(np.random.default_rng(1), 0.0), _table(np.random.default_rng(2), -0.02)
    r = paired_bootstrap(ref, alt, seed=3, replicates=200)
    est = mod.replicate_estimates(ref, alt, 3, 200)
    assert np.allclose(np.quantile(est, [.025, .975]), r["ci95"])
    assert r["delta_mse"] < 0 and np.mean(est < 0) > .9


def test_abs_error_view_gives_absolute_error():
    o = {"prediction": np.array([[.2, .9]], np.float32), "target": np.array([[.5, .5]], np.float32)}
    v = mod.abs_error_view(o)
    assert np.allclose((v["prediction"] - v["target"]) ** 2, np.abs(o["prediction"] - o["target"]), atol=1e-6)
