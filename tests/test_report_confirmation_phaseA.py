import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("_rep", ROOT / "scripts/report_confirmation_phaseA.py")
rep = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rep)


def test_adapter_params_scale_with_dim():
    assert rep.adapter_params(256) == 2 * 256 + 256 * 256 + 256 + 512
    assert rep.adapter_params(128) < rep.adapter_params(256) < rep.adapter_params(512)


def test_curve_stats_linear_decline_and_best_in_last10():
    mse = 0.02 - 1e-5 * np.arange(120)
    c = rep.curve_stats(mse)
    assert c["best_epoch"] == 119 and c["best_in_last10"]
    assert c["slope_last10_pct_per_epoch"] < 0
    assert abs(c["noise_sd_detrended_last20"]) < 1e-12
    assert c["mse_ep60"] == mse[60] and c["final_mse"] == mse[-1]


def test_curve_stats_plateau_not_budget_limited():
    mse = np.concatenate([np.linspace(0.03, 0.015, 60), np.full(60, 0.015)])
    mse[70] = 0.0149
    c = rep.curve_stats(mse)
    assert c["best_epoch"] == 70 and not c["best_in_last10"]
    assert abs(c["slope_last20_pct_per_epoch"]) < 1e-9


def test_seed_summary_and_sign_counts():
    df = pd.DataFrame({"arm": ["a"] * 3 + ["b"] * 3, "mse": [1.0, 2.0, 3.0, 4.0, 4.0, 4.0]})
    s = rep.seed_summary(df, ["mse"]).set_index("arm")
    assert s.loc["a", "mse_mean"] == 2.0 and abs(s.loc["a", "mse_sd"] - 1.0) < 1e-12 and s.loc["b", "mse_sd"] == 0.0
    c = pd.DataFrame({"comparator": ["x"] * 3, "metric": ["mse"] * 3, "delta": [1.0, 1.0, -1.0], "ci_lo": [0.5, -0.1, -2.0], "ci_hi": [1.5, 2.0, -0.5]})
    r = rep.sign_counts(c).iloc[0]
    assert (r.reference_better, r.ci_excludes0_reference_better, r.ci_excludes0_comparator_better) == (2, 1, 1)


def test_md_table_format():
    t = rep.md_table(pd.DataFrame({"a": ["x"], "v": [0.123456789]}), {"v": "{:.3f}"})
    assert "| x | 0.123 |" in t
