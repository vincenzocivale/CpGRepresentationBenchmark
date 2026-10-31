#!/usr/bin/env python3
"""Summary of the patient-kNN baseline on the TEST patients (R9; CPU only).

Input: the per-entry predictions written by `scripts/run_knn_imputation_baseline.py --split test --save-predictions ...`
(k=10, mask fraction 0.50, mask seed 17001, panels of 2,048, same MaskingDataset draw as every arm).
Output summary.json: MSE, MAE, skill (1 - MSE / MSE of the empirical train-patient prior), n_samples, n_entries and 95% percentile
intervals from a bootstrap that resamples TEST SAMPLES only (1,000 resamples, no genomic blocks: the descriptive scheme of the
Methods). Pairing check: the targets, columns and sample order are compared with the saved test predictions of a frozen arm.

  python scripts/hal/knn_test_summary.py --predictions P.npz --raw RAW.json --out summary.json --check-against ARM_predictions.npz
"""
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--raw", required=True, help="summary json written by run_knn_imputation_baseline.py")
    ap.add_argument("--check-against", required=True, help="predictions.npz of a frozen arm on the same split and mask")
    ap.add_argument("--out", required=True)
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    h = np.load(a.predictions); raw = json.load(open(a.raw))
    assert raw["patient_view"] == "test", raw["patient_view"]
    k = int(h["k"]); P, T, R = h["prediction"], h["target"], h["prior_prediction"]
    se, se_ref, ae = (P - T) ** 2, (R - T) ** 2, np.abs(P - T)
    n_s, n_e = P.shape
    arm = np.load(a.check_against)
    order = np.argsort(arm["sample_index"], kind="stable"); same_order = np.array_equal(arm["sample_index"], h["sample_index"])
    if not same_order:  # compare sample by sample after aligning on the matrix row
        i = {int(s): j for j, s in enumerate(arm["sample_index"])}
        idx = np.array([i[int(s)] for s in h["sample_index"]])
    else:
        idx = np.arange(n_s)
    cols_equal = bool(np.array_equal(arm["target_matrix_column"][idx], h["target_matrix_column"]))
    target_max_abs = float(np.max(np.abs(arm["target"][idx] - T)))
    prior_max_abs = float(np.max(np.abs(arm["prior_prediction"][idx] - R)))
    assert cols_equal and target_max_abs < 1e-6, "kNN panels/targets differ from the arms' test panels"

    mse, mae, prior_mse = se.mean(), ae.mean(), se_ref.mean()
    skill = 1 - mse / prior_mse
    chk = raw["results"]["mask_0.50"][f"k{k}"]
    assert abs(chk["mse"] - mse) < 1e-6 and abs(chk["skill_vs_prior"] - skill) < 1e-5, (chk["mse"], mse)
    rng = np.random.default_rng(a.seed)
    s_mse, s_ref, s_ae = se.sum(1), se_ref.sum(1), ae.sum(1)  # equal entries per sample (1,024), so per-sample sums are exact
    bm, bs, ba = [], [], []
    for _ in range(a.replicates):
        d = rng.integers(0, n_s, n_s)
        m, r = s_mse[d].sum() / (n_s * n_e), s_ref[d].sum() / (n_s * n_e)
        bm.append(m); bs.append(1 - m / r); ba.append(s_ae[d].sum() / (n_s * n_e))
    q = lambda x: [float(v) for v in np.quantile(x, [.025, .975])]
    out = {"method": "knn_patient_neighbours", "split": "test", "k": k, "k_source": "fixed a priori (validation primary k=10)",
           "mask_fraction": 0.5, "mask_seed": 17001, "panel_size": 2048, "panel_repeats": 1,
           "n_samples": int(n_s), "n_entries": int(n_s * n_e),
           "mse": float(mse), "mae": float(mae), "prior_mse": float(prior_mse), "skill_vs_prior": float(skill),
           "mse_ci95": q(bm), "mae_ci95": q(ba), "skill_ci95": q(bs),
           "bootstrap": {"unit": "test samples (patients) only; no genomic blocks", "replicates": a.replicates, "seed": a.seed,
                         "ci": "two-sided 95% percentile interval"},
           "pairing_check": {"reference_predictions": a.check_against, "target_matrix_column_identical": cols_equal,
                             "max_abs_target_diff": target_max_abs, "max_abs_prior_diff": prior_max_abs,
                             "same_sample_order": bool(same_order)},
           "metrics_from_benchmark_function": chk, "source_predictions": a.predictions, "source_raw_json": a.raw}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k_: out[k_] for k_ in ["mse", "mse_ci95", "skill_vs_prior", "skill_ci95", "n_samples", "n_entries", "prior_mse"]}, indent=1))


if __name__ == "__main__":
    main()
