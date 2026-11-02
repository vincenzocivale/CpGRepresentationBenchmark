#!/usr/bin/env python3
"""R14: does a frozen representation add anything beyond retrieving similar training patients? (CPU only; saved predictions only.)

Per arm (8 representations) and seed (17/42/97), at masking fraction 0.50 (mask seed 17001, panels of 2,048, 1,024 target entries/sample):
  blend    beta_hat = w_a*arm + w_k*kNN + (1 - w_a - w_k)*prior, weights by least squares on beta over the VALIDATION entries
           (validation kNN k=10 predictions rerun with --save-predictions + the arm's saved validation predictions), applied once,
           unchanged, to the TEST predictions (R9 kNN file + the arm's frozen test predictions)
  avg5050  parameter-free (arm + kNN) / 2
Reported on TEST: MSE and skill (1 - MSE / MSE of the per-CpG train mean) of kNN alone, arm alone, blend, avg5050, with 95% intervals;
relative MSE of blend/avg5050 vs kNN, and of each S2F / ENCODE blend vs the CpGPT blend, with the paired crossed bootstrap of the main
contrasts (samples x 1-Mb blocks, 2,000 replicates, seed 17, same draw for every curve; `weights` is the one of gse90496_contrasts.py and
reproduces statistics.paired_bootstrap, checked below). Seed 'mean' = mean of the per-seed estimates, same draw, as in the main contrasts.
Pairing: every prediction file must match the kNN file entry by entry (sample order, target column, target beta, prior) or the script stops.

  python scripts/hal/knn_blend.py --out-dir outputs/paper_knn_blend_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gse90496_contrasts import ARMS, CPGPT, REF, S2F, SEEDS, ci_p, weights  # noqa: E402

ROOT = HERE.parents[1]
VS_CPGPT = S2F + [REF]


def aligned(h: dict, ref: dict, what: str) -> np.ndarray:
    """Prediction of h with rows in the sample order of ref, after checking that every entry pairs with ref."""
    pos = {int(s): i for i, s in enumerate(h["sample_index"])}
    assert len(pos) == len(h["sample_index"]) == len(ref["sample_index"]), f"{what}: sample sets differ"
    idx = np.array([pos[int(s)] for s in ref["sample_index"]])
    assert np.array_equal(h["target_matrix_column"][idx], ref["target_matrix_column"]), f"{what}: panels/masks differ"
    assert np.allclose(h["target"][idx], ref["target"], atol=1e-7), f"{what}: targets differ"
    assert np.allclose(h["prior_prediction"][idx], ref["prior_prediction"], atol=1e-6), f"{what}: prior differs"
    return h["prediction"][idx].astype(np.float64)


def fit_blend(arm: np.ndarray, knn: np.ndarray, prior: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least squares of y on w_a*arm + w_k*knn + (1 - w_a - w_k)*prior; returns (w_arm, w_knn, w_prior)."""
    X = np.stack([(arm - prior).ravel(), (knn - prior).ravel()], 1)
    wa, wk = np.linalg.lstsq(X, (y - prior).ravel(), rcond=None)[0]
    assert np.isfinite([wa, wk]).all(), (wa, wk)
    return float(wa), float(wk), float(1 - wa - wk)


class Cells:
    """Crossed sample x 1-Mb-block bootstrap of entry-level squared errors (cell sums; same draws for every curve)."""

    def __init__(self, patient: np.ndarray, block: np.ndarray, reps: int, seed: int):
        assert len(set(patient)) == len(patient) == block.shape[0]
        _, pi = np.unique(patient, return_inverse=True)  # sorted order, as statistics.paired_bootstrap
        blocks, bi = np.unique(block, return_inverse=True)
        self.P, self.B, self.E = len(patient), len(blocks), block.shape[1]
        self.flat = (pi[:, None] * self.B + bi.reshape(block.shape)).ravel()
        self.pw, self.bw = weights(self.P, self.B, reps, seed)
        self.n = self._boot(self._cell(np.ones(block.shape)))

    def _cell(self, x):
        return np.bincount(self.flat, weights=x.ravel(), minlength=self.P * self.B).reshape(self.P, self.B)

    def _boot(self, c):
        return np.sum((self.pw @ c) * self.bw, 1)

    def mse(self, se: np.ndarray):
        """(point MSE, bootstrap MSE vector)."""
        c = self._cell(se)
        return c.sum() / (self.P * self.E), self._boot(c) / self.n


def contrast(a, b) -> dict:
    """a, b = (point, boot) MSEs; relative difference (a - b) / b with percentile interval and bootstrap P."""
    (pa, ba), (pb, bb) = a, b
    d, de = pa - pb, ba - bb
    ci, p = ci_p(d, de)
    rci, _ = ci_p(d / pb, de / bb)
    return {"delta_mse": d, "delta_mse_ci_lo": ci[0], "delta_mse_ci_hi": ci[1], "rel_delta_mse": d / pb,
            "rel_delta_mse_ci_lo": rci[0], "rel_delta_mse_ci_hi": rci[1], "p_boot": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--knn-val", default="outputs/paper_knn_blend_v1/knn_k10_mask0.50_validation_predictions.npz")
    ap.add_argument("--knn-test", default="outputs/paper_knn_test_v1/knn_k10_mask0.50_test_predictions.npz")
    ap.add_argument("--manifest", default="configs/experiments/test_confirmation_v2/freeze_manifest.yaml")
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    root = Path(a.root)
    sys.path.insert(0, str(root / "src"))
    from cpg_repr_benchmark.data.methylation import read_axis
    from cpg_repr_benchmark.data.splits import patient_id
    from cpg_repr_benchmark.encode_atlas.statistics import paired_bootstrap, prediction_table
    out = root / a.out_dir
    out.mkdir(parents=True, exist_ok=True)

    man = yaml.safe_load((root / a.manifest).read_text())["runs"]
    arm_file = lambda arm, s, split: root / man[f"{arm}__seed{s}"]["run_dir"] / f"evaluation/{split}/seen/mask_0.50" / "predictions.npz"
    kv, kt = dict(np.load(root / a.knn_val)), dict(np.load(root / a.knn_test))
    assert int(kv["k"]) == int(kt["k"]) == 10
    raw_names = read_axis(root / "data/methylation/tcga_array_official_full.h5")[1]
    names = np.asarray([patient_id(x) for x in raw_names])
    registry = pd.read_parquet(root / "outputs/encode_atlas_v1/loci.parquet")
    blockstr = registry["chr"].astype(str).to_numpy() + ":" + (registry["pos"].to_numpy() // 1_000_000).astype(str)
    cells = Cells(names[kt["sample_index"]], blockstr[kt["target_matrix_column"]], a.replicates, a.seed)
    y_v, pi_v, y_t, pi_t = kv["target"].astype(np.float64), kv["prior_prediction"].astype(np.float64), kt["target"].astype(np.float64), kt["prior_prediction"].astype(np.float64)
    k_v, k_t = kv["prediction"].astype(np.float64), kt["prediction"].astype(np.float64)
    print(f"test: {cells.P} samples, {cells.B} blocks, {cells.P * cells.E} entries", flush=True)

    se = lambda p: (p - y_t) ** 2
    prior_mse = cells.mse(se(pi_t))
    est, rows, wts = {"knn": cells.mse(se(k_t))}, [], {}
    # kNN alone must reproduce R9 (skill 0.587) and the stored validation k=10 run
    r9 = json.loads((root / "outputs/paper_knn_test_v1/summary.json").read_text())
    assert abs(est["knn"][0] - r9["mse"]) < 1e-6 and abs(1 - est["knn"][0] / prior_mse[0] - r9["skill_vs_prior"]) < 1e-5, "kNN test does not reproduce R9"
    assert abs(1 - est["knn"][0] / prior_mse[0] - 0.587) < 1e-3
    v0 = json.loads((root / "outputs/b1_b2_baselines/knn/summary.json").read_text())["results"]["mask_0.50"]["k10"]["mse"]
    assert abs(((k_v - y_v) ** 2).mean() - v0) < 1e-6, "validation kNN rerun does not reproduce the stored k=10 validation MSE"

    preds_t = {}
    for arm in ARMS:
        for s in SEEDS:
            av = aligned(dict(np.load(arm_file(arm, s, "validation"))), kv, f"{arm} s{s} validation")
            at = aligned(dict(np.load(arm_file(arm, s, "test"))), kt, f"{arm} s{s} test")
            for sp, p in (("validation", av), ("test", at)):  # recomputed MSE = the benchmark's metrics.json
                m = json.loads((arm_file(arm, s, sp).parent / "metrics.json").read_text())
                assert abs(((p - (y_v if sp == "validation" else y_t)) ** 2).mean() - m["mse"]) < 2e-6, (arm, s, sp)
            wa, wk, wp = fit_blend(av, k_v, pi_v, y_v)
            blend = pi_t + wa * (at - pi_t) + wk * (k_t - pi_t)
            wts[(arm, s)] = {"w_arm": wa, "w_knn": wk, "w_prior": wp,
                             "val_mse_blend": float(((pi_v + wa * (av - pi_v) + wk * (k_v - pi_v) - y_v) ** 2).mean()),
                             "frac_blend_outside_unit": float(((blend < 0) | (blend > 1)).mean())}
            preds_t[(arm, s)] = {"arm": at, "blend": blend, "avg5050": (at + k_t) / 2}
            for v, p in preds_t[(arm, s)].items():
                est[(arm, s, v)] = cells.mse(se(p))
    # seed 'mean': mean of per-seed point estimates and bootstrap vectors (same draw)
    for arm in ARMS:
        for v in ("arm", "blend", "avg5050"):
            est[(arm, "mean", v)] = (np.mean([est[(arm, s, v)][0] for s in SEEDS]), np.mean([est[(arm, s, v)][1] for s in SEEDS], 0))
        wts[(arm, "mean")] = {k: float(np.mean([wts[(arm, s)][k] for s in SEEDS])) for k in wts[(arm, 17)]}

    skill = lambda e: (1 - e[0] / prior_mse[0], *np.quantile(1 - e[1] / prior_mse[1], [.025, .975]))
    row = lambda arm, seed, v, e, w={}: {"arm": arm, "seed": seed, "variant": v, "mse": e[0], "mse_ci_lo": np.quantile(e[1], .025), "mse_ci_hi": np.quantile(e[1], .975),
                                         **dict(zip(("skill", "skill_ci_lo", "skill_ci_hi"), skill(e))), **w}
    rows.append(row("knn", "-", "knn", est["knn"]))
    for arm in ARMS:
        for s in [*SEEDS, "mean"]:
            for v in ("arm", "blend", "avg5050"):
                rows.append(row(arm, s, v, est[(arm, s, v)], wts[(arm, s)] if v == "blend" else {}))
    ctr = []
    for arm in ARMS:
        for s in [*SEEDS, "mean"]:
            for v in ("blend", "avg5050"):
                ctr.append({"arm": arm, "reference": "knn", "seed": s, "variant": v, **contrast(est[(arm, s, v)], est["knn"])})
            if arm in VS_CPGPT:
                for v in ("blend", "avg5050"):
                    ctr.append({"arm": arm, "reference": CPGPT, "seed": s, "variant": v, **contrast(est[(arm, s, v)], est[(CPGPT, s, v)])})
    res, ctr = pd.DataFrame(rows), pd.DataFrame(ctr)
    assert np.isfinite(res[["mse", "skill"]].to_numpy()).all() and np.isfinite(ctr.rel_delta_mse.to_numpy()).all()

    # the contrasts must equal the repository's paired_bootstrap (pre-specified test implementation)
    checks = []
    for tag, p in (("arm", preds_t[("enformer_latent_native", 17)]["arm"]), ("blend", preds_t[("enformer_latent_native", 17)]["blend"])):
        r = paired_bootstrap(prediction_table(kt, registry, raw_names), prediction_table({**kt, "prediction": p}, registry, raw_names),
                             seed=a.seed, replicates=a.replicates)
        mine = contrast(cells.mse(se(p)), est["knn"])
        ok = abs(r["relative_delta_mse"] - mine["rel_delta_mse"]) < 1e-9 and np.allclose(r["relative_ci95"], [mine["rel_delta_mse_ci_lo"], mine["rel_delta_mse_ci_hi"]], atol=1e-9)
        assert ok, (tag, r, mine)
        checks.append({"check": f"enformer s17 {tag} vs kNN equals statistics.paired_bootstrap", "ok": bool(ok)})

    res.to_csv(out / "blend_results.csv", index=False)
    ctr.to_csv(out / "blend_contrasts.csv", index=False)
    mean = lambda v, c: ctr[(ctr.seed == "mean") & (ctr.variant == v) & (ctr.reference == c)].set_index("arm")
    summ = {"split": "test", "mask_fraction": 0.5, "k": 10, "n_samples": cells.P, "n_blocks": cells.B, "n_entries": cells.P * cells.E,
            "replicates": a.replicates, "seed": a.seed, "prior_mse": prior_mse[0],
            "knn_alone": res[res.arm == "knn"].iloc[0].to_dict(), "self_checks": checks,
            "per_arm_seed_mean": {arm: {"skill_arm": res[(res.arm == arm) & (res.seed == "mean") & (res.variant == "arm")].skill.iloc[0],
                                         "skill_blend": res[(res.arm == arm) & (res.seed == "mean") & (res.variant == "blend")].skill.iloc[0],
                                         "skill_avg5050": res[(res.arm == arm) & (res.seed == "mean") & (res.variant == "avg5050")].skill.iloc[0],
                                         **wts[(arm, "mean")],
                                         "blend_vs_knn_rel_mse": mean("blend", "knn").loc[arm, ["rel_delta_mse", "rel_delta_mse_ci_lo", "rel_delta_mse_ci_hi"]].tolist(),
                                         "avg5050_vs_knn_rel_mse": mean("avg5050", "knn").loc[arm, ["rel_delta_mse", "rel_delta_mse_ci_lo", "rel_delta_mse_ci_hi"]].tolist(),
                                         **({"blend_vs_cpgpt_blend_rel_mse": mean("blend", CPGPT).loc[arm, ["rel_delta_mse", "rel_delta_mse_ci_lo", "rel_delta_mse_ci_hi"]].tolist()} if arm in VS_CPGPT else {})}
                                 for arm in ARMS},
            "pairing": "all 48 prediction files (24 validation + 24 test) matched the kNN files entry by entry (sample order, target column, target beta, prior)"}
    (out / "blend_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    pd.set_option("display.width", 220)
    print(res[res.seed.isin(["-", "mean"])][["arm", "variant", "mse", "skill", "w_arm", "w_knn", "w_prior"]].round(5).to_string(index=False))
    print(ctr[ctr.seed == "mean"][["arm", "reference", "variant", "rel_delta_mse", "rel_delta_mse_ci_lo", "rel_delta_mse_ci_hi"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
