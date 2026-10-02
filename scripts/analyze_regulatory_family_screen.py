"""Analysis of the regulatory family screen (docs/REGULATORY_FAMILY_SCREEN.md).

Validation patients only, 50% masking, one decoder seed. Reads only: per-run summary/history/predictions, the
embedding stores (to find all-zero rows) and the binary track CSR of the feature store (never `/dense`, never
methylation values; the methylation file is opened for the `sample_name` axis only).
Writes tables to outputs/regulatory_family_screen_v1/analysis/.

    python scripts/analyze_regulatory_family_screen.py [--replicates 2000] [--diag-replicates 1000]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cpg_repr_benchmark.data.methylation import read_axis
from cpg_repr_benchmark.encode_atlas.compression import TrackStore
from cpg_repr_benchmark.encode_atlas.feature_sets import resolve_feature_set
from cpg_repr_benchmark.encode_atlas.statistics import paired_bootstrap, prediction_table

SCREEN = ROOT / "outputs/regulatory_family_screen_v1"
CAMPAIGN = ROOT / "outputs/encode_atlas_v1"
ARMS = ["regulatory_histone", "regulatory_histone_dnase", "regulatory_histone_tf", "regulatory_clean"]
REF = "regulatory_histone"
SEED = 17
FIT_CHR = {f"chr{i}" for i in range(1, 20)}
STORE = ROOT / "data/derived/functional_annotations/reference_functional_features_tcga_array_all_v1_cpgidx_fixed.h5"
CONTRASTS = [("regulatory_histone", "regulatory_histone_dnase"), ("regulatory_histone", "regulatory_histone_tf"),
             ("regulatory_histone", "regulatory_clean"), ("regulatory_histone_dnase", "regulatory_clean"),
             ("regulatory_histone_tf", "regulatory_clean"), ("regulatory_histone_dnase", "regulatory_histone_tf")]


def run_dir(arm: str) -> Path:
    assert (SCREEN / "logs" / f"{arm}__seed{SEED}.done").exists(), f"{arm}: no .done marker"
    lines = (SCREEN / "logs" / f"{arm}__seed{SEED}.log").read_text().splitlines()
    return Path([x for x in lines if x.startswith("RUN_DIR=")][-1].split("=", 1)[1])


def mtime(p: Path) -> float:
    return p.stat().st_mtime


def per_arm_metrics(arm: str):
    d = run_dir(arm)
    cfg = yaml.safe_load((d / "resolved_config.yaml").read_text())
    assert cfg["evaluation"]["patient_view"] == "validation", "non-validation view"
    assert cfg["evaluation"]["require_patient_view"] == "validation"
    metrics = json.loads((d / "evaluation/seen/mask_0.50/metrics.json").read_text())
    assert metrics["patient_view"] == "validation"
    hist = json.loads((d / "history.json").read_text())
    best = min(hist, key=lambda r: r["validation_mse"])
    last = hist[-1]
    with np.load(d / "evaluation/seen/mask_0.50/predictions.npz") as h:
        prior_mse = float(np.mean((h["prior_prediction"].astype(np.float64) - h["target"]) ** 2))
    t_start = mtime(d / "experiment.json")          # splits/prior done, training starts
    t_train_end = mtime(d / "history.json")         # last epoch written
    t_end = mtime(d / "summary.json")               # after the 50% evaluation
    t_create = pd.Timestamp(d.name.split("-")[0], tz="UTC").timestamp()  # run dir creation (embedding load etc.)
    row = {"arm": arm, "run_dir": d.name, "n_pairs": metrics["n_pairs"], "mse": metrics["mse"], "mae": metrics["mae"],
           "mas_pcc": metrics["mas_pcc"], "mac_pcc": metrics["mac_pcc"], "prior_mse_metrics": metrics["prior_mse"],
           "prior_mse_predictions": prior_mse, "skill_vs_prior": metrics["skill_vs_prior"],
           "best_epoch": best["epoch"], "n_epochs": len(hist), "best_val_mse": best["validation_mse"],
           "last_epoch_val_mse": last["validation_mse"], "last_epoch_val_mae": last["validation_mae"],
           "last_epoch_val_mas_pcc": last["validation_mas_pcc"], "last_epoch_val_mac_pcc": last["validation_mac_pcc"],
           "last_minus_best_mse": last["validation_mse"] - best["validation_mse"],
           "train_seconds": t_train_end - t_start, "train_plus_eval_seconds": t_end - t_start,
           "run_wall_seconds": t_end - t_create}
    return row, hist, d


def abs_error_view(outputs: dict) -> dict:
    """Outputs whose squared error equals the ABSOLUTE error, so prediction_table/paired_bootstrap give MAE."""
    err = np.sqrt(np.abs(outputs["prediction"].astype(np.float64) - outputs["target"].astype(np.float64)))
    return {**outputs, "target": outputs["target"].astype(np.float64),
            "prediction": outputs["target"].astype(np.float64) + err}


def replicate_estimates(ref: pd.DataFrame, alt: pd.DataFrame, seed: int, replicates: int):
    """Bootstrap replicates of delta (same RNG stream as statistics.paired_bootstrap) for the fraction favoring alt."""
    key = ["patient", "column", "block"]
    pairs = ref.merge(alt, on=key, suffixes=("_r", "_a"), validate="one_to_one")
    delta = (pairs.squared_error_a - pairs.squared_error_r).to_numpy()
    obs = pairs.n_r.to_numpy()
    _, pidx = np.unique(pairs.patient, return_inverse=True)
    _, bidx = np.unique(pairs.block, return_inverse=True)
    npat, nblk = pidx.max() + 1, bidx.max() + 1
    flat = pidx * nblk + bidx
    dc = np.bincount(flat, weights=obs * delta, minlength=npat * nblk).reshape(npat, nblk)
    cc = np.bincount(flat, weights=obs, minlength=npat * nblk).reshape(npat, nblk)
    rng = np.random.default_rng(seed)
    pw, bw = [], []
    for _ in range(replicates):
        pw.append(rng.multinomial(npat, np.full(npat, 1 / npat)))
        bw.append(rng.multinomial(nblk, np.full(nblk, 1 / nblk)))
    pw, bw = np.asarray(pw), np.asarray(bw)
    n = np.sum((pw @ cc) * bw, axis=1)
    est = np.sum((pw @ dc) * bw, axis=1)
    return est[n > 0] / n[n > 0]


def boot(ref, alt, seed, reps, subset=None):
    if subset is not None:
        ref, alt = ref[subset(ref)], alt[subset(alt)]
    r = paired_bootstrap(ref, alt, seed=seed, replicates=reps)
    mse_ref = r["delta_mse"] / r["relative_delta_mse"]
    return {"mse_reference": mse_ref, "mse_alternative": mse_ref + r["delta_mse"], "delta": r["delta_mse"],
            "relative": r["relative_delta_mse"], "ci_lo": r["ci95"][0], "ci_hi": r["ci95"][1],
            "rel_ci_lo": r["relative_ci95"][0], "rel_ci_hi": r["relative_ci95"][1], "n_pairs": r["n_pairs"],
            "n_patients": r["n_patients"], "n_blocks": r["n_blocks"]}


def zero_rows(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h:
        ids = h["cpg_idx"][:]
        emb = h["embedding"]
        z = np.empty(len(ids), bool)
        for s in range(0, len(ids), 50_000):
            z[s:s + 50_000] = ~np.any(emb[s:s + 50_000] != 0, axis=1)
    return ids, z


def quartile_labels(values: np.ndarray):
    edges = np.quantile(values, [.25, .5, .75])
    return np.searchsorted(edges, values, side="right"), edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--diag-replicates", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-diagnostics", action="store_true")
    args = ap.parse_args()
    os.nice(10)
    out = SCREEN / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. per-arm metrics -------------------------------------------------------------------------------
    rows, hists, dirs = [], {}, {}
    for arm in ARMS:
        row, hist, d = per_arm_metrics(arm)
        rows.append(row), hists.update({arm: hist}), dirs.update({arm: d})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "per_arm_metrics.csv", index=False)
    pd.DataFrame([{"arm": a, **r} for a, h in hists.items() for r in h]).to_csv(out / "history_all_arms.csv", index=False)

    # ---- 2. tables + pairing checks ---------------------------------------------------------------------------
    registry = pd.read_parquet(CAMPAIGN / "loci.parquet")
    cfg0 = yaml.safe_load((dirs[REF] / "resolved_config.yaml").read_text())
    _, names = read_axis(ROOT / cfg0["dataset"]["methylation_h5"])
    raw, tab_sq, tab_abs = {}, {}, {}
    for arm in ARMS:
        with np.load(dirs[arm] / "evaluation/seen/mask_0.50/predictions.npz") as h:
            o = {k: h[k] for k in h}
        raw[arm] = o
        tab_sq[arm] = prediction_table(o, registry, names)
        tab_abs[arm] = prediction_table(abs_error_view(o), registry, names)
    ref_o = raw[REF]
    pairing = {"n_validation_patients": len(ref_o["sample_index"]), "panel_loci_per_patient":
               int(ref_o["target_matrix_column"].shape[1])}
    for arm in ARMS[1:]:
        o = raw[arm]
        pairing[arm] = {k: bool(np.array_equal(o[k], ref_o[k]))
                        for k in ("sample_index", "target_matrix_column", "panel_repeat", "prior_prediction", "target")}
    pairing["all_identical"] = all(v for a in ARMS[1:] for v in pairing[a].values())
    (out / "pairing_checks.json").write_text(json.dumps(pairing, indent=2))
    if not pairing["all_identical"]:
        raise SystemExit("pairing differs across arms; see pairing_checks.json")

    # ---- 3. paired contrasts ---------------------------------------------------------------------------------
    comp = []
    for a, b in CONTRASTS:
        for metric, tabs in (("mse", tab_sq), ("mae", tab_abs)):
            r = boot(tabs[a], tabs[b], args.seed, args.replicates)
            est = replicate_estimates(tabs[a], tabs[b], args.seed, args.replicates)
            assert abs(np.quantile(est, .025) - r["ci_lo"]) < 1e-12 and abs(np.quantile(est, .975) - r["ci_hi"]) < 1e-12
            comp.append({"reference": a, "alternative": b, "metric": metric, **r,
                         "frac_replicates_favoring_alt": float(np.mean(est < 0)), "replicates": args.replicates,
                         "seed": args.seed})
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "paired_contrasts.csv", index=False)

    # ---- 4. criterion --------------------------------------------------------------------------------------
    crit = []
    for arm in ARMS[1:]:
        m = comp[(comp.reference == REF) & (comp.alternative == arm) & (comp.metric == "mse")].iloc[0]
        a = comp[(comp.reference == REF) & (comp.alternative == arm) & (comp.metric == "mae")].iloc[0]
        c1, c2, c3 = bool(m.relative <= -0.01), bool(m.ci_hi < 0), bool(a.delta < 0)
        crit.append({"arm": arm, "rel_mse_change": m.relative, "ge_1pct_improvement": c1, "mse_ci_below_0": c2,
                     "mae_same_sign": c3, "mae_ci_below_0": bool(a.ci_hi < 0), "passes": c1 and c2 and c3})
    pd.DataFrame(crit).to_csv(out / "criterion.csv", index=False)
    if args.skip_diagnostics:
        return

    # ---- 5. diagnostics ----------------------------------------------------------------------------------------
    loci_ids = registry.cpg_idx.to_numpy()
    chrom = registry.chr.astype(str).to_numpy()
    in_fit = np.isin(chrom, list(FIT_CHR))
    diag = []

    def add(kind, group, label, ref_arm, arm, subset):
        for metric, tabs in (("mse", tab_sq), ("mae", tab_abs)):
            r = boot(tabs[ref_arm], tabs[arm], args.seed, args.diag_replicates, subset)
            diag.append({"diagnostic": kind, "group": group, "label": label, "reference": ref_arm, "alternative": arm,
                         "metric": metric, **r})

    # (c) in-fit vs out-of-fit
    for arm in ARMS[1:]:
        for g, mask in (("in_fit_chr1_19", in_fit), ("out_of_fit_chr20_22", ~in_fit)):
            add("fit_scope", g, g, REF, arm, lambda t, m=mask: m[t.column.to_numpy()])
    # absolute MSE per arm by fit scope (own MSE incl. histone)
    scope_rows = []
    for arm in ARMS:
        t = tab_sq[arm]
        for g, mask in (("in_fit_chr1_19", in_fit), ("out_of_fit_chr20_22", ~in_fit)):
            s = t[mask[t.column.to_numpy()]]
            scope_rows.append({"arm": arm, "group": g, "mse": float(np.average(s.squared_error, weights=s.n)),
                               "n_pairs": len(s), "n_loci": int(s.column.nunique())})
    pd.DataFrame(scope_rows).to_csv(out / "mse_by_fit_scope.csv", index=False)

    # (a) all-zero embedding rows, and (b) track coverage quartiles
    store = TrackStore(STORE)  # reads only cpg_idx, track_indptr, track_indices
    assert set(store.datasets_read) <= {"cpg_idx", "track_indptr", "track_indices"}
    srow = store.rows_for(loci_ids)
    cov = {}
    for arm in ARMS:
        fs = resolve_feature_set(arm)
        cols = np.asarray(fs.feature_columns)
        cov[arm] = np.asarray(store.matrix[:, cols].sum(axis=1)).ravel()[srow]
    zero_info = []
    for arm in ARMS:
        ids, z = zero_rows(ROOT / yaml.safe_load((dirs[arm] / "resolved_config.yaml").read_text())
                           ["representation"]["store_h5"])
        order = np.argsort(ids)
        zl = z[order[np.searchsorted(ids[order], loci_ids)]]
        in_panel = np.zeros(len(loci_ids), bool)
        in_panel[np.unique(raw[arm]["target_matrix_column"])] = True
        zero_info.append({"arm": arm, "zero_loci_universe": int(zl.sum()), "n_loci_universe": len(zl),
                          "zero_loci_in_eval_panel": int((zl & in_panel).sum()), "loci_in_eval_panel": int(in_panel.sum()),
                          "zero_equals_no_track_overlap": bool(np.array_equal(zl, cov[arm] == 0)),
                          "zero_loci_chr20_22": int((zl & ~in_fit).sum())})
        t = tab_sq[arm]
        zmask = zl[t.column.to_numpy()]
        zero_info[-1].update({"mse_zero_rows": float(np.average(t.squared_error[zmask], weights=t.n[zmask]))
                              if zmask.any() else np.nan,
                              "mse_nonzero_rows": float(np.average(t.squared_error[~zmask], weights=t.n[~zmask])),
                              "pairs_zero_rows": int(zmask.sum())})
        if arm != REF:
            if zmask.sum() > 0:
                add("zero_embedding", "zero_rows", "all-zero embedding", REF, arm, lambda tt, m=zl: m[tt.column.to_numpy()])
            add("zero_embedding", "nonzero_rows", "non-zero embedding", REF, arm, lambda tt, m=zl: ~m[tt.column.to_numpy()])
        zero_info[-1]["_zl"] = zl
    zdf = pd.DataFrame([{k: v for k, v in r.items() if k != "_zl"} for r in zero_info])
    zdf.to_csv(out / "zero_embedding_by_arm.csv", index=False)

    # (b) quartiles: (i) arm's own total track count, (ii) histone-only track count (common stratifier)
    qinfo = []
    for kind, getter in (("own_track_count", lambda arm: cov[arm]), ("histone_track_count", lambda arm: cov[REF])):
        for arm in ARMS[1:]:
            lab, _ = quartile_labels(getter(arm))
            for q in range(4):
                sel = lab == q
                qinfo.append({"stratifier": kind, "arm": arm, "quartile": q + 1, "n_loci": int(sel.sum()),
                              "count_min": float(getter(arm)[sel].min()) if sel.any() else np.nan,
                              "count_max": float(getter(arm)[sel].max()) if sel.any() else np.nan})
                if sel.sum():
                    add(f"quartile_{kind}", f"Q{q + 1}", f"{kind} Q{q + 1}", REF, arm,
                        lambda t, s=sel: s[t.column.to_numpy()])
    pd.DataFrame(qinfo).to_csv(out / "quartile_definitions.csv", index=False)
    pd.DataFrame(diag).to_csv(out / "diagnostic_strata_contrasts.csv", index=False)
    cov_summary = {arm: {"mean": float(cov[arm].mean()), "median": float(np.median(cov[arm])),
                         "zero_fraction": float((cov[arm] == 0).mean())} for arm in ARMS}
    (out / "coverage_summary.json").write_text(json.dumps(cov_summary, indent=2))
    print(metrics.T.to_string())
    print(comp.to_string())
    print(pd.DataFrame(crit).to_string())


if __name__ == "__main__":
    main()
