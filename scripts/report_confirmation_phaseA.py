#!/usr/bin/env python3
# ruff: noqa: ISC004
"""Descriptive report of the Phase A confirmation runs (4 arms x seeds 17/42/97; VALIDATION ONLY; test data is never read).

Reads only: <run>/{confirmation_status.json,history.json,evaluation/validation/seen/mask_*/{metrics.json,predictions.npz}} and the paired-bootstrap CSVs
written by scripts/analyze_confirmation_matrix.py (analysis dir). Writes CSVs + a figure to the analysis dir and a markdown report to docs/.
No selection / decision rule is applied; statements are descriptive. Never writes into run directories.

    python scripts/report_confirmation_phaseA.py [--analysis-dir DIR] [--md PATH] [--no-zero-rows]
    <matplotlib env> python scripts/report_confirmation_phaseA.py --plot-only      # png of validation MSE vs epoch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FRACTIONS = (0.15, 0.30, 0.50, 0.70, 0.90)
ARMS = ("regulatory_histone_dnase", "functional_annotations_pca", "cpgpt_large_locus", "deepcpg_dna_locus")
CANDIDATE = "regulatory_histone_dnase"
SEEDS = (17, 42, 97)
DIMS = {"regulatory_histone_dnase": 256, "functional_annotations_pca": 256, "cpgpt_large_locus": 512, "deepcpg_dna_locus": 128}
LABEL = "validation only, trained_validation_frozen, NOT confirmed, test never read"


# ------------------------------------------------------------------------------------------------ pure helpers
def adapter_params(d: int, latent: int = 256) -> int:
    """LayerNorm(d) -> Linear(d, latent) -> GELU -> LayerNorm(latent)."""
    return 2 * d + d * latent + latent + 2 * latent


def curve_stats(mse) -> dict:
    """Last-epoch behaviour of one validation-MSE curve (epochs 0-based)."""
    m = np.asarray(mse, float)
    n = len(m)
    out = {"n_epochs": n, "best_epoch": int(np.argmin(m)), "final_mse": float(m[-1])}
    out["best_in_last10"] = bool(out["best_epoch"] >= n - 10)
    for k in (10, 20):
        w, x = m[-k:], np.arange(k)
        slope = float(np.polyfit(x, w, 1)[0])
        out[f"slope_last{k}_pct_per_epoch"] = 100.0 * slope / float(w.mean())
        if k == 20:
            res = w - np.polyval(np.polyfit(x, w, 1), x)
            out["noise_sd_detrended_last20"] = float(np.sqrt((res ** 2).sum() / (k - 2)))
            out["noise_sd_pct_of_mse"] = 100.0 * out["noise_sd_detrended_last20"] / float(w.mean())
    out["mean_last10"] = float(m[-10:].mean())
    for e in (60, 90, 110):
        if e < n:
            out[f"mse_ep{e}"] = float(m[e])
    return out


def seed_summary(df: pd.DataFrame, cols) -> pd.DataFrame:
    rows = []
    for arm, d in df.groupby("arm", sort=False):
        r = {"arm": arm, "n_seeds": len(d)}
        for c in cols:
            v = d[c].to_numpy(float)
            r[f"{c}_mean"], r[f"{c}_sd"] = float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))
            r[f"{c}_min"], r[f"{c}_max"] = float(v.min()), float(v.max())
        rows.append(r)
    return pd.DataFrame(rows)


def sign_counts(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Per comparator x metric: number of (seed, fraction) cells in which the reference is better (delta > 0) and CI excludes 0."""
    rows = []
    for (c, m), d in contrasts.groupby(["comparator", "metric"]):
        rows.append({"comparator": c, "metric": m, "cells": len(d), "reference_better": int((d.delta > 0).sum()),
                     "ci_excludes0_reference_better": int((d.ci_lo > 0).sum()), "ci_excludes0_comparator_better": int((d.ci_hi < 0).sum())})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    names = [str(c) for c in cols]
    lines = ["| " + " | ".join(names) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            cells.append(f.format(v) if (f and isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ loading
def load_all(runs_root: Path) -> dict:
    out = {}
    for arm in ARMS:
        for s in SEEDS:
            log = runs_root / "logs" / f"{arm}__seed{s}.log"
            rd = Path([x for x in log.read_text().splitlines() if x.startswith("RUN_DIR=")][-1].split("=", 1)[1])
            st = json.loads((rd / "confirmation_status.json").read_text())
            hist = json.loads((rd / "history.json").read_text())
            met = {f: json.loads((rd / "evaluation" / "validation" / "seen" / f"mask_{f:.2f}" / "metrics.json").read_text()) for f in FRACTIONS}
            for f, m in met.items():
                if m.get("patient_view") != "validation":
                    raise PermissionError(f"{arm}/{s}/{f}: not a validation metrics file")
            out[(arm, s)] = {"dir": rd, "status": st, "hist": hist, "metrics": met}
    return out


def run_tables(runs: dict):
    per_run, per_frac = [], []
    for (arm, s), r in runs.items():
        m50, st = r["metrics"][0.50], r["status"]
        per_run.append({"arm": arm, "seed": s, "dim": DIMS[arm], "best_epoch": st["best_epoch_0based"], "epochs_run": st["epochs_run"],
                        "mse": m50["mse"], "mae": m50["mae"], "mas_pcc": m50["mas_pcc"], "mac_pcc": m50["mac_pcc"],
                        "prior_mse": m50["prior_mse"], "skill_vs_prior": m50["skill_vs_prior"],
                        "wall_clock_h": st["wall_clock_seconds"] / 3600, "best_pt_path": st["best_pt_path"], "best_pt_sha256": st["best_pt_sha256"]})
        for f, m in r["metrics"].items():
            per_frac.append({"arm": arm, "seed": s, "mask_fraction": f, "mse": m["mse"], "mae": m["mae"], "mas_pcc": m["mas_pcc"],
                             "mac_pcc": m["mac_pcc"], "prior_mse": m["prior_mse"], "skill_vs_prior": m["skill_vs_prior"]})
    return pd.DataFrame(per_run), pd.DataFrame(per_frac)


def curve_table(runs: dict) -> pd.DataFrame:
    rows = []
    for (arm, s), r in runs.items():
        mse = [h["validation_mse"] for h in r["hist"]]
        rows.append({"arm": arm, "seed": s, **curve_stats(mse), "nonfinite_losses": int(sum(
            not np.isfinite(h[k]) for h in r["hist"] for k in ("train_mse", "validation_mse") if k in h))})
    return pd.DataFrame(rows)


def last10_gap(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SEEDS:
        ref = curves[(curves.arm == CANDIDATE) & (curves.seed == s)].mean_last10.iloc[0]
        for a in ARMS:
            if a != CANDIDATE:
                v = curves[(curves.arm == a) & (curves.seed == s)].mean_last10.iloc[0]
                rows.append({"comparator": a, "seed": s, "delta_last10": v - ref, "rel_pct_last10": 100 * (v - ref) / ref})
    return pd.DataFrame(rows)


def zero_row_table(runs: dict, registry: pd.DataFrame, store_path: Path) -> pd.DataFrame:
    """MSE at 0.50 on loci whose candidate embedding row is all-zero vs the others (same loci for every arm; predictions only)."""
    import h5py
    with h5py.File(store_path, "r") as h:
        ids, emb = h["cpg_idx"][:], h["embedding"]
        zero = np.zeros(len(ids), bool)
        for i in range(0, len(ids), 50000):
            zero[i:i + 50000] = ~np.any(emb[i:i + 50000] != 0, axis=1)
    order = np.argsort(ids)
    zl = zero[order[np.searchsorted(ids[order], registry.cpg_idx.to_numpy())]]  # indexed by matrix column
    rows = []
    for (arm, s), r in runs.items():
        with np.load(r["dir"] / "evaluation" / "validation" / "seen" / "mask_0.50" / "predictions.npz") as h:
            col, y, p, pr = h["target_matrix_column"], h["target"], h["prediction"], h["prior_prediction"]
        z = zl[col]
        se, sp = (p - y) ** 2, (pr - y) ** 2
        rows.append({"arm": arm, "seed": s, "zero_loci_total": int(zl.sum()), "pairs_zero": int(z.sum()), "pairs_total": int(z.size),
                     "share_pairs_zero_pct": 100 * float(z.mean()), "mse_zero_rows": float(se[z].mean()) if z.any() else float("nan"),
                     "mse_nonzero_rows": float(se[~z].mean()), "prior_mse_zero_rows": float(sp[z].mean()) if z.any() else float("nan"),
                     "prior_mse_nonzero_rows": float(sp[~z].mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ plot
def plot_curves(runs: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"regulatory_histone_dnase": "#1b6ca8", "functional_annotations_pca": "#d1792b", "cpgpt_large_locus": "#2e8b57",
              "deepcpg_dna_locus": "#8e4a9c"}
    styles = {17: "-", 42: "--", 97: ":"}
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 2]})
    for (arm, s), r in runs.items():
        m = np.array([h["validation_mse"] for h in r["hist"]])
        for a, lo in ((ax[0], 0), (ax[1], 90)):
            if a is ax[1] and arm not in (CANDIDATE, "functional_annotations_pca"):
                continue
            a.plot(np.arange(len(m))[lo:], m[lo:], color=colors[arm], ls=styles[s], lw=1.4)
    for a, lo in ((ax[0], 0), (ax[1], 90)):
        a.set_xlabel("epoch (0-based)")
        a.grid(alpha=.25)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylim(0.0140, 0.0215)
    ax[1].set_ylim(0.01462, 0.01505)
    ax[0].set_ylabel("validation MSE @ mask 0.50 (best.pt selection metric)")
    ends = {a: np.mean([runs[(a, s)]["hist"][-1]["validation_mse"] for s in SEEDS]) for a in (CANDIDATE, "functional_annotations_pca")}
    for a, y in ends.items():
        ax[1].annotate(a, (119.5, y), color=colors[a], fontsize=9, va="top" if a != CANDIDATE else "bottom", xytext=(4, 0),
                       textcoords="offset points", annotation_clip=False)
    for a in ARMS:
        if a not in ends:
            ax[0].annotate(a, (119.5, np.mean([runs[(a, s)]["hist"][-1]["validation_mse"] for s in SEEDS])), color=colors[a], fontsize=8,
                           va="bottom", xytext=(-4, 12), textcoords="offset points", ha="right")
    ax[1].set_xlim(90, 119)
    ax[0].set_title("all epochs (line style: solid 17, dashed 42, dotted 97)")
    ax[1].set_title("epochs 90-119, candidate vs functional only (zoom)")
    fig.suptitle("Phase A: " + LABEL, fontsize=10)
    fig.subplots_adjust(right=0.80)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ markdown
def build_markdown(per_run, per_frac, summ50, summ_frac, contrasts, across, signs, curves, gap10, zero, ref_f, integrity, png_rel) -> str:
    f6 = {c: "{:.6f}" for c in ("mse", "mae", "mas_pcc", "mac_pcc", "prior_mse", "skill_vs_prior")}
    L = ["# Regulatory confirmation matrix, Phase A results\n",
         f"> **{LABEL}.** Descriptive of representation arms under the frozen protocol (120 epochs, early stopping off, batch 8, `best.pt` by "
         "validation MSE @ 0.50, evaluated at 0.15/0.30/0.50/0.70/0.90 on VALIDATION patients). No change to the candidate, the protocol or the matrix. "
         "No selection or decision rule is applied here.\n",
         "Generated by `scripts/report_confirmation_phaseA.py` from `outputs/regulatory_confirmation_v1/` (CSVs under `analysis/`).\n",
         "## 1. Integrity\n", f"- 12/12 runs `trained_validation_frozen`: **{integrity['n_trained_validation_frozen']}/{integrity['n_expected']}**; "
         f"integrity check ok = **{integrity['ok']}**, anomalies = {integrity['anomalies'] or 'none'}; `test_read` = {integrity['aggregate_test_read']}; "
         f"`test` paths found under `benchmark/` = {len(integrity['evaluation_test_paths_found'])}.",
         "- Details (best.pt sha256 recomputed, store sha256 vs matrix.yaml, config hash, stray dirs): `analysis/integrity.json`.\n",
         "## 2. Per arm and seed at mask 0.50 (final evaluation of `best.pt`)\n",
         "Best epoch is 0-based (last epoch = 119). Skill = 1 - MSE/prior-MSE (prior-only MSE taken from the same predictions).\n",
         md_table(per_run.drop(columns=["best_pt_path", "best_pt_sha256", "wall_clock_h"]), f6), "",
         "### Seed mean +/- SD (n=3) at 0.50\n",
         md_table(summ50.assign(**{c: summ50[c] for c in []})[["arm", "best_epoch_mean", "mse_mean", "mse_sd", "mae_mean", "mae_sd", "mas_pcc_mean",
                                                             "mas_pcc_sd", "mac_pcc_mean", "mac_pcc_sd", "skill_vs_prior_mean", "skill_vs_prior_sd"]],
                  {c: "{:.6f}" for c in summ50.columns if c not in ("arm", "best_epoch_mean")} | {"best_epoch_mean": "{:.1f}"}), "",
         "## 3. All five mask fractions (seed mean +/- SD)\n", md_table(summ_frac, {c: "{:.6f}" for c in summ_frac.columns if c not in ("arm", "mask_fraction")}), "",
         "Per arm/seed/fraction values: `analysis/per_run_all_fractions.csv`.\n",
         "## 4. Paired contrasts vs the frozen candidate `regulatory_histone_dnase`\n",
         "delta = comparator - candidate (MSE or MAE); **positive = candidate better**. rel = delta / candidate value. Uncertainty: crossed patient x 1 Mb "
         "block bootstrap, 2000 replicates, seed 17 (`encode_atlas.statistics.paired_bootstrap`). `fav. comparator` = fraction of replicates in which the "
         "comparator has lower MSE/MAE. Pairing (sample_index, target_matrix_column, panel_repeat, target, prior_prediction) was verified identical across "
         "arms within every seed and fraction (`analysis/pairing_checks.json`; the analysis script aborts otherwise).\n"]
    c50 = contrasts[(contrasts.mask_fraction == 0.5)]
    for metric in ("mse", "mae"):
        t = c50[c50.metric == metric].copy()
        t["rel_pct"] = 100 * t.relative
        t["rel_ci_pct"] = [f"[{100 * a:.2f}, {100 * b:.2f}]" for a, b in zip(t.rel_ci_lo, t.rel_ci_hi)]
        t["ci95"] = [f"[{a:.2e}, {b:.2e}]" for a, b in zip(t.ci_lo, t.ci_hi)]
        L += [f"### {metric.upper()} at 0.50, per seed\n", md_table(t[["comparator", "seed", "delta", "ci95", "rel_pct", "rel_ci_pct",
                                                                      "frac_replicates_favoring_alt", "d_mas_pcc", "d_mac_pcc"]].rename(
            columns={"frac_replicates_favoring_alt": "fav. comparator"}), {"delta": "{:.2e}", "rel_pct": "{:+.2f}", "fav. comparator": "{:.3f}",
                                                                           "d_mas_pcc": "{:+.4f}", "d_mac_pcc": "{:+.4f}"}), ""]
    a50 = across[across.mask_fraction == 0.5].copy()
    L += ["### Across seeds at 0.50 (mean of the three per-seed paired estimates)\n",
          "No across-seed bootstrap CI exists (seeds are not a sampling unit; n=3). `k_ci_excl0` = number of seeds whose per-seed 95% CI excludes 0 "
          "in the stated direction.\n",
          md_table(a50[["comparator", "metric", "mean_delta", "mean_relative", "min_relative", "max_relative", "k_candidate_better",
                        "k_ci_excludes0_candidate_better", "k_ci_excludes0_comparator_better", "mean_d_mas_pcc", "mean_d_mac_pcc"]],
                   {"mean_delta": "{:.2e}", "mean_relative": "{:+.4f}", "min_relative": "{:+.4f}", "max_relative": "{:+.4f}",
                    "mean_d_mas_pcc": "{:+.4f}", "mean_d_mac_pcc": "{:+.4f}"}), "",
          "### All fractions: per-seed values, relative delta (%) with 95% CI, MSE\n"]
    t = contrasts[contrasts.metric == "mse"].copy()
    t["rel [95% CI] %"] = [f"{100 * r:+.2f} [{100 * a:+.2f}, {100 * b:+.2f}]" for r, a, b in zip(t.relative, t.rel_ci_lo, t.rel_ci_hi)]
    piv = t.pivot_table(index=["comparator", "mask_fraction"], columns="seed", values="rel [95% CI] %", aggfunc="first").reset_index()
    L += [md_table(piv), "", "### All fractions: across-seed mean relative delta (%), MSE and MAE\n"]
    a = across.copy()
    a["mean_rel_pct"], a["min_rel_pct"], a["max_rel_pct"] = 100 * a.mean_relative, 100 * a.min_relative, 100 * a.max_relative
    L += [md_table(a[["comparator", "mask_fraction", "metric", "mean_rel_pct", "min_rel_pct", "max_rel_pct", "k_candidate_better",
                      "k_ci_excludes0_candidate_better", "k_ci_excludes0_comparator_better"]].sort_values(["comparator", "metric", "mask_fraction"]),
                   {"mean_rel_pct": "{:+.2f}", "min_rel_pct": "{:+.2f}", "max_rel_pct": "{:+.2f}"}), "",
          "### Sign consistency (15 cells = 3 seeds x 5 fractions per comparator and metric)\n", md_table(signs), "",
          "Descriptive reading: see the section 'Descriptive summary' below; the counts above are the evidence.\n"]
    if ref_f is not None:
        L += ["### Functional (`functional_annotations_pca`) as reference: other arms vs functional, across seeds\n",
              "Same bootstrap, reference = functional; positive delta = functional better. CSV: `analysis_ref_functional/`.\n",
              md_table(ref_f[ref_f.mask_fraction == 0.5][["comparator", "metric", "mean_delta", "mean_relative", "min_relative", "max_relative",
                                                          "k_candidate_better", "k_ci_excludes0_candidate_better",
                                                          "k_ci_excludes0_comparator_better"]],
                       {"mean_delta": "{:.2e}", "mean_relative": "{:+.4f}", "min_relative": "{:+.4f}", "max_relative": "{:+.4f}"}),
              "\n(`k_candidate_*` columns here count seeds in which the functional reference is better.)\n"]
    L += ["## 5. Curve behaviour (validation MSE @ 0.50 per epoch, from `history.json`)\n",
          "Slope = OLS slope over the window as % of the window-mean MSE per epoch (negative = still improving). Noise = SD of detrended last-20 MSE. "
          f"Figure: `{png_rel}`.\n",
          md_table(curves[["arm", "seed", "best_epoch", "best_in_last10", "mse_ep60", "mse_ep90", "mse_ep110", "final_mse", "slope_last10_pct_per_epoch",
                           "slope_last20_pct_per_epoch", "noise_sd_detrended_last20", "noise_sd_pct_of_mse", "nonfinite_losses"]],
                   {"mse_ep60": "{:.5f}", "mse_ep90": "{:.5f}", "mse_ep110": "{:.5f}", "final_mse": "{:.5f}", "slope_last10_pct_per_epoch": "{:+.3f}",
                    "slope_last20_pct_per_epoch": "{:+.3f}", "noise_sd_detrended_last20": "{:.2e}", "noise_sd_pct_of_mse": "{:.3f}"}), "",
          "### Arm gap from last-10-epoch averages (sensitivity to the best-epoch pick), comparator - candidate\n",
          md_table(gap10, {"delta_last10": "{:+.2e}", "rel_pct_last10": "{:+.2f}"}), ""]
    if zero is not None:
        L += ["## 6. Zero-embedding rows (candidate store: all-zero rows) at 0.50\n",
              "Loci whose candidate embedding row is all-zero, applied to every arm's predictions (same loci/pairs for all arms). Descriptive only.\n",
              md_table(zero, {"share_pairs_zero_pct": "{:.3f}", "mse_zero_rows": "{:.5f}", "mse_nonzero_rows": "{:.5f}", "prior_mse_zero_rows": "{:.5f}",
                              "prior_mse_nonzero_rows": "{:.5f}"}), ""]
    L += ["## 7. Resources and checkpoints\n", md_table(per_run[["arm", "seed", "dim", "wall_clock_h", "best_pt_sha256"]], {"wall_clock_h": "{:.2f}"}),
          "\nWall-clock is per run, but the three seed orchestrators shared the GPU/CPU concurrently, so wall times are not clean per-arm cost "
          "measurements. Full best.pt paths: `analysis/per_run_table.csv`.\n",
          "Adapter parameters (`LayerNorm(d) -> Linear(d,256) -> GELU -> LayerNorm(256)`): " + ", ".join(
              f"{d}D = {adapter_params(d):,}" for d in sorted(set(DIMS.values()))) + ".\n",
          "## 8. Caveats (from `docs/REGULATORY_CONFIRMATION_FAIRNESS.md`)\n",
          "- Native dimensions differ (candidate 256, functional 256, CpGPT-large 512, DeepCpG 128); adapter capacity grows with dimension; no dimension "
          "matching in this result.\n- Fit scope: candidate SVD fit on chr1-19 loci only; functional SVD fit on all 408,399 loci (transductive); "
          "CpGPT/DeepCpG external.\n- dtype: candidate float32, others float16.\n- The candidate has 849 all-zero rows (no track overlap).\n"
          "- Functional includes dense genomic-context inputs (part of the functional input, not independent validation); the candidate has none.\n"
          "- CpGPT and DeepCpG encoders were pretrained on methylation (external); the candidate was not.\n"
          "- Seeds change only the reconstructor init (masks/panels identical); n = 3; validation patients are the same for all arms and were used "
          "for checkpoint selection (best epoch), so these numbers are not an untouched test estimate.\n- Modern sequence FMs are not part of Phase A.\n"]
    return "\n".join(L)


def descriptive(signs: pd.DataFrame, across: pd.DataFrame) -> str:
    lines = ["## Descriptive summary (no decision rule applied)\n"]
    for c in signs.comparator.unique():
        parts = []
        for _, r in signs[signs.comparator == c].iterrows():
            d = "candidate" if r.reference_better > r.cells / 2 else "comparator"
            parts.append(f"{r.metric.upper()}: candidate lower-error in {r.reference_better}/{r.cells} seed x fraction cells "
                         f"(CI excludes 0 favouring candidate {r.ci_excludes0_reference_better}, favouring comparator "
                         f"{r.ci_excludes0_comparator_better}); majority = {d}")
        lines.append(f"- `{c}`: " + "; ".join(parts) + ".")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=ROOT / "outputs/regulatory_confirmation_v1")
    ap.add_argument("--analysis-dir", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=ROOT / "docs/REGULATORY_CONFIRMATION_PHASE_A_RESULTS.md")
    ap.add_argument("--no-zero-rows", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args(argv)
    ad = a.analysis_dir or a.runs_root / "analysis"
    ad.mkdir(parents=True, exist_ok=True)
    runs = load_all(a.runs_root)
    png = ad / "validation_mse_vs_epoch.png"
    if a.plot_only:
        plot_curves(runs, png)
        print(png)
        return 0
    per_run, per_frac = run_tables(runs)
    per_run.to_csv(ad / "per_run_table.csv", index=False)
    per_frac.to_csv(ad / "per_run_all_fractions.csv", index=False)
    summ50 = seed_summary(per_run, ["best_epoch", "mse", "mae", "mas_pcc", "mac_pcc", "skill_vs_prior"])
    sf = seed_summary(per_frac.assign(arm_f=per_frac.arm + "@" + per_frac.mask_fraction.map("{:.2f}".format)).drop(columns="arm").rename(
        columns={"arm_f": "arm"}), ["mse", "mae", "mas_pcc", "mac_pcc", "skill_vs_prior"])
    sf[["arm", "mask_fraction"]] = sf.arm.str.split("@", expand=True)
    sf["mse"] = [f"{m:.6f} +/- {s:.6f}" for m, s in zip(sf.mse_mean, sf.mse_sd)]
    sf["mae"] = [f"{m:.6f} +/- {s:.6f}" for m, s in zip(sf.mae_mean, sf.mae_sd)]
    sf["mas_pcc"] = [f"{m:.4f} +/- {s:.4f}" for m, s in zip(sf.mas_pcc_mean, sf.mas_pcc_sd)]
    sf["mac_pcc"] = [f"{m:.4f} +/- {s:.4f}" for m, s in zip(sf.mac_pcc_mean, sf.mac_pcc_sd)]
    sf["skill"] = [f"{m:.4f} +/- {s:.4f}" for m, s in zip(sf.skill_vs_prior_mean, sf.skill_vs_prior_sd)]
    summ_frac = sf[["arm", "mask_fraction", "mse", "mae", "mas_pcc", "mac_pcc", "skill"]]
    summ50.to_csv(ad / "seed_summary_0.50.csv", index=False)
    sf.drop(columns=["mse", "mae", "mas_pcc", "mac_pcc", "skill"]).to_csv(ad / "seed_summary_all_fractions.csv", index=False)
    curves = curve_table(runs)
    curves.to_csv(ad / "curve_behaviour.csv", index=False)
    gap10 = last10_gap(curves)
    gap10.to_csv(ad / "last10_gap.csv", index=False)
    contrasts, across = pd.read_csv(ad / "paired_contrasts.csv"), pd.read_csv(ad / "across_seeds_summary.csv")
    signs = sign_counts(contrasts)
    signs.to_csv(ad / "sign_counts.csv", index=False)
    rf = ad.parent / "analysis_ref_functional" / "across_seeds_summary.csv"
    ref_f = pd.read_csv(rf) if rf.is_file() else None
    zero = None
    if not a.no_zero_rows:
        reg = pd.read_parquet(ROOT / "outputs/encode_atlas_v1/loci.parquet")
        st = yaml.safe_load((runs[(CANDIDATE, 17)]["dir"] / "resolved_config.yaml").read_text())["representation"]["store_h5"]
        zero = zero_row_table(runs, reg, ROOT / st)
        zero.to_csv(ad / "zero_rows.csv", index=False)
    integrity = json.loads((ad / "integrity.json").read_text())
    try:
        plot_curves(runs, png)
    except ImportError:
        print("matplotlib missing: run with --plot-only in an env that has it")
    md = build_markdown(per_run, per_frac, summ50, summ_frac, contrasts, across, signs, curves, gap10, zero, ref_f, integrity,
                        f"outputs/regulatory_confirmation_v1/analysis/{png.name}")
    md += "\n" + descriptive(signs, across)
    a.md.write_text(md)
    print(f"wrote {a.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
