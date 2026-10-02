#!/usr/bin/env python3
"""Aggregate 450K->EPIC upscaling runs into tidy CSVs, markdown tables and figures.

    python scripts/summarize_upscaling.py --outputs outputs --results outputs/upscale_450k_epic/_summary

Writes under --results:
  runs_long.csv        one row per (run, eval set, locus view) with every metric
  table_<view>.md      mean +- sd over seeds, per arm and eval set (headline metrics)
  fig_skill_real.png   skill vs. the train-mean prior on real pairs / simulated test, per arm
  fig_scatter_real.png predicted vs. measured EPIC beta on the real pairs, best neural arm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cpg_repr_benchmark.viz.style import CATEGORICAL_PALETTE, apply_paper_style, savefig

ARM_ORDER = ["baseline_prior_mean", "baseline_ridge_pca", "control_constant", "control_random_stable", "functional_annotations_pca_450k_fit"]
ARM_LABEL = {
    "baseline_prior_mean": "Train-mean prior",
    "baseline_ridge_pca": "Linear PCA-ridge",
    "control_constant": "Constant locus code",
    "control_random_stable": "Random locus code",
    "functional_annotations_pca_450k_fit": "Functional PCA",
}
SETS = ["sim_test", "real_GSE86833_blood", "real_GSE86833_cells", "real_GSE92580_FF", "real_primary"]
SET_LABEL = {
    "sim_test": "Simulated\n(held-out EPIC studies)",
    "real_GSE86833_blood": "Real pairs:\nblood (n=5)",
    "real_GSE86833_cells": "Real pairs:\ncell lines/fibroblasts (n=10)",
    "real_GSE92580_FF": "Real pairs:\nbrain tumour FF (n=9)",
    "real_primary": "Real pairs:\nall primary (n=24)",
}
METRICS = ["mse", "skill_vs_prior", "sample_pearson", "locus_pearson_variable", "skill_vs_prior_variable", "mae"]


def load(outputs: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(outputs.glob("upscale_450k_epic/*/*/*/seed_*/*/summary.json")):
        s = json.loads(path.read_text())
        for eset, views in s["evaluation"].items():
            for view, entry in views.items():
                rows.append({"run_dir": str(path.parent), "arm": s["representation"], "seed": s["seed"], "eval_set": eset,
                             "view": view, "n_loci": entry["n_loci"], **entry["summary"]})
    return pd.DataFrame(rows)


def tables(df: pd.DataFrame, out: Path) -> None:
    for view, sub in df.groupby("view"):
        lines = [f"# {view}: mean ± sd over seeds (n runs per arm in brackets)\n"]
        for metric in METRICS:
            lines.append(f"\n## {metric}\n")
            header = "| arm | " + " | ".join(s for s in SETS if s in set(sub.eval_set)) + " |"
            lines += [header, "|" + "---|" * (header.count("|") - 1)]
            for arm in ARM_ORDER:
                a = sub[sub.arm == arm]
                if a.empty:
                    continue
                cells = []
                for s in SETS:
                    if s not in set(sub.eval_set):
                        continue
                    v = a[a.eval_set == s][metric].dropna()
                    cells.append("-" if v.empty else f"{v.mean():.4f}" + (f" ± {v.std(ddof=1):.4f}" if len(v) > 1 else ""))
                lines.append(f"| {ARM_LABEL[arm]} [{a.seed.nunique()}] | " + " | ".join(cells) + " |")
        (out / f"table_{view}.md").write_text("\n".join(lines) + "\n")


def fig_skill(df: pd.DataFrame, out: Path) -> None:
    apply_paper_style()
    sub = df[(df.view == "seen_loci") & df.eval_set.isin(SETS) & df.arm.isin(ARM_ORDER)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    width = 0.8 / len(ARM_ORDER)
    for ax, metric, title in zip(axes, ["skill_vs_prior", "skill_vs_prior_variable"],
                                 ["All EPIC-only loci", "Most variable decile of loci"]):
        for i, arm in enumerate(ARM_ORDER):
            for j, s in enumerate(SETS):
                v = sub[(sub.arm == arm) & (sub.eval_set == s)][metric].dropna()
                if v.empty:
                    continue
                x = j + (i - (len(ARM_ORDER) - 1) / 2) * width
                ax.bar(x, v.mean(), width * 0.9, color=CATEGORICAL_PALETTE[i], label=ARM_LABEL[arm] if j == 0 else None)
                if len(v) > 1:
                    ax.errorbar(x, v.mean(), v.std(ddof=1), color="#333", lw=0.8, capsize=1.5)
        ax.axhline(0, color="#888", lw=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(len(SETS)))
        ax.set_xticklabels([SET_LABEL[s] for s in SETS], fontsize=6.5)
    axes[0].set_ylabel("Skill vs. train-mean prior\n(1 - MSE / prior MSE)")
    axes[0].legend(fontsize=6.5, ncol=1, frameon=False)
    savefig(fig, out / "fig_skill_real.png")


def fig_scatter(df: pd.DataFrame, out: Path) -> None:
    sub = df[(df.arm == "functional_annotations_pca_450k_fit") & (df.view == "seen_loci") & (df.eval_set == "real_primary")]
    if sub.empty:
        return
    run = Path(sub.sort_values("mse").iloc[0]["run_dir"])
    protocol = np.load(run / "protocol_snapshot.npz")["target_train_mask"]
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharex=True, sharey=True)
    for ax, name, title in zip(axes, ["real_GSE86833_blood", "real_GSE86833_cells", "real_GSE92580_FF"],
                               ["Blood", "Cell lines / fibroblasts", "Brain tumour (FF)"]):
        p = run / "evaluation" / name / "predictions.npz"
        if not p.exists():
            continue
        d = np.load(p)
        pred, true = d["prediction"][:, protocol].astype(np.float32), d["target"][:, protocol].astype(np.float32)
        m = np.isfinite(true)
        idx = np.random.default_rng(0).choice(int(m.sum()), size=min(200_000, int(m.sum())), replace=False)
        ax.hexbin(true[m][idx], pred[m][idx], gridsize=60, bins="log", cmap="Blues", mincnt=1)
        ax.plot([0, 1], [0, 1], color="#888", lw=0.6)
        ax.set_title(f"{title}  (r={np.corrcoef(true[m][idx], pred[m][idx])[0, 1]:.2f})", fontsize=8)
        ax.set_xlabel("Measured EPIC β")
    axes[0].set_ylabel("Reconstructed from 450K β")
    savefig(fig, out / "fig_scatter_real.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", type=Path, default=Path("outputs"))
    p.add_argument("--results", type=Path, default=Path("outputs/upscale_450k_epic/_summary"))
    args = p.parse_args()
    df = load(args.outputs)
    args.results.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.results / "runs_long.csv", index=False)
    tables(df, args.results)
    fig_skill(df, args.results)
    fig_scatter(df, args.results)
    print(f"{len(df)} rows from {df.run_dir.nunique()} runs -> {args.results}")


if __name__ == "__main__":
    main()
