#!/usr/bin/env python3
"""Generate paper-style figures from current outputs/masking and outputs/bio_validation runs.

Results are still partial (few representations/seeds) - figures show exactly what
has run so far, with no synthetic error bars or extrapolation.

Masking is genomewide-only: only datasets with "genomewide" in their name are
plotted, and there is no seen/unseen_locus split (unseen_locus is not part of
the genomewide masking task).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cpg_repr_benchmark.viz.aggregate import load_bio_validation_df, load_masking_df
from cpg_repr_benchmark.viz.style import (
    KNOWN_SET_CATEGORIES,
    KNOWN_SET_CATEGORY_LABELS,
    apply_paper_style,
    display_name,
    pretty_label,
    representation_colors,
    savefig,
)

# Temporarily excluded from all figures - remove this entry to bring it back.
EXCLUDED_REPRESENTATIONS = {"methylgpt_locus"}


def plot_reconstruction_vs_mask(df: pd.DataFrame, out_dir: Path) -> None:
    genomewide = df[df["dataset"].fillna("").str.contains("genomewide", case=False)]
    if genomewide.empty:
        print("skip reconstruction_vs_mask: no genomewide masking runs found")
        return

    for dataset, sub in genomewide.groupby("dataset"):
        reps = sorted(sub["representation"].dropna().unique())
        colors = representation_colors(reps)

        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        for rep in reps:
            rep_df = sub[sub["representation"] == rep].sort_values("mask_fraction")
            ax.plot(
                rep_df["mask_fraction"],
                rep_df["mse"],
                marker="o",
                color=colors.get(rep, "#898781"),
                label=display_name(rep),
            )
        ax.set_xlabel("mask fraction")
        ax.set_ylabel("MSE")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        savefig(fig, out_dir / f"reconstruction_vs_mask__{dataset}")
        plt.close(fig)


def plot_bio_probes_auc(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[(df["group"] == "genomic_context") & (df["metric"] == "auc")]
    if sub.empty:
        print("skip bio_probes_auc: no genomic_context AUC found")
        return
    classes = sorted(sub["name"].unique())
    reps = sorted(sub["representation"].dropna().unique())
    colors = representation_colors(reps)

    n_reps = len(reps)
    width = 0.8 / max(n_reps, 1)
    x = range(len(classes))

    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    for i, rep in enumerate(reps):
        rep_df = sub[sub["representation"] == rep].set_index("name")
        values = [rep_df["value"].get(cls, float("nan")) for cls in classes]
        offsets = [xi + (i - (n_reps - 1) / 2) * width for xi in x]
        ax.bar(offsets, values, width=width, color=colors.get(rep, "#898781"), label=display_name(rep))

    ax.axhline(0.5, color="#c3c2b7", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels([pretty_label(cls) for cls in classes])
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    savefig(fig, out_dir / "bio_probes_auc")
    plt.close(fig)


def plot_bio_known_sets_auc(df: pd.DataFrame, out_dir: Path) -> None:
    all_sub = df[(df["group"] == "known_cpg_sets") & (df["metric"] == "auc")].assign(
        value=lambda d: pd.to_numeric(d["value"], errors="coerce")
    )
    if all_sub.empty:
        print("skip bio_known_sets_*: no known_cpg_sets AUC found")
        return

    for category, set_names in KNOWN_SET_CATEGORIES.items():
        sub = all_sub[all_sub["name"].isin(set_names)]
        if sub.empty:
            continue
        sets_ = [s for s in set_names if s in sub["name"].unique()]
        reps = sorted(sub["representation"].dropna().unique())
        colors = representation_colors(reps)

        n_reps = len(reps)
        width = 0.8 / max(n_reps, 1)
        x = range(len(sets_))

        fig, ax = plt.subplots(figsize=(max(3.6, 0.9 * len(sets_) + 1.4), 2.8))
        for i, rep in enumerate(reps):
            rep_df = sub[sub["representation"] == rep].set_index("name")
            values = [rep_df["value"].get(s, float("nan")) for s in sets_]
            offsets = [xi + (i - (n_reps - 1) / 2) * width for xi in x]
            ax.bar(offsets, values, width=width, color=colors.get(rep, "#898781"), label=display_name(rep))

        ax.axhline(0.5, color="#c3c2b7", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_xticks(list(x))
        ax.set_xticklabels([pretty_label(s) for s in sets_], rotation=30, ha="right")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.4, 1.0)
        ax.set_title(KNOWN_SET_CATEGORY_LABELS.get(category, category), loc="left", fontweight="normal")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        savefig(fig, out_dir / f"bio_known_sets_{category}")
        plt.close(fig)


def plot_bio_clocks_pearson(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[(df["group"] == "clock_coefficients") & (df["metric"] == "pearson")]
    if sub.empty:
        print("skip bio_clocks_pearson: no clock_coefficients pearson found")
        return
    clocks = sorted(sub["name"].unique())
    reps = sorted(sub["representation"].dropna().unique())
    colors = representation_colors(reps)

    n_reps = len(reps)
    width = 0.8 / max(n_reps, 1)
    x = range(len(clocks))

    fig, ax = plt.subplots(figsize=(max(3.6, 0.9 * len(clocks) + 1.4), 2.8))
    for i, rep in enumerate(reps):
        rep_df = sub[sub["representation"] == rep].set_index("name")
        values = [rep_df["value"].get(clock, float("nan")) for clock in clocks]
        offsets = [xi + (i - (n_reps - 1) / 2) * width for xi in x]
        ax.bar(offsets, values, width=width, color=colors.get(rep, "#898781"), label=display_name(rep))

    ax.axhline(0.0, color="#c3c2b7", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels([pretty_label(clock) for clock in clocks], rotation=20, ha="right")
    ax.set_ylabel("pearson r")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="center left", bbox_to_anchor=(1.0, 0.5))
    savefig(fig, out_dir / "bio_clocks_pearson")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate result figures from outputs/")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    apply_paper_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    masking_df = load_masking_df(args.outputs)
    masking_df = masking_df[~masking_df["representation"].isin(EXCLUDED_REPRESENTATIONS)]
    plot_reconstruction_vs_mask(masking_df, args.out_dir)

    bio_df = load_bio_validation_df(args.outputs)
    bio_df = bio_df[~bio_df["representation"].isin(EXCLUDED_REPRESENTATIONS)]
    plot_bio_probes_auc(bio_df, args.out_dir)
    plot_bio_known_sets_auc(bio_df, args.out_dir)
    plot_bio_clocks_pearson(bio_df, args.out_dir)

    print(f"Figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
