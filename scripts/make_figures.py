#!/usr/bin/env python3
"""Final figures, read from frozen result CSVs under outputs/. No experiment is run.

fig1 TCGA masking reconstruction | fig2 external GSE40279 (test) | fig3 biological validation v2 | fig4 EWAS v2 vs CpGPT
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

O = Path("outputs")
SRC = {
    "tcga": O / "regulatory_confirmation_v1/analysis/per_run_all_fractions.csv",
    "ext": O / "external_reconstruction_v1/analysis/A_v1.2/test/per_run_metrics.csv",
    "bio": O / "biological_validation_v2/bioval-v2-protocol-freeze-v1/report/summary_primary.csv",
    "ewas": O / "ewas_v2/ewas-v2-protocol-freeze-v1/family_summary.csv",
}
# Okabe-Ito (colour-blind safe); candidate first.
ARMS = {
    "regulatory_histone_dnase": ("Histone+DNase (ours)", "#0072B2"),
    "regulatory_histone_dnase_v1": ("Histone+DNase (ours)", "#0072B2"),
    "functional_annotations_pca": ("Functional PCA", "#009E73"),
    "cpgpt_large_locus": ("CpGPT-large", "#D55E00"),
    "deepcpg_dna_locus": ("DeepCpG", "#E69F00"),
}
ENDPOINTS = {"loyfer_profile": "Tissue profile\nsimilarity", "microc_H1_intra10kb": "Micro-C H1\n(<10 kb)",
             "fantom5_membership": "FANTOM5\nmembership", "rt_consensus": "Replication\ntiming"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def _mse_vs_mask(ax, path, title):
    d = pd.read_csv(path)
    d = d.groupby(["arm", "mask_fraction"]).mse.agg(["mean", "std"]).reset_index()
    for arm, g in d.groupby("arm"):
        label, c = ARMS[arm]
        ax.errorbar(g.mask_fraction, g["mean"], g["std"], label=label, color=c, marker="o", ms=3, capsize=2)
    ax.set(xlabel="Mask fraction", ylabel="Masked-CpG MSE (mean ± sd, 3 seeds)", title=title)
    ax.legend(frameon=False)


def fig1_2(out):
    for key, name, title in (("tcga", "fig1_tcga_masking", "TCGA (held-out patients)"),
                             ("ext", "fig2_external_gse40279", "GSE40279 external test")):
        fig, ax = plt.subplots(figsize=(4, 3.2))
        _mse_vs_mask(ax, SRC[key], title)
        _save(fig, out, name)


def fig3(out):
    d = pd.read_csv(SRC["bio"])
    fig, axes = plt.subplots(1, len(ENDPOINTS), figsize=(9, 3), sharey=False)
    for ax, (ep, lab) in zip(axes, ENDPOINTS.items()):
        g = d[d.endpoint == ep].set_index("arm")
        for i, (arm, (name, c)) in enumerate(a for a in ARMS.items() if a[0] != "regulatory_histone_dnase"):
            if arm not in g.index:
                continue
            r = g.loc[arm]
            ax.errorbar(i, r.value, [[r.value - r.ci_lo], [r.ci_hi - r.value]], fmt="o", color=c, capsize=2)
        ax.axhline(0, color="0.7", lw=.6)
        ax.set(title=lab, xticks=[])
    axes[0].set_ylabel("Endpoint value (95% CI)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=n) for k, (n, c) in ARMS.items() if k != "regulatory_histone_dnase"]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.02))
    fig.tight_layout(rect=(0, .1, 1, 1))
    _save(fig, out, "fig3_biological_validation")


def fig4(out):
    d = pd.read_csv(SRC["ewas"])
    d = d[(d.design == "main") & (d.contrast == "primary_vs_cpgpt") & (d.metric == "auroc") & (d.scope == "independent")]
    d = d.groupby("family")[["n_candidate_better", "n_spans_zero", "n_candidate_worse"]].sum()
    fig, ax = plt.subplots(figsize=(4, 2.6))
    left = 0
    for col, lab, c in (("n_candidate_better", "Ours better", "#0072B2"), ("n_spans_zero", "CI spans 0", "0.75"),
                        ("n_candidate_worse", "CpGPT better", "#D55E00")):
        ax.barh(d.index, d[col], left=left, color=c, label=lab)
        left = left + d[col]
    ax.set(xlabel="EWAS tasks (AUROC, independent scope)", title="EWAS v2 vs CpGPT-large")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _save(fig, out, "fig4_ewas_v2")


def _save(fig, out, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=O / "figures_final")
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    fig1_2(out)
    fig3(out)
    fig4(out)
    print(f"wrote figures to {out}")
