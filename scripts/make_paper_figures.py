#!/usr/bin/env python3
"""Draft main-text figures (4-figure story), read from frozen CSVs under outputs/. No experiment is run.

Fig1 construction + feature-family selection | Fig2 biology (FANTOM5/Micro-C/RT) | Fig3 EWAS, all 3 arms | Fig4 reconstruction
Writes paper/figures/figN_*.{pdf,png}. Early draft: styling is shared via the constants below.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import FancyBboxPatch

O = Path("outputs")
HD, CP, DC = "regulatory_histone_dnase", "cpgpt_large_locus", "deepcpg_dna_locus"
# order and colours are shared by every figure (Okabe-Ito)
ARMS = {HD: ("Histone+DNase", "#0072B2"), CP: ("CpGPT-large", "#D55E00"), DC: ("DeepCpG-DNA", "#E69F00")}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "axes.titlesize": 9, "axes.titleweight": "bold"})


def _save(fig, out, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _panel(ax, letter):
    ax.text(-0.02, 1.08, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


# ---------------------------------------------------------------- Fig 1
def fig1(out):
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.45, 1], hspace=0.5, wspace=0.9)
    # A: schematic
    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); _panel(ax, "A")
    ax.set(xlim=(0, 100), ylim=(0, 30))
    boxes = ["CpG locus", "1,959 Histone +\n533 DNase tracks", "Binary\noverlap vector", "Global\nTruncatedSVD", "256-D\nembedding"]
    xs = np.linspace(1, 81, 5)
    for x, t in zip(xs, boxes):
        ax.add_patch(FancyBboxPatch((x, 13), 17, 12, boxstyle="round,pad=0.4", fc="#E8F1F8" if t != boxes[-1] else "#0072B2", ec="#0072B2"))
        ax.text(x + 8.5, 19, t, ha="center", va="center", fontsize=7.5, color="white" if t == boxes[-1] else "black")
    for x in xs[:-1]:
        ax.annotate("", (x + 19.2, 19), (x + 17.8, 19), arrowprops=dict(arrowstyle="->", color="0.3"))
    for x, t in zip((12, 40, 68), ("patient-independent", "no methylation input", "no phenotype labels")):
        ax.text(x + 8, 6, t, ha="center", fontsize=6.5, style="italic", color="0.35",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75"))
    # B: % MSE reduction vs context-only (50% masking, mean of 3 seeds)
    d = pd.read_csv(O / "encode_atlas_v1/reconstruction_metrics.csv")
    d = d[(d.stage == "confirm") & (d.view == "seen") & ~d.ood & ~d.pilot & (d.mask_fraction == 0.5)]
    mse = d.groupby(["arm", "seed"]).mse.mean().unstack()
    rows = [("add/target/H3K4me3", "+ H3K4me3"), ("add/assay/Histone ChIP-seq", "Histone"),
            ("full", "Full functional"), ("drop/assay/Histone ChIP-seq", "Full − Histone")]
    red = {a: 100 * (1 - mse.loc[a] / mse.loc["context_only"]) for a, _ in rows}
    ax = fig.add_subplot(gs[1, 0]); _panel(ax, "B")
    ax.barh(range(len(rows)), [red[a].mean() for a, _ in rows], color="#0072B2", height=.6)
    for i, (a, _) in enumerate(rows):
        ax.plot(red[a], [i] * len(red[a]), "o", ms=2.5, color="0.2")
        ax.text(red[a].max() + .9, i, f"{red[a].mean():.1f}%", va="center", fontsize=8)
    ax.axvline(0, color="0.3", lw=.8)
    ax.set(yticks=range(len(rows)), yticklabels=[r[1] for r in rows], xlim=(0, 27),
           xlabel="MSE reduction vs Context only (%)\n(50% masking; dots = seeds)", title="Which information matters?")
    ax.invert_yaxis()
    # C: paired relative dMSE
    p = pd.read_csv(O / "regulatory_confirm_v1/analysis/paired_contrasts.csv")
    p = p[p.metric == "mse"]
    steps = [("regulatory_histone", "regulatory_histone_dnase", "Histone →\nHistone+DNase", "+533 DNase tracks"),
             ("regulatory_histone_dnase", "regulatory_clean", "Histone+DNase →\n+ TF", "+1,673 TF tracks")]
    ax = fig.add_subplot(gs[1, 1]); _panel(ax, "C")
    for i, (a, b, lab, note) in enumerate(steps):
        v = -100 * p[(p.reference == a) & (p.alternative == b)].groupby("seed").relative.mean()
        ax.bar(i, v.mean(), color="#0072B2", width=.55)
        ax.plot([i] * len(v), v, "o", ms=2.5, color="0.2")
        ax.text(i, v.max() + .08, f"{v.mean():.2f}%", ha="center", fontsize=8)
        ax.text(i, -.28, note, ha="center", fontsize=7, style="italic", color="0.35")
    ax.set(xticks=range(2), xticklabels=[s[2] for s in steps], ylim=(-.4, 2.3), ylabel="MSE reduction (%)\n(dots = seeds)",
           title="Why Histone+DNase?")
    ax.axhline(0, color="0.3", lw=.8)
    _save(fig, out, "fig1_construction")


# ---------------------------------------------------------------- Fig 2
def _legend(fig):
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c in ARMS.values()], [n for n, _ in ARMS.values()],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.04))


def fig2(out):
    d = pd.read_csv(O / "biological_validation_v2/bioval-v2-protocol-freeze-v1/report/summary_primary.csv")
    d["arm"] = d.arm.replace({"regulatory_histone_dnase_v1": HD})
    # (endpoint, title, ylabel, reference line, axis floor)
    panels = [("fantom5_membership", "Enhancer identity", "AUROC (dashed = chance)", 0.5, 0.5),
              ("microc_H1_intra10kb", "3D chromatin contacts", "AUROC (dashed = chance)", 0.5, 0.5),
              ("rt_consensus", "Replication timing", "Out-of-fold Spearman ρ", 0.0, 0.0)]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7))
    for ax, (ep, title, yl, ref, floor), L in zip(axes, panels, "ABC"):
        g = d[d.endpoint == ep].set_index("arm")
        for i, (arm, (_, c)) in enumerate(ARMS.items()):
            r = g.loc[arm]
            ax.bar(i, r.value - floor, bottom=floor, color=c, width=.65, yerr=[[r.value - r.ci_lo], [r.ci_hi - r.value]], capsize=2)
            ax.text(i, r.ci_hi, f"{r.value:.3f}", ha="center", va="bottom", fontsize=7)
        ax.axhline(ref, color="0.3", ls="--", lw=.8)
        ax.set(xticks=[], ylabel=yl, title=title, ylim=(floor, g.ci_hi.max() * 1.08))
        _panel(ax, L)
    _legend(fig)
    _save(fig, out, "fig2_biology")


# ---------------------------------------------------------------- Fig 3
# Exact prospective EWAS-v3 panel; no filtering/ordering by representation performance.
EWAS_LABELS = {
    "melanoma": "Melanoma vs nevi", "oral_pharyngeal_scc": "Oral/pharyngeal SCC",
    "breast_cancer": "Breast cancer (prevalent)",
    "colorectal_cancer": "Colorectal cancer (prevalent)",
    "lung_cancer": "Lung cancer (incident)", "ibd_family": "Crohn's disease",
    "rheumatoid_arthritis": "Rheumatoid arthritis", "psoriasis": "Psoriasis",
    "graves_disease": "Graves' disease", "adiposity_bmi": "BMI",
    "glycemic_t2d": "Type 2 diabetes (incident)", "lipid_hdl": "HDL cholesterol",
    "cardiovascular_ihd": "Ischemic heart disease (incident)",
}


def fig3(out):
    root = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load((root / "configs/ewas_v3/main_trait_panel.yaml").read_text())
    a = pd.read_csv(root / "data/registry/ewas_v3_results/main_results.csv")
    a["arm"] = a.representation.replace({"regulatory_histone_dnase_v1": HD})
    order = [t["task"] for t in manifest["main_tasks"]]
    assert set(a.task) == set(order) == set(EWAS_LABELS)
    assert len(a) == len(order) * len(ARMS) and set(a.arm) == set(ARMS)
    assert (a.status == "complete").all() and not a.duplicated(["task", "arm"]).any()
    for task in manifest["main_tasks"]:
        assert set(a.loc[a.task == task["task"], "source_analysis_id"]) == {task["source_analysis_id"]}
    flags = pd.read_csv(root / "data/registry/ewas_v3/primary_overlap_flags.csv")
    correlated = set(flags.trait_a) | set(flags.trait_b)
    families = ["Cancer", "Immune/inflammatory", "Cardiometabolic"]
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.2), gridspec_kw={"height_ratios": [5, 4, 4]})
    h = .26
    for ax, family, letter in zip(axes, families, "ABC"):
        tasks = [t["task"] for t in manifest["main_tasks"] if t["category"] == family]
        for k, (arm, (lab, color)) in enumerate(ARMS.items()):
            g = a[(a.arm == arm) & (a.category == family)].set_index("task").loc[tasks]
            y = np.arange(len(tasks)) + (k - 1) * h
            ax.barh(y, g.AUROC - .5, left=.5, height=h * .92, color=color, label=lab,
                    xerr=[g.AUROC - g.ci_lo, g.ci_hi - g.AUROC], error_kw=dict(lw=.8, capsize=1.5))
        ax.axvline(.5, color="0.3", ls="--", lw=.8)
        ax.set(yticks=range(len(tasks)),
               yticklabels=[EWAS_LABELS[t] + ("†" if t in correlated else "") for t in tasks],
               xlim=(.4, 1), title=family)
        ax.text(.5, -.62, "chance", fontsize=6.5, ha="center", color="0.3")
        ax.invert_yaxis()
        _panel(ax, letter)
    axes[-1].set_xlabel("Pooled out-of-fold AUROC (95% CI)")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c in ARMS.values()],
               [name for name, _ in ARMS.values()], loc="lower center", ncol=3,
               frameon=False, fontsize=8, bbox_to_anchor=(.5, .062))
    fig.text(.02, .036, "EWAS Catalog-associated CpGs: p < 10⁻⁴; fixed 408,399-CpG universe.",
             fontsize=6.5, color="0.35")
    fig.text(.02, .016, "† >50% overlap with another selected task; shared cohorts are not independent replications.",
             fontsize=6.5, color="0.35")
    fig.tight_layout(rect=(0, .12, 1, .985), h_pad=2.0)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig3_ewas.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4
def fig4(out):
    src = {"TCGA": ("TCGA (held-out patients)", O / "regulatory_confirmation_v1/analysis/per_run_all_fractions.csv"),
           "GSE40279": ("GSE40279 (independent cohort)", O / "external_reconstruction_v1/analysis/A_v1.2/test/per_run_metrics.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.9))
    for ax, (title, path), L in zip(axes, src.values(), "AB"):
        d = pd.read_csv(path)
        for arm, (lab, c) in ARMS.items():
            g = d[d.arm == arm]
            m = g.groupby("mask_fraction").mse
            ax.fill_between(m.min().index * 100, m.min(), m.max(), color=c, alpha=.2, lw=0)
            ax.plot(m.mean().index * 100, m.mean(), "-o", color=c, ms=3)
            ax.plot(g.mask_fraction * 100, g.mse, ".", color=c, alpha=.4, ms=3)
        ax.set(xlabel="Masking fraction (%)", ylabel="Masked-CpG MSE", title=title)
        _panel(ax, L)
    _legend(fig)
    _save(fig, out, "fig4_reconstruction")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    ap.add_argument("--figure", choices=["1", "2", "3", "4", "all"], default="all")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    functions = {"1": fig1, "2": fig2, "3": fig3, "4": fig4}
    for f in (functions.values() if args.figure == "all" else [functions[args.figure]]):
        f(out)
    print(f"wrote figures to {out}")
