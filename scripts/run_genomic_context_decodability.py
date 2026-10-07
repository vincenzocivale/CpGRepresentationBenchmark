#!/usr/bin/env python3
"""genomic_context_decodability_v1: representation characterization (not independent validation).

  fit      --arm A     chromosome-blocked probes -> pooled OOF predictions (outputs/.../oof/A.npz)
  analyze              bootstrap CIs (B=2000, shared draws) + paired contrasts -> results.json, tables
  figure               main 3-panel bar figure + supplementary confusion matrices / secondary metrics

Example: for a in regulatory_histone_dnase_v1 cpgpt_large_locus deepcpg_dna_locus; do
           python scripts/run_genomic_context_decodability.py fit --arm $a; done
         python scripts/run_genomic_context_decodability.py analyze && ... figure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cpg_repr_benchmark import genomic_context_decodability as gcd
from cpg_repr_benchmark.biological_validation_v2.splits import chromosome_blocked_folds
from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import (
    BlockBootstrap,
    block_index,
    contrast,
    holm_adjust,
    percentile_ci,
)
from cpg_repr_benchmark.bioval_v2_launch.embedding_eval import ArmSpec, EmbeddingStore, load_arm_specs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/genomic_context_decodability_v1"
CHROMS = [f"chr{i}" for i in range(1, 23)]
LABEL = {"regulatory_histone_dnase_v1": "Histone+DNase", "cpgpt_large_locus": "CpGPT-large",
         "deepcpg_dna_locus": "DeepCpG-DNA", "functional_annotations_pca": "Functional PCA (legacy)", "alphagenome_locus": "AlphaGenome (post-hoc)", "regulatory_raw2492": "Raw 2,492 tracks (post-hoc)",
         "regulatory_lpca256": "Logistic PCA-256 (post-hoc)", "regulatory_lpca512": "Logistic PCA-512 (post-hoc)", "regulatory_svd512": "SVD-512 (post-hoc)"}
COLOR = {"regulatory_histone_dnase_v1": "#0072B2", "cpgpt_large_locus": "#D55E00",
         "deepcpg_dna_locus": "#E69F00", "functional_annotations_pca": "#009E73", "alphagenome_locus": "#CC79A7", "regulatory_raw2492": "#56B4E9",
         "regulatory_lpca256": "#F0E442", "regulatory_lpca512": "#999999", "regulatory_svd512": "#000000"}


def folds_for(labels):
    rec = chromosome_blocked_folds(CHROMS, 5, seed=17)  # the frozen bioval-v2 folds
    return labels.chrom.map(rec).to_numpy().astype(int)


def cmd_fit(a):
    spec = load_arm_specs(ROOT, (a.arm,))[a.arm]  # registered matrix arms + post-hoc arms (configs/posthoc/*.yaml)
    store = EmbeddingStore(spec, ROOT, verify_sha=not a.skip_sha)
    labels = gcd.load_labels(ROOT)
    fold = folds_for(labels)
    X = store.load(labels.cpg_idx.to_numpy())  # no per-arm intersection: missing loci raise
    n_zero = int((~X.any(1)).sum())
    print(f"[{a.arm}] n={len(X)} dim={X.shape[1]} zero-vector loci kept={n_zero}", flush=True)
    oof = gcd.fit_arm(X, labels, fold)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "oof").mkdir(exist_ok=True)
    np.savez_compressed(OUT / "oof" / f"{a.arm}.npz", **oof)
    (OUT / "oof" / f"{a.arm}.json").write_text(json.dumps(
        {"identity": store.identity(n_zero), "n_loci": len(X), "n_zero_vector_loci_kept": n_zero,
         "ridge_alphas_grid": gcd.RIDGE_ALPHAS, "ridge_alpha_chosen_per_fold": oof["ridge_alpha"].tolist(),
         "logreg": {"C": gcd.LOGREG_C, "class_weight": "balanced", "max_iter": gcd.LOGREG_MAX_ITER,
                    "n_nonconverged_folds_context_gene": oof["n_nonconverged_folds"].tolist()}},
        indent=2, default=str))
    print(f"[{a.arm}] done")


def arms_available():
    return [x for x in (*gcd.MAIN_ARMS, *gcd.SUPPLEMENTARY_ARMS) if (OUT / "oof" / f"{x}.npz").exists()]


def cmd_analyze(_a):
    labels = gcd.load_labels(ROOT)
    boot = BlockBootstrap(np.unique(labels.chrom.to_numpy()), n_boot=gcd.N_BOOT, seed=gcd.SEED)
    blk = block_index(labels.chrom.to_numpy(), boot.names)
    res = {"protocol": "genomic_context_decodability_v1", "n_loci": len(labels), "n_boot": gcd.N_BOOT,
           "seed": gcd.SEED, "boot_fingerprint": boot.fingerprint(), "arms": {}, "contrasts": {}}
    reps = {}
    for arm in arms_available():
        oof = dict(np.load(OUT / "oof" / f"{arm}.npz"))
        res["arms"][arm] = {}
        for ep, (scorer, stats) in gcd.endpoint_scorers(labels, oof).items():
            body = {}
            for name, fn in stats.items():
                point = fn(np.ones(len(labels)))
                rep = boot.replicates(fn, blk)
                lo, hi = percentile_ci(rep)
                body[name] = {"value": point, "ci_lo": lo, "ci_hi": hi, "class": "primary" if name == gcd.ENDPOINTS[ep][0] else "secondary"}
                reps[f"{arm}|{ep}|{name}"] = rep
            if ep != "tss_distance":
                c = scorer.confusion()
                body["confusion_counts"] = c.tolist()
                body["confusion_row_normalised"] = (c / c.sum(1, keepdims=True)).tolist()
            res["arms"][arm][ep] = body
            print(f"[{arm}] {ep} {gcd.ENDPOINTS[ep][0]} = {body[gcd.ENDPOINTS[ep][0]]['value']:.4f}", flush=True)
    ref = gcd.MAIN_ARMS[0]
    holm_in = {}
    for other in res["arms"]:
        if other == ref or ref not in res["arms"]:
            continue
        for ep, (prim, secs) in gcd.ENDPOINTS.items():
            for st in (prim, *secs):
                a, b = res["arms"][ref][ep][st], res["arms"][other][ep][st]
                c = contrast(a["value"], reps[f"{ref}|{ep}|{st}"], b["value"], reps[f"{other}|{ep}|{st}"])
                key = f"{ref}-{other}|{ep}|{st}"
                c["class"] = a["class"]
                c["supplementary_arm"] = other in gcd.SUPPLEMENTARY_ARMS
                res["contrasts"][key] = c
                if a["class"] == "primary" and other in gcd.MAIN_ARMS:
                    holm_in[key] = c["p"]
    for k, v in holm_adjust(holm_in).items():
        res["contrasts"][k]["p_holm"] = v
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(OUT / "replicates.npz", **reps)
    rows = [{"arm": arm, "endpoint": ep, "stat": st, **{k: v[k] for k in ("value", "ci_lo", "ci_hi", "class")}}
            for arm, d in res["arms"].items() for ep, e in d.items() for st, v in e.items() if isinstance(v, dict) and "value" in v]
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame([{"contrast": k, **v} for k, v in res["contrasts"].items()]).to_csv(OUT / "contrasts.csv", index=False)
    print(f"wrote {OUT}/results.json, summary.csv, contrasts.csv")


def cmd_figure(_a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                         "axes.titlesize": 9, "axes.titleweight": "bold"})  # same style as paper/figures/fig2_biology
    res = json.loads((OUT / "results.json").read_text())
    main = [a for a in (*gcd.MAIN_ARMS, "alphagenome_locus") if a in res["arms"]]
    # (endpoint, title, ylabel, chance/axis floor)
    panels = [("cpg_context", "CpG context", "Macro AUROC (dashed = chance)", 0.5),
              ("gene_region", "Gene region", "Macro AUROC (dashed = chance)", 0.5),
              ("tss_distance", "TSS distance", "Out-of-fold Spearman ρ", 0.0)]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7))
    for ax, (ep, title, yl, floor), letter in zip(axes, panels, "ABC", strict=True):
        prim = gcd.ENDPOINTS[ep][0]
        hi = 0
        for i, arm in enumerate(main):
            r = res["arms"][arm][ep][prim]
            ax.bar(i, r["value"] - floor, bottom=floor, color=COLOR[arm], width=.65,
                   yerr=[[r["value"] - r["ci_lo"]], [r["ci_hi"] - r["value"]]], capsize=2)
            ax.text(i, r["ci_hi"], f"{r['value']:.3f}", ha="center", va="bottom", fontsize=7)
            hi = max(hi, r["ci_hi"])
        ax.axhline(floor, color="0.3", ls="--", lw=.8)
        ax.set(xticks=[], ylabel=yl, title=title, ylim=(floor, hi * 1.08))
        ax.text(-0.02, 1.08, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=COLOR[a]) for a in main], [LABEL[a] for a in main],
               loc="lower center", ncol=len(main), frameon=False, bbox_to_anchor=(.5, -.04))
    fig.tight_layout()
    paper = ROOT / "paper/figures"
    for ext in ("pdf", "png"):
        for d in (OUT, paper):
            fig.savefig(d / f"fig5_genomic_context.{ext}" if d == paper else d / f"fig_genomic_context_decodability.{ext}",
                        bbox_inches="tight", dpi=300)
    plt.close(fig)
    classes = {"cpg_context": gcd.CONTEXT_CLASSES, "gene_region": gcd.GENE_CLASSES}
    fig, axes = plt.subplots(2, len(main), figsize=(2.9 * len(main), 5.6), squeeze=False)
    for r, ep in enumerate(classes):
        for c, arm in enumerate(main):
            ax, m = axes[r, c], np.array(res["arms"][arm][ep]["confusion_row_normalised"])
            ax.imshow(m, vmin=0, vmax=1, cmap="Blues")
            k = len(classes[ep])
            ax.set(xticks=range(k), yticks=range(k), title=f"{LABEL[arm]}\n{ep.replace('_', ' ')}")
            ax.set_xticklabels(classes[ep], rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(classes[ep], fontsize=7)
            for i in range(k):
                for j in range(k):
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if m[i, j] > .5 else "black")
            ax.set_xlabel("predicted") if r == 1 else None
            ax.set_ylabel("true") if c == 0 else None
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figS_genomic_context_confusion.{ext}", bbox_inches="tight")
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--arm", required=True, choices=[*gcd.MAIN_ARMS, *gcd.SUPPLEMENTARY_ARMS])
    f.add_argument("--skip-sha", action="store_true")
    f.set_defaults(fn=cmd_fit)
    sub.add_parser("analyze").set_defaults(fn=cmd_analyze)
    sub.add_parser("figure").set_defaults(fn=cmd_figure)
    args = p.parse_args()
    args.fn(args)
