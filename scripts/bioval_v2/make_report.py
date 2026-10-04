# ruff: noqa: C408, UP031, ISC004
"""Build the biological_validation_v2 results report from ALREADY COMPUTED outputs (read-only).

Reads  outputs/biological_validation_v2/<tag>/<arm>/{run_manifest.json,endpoints/*.json,endpoints/*.replicates.npz}
       outputs/biological_validation_v2/<tag>/contrasts/*.json|csv
Writes only  <root>/report/{summary_primary.csv,summary_contrasts.csv,figures/*.png}  and  docs/BIOLOGICAL_VALIDATION_V2_RESULTS.md.
Nothing is recomputed: values, CIs, p-values and Holm adjustments are copied from the stored files. The only arithmetic is a
sign/direction cross-check (delta recomputed from stored point values and stored bootstrap replicates).
It refuses to write under any frozen path and imports no freeze / evaluation module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TAG = "bioval-v2-protocol-freeze-v1"
DEFAULT_ROOT = REPO / "outputs" / "biological_validation_v2" / TAG
DEFAULT_DOC = REPO / "docs" / "BIOLOGICAL_VALIDATION_V2_RESULTS.md"

REF = "regulatory_histone_dnase_v1"
COMPARATORS = ["functional_annotations_pca", "cpgpt_large_locus", "deepcpg_dna_locus"]
ARMS = [REF, *COMPARATORS]
SHORT = {
    REF: "Regulatory (histone+DNase)",
    "functional_annotations_pca": "Functional PCA (legacy)",
    "cpgpt_large_locus": "CpGPT-large",
    "deepcpg_dna_locus": "DeepCpG-DNA",
}
# Okabe-Ito based, fixed per arm across all figures
COLORS = {REF: "#D55E00", "functional_annotations_pca": "#0072B2", "cpgpt_large_locus": "#009E73",
          "deepcpg_dna_locus": "#CC79A7"}

# (endpoint, stat, label, metric description)
PRIMARY = [
    ("loyfer_profile", "spearman_pooled_gt1Mb_inter_eqw", "Loyfer WGBS profile similarity",
     "weighted Spearman(profile Pearson, embedding cosine), >1 Mb + interchromosomal, equal stratum weight; higher = better"),
    ("microc_H1_intra10kb", "auroc_contact_vs_noncontact", "H1 Micro-C intra 10 kb",
     "AUROC of cosine, contact vs matched non-contact; higher = better"),
    ("fantom5_membership", "auroc_oof_bg_enh5k", "FANTOM5 enhancer membership",
     "blocked-CV linear-probe AUROC vs matched controls (pool bg_enh5k); higher = better"),
    ("rt_consensus", "spearman_oof_consensus", "Replication timing (consensus z)",
     "out-of-fold ridge-probe Spearman vs rt_consensus_z; higher = better"),
]
STRATA = ["<1kb", "1-10kb", "10-100kb", "100kb-1Mb", ">1Mb", "interchromosomal"]
FROZEN_PREFIXES = [
    "docs/BIOLOGICAL_VALIDATION_V2.md", "docs/bioval_v2_prep", "configs/biological_validation_v2", "data/derived/bioval_v2",
    "docs/BIOLOGICAL_VALIDATION_V2_LAUNCH_REGISTRATION.md", "docs/BIOLOGICAL_VALIDATION_V2_EXECUTION_AUDIT.md",
    "src/cpg_repr_benchmark/biological_validation_v2", "src/cpg_repr_benchmark/bioval_v2_launch",
    "scripts/bioval_v2/run_evaluation.py",
]


# ----------------------------------------------------------------------------------------------- safety
def assert_writable(path: Path, root: Path, doc: Path, repo: Path = REPO) -> None:
    """Allow only <root>/report/** and the results doc; refuse every frozen path."""
    p = Path(path).resolve()
    for fp in FROZEN_PREFIXES:
        f = (repo / fp).resolve()
        if p == f or f in p.parents:
            raise PermissionError(f"refusing to write under frozen path: {p}")
    rep = (Path(root) / "report").resolve()
    if p == Path(doc).resolve() or rep in p.parents or p == rep:
        return
    raise PermissionError(f"refusing to write outside report dir / results doc: {p}")


# ----------------------------------------------------------------------------------------------- loading
def load_arm(root: Path, arm: str) -> dict:
    d = {"manifest": json.loads((root / arm / "run_manifest.json").read_text()), "endpoints": {}}
    for f in sorted((root / arm / "endpoints").glob("*.json")):
        d["endpoints"][f.stem] = json.loads(f.read_text())
    return d


def load_contrasts(root: Path) -> dict[str, pd.DataFrame]:
    return {c: pd.read_csv(root / "contrasts" / f"{REF}__vs__{c}.csv") for c in COMPARATORS}


def stat_of(arms: dict, arm: str, ep: str, stat: str) -> dict:
    return arms[arm]["endpoints"][ep]["stats"][stat]


# ----------------------------------------------------------------------------------------------- tables
def primary_summary(arms: dict) -> pd.DataFrame:
    rows = []
    for ep, st, _lab, _desc in PRIMARY:
        for a in ARMS:
            s = stat_of(arms, a, ep, st)
            rows.append(dict(endpoint=ep, arm=a, value=s["value"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"]))
    return pd.DataFrame(rows)


def contrast_summary(contrasts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keep = ["comparator", "endpoint", "stat", "stat_class", "ref_value", "comp_value", "delta", "ci_lo", "ci_hi", "p",
            "in_holm_family", "holm_p_adjusted", "supports_regulatory"]
    return pd.concat([c[keep] for c in contrasts.values()], ignore_index=True)


def classify(delta: float, lo: float, hi: float) -> str:
    """Direction of delta = regulatory - comparator (all registered statistics are higher-is-better)."""
    if lo > 0:
        return "regulatory higher"
    if hi < 0:
        return "regulatory lower"
    return "CI includes 0"


def reading(label: str, comp: str, delta: float, lo: float, hi: float, holm: float | None) -> str:
    cls = classify(delta, lo, hi)
    c = SHORT[comp]
    h = "" if holm is None or pd.isna(holm) else f"; Holm p={holm:.3g}"
    if cls == "CI includes 0":
        return (f"No detectable difference vs {c}: delta {delta:+.4f} (CI {lo:+.4f} to {hi:+.4f} includes 0){h}")
    if cls == "regulatory higher":
        return f"Regulatory better than {c} by {abs(delta):.4f}{h}"
    return f"{c} better than regulatory by {abs(delta):.4f}{h}"


def fmt(v, lo=None, hi=None, nd=4) -> str:
    if lo is None:
        return f"{v:.{nd}f}"
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def sign_check(root: Path, arms: dict, contrasts: dict[str, pd.DataFrame]) -> list[dict]:
    """Arithmetic cross-check of the stored contrasts against the stored endpoint values/replicates (no re-evaluation)."""
    out = []
    for comp in COMPARATORS:
        c = contrasts[comp]
        for ep, st, _l, _d in PRIMARY:
            row = c[(c.endpoint == ep) & (c.stat == st)].iloc[0]
            v_ref = stat_of(arms, REF, ep, st)["value"]
            v_cmp = stat_of(arms, comp, ep, st)["value"]
            r_ref = np.load(root / REF / "endpoints" / f"{ep}.replicates.npz")[st]
            r_cmp = np.load(root / comp / "endpoints" / f"{ep}.replicates.npz")[st]
            d = r_ref - r_cmp
            lo, hi = np.percentile(d, [2.5, 97.5])
            out.append(dict(
                comparator=comp, endpoint=ep,
                delta_from_values=v_ref - v_cmp, delta_in_contrast_file=float(row.delta),
                sign_agrees=bool(np.sign(v_ref - v_cmp) == np.sign(row.delta)),
                ci_from_replicates=(float(lo), float(hi)), ci_in_contrast_file=(float(row.ci_lo), float(row.ci_hi)),
                max_abs_diff=float(max(abs(v_ref - v_cmp - row.delta), abs(lo - row.ci_lo), abs(hi - row.ci_hi))),
                k_le0_from_replicates=int((d <= 0).sum()), k_le0_in_file=int(row.k_le0)))
    return out


# ----------------------------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150})
    return plt


def _dot_panel(ax, arms, ep, st, title, ref_line=None, per_fold_key=None):
    for i, a in enumerate(ARMS):
        s = stat_of(arms, a, ep, st)
        y = len(ARMS) - 1 - i
        ax.errorbar(s["value"], y, xerr=[[s["value"] - s["ci_lo"]], [s["ci_hi"] - s["value"]]], fmt="o", color=COLORS[a],
                    ms=7, capsize=3, lw=1.6)
        if per_fold_key and s.get("meta", {}).get(per_fold_key):
            vals = list(s["meta"][per_fold_key].values())
            ax.scatter(vals, [y] * len(vals), s=14, color=COLORS[a], alpha=0.45, marker="|", zorder=1)
        ax.annotate(f"{s['value']:.3f}", (s["value"], y), textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8, color="#333")
    ax.set_yticks(range(len(ARMS)))
    ax.set_yticklabels([SHORT[a] for a in reversed(ARMS)])
    ax.set_title(title, loc="left", fontsize=10)
    if ref_line is not None:
        ax.axvline(ref_line, color="#888", lw=1, ls="--")
    ax.grid(axis="y", visible=False)


def make_figures(arms: dict, contrasts: dict, figdir: Path, write) -> list[Path]:
    plt = _plt()
    paths = []

    def save(fig, name):
        p = figdir / name
        write(p)
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(p)

    # 1 Loyfer strata
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axvspan(3.5, 5.5, color="#f3efe0", zorder=0)
    ax.axhline(0, color="#666", lw=1)
    off = np.linspace(-0.27, 0.27, len(ARMS))
    for a, o in zip(ARMS, off):
        v, lo, hi = [], [], []
        for s_ in STRATA:
            s = stat_of(arms, a, "loyfer_profile", f"spearman_{s_}")
            v.append(s["value"]); lo.append(s["ci_lo"]); hi.append(s["ci_hi"])
        v, lo, hi = map(np.array, (v, lo, hi))
        x = np.arange(len(STRATA)) + o
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="o", color=COLORS[a], ms=5, capsize=2, lw=1.3, label=SHORT[a])
    ax.set_xticks(range(len(STRATA)))
    ax.set_xticklabels(STRATA)
    ax.set_ylabel("Spearman(profile similarity, cosine)")
    ax.set_title("Loyfer WGBS: per-stratum Spearman (shaded = the two primary strata, pooled)", loc="left", fontsize=10)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper right")
    save(fig, "fig1_loyfer_strata.png")

    # 2 Micro-C
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    _dot_panel(axs[0], arms, "microc_H1_intra10kb", "auroc_contact_vs_noncontact", "AUROC contact vs non-contact (primary)", 0.5)
    _dot_panel(axs[1], arms, "microc_H1_intra10kb", "delta_cosine_paired_mean", "paired delta-cosine, pos - neg (secondary)", 0.0)
    axs[1].set_yticklabels([])
    save(fig, "fig2_microc_auroc.png")

    # 3 FANTOM5
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    _dot_panel(ax, arms, "fantom5_membership", "auroc_oof_bg_enh5k", "FANTOM5 membership AUROC (bg_enh5k, blocked linear probe)", 0.5,
               "per_fold_auroc")
    ax.set_xlabel("AUROC (ticks = per-fold)")
    save(fig, "fig3_fantom5_auroc.png")

    # 4 RT
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    _dot_panel(ax, arms, "rt_consensus", "spearman_oof_consensus", "RT consensus out-of-fold Spearman", None, None)
    for i, a in enumerate(ARMS):
        pf = stat_of(arms, a, "rt_consensus", "spearman_oof_consensus")["meta"]["per_fold_spearman"]
        vals = list(pf.values())
        ax.scatter(vals, [len(ARMS) - 1 - i] * len(vals), s=22, color=COLORS[a], alpha=0.45, marker="|", zorder=1)
    ax.set_xlabel("Spearman (ticks = per-fold)")
    save(fig, "fig4_rt_spearman.png")

    # 5 forest
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (ep, st, lab, _d) in zip(axs, PRIMARY):
        for j, comp in enumerate(COMPARATORS):
            r = contrasts[comp]
            r = r[(r.endpoint == ep) & (r.stat == st)].iloc[0]
            y = len(COMPARATORS) - 1 - j
            ax.errorbar(r.delta, y, xerr=[[r.delta - r.ci_lo], [r.ci_hi - r.delta]], fmt="o", color=COLORS[comp], capsize=3, ms=7)
            ax.annotate(f"{r.delta:+.3f}", (r.delta, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
        ax.axvline(0, color="#666", lw=1)
        ax.set_ylim(-0.5, len(COMPARATORS) - 0.4)
        ax.margins(x=0.15)
        ax.set_yticks(range(len(COMPARATORS)))
        ax.set_yticklabels([SHORT[c] for c in reversed(COMPARATORS)] if ax is axs[0] else [])
        ax.set_title(lab, fontsize=9, loc="left", pad=10)
        ax.set_xlabel("delta = regulatory - comparator")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Paired deltas (95% chromosome-block bootstrap CI); right of 0 = regulatory better", x=0.01, y=1.08, ha="left", fontsize=10)
    save(fig, "fig5_forest_primary_deltas.png")
    return paths


# ----------------------------------------------------------------------------------------------- markdown
def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def stat_row_table(arms, items, nd=4):
    rows = []
    for ep, st, lab in items:
        row = [lab]
        for a in ARMS:
            try:
                s = stat_of(arms, a, ep, st)
                row.append(fmt(s["value"], s["ci_lo"], s["ci_hi"], nd))
            except KeyError:
                row.append("n/a")
        rows.append(row)
    return md_table(["Statistic", *[SHORT[a] for a in ARMS]], rows)


def build_markdown(arms, contrasts, checks, root: Path) -> str:
    m = arms[REF]["manifest"]
    L: list[str] = []
    A = L.append
    A("# biological_validation_v2: RESULTS (report only)\n")
    A("Generated by `scripts/bioval_v2/make_report.py` from already-computed outputs (read-only; nothing recomputed).\n")
    A(md_table(["Item", "Value"], [
        ["Protocol freeze tag / commit", f"`{TAG}` / `27d8bce`"],
        ["Registration commit", "`4ffa2cd` (`docs/BIOLOGICAL_VALIDATION_V2_LAUNCH_REGISTRATION.md`)"],
        ["Code commit (run HEAD)", f"`{m['head_commit'][:7]}`"],
        ["Computed on", "2026-10-04 (runs 09:33-11:15 UTC)"],
        ["Seed / bootstrap", f"{m['seed']} / B={m['n_bootstrap']}, 22 chromosome blocks, one shared draw table (sha256 `{m['bootstrap_draw_table_sha256'][:12]}...`)"],
        ["Library versions", ", ".join(f"{k} {v}" for k, v in m["library_versions"].items())],
        ["Statements", "No methodology changed after the first real run. The candidate was not modified by these results. The TCGA test set was never used (`tcga_test_set_referenced=false` in all 4 manifests). No freeze module imported, no freeze command used."],
        ["Output root", f"`{root.relative_to(REPO) if root.is_relative_to(REPO) else root}`"],
    ]))
    A("\n**Sign convention (registered, all statistics higher-is-better):** contrast delta = regulatory_histone_dnase_v1 MINUS comparator. "
      "delta > 0 means the regulatory arm is better; delta < 0 means the comparator is better. Holm (alpha 0.05) is applied over the 4 primary "
      "endpoints, per contrast. 'Regulatory supports' requires Holm p < 0.05 AND delta > 0. No aggregate score is built. Arms: "
      + "; ".join(f"`{a}` = {SHORT[a]}" for a in ARMS) + ".\n")

    # (a)
    A("## 1. PRIMARY endpoints x arms (value with 95% chromosome-block bootstrap CI)\n")
    rows = []
    for ep, st, lab, desc in PRIMARY:
        row = [f"**{lab}**<br>`{st}`<br>{desc}"]
        for a in ARMS:
            s = stat_of(arms, a, ep, st)
            row.append(fmt(s["value"], s["ci_lo"], s["ci_hi"]))
        rows.append(row)
    A(md_table(["Endpoint / metric / direction", *[SHORT[a] for a in ARMS]], rows))
    A("\nSupplementary values attached to the primaries (same arms, same CIs):\n")
    A(stat_row_table(arms, [
        ("microc_H1_intra10kb", "delta_cosine_paired_mean", "Micro-C paired delta-cosine (pos - neg), secondary effect size"),
        ("rt_consensus", "r2_oof_consensus", "RT consensus out-of-fold R2 (pooled)"),
    ]))
    A("\nPer-fold values (point estimates, no CI; chromosome-blocked folds 0-4):\n")
    rows = []
    for a in ARMS:
        rt = stat_of(arms, a, "rt_consensus", "spearman_oof_consensus")["meta"]["per_fold_spearman"]
        r2 = stat_of(arms, a, "rt_consensus", "r2_oof_consensus")["meta"]["per_fold_r2"]
        fa = stat_of(arms, a, "fantom5_membership", "auroc_oof_bg_enh5k")["meta"]["per_fold_auroc"]
        rows.append([SHORT[a], " / ".join(f"{v:.3f}" for v in rt.values()), " / ".join(f"{v:.3f}" for v in r2.values()),
                     " / ".join(f"{v:.3f}" for v in fa.values())])
    A(md_table(["Arm", "RT Spearman per fold", "RT R2 per fold", "FANTOM5 AUROC per fold"], rows))
    A("\nFigures: ![Loyfer](../outputs/biological_validation_v2/%s/report/figures/fig1_loyfer_strata.png) "
      "![MicroC](../outputs/biological_validation_v2/%s/report/figures/fig2_microc_auroc.png) "
      "![FANTOM5](../outputs/biological_validation_v2/%s/report/figures/fig3_fantom5_auroc.png) "
      "![RT](../outputs/biological_validation_v2/%s/report/figures/fig4_rt_spearman.png)\n" % ((TAG,) * 4))

    # (b)
    A("## 2. PAIRED comparisons: regulatory_histone_dnase_v1 vs each comparator (4 primaries)\n")
    A("delta = regulatory - comparator (higher-is-better metrics). CI = 95% percentile of the paired delta replicates. raw p = two-sided "
      "bootstrap p; its floor is (1+0)/(1+1000) = 0.002, so 0.002 means 'no bootstrap replicate crossed 0', not a measured tiny p. "
      "Holm p = Holm over the 4 primaries within the contrast (floor 0.008).\n")
    rows = []
    for comp in COMPARATORS:
        c = contrasts[comp]
        for ep, st, lab, _d in PRIMARY:
            r = c[(c.endpoint == ep) & (c.stat == st)].iloc[0]
            rows.append([SHORT[comp], lab, f"{r.delta:+.4f}", f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]", f"{r.p:.3f}",
                         f"{r.holm_p_adjusted:.3f}", "yes" if bool(r.supports_regulatory) else "no",
                         reading(lab, comp, r.delta, r.ci_lo, r.ci_hi, r.holm_p_adjusted)])
    A(md_table(["Comparator", "Endpoint", "delta (reg - comp)", "95% CI", "raw p", "Holm p", "supports regulatory (Holm<0.05 & delta>0)", "Plain-language reading"], rows))
    A("\nForest plot: ![forest](../outputs/biological_validation_v2/%s/report/figures/fig5_forest_primary_deltas.png)\n" % TAG)
    A("**Sign double check (performed by the report script, arithmetic only).** For each of the 12 cells, delta was recomputed from the stored "
      "per-arm point values in `endpoints/<endpoint>.json` and, for the CI, from the stored replicate arrays "
      "(`endpoints/*.replicates.npz`, regulatory minus comparator, 2.5/97.5 percentiles); both match the contrast files:\n")
    rows = [[SHORT[x["comparator"]], x["endpoint"], f"{x['delta_from_values']:+.5f}", f"{x['delta_in_contrast_file']:+.5f}",
             "yes" if x["sign_agrees"] else "NO", f"{x['max_abs_diff']:.1e}", f"{x['k_le0_from_replicates']} / {x['k_le0_in_file']}"]
            for x in checks]
    A(md_table(["Comparator", "Endpoint", "reg - comp from raw values", "delta in contrast file", "sign agrees", "max abs diff (delta, CI)", "k(delta*<=0) replicates / file"], rows))
    A("\nExample (Loyfer vs DeepCpG): regulatory = %.4f, DeepCpG = %.4f, so regulatory - DeepCpG = %.4f < 0: DeepCpG is better on Loyfer.\n"
      % (stat_of(arms, REF, "loyfer_profile", PRIMARY[0][1])["value"], stat_of(arms, "deepcpg_dna_locus", "loyfer_profile", PRIMARY[0][1])["value"],
         stat_of(arms, REF, "loyfer_profile", PRIMARY[0][1])["value"] - stat_of(arms, "deepcpg_dna_locus", "loyfer_profile", PRIMARY[0][1])["value"]))

    # (c)
    A("## 3. Loyfer WGBS: strata and secondary marker-kNN\n")
    A("Unweighted per-stratum Spearman (descriptive; pairs kept after zero-locus exclusion). **The primary is the pooled equal-weight statistic of the "
      ">1 Mb and interchromosomal strata only** (shown in Section 1). Short strata mostly measure genomic proximity.\n")
    items = [("loyfer_profile", f"spearman_{s}", f"{s}" + ("  **(primary stratum)**" if s in (">1Mb", "interchromosomal") else "")) for s in STRATA]
    A(stat_row_table(arms, items))
    A("\nDescriptive alternatives (never replace the primary):\n")
    A(stat_row_table(arms, [
        ("loyfer_profile", "spearman_pooled_gt1Mb_inter_eqw", "PRIMARY: weighted pooled Spearman, >1Mb + inter, equal stratum weight"),
        ("loyfer_profile", "spearman_mean_of_gt1Mb_and_inter", "mean of the two per-stratum Spearman values"),
        ("loyfer_profile", "spearman_all_pairs", "Spearman over all pairs (386,268 minus exclusions), descriptive"),
    ]))
    A("\nPaired per-stratum deltas (regulatory - comparator; raw p only, descriptive, no claim, outside the Holm family):\n")
    rows = []
    for s in STRATA:
        row = [s]
        for comp in COMPARATORS:
            c = contrasts[comp]
            r = c[(c.endpoint == "loyfer_profile") & (c.stat == f"spearman_{s}")].iloc[0]
            row.append(f"{r.delta:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")
        rows.append(row)
    A(md_table(["Stratum", *[f"vs {SHORT[c]}" for c in COMPARATORS]], rows))
    A("\nMarker-kNN (SECONDARY, k = 10; marker hg38 provenance verdict WEAK; low power: 1,257 markers = 157 U25 + 1,100 U250-only; "
      "enrichment = observed same-label neighbour fraction / expected; CI = chromosome-block bootstrap of marker loci). "
      "The one-sided within-chromosome permutation p-value (999 permutations, seed 17; floor 0.001) is 0.001 for every arm and set. "
      "**Caveat on that p-value:** the frozen `metrics.knn_enrichment(n_perm>0)` is defective (marker mask not recomputed after permutation; raises KeyError or "
      "evaluates the null at the wrong loci), so the observed enrichment uses the frozen function with n_perm=0 and the p-value comes from the new routine "
      "`launch_endpoints.knn_permutation_pvalue`, which is not a frozen artifact. The p-value is descriptive only; since it saturates at its floor "
      "for all arms it carries no discriminating information.\n")
    A(stat_row_table(arms, [
        ("loyfer_marker_knn", "enrichment_github_U25_U250", "enrichment, github U25+U250 (secondary, k=10)"),
        ("loyfer_marker_knn", "enrichment_github_U25_only", "enrichment, U25 only (sensitivity, 157 markers)"),
        ("loyfer_marker_knn", "enrichment_paper_lifted_U25_U250", "enrichment, paper-lifted U25+U250 (sensitivity)"),
    ], nd=1))
    rows = []
    for comp in COMPARATORS:
        c = contrasts[comp]
        r = c[(c.endpoint == "loyfer_marker_knn") & (c.stat == "enrichment_github_U25_U250")].iloc[0]
        rows.append([SHORT[comp], f"{r.delta:+.1f} [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}]", f"{r.p:.3f}"])
    A("\nPaired delta (regulatory - comparator), secondary set, raw p only:\n")
    A(md_table(["Comparator", "delta enrichment [95% CI]", "raw p"], rows))
    A("\nNote the kNN ordering (regulatory ~ functional > CpGPT >> DeepCpG) is the opposite of the pair-level Loyfer primary ordering "
      "(DeepCpG ~ CpGPT > functional > regulatory). The two Loyfer readouts measure different things (local neighbourhood purity of marker loci vs "
      "genome-wide cosine/profile-similarity rank agreement); whether kNN neighbours are largely genomic neighbours was not checked.\n")

    # (d)
    A("## 4. SECONDARY, SENSITIVITY and EXPLORATORY results (not mixed into the primaries; none enters the Holm family)\n")
    A("### 4.1 SECONDARY\n")
    A(stat_row_table(arms, [
        ("compartment_E1_probe_GM12878_100kb", "spearman_E1_vs_oof_pred", "GM12878 100 kb compartment E1 probe: Spearman"),
        ("compartment_E1_probe_GM12878_100kb", "r2_oof", "GM12878 E1 probe: R2"),
        ("compartment_E1_probe_H1_100kb", "spearman_E1_vs_oof_pred", "H1 100 kb compartment E1 probe: Spearman"),
        ("compartment_E1_probe_H1_100kb", "r2_oof", "H1 E1 probe: R2"),
        ("fantom5_activity_similarity", "spearman_all_pairs", "FANTOM5 activity-profile similarity: Spearman, all pairs (171,017 minus exclusions)"),
        ("fantom5_activity_similarity", "spearman_intra", "  intra-chromosomal (descriptive)"),
        ("fantom5_activity_similarity", "spearman_inter", "  inter-chromosomal (descriptive)"),
        ("microc_H1_intra10kb", "delta_cosine_paired_mean", "H1 Micro-C paired delta-cosine"),
        ("microc_H1_intra10kb", "auroc_distance_bin_[10000,100000)", "H1 Micro-C AUROC, distance bin [10 kb, 100 kb)"),
        ("microc_H1_intra10kb", "auroc_distance_bin_[100000,1e+06)", "H1 Micro-C AUROC, distance bin [100 kb, 1 Mb)"),
        ("microc_H1_intra10kb", "auroc_distance_bin_[1e+06,1e+07)", "H1 Micro-C AUROC, distance bin [1 Mb, 10 Mb)"),
        ("rt_consensus", "r2_oof_consensus", "RT consensus R2"),
    ]))
    A("\nThe E1 target orientation is GC-derived (protocol caveat 4), and compartments are `partially_related` to histone/DNase inputs.\n")
    A("### 4.2 SENSITIVITY (declared variants of a primary target)\n")
    A(stat_row_table(arms, [
        ("microc_HFFc6_intra10kb", "auroc_contact_vs_noncontact", "HFFc6 Micro-C intra 10 kb: AUROC"),
        ("microc_HFFc6_intra10kb", "delta_cosine_paired_mean", "HFFc6 Micro-C: paired delta-cosine"),
        ("fantom5_membership", "auroc_oof__win500", "FANTOM5 +-500 bp window (26,880 pairs): AUROC"),
        ("fantom5_membership", "auroc_oof__tss5k_pool", "FANTOM5 TSS-filtered pool (3,349 pairs): AUROC (poorly balanced: SMD tss_dist -0.117)"),
        ("loyfer_profile", "spearman_pooled_eqw__profile_spearman", "Loyfer primary using profile Spearman instead of Pearson"),
        ("loyfer_profile", "pearsoncorr_pooled_eqw__profile_pearson", "Loyfer: Pearson correlation instead of Spearman"),
        ("loyfer_profile", "spearman_pooled_eqw__min30_shared_groups", "Loyfer primary, pairs with >=30 shared groups"),
    ]))
    A("\nPaired delta (regulatory - comparator) for selected sensitivities (raw p only, no claim):\n")
    sel = [("microc_HFFc6_intra10kb", "auroc_contact_vs_noncontact", "HFFc6 AUROC"),
           ("microc_HFFc6_intra10kb", "delta_cosine_paired_mean", "HFFc6 delta-cosine"),
           ("fantom5_membership", "auroc_oof__win500", "FANTOM5 win500 AUROC"),
           ("fantom5_membership", "auroc_oof__tss5k_pool", "FANTOM5 TSS-pool AUROC"),
           ("loyfer_profile", "spearman_pooled_eqw__profile_spearman", "Loyfer (profile Spearman)"),
           ("loyfer_profile", "spearman_pooled_eqw__min30_shared_groups", "Loyfer (>=30 shared groups)"),
           ("microc_H1_intra10kb", "delta_cosine_paired_mean", "H1 delta-cosine (secondary)")]
    rows = []
    for ep, st, lab in sel:
        row = [lab]
        for comp in COMPARATORS:
            c = contrasts[comp]
            r = c[(c.endpoint == ep) & (c.stat == st)].iloc[0]
            row.append(f"{r.delta:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] (p={r.p:.3f})")
        rows.append(row)
    A(md_table(["Statistic", *[f"vs {SHORT[c]}" for c in COMPARATORS]], rows))
    A("\nPer-line RT (15 lines; Spearman of out-of-fold ridge prediction vs the line's own RT; sensitivity):\n")
    lines = sorted(k[len("spearman_oof_line_"):] for k in arms[REF]["endpoints"]["rt_consensus"]["stats"] if k.startswith("spearman_oof_line_"))
    A(stat_row_table(arms, [("rt_consensus", f"spearman_oof_line_{ln}", ln) for ln in lines], nd=3))
    A("\n### 4.3 EXPLORATORY\n")
    A("H1 inter-chromosomal 1 Mb (rows with fold >= 0 only; protocol: 98.3% drop, weak balance SMD 0.30; 9,931 matched pairs before zero-locus exclusion) "
      "and PMD (Decato 2020, methylation-derived; ridge-regression probe on the 0/1 label, AUROC of out-of-fold scores; "
      "registered as ridge not logistic).\n")
    A(stat_row_table(arms, [
        ("microc_H1_inter1Mb", "auroc_contact_vs_noncontact", "H1 inter 1 Mb: AUROC"),
        ("microc_H1_inter1Mb", "delta_cosine_paired_mean", "H1 inter 1 Mb: paired delta-cosine"),
        ("pmd_probe", "auroc_oof_pmd_in_any", "PMD membership (in any sample): AUROC"),
    ]))
    rows = []
    for ep, st, lab in [("microc_H1_inter1Mb", "auroc_contact_vs_noncontact", "H1 inter AUROC"),
                        ("microc_H1_inter1Mb", "delta_cosine_paired_mean", "H1 inter delta-cosine"),
                        ("pmd_probe", "auroc_oof_pmd_in_any", "PMD AUROC")]:
        row = [lab]
        for comp in COMPARATORS:
            c = contrasts[comp]
            r = c[(c.endpoint == ep) & (c.stat == st)].iloc[0]
            row.append(f"{r.delta:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] (p={r.p:.3f})")
        rows.append(row)
    A("\nPaired delta (regulatory - comparator), raw p only:\n")
    A(md_table(["Statistic", *[f"vs {SHORT[c]}" for c in COMPARATORS]], rows))
    A("\n### 4.4 Declared in the protocol / registration but NOT run in this launch\n")
    A("- Loyfer: `mask_lenient`, single-sample-group exclusion, U25-only for Loyfer pairs (would require re-deriving profile-similarity targets from `loyfer_celltype_beta.h5`, a new freeze artifact).\n"
      "- FANTOM5: expression-threshold variants (N = 1/3/10/20 libraries), additional chromatin-covariate matching, sample-level vs collapsed activity variants, per-category activity analyses (no frozen pair/membership files).\n"
      "- Replication timing: residualisation for GC / CpG density / TSS distance (or gene density), and the normal-vs-tumour line split (covariate file / line assignment not frozen). The 15 per-line RT results are reported above.\n"
      "- 3D: 50 kb compartments, K-per-bin-pair sensitivity, distance-only/covariate baseline comparison, HFFc6 compartments (registered not evaluated), `stratified_by_distance` for Loyfer.\n"
      "- No cosine=0 sensitivity for the zero rows, no contrast among the three comparators, no equivalence/non-inferiority margins.\n")

    # (f)
    A("## 5. Complete caveat list\n")
    cav = [
        "**Relatedness of endpoints to the representation inputs (protocol section 3).** Regulatory inputs = 2,492 ENCODE histone ChIP + DNase tracks (the registered 'histone+DNase' candidate; TF/CTCF tracks are not in its store). "
        "Loyfer WGBS, RT and H1 inter: independent. H1/HFFc6 intra contacts: `partially_related` (CTCF/RAD21/SMC3 anchor biology; the regulatory candidate has no CTCF/cohesin tracks but histone/DNase relate to anchors). "
        "Compartments: partially related to histone/DNase and sequence (E1 oriented by GC). **FANTOM5 membership and activity: `partially_related`** (H3K27ac/H3K4me1/DNase mark the same enhancers; GENCODE TSS is a matching covariate), so the FANTOM5 results are not independent validation for the histone/DNase-based arms. PMD is independent of inputs but methylation-derived.",
        "**The functional baseline has dense genomic-context inputs** (island/shore/shelf, gene region, cCRE V4, TSS distance) plus 4,165 binary tracks incl. TF/CTCF; these correlate with enhancer, TSS and compartment structure. The comparison regulatory vs functional is therefore between different input sets, not only different compressions.",
        "**Fit-scope mismatch.** The regulatory SVD was fit on chr1-19 (388,599 loci); chr20-22 are transformed only (out-of-fit). Functional PCA and CpGPT/DeepCpG were fit/pretrained without that restriction. chr20-22 are 3 of 22 bootstrap blocks and fall in chromosome-blocked folds like any other.",
        "**Dimension/dtype differences:** 256 (float32), 256 (float16), 512 (float16), 128 (float16). Probe capacity and cosine geometry differ with dimension; no dimension matching was attempted. All arms loaded as float32.",
        "**Zero-row exclusion.** 849 regulatory loci have all-zero rows. For cosine endpoints every pair/row touching any such locus is dropped for ALL arms (Loyfer 1,593 pairs of 386,268; H1 intra 15,474 rows of 5,379,842, 2,674,501 of 2,689,921 matched pairs kept for delta-cosine; HFFc6 22,205 rows; H1 inter 488 rows of 19,862; FANTOM5 activity 176 pairs; kNN candidates 407,550). Linear-probe endpoints (FANTOM5 membership, RT, PMD, compartments) keep zero rows as valid features (e.g. 23 zero rows among the FANTOM5 probe set for the regulatory arm only); these rows are plausibly easy negatives for the candidate. No cosine=0 sensitivity.",
        "**Methylation pretraining.** CpGPT and DeepCpG are trained on methylation (modality shared with Loyfer WGBS and PMD). The Loyfer axis is therefore not modality-independent for those comparators, and their advantage there may reflect methylation exposure; the extraction pipelines are nominally patient-independent, but the exact pretraining data overlap with the Loyfer atlas was not audited here. The regulatory and functional arms have no methylation input.",
        "**Weighted-Spearman definition.** 'Equal stratum weight' was operationalised at launch (not in frozen code) as the weighted Pearson correlation of weighted mid-rank fractions, item weight 1/n_stratum. The descriptive mean-of-two-strata alternative is reported; the two agree closely in every arm (identical to 4 decimals in every arm).",
        "**Probe grids.** Ridge alpha grid 1e-2..1e4 in scikit-learn convention (not divided by n; with n ~ 3e5 weak regularisation; chosen alphas of 1e3-1e4 e.g. 1e3 in all five RT folds for the regulatory arm, 1e4 for its PMD probe); logistic C 1e-4..10; nested leave-one-frozen-fold-out inner CV; PMD uses ridge regression rather than logistic. Grid edges were not extended (would be a methodology change).",
        "**Single seed and bootstrap scope.** Single seed (17) for the folds, draws and probes; probe predictions are NOT refitted inside the bootstrap, so CIs reflect chromosome resampling only, not probe-fitting variance or seed variance.",
        "**Block bootstrap.** 22 chromosome blocks, B=1000: coarse CIs; raw p floor (1+k)/(1+B) = 0.002 (k=0), Holm-adjusted floor 0.008 for the best-case 4-endpoint family; values at the floor are not measured effect-size probabilities. Chromosomes are heterogeneous (size, GC, gene density), and genome-wide dependence (e.g. shared sequence families, long-range structure) is not captured by chromosome blocks. Inter-chromosomal pairs are assigned to the block of cpg_i, which only approximates their dependence.",
        "**Frozen `knn_enrichment` bug workaround** (see Section 3): p-value from a new routine, descriptive only; marker hg38 provenance WEAK; low power; enrichment values are very large because expected same-label fractions are tiny, and U25-only values are unstable (e.g. very wide CIs).",
        "**Sensitivities not run** (Section 4.4), including RT covariate residualisation: the RT result may partly reflect GC / gene-density content that the candidate encodes through histone/DNase signal.",
        "**Intra-chromosomal distance confounds.** Loyfer short strata largely measure genomic proximity; Micro-C controls are distance-matched on a log grid but not exactly (median distance difference ~2.1 kb in pairs) and cosine similarity of any locus-embedding may vary with distance; Micro-C AUROC pools all distances and (on the registered distance bins) is heterogeneous.",
        "**No multiplicity control across contrasts** (3 contrasts, each with its own Holm family of 4) and none across the secondary/sensitivity/exploratory statistics (raw p only). Holm adjusts the family; it is NOT a selection rule, and no endpoint was used to select a representation.",
        "**Effect sizes versus significance.** With millions of pairs and 22 blocks, small differences (e.g. Micro-C regulatory vs functional AUROC delta +0.0008) can be 'significant' in the secondary delta-cosine yet immaterial; conversely wide CIs on chromosomal blocks can hide genuine small differences. The protocol defines NO equivalence margin.",
        "**Other data caveats (protocol section 5):** single cell line per 3D system, different Micro-C depth, RT Rep1 only with liftover hg19->hg38, FANTOM5 permissive set (median 1 library per group, 90.8% zeros), Loyfer sample imbalance (205 samples = 168 donors), marker sets hypomethylated only, H1 inter weakly balanced.",
    ]
    for i, t in enumerate(cav, 1):
        A(f"{i}. {t}")

    # (g)
    A("\n## 6. Pattern classification (cautious; numbers only)\n")
    A("Patterns requested: **A** = regulatory > sequence models on all/almost all endpoints; **B** = regulatory ~ legacy functional but > sequence models; "
      "**C** = legacy functional > regulatory also biologically; **D** = endpoint-specific / mixed. The protocol defines **no equivalence margin**, so "
      "'approximately equal' is read here only as 'paired 95% CI includes 0' (strict) and effect sizes are described alongside. "
      "No thresholds were invented.\n")
    rows = []
    for ep, st, lab, _d in PRIMARY:
        row = [lab]
        for comp in COMPARATORS:
            r = contrasts[comp]
            r = r[(r.endpoint == ep) & (r.stat == st)].iloc[0]
            row.append(f"{classify(r.delta, r.ci_lo, r.ci_hi)} ({r.delta:+.3f})")
        rows.append(row)
    A(md_table(["Primary endpoint", *[f"vs {SHORT[c]}" for c in COMPARATORS]], rows))
    A("""
Reading by endpoint (primaries):
- **Loyfer WGBS profile (independent target):** regulatory is the LOWEST of the four arms and its primary value is negative (about -0.06), significantly below functional (delta -0.025), CpGPT (-0.169) and DeepCpG (-0.178) (all Holm p = 0.008, i.e. at the floor). Sequence/methylation-pretrained models are best (DeepCpG ~ CpGPT; their mutual difference was not tested; their CIs overlap). This is unfavourable for the regulatory candidate and is **not** consistent with pattern A. Note functional is also below both sequence models, so for this endpoint the order is sequence models > functional > regulatory (not pattern C either, since functional vs regulatory differ by a small amount in functional's favour).
- **H1 Micro-C intra 10 kb (partially related):** regulatory ~ functional (AUROC delta +0.0008, CI includes 0, Holm p = 0.16; the secondary delta-cosine favours regulatory by +0.0017, CI excludes 0, raw p only); regulatory > CpGPT (+0.037) and DeepCpG (+0.066), Holm-supported. Absolute AUROC values are modest (0.50-0.57). Pattern B on this endpoint.
- **FANTOM5 membership (partially related):** functional > regulatory by 0.017 (CI excludes 0); regulatory >> CpGPT (+0.132) and DeepCpG (+0.210). Pattern C-like against the functional arm (small effect, favouring the legacy arm) with regulatory > sequence models. Because FANTOM5 is partially related to the input tracks, and the functional arm additionally contains TF tracks and cCRE/TSS columns, this does not isolate biology.
- **RT consensus (independent):** functional > regulatory by 0.015 Spearman (CI excludes 0); regulatory >> CpGPT (+0.217) and DeepCpG (+0.344). Same shape as FANTOM5: functional slightly ahead, both far above sequence models.

**Which pattern(s) the data support.** No single pattern describes all four primaries: the overall result is **D (endpoint-specific / mixed)**. On three of four primaries (Micro-C, FANTOM5, RT) regulatory is clearly above both sequence models (B/A-like) and the functional arm is equal (Micro-C) or modestly higher (FANTOM5, RT: +0.017 AUROC, +0.015 Spearman), i.e. regulatory does NOT reach functional there, though the gap is small relative to the gap to the sequence models (a strict 'approximately equal' reading fails because the CIs exclude 0, and no equivalence margin exists to justify calling them equal). On the independent Loyfer methylation-program endpoint the order reverses: sequence models > functional > regulatory, and regulatory is below zero. Regulatory is therefore not better than the legacy functional baseline on any primary endpoint under this protocol (supported only as 'no detectable difference' on Micro-C AUROC). Pattern A is contradicted by Loyfer; pattern B holds for Micro-C only; pattern C holds in a weak, small-effect form for FANTOM5 and RT (and Loyfer), and 'also biologically' is limited by the fact that functional has more inputs. Secondary evidence is directionally the same: E1 probes and FANTOM5 activity similarity show regulatory ~ functional >> sequence models (CIs overlap for regulatory vs functional), the marker-kNN secondary shows regulatory ~ functional > CpGPT >> DeepCpG, and the HFFc6 sensitivity (regulatory above functional on both AUROC and delta-cosine in point estimates) is consistent with the Micro-C secondary. Uncertainty: single seed, 22-block CIs, partially related targets, fit-scope and dimension differences, and unrun sensitivities (Section 5).
""")
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    a = ap.parse_args(argv)
    root, doc = a.root.resolve(), a.doc.resolve()

    def write_ok(p):
        assert_writable(p, root, doc)
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    arms = {x: load_arm(root, x) for x in ARMS}
    contrasts = load_contrasts(root)
    checks = sign_check(root, arms, contrasts)
    bad = [c for c in checks if not c["sign_agrees"] or c["max_abs_diff"] > 1e-6]
    if bad:
        raise SystemExit(f"sign/value cross-check failed for {len(bad)} cell(s): {bad}")
    rep = root / "report"
    p1, p2 = rep / "summary_primary.csv", rep / "summary_contrasts.csv"
    write_ok(p1); primary_summary(arms).to_csv(p1, index=False)
    write_ok(p2); contrast_summary(contrasts).to_csv(p2, index=False)
    make_figures(arms, contrasts, rep / "figures", write_ok)
    write_ok(doc)
    doc.write_text(build_markdown(arms, contrasts, checks, root))
    print("wrote", p1, p2, doc)


if __name__ == "__main__":
    main()
