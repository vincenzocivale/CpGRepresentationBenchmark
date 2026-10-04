# ruff: noqa: C408
"""Report-only builder for the biological_validation_v2 follow-up results (Part 0 recap, Part A sensitivities, Part B exploratory).

Reads ALREADY COMPUTED files only (nothing is recomputed; the only operations are copying, formatting, sorting and the registered
sign-based verdict classification of already stored deltas / CIs):
  <root>/<arm>/endpoints/*.json, <root>/contrasts/*.csv, <root>/report/summary_primary.csv      (frozen primary, READ ONLY)
  <root>/sensitivity/**                                                                           (Part A)
  <root>/exploratory_posthoc_loyfer/**                                                            (Part B)
Writes ONLY  <root>/exploratory_posthoc_loyfer/figures_followup/*.png  and  docs/BIOLOGICAL_VALIDATION_V2_SENSITIVITY_AND_EXPLORATORY_RESULTS.md.
Refuses every frozen path. Imports no freeze / evaluation / follow-up-run module.
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
DEFAULT_DOC = REPO / "docs" / "BIOLOGICAL_VALIDATION_V2_SENSITIVITY_AND_EXPLORATORY_RESULTS.md"

REF = "regulatory_histone_dnase_v1"
FUN, CPG, DEE = "functional_annotations_pca", "cpgpt_large_locus", "deepcpg_dna_locus"
COMPARATORS = [FUN, CPG, DEE]
ARMS = [REF, *COMPARATORS]
SEQ = [CPG, DEE]
SHORT = {REF: "Regulatory", FUN: "Functional PCA", CPG: "CpGPT-large", DEE: "DeepCpG-DNA"}
COLORS = {REF: "#D55E00", FUN: "#0072B2", CPG: "#009E73", DEE: "#CC79A7"}  # identical to scripts/bioval_v2/make_report.py
GEOMS = ["cosine", "centered_cosine", "pc1_removed_cosine", "neg_euclidean"]
GEOM_LABEL = {"cosine": "cosine (primary)", "centered_cosine": "centered cosine", "pc1_removed_cosine": "PC1-removed cosine",
              "neg_euclidean": "negative Euclidean"}
VCELLS = ["V_Q1", "V_Q2", "V_Q3", "V_Q4", "V_top10pct"]
MCELLS = ["M_Q1", "M_Q2", "M_Q3", "M_Q4"]
FROZEN_PREFIXES = [
    "docs/BIOLOGICAL_VALIDATION_V2.md", "docs/bioval_v2_prep", "configs/biological_validation_v2", "data/derived/bioval_v2",
    "docs/BIOLOGICAL_VALIDATION_V2_LAUNCH_REGISTRATION.md", "docs/BIOLOGICAL_VALIDATION_V2_EXECUTION_AUDIT.md",
    "docs/BIOLOGICAL_VALIDATION_V2_RESULTS.md", "docs/BIOLOGICAL_VALIDATION_V2_SENSITIVITY_AND_EXPLORATORY_REGISTRATION.md",
    "docs/BIOLOGICAL_VALIDATION_V2_EXPLORATORY_POSTHOC_PROVENANCE_AUDIT.md",
    "src/cpg_repr_benchmark/biological_validation_v2", "src/cpg_repr_benchmark/bioval_v2_launch",
    "src/cpg_repr_benchmark/bioval_v2_followup", "scripts/bioval_v2/run_evaluation.py", "scripts/bioval_v2/run_followup.py",
]


# ----------------------------------------------------------------------------------------------- safety
def assert_writable(path: Path, root: Path, doc: Path, repo: Path = REPO) -> None:
    """Allow only <root>/exploratory_posthoc_loyfer/figures_followup/** and the follow-up results doc; refuse every frozen path."""
    p = Path(path).resolve()
    if p == Path(doc).resolve():
        return
    for fp in FROZEN_PREFIXES:
        f = (repo / fp).resolve()
        if p == f or f in p.parents:
            raise PermissionError(f"refusing to write under frozen path: {p}")
    figdir = (Path(root) / "exploratory_posthoc_loyfer" / "figures_followup").resolve()
    if figdir in p.parents:
        return
    raise PermissionError(f"refusing to write outside exploratory_posthoc_loyfer/figures_followup / follow-up results doc: {p}")


# ----------------------------------------------------------------------------------------------- loading / formatting
def jload(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def fv(v, lo=None, hi=None, nd=3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    s = f"{v:.{nd}f}"
    return s if lo is None else f"{s} [{lo:.{nd}f}, {hi:.{nd}f}]"


def fd(d: dict, nd=3, p=True) -> str:
    s = f"{d['delta']:+.{nd}f} [{d['ci_lo']:+.{nd}f}, {d['ci_hi']:+.{nd}f}]"
    return s + (f" p={d['p']:.3f}" if p and "p" in d else "")


def md_table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def sign_class(prim_delta: float, d: dict) -> str:
    """Registered operational classification of one candidate-vs-comparator delta (sign only; no new statistic)."""
    if np.sign(d["delta"]) == np.sign(prim_delta):
        return "same_sign"
    return "sign_flip_CI_excludes_0" if (d["ci_lo"] > 0 or d["ci_hi"] < 0) else "sign_flip_CI_includes_0"


def verdict_of(classes: list[str]) -> str:
    if any(c == "sign_flip_CI_excludes_0" for c in classes):
        return "QUALITATIVELY_CHANGES"
    if all(c == "same_sign" for c in classes):
        return "CONFIRMS"
    return "INCONCLUSIVE"


class Frozen:
    """Read-only view of the frozen primary results."""

    def __init__(self, root: Path):
        self.root = root
        self.ep = {a: {} for a in ARMS}
        for a in ARMS:
            for f in sorted((root / a / "endpoints").glob("*.json")):
                self.ep[a][f.stem] = jload(f)
        self.con = {}
        for c in COMPARATORS:
            df = pd.read_csv(root / "contrasts" / f"{REF}__vs__{c}.csv")
            self.con[c] = {(r.endpoint, r.stat): dict(delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi, p=r.p) for r in df.itertuples()}

    def stat(self, arm, ep, st):
        return self.ep[arm][ep]["stats"][st]

    def contrast(self, comp, ep, st):
        return self.con[comp][(ep, st)]


# ----------------------------------------------------------------------------------------------- Part A
# (n, axis, protocol analysis, kind, sub-stats [(label, endpoint, stat, ref_endpoint, ref_stat)] , note)
L_REF = ("loyfer_profile", "spearman_pooled_gt1Mb_inter_eqw")
H1_REF = ("microc_H1_intra10kb", "auroc_contact_vs_noncontact")
F5_REF = ("fantom5_membership", "auroc_oof_bg_enh5k")
ACT_REF = ("fantom5_activity_similarity", "spearman_all_pairs")
RT_REF = ("rt_consensus", "spearman_oof_consensus")
LINES = ["BG02ES", "BJ", "GM06990", "GM12801", "GM12812", "GM12813", "GM12878", "HELAS3", "HEPG2", "HUVEC", "IMR90", "K562", "MCF7", "NHEK", "SKNSH"]
EARLIER = {
    1: ("Loyfer", "Pearson vs Spearman", [("profile Spearman", "loyfer_profile", "spearman_pooled_eqw__profile_spearman", *L_REF),
                                           ("Pearson corr", "loyfer_profile", "pearsoncorr_pooled_eqw__profile_pearson", *L_REF)], ""),
    2: ("Loyfer", "per distance stratum", [(s, "loyfer_profile", f"spearman_{s}", *L_REF) for s in ["<1kb", "1-10kb", "10-100kb", "100kb-1Mb", ">1Mb", "interchromosomal"]],
        "Strata are a decomposition of the primary, not a variant of its target; the rule is applied mechanically to each stratum (reference = pooled primary deltas)."),
    3: ("Loyfer", "pairs with < 30 shared groups removed", [("min30", "loyfer_profile", "spearman_pooled_eqw__min30_shared_groups", *L_REF)], ""),
    6: ("Loyfer marker kNN", "marker sets paper S4A/S4B lifted; U25 only", [("paper-lifted U25+U250", "loyfer_marker_knn", "enrichment_paper_lifted_U25_U250", "loyfer_marker_knn", "enrichment_github_U25_U250"),
                                                                           ("github U25 only", "loyfer_marker_knn", "enrichment_github_U25_only", "loyfer_marker_knn", "enrichment_github_U25_U250")],
        "Reference = the secondary github U25+U250 enrichment (its primary delta vs Functional is +2.3 with CI including 0, so the sign reference vs Functional is fragile)."),
    7: ("3D", "HFFc6 Micro-C intra", [("AUROC", "microc_HFFc6_intra10kb", "auroc_contact_vs_noncontact", *H1_REF),
                                      ("delta-cosine", "microc_HFFc6_intra10kb", "delta_cosine_paired_mean", "microc_H1_intra10kb", "delta_cosine_paired_mean")],
        "H1 AUROC delta vs Functional is +0.0008 with CI including 0, so sign agreement there is weakly informative."),
    8: ("3D", "per distance bin (H1)", [(b, "microc_H1_intra10kb", f"auroc_distance_bin_{b}", *H1_REF) for b in ["[10000,100000)", "[100000,1e+06)", "[1e+06,1e+07)"]],
        "The non-confirming bin comes from the Functional contrast only: the primary H1 delta vs Functional is +0.0008 (CI includes 0), so any bin with a slightly negative delta whose CI excludes 0 is mechanically a sign flip; the contrasts vs CpGPT-large and DeepCpG-DNA keep the primary sign (vs CpGPT-large with CI including 0 in this bin)."),
    13: ("FANTOM5", "500 bp window; TSS-filtered pool", [("win500", "fantom5_membership", "auroc_oof__win500", *F5_REF), ("tss5k pool", "fantom5_membership", "auroc_oof__tss5k_pool", *F5_REF)],
         "TSS-filtered pool is poorly balanced (SMD tss_dist -0.117, 3,349 pairs)."),
    18: ("FANTOM5", "activity similarity intra / inter", [("intra", "fantom5_activity_similarity", "spearman_intra", *ACT_REF), ("inter", "fantom5_activity_similarity", "spearman_inter", *ACT_REF)], ""),
    19: ("RT", "per single line (15)", [(l, "rt_consensus", f"spearman_oof_line_{l}", *RT_REF) for l in LINES], "Per-line Spearman; reference = consensus RT primary."),
}
DONE_NOW = {  # SUMMARY item -> (row n, axis, analysis, stat key in the sensitivity endpoint json)
    "loyfer_mask_lenient": (4, "Loyfer", "mask_lenient (>=1 valid donor)", "spearman_pooled_gt1Mb_inter_eqw"),
    "loyfer_drop_single_sample_groups": (5, "Loyfer", "single-sample groups excluded (4 groups)", "spearman_pooled_gt1Mb_inter_eqw"),
    "compartment_E1_probe_H1_50kb": (9, "3D", "E1 compartments at 50 kb, H1", "spearman_E1_vs_oof_pred"),
    "compartment_E1_probe_GM12878_50kb": (9, "3D", "E1 compartments at 50 kb, GM12878", "spearman_E1_vs_oof_pred"),
    "fantom5_membership_N1": (14, "FANTOM5", "expression threshold N = 1 library", "auroc_oof__N1"),
    "fantom5_membership_N3": (14, "FANTOM5", "expression threshold N = 3", "auroc_oof__N3"),
    "fantom5_membership_N10": (14, "FANTOM5", "expression threshold N = 10", "auroc_oof__N10"),
    "fantom5_membership_N20": (14, "FANTOM5", "expression threshold N = 20", "auroc_oof__N20"),
    "fantom5_activity_sample_level": (16, "FANTOM5", "sample level (1,829 libraries) vs collapsed", "spearman_all_pairs"),
    "fantom5_activity_cat_cell_lines": (17, "FANTOM5", "category: cell lines", "spearman_all_pairs"),
    "fantom5_activity_cat_primary_cells": (17, "FANTOM5", "category: primary cells", "spearman_all_pairs"),
    "fantom5_activity_cat_tissues": (17, "FANTOM5", "category: tissues", "spearman_all_pairs"),
}
NOT_RUNNABLE = {
    10: ("3D", "K per bin pair", "needs `freeze_4dn_pairs` re-projection with K != 20; the protocol names no alternative K. Decision D-A3 (K values + permission to run the freeze command) not given; command NOT executed."),
    11: ("3D", "distance-only / covariate baseline comparison", "protocol text 'baseline di sola distanza/covariate (dichiarato)' leaves score, model and covariate set undefined (distance-matched pairs give ~0.5 by construction). Decision D-A5 (definition of the baseline) not given; NOT executed."),
    15: ("FANTOM5", "additional matching on chromatin covariates (post hoc)", "needs a new matching (`freeze_fantom5_matching`) with chromatin covariates that are not specified. Decision D-A4 (covariates + permission for a new matching) not given; command NOT executed."),
    20: ("RT", "partial correlation / residualisation for GC, CpG-island density, gene/TSS density", "covariate table is frozen (`covariates/universe_covariates.parquet`) but the residualisation model, which variable is residualised, fit scope and 'gene density' (only `tss_dist` exists) are undefined. Decision D-A6 (residualisation specification) not given; NOT executed."),
    21: ("RT", "normal vs tumour lines only", "the normal/tumour assignment of the 15 UW lines is not frozen (hESC, LCLs, K562, SK-N-SH, HeLa-S3 ...) and the consensus would need refits on subsets. Decision D-A7 (frozen assignment) not given; NOT executed."),
}
PMD_ROW = (22, "RT / PMD", "PMD probe (exploratory)", [("PMD membership AUROC", "pmd_probe", "auroc_oof_pmd_in_any", None, None)], "Different target (PMD, methylation-derived) with no registered primary reference: verdict rule not applicable.")
E1_ROW = (12, "3D", "E1 probe 100 kb H1 / GM12878; delta-cosine (secondary endpoints)",
          [("H1 E1 100kb Spearman", "compartment_E1_probe_H1_100kb", "spearman_E1_vs_oof_pred", None, None),
           ("GM12878 E1 100kb Spearman", "compartment_E1_probe_GM12878_100kb", "spearman_E1_vs_oof_pred", None, None),
           ("H1 delta-cosine", "microc_H1_intra10kb", "delta_cosine_paired_mean", None, None)],
          "These are the secondary endpoints themselves (they are the references of rows 7-9): verdict rule not applicable.")


def order_by(vals: dict) -> list[str]:
    return sorted(vals, key=lambda a: -vals[a])


def done_earlier_row(fz: Frozen, n, spec):
    axis, analysis, subs, note = spec[0], spec[1], spec[2], spec[3]
    vals, cons, classes, ord_ident = [], [], [], 0
    for lab, ep, st, rep, rst in subs:
        v = {a: fz.stat(a, ep, st) for a in ARMS}
        c = {k: fz.contrast(k, ep, st) for k in COMPARATORS}
        vals.append((lab, v)); cons.append((lab, c))
        if rep:
            cl = [sign_class(fz.contrast(k, rep, rst)["delta"], c[k]) for k in COMPARATORS]
            classes.append((lab, cl))
            pv = {a: fz.stat(a, rep, rst)["value"] for a in ARMS}
            ord_ident += order_by({a: v[a]["value"] for a in ARMS}) == order_by(pv)
    return axis, analysis, vals, cons, classes, ord_ident, note


def fmt_values(vals, nd=3, with_ci=True, maxlines=None):
    lines = []
    for lab, v in vals:
        if with_ci:
            body = "; ".join(f"{SHORT[a][:4]} {fv(v[a]['value'], v[a].get('ci_lo'), v[a].get('ci_hi'), nd)}" for a in ARMS)
        else:
            body = "; ".join(f"{SHORT[a][:4]} {fv(v[a]['value'], nd=nd)}" for a in ARMS)
        lines.append((f"{lab}: " if len(vals) > 1 else "") + body)
    return "<br>".join(lines)


def partA_rows(fz: Frozen, root: Path):
    summ = pd.read_csv(root / "sensitivity" / "SUMMARY.csv")
    done_now = {r.item: r for r in summ.itertuples()}
    rows = {}
    verdicts = {}
    # earlier / frozen primary launch
    for n, spec in EARLIER.items():
        axis, analysis, vals, cons, classes, oi, note = done_earlier_row(fz, n, spec)
        if n == 19:  # compact: range over the 15 lines
            rng = []
            for a in ARMS:
                vs = [v[a]["value"] for _, v in vals]
                rng.append(f"{SHORT[a][:4]} {min(vs):.3f} to {max(vs):.3f}")
            res = "per-line range: " + "; ".join(rng) + " (15 lines, table in the primary results doc section 4.2)"
            ct = []
            for k in COMPARATORS:
                ds = [c[k]["delta"] for _, c in cons]
                ct.append(f"vs {SHORT[k][:4]}: delta {min(ds):+.3f} to {max(ds):+.3f}, CI excludes 0 in {sum(1 for _, c in cons if c[k]['ci_lo'] > 0 or c[k]['ci_hi'] < 0)}/15 lines")
            con = "<br>".join(ct)
        else:
            res = fmt_values(vals, with_ci=(len(vals) <= 2))
            con = "<br>".join(((f"{lab}: " if len(cons) > 1 else "") + "; ".join(f"vs {SHORT[k][:4]} {fd(c[k])}" for k in COMPARATORS)) for lab, c in cons)
        vs = [verdict_of(cl) for _, cl in classes]
        v = verdict_of([x for _, cl in classes for x in cl])
        if any(x != "CONFIRMS" for x in vs):
            bad = [lab for (lab, _), x in zip(classes, vs) if x != "CONFIRMS"]
            vtxt = f"**{v}** (mechanical rule; non-confirming sub-statistic(s): {', '.join(bad)}; other {len(vs) - len(bad)}/{len(vs)} CONFIRMS)"
        else:
            vtxt = f"**{v}**"
        vtxt += f"; arm ordering identical to primary in {oi}/{len(classes)} sub-statistic(s)"
        rows[n] = [n, f"{axis}: {analysis}", "DONE earlier (frozen primary launch; values copied from the frozen endpoint/contrast files)", res, con, vtxt, note]
        verdicts[n] = v
    for n, spec in ((12, E1_ROW), (22, PMD_ROW)):
        axis, analysis, subs, note = spec[1], spec[2], spec[3], spec[4]
        vals = [(lab, {a: fz.stat(a, ep, st) for a in ARMS}) for lab, ep, st, _, _ in subs]
        cons = [(lab, {k: fz.contrast(k, ep, st) for k in COMPARATORS}) for lab, ep, st, _, _ in subs]
        res = fmt_values(vals, with_ci=(len(vals) <= 1))
        con = "<br>".join(((f"{lab}: " if len(cons) > 1 else "") + "; ".join(f"vs {SHORT[k][:4]} {fd(c[k])}" for k in COMPARATORS)) for lab, c in cons)
        rows[n] = [n, f"{axis}: {analysis}", "DONE earlier (frozen primary launch)", res, con, "n/a (rule not applicable)", note]
        verdicts[n] = "NA"
    # done now
    for item, (n, axis, analysis, key) in DONE_NOW.items():
        r = done_now[item]
        vals = []
        v = {}
        for a in ARMS:
            s = jload(root / "sensitivity" / a / "endpoints" / f"{item}.json")["stats"][key]
            v[a] = s
        vals.append((item, v))
        sc = json.loads(r.sensitivity_contrasts)
        con = "; ".join(f"vs {SHORT[k][:4]} {fd(sc[k])}" for k in COMPARATORS)
        pc = json.loads(r.per_contrast)
        nn = json.loads(r.n_pairs_or_rows)[REF]
        ordtxt = " > ".join(SHORT[a][:4] for a in json.loads(r.ordering_sensitivity)[::-1])
        same = sum(1 for k in COMPARATORS if pc[k] == "same_sign")
        vtxt = f"**{r.verdict}** ({same}/3 contrasts same_sign); ordering {ordtxt}; identical to primary ordering: {r.ordering_identical}"
        cav = f"n = {nn:,} pairs/rows."
        tag = analysis
        rows.setdefault(n, []).append([n, f"{axis}: {analysis}", "DONE NOW (follow-up run, `sensitivity/`)", f"{tag}: " + fmt_values(vals), f"{tag}: " + con, f"{tag}: " + vtxt, cav])
        verdicts.setdefault(("now", n), []).append(r.verdict)
    # merge multi-item rows (9, 14, 17)
    merged = {}
    for n, v in list(rows.items()):
        if v and isinstance(v[0], list):
            first = v[0]
            m = [n, f"{first[1].split(':')[0]}: " + {9: "E1 compartments at 50 kb (H1, GM12878)", 14: "FANTOM5 expression thresholds N = 1/3/10/20 libraries (membership subset of frozen pairs)",
                                                     16: "FANTOM5 sample level vs collapsed", 17: "FANTOM5 activity per category (cell lines / primary cells / tissues)",
                                                     4: "mask_lenient (>= 1 valid donor)", 5: "single-sample groups excluded (4 groups)"}[n],
                 first[2], "<br>".join(x[3] for x in v), "<br>".join(x[4] for x in v), "<br>".join(x[5] for x in v), "<br>".join(x[6] for x in v)]
            merged[n] = m
        else:
            merged[n] = v
    for n, (axis, analysis, why) in NOT_RUNNABLE.items():
        merged[n] = [n, f"{axis}: {analysis}", f"NOT RUNNABLE: {why.split('. Decision')[0]}", "-", "-", "n/a", "Decision" + why.split(". Decision", 1)[1] if ". Decision" in why else why]
    # collect verdict per protocol row
    final = {}
    for n in range(1, 23):
        if n in NOT_RUNNABLE:
            final[n] = "NOT_RUNNABLE"
        elif ("now", n) in verdicts:
            vs = verdicts[("now", n)]
            final[n] = "QUALITATIVELY_CHANGES" if "QUALITATIVELY_CHANGES" in vs else ("CONFIRMS" if all(x == "CONFIRMS" for x in vs) else "INCONCLUSIVE")
        else:
            final[n] = verdicts[n]
    return [merged[n] for n in range(1, 23)], final


# ----------------------------------------------------------------------------------------------- Part B loading
class Expl:
    def __init__(self, root: Path):
        self.d = root / "exploratory_posthoc_loyfer"
        self.sim = {a: jload(self.d / a / "endpoints" / "exploratory_similarity.json") for a in ARMS}
        self.rdg = {a: jload(self.d / a / "endpoints" / "B3_profile_ridge.json") for a in ARMS}
        self.abc = jload(self.d / "interpretation_ABC.json")
        self.rank = pd.read_csv(self.d / "similarity_ranking_by_cell.csv")
        self.geo = pd.read_csv(self.d / "geometry_within_arm_deltas.csv")
        self.fvs = pd.read_csv(self.d / "ridge_functional_vs_sequence.csv")
        self.scon = {c: pd.read_csv(self.d / "contrasts" / f"similarity__{REF}__vs__{c}.csv").set_index("stat") for c in COMPARATORS}
        self.rcon = {c: pd.read_csv(self.d / "contrasts" / f"ridge__{REF}__vs__{c}.csv") for c in COMPARATORS}

    def S(self, arm, geom, cell, fam="spearman_pooled_eqw"):
        return self.sim[arm]["stats"][f"{fam}__{geom}__{cell}"]

    def scon_row(self, comp, geom, cell, fam="spearman_pooled_eqw"):
        r = self.scon[comp].loc[f"{fam}__{geom}__{cell}"]
        return dict(delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi, p=r.p)

    def R(self, arm, target, subset):
        return self.rdg[arm]["targets"][target]["subsets"][subset]["model"]

    def Rbase(self, arm, target, subset):
        return self.rdg[arm]["targets"][target]["subsets"][subset]["train_mean_baseline"]

    def rcon_row(self, comp, target, subset, metric, group=None):
        df = self.rcon[comp]
        m = (df.target == target) & (df.subset == subset) & (df.metric == metric)
        m &= df.group.isna() if group is None else (df.group == group)
        r = df[m].iloc[0]
        return dict(delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi, p=r.p)


def _pm(m: dict) -> str:
    ci = m.get("pearson_mean_over_cpgs_ci")
    return fv(m["pearson_mean_over_cpgs"], *ci) if ci else fv(m["pearson_mean_over_cpgs"])


def excl(d: dict) -> str:
    return "above (CI>0)" if d["ci_lo"] > 0 else "below (CI<0)" if d["ci_hi"] < 0 else "CI includes 0"


# ----------------------------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150})
    return plt


def make_figures(ex: Expl, figdir: Path, write) -> list[Path]:
    plt = _plt()
    paths = []

    def save(fig, name):
        p = figdir / name
        write(p)
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(p)

    # (1) Loyfer ranking across variance quartiles (pooled primary-style statistic, cosine)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = np.arange(len(VCELLS))
    for i, a in enumerate(ARMS):
        v = np.array([ex.S(a, "cosine", c)["value"] for c in VCELLS])
        lo = np.array([ex.S(a, "cosine", c)["ci_lo"] for c in VCELLS])
        hi = np.array([ex.S(a, "cosine", c)["ci_hi"] for c in VCELLS])
        off = (i - 1.5) * 0.09
        ax.errorbar(xs + off, v, yerr=[v - lo, hi - v], fmt="o-", color=COLORS[a], lw=1.2, ms=5, capsize=2, label=SHORT[a])
    ax.axhline(0, color="#666", lw=1)
    ax.set_xticks(xs, ["V_Q1\n(least var.)", "V_Q2", "V_Q3", "V_Q4\n(most var.)", "V_top10%"])
    ax.set_ylabel("pooled Spearman (>1 Mb + inter), cosine")
    ax.set_title("EXPLORATORY POST HOC: Loyfer ranking by min-variance stratum\n(primary-style statistic within each cell; 95% chromosome-block CI)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, "fig1_loyfer_variance_strata.png")

    # (2) geometry diagnostics
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    xs = np.arange(len(GEOMS))
    for i, a in enumerate(ARMS):
        st = [ex.S(a, g, "all") for g in GEOMS]
        v = np.array([s["value"] for s in st]); lo = np.array([s["ci_lo"] for s in st]); hi = np.array([s["ci_hi"] for s in st])
        ax.errorbar(xs + (i - 1.5) * 0.12, v, yerr=[v - lo, hi - v], fmt="o", color=COLORS[a], ms=6, capsize=2, label=SHORT[a])
    ax.axhline(0, color="#666", lw=1)
    ax.set_xticks(xs, [GEOM_LABEL[g] for g in GEOMS], fontsize=9)
    ax.set_ylabel("pooled Spearman (>1 Mb + inter)")
    ax.set_title("EXPLORATORY POST HOC: Loyfer pair similarity under four geometries\n(same 384,675 pairs; primary-style pooled statistic)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, "fig2_geometry_diagnostics.png")

    # (3) ridge R2 T2 variable CpGs + per-group heatmap
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 6.2), gridspec_kw={"width_ratios": [1, 1.25]})
    for i, a in enumerate(ARMS):
        m = ex.R(a, "centered", "variable_cpgs")
        lo, hi = m["r2_global_ci"]
        ax.errorbar(m["r2_global"], len(ARMS) - 1 - i, xerr=[[m["r2_global"] - lo], [hi - m["r2_global"]]], fmt="o", color=COLORS[a], ms=7, capsize=3)
        ax.text(m["r2_global"], len(ARMS) - 1 - i + 0.25, f"{m['r2_global']:.3f}", ha="center", fontsize=8, color="#333")
    ax.axvline(0, color="#666", lw=1); ax.axvline(0.02, color="#888", lw=1, ls="--"); ax.set_ylim(-0.6, 3.8)
    ax.set_yticks(range(len(ARMS)), [SHORT[a] for a in ARMS][::-1])
    ax.set_xlabel("global out-of-fold R2 (T2 centred profile, variable CpGs)")
    ax.set_title("Ridge R2 (95% CI); dashed = registered 0.02 near-baseline threshold", fontsize=9)
    groups = ex.rdg[REF]["groups"]
    mat = np.array([ex.R(a, "centered", "variable_cpgs")["r2_per_group"] for a in ARMS]).T
    order = np.argsort(-mat[:, 0])
    im = ax2.imshow(mat[order], aspect="auto", cmap="viridis", vmin=0, vmax=max(0.5, np.nanmax(mat)))
    ax2.set_xticks(range(len(ARMS)), [SHORT[a] for a in ARMS], rotation=20, fontsize=8)
    ax2.set_yticks(range(len(groups)), [groups[i] for i in order], fontsize=6.5)
    ax2.grid(False)
    fig.colorbar(im, ax=ax2, fraction=0.04, label="per-group R2")
    ax2.set_title("Per-cell-group R2 (39 groups, sorted by Regulatory)", fontsize=9)
    fig.suptitle("EXPLORATORY POST HOC: B3 profile ridge, T2 centred target, variable CpGs", fontsize=10)
    save(fig, "fig3_ridge_r2_and_per_group.png")

    # (4) cosine vs ridge dissociation
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, geom in zip(axes, ["cosine", "centered_cosine"]):
        for a in ARMS:
            s = ex.S(a, geom, "all"); m = ex.R(a, "centered", "variable_cpgs"); lo, hi = m["r2_global_ci"]
            ax.errorbar(s["value"], m["r2_global"], xerr=[[s["value"] - s["ci_lo"]], [s["ci_hi"] - s["value"]]], yerr=[[m["r2_global"] - lo], [hi - m["r2_global"]]],
                        fmt="o", color=COLORS[a], ms=8, capsize=2, label=SHORT[a])
        ax.axvline(0, color="#666", lw=1); ax.axhline(0.02, color="#888", lw=1, ls="--")
        ax.set_xlabel(f"Loyfer pooled Spearman, {GEOM_LABEL[geom]}")
    axes[0].set_ylabel("ridge global R2 (T2, variable CpGs)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("EXPLORATORY POST HOC: pair-similarity (cosine) vs linear decodability of the profile (ridge)\nfour points per panel, one per arm; not a causal analysis", fontsize=10)
    save(fig, "fig4_cosine_vs_ridge_dissociation.png")
    return paths


# ----------------------------------------------------------------------------------------------- markdown
def build_markdown(fz: Frozen, ex: Expl, root: Path, figrel: str, summ_primary: pd.DataFrame) -> str:
    L = []
    A = L.append
    abc = ex.abc
    rd = abc["readings"]
    pr = rd["centered/variable_cpgs"]
    A("# biological_validation_v2: SENSITIVITY and EXPLORATORY POST-HOC RESULTS (report only)\n")
    A("Generated by `scripts/bioval_v2/make_followup_report.py` from already-computed outputs. **Nothing was recomputed**: values, CIs and raw p-values are copied from the stored files; "
      "the only operations are formatting and the registered sign-based verdict classification of stored deltas. No Holm adjustment is recomputed or extended; every statistic below is descriptive (raw bootstrap p, floor 0.002). "
      "No aggregate score is built. The TCGA test set is not touched. Frozen primary fingerprint (gate): `38b9da12700a8fc83dc08f5a01aaa76b61f0d5097bffe395d8eb5225f1f02094`.\n")
    A("Registration: `docs/BIOLOGICAL_VALIDATION_V2_SENSITIVITY_AND_EXPLORATORY_REGISTRATION.md`. Provenance audit: `docs/BIOLOGICAL_VALIDATION_V2_EXPLORATORY_POSTHOC_PROVENANCE_AUDIT.md`. "
      "Sign convention everywhere: delta = Regulatory minus comparator, higher-is-better. Arms: Regulatory = `regulatory_histone_dnase_v1` (candidate), Functional PCA, CpGPT-large, DeepCpG-DNA (the last two = SEQ).\n")
    A("The three parts below are strictly separated and carry different evidential status: PART 0 frozen primary (unchanged), PART A preregistered sensitivities (descriptive), PART B exploratory post hoc (conceived after seeing the primary results).\n")

    # ---------------- PART 0
    A("\n---\n\n# PART 0 - FROZEN PRIMARY RESULTS (recap only; unchanged; see `docs/BIOLOGICAL_VALIDATION_V2_RESULTS.md`)\n")
    A("Copied from `outputs/.../report/summary_primary.csv` (not recomputed). Value [95% chromosome-block bootstrap CI].\n")
    names = {"loyfer_profile": "Loyfer WGBS profile similarity (pooled Spearman, >1 Mb + inter)", "microc_H1_intra10kb": "H1 Micro-C intra 10 kb (AUROC)",
             "fantom5_membership": "FANTOM5 enhancer membership (AUROC)", "rt_consensus": "Replication timing consensus (Spearman)"}
    rows = []
    for ep, nm in names.items():
        r = [nm]
        for a in ARMS:
            s = summ_primary[(summ_primary.endpoint == ep) & (summ_primary.arm == a)].iloc[0]
            r.append(fv(s.value, s.ci_lo, s.ci_hi, 4))
        rows.append(r)
    A(md_table(["Primary endpoint", *[SHORT[a] for a in ARMS]], rows))
    A("\nThe frozen reading stands unchanged: on the Loyfer primary the candidate is the lowest arm and is below the CpGRepresentation sequence models (CpGPT-large 0.107, DeepCpG-DNA 0.116) and slightly below Functional PCA (-0.038); "
      "on Micro-C, FANTOM5 and RT it is above both sequence models and equal (Micro-C) or slightly below (FANTOM5, RT) Functional PCA. Nothing in Parts A or B amends, replaces or re-weights these results.\n")

    # ---------------- PART A
    rowsA, final = partA_rows(fz, root)
    cnt = {k: sum(1 for v in final.values() if v == k) for k in ["CONFIRMS", "QUALITATIVELY_CHANGES", "INCONCLUSIVE", "NOT_RUNNABLE", "NA"]}
    A("\n---\n\n# PART A - PREREGISTERED SENSITIVITY RESULTS (descriptive; raw p only; no Holm; no claim)\n")
    A("Complete list of the 22 protocol sensitivity / secondary analyses of the registration (section 1). Verdict rule (registered, section 2.4): reference = frozen primary statistic of the same family and its three frozen candidate-vs-comparator deltas; "
      "**CONFIRMS** = all 3 contrasts keep the primary sign; **QUALITATIVELY_CHANGES** = at least one sign flip whose sensitivity CI excludes 0; otherwise **INCONCLUSIVE**. "
      "Rows done at the primary launch were classified with the same rule *retrospectively and mechanically* from the stored frozen contrasts (this retrospective application is a report-time step, not a registered output; where a row has several sub-statistics the worst case is shown and the sub-statistics are named). "
      "Rows whose statistic is itself the reference (row 12) or that have no registered reference family (row 22) get no verdict. Values in cells: arms abbreviated Regu / Func / CpGP / Deep.\n")
    A(f"**Counts over the 22 protocol rows: CONFIRMS {cnt['CONFIRMS']}, QUALITATIVELY_CHANGES {cnt['QUALITATIVELY_CHANGES']}, INCONCLUSIVE {cnt['INCONCLUSIVE']}, verdict not applicable {cnt['NA']}, NOT RUNNABLE {cnt['NOT_RUNNABLE']}.** "
      "Of these 22 rows, 11 were done at the primary launch, 6 protocol rows (12 follow-up items: rows 4, 5, 9, 14, 16, 17) were run now, and 5 are not runnable. "
      "Among the 12 items run now (`sensitivity/SUMMARY.csv`): 12 CONFIRMS, 0 QUALITATIVELY_CHANGES, 0 INCONCLUSIVE; the arm ordering is identical to the primary ordering in all 12.\n")
    A(md_table(["#", "Protocol-specified analysis", "Status", "Result per arm (value [CI] where shown)", "Candidate vs comparator (delta reg - comp [CI], raw p; descriptive)", "Verdict (registered rule) and arm ordering", "Caveats"], rowsA))
    A("\n**Decisions needing the user (not taken by anyone but flagged here).**\n")
    A("- **D-A1 (taken as 'yes' by the orchestrator, FLAG FOR USER REVIEW).** In-memory re-derivation of the Loyfer `mask_lenient` target and of the single-sample-group exclusion from the FROZEN files (`loyfer_celltype_beta.h5`, frozen pair list). "
      "It re-implements (never runs) the frozen target code, writes nothing to `data/derived`, and the kernel reproduced the frozen `loyfer_pearson` / `n_shared_groups` of all 386,268 pairs exactly (max abs diff 8.9e-16, asserted at run time). "
      "The orchestrator interpreted this as read-only-from-frozen, consistent with the user's instruction to 'keep matching and stores unchanged'. The user has not explicitly approved this reading.")
    A("- **D-A2 (taken as 'yes' by the orchestrator, FLAG FOR USER REVIEW).** FANTOM5 N-threshold sensitivities (N = 1/3/10/20 libraries) as a SUBSET of the frozen pairs (positives expressed in >= N libraries together with THEIR FROZEN matched controls), without re-matching. "
      "Same orchestrator interpretation (read-only-from-frozen, 'keep matching unchanged'); numeric covariate balance of the subsets was not re-verified; the PREP defines N for the activity secondary while the yaml lists it under the membership primary (membership reading used). Not explicitly approved by the user.")
    A("- **D-A3, D-A4, D-A5, D-A6, D-A7: NOT executed** (rows 10, 15, 11, 20, 21): each requires a new frozen artifact, a freeze command (`freeze_4dn_pairs`, `freeze_fantom5_matching`) or an analysis specification that the protocol does not provide. No freeze/prepare/build command was run or imported.\n")
    A("Reading of Part A (descriptive): every sensitivity actually run now leaves the primary signs and the arm ordering unchanged (Loyfer deltas vs Functional -0.025, vs CpGPT -0.17 to -0.18, vs DeepCpG -0.18 to -0.19 under mask_lenient and single-group exclusion; "
      "FANTOM5 membership AUROC delta vs CpGPT +0.13 to +0.15 and vs DeepCpG +0.21 to +0.24 for N = 1..20; E1 at 50 kb and FANTOM5 activity variants likewise). Among the rows done at the primary launch, classified retrospectively and mechanically, three are not CONFIRMS: row 2 (the <1 kb Loyfer stratum flips sign vs CpGPT-large and DeepCpG-DNA; strata are a proximity-dominated decomposition of the primary, not a target variant), "
      "row 6 (INCONCLUSIVE: github U25-only enrichment vs Functional flips sign with CI including 0, the reference delta being +2.3 with CI including 0) and row 8 (the 1-10 Mb H1 bin vs Functional, whose primary reference delta is +0.0008 with CI including 0). "
      "Caveats: the sensitivities share the same arms, folds, bootstrap draw table and single seed as the primary, so they are variants of the same experiment and not independent replications; "
      "the sign rule is weak when the primary delta is near zero (Micro-C and Functional, +0.0008, CI includes 0); 5 of 22 protocol rows remain not runnable.\n")

    # ---------------- PART B
    A("\n---\n\n# PART B - EXPLORATORY POST HOC (`exploratory_posthoc_loyfer`)\n")
    A("**EXPLORATORY POST HOC. Conceived AFTER seeing the primary results** (candidate clearly below CpGPT-large and DeepCpG-DNA and slightly below Functional PCA on the Loyfer primary). It **cannot replace the frozen Loyfer primary**, "
      "designates no new primary, enters no Holm family, adds no data, retrains or creates no representation and uses no TCGA data. One seed (17), 22 chromosome blocks, B = 1000, intra-chromosome folds are the frozen chromosome-blocked folds. "
      "Definitions (strata edges, geometries, ridge protocol, thresholds, A/B/C rules) were registered before any embedding statistic was computed. All tables below are copied from `exploratory_posthoc_loyfer/`.\n")

    # B1
    A("\n## B1. Strata of the 386,268 frozen Loyfer pairs (384,675 after the D2 zero-locus exclusion)\n")
    A("V = min over the two loci of the per-CpG variance of the beta profile; V_Q1..V_Q4 are quartiles of V of the frozen pairs (edges 0.000369, 0.001334, 0.008412; V_top10pct = V >= 0.026695); M = mean of the two per-CpG mean betas (edges 0.2654, 0.4667, 0.6981). "
      "The registered ranking statistic is the **primary-style pooled statistic** (weighted Spearman over >1 Mb + interchromosomal pairs inside the cell, weight 1/n_stratum). Unweighted Spearman over all pairs of a cell mixes in short-distance pairs and is shown only for completeness. "
      "Pair counts per cell: V_Q1..Q4 96,338 / 96,244 / 95,920 / 96,173; V_top10pct 38,555; M_Q1..Q4 96,535 / 96,263 / 96,016 / 95,861.\n")
    A("**Pooled primary-style statistic per cell (cosine, as in the primary), value [95% CI]:**\n")
    rows = []
    for c in ["all", *VCELLS, *MCELLS]:
        r = [c]
        vals = {a: ex.S(a, "cosine", c)["value"] for a in ARMS}
        for a in ARMS:
            s = ex.S(a, "cosine", c)
            r.append(fv(s["value"], s["ci_lo"], s["ci_hi"]))
        order = order_by(vals)
        rk = ex.rank[ex.rank.stat == f"spearman_pooled_eqw__cosine__{c}"].iloc[0]
        r.append(" > ".join(SHORT[a].split()[0] for a in order))
        r.append("yes" if bool(rk.same_ordering_as_frozen_primary) else "no")
        r.append(str(int(rk.candidate_rank)))
        rows.append(r)
    A(md_table(["Cell", *[SHORT[a] for a in ARMS], "Ordering (best first)", "Same ordering as frozen primary", "Candidate rank"], rows))
    A("\n**Candidate minus comparator, pooled primary-style statistic (cosine), delta [CI] (raw p, descriptive):**\n")
    rows = []
    for c in ["all", "V_Q3", "V_Q4", "V_top10pct"]:
        rows.append([c, *[f"{fd(ex.scon_row(k, 'cosine', c))}" for k in COMPARATORS]])
    A(md_table(["Cell", "vs Functional PCA", "vs CpGPT-large", "vs DeepCpG-DNA"], rows))
    A("\n**Unweighted Spearman over all pairs of the cell (cosine; descriptive; includes short-distance pairs), value:**\n")
    rows = []
    for c in ["all", "V_Q4", "V_top10pct", "chrom_intra", "chrom_inter"]:
        rk = ex.rank[ex.rank.stat == f"spearman__cosine__{c}"].iloc[0]
        rows.append([c, *[fv(ex.S(a, 'cosine', c, 'spearman')['value']) for a in ARMS], rk.ordering.replace("regulatory_histone_dnase_v1", "Regulatory").replace("functional_annotations_pca", "Functional").replace("cpgpt_large_locus", "CpGPT").replace("deepcpg_dna_locus", "DeepCpG").replace(">", " > ")])
    A(md_table(["Cell", *[SHORT[a] for a in ARMS], "Ordering"], rows))
    vq4 = {a: ex.S(a, "cosine", "V_Q4")["value"] for a in ARMS}
    vt = {a: ex.S(a, "cosine", "V_top10pct")["value"] for a in ARMS}
    A("\n**Does the arm ranking change in truly variable-methylation pairs?** "
      f"On the registered pooled statistic the candidate remains the lowest arm in V_Q4 (ordering {' > '.join(SHORT[a].split()[0] for a in order_by(vq4))}; candidate {vq4[REF]:.3f}, CI includes 0) and is third of four in V_top10pct "
      f"(ordering {' > '.join(SHORT[a].split()[0] for a in order_by(vt))}; candidate {vt[REF]:.3f}, DeepCpG-DNA drops to last). The ordering differs from the frozen primary ordering in V_Q3, V_Q4 and V_top10pct, but the candidate stays numerically below CpGPT-large in every variance cell (V_Q1 differences are not distinguishable from 0) and below Functional PCA in V_Q2..V_top10pct. "
      "So the sequence-model advantage on the primary-style statistic is NOT confined to low-variance pairs: it is present in the more variable cells (CpGPT-large leads in V_Q3, V_Q4 and V_top10pct), "
      "whereas DeepCpG-DNA's advantage shrinks in V_Q4 (candidate minus DeepCpG -0.062) and reverses in V_top10pct (DeepCpG 0.041 vs candidate 0.068; delta +0.026 [+0.007, +0.048], see table). Every cell's statistic is small in absolute value in V_Q1 (all |values| <= 0.016: the statistic is uninformative there). "
      "With the unweighted statistic (mixing in near pairs, which is dominated by genomic proximity) the picture is different: in V_Q4 and V_top10pct the ordering is Functional > Regulatory > CpGPT > DeepCpG, i.e. the candidate is second; this statistic is not the registered ranking statistic and is shown for transparency only. "
      "The ranking therefore depends on the statistic (pooled long-range vs all-pairs) and no statement beyond that is made.\n")
    A("Mean-quartile and chromosome strata: see the ranking rows above; the pooled candidate value is lowest in M_Q1, M_Q2, M_Q3 and M_Q4 (candidate below CpGPT-large in every M cell).\n")

    # B2
    A("\n## B2. Geometry diagnostics (same 384,675 pairs; four similarities)\n")
    A("`centered_cosine`: cosine(e - mu), mu = mean over the 407,550 universe loci (849 D2 zero loci excluded; identical for every arm). `pc1_removed_cosine`: additionally remove the top principal component. `neg_euclidean`: -||e_i - e_j||. "
      "PC1 explained variance ratio of the centred embeddings (stored): " + ", ".join(f"{SHORT[a]} {ex.sim[a]['meta']['pc1_explained_variance_ratio']:.3f}" for a in ARMS) + ".\n")
    A("**Pooled primary-style statistic, all pairs, value [CI] (within-arm delta vs cosine [CI] in parentheses):**\n")
    rows = []
    for g in GEOMS:
        r = [GEOM_LABEL[g]]
        for a in ARMS:
            s = ex.S(a, g, "all")
            cell = fv(s["value"], s["ci_lo"], s["ci_hi"])
            if g != "cosine":
                d = ex.geo[(ex.geo.arm == a) & (ex.geo.geometry == g) & (ex.geo.stat_family == "spearman_pooled_eqw") & (ex.geo.cell == "all")].iloc[0]
                cell += f"<br>({d.delta:+.3f} [{d.ci_lo:+.3f}, {d.ci_hi:+.3f}])"
            r.append(cell)
        rows.append(r)
    A(md_table(["Geometry", *[SHORT[a] for a in ARMS]], rows))
    A("\n**Candidate minus comparator under each geometry (pooled primary-style statistic) delta [CI] (raw p), with position of the candidate:**\n")
    rows = []
    for c in ["all", "V_Q4", "V_top10pct"]:
        for g in GEOMS:
            r = [c, GEOM_LABEL[g]]
            for k in COMPARATORS:
                d = ex.scon_row(k, g, c)
                r.append(f"{fd(d, p=False)} ({excl(d)})")
            rows.append(r)
    A(md_table(["Cell", "Geometry", "vs Functional PCA", "vs CpGPT-large", "vs DeepCpG-DNA"], rows))
    def _gdelta(a, g):
        return ex.geo[(ex.geo.arm == a) & (ex.geo.geometry == g) & (ex.geo.stat_family == "spearman_pooled_eqw") & (ex.geo.cell == "all")].iloc[0]

    imp = {a: [g for g in GEOMS[1:] if _gdelta(a, g).delta >= 0.03 and _gdelta(a, g).ci_lo > 0] for a in ARMS}
    A("\n**Which geometry improves which arm (registered IMPROVES criterion: within-arm delta >= 0.03 with CI > 0).** "
      + "; ".join(f"{SHORT[a]}: {', '.join(imp[a]) if imp[a] else 'none'}" for a in ARMS) + ". "
      "The two ENCODE-feature arms (Regulatory, Functional PCA) improve by +0.13 to +0.26 under all three alternative geometries (largest under negative Euclidean), whereas CpGPT-large and DeepCpG-DNA do not improve under any of them "
      "(CpGPT-large -0.012 / -0.034 / -0.030; DeepCpG-DNA -0.006 / -0.040 / +0.002 for centred / PC1-removed / Euclidean; the DeepCpG Euclidean gain has CI > 0 but is far below the 0.03 threshold). "
      "In other words the cosine Loyfer advantage of the two sequence arms is specific to the cosine geometry on the raw embedding, while the candidate's very negative cosine value is strongly geometry-dependent.\n")
    A(f"**Registered GO outcome: {pr['geometry_outcome']}.** The candidate's own improvements (all CI > 0): centred cosine {fd(abc['geometry_within_candidate']['centered_cosine'], p=False)}, "
      f"PC1-removed {fd(abc['geometry_within_candidate']['pc1_removed_cosine'], p=False)}, negative Euclidean {fd(abc['geometry_within_candidate']['neg_euclidean'], p=False)}. "
      "Under each of these geometries, applied to every arm, the candidate is not significantly below either SEQ arm on the pooled primary-style statistic: "
      "it is in fact significantly ABOVE both (candidate minus CpGPT-large: centred " + fd(abc['geometry_gap_vs_seq']['centered_cosine'][CPG], p=False) + ", PC1-removed " + fd(abc['geometry_gap_vs_seq']['pc1_removed_cosine'][CPG], p=False) + ", Euclidean " + fd(abc['geometry_gap_vs_seq']['neg_euclidean'][CPG], p=False) +
      "; minus DeepCpG-DNA: " + ", ".join(fd(abc['geometry_gap_vs_seq'][g][DEE], p=False) for g in ["centered_cosine", "pc1_removed_cosine", "neg_euclidean"]) + "). "
      "Magnitude: the registered label 'CLOSES' means only 'not significantly below'; here the pooled-statistic gap goes from about -0.17 / -0.18 (cosine) to +0.08 / +0.06 (centred), +0.013 / +0.010 (PC1-removed, barely above 0) and +0.13 / +0.08 (Euclidean), i.e. it is reversed in sign, by a very different amount per geometry (smallest after PC1 removal, largest under Euclidean). "
      "Important qualifications: (a) this holds for the all-pairs pooled statistic; in the most variable pairs (V_top10pct) the candidate remains significantly below CpGPT-large under cosine (-0.097), centred cosine (-0.074) and negative Euclidean (-0.029) and is indistinguishable from it only after PC1 removal (-0.011, CI includes 0); in V_Q4 it is indistinguishable from CpGPT-large under centred and PC1-removed cosine and above it under Euclidean (see table); "
      "(b) the alternative geometries were chosen after seeing the primary result and none is a new primary; (c) the candidate's PC1 carries 46% of the variance of the centred embedding (DeepCpG-DNA 53%, CpGPT-large 6%), consistent with a dominant shared direction that cosine on the raw embedding is sensitive to, but this is a description, not a mechanistic test; "
      "(d) 'closes' does NOT mean the candidate is better on Loyfer: the frozen primary stands.\n")

    # B3
    A("\n## B3. Chromosome-blocked multi-output ridge: methylation profile from the CpG representation\n")
    A("Target: frozen WGBS beta profile over 39 cell groups for the 382,041 CpGs observed in all groups (192,339 'variable' CpGs with per-CpG variance >= 0.008412). T1 raw (39 betas), T2 centred (beta minus per-CpG mean: the profile shape), T3 level (per-CpG mean). "
      "Frozen chromosome-blocked folds (5), `chrom_probes.ridge_blocked_cv` unchanged (alpha grid 1e-2..1e4, inner leave-one-fold-out CV). Metrics are out-of-fold; the reference predictor is the train-fold per-group mean (R2 about 0 by construction on all CpGs; it is clearly negative on the variable-CpG subset, e.g. T1 -0.148 and T3 -0.242, because train-fold means mispredict variable CpGs).\n")
    A("**Global R2 [95% chromosome-block CI] (Pearson over all CpG x group cells in parentheses).**\n")
    rows = []
    for t, tl in [("raw", "T1 raw"), ("centered", "T2 centred"), ("level", "T3 level")]:
        for s, sl in [("all_cpgs", "all CpGs"), ("variable_cpgs", "variable CpGs")]:
            r = [f"{tl}, {sl}"]
            for a in ARMS:
                m = ex.R(a, t, s)
                r.append(f"{fv(m['r2_global'], *m['r2_global_ci'])} ({m['pearson_pooled']:.3f})")
            b = ex.Rbase(REF, t, s)
            r.append(f"{b['r2_global']:.3f}")
            rows.append(r)
    A(md_table(["Target, subset", *[SHORT[a] for a in ARMS], "Train-mean baseline R2"], rows))
    A("\n**Mean per-CpG Pearson across the 39 groups (shape readout):**\n")
    rows = []
    for t, tl in [("raw", "T1 raw"), ("centered", "T2 centred")]:
        for s, sl in [("all_cpgs", "all CpGs"), ("variable_cpgs", "variable CpGs")]:
            rows.append([f"{tl}, {sl}", *[_pm(ex.R(a, t, s)) for a in ARMS]])
    A(md_table(["Target, subset", *[SHORT[a] for a in ARMS]], rows))
    A("\n**Candidate minus comparator, global R2 (chromosome-block bootstrap; raw p; descriptive), and Functional minus sequence arms:**\n")
    rows = []
    for t, tl in [("raw", "T1"), ("centered", "T2"), ("level", "T3")]:
        for s, sl in [("all_cpgs", "all"), ("variable_cpgs", "variable")]:
            r = [f"{tl} {sl}"]
            for k in COMPARATORS:
                r.append(fd(ex.rcon_row(k, t, s, "r2_global")))
            for k in SEQ:
                x = ex.fvs[(ex.fvs.comparator == k) & (ex.fvs.target == t) & (ex.fvs.subset == s) & (ex.fvs.metric == "r2_global")].iloc[0]
                r.append(fd(dict(delta=x.delta, ci_lo=x.ci_lo, ci_hi=x.ci_hi, p=x.p)))
            rows.append(r)
    A(md_table(["Target, subset", "Regu - Func", "Regu - CpGPT", "Regu - Deep", "Func - CpGPT", "Func - Deep"], rows))
    A("\n**Per-fold global R2, T2 centred, variable CpGs (point estimates):**\n")
    rows = []
    for a in ARMS:
        pf = ex.R(a, "centered", "variable_cpgs")["per_fold"]
        rows.append([SHORT[a], " / ".join(f"{pf[str(f)]['r2_global']:.3f}" for f in ex.rdg[a]["folds"])])
    A(md_table(["Arm", "folds 0..4"], rows))
    A("\n**Chosen ridge alpha per fold (median over the targets of that set; grid 1e-2..1e4):**\n")
    rows = []
    for a in ARMS:
        r = [SHORT[a]]
        for t in ["raw", "centered", "level"]:
            r.append(" / ".join(f"{x:g}" for x in ex.rdg[a]["targets"][t]["alpha_per_fold_median_over_targets"]))
        rows.append(r)
    A(md_table(["Arm", "T1 raw", "T2 centred", "T3 level"], rows))
    A("CpGPT-large selects the top of the grid (alpha = 1e4) in all five folds for T2 and the bottom (0.01) in several folds for T1/T3; the Regulatory arm selects 1e3 in all folds for T2. "
      "A grid-edge choice means the optimum may lie outside the registered grid for that arm (grid not extended: that would be a methodology change), so the CpGPT-large T2 R2 could be a slight underestimate of what a wider grid would give.\n")
    # per group
    mat = {a: np.array(ex.R(a, "centered", "variable_cpgs")["r2_per_group"]) for a in ARMS}
    groups = ex.rdg[REF]["groups"]
    A("\n**Per-cell-group R2, T2 centred, variable CpGs (39 groups).** Range per arm: " + "; ".join(f"{SHORT[a]} {mat[a].min():.3f} to {mat[a].max():.3f} (median {np.median(mat[a]):.3f})" for a in ARMS) + ". "
      "Heatmap: `" + figrel + "/fig3_ridge_r2_and_per_group.png`. Full table:\n")
    rows = []
    for i, g in enumerate(groups):
        r = [g]
        for a in ARMS:
            r.append(f"{mat[a][i]:.3f}")
        dd = {k: ex.rcon_row(k, "centered", "variable_cpgs", "r2_per_group", g) for k in SEQ}
        r.append(" / ".join(f"{dd[k]['delta']:+.3f}" + ("*" if dd[k]["ci_lo"] > 0 or dd[k]["ci_hi"] < 0 else "") for k in SEQ))
        rows.append(r)
    A(md_table(["Group", *[SHORT[a] for a in ARMS], "Regu minus CpGPT / minus Deep (* = CI excludes 0)"], rows))
    npos = {k: sum(1 for g in groups if ex.rcon_row(k, "centered", "variable_cpgs", "r2_per_group", g)["ci_lo"] > 0) for k in COMPARATORS}
    A(f"\nGroups (of 39) in which the candidate's per-group R2 is significantly above the comparator (CI > 0): vs Functional {npos[FUN]}, vs CpGPT-large {npos[CPG]}, vs DeepCpG-DNA {npos[DEE]}.\n")
    m_ = {a: ex.R(a, "centered", "variable_cpgs") for a in ARMS}
    A(f"**Reading.** Linear decodability of the centred profile (T2, variable CpGs) is high for the Regulatory arm (R2 {m_[REF]['r2_global']:.3f}) and equally for Functional PCA ({m_[FUN]['r2_global']:.3f}; Regulatory minus Functional {fd(ex.rcon_row(FUN, 'centered', 'variable_cpgs', 'r2_global'), p=False)}), "
      f"and poor for CpGPT-large ({m_[CPG]['r2_global']:.3f}) and DeepCpG-DNA ({m_[DEE]['r2_global']:.3f}, below the registered 0.02 near-baseline threshold). Regulatory and Functional exceed both sequence arms in all six target/subset combinations (T1/T2/T3 x all/variable CpGs), and the Regulatory-Functional differences are at most 0.022 in absolute R2 (Functional slightly higher for T1/T3, equal for T2). Per group, the candidate is significantly above CpGPT-large and DeepCpG-DNA in 39/39 groups, and above Functional PCA in 22/39. "
      "Functional is not below either sequence arm (FO = NOT_BELOW), i.e. the ENCODE-feature arms behave alike here.\n")

    # B4
    A("\n## B4. Provenance audit summary (documentation only; `docs/BIOLOGICAL_VALIDATION_V2_EXPLORATORY_POSTHOC_PROVENANCE_AUDIT.md`)\n")
    A("Labels preserved from the audit: VERIFIED-IN-REPO, VERIFIED-LOCAL-SIBLING, VERIFIED-EXTERNAL, UNVERIFIED.\n")
    A("- **CpGPT-large:** pretrained with methylation-value reconstruction/uncertainty, beta-distribution and contrastive losses (`large.yaml`: VERIFIED-LOCAL-SIBLING); corpus = Illumina array bulk samples only (27k/450k/EPIC/EPICv2; no WGBS) (VERIFIED-LOCAL-SIBLING README and VERIFIED-EXTERNAL GitHub README; the ~150,000-sample / >2,000-study figures are VERIFIED-EXTERNAL only through a fetch-tool summary of the bioRxiv v5, preprint not read in full); "
      "the 512-d locus vector is a trained projection of frozen Nucleotide Transformer v2 DNA embeddings (VERIFIED-EXTERNAL summary). Exact corpus composition and whether Loyfer/GSE186458 samples are in CpGCorpus: UNVERIFIED. Whether the locus vector carries methylation-derived information beyond sequence: UNVERIFIED as a measurement, VERIFIED as a training fact.")
    A("- **DeepCpG-DNA (HCC):** DNA module of the zoo model trained on Hou et al. 2016 scRRBS, 25 human hepatocellular carcinoma cells (VERIFIED-IN-REPO README; VERIFIED-EXTERNAL zoo page); the objective of the DNA module (per-cell methylation state from a 1001 bp window) is stated in the repo and is UNVERIFIED externally in this session (paper paywalled); which benchmark CpGs were in the RRBS training set: UNVERIFIED.")
    A("- **Overlap with Loyfer (WGBS, 39 sorted-cell groups):** direct data overlap UNVERIFIED (unlikely for CpGPT: arrays vs WGBS; not applicable for DeepCpG). **Direct methylation supervision: YES for both (VERIFIED).** Indirect overlap plausible: CpGPT likely (array probes = the benchmark universe); DeepCpG partial (level/CpG-density determinants only) (non-binding, UNVERIFIED empirically). "
      "A large part of cross-cell-type methylation structure is sequence-derivable without methylation labels, so the contrast cannot attribute an advantage to supervision rather than sequence information.")
    A("- **Documentary outcome PO = DIRECT_METHYLATION_SUPERVISION for both sequence arms** (fixed by the audit; independent of the results). The audit never excludes a comparator or invalidates a result.\n")

    # A/B/C
    A("\n## B5. A / B / C adjudication per the registered rules (`interpretation_ABC.json`)\n")
    A("Registered inputs: RO (ridge outcome from candidate-minus-SEQ deltas), GO (geometry outcome), FO (functional below both SEQ on ridge?), PO (provenance). Thresholds R2_NEAR_BASELINE = 0.02, GEO_IMPROVEMENT = 0.03; 'significantly' = paired 95% CI excludes 0. Primary reading = T2 centred, variable CpGs, global R2; consistency readings T2 all CpGs and T1 all CpGs.\n")
    rows = []
    for k, v in rd.items():
        rows.append([k + (" (PRIMARY reading)" if k == abc["primary_reading"] else ""), v["ridge_outcome"], v["geometry_outcome"], v["functional_outcome"], v["provenance_outcome"], v["A"], v["B"], v["C"]])
    A(md_table(["Reading", "RO", "GO", "FO", "PO", "A", "B", "C"], rows))
    sg = {k: fd(pr["candidate_vs_seq"][k], p=False) for k in SEQ}
    A(f"\nThe three readings agree (no 'subset/target dependent' qualification is needed). Candidate minus SEQ on the primary reading: CpGPT-large {sg[CPG]}, DeepCpG-DNA {sg[DEE]}, hence RO = {pr['ridge_outcome']} (candidate not significantly below either; significantly above both). "
      f"Ridge R2 values: " + ", ".join(f"{SHORT[a]} {v:.3f}" for a, v in pr["M_ridge_values"].items()) + ".\n")
    A("**Resulting reading under the registered mechanical table (RO = HIGHER, GO = CLOSES, FO = NOT_BELOW, PO = DIRECT_METHYLATION_SUPERVISION):**\n")
    A("- **A (the regulatory representation lacks long-range methylation-program information): not supported.** The profile shape across 39 cell groups is linearly decodable from the candidate (R2 0.279 on variable CpGs, 0.245 on all CpGs) far above the baseline and far above the sequence arms.")
    A("- **B (the information is present but cosine geometry does not expose it): supported (strong) by the registered table.** 'Supported' is the mechanical output of the registered table (RO and GO), not a demonstration that the raw-embedding cosine 'hides' a methylation program; operationally: a supervised linear readout recovers the profile, and several alternative similarities (centred cosine, Euclidean, PC1-removed) lift the candidate's pooled Loyfer statistic from -0.063 to +0.174 / +0.202 / +0.086, above the sequence arms under those geometries.")
    A("- **C (result materially affected by methylation pretraining/provenance of the comparators): not supported by ridge evidence** (the comparator advantage disappears under the supervised readout, where they are worse). It is NOT excluded: both sequence arms have direct methylation supervision (PO), the design cannot separate 'ENCODE-feature arms carry the information' from 'sequence arms were trained on methylation', and their cosine advantage exists.")
    A("- **Combinations:** A and C are both 'not supported' (not 'confounded', because FO = NOT_BELOW); B is the only supported reading. No reading is promoted to a primary or to a Holm family.\n")
    A("**What the evidence does NOT show.** See the caveats below; in particular it does not show that the candidate is better than the sequence models on Loyfer, that the candidate 'captures long-range methylation programs' in the sense of the primary endpoint, or that sequence embeddings lack methylation information.\n")

    # caveats
    A("\n## B6. Caveats (do not omit)\n")
    cav = [
        "**(i) Decodability is not the primary construct.** The ridge shows that the methylation profile is linearly decodable from the regulatory embedding (and equally from the functional one) but poorly from CpGPT/DeepCpG embeddings. This is a supervised linear readout on a target that is cell-type-specific; histone and DNase tracks are cell-type-specific chromatin measurements, so linear decodability of cell-type methylation profiles from them is plausible and does not by itself show that the regulatory embedding 'captures long-range methylation programs' in the sense of the primary endpoint (pair-similarity rank agreement at >1 Mb / interchromosomal distance). It also does not show that the sequence embeddings lack methylation information: they may encode it nonlinearly or in a geometry that ridge does not read, and their cosine Loyfer advantage exists.",
        "**(ii) Different task.** The ridge readout predicts profiles per locus (supervised, fitted on training chromosomes) whereas the primary measures unsupervised pair similarity; neither implies the other.",
        "**(iii) Same ridge protocol/alpha grid for all arms.** This may disadvantage the 128-D and 512-D sequence embeddings (regularisation/scaling; StandardScaler per feature, grid 1e-2..1e4 in sklearn convention). Chosen alphas are reported in B3; CpGPT-large reaches the grid top (1e4) in all folds for T2 and the grid bottom in some folds for T1/T3, so its optimum may lie outside the grid.",
        "**(iv) Fit scope, dimension and dtype.** The regulatory SVD was fit on chr1-19 only (chr20-22 transformed out-of-fit); dimensions 256 / 256 / 512 / 128 and float32 / float16 differ; no dimension matching was attempted; 798 regulatory zero-embedding rows remain in the ridge set as valid features (D2 probe convention).",
        "**(v) Post hoc, single seed, 22 chromosome blocks, intra-chromosome folds.** Conceived after the primary results; one seed; CIs reflect chromosome resampling only (probe predictions are not refitted inside the bootstrap); raw p floor 0.002; folds are chromosome-blocked but inter-chromosomal and long-range dependencies are not captured by blocks; no multiplicity control.",
        "**(vi) Provenance.** Both sequence arms have direct methylation supervision (CpGPT-large: array pretraining; DeepCpG-DNA: scRRBS HCC) with indirect overlap to Loyfer WGBS plausible. C is therefore 'not supported by ridge evidence' but cannot be excluded by the design. Also 'CLOSES' is the registered label for 'a geometry shrinks the gap until the candidate is not significantly below': the magnitude is reported (gap about -0.17 under cosine; +0.08 centred, +0.01 PC1-removed, +0.13 Euclidean versus CpGPT-large) and it depends strongly on the geometry; in V_top10pct the candidate stays below CpGPT-large under cosine, centred cosine and Euclidean.",
        "**(vii) No claim of superiority.** Do not conclude that the candidate is 'better' on Loyfer: the frozen primary (candidate below both sequence models and slightly below Functional PCA) stands unchanged; Part B cannot replace it.",
        "**(viii) Geometry choices.** The alternative geometries were selected after seeing the primary result; centring uses a mean over the universe (identical for every arm) and is a transformation that a user of the raw embedding may or may not apply; the sequence arms do not benefit from any of them.",
        "**(ix) Part A scope.** Sensitivities are descriptive variants of the same experiment (same arms, folds, draw table, seed); 5 of 22 protocol rows are not runnable; D-A1 and D-A2 were interpreted by the orchestrator, not approved by the user.",
    ]
    for c in cav:
        A("- " + c)
    A("\n## Figures (`" + figrel + "`; arm colours as in `report/figures`)\n")
    for n, d in [("fig1_loyfer_variance_strata.png", "Loyfer ranking across variance strata (pooled primary-style statistic)"),
                 ("fig2_geometry_diagnostics.png", "four geometries x four arms (pooled statistic, all pairs)"),
                 ("fig3_ridge_r2_and_per_group.png", "ridge R2 (T2, variable CpGs) with CI and per-cell-group R2 heatmap"),
                 ("fig4_cosine_vs_ridge_dissociation.png", "cosine Loyfer Spearman versus ridge R2 per arm (and centred cosine versus ridge)")]:
        A(f"- `{n}`: {d}")
    A("\n## Integrity statement\n")
    A("No statistic was recomputed; no Holm correction was recomputed or extended; no frozen artifact, registration or primary result was modified; the candidate and the protocol were not changed; no freeze/prepare/build/run_followup/run_evaluation command was run; no TCGA data, training or GPU was used; nothing was committed. "
      "The only files written by this report are this document and `exploratory_posthoc_loyfer/figures_followup/*.png`.")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    a = ap.parse_args(argv)
    root, doc = a.root.resolve(), a.doc.resolve()

    def write_ok(p):
        assert_writable(p, root, doc)
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    fz, ex = Frozen(root), Expl(root)
    summ = pd.read_csv(root / "report" / "summary_primary.csv")
    figdir = root / "exploratory_posthoc_loyfer" / "figures_followup"
    make_figures(ex, figdir, write_ok)
    figrel = "outputs/biological_validation_v2/" + TAG + "/exploratory_posthoc_loyfer/figures_followup"
    write_ok(doc)
    doc.write_text(build_markdown(fz, ex, root, figrel, summ))
    print("wrote", doc)


if __name__ == "__main__":
    main()
