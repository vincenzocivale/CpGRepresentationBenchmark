"""Supplementary figures/tables S1-S14 and S-final placeholder."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa: F401,F403,E402

O = ROOT / "outputs"
BV = O / "biological_validation_v2/bioval-v2-protocol-freeze-v1"
FR = [0.15, 0.3, 0.5, 0.7, 0.9]
CAND, CP, DC = MAIN_ARMS
LEG = "functional_annotations_pca"
ALL4 = MAIN_ARMS + [LEG]
ci = lambda s, k: json.loads(s)[k]  # noqa: E731


def new(name, title, tags=()):
    return Build(name, __file__, title=title, tags=tags)


# ============================================================ S1 full ENCODE attribution campaign
def s1():
    b = new("S1_encode_campaign", "Full ENCODE attribution campaign")
    pc = b.csv(O / "encode_atlas_v1/paired_comparisons.csv")
    pc = pc[(pc.view == "seen") & (~pc.ood)].copy()
    pc["rel_pct"] = 100 * pc.relative_delta_mse
    pc["lo"] = 100 * pc.relative_ci95.apply(lambda s: ci(s, 0)); pc["hi"] = 100 * pc.relative_ci95.apply(lambda s: ci(s, 1))
    d = pc[pc.stage == "discovery"][["arm", "family", "seed", "rel_pct", "lo", "hi", "n_patients", "n_blocks"]].sort_values("rel_pct")
    cf = pc[pc.stage == "confirm"].groupby(["arm", "family"]).agg(rel_pct_mean=("rel_pct", "mean"), n_seeds=("seed", "nunique")).reset_index()
    cols = {"arm": "campaign arm id", "family": "arm family (arms.json)", "seed": "seed", "rel_pct": "100*(arm - full)/full MSE@50% (paired)", "lo": "paired 95% CI lower (patient x 1 Mb block bootstrap)",
            "hi": "upper", "n_patients": "patients", "n_blocks": "blocks", "rel_pct_mean": "mean over 3 confirm-stage seeds of rel_pct", "n_seeds": "number of seeds"}
    b.source("discovery", d, cols, "paired_comparisons.csv stage=discovery, view=seen, ood=False")
    b.source("confirm", cf, cols, "stage=confirm, mean over seeds")
    fams = sorted(d.family.unique()); pal = [OI[k] for k in ("blue", "orange", "green", "verm", "purple", "sky", "grey", "black")]
    fcol = {f: pal[i % 8] for i, f in enumerate(fams)}
    fig, ax = plt.subplots(figsize=(7.2, 11))
    for i, (_, r) in enumerate(d.iterrows()):
        ax.errorbar(r.rel_pct, i, xerr=[[r.rel_pct - r.lo], [r.hi - r.rel_pct]], fmt="o", ms=2.5, color=fcol[r.family], lw=0.6, capsize=0)
    cm = cf.set_index("arm").rel_pct_mean
    for i, (_, r) in enumerate(d.iterrows()):
        if r.arm in cm.index: ax.plot(cm[r.arm], i, marker="D", mfc="none", mec="black", ms=4, ls="none")
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.arm, fontsize=3.8); ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("paired relative ΔMSE@50% vs Full functional (%), (arm − full)/full; seed 17, discovery")
    for f in fams: ax.plot([], [], "o", color=fcol[f], label=f)
    ax.plot([], [], "D", mfc="none", mec="black", label="mean of 3 confirm seeds"); ax.legend(fontsize=5.5, loc="lower right", ncol=2)
    ax.set_title("DISCOVERY EVIDENCE: all campaign arms (x-axis not zero-bounded: effect sizes relative to Full)", fontsize=7)
    b.save(fig, "S1_encode_campaign", dpi=300)
    b.caption("**Supplementary Figure S1. Full ENCODE attribution campaign (discovery evidence).** Paired relative MSE difference at 50% masking of every campaign arm against the Full functional representation, (arm - Full)/Full; discovery stage, seed 17, 95% CI from the paired patient x 1 Mb block bootstrap (2,000 replicates). Open diamonds: mean over 3 seeds for arms with confirm-stage runs. Colours: arm family. Discovery evidence only.")
    b.finish(filters=["paired_comparisons.csv view=seen, ood=False; stage discovery (points) and confirm (diamonds)"], caption_stats=["paired relative delta vs full; CI bootstrap patient x 1Mb block"])


# ============================================================ S2/S3 focused contrasts
def focused(name, title, contrasts, mse_arms, label, cap):
    b = new(name, title)
    fc = b.csv(O / "encode_atlas_v1/focused_contrasts.csv"); rm = b.csv(O / "encode_atlas_v1/reconstruction_metrics.csv")
    x = fc[fc.contrast.isin(contrasts)].copy()
    x["rel_pct"] = 100 * x.relative_delta_mse; x["lo"] = 100 * x.relative_ci95.apply(lambda s: ci(s, 0)); x["hi"] = 100 * x.relative_ci95.apply(lambda s: ci(s, 1))
    x = x[["contrast", "baseline", "alternative", "seed", "rel_pct", "lo", "hi", "n_patients", "n_blocks"]]
    m = rm[(rm.stage == "confirm") & (~rm.ood) & (~rm.pilot) & (rm.view == "seen") & (rm["mask"] == "mask_0.50") & rm.arm.isin(mse_arms)][["arm", "seed", "mse"]]
    b.source("contrasts", x, {"contrast": "focused contrast name", "baseline": "baseline arm", "alternative": "alternative arm", "seed": "seed", "rel_pct": "100*(alternative - baseline)/baseline MSE@50% (paired; negative favours alternative)",
                              "lo": "paired 95% CI lower", "hi": "upper", "n_patients": "patients", "n_blocks": "1 Mb blocks"}, f"focused_contrasts.csv contrast in {contrasts}")
    b.source("mse", m, {"arm": "arm", "seed": "seed", "mse": "MSE@50% masking, confirm stage"}, "reconstruction_metrics.csv confirm/seen/mask_0.50")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(wspace=0.55, width_ratios=[1.3, 1]))
    ax = axs[0]; panel_letter(ax, "A"); y = 0; yt = []; yl = []
    for k, c in enumerate(contrasts):
        for sd in (17, 42, 97):
            r = x[(x.contrast == c) & (x.seed == sd)].iloc[0]
            y = forest(ax, [dict(est=r.rel_pct, lo=r.lo, hi=r.hi, marker=SEED_MARK[sd], color=[OI["blue"], OI["purple"], OI["verm"]][k])], y); yt.append(y - 1); yl.append(f"{c.replace('_', ' ')}  s{sd}")
        y += 0.8
    ax.axvline(0, color="black", lw=0.6); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=5.5); ax.set_xlabel("paired relative ΔMSE@50% (%)\nnegative favours alternative"); ax.set_title("per-seed paired contrasts (95% CI)", fontsize=7)
    ax = axs[1]; panel_letter(ax, "B")
    for i, a in enumerate(mse_arms):
        v = m[m.arm == a].sort_values("seed"); ax.bar(i, v.mse.mean(), color=OI["grey"], alpha=0.6, edgecolor="black", lw=0.5, width=0.7); ax.scatter(i + jitter(3), v.mse, s=8, c="black", zorder=3, linewidths=0)
    ax.set_xticks(range(len(mse_arms))); ax.set_xticklabels([a.replace("control/", "").replace("add/", "+").replace("only/", "only ") for a in mse_arms], rotation=35, ha="right", fontsize=5.5)
    ax.set_ylim(0, m.mse.max() * 1.12); ax.set_ylabel("MSE@50% (3 seeds)"); ax.set_title("absolute MSE (zero-based)", fontsize=7)
    b.save(fig, name); b.caption(cap)
    b.finish(filters=[f"focused_contrasts.csv: {contrasts}"], caption_stats=["paired 95% CI: patient x 1 Mb block bootstrap, 2000 replicates; per seed; seeds = optimisation replicates"])
    return b


def s2():
    focused("S2_matched_histone", "Matched histone vs non-histone", ["matched_assay_identity"], ["control/non_histone_prevalence_matched", "control/histone_prevalence_matched", "full"],
            "S2", "**Supplementary Figure S2. Matched histone vs non-histone controls (discovery evidence).** Panels: (A) paired relative MSE difference between prevalence-matched histone and non-histone track sets (same number of tracks, matched prevalence support) at 50% masking per seed, with paired patient x 1 Mb block bootstrap 95% CI; negative favours histone tracks. (B) MSE@50% of the matched arms (bars = mean of 3 seeds, dots = seeds; zero-based) with Full for reference. Interpret only on the shared prevalence support.")


def s3():
    focused("S3_h3k4me3", "H3K4me3 and histone-mark attribution", ["h3k4me3_increment", "histone_sufficiency"], ["context_only", "add/target/H3K4me3", "only/target/H3K4me3", "add/assay/Histone ChIP-seq", "full"],
            "S3", "**Supplementary Figure S3. H3K4me3 and histone attribution (discovery evidence).** (A) Per-seed paired relative MSE@50% difference for adding H3K4me3 tracks to the context-only representation (h3k4me3_increment) and for adding all histone tracks (histone_sufficiency), with paired patient x 1 Mb block bootstrap 95% CI (negative favours the added tracks). (B) MSE@50% for the corresponding arms (bars = seed mean; dots = seeds; zero-based).")


# ============================================================ S4 SVD energy / effective rank
def s4():
    b = new("S4_svd_compression", "SVD energy and effective rank")
    r = b.json(O / "regulatory_compression_reports/report.json"); b.track(O / "regulatory_compression_reports/report.md")
    E = r["embeddings"]
    keys = [k for k in E if k.endswith("global_svd256__discovery_chr1_19")]
    lab = {"regulatory_histone": "Histone", "regulatory_clean": "Clean (Hist+DNase+TF)", "regulatory_all_experimental": "All experimental"}
    rows = []; cum = []
    for k in keys:
        e = E[k]; en = np.array(e["energy_per_component"]); c = np.cumsum(en) / e["total_energy"]
        for i, v in enumerate(c): cum.append(dict(feature_set=e["feature_set"], component=i + 1, cumulative_energy=float(v)))
        rows.append(dict(feature_set=e["feature_set"], n_features=e["n_features"], dim=e["dim"], energy_explained_256=e["explained_energy_cum"]["256"], variance_explained_256=e["explained_variance_cum"]["256"],
                         pr_uncentered=e["effective_rank_uncentered"]["participation_ratio"], entropy_rank_uncentered=e["effective_rank_uncentered"]["exp_entropy_rank"],
                         pr_centered=e["effective_rank_centered"]["participation_ratio"], entropy_rank_centered=e["effective_rank_centered"]["exp_entropy_rank"],
                         block_energy_histone=e["block_only_energy"].get("histone", np.nan), block_energy_tf=e["block_only_energy"].get("tf", np.nan), block_energy_accessibility=e["block_only_energy"].get("accessibility", np.nan),
                         input_share_histone=e["input_energy_share"].get("histone", np.nan), input_share_tf=e["input_energy_share"].get("tf", np.nan), input_share_accessibility=e["input_energy_share"].get("accessibility", np.nan)))
    S = pd.DataFrame(rows); C = pd.DataFrame(cum)
    b.source("summary", S, {c: c.replace("_", " ") + " (compression report.json; global SVD-256, discovery chr1-19 fit)" for c in S.columns}, "report.json embeddings *__global_svd256__discovery_chr1_19")
    b.source("cumulative_energy", C, {"feature_set": "feature set", "component": "SVD component (1-based)", "cumulative_energy": "cumulative energy_per_component / total_energy"}, "report.json energy_per_component")
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw=dict(wspace=0.45))
    cc = {"regulatory_histone": OI["sky"], "regulatory_clean": OI["verm"], "regulatory_all_experimental": OI["purple"]}
    ax = axs[0]; panel_letter(ax, "A")
    for fs_, g in C.groupby("feature_set"): ax.plot(g.component, g.cumulative_energy, color=cc[fs_], label=lab[fs_])
    ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_xlabel("SVD component"); ax.set_ylabel("cumulative energy fraction"); ax.legend(fontsize=5.5, loc="lower right")
    ax = axs[1]; panel_letter(ax, "B"); w = 0.2; xs = np.arange(4); names = ["PR\nuncentered", "exp-entropy\nuncentered", "PR\ncentered", "exp-entropy\ncentered"]
    for i, (_, r_) in enumerate(S.iterrows()): ax.bar(xs + (i - 1) * w, [r_.pr_uncentered, r_.entropy_rank_uncentered, r_.pr_centered, r_.entropy_rank_centered], w, color=cc[r_.feature_set], edgecolor="black", lw=0.4)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=5.5); ax.set_ylabel("effective rank"); ax.set_title("of 256 dimensions", fontsize=7)
    ax = axs[2]; panel_letter(ax, "C"); Sm = S[S.feature_set != "regulatory_histone"]
    for i, (_, r_) in enumerate(Sm.iterrows()):
        bot = 0
        for blk, col in [("histone", OI["sky"]), ("tf", OI["purple"]), ("accessibility", OI["blue"])]:
            v = r_[f"block_energy_{blk}"]; ax.bar(i, v, bottom=bot, color=col, edgecolor="black", lw=0.4, width=0.6, label=blk if i == 0 else None); bot += v
    ax.set_xticks(range(len(Sm))); ax.set_xticklabels([lab[f] for f in Sm.feature_set], fontsize=5.5, rotation=15); ax.set_ylabel("fraction of embedding energy by block"); ax.legend(fontsize=5.5)
    b.save(fig, "S4_svd_compression")
    b.note("The compression report does not contain the frozen Histone+DNase (2,492-track) set: only histone, clean and all-experimental global SVD-256 sets are available; Histone+DNase compression statistics would need a new report (not run).")
    b.caption("**Supplementary Figure S4. SVD energy and effective rank of regulatory feature sets (discovery chr1-19 fit, global SVD, 256 dimensions).** (A) Cumulative energy fraction vs component. (B) Effective rank (participation ratio and exponential-entropy rank) of the uncentred and centred spectra. (C) Fraction of embedding energy contributed by each assay block. Source: regulatory_compression_reports/report.json. The frozen Histone+DNase set is not part of this report.")
    b.finish(filters=["report.json: global_svd256 discovery_chr1_19 embeddings"], caption_stats=["energy fraction = sum energy_per_component / total_energy", "PR = participation ratio of squared singular values", "descriptive; no inference"],
             missing_results=["compression statistics for the frozen regulatory_histone_dnase feature set (not in report.json)"])


# ============================================================ S5 training curves
def s5():
    b = new("S5_training_curves", "3-seed family selection training curves")
    c = b.csv(O / "regulatory_confirm_v1/analysis/validation_curves_all_runs.csv")
    b.source("curves", c[["arm", "seed", "epoch", "validation_mse", "validation_mae"]], {"arm": "arm", "seed": "seed", "epoch": "epoch (0-based)", "validation_mse": "validation MSE@0.50 at epoch end", "validation_mae": "validation MAE"}, "validation_curves_all_runs.csv, all rows")
    col = {"regulatory_histone": OI["sky"], "regulatory_histone_dnase": OI["blue"], "regulatory_clean": OI["verm"]}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.3))
    for k, ax in enumerate(axs):
        panel_letter(ax, "AB"[k])
        for a, g in c.groupby("arm"):
            for sd, h in g.groupby("seed"): ax.plot(h.epoch, h.validation_mse, color=col[a], lw=0.8, ls=["-", "--", ":"][[17, 42, 97].index(sd)], label=f"{a.replace('regulatory_', '')}" if sd == 17 else None)
        ax.set_xlabel("epoch")
        if k == 0: ax.set_ylabel("validation MSE@50%"); ax.set_ylim(0, c.validation_mse.max() * 1.05); ax.set_title("full range (zero-based)", fontsize=7)
        else: ax.set_xlim(30, c.epoch.max() + 1); lo = c[c.epoch >= 30].validation_mse; ax.set_ylim(lo.min() * 0.999, lo.max() * 1.001); ax.set_title("ZOOM: epochs 30-79, y-axis truncated", fontsize=7, color=OI["verm"])
    axs[0].legend(fontsize=6); axs[0].text(0.5, 0.5, "line style = seed (17 solid, 42 dashed, 97 dotted)", transform=axs[0].transAxes, fontsize=5.5, ha="center")
    b.save(fig, "S5_training_curves", dpi=600)
    b.caption("**Supplementary Figure S5. Validation MSE at 50% masking per epoch for the 3-seed family-selection runs (batch 8, 80 epochs).** One line per arm and seed (solid/dashed/dotted: seeds 17/42/97). (A) Full range, y-axis from zero. (B) Zoom on epochs 30-79; the y-axis is truncated to show the late-phase differences and is not zero-based. Several runs reach their best epoch in the last 10 epochs (fixed budget; not converged).")
    b.finish(filters=["validation_curves_all_runs.csv all rows"], caption_stats=["curves are validation MSE at epoch end; no CI"])


# ============================================================ S6 TCGA all metrics
def s6():
    b = new("S6_tcga_all_metrics", "TCGA all metrics x masking fraction")
    d = b.csv(O / "regulatory_confirmation_v1/analysis/per_run_metrics.csv")[["arm", "seed", "mask_fraction", "mse", "mae", "mas_pcc", "mac_pcc"]]
    b.source("metrics", d, {"arm": "arm", "seed": "seed", "mask_fraction": "masking fraction", "mse": "validation MSE", "mae": "validation MAE", "mas_pcc": "MAS-PCC (patient-level)", "mac_pcc": "MAC-PCC (locus-level)"}, "per_run_metrics.csv all rows (incl. legacy control)")
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), gridspec_kw=dict(wspace=0.35, hspace=0.4))
    for k, (ax, (m, l)) in enumerate(zip(axs.flat, [("mse", "MSE"), ("mae", "MAE"), ("mas_pcc", "MAS-PCC"), ("mac_pcc", "MAC-PCC")])):
        lines_with_seeds(ax, d, ALL4, m); ax.set_xticks(FR); ax.set_xlabel("masking fraction"); ax.set_ylabel(l + (" (higher better)" if "pcc" in m else " (lower better)")); panel_letter(ax, "ABCD"[k])
        if m in ("mse", "mae"): ax.set_ylim(0, None)
        else: ax.set_title("y-axis not zero-based" , fontsize=6, color=OI["verm"])
    axs[0, 0].legend(fontsize=5.5, loc="lower right")
    b.save(fig, "S6_tcga_all_metrics"); b.caption("**Supplementary Figure S6. TCGA Phase A validation: all metrics vs masking fraction.** Lines: seed mean; dots: seeds 17/42/97. Dashed grey: legacy Functional PCA sensitivity control. MSE and MAE axes start at zero; correlation panels use a data-range axis.")
    b.finish(filters=["per_run_metrics.csv"], caption_stats=["mean over 3 seeds; dots = seeds"])


# ============================================================ S7 TCGA per-seed paired deltas
def s7():
    b = new("S7_tcga_paired_deltas", "TCGA per-seed paired deltas, all fractions")
    c = b.csv(O / "regulatory_confirmation_v1/analysis/paired_contrasts.csv"); c = c[c.metric == "mse"].copy()
    d = c[["seed", "mask_fraction", "comparator", "relative", "rel_ci_lo", "rel_ci_hi", "n_patients", "n_blocks"]].copy()
    for k in ("relative", "rel_ci_lo", "rel_ci_hi"): d[k] = 100 * d[k]
    d = d.rename(columns={"relative": "relative_pct", "rel_ci_lo": "ci_lo_pct", "rel_ci_hi": "ci_hi_pct"})
    b.source("paired", d, {"seed": "seed", "mask_fraction": "masking fraction", "comparator": "comparator (reference = candidate)", "relative_pct": "100*(comparator - candidate)/candidate MSE", "ci_lo_pct": "paired 95% CI lower (patient x 1 Mb block bootstrap, 2000 rep.)", "ci_hi_pct": "upper", "n_patients": "patients", "n_blocks": "blocks"}, "paired_contrasts.csv metric=mse")
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw=dict(wspace=0.4))
    for k, (ax, a) in enumerate(zip(axs, [CP, DC, LEG])):
        panel_letter(ax, "ABC"[k])
        for sd in (17, 42, 97):
            g = d[(d.comparator == a) & (d.seed == sd)].sort_values("mask_fraction"); off = (sd - 42) / 97 * 0.03
            ax.errorbar(g.mask_fraction + off, g.relative_pct, yerr=[g.relative_pct - g.ci_lo_pct, g.ci_hi_pct - g.relative_pct], fmt=SEED_MARK[sd], color=ARM_COLOR[a], ms=3.5, capsize=1.5, lw=0.8, label=f"seed {sd}")
        ax.axhline(0, color="black", lw=0.6); ax.set_xticks(FR); ax.set_xlabel("masking fraction"); ax.set_title(ARM_LABEL[a], fontsize=7); ax.legend(fontsize=5.5)
    axs[0].set_ylabel("paired relative ΔMSE (%)\n(comparator − candidate)/candidate")
    b.save(fig, "S7_tcga_paired_deltas"); b.caption("**Supplementary Figure S7. TCGA Phase A: per-seed paired relative MSE differences at all masking fractions.** (comparator - candidate)/candidate with 95% CI from the paired patient x 1 Mb block bootstrap (2,000 replicates; bootstrap seed = training seed). Panel C is the legacy Functional PCA sensitivity control (negative values favour the control).")
    b.finish(filters=["paired_contrasts.csv metric=mse"], caption_stats=["CI paired bootstrap, per seed"])


# ============================================================ S8 chromosome robustness
def s8():
    b = new("S8_chr20_22_robustness", "chr20-22 out-of-fit robustness")
    m = b.csv(O / "regulatory_family_screen_v1/analysis/mse_by_fit_scope.csv"); dc = b.csv(O / "regulatory_family_screen_v1/analysis/diagnostic_strata_contrasts.csv")
    dc = dc[(dc.diagnostic == "fit_scope") & (dc.metric == "mse")].copy()
    for k in ("relative", "rel_ci_lo", "rel_ci_hi"): dc[k] *= 100
    dc = dc[["group", "reference", "alternative", "relative", "rel_ci_lo", "rel_ci_hi", "n_pairs", "n_loci" if "n_loci" in dc else "n_blocks"]]
    b.source("mse_by_scope", m, {"arm": "family-screen arm", "group": "in_fit_chr1_19 / out_of_fit_chr20_22", "mse": "validation MSE@0.5", "n_pairs": "pairs", "n_loci": "loci"}, "mse_by_fit_scope.csv")
    b.source("contrasts", dc, {"group": "fit scope", "reference": "reference arm", "alternative": "alternative arm", "relative": "100*(alt-ref)/ref MSE", "rel_ci_lo": "paired CI lower (%)", "rel_ci_hi": "upper (%)", "n_pairs": "pairs", dc.columns[-1]: "blocks"}, "diagnostic_strata_contrasts.csv diagnostic=fit_scope metric=mse")
    arms = list(m.arm.unique()); fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.4)); ax = axs[0]; panel_letter(ax, "A")
    for j, g in enumerate(["in_fit_chr1_19", "out_of_fit_chr20_22"]):
        v = m[m.group == g].set_index("arm").loc[arms].mse; ax.bar(np.arange(len(arms)) + (j - 0.5) * 0.38, v.values, 0.36, color=[OI["sky"] if j == 0 else OI["verm"]] * len(arms), alpha=0.9 if j == 0 else 0.6, edgecolor="black", lw=0.4, label=g)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([a.replace("regulatory_", "") for a in arms], fontsize=6); ax.set_ylim(0, m.mse.max() * 1.15); ax.set_ylabel("validation MSE@50% (1 seed)"); ax.legend(fontsize=6)
    ax = axs[1]; panel_letter(ax, "B"); y = 0; yl = []
    for _, r in dc.iterrows():
        ax.errorbar(r.relative, y, xerr=[[r.relative - r.rel_ci_lo], [r.rel_ci_hi - r.relative]], fmt="o", color=OI["blue"] if "out" not in r.group else OI["verm"], ms=3.5, capsize=2, lw=0.9); yl.append(f"{r.reference.replace('regulatory_','')}→{r.alternative.replace('regulatory_','')} ({'out' if 'out' in r.group else 'in'}-fit)"); y += 1
    ax.axvline(0, color="black", lw=0.6); ax.set_yticks(range(len(yl))); ax.set_yticklabels(yl, fontsize=5.3); ax.set_xlabel("paired relative ΔMSE (%)")
    b.save(fig, "S8_chr20_22_robustness")
    b.note("MISSING: no per-chromosome source exists for TCGA Phase A (regulatory_confirmation_v1/analysis has no chromosome-stratified file); S8 uses the 1-seed family screen only.")
    b.caption("**Supplementary Figure S8. In-fit (chr1-19) vs out-of-fit (chr20-22) loci, family screen (1 seed, batch 32, 30 epochs).** (A) Validation MSE@50% by locus scope. Absolute MSE depends on the locus mix. (B) Paired relative MSE differences by scope with 95% paired patient x 1 Mb block bootstrap CI. chr20-22 were transformed but not used to fit the SVD. No per-chromosome analysis of the Phase A confirmation exists.")
    b.finish(filters=["mse_by_fit_scope.csv; diagnostic_strata_contrasts.csv diagnostic=fit_scope metric=mse"], caption_stats=["CI paired bootstrap patient x 1Mb block"], missing_results=["TCGA Phase A per-chromosome (chr20-22) analysis (no machine-readable source)"])


# ============================================================ S9 external validation + test, all metrics
def s9():
    b = new("S9_external_all_metrics", "GSE40279 all metrics x masking: validation and test", ["external-recon-protocol-freeze-v1.2", "external-recon-test-authorization-v1"])
    A = O / "external_reconstruction_v1/analysis/A_v1.2"
    v = b.csv(A / "validation/per_run_metrics.csv"); t = b.csv(A / "test/per_run_metrics.csv"); v["split"] = "validation"; t["split"] = "test"
    d = pd.concat([v, t])[["split", "arm", "seed", "mask_fraction", "mse", "mae", "mas_pcc", "mac_pcc"]]
    b.source("metrics", d, {"split": "validation or test (separate rows of panels)", "arm": "arm", "seed": "seed", "mask_fraction": "masking fraction", "mse": "MSE", "mae": "MAE", "mas_pcc": "MAS-PCC", "mac_pcc": "MAC-PCC"}, "A_v1.2/{validation,test}/per_run_metrics.csv")
    fig, axs = plt.subplots(2, 4, figsize=(7.6, 4.2), gridspec_kw=dict(wspace=0.5, hspace=0.5))
    for r, sp in enumerate(["validation", "test"]):
        for c, (m, l) in enumerate([("mse", "MSE"), ("mae", "MAE"), ("mas_pcc", "MAS-PCC"), ("mac_pcc", "MAC-PCC")]):
            ax = axs[r, c]; lines_with_seeds(ax, d[d.split == sp], ALL4, m); ax.set_xticks([0.15, 0.5, 0.9]); ax.set_xlabel("masking fraction", fontsize=6); ax.set_ylabel(l, fontsize=6.5)
            if m in ("mse", "mae"): ax.set_ylim(0, None)
            if c == 0: ax.set_title(f"{sp.upper()} split", fontsize=7, loc="left", color=OI["verm"] if sp == "test" else "black", fontweight="bold")
    axs[0, 0].legend(fontsize=4.8, loc="lower right"); panel_letter(axs[0, 0], "A", dx=-0.35); panel_letter(axs[1, 0], "B", dx=-0.35)
    b.save(fig, "S9_external_all_metrics"); b.caption("**Supplementary Figure S9. GSE40279: all metrics vs masking fraction, validation split (row A, used for model selection) and one-shot independent test split (row B).** Lines: seed mean; dots: seeds; dashed grey: legacy control. MSE/MAE axes from zero.")
    b.finish(filters=["A_v1.2 validation and test per_run_metrics"], caption_stats=["mean over seeds, dots seeds"])


# ============================================================ S10 GSE40279 convergence
def s10():
    b = new("S10_external_convergence", "GSE40279 validation MSE vs epoch", ["external-recon-protocol-freeze-v1.2"])
    st = b.json(O / "external_reconstruction_v1/test_eval_status.json"); rows = []
    for r in st["runs"]:
        hp = Path(r["run_dir"]) / "history.json"; h = json.load(open(b.track(hp)))
        for e in h: rows.append(dict(arm=r["arm"], seed=r["seed"], epoch=e["epoch"], validation_mse=e["validation_mse"]))
    d = pd.DataFrame(rows); assert d.groupby(["arm", "seed"]).size().eq(120).all()
    b.source("history", d, {"arm": "arm", "seed": "seed", "epoch": "epoch (0-based)", "validation_mse": "validation MSE@0.50 per epoch (history.json)"}, "history.json of the 12 frozen external runs (run dirs from test_eval_status.json); only validation_mse column used")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.3))
    for k, ax in enumerate(axs):
        panel_letter(ax, "AB"[k])
        for a in ALL4:
            for sd in (17, 42, 97):
                g = d[(d.arm == a) & (d.seed == sd)]; ax.plot(g.epoch, g.validation_mse, color=ARM_COLOR[a], lw=0.8, ls=["-", "--", ":"][[17, 42, 97].index(sd)], label=ARM_LABEL[a] if sd == 17 else None)
        ax.set_xlabel("epoch")
        if k == 0: ax.set_ylim(0, d.validation_mse.max() * 1.05); ax.set_ylabel("validation MSE@50%"); ax.legend(fontsize=5.5)
        else: z = d[d.epoch >= 40]; ax.set_xlim(40, 120); ax.set_ylim(z.validation_mse.min() * 0.998, z.validation_mse.max() * 1.002); ax.set_title("ZOOM epochs 40-119, y truncated", fontsize=7, color=OI["verm"])
    b.save(fig, "S10_external_convergence", dpi=600); b.caption("**Supplementary Figure S10. GSE40279 training convergence: validation MSE@50% vs epoch for every arm and seed (120 epochs).** Line style encodes seed (solid/dashed/dotted: 17/42/97). (B) zoom with truncated y-axis. Only validation values are shown.")
    b.finish(filters=["history.json validation_mse"], caption_stats=["no CI"])


# ============================================================ S11 B-strict / B-recalibrated
def s11():
    b = new("S11_transfer_B", "TCGA to GSE40279 transfer (validation only)", ["external-recon-protocol-freeze-v1.2"])
    A = O / "external_reconstruction_v1/analysis"; fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2), gridspec_kw=dict(wspace=0.38, hspace=0.45)); allm = []; allc = []
    for r, (var, lab) in enumerate([("B_strict_v1.2", "B-strict"), ("B_recalibrated_v1.2", "B-recalibrated")]):
        d = b.csv(A / var / "validation/per_run_metrics.csv"); d["variant"] = lab; allm.append(d[["variant", "arm", "seed", "mask_fraction", "mse", "mae", "mas_pcc", "mac_pcc"]])
        ax = axs[r, 0]; lines_with_seeds(ax, d, MAIN_ARMS, "mse"); ax.set_xticks(FR); ax.set_ylim(0, d[d.arm.isin(MAIN_ARMS)].mse.max() * 1.1); ax.set_xlabel("masking fraction"); ax.set_ylabel(f"{lab} validation MSE"); panel_letter(ax, "AC"[r])
        if r == 0: ax.legend(fontsize=5.5)
        ax = axs[r, 1]; panel_letter(ax, "BD"[r]); y = 0; yl = []; yt = []
        for blk, comp in [("secondary", DC), ("primary", CP)]:
            s = b.csv(A / var / f"validation/{blk}_contrast_per_seed.csv"); s = s[(s.mask_fraction == 0.5) & (s.metric == "mse") & (s.comparator == comp)].sort_values("seed")
            for _, q in s.iterrows():
                allc.append(dict(variant=lab, block=blk, comparator=comp, seed=int(q.seed), relative_pct=100 * q.relative, ci_lo_pct=100 * q.rel_ci_lo, ci_hi_pct=100 * q.rel_ci_hi))
                hol = q.rel_ci_lo <= 0 <= q.rel_ci_hi
                y = forest(ax, [dict(est=100 * q.relative, lo=100 * q.rel_ci_lo, hi=100 * q.rel_ci_hi, marker=SEED_MARK[int(q.seed)], color=ARM_COLOR[comp], hollow=bool(hol))], y); yt.append(y - 1); yl.append(f"{ARM_LABEL[comp].split()[0]} s{int(q.seed)}")
            y += 0.7
        ax.axvline(0, color="black", lw=0.6); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=5.5); ax.set_xlabel("paired relative ΔMSE@50% (%)"); ax.set_title(f"{lab}: (comparator − candidate)/candidate", fontsize=6.5)
    b.source("metrics", pd.concat(allm), {"variant": "B-strict or B-recalibrated", "arm": "arm", "seed": "seed", "mask_fraction": "masking fraction", "mse": "MSE", "mae": "MAE", "mas_pcc": "MAS-PCC", "mac_pcc": "MAC-PCC"}, "B_*_v1.2/validation/per_run_metrics.csv")
    b.source("contrasts", pd.DataFrame(allc), {"variant": "variant", "block": "primary (CpGPT) / secondary (DeepCpG)", "comparator": "comparator", "seed": "seed", "relative_pct": "100*(comparator-candidate)/candidate MSE@0.5", "ci_lo_pct": "paired 95% CI lower", "ci_hi_pct": "upper"}, "validation split; mask 0.5; mse")
    b.save(fig, "S11_transfer_B"); b.caption("**Supplementary Figure S11. Experiment B (TCGA-trained checkpoints evaluated on GSE40279; validation split only; secondary).** Top: B-strict (original TCGA prior); bottom: B-recalibrated (prior recomputed from the 524 GSE40279 training subjects). Left: MSE vs masking fraction (lines = mean, dots = seeds; zero-based). Right: per-seed paired relative MSE difference at 50% masking with 95% paired block bootstrap CI; hollow markers: CI includes 0. Absolute B MSE is not comparable with experiment A or TCGA. The test split was not evaluated for B.")
    b.finish(filters=["B_strict_v1.2 and B_recalibrated_v1.2 validation per_run_metrics; primary/secondary contrasts per seed at mask 0.5 mse"], caption_stats=["paired 95% CI individuals x 1Mb blocks, 2000 replicates"])


# ============================================================ S12 complete biological validation
def s12():
    b = new("S12_bioval_complete", "Complete biological validation including legacy Functional PCA", ["bioval-v2-protocol-freeze-v1"])
    sp = b.csv(BV / "report/summary_primary.csv"); sc = b.csv(BV / "report/summary_contrasts.csv")
    ARMS = ["regulatory_histone_dnase_v1", "cpgpt_large_locus", "deepcpg_dna_locus", "functional_annotations_pca"]
    b.source("primary", sp, {"endpoint": "primary endpoint", "arm": "arm", "value": "point estimate", "ci_lo": "frozen 95% bootstrap CI lower", "ci_hi": "upper"}, "summary_primary.csv all rows")
    pr = sc[sc.stat_class == "primary"][["comparator", "endpoint", "stat", "ref_value", "comp_value", "delta", "ci_lo", "ci_hi", "p", "holm_p_adjusted", "supports_regulatory"]]
    b.source("primary_contrasts", pr, {"comparator": "comparator arm (reference = candidate regulatory_histone_dnase_v1)", "endpoint": "endpoint", "stat": "statistic", "ref_value": "candidate value", "comp_value": "comparator value", "delta": "candidate - comparator",
                                         "ci_lo": "frozen paired 95% bootstrap CI lower", "ci_hi": "upper", "p": "raw permutation/bootstrap p", "holm_p_adjusted": "Holm-adjusted p over the 12-contrast primary family", "supports_regulatory": "frozen flag: contrast supports the candidate"}, "summary_contrasts.csv stat_class=primary")
    write_table(b, "S12_bioval_primary_contrasts", pr.round(4), "Primary biological-validation contrasts (candidate minus comparator), Holm-adjusted p values.", "tab:s12", {c: c for c in pr.columns}, colfmt="l" * len(pr.columns))
    EPS = [("microc_H1_intra10kb", "H1 Micro-C AUROC"), ("fantom5_membership", "FANTOM5 membership AUROC"), ("rt_consensus", "Replication timing ρ"), ("loyfer_profile", "Loyfer long-range ρ")]
    fig, axs = plt.subplots(1, 4, figsize=(8.0, 3.0), gridspec_kw=dict(wspace=0.7))
    for k, (ax, (ep, t)) in enumerate(zip(axs, EPS)):
        d = sp[sp.endpoint == ep].set_index("arm").loc[ARMS]
        for i, a in enumerate(ARMS):
            r = d.loc[a]; ak = {"regulatory_histone_dnase_v1": "regulatory_histone_dnase_v1"}.get(a, a)
            ax.errorbar(i, r.value, yerr=[[r.value - r.ci_lo], [r.ci_hi - r.value]], fmt=ARM_MARK[ak], color=ARM_COLOR[ak], ms=5, capsize=2.5, mfc="white" if a == LEG else ARM_COLOR[ak])
        ax.axhline(0.5 if "AUROC" in t else 0, color="black", ls=":", lw=0.7); ax.set_xticks(range(4)); ax.set_xticklabels(["Hist+\nDNase", "CpGPT", "DeepCpG", "Func.\nPCA\n(legacy)"], fontsize=5.5); ax.set_title(t, fontsize=6.8); panel_letter(ax, "ABCD"[k], dx=-0.3)
        if "AUROC" in t: ax.text(0.02, 0.02, "axis not zero-based", transform=ax.transAxes, fontsize=5, style="italic")
    b.save(fig, "S12_bioval_complete")
    b.caption("**Supplementary Figure S12. Complete frozen biological validation including the legacy Functional PCA control.** All four primary endpoints, all four arms (open marker: legacy control). Points: statistic; bars: frozen 95% bootstrap CI; dotted line: chance. Paired primary contrasts with Holm-adjusted p values are in the accompanying table (Holm family: 12 contrasts; adjusted p floor 0.008 with 1000 bootstrap replicates). The candidate is lower than both sequence models on the Loyfer endpoint.")
    b.finish(filters=["summary_primary.csv all arms; summary_contrasts.csv stat_class=primary"], caption_stats=["CI frozen paired bootstrap (1000 replicates, seed 17)", "delta = candidate - comparator"])


# ============================================================ S13 sensitivities
def s13():
    b = new("S13_bioval_sensitivities", "Frozen biological sensitivities", ["bioval-v2-protocol-freeze-v1"])
    s = b.csv(BV / "sensitivity/SUMMARY.csv"); b.track(BV / "sensitivity/SUMMARY.json")
    s["final_status"] = np.where(s.status == "done", s.verdict, "NOT_EXECUTED_REQUIRES_NEW_PREREGISTRATION_OR_ARTIFACTS")
    t = s[["item", "status", "protocol_status", "final_status", "ordering_identical"]].copy()
    t["status"] = t.status.str.slice(0, 70)
    b.source("verdicts", t, {"item": "sensitivity item id", "status": "done, or truncated reason it was not run (SUMMARY.csv)", "protocol_status": "protocol status (SUMMARY.csv)", "final_status": "verdict CONFIRMS/CHANGES/INCONCLUSIVE for executed rows; NOT_EXECUTED_REQUIRES_NEW_PREREGISTRATION_OR_ARTIFACTS otherwise", "ordering_identical": "arm ordering identical to primary (executed rows)"}, "sensitivity/SUMMARY.csv all rows")
    write_table(b, "S13_bioval_sensitivities", t, "Frozen biological sensitivity analyses. Executed rows: verdict relative to the frozen primary; the five remaining items were not executed.", "tab:s13", {c: c for c in t.columns}, colfmt="lllll")
    fig, ax = plt.subplots(figsize=(7.2, 4.6)); ax.axis("off"); col = {"CONFIRMS": "#B7E1CD", "CHANGES": "#F4B6B6", "INCONCLUSIVE": "#FFE8A3"}
    for i, (_, r) in enumerate(t.iterrows()):
        c = col.get(r.final_status, "#DDDDDD"); yy = 1 - (i + 1) / (len(t) + 1)
        ax.add_patch(plt.Rectangle((0.52, yy - 0.02), 0.46, 0.04, transform=ax.transAxes, fc=c, ec="black", lw=0.4)); ax.text(0.02, yy, r["item"], transform=ax.transAxes, fontsize=6.3, va="center")
        ax.text(0.75, yy, r.final_status.replace("NOT_EXECUTED_REQUIRES_NEW_PREREGISTRATION_OR_ARTIFACTS", "NOT_EXECUTED (needs new prereg./artifacts)"), transform=ax.transAxes, fontsize=5.8, va="center", ha="center")
    ax.text(0.02, 1.02, "Frozen biological sensitivities (per-row verdict)", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    b.save(fig, "S13_bioval_sensitivities"); b.caption("**Supplementary Figure/Table S13. Frozen biological sensitivity analyses.** Per-row verdict of each pre-registered sensitivity relative to the frozen primary conclusion (CONFIRMS / CHANGES / INCONCLUSIVE). Five items were not executed because they require new pre-registration or new artifacts (label NOT_EXECUTED_REQUIRES_NEW_PREREGISTRATION_OR_ARTIFACTS).")
    b.finish(filters=["sensitivity/SUMMARY.csv"], caption_stats=["verdict as computed by the frozen sensitivity runner"])
    b.note(f"n executed={int((s.status=='done').sum())}, n not executed={int((s.status!='done').sum())}")


# ============================================================ S14 exploratory post-hoc Loyfer diagnostics (already frozen)
def s14():
    b = new("S14_loyfer_exploratory_posthoc", "EXPLORATORY POST-HOC Loyfer diagnostics", ["bioval-v2-protocol-freeze-v1"])
    E = BV / "exploratory_posthoc_loyfer"
    g = b.csv(E / "geometry_within_arm_deltas.csv"); r = b.csv(E / "ridge_summary_by_arm.csv")
    g = g[(g.cell == "all") & (g.stat_family == "spearman")][["arm", "geometry", "delta", "ci_lo", "ci_hi", "p", "B_eff"]]
    rr = r[(r.kind == "model")][["arm", "target", "subset", "r2_global", "r2_macro"]]
    b.source("geometry", g, {"arm": "arm", "geometry": "centered_cosine / pc1_removed_cosine / neg_euclidean", "delta": "within-arm geometry minus cosine, Spearman over all pairs (EXPLORATORY POST-HOC)", "ci_lo": "bootstrap CI lower", "ci_hi": "upper", "p": "raw p (no Holm)", "B_eff": "bootstrap replicates"}, "geometry_within_arm_deltas.csv cell=all stat_family=spearman")
    b.source("ridge", rr, {"arm": "arm", "target": "raw / centered / level", "subset": "all_cpgs / variable_cpgs", "r2_global": "ridge R^2 (global)", "r2_macro": "ridge R^2 (macro)"}, "ridge_summary_by_arm.csv kind=model")
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw=dict(wspace=0.4)); ARMS = ["regulatory_histone_dnase_v1", "functional_annotations_pca", "cpgpt_large_locus", "deepcpg_dna_locus"]
    ax = axs[0]; panel_letter(ax, "A"); geos = ["centered_cosine", "pc1_removed_cosine", "neg_euclidean"]
    for i, a in enumerate(ARMS):
        for j, ge in enumerate(geos):
            q = g[(g.arm == a) & (g.geometry == ge)].iloc[0]; ax.errorbar(i + (j - 1) * 0.2, q.delta, yerr=[[q.delta - q.ci_lo], [q.ci_hi - q.delta]], fmt=["o", "s", "^"][j], color=ARM_COLOR[a if a != "regulatory_histone_dnase_v1" else a], ms=4, capsize=2, mfc="white" if j == 1 else ARM_COLOR[a])
    ax.axhline(0, color="black", lw=0.6); ax.set_xticks(range(4)); ax.set_xticklabels(["Hist+\nDNase", "Func.\nPCA", "CpGPT", "DeepCpG"], fontsize=6); ax.set_ylabel("Δ Spearman (geometry − cosine)"); ax.set_title("markers: centered (o), PC1-removed (square), −Euclid. (triangle)", fontsize=5.8)
    ax = axs[1]; panel_letter(ax, "B"); targets = ["raw", "centered", "level"]
    for i, a in enumerate(ARMS):
        for j, tg in enumerate(targets):
            q = rr[(rr.arm == a) & (rr.target == tg) & (rr.subset == "variable_cpgs")]
            if len(q): ax.bar(i + (j - 1) * 0.26, q.r2_global.iloc[0], 0.24, color=ARM_COLOR[a], alpha=[0.95, 0.65, 0.35][j], edgecolor="black", lw=0.4)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Hist+\nDNase", "Func.\nPCA", "CpGPT", "DeepCpG"], fontsize=6); ax.set_ylabel("ridge R$^2$ (variable CpGs)"); ax.set_title("bars: target raw / centered / level (dark to light)", fontsize=6)
    fig.suptitle("EXPLORATORY POST-HOC (already frozen; not part of the primary or Fig 5)", color=OI["verm"], fontsize=8, fontweight="bold", y=1.02)
    b.save(fig, "S14_loyfer_exploratory_posthoc"); b.caption("**Supplementary Figure S14. EXPLORATORY POST-HOC Loyfer diagnostics (frozen exploratory results; not pre-registered).** (A) Change in the Loyfer long-range Spearman (all strata) when replacing raw cosine by centered cosine, PC1-removed cosine or negative Euclidean similarity, per arm, with bootstrap 95% CI; raw p without multiplicity control, descriptive only. (B) Ridge-probe R^2 for predicting methylation profile (variable CpGs) for raw/centered/level targets. These do not amend the primary Loyfer result.")
    b.finish(filters=["exploratory_posthoc_loyfer/geometry_within_arm_deltas.csv cell=all, stat_family=spearman", "ridge_summary_by_arm.csv kind=model"], caption_stats=["exploratory post-hoc; descriptive; no multiplicity control"])


def sfinal():
    b = new("S_final_loyfer_audit_placeholder", "Loyfer audit (pending)")
    placeholder_figure(b, "S_final_loyfer_audit_placeholder", "Supplement S-final: Loyfer failure audit", ["Exploratory post-hoc audit in progress / paused", "Will be populated only after the audit is final and frozen", "(no audit output was read)"])
    b.caption("**Supplementary Figure S-final (placeholder). Loyfer failure audit.** Pending: exploratory post-hoc audit not final. No data.")
    b.finish(placeholder=True, missing_results=["Loyfer failure audit final outputs"])


if __name__ == "__main__":
    for f in (s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, sfinal):
        f()
