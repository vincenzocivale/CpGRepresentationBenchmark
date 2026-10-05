"""Figure 1: regulatory representation design + discovery/selection evidence."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa: F401,F403,E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

b = Build("fig1", __file__, title="Regulatory representation design, family attribution and family selection")
O = ROOT / "outputs"

# ------------------------------------------------------------------ 1A schematic counts (read, not typed)
fz = b.json(ROOT / "configs/frozen/regulatory_histone_dnase_v1.json")
arms_json = b.json(O / "encode_atlas_v1/arms.json")
tpa = fz["feature_set"]["tracks_per_assay"]
n_hist, n_dnase = int(tpa["Histone ChIP-seq"]), int(tpa["DNase-seq"])
n_fit = int(fz["fit_loci"]["n"])
dim = int(fz["embedding"]["shape"][1])
n_univ = int(fz["embedding"]["shape"][0])
b.check("histone tracks (frozen config vs encode arms.json)", len(arms_json["only/assay/Histone ChIP-seq"]["tracks"]), n_hist, 0, "encode_atlas_v1/arms.json")
b.check("DNase tracks (frozen config vs encode arms.json)", len(arms_json["only/assay/DNase-seq"]["tracks"]), n_dnase, 0, "encode_atlas_v1/arms.json")
b.check("histone tracks = 1,959", 1959, n_hist, 0)
b.check("DNase tracks = 533", 533, n_dnase, 0)
b.check("discovery fit loci chr1-19 = 388,599", 388599, n_fit, 0)
b.check("embedding dim = 256", 256, dim, 0)
b.source("A_counts", pd.DataFrame([
    dict(item="histone_chipseq_tracks", value=n_hist), dict(item="dnase_seq_tracks", value=n_dnase),
    dict(item="total_tracks", value=n_hist + n_dnase), dict(item="svd_dim", value=dim),
    dict(item="fit_loci_discovery_chr1_19", value=n_fit), dict(item="loci_in_store", value=n_univ),
    dict(item="dense_columns", value=int(fz["dense_columns"])), dict(item="beta_values_used_0_false", value=int(bool(fz["beta_values_used"])))]),
    {"item": "quantity shown in the schematic", "value": "count read from configs/frozen/regulatory_histone_dnase_v1.json"},
    "configs/frozen/regulatory_histone_dnase_v1.json: feature_set.tracks_per_assay, fit_loci.n, embedding.shape")

# ------------------------------------------------------------------ 1B ENCODE attribution (campaign confirm stage, 3 seeds)
rm = b.csv(O / "encode_atlas_v1/reconstruction_metrics.csv")
pc = b.csv(O / "encode_atlas_v1/paired_comparisons.csv")
B_ARMS = [("context_only", "Context only"), ("add/assay/Histone ChIP-seq", "Context +\nHistone"),
          ("drop/assay/Histone ChIP-seq", "Full −\nHistone"), ("full", "Full\nfunctional"),
          ("fm/cpgpt_locus_large", "CpGPT-large\n(SVD-256)"), ("fm/deepcpg_dna_locus", "DeepCpG-DNA")]
sel = rm[(rm.stage == "confirm") & (~rm.ood) & (~rm.pilot) & (rm.view == "seen") & (rm["mask"] == "mask_0.50")
         & rm.arm.isin([a for a, _ in B_ARMS])][["arm", "seed", "mse", "mae", "n_pairs"]].copy()
assert sel.groupby("arm").seed.nunique().eq(3).all(), "need 3 seeds per arm"
full_by_seed = sel[sel.arm == "full"].set_index("seed").mse
pcs = pc[(pc.stage == "confirm") & (pc.view == "seen") & (~pc.ood) & pc.arm.isin([a for a, _ in B_ARMS])].copy()
import json as _json  # noqa: E402
pcs["rel_ci_lo"] = pcs.relative_ci95.apply(lambda s: _json.loads(s)[0])
pcs["rel_ci_hi"] = pcs.relative_ci95.apply(lambda s: _json.loads(s)[1])
sel = sel.merge(pcs[["arm", "seed", "relative_delta_mse", "rel_ci_lo", "rel_ci_hi"]], on=["arm", "seed"], how="left")
sel["arm_label"] = sel.arm.map(dict((a, l.replace("\n", " ")) for a, l in B_ARMS))
colsB = {"arm": "campaign arm id (encode_atlas_v1)", "seed": "optimisation seed", "mse": "MSE at 50% masking on the campaign evaluation patients (reconstruction_metrics.csv, mask_0.50)",
         "mae": "MAE at 50% masking", "n_pairs": "evaluated (patient, locus) pairs",
         "relative_delta_mse": "(arm - full)/full paired relative MSE difference (paired_comparisons.csv); NaN for the reference arm 'full'",
         "rel_ci_lo": "lower bound of paired 95% bootstrap CI (patient x 1 Mb block) of relative_delta_mse",
         "rel_ci_hi": "upper bound of the same CI", "arm_label": "plot label"}
b.source("B", sel.sort_values(["arm", "seed"]), colsB,
         "stage=confirm, ood=False, pilot=False, view=seen, mask=mask_0.50; paired: arm vs full, same filters")

# ------------------------------------------------------------------ 1C family selection
fs = b.csv(O / "regulatory_family_screen_v1/analysis/per_arm_metrics.csv")[["arm", "mse", "mae", "best_epoch", "n_epochs"]]
fs["protocol"] = "family_screen_1seed_batch32_30ep"
fs["seed"] = 17
fsc = b.csv(O / "regulatory_family_screen_v1/analysis/criterion.csv")
cf = b.csv(O / "regulatory_confirm_v1/analysis/per_run_metrics.csv")[["arm", "seed", "mse", "mae", "best_epoch", "epochs_run", "not_converged"]]
cf = cf.rename(columns={"epochs_run": "n_epochs"})
cf["protocol"] = "confirm_3seeds_batch8_80ep"
C_ARMS = ["regulatory_histone", "regulatory_histone_dnase", "regulatory_histone_tf", "regulatory_clean"]
C_LAB = {"regulatory_histone": "Histone", "regulatory_histone_dnase": "Histone\n+DNase", "regulatory_histone_tf": "Histone\n+TF", "regulatory_clean": "Clean"}
assert set(fs.arm) == set(C_ARMS), fs.arm.tolist()
b.source("C_family_screen", fs, {"arm": "regulatory family-screen arm", "mse": "validation MSE at 50% masking of the best checkpoint", "mae": "validation MAE",
                                 "best_epoch": "0-based best epoch", "n_epochs": "epochs run", "protocol": "protocol tag (1 seed, batch 32, 30 epochs)", "seed": "seed (single)"},
         "regulatory_family_screen_v1/analysis/per_arm_metrics.csv, all 4 arms")
b.source("C_confirmation", cf, {"arm": "arm", "seed": "seed", "mse": "validation MSE at 50% masking (best checkpoint)", "mae": "validation MAE",
                                "best_epoch": "0-based best epoch", "n_epochs": "epochs run (80 = fixed budget)", "not_converged": "True if best epoch in last 10 epochs",
                                "protocol": "protocol tag (3 seeds, batch 8, 80 epochs)"},
         "regulatory_confirm_v1/analysis/per_run_metrics.csv, arms histone/+DNase/clean (no +TF in confirmation)")

# ------------------------------------------------------------------ 1D forest
pcn = b.csv(O / "regulatory_confirm_v1/analysis/paired_contrasts.csv")
sm = b.csv(O / "regulatory_confirm_v1/analysis/across_seeds_summary.csv")
CONTRASTS = [("regulatory_histone", "regulatory_histone_dnase", "Histone → Histone+DNase"),
             ("regulatory_histone_dnase", "regulatory_clean", "Histone+DNase → Clean")]
rows = []
for ref, alt, lab in CONTRASTS:
    x = pcn[(pcn.reference == ref) & (pcn.alternative == alt) & (pcn.metric == "mse")].sort_values("seed")
    for _, r in x.iterrows():
        rows.append(dict(contrast=lab, reference=ref, alternative=alt, seed=int(r.seed), relative_pct=100 * r.relative,
                         ci_lo_pct=100 * r.rel_ci_lo, ci_hi_pct=100 * r.rel_ci_hi, n_patients=int(r.n_patients), n_blocks=int(r.n_blocks), replicates=int(r.replicates), bootstrap_seed=int(r.bootstrap_seed)))
    s = sm[(sm.reference == ref) & (sm.alternative == alt) & (sm.metric == "mse")].iloc[0]
    rows.append(dict(contrast=lab, reference=ref, alternative=alt, seed=-1, relative_pct=100 * s.mean_relative, ci_lo_pct=np.nan, ci_hi_pct=np.nan,
                     n_patients=np.nan, n_blocks=np.nan, replicates=np.nan, bootstrap_seed=np.nan))
D = pd.DataFrame(rows)
colsD = {"contrast": "paired contrast (reference -> alternative)", "reference": "reference arm", "alternative": "alternative arm", "seed": "optimisation seed; -1 = unweighted mean over the 3 seeds (across_seeds_summary.mean_relative)",
         "relative_pct": "100*(alternative - reference)/reference of validation MSE@0.5; negative favours the alternative",
         "ci_lo_pct": "lower bound, paired 95% CI (patient x 1 Mb block bootstrap), percent; NaN for the mean row", "ci_hi_pct": "upper bound",
         "n_patients": "patients resampled", "n_blocks": "1 Mb genomic blocks resampled", "replicates": "bootstrap replicates", "bootstrap_seed": "bootstrap RNG seed"}
b.source("D", D, colsD, "metric=mse; reference/alternative as listed; seeds 17/42/97; mean row from across_seeds_summary.csv")
m1 = D[(D.seed == -1)].relative_pct.tolist()
b.check("Histone->Histone+DNase mean rel dMSE (%)", -1.75, m1[0], 0.01)
b.check("Histone+DNase->Clean mean rel dMSE (%)", -0.51, m1[1], 0.01)

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(7.4, 9.2))
gs = fig.add_gridspec(3, 8, height_ratios=[0.6, 1.2, 0.95], hspace=0.5, wspace=1.4, top=0.975, bottom=0.06, left=0.09, right=0.985)
# ---- 1A
axA = fig.add_subplot(gs[0, :])
axA.set_xlim(0, 100); axA.set_ylim(0, 22); axA.axis("off")
panel_letter(axA, "A", dx=-0.04, dy=0.98)


def box(x, y, w, h, text, fc="white", ec="black", fs=7, bold=False, ls="-"):
    axA.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=0.9, ls=ls))
    axA.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(x0, y0, x1, y1):
    axA.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="black"))


box(1, 8, 14, 6, "CpG locus\n(genomic pos.)", fc="#EEEEEE")
box(20, 7.5, 17, 7, "ENCODE experimental\nannotations\n(binary peak overlap)", fs=6.5, fc="#EAF3FA")
box(43, 13, 20, 5.2, f"Histone ChIP-seq\n{n_hist:,} tracks", fc=OI["sky"] + "55")
box(43, 3.8, 20, 5.2, f"DNase-seq\n{n_dnase:,} tracks", fc=OI["blue"] + "40")
box(69, 8, 14, 6, "Global\nTruncatedSVD", fc="white")
box(88, 8, 11, 6, f"{dim}-D CpG\nembedding", fc=OI["blue"] + "55", bold=True)
arrow(14.5, 11, 19.5, 11); arrow(36.5, 12, 42.5, 15.5); arrow(36.5, 10, 42.5, 6.4)
arrow(63.5, 15.5, 68.5, 12.5); arrow(63.5, 6.4, 68.5, 9.5); arrow(83.5, 11, 87.5, 11)
axA.text(50, 0.4, "patient-independent  |  no beta values  |  no dense genomic annotations  |  "
         f"unsupervised, fit on discovery chr1–19 ({n_fit:,} loci); chr20–22 transformed only",
         ha="center", va="center", fontsize=6.2, style="italic")
axA.text(50, 20.7, "Regulatory (ENCODE-track-only) locus representation: Histone+DNase, native frozen", ha="center", fontsize=8.5, fontweight="bold")

# ---- 1B
axB = fig.add_subplot(gs[1, 0:4])
panel_letter(axB, "B", dx=-0.2)
x = np.arange(len(B_ARMS))
tops = []
for i, (a, lab) in enumerate(B_ARMS):
    v = sel[sel.arm == a].sort_values("seed")
    axB.bar(i, v.mse.mean(), color=DISC_COLOR[a], edgecolor="black", lw=0.6, width=0.7, alpha=0.9)
    axB.scatter(i + jitter(len(v)), v.mse, s=9, c="black", zorder=3, linewidths=0)
    rel_ = v.relative_delta_mse.mean()
    tops.append(v.mse.max())
    if a != "full":
        lo, hi = v.rel_ci_lo.mean(), v.rel_ci_hi.mean()
        axB.text(i, v.mse.max() * 1.015, f"{100 * rel_:+.1f}%", ha="center", va="bottom", fontsize=6.5)
    else:
        axB.text(i, v.mse.max() * 1.015, "ref.", ha="center", va="bottom", fontsize=6.5)
axB.set_xticks(x); axB.set_xticklabels([l for _, l in B_ARMS], fontsize=5.6, rotation=30, ha="right")
axB.set_ylim(0, max(tops) * 1.10)
axB.set_ylabel("MSE at 50% masking (3 seeds)")
axB.set_title("ENCODE family attribution\nDISCOVERY EVIDENCE (campaign), not confirmation", fontsize=7.5, color=OI["verm"], fontweight="bold")
axB.set_xlabel("bars: seed mean; dots: seeds 17/42/97; labels: mean paired \u0394 vs Full, (arm\u2212Full)/Full", fontsize=6)

# ---- 1C
axC1 = fig.add_subplot(gs[1, 4:6])
axC2 = fig.add_subplot(gs[1, 6:8])
panel_letter(axC1, "C", dx=-0.5)
for i, a in enumerate(C_ARMS):
    v = fs[fs.arm == a].mse.iloc[0]
    axC1.bar(i, v, color=DISC_COLOR[a], edgecolor="black", lw=0.6, width=0.7)
    axC1.scatter([i], [v], s=9, c="black", zorder=3, linewidths=0)
axC1.set_xticks(range(4)); axC1.set_xticklabels([C_LAB[a] for a in C_ARMS], fontsize=6)
axC1.set_ylim(0, fs.mse.max() * 1.15); axC1.set_ylabel("validation MSE at 50% masking")
axC1.set_title("family screen\n1 seed, batch 32, 30 ep.", fontsize=7)
C3 = ["regulatory_histone", "regulatory_histone_dnase", "regulatory_clean"]
for i, a in enumerate(C3):
    v = cf[cf.arm == a].sort_values("seed")
    axC2.bar(i, v.mse.mean(), color=DISC_COLOR[a], edgecolor="black", lw=0.6, width=0.7)
    axC2.scatter(i + jitter(len(v)), v.mse, s=9, c="black", zorder=3, linewidths=0)
axC2.set_xticks(range(3)); axC2.set_xticklabels([C_LAB[a] for a in C3], fontsize=6)
axC2.set_ylim(0, cf.mse.max() * 1.15)
axC2.set_title("confirmation\n3 seeds, batch 8, 80 ep.", fontsize=7)
axC2.set_ylabel("")
fig.text(0.74, 0.545, "Absolute MSE differs between the two protocols (separate axes; not comparable)", fontsize=5.8, ha="center", style="italic")

# ---- 1D forest (below, in remaining space: use inset-like axes under C? place as separate row via new axes)
axD = fig.add_subplot(gs[2, 1:7])
panel_letter(axD, "D", dx=-0.16, dy=1.12)
ypos, ylabs = [], []
y = 0
for lab in [c[2] for c in CONTRASTS][::-1]:
    sub = D[D.contrast == lab]
    for _, r in sub[sub.seed != -1].iterrows():
        axD.errorbar(r.relative_pct, y, xerr=[[r.relative_pct - r.ci_lo_pct], [r.ci_hi_pct - r.relative_pct]], fmt=SEED_MARK[int(r.seed)], color=OI["blue"],
                     ms=4, capsize=2, lw=1)
        ypos.append(y); ylabs.append(f"seed {int(r.seed)}"); y += 1
    mrow = sub[sub.seed == -1].iloc[0]
    axD.plot(mrow.relative_pct, y, marker="D", color="black", ms=5)
    axD.text(mrow.relative_pct, y + 0.45, f"mean {mrow.relative_pct:+.2f}%", ha="center", fontsize=6.5)
    ypos.append(y); ylabs.append("mean of seeds"); y += 1.8
axD.axvline(0, color="black", lw=0.7)
axD.set_yticks(ypos); axD.set_yticklabels(ylabs, fontsize=6)
axD.set_xlabel("paired ΔMSE@50% (%), alternative vs reference; <0 favours alternative")
axD.set_ylim(-0.8, y+0.3)
axD.text(0.01, 0.97, "Histone \u2192 Histone+DNase", transform=axD.transAxes, ha="left", va="top", fontsize=6.5, fontweight="bold")
axD.text(0.01, 0.45, "Histone+DNase \u2192 Clean", transform=axD.transAxes, ha="left", va="top", fontsize=6.5, fontweight="bold")
b.save(fig, "fig1")

hist = b.checks
b.caption(f"""**Figure 1. Regulatory (ENCODE-track-only) CpG-locus representation: design, family attribution and family selection.**
(A) Schematic. The locus representation is built only from patient-independent ENCODE experimental annotations ({n_hist:,} Histone ChIP-seq and {n_dnase:,} DNase-seq tracks; binary peak overlap), compacted by an unsupervised global TruncatedSVD to {dim} dimensions; no methylation values and no dense genomic annotations are used; the SVD is fit on discovery chr1-19 ({n_fit:,} loci) and chr20-22 are transformed only.
(B) DISCOVERY EVIDENCE (ENCODE attribution campaign, not the frozen confirmation): MSE at 50% masking, 3 optimisation seeds (17, 42, 97; dots), bars show the seed mean; y-axis starts at zero. Labels give the mean paired relative difference vs the Full functional representation, (arm - Full)/Full, from the paired patient x 1 Mb genomic-block bootstrap (2,000 replicates; per-seed CIs in the source data). CpGPT-large is the SVD-256 compact variant used in the campaign. Evaluation split of the campaign differs from the Phase-A validation split, so absolute values are not comparable with Figure 2.
(C) Family selection. Left: family screen (1 seed, batch 32, 30 epochs; four arms Histone, +DNase, +TF, Clean). Right: 3-seed confirmation (batch 8, 80 epochs; Histone, +DNase, Clean; dots are seeds). The two sub-panels use separate axes because absolute MSE is not comparable across protocols.
(D) Paired relative MSE difference (alternative - reference)/reference at 50% masking per seed with 95% paired patient x 1 Mb block bootstrap CI (2,000 replicates; bootstrap seed = optimisation seed), and the unweighted mean over seeds (diamond; mean = {m1[0]:+.2f}% and {m1[1]:+.2f}%). Negative values favour the alternative. Clean's small MSE advantage over Histone+DNase is not evidence of equivalence.""")
b.finish(filters=["1B: encode_atlas_v1 reconstruction_metrics.csv stage=confirm, ood=False, pilot=False, view=seen, mask_0.50; arms context_only, add/assay/Histone ChIP-seq, drop/assay/Histone ChIP-seq, full, fm/cpgpt_locus_large, fm/deepcpg_dna_locus; paired_comparisons.csv same filters (arm vs full)",
                  "1C: regulatory_family_screen_v1 per_arm_metrics.csv (4 arms); regulatory_confirm_v1 per_run_metrics.csv (3 arms x 3 seeds)",
                  "1D: regulatory_confirm_v1 paired_contrasts.csv metric=mse; across_seeds_summary.csv mean_relative"],
         caption_stats=["MSE: mean squared error on masked methylation values at 50% masking", "relative delta = (alternative-reference)/reference; CI = paired 95% percentile bootstrap resampling patients and 1 Mb genomic blocks, 2000 replicates",
                        "bars = seed means, dots = individual seeds (N=3), no significance stars", "1C protocols are not comparable on a common axis"])
