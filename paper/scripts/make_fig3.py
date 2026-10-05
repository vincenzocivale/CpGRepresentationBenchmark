"""Figure 3: GSE40279 independent confirmation (one-shot TEST split only)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa: F401,F403,E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

b = Build("fig3", __file__, tags=["external-recon-protocol-freeze-v1.2", "external-recon-test-authorization-v1"],
          title="GSE40279 independent confirmation (one-shot test)")
T = ROOT / "outputs/external_reconstruction_v1/analysis/A_v1.2/test"
tcga = ROOT / "outputs/regulatory_confirmation_v1/analysis"
man = b.json(ROOT / "configs/external/gse40279_v1_split_manifest.json")
b.track(ROOT / "configs/external/gse40279_v1_freeze_manifest.json")
cnt = man["counts"]; ntot = sum(cnt.values())
b.check("GSE40279 individuals total = 656", 656, ntot, 0); b.check("train=524", 524, cnt["train"], 0); b.check("val=66", 66, cnt["validation"], 0); b.check("test=66", 66, cnt["test"], 0)
b.source("A_split", pd.DataFrame([dict(split=k, n_individuals=v) for k, v in [("total", ntot), ("train", cnt["train"]), ("validation", cnt["validation"]), ("test", cnt["test"])]]),
         {"split": "split name", "n_individuals": "unique individuals (configs/external/gse40279_v1_split_manifest.json counts)"}, "counts block")
pr = b.csv(T / "per_run_metrics.csv")
prim_s = b.csv(T / "primary_contrast_per_seed.csv"); prim_a = b.csv(T / "primary_contrast_across_seeds.csv")
sec_s = b.csv(T / "secondary_contrast_per_seed.csv"); sec_a = b.csv(T / "secondary_contrast_across_seeds.csv")
FR = [0.15, 0.3, 0.5, 0.7, 0.9]
CAND, CP, DC = MAIN_ARMS
d = pr[pr.arm.isin(MAIN_ARMS)][["arm", "seed", "mask_fraction", "mse", "n_pairs"]].copy()
assert len(d) == 45
b.source("B", d, {"arm": "arm", "seed": "seed", "mask_fraction": "masking fraction", "mse": "GSE40279 TEST MSE (one-shot, frozen best.pt)", "n_pairs": "evaluated pairs"}, "A_v1.2/test/per_run_metrics.csv, main arms")

rows = []
for tag, ps, pa, comp in [("primary", prim_s, prim_a, CP), ("secondary", sec_s, sec_a, DC)]:
    x = ps[(ps.mask_fraction == 0.5) & (ps.metric == "mse") & (ps.comparator == comp)].sort_values("seed")
    for _, r in x.iterrows():
        rows.append(dict(block=tag, comparator=comp, seed=int(r.seed), relative_pct=100 * r.relative, ci_lo_pct=100 * r.rel_ci_lo, ci_hi_pct=100 * r.rel_ci_hi, ci_includes_zero=bool(r.rel_ci_lo <= 0 <= r.rel_ci_hi), n_patients=int(r.n_patients), n_blocks=int(r.n_blocks)))
    s = pa[(pa.mask_fraction == 0.5) & (pa.metric == "mse") & (pa.comparator == comp)].iloc[0]
    rows.append(dict(block=tag, comparator=comp, seed=-1, relative_pct=100 * s.mean_relative, ci_lo_pct=np.nan, ci_hi_pct=np.nan, ci_includes_zero=np.nan, n_patients=np.nan, n_blocks=np.nan))
Cdf = pd.DataFrame(rows)
colsC = {"block": "registered role: primary (CpGPT) / secondary (DeepCpG)", "comparator": "comparator arm", "seed": "seed; -1 = mean over seeds of the per-seed relative differences",
         "relative_pct": "100*(comparator - candidate)/candidate MSE@0.50, TEST split", "ci_lo_pct": "paired 95% CI lower (patient x 1 Mb block bootstrap, 2000 replicates, 66 test individuals)", "ci_hi_pct": "upper",
         "ci_includes_zero": "True if CI contains 0", "n_patients": "individuals resampled", "n_blocks": "genomic blocks resampled"}
b.source("C", Cdf, colsC, "mask_fraction=0.5, metric=mse, test split; primary_/secondary_contrast_per_seed.csv, across_seeds")
mp = Cdf[(Cdf.block == "primary") & (Cdf.seed == -1)].relative_pct.iloc[0]; ms = Cdf[(Cdf.block == "secondary") & (Cdf.seed == -1)].relative_pct.iloc[0]
b.check("external test rel dMSE@0.50 CpGPT (%)", 1.82, mp, 0.01); b.check("external test rel dMSE@0.50 DeepCpG (%)", 6.13, ms, 0.01)
nz = int(Cdf[(Cdf.block == "primary") & (Cdf.seed != -1)].ci_includes_zero.sum())
b.note(f"{nz}/3 per-seed CIs of the primary (CpGPT) contrast include 0 at 0.50")

# 3D
t_s = pd.read_csv(b.track(tcga / "across_seeds_summary.csv"))
rowsD = []
for comp in (CP, DC):
    x = t_s[(t_s.comparator == comp) & (t_s.mask_fraction == 0.5) & (t_s.metric == "mse")].iloc[0]
    rowsD.append(dict(cohort="TCGA (Phase A validation)", comparator=comp, mean_relative_pct=100 * x.mean_relative, min_pct=100 * x.min_relative, max_pct=100 * x.max_relative))
    e = Cdf[(Cdf.comparator == comp)]
    rowsD.append(dict(cohort="GSE40279 (independent test)", comparator=comp, mean_relative_pct=e[e.seed == -1].relative_pct.iloc[0], min_pct=e[e.seed != -1].relative_pct.min(), max_pct=e[e.seed != -1].relative_pct.max()))
Ddf = pd.DataFrame(rowsD)
b.source("D", Ddf, {"cohort": "cohort/split", "comparator": "comparator arm", "mean_relative_pct": "100*mean over seeds of (comparator-candidate)/candidate MSE@0.50", "min_pct": "min over seeds", "max_pct": "max over seeds"},
         "TCGA: regulatory_confirmation_v1 across_seeds_summary mse@0.5; GSE40279: test contrasts @0.5")
b.check("TCGA CpGPT %", 12.8, Ddf.iloc[0].mean_relative_pct, 0.1); b.check("TCGA DeepCpG %", 32.7, Ddf.iloc[2].mean_relative_pct, 0.1)

fig = plt.figure(figsize=(7.4, 6.0))
gs = fig.add_gridspec(2, 2, wspace=0.38, hspace=0.5)
ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off"); panel_letter(ax, "A", dx=-0.05)
ax.text(5, 5.6, f"GSE40279 (whole blood, 450K): {ntot} individuals", ha="center", fontsize=7.5, fontweight="bold")
x0 = 0.3
for lab, n, col, w in [("train", cnt["train"], "#DDDDDD", 5.6), ("val", cnt["validation"], "#FFF3C4", 1.8), ("test", cnt["test"], OI["verm"] + "55", 1.8)]:
    ax.add_patch(FancyBboxPatch((x0, 3.2), w - 0.1, 1.2, boxstyle="round,pad=0.03", fc=col, ec="black", lw=0.8)); ax.text(x0 + w / 2 - 0.05, 3.8, f"{lab}\nn={n}", ha="center", va="center", fontsize=6.8); x0 += w
ax.text(0.3, 2.5, "train: fit frozen models (120 epochs, 3 seeds);\nvalidation: checkpoint selection (best val MSE@0.50);", fontsize=6.3, va="top")
ax.text(0.3, 1.3, "test: accessed ONCE after authorization\n(tag external-recon-test-authorization-v1);\nno retraining, no re-selection.", fontsize=6.3, va="top", color=OI["verm"])
ax = fig.add_subplot(gs[0, 1]); panel_letter(ax, "B")
lines_with_seeds(ax, d, MAIN_ARMS, "mse"); ax.set_xticks(FR); ax.set_ylim(0, d.mse.max() * 1.12)
ax.set_xlabel("masking fraction"); ax.set_ylabel("TEST MSE"); ax.legend(fontsize=6, loc="lower right"); ax.set_title("independent test (mean line, dots = seeds)", fontsize=7.5)
ax = fig.add_subplot(gs[1, 0]); panel_letter(ax, "C")
y = 0; yt = []; yl = []
for blk, comp in [("secondary", DC), ("primary", CP)]:
    sub = Cdf[Cdf.block == blk]
    for sd in (17, 42, 97):
        r = sub[sub.seed == sd].iloc[0]
        y = forest(ax, [dict(est=r.relative_pct, lo=r.ci_lo_pct, hi=r.ci_hi_pct, marker=SEED_MARK[sd], color=ARM_COLOR[comp], hollow=bool(r.ci_includes_zero))], y)
        yt.append(y - 1); yl.append(f"seed {sd}")
        if r.ci_includes_zero: ax.text(r.ci_hi_pct + 0.2, y - 1, "CI includes 0", fontsize=5.8, va="center")
    mu = sub[sub.seed == -1].relative_pct.iloc[0]; ax.plot(mu, y, "D", color="black", ms=5)
    ax.text(mu, y + 0.45, f"{ARM_LABEL[comp]} ({blk}): mean {mu:+.2f}%", ha="center", fontsize=6.2); yt.append(y); yl.append("mean"); y += 1.9
ax.axvline(0, color="black", lw=0.6); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6); ax.set_ylim(-0.7, y); ax.set_xlim(-1, 11)
ax.set_xlabel("paired relative ΔMSE@50% (%), TEST"); ax.set_title("hollow marker: 95% CI includes 0", fontsize=7)
ax = fig.add_subplot(gs[1, 1]); panel_letter(ax, "D")
cohorts = list(Ddf.cohort.unique())
for k, comp in enumerate((CP, DC)):
    e = Ddf[Ddf.comparator == comp]
    xs = [cohorts.index(c) + (k - 0.5) * 0.18 for c in e.cohort]
    ax.plot(xs, e.mean_relative_pct, "-", color=ARM_COLOR[comp], lw=1)
    ax.errorbar(xs, e.mean_relative_pct, yerr=[e.mean_relative_pct - e.min_pct, e.max_pct - e.mean_relative_pct], fmt=ARM_MARK[comp], color=ARM_COLOR[comp], ms=5, capsize=2, label=f"vs {ARM_LABEL[comp]}")
ax.set_xticks(range(2)); ax.set_xticklabels(["TCGA\n(validation)", "GSE40279\n(independent test)"], fontsize=6.5); ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0, None); ax.set_ylabel("relative improvement (%)\n(comparator − candidate)/candidate"); ax.legend(fontsize=6, loc="upper right")
ax.set_title("relative effects only; no absolute MSE across cohorts", fontsize=7)
b.save(fig, "fig3")
b.caption(f"""**Figure 3. Independent confirmation in GSE40279 (one-shot test split).**
(A) Protocol: {ntot} individuals split {cnt['train']}/{cnt['validation']}/{cnt['test']} (train/validation/test; sex x age-tercile stratified). Frozen models (3 seeds each) were selected on validation MSE@0.50 and the test split was accessed once, after authorization (tag external-recon-test-authorization-v1). (B) TEST MSE vs masking fraction for the candidate (Histone+DNase), CpGPT-large and DeepCpG-DNA; lines = seed mean, dots = seeds; y-axis from zero. (C) Paired relative MSE difference (comparator - candidate)/candidate at 50% masking, per seed, with 95% CI from a paired individual x 1 Mb genomic-block bootstrap (2,000 replicates); CpGPT-large is the registered primary contrast (mean {mp:+.2f}%), DeepCpG-DNA the secondary one (mean {ms:+.2f}%). Hollow markers: the CI includes zero ({nz}/3 seeds for the primary contrast). Diamonds: unweighted mean over seeds. (D) Relative improvement of the candidate over each comparator in TCGA (Phase A validation) and in the GSE40279 test split (mean over seeds, bars = min-max across seeds). Absolute MSE is not compared across cohorts because the cohorts differ in tissue and scale. No significance stars.""")
b.finish(filters=["A_v1.2/test per_run_metrics (main arms); primary_/secondary_contrast_per_seed & across_seeds at mask_fraction=0.5, metric=mse", "TCGA regulatory_confirmation_v1/analysis/across_seeds_summary.csv mse@0.5"],
         caption_stats=["CI: paired 95% bootstrap, individuals x 1 Mb blocks, 2000 replicates", "relative delta=(comparator-candidate)/candidate", "bars in 3D = min/max across seeds"])
