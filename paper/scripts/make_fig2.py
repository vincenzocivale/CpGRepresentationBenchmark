"""Figure 2: TCGA reconstruction, Phase A frozen validation (validation patients only)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa: F401,F403,E402

b = Build("fig2", __file__, title="TCGA reconstruction (Phase A, frozen validation)")
A = ROOT / "outputs/regulatory_confirmation_v1/analysis"
pr = b.csv(A / "per_run_metrics.csv")
pc = b.csv(A / "paired_contrasts.csv")
sm = b.csv(A / "across_seeds_summary.csv")
FR = [0.15, 0.3, 0.5, 0.7, 0.9]
CAND, CP, DC = MAIN_ARMS
cmp_arms = [CP, DC]

d = pr[pr.arm.isin(MAIN_ARMS)][["arm", "seed", "mask_fraction", "mse"]].copy()
assert len(d) == 45
b.source("A", d, {"arm": "arm id", "seed": "seed", "mask_fraction": "masking fraction", "mse": "validation MSE (TCGA, Phase A, best.pt by val MSE@0.50)"},
         "per_run_metrics.csv, main arms, 5 fractions x 3 seeds")

# 2B relative delta
rel_rows = []
for a in cmp_arms:
    for f in FR:
        x = pc[(pc.comparator == a) & (pc.mask_fraction == f) & (pc.metric == "mse")]
        for _, r in x.iterrows():
            rel_rows.append(dict(comparator=a, mask_fraction=f, seed=int(r.seed), relative_pct=100 * r.relative, ci_lo_pct=100 * r.rel_ci_lo, ci_hi_pct=100 * r.rel_ci_hi, mean_over_seeds_pct=np.nan))
        s = sm[(sm.comparator == a) & (sm.mask_fraction == f) & (sm.metric == "mse")].iloc[0]
        rel_rows.append(dict(comparator=a, mask_fraction=f, seed=-1, relative_pct=100 * s.mean_relative, ci_lo_pct=np.nan, ci_hi_pct=np.nan, mean_over_seeds_pct=100 * s.mean_relative))
R = pd.DataFrame(rel_rows)
colsR = {"comparator": "comparator arm", "mask_fraction": "masking fraction", "seed": "seed; -1 = mean over 3 seeds of the per-seed relative differences (across_seeds_summary.mean_relative)",
         "relative_pct": "100*(comparator - candidate)/candidate MSE; positive = candidate lower error", "ci_lo_pct": "lower bound paired 95% CI (patient x 1 Mb block bootstrap, 2000 replicates, bootstrap seed = seed)",
         "ci_hi_pct": "upper bound", "mean_over_seeds_pct": "duplicate of relative_pct on the mean row"}
b.source("B", R, colsR, "paired_contrasts.csv metric=mse, comparator in {cpgpt, deepcpg}; across_seeds_summary.csv metric=mse")

# 2C / 2D at 0.50
at = d[d.mask_fraction == 0.5]
C = at.groupby("arm").mse.agg(mean="mean", sd=lambda v: v.std(ddof=1)).reset_index()
C2 = at.merge(C, on="arm")
b.source("C", C2, {"arm": "arm id", "seed": "seed", "mask_fraction": "0.5", "mse": "validation MSE at 50% masking", "mean": "mean over 3 seeds", "sd": "sample SD (ddof=1) over 3 seeds"}, "mask_fraction=0.5")
exp = {CAND: 0.014784, CP: 0.016672, DC: 0.019624}
for a, e in exp.items():
    b.check(f"TCGA MSE@0.50 {a}", e, float(C[C.arm == a]["mean"].iloc[0]), 1e-5)
D = R[(R.mask_fraction == 0.5)].copy()
b.source("D", D, colsR, "mask_fraction=0.5 rows of panel B table")
for a, e in [(CP, 12.8), (DC, 32.7)]:
    b.check(f"TCGA rel dMSE@0.50 mean {a} (%)", e, float(D[(D.comparator == a) & (D.seed == -1)].relative_pct.iloc[0]), 0.1)
# alt definition: ratio of seed-mean MSEs
for a in cmp_arms:
    ratio = 100 * (C[C.arm == a]["mean"].iloc[0] / C[C.arm == CAND]["mean"].iloc[0] - 1)
    b.note(f"rel dMSE@0.50 {a}: mean of per-seed relatives = {D[(D.comparator==a)&(D.seed==-1)].relative_pct.iloc[0]:.3f}% vs ratio of seed-mean MSEs = {ratio:.3f}% (both reported by definition; plotted = mean of per-seed relatives)")

fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8), gridspec_kw=dict(wspace=0.4, hspace=0.42))
ax = axs[0, 0]; panel_letter(ax, "A")
lines_with_seeds(ax, d, MAIN_ARMS, "mse")
ax.set_xlabel("masking fraction"); ax.set_ylabel("validation MSE"); ax.set_xticks(FR)
ax.set_ylim(0, d.mse.max() * 1.1); ax.legend(loc="lower right", fontsize=6)
ax.set_title("MSE vs masking (mean line; dots = seeds)", fontsize=7.5)
ax = axs[0, 1]; panel_letter(ax, "B")
for a in cmp_arms:
    m = R[(R.comparator == a) & (R.seed == -1)]
    ax.plot(m.mask_fraction, m.relative_pct, color=ARM_COLOR[a], marker=ARM_MARK[a], ms=3.5, label=f"{ARM_LABEL[a]} vs candidate")
    s = R[(R.comparator == a) & (R.seed != -1)]
    ax.scatter(s.mask_fraction + (s.seed.map({17: -1, 42: 0, 97: 1}) * 0.008), s.relative_pct, s=7, color=ARM_COLOR[a], alpha=0.55, edgecolors="none")
ax.axhline(0, color="black", lw=0.6); ax.set_xticks(FR); ax.set_ylim(bottom=0)
ax.set_xlabel("masking fraction"); ax.set_ylabel("relative ΔMSE (%) = (comparator − candidate)/candidate")
ax.legend(fontsize=6, loc="center"); ax.set_title("(comparator \u2212 candidate)/candidate", fontsize=7.5)
ax = axs[1, 0]; panel_letter(ax, "C")
for i, a in enumerate(MAIN_ARMS):
    v = at[at.arm == a].sort_values("seed")
    mu, sd = v.mse.mean(), v.mse.std(ddof=1)
    ax.bar(i, mu, color=ARM_COLOR[a], alpha=0.85, edgecolor="black", lw=0.6, width=0.6)
    ax.errorbar(i, mu, yerr=sd, color="black", capsize=3, lw=0.9)
    ax.scatter(i + jitter(3, 0.15), v.mse, s=10, c="black", zorder=3, linewidths=0)
ax.set_xticks(range(3)); ax.set_xticklabels([ARM_LABEL[a].replace(" (candidate)", "\n(candidate)") for a in MAIN_ARMS], fontsize=6)
ax.set_ylim(0, at.mse.max() * 1.15); ax.set_ylabel("validation MSE at 50% masking"); ax.set_title("bar = mean, error bar = SD (n=3), dots = seeds", fontsize=7)
ax = axs[1, 1]; panel_letter(ax, "D")
y = 0; yt, yl = [], []
for a in [DC, CP]:
    for sd in (17, 42, 97):
        r = D[(D.comparator == a) & (D.seed == sd)].iloc[0]
        y = forest(ax, [dict(est=r.relative_pct, lo=r.ci_lo_pct, hi=r.ci_hi_pct, marker=SEED_MARK[sd], color=ARM_COLOR[a])], y)
        yt.append(y - 1); yl.append(f"seed {sd}")
    mu = D[(D.comparator == a) & (D.seed == -1)].relative_pct.iloc[0]
    ax.plot(mu, y, "D", color="black", ms=5); ax.text(mu, y + 0.4, f"{ARM_LABEL[a]}: mean {mu:+.1f}%", ha="center", fontsize=6.3)
    yt.append(y); yl.append("mean"); y += 1.8
ax.axvline(0, color="black", lw=0.6); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6); ax.set_ylim(-0.7, y)
ax.set_xlabel("paired relative ΔMSE@50% (%), (comparator − candidate)/candidate"); ax.set_xlim(0, 42)
b.save(fig, "fig2")
cp_m = D[(D.comparator == CP) & (D.seed == -1)].relative_pct.iloc[0]; dc_m = D[(D.comparator == DC) & (D.seed == -1)].relative_pct.iloc[0]
b.caption(f"""**Figure 2. Reconstruction of masked methylation in TCGA (Phase A, frozen protocol, validation patients only; test patients not used).**
Main comparators: Histone+DNase regulatory representation (candidate), CpGPT-large and DeepCpG-DNA (3 seeds each, 120 epochs, batch 8). The legacy Functional PCA control is excluded from this figure (see Supplement). (A) Validation MSE vs masking fraction; lines = seed mean, dots = individual seeds (17, 42, 97). (B) Relative excess error of each comparator vs the candidate, (comparator - candidate)/candidate, mean over seeds (line) and per-seed values (dots); positive values mean the candidate has lower error. (C) MSE at 50% masking: bars = mean of 3 seeds, error bars = sample SD (ddof=1), dots = seeds; y-axis starts at zero. (D) Paired relative MSE difference at 50% masking per seed with 95% CI from a paired patient x 1 Mb genomic-block bootstrap (2,000 replicates; 918 validation patients; bootstrap seed = training seed); diamonds = unweighted mean of the three per-seed relative differences (CpGPT-large {cp_m:+.1f}%, DeepCpG-DNA {dc_m:+.1f}%). The across-seed summary is a descriptive mean over seeds; seeds are optimisation replicates, not independent biological samples. No significance stars are used.""")
b.finish(filters=["per_run_metrics.csv: arms main x seeds 17/42/97 x fractions 0.15..0.9", "paired_contrasts.csv: metric=mse, reference=regulatory_histone_dnase, comparators cpgpt/deepcpg", "across_seeds_summary.csv: metric=mse"],
         caption_stats=["relative delta=(comparator-candidate)/candidate", "CI: paired patient x 1 Mb block bootstrap, 2000 replicates, per seed", "SD: sample SD of 3 seed values"])
