"""Figure 4: biological characterisation (frozen bioval-v2 primary endpoints)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa: F401,F403,E402

b = Build("fig4", __file__, tags=["bioval-v2-protocol-freeze-v1"], title="Biological characterisation (frozen primary endpoints)")
P = ROOT / "outputs/biological_validation_v2/bioval-v2-protocol-freeze-v1/report/summary_primary.csv"
sp = b.csv(P)
ARMS = ["regulatory_histone_dnase_v1", "cpgpt_large_locus", "deepcpg_dna_locus"]
PAN = [("A", "microc_H1_intra10kb", "H1 Micro-C contact", "AUROC", 0.5, (0.5685, 0.531, 0.503)),
       ("B", "fantom5_membership", "FANTOM5 enhancer\nmembership", "AUROC", 0.5, (0.833, 0.701, 0.623)),
       ("C", "rt_consensus", "Replication timing\n(out-of-fold)", "out-of-fold Spearman ρ", 0.0, (0.703, 0.486, 0.359)),
       ("D", "loyfer_profile", "Loyfer long-range\nmethylation program", "weighted Spearman ρ (>1 Mb + interchrom.)", 0.0, (-0.063, 0.107, 0.116))]
tol = {"microc_H1_intra10kb": 0.0006, "fantom5_membership": 0.0006, "rt_consensus": 0.0006, "loyfer_profile": 0.0006}
cols = {"endpoint": "frozen primary endpoint id", "arm": "arm id", "value": "point estimate on the full frozen pair/locus set", "ci_lo": "frozen 95% bootstrap CI lower", "ci_hi": "upper",
        "null_reference": "chance/null value shown as dotted line (AUROC 0.5; Spearman 0)"}
fig, axs = plt.subplots(1, 4, figsize=(8.2, 3.3), gridspec_kw=dict(wspace=0.75))
for ax, (L, ep, title, ylab, null, exp) in zip(axs, PAN):
    d = sp[(sp.endpoint == ep) & sp.arm.isin(ARMS)].set_index("arm").loc[ARMS].reset_index()
    d["null_reference"] = null
    b.source(f"{L}", d[["endpoint", "arm", "value", "ci_lo", "ci_hi", "null_reference"]], cols, f"summary_primary.csv endpoint={ep}, 3 main arms (legacy Functional PCA row omitted: Supplement S12)")
    for i, (a, e) in enumerate(zip(ARMS, exp)):
        r = d[d.arm == a].iloc[0]
        ax.errorbar(i, r.value, yerr=[[r.value - r.ci_lo], [r.ci_hi - r.value]], fmt=ARM_MARK[a], color=ARM_COLOR[a], ms=6, capsize=3, lw=1.2)
        ax.text(i, r.ci_hi + 0.01 * (1 if ep != "loyfer_profile" else 1), f"{r.value:.3f}", ha="center", fontsize=6.3, va="bottom")
        b.check(f"{ep} {a}", e, r.value, tol[ep] + (0.0006 if ep == "loyfer_profile" else 0), "brief (rounded)")
    ax.axhline(null, color="black", ls=":", lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Hist+\nDNase", "CpGPT-\nlarge", "DeepCpG-\nDNA"], fontsize=6)
    lo = min(0, d.ci_lo.min() - 0.05) if ep == "loyfer_profile" else 0
    if ep in ("microc_H1_intra10kb", "fantom5_membership"):
        lo = 0.4
    ax.set_ylim(lo, d.ci_hi.max() + 0.18 * (d.ci_hi.max() - lo))
    ax.set_title(title, fontsize=6.8); ax.set_ylabel(ylab, fontsize=6.3); panel_letter(ax, L, dx=-0.3)
    if ep in ("microc_H1_intra10kb", "fantom5_membership"):
        ax.text(0.02, 0.02, "axis starts at 0.4", transform=ax.transAxes, fontsize=5, style="italic")
b.save(fig, "fig4")
vals = {ep: sp[(sp.endpoint == ep)].set_index("arm").value for _, ep, *_ in PAN}
b.caption(f"""**Figure 4. Biological characterisation of the locus representations (frozen primary endpoints of the pre-registered biological validation v2; protocol tag bioval-v2-protocol-freeze-v1).**
Each panel has its own axis and metric; no normalisation across metrics. Points: statistic on the full frozen set; bars: frozen 95% bootstrap CI as stored in summary_primary.csv; dotted line: chance/null (AUROC 0.5; Spearman 0). (A) H1 Micro-C intra-chromosomal contact vs matched non-contact AUROC using cosine similarity of the embeddings. (B) FANTOM5 enhancer membership, blocked cross-validated linear-probe AUROC. (C) Replication timing (consensus), out-of-fold Spearman correlation of a ridge probe. (D) Long-range (>1 Mb and inter-chromosomal) weighted Spearman correlation between embedding cosine similarity and Loyfer WGBS methylation-profile similarity: the candidate is negative ({vals['loyfer_profile']['regulatory_histone_dnase_v1']:+.3f}) and lower than CpGPT-large ({vals['loyfer_profile']['cpgpt_large_locus']:+.3f}) and DeepCpG-DNA ({vals['loyfer_profile']['deepcpg_dna_locus']:+.3f}); this primary endpoint is unfavourable to the candidate and is shown with the same prominence as the others. The legacy Functional PCA control is shown in Supplement S12. In panels A and B the y-axis starts at 0.4 (stated in-panel; chance = 0.5); C and D include zero. No significance stars.""")
b.finish(filters=["summary_primary.csv: endpoints microc_H1_intra10kb, fantom5_membership, rt_consensus, loyfer_profile; arms regulatory_histone_dnase_v1, cpgpt_large_locus, deepcpg_dna_locus"],
         caption_stats=["CI = frozen 95% bootstrap (paired patient/locus/pair-level bootstrap as registered; 1000 replicates per run_manifest n_bootstrap)", "no multiplicity annotations in figure; Holm-adjusted contrasts in S12"])
