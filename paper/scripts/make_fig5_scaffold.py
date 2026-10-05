"""Figure 5 SCAFFOLD (Loyfer mechanism). No data, no reading of any audit output."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa: F401,F403,E402

b = Build("fig5_scaffold", __file__, title="Loyfer mechanism (scaffold, pending audit)")
PANELS = [("A", "Frozen native cosine\n(per-arm Loyfer primary)"), ("B", "Raw Histone+DNase similarity"),
          ("C", "SVD / native geometry"), ("D", "PC-removal / centering diagnostic"), ("E", "OOF predicted methylation-profile similarity")]
fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.8), gridspec_kw=dict(wspace=0.3, hspace=0.35))
for ax, (L, t) in zip(axs.flat, PANELS + [("", "")]):
    if not L:
        ax.axis("off"); ax.text(0.5, 0.5, "SCAFFOLD ONLY\nno data\n\nWill be populated from the\nfinal Loyfer failure audit\n(not yet available)", ha="center", va="center", fontsize=8, color=OI["grey"]); continue
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=7.5); panel_letter(ax, L, dx=-0.05)
    for sp in ax.spines.values(): sp.set_linestyle("--"); sp.set_color(OI["grey"])
    ax.text(0.5, 0.5, "EXPLORATORY —\npending Loyfer audit", ha="center", va="center", fontsize=9, fontweight="bold", color=OI["verm"], rotation=20, transform=ax.transAxes)
b.save(fig, "fig5_scaffold", dpi=200)
b.caption("**Figure 5 (scaffold). Mechanism of the Loyfer long-range methylation-program result.** EXPLORATORY - pending Loyfer audit. Panels (to be populated from the final audit): (A) frozen native cosine; (B) raw Histone+DNase similarity; (C) SVD/native geometry; (D) PC-removal/centering diagnostic; (E) out-of-fold predicted methylation-profile similarity. No numbers are shown.")
b.note("Scaffold: no result file was read; the Loyfer failure-audit directory was not accessed.")
b.finish(placeholder=True, missing_results=["Loyfer failure audit outputs (exploratory post-hoc audit in progress/paused)"],
         caption_stats=["none (scaffold)"])
