"""Registered constants + item registry of the follow-up (fixed BEFORE any real computation)."""
from __future__ import annotations

PROTOCOL_TAG = "bioval-v2-protocol-freeze-v1"
BASE_OUT = f"outputs/biological_validation_v2/{PROTOCOL_TAG}"
SENS_OUT = f"{BASE_OUT}/sensitivity"
EXPL_OUT = f"{BASE_OUT}/exploratory_posthoc_loyfer"
FOLLOWUP_REGISTRATION_DOC = "docs/BIOLOGICAL_VALIDATION_V2_SENSITIVITY_AND_EXPLORATORY_REGISTRATION.md"
LAUNCH_REGISTRATION_DOC = "docs/BIOLOGICAL_VALIDATION_V2_LAUNCH_REGISTRATION.md"
FOLLOWUP_CODE_FILES = (
    "src/cpg_repr_benchmark/bioval_v2_followup/__init__.py",
    "src/cpg_repr_benchmark/bioval_v2_followup/registry.py",
    "src/cpg_repr_benchmark/bioval_v2_followup/gate.py",
    "src/cpg_repr_benchmark/bioval_v2_followup/targets.py",
    "src/cpg_repr_benchmark/bioval_v2_followup/strata.py",
    "src/cpg_repr_benchmark/bioval_v2_followup/geometry.py",
    "src/cpg_repr_benchmark/bioval_v2_followup/profile_ridge.py",
    "src/cpg_repr_benchmark/bioval_v2_followup/compare.py",
    "src/cpg_repr_benchmark/bioval_v2_followup/interpret.py",
    "src/cpg_repr_benchmark/bioval_v2_followup/drivers.py",
    "scripts/bioval_v2/run_followup.py",
    "tests/test_bioval_v2_followup.py",
)

# ---- Part B1 strata (fixed from FROZEN data only: quantiles of the 386,268 frozen pairs, 6 significant digits) -------
# V = min over the pair of the per-CpG variance (ddof=0) of the WGBS beta profile over the groups observed under mask_primary.
V_EDGES = (0.000369276, 0.00133439, 0.00841199)   # quartile edges of V (Q1|Q2|Q3|Q4)
V_TOP10 = 0.0266948                               # 90th percentile of V (overlapping extra cell "V_top10pct")
# M = mean over the pair of the per-CpG mean beta (mask_primary groups)
M_EDGES = (0.2654, 0.466704, 0.698094)            # quartile edges of M
EDGE_RTOL = 1e-4                                  # run-time assertion: recomputed quantiles agree with the registered edges
VARIABLE_CPG_VAR = V_EDGES[2]                     # B3 "variable CpG" = per-CpG variance >= this value (= V Q4 lower edge)

MIN_SHARED = 20
POOLED_STRATA = (">1Mb", "interchromosomal")
ALL_STRATA = ("<1kb", "1-10kb", "10-100kb", "100kb-1Mb", ">1Mb", "interchromosomal")
LOYFER_PRIMARY_STAT = "spearman_pooled_gt1Mb_inter_eqw"

# ---- Part A items -----------------------------------------------------------------------------------------------------
DONE, RUNNABLE, NEW_ARTIFACT, AMBIGUOUS = "DONE", "RUNNABLE_FROM_FROZEN", "REQUIRES_NEW_ARTIFACT", "PROTOCOL_AMBIGUOUS"
# id -> (status, primary reference (endpoint, stat) in the frozen results, short note)
ITEMS = {
    "loyfer_mask_lenient": (RUNNABLE, ("loyfer_profile", LOYFER_PRIMARY_STAT), "profile Pearson recomputed in memory with mask_lenient"),
    "loyfer_drop_single_sample_groups": (RUNNABLE, ("loyfer_profile", LOYFER_PRIMARY_STAT), "4 single-sample groups removed from profiles"),
    "compartment_E1_probe_H1_50kb": (RUNNABLE, ("compartment_E1_probe_H1_100kb", "spearman_E1_vs_oof_pred"), "frozen 50 kb E1 table"),
    "compartment_E1_probe_GM12878_50kb": (RUNNABLE, ("compartment_E1_probe_GM12878_100kb", "spearman_E1_vs_oof_pred"), "frozen 50 kb E1 table"),
    "fantom5_membership_N1": (RUNNABLE, ("fantom5_membership", "auroc_oof_bg_enh5k"), "positives in enhancers expressed in >=1 libraries + frozen matched controls"),
    "fantom5_membership_N3": (RUNNABLE, ("fantom5_membership", "auroc_oof_bg_enh5k"), "idem >=3"),
    "fantom5_membership_N10": (RUNNABLE, ("fantom5_membership", "auroc_oof_bg_enh5k"), "idem >=10"),
    "fantom5_membership_N20": (RUNNABLE, ("fantom5_membership", "auroc_oof_bg_enh5k"), "idem >=20"),
    "fantom5_activity_sample_level": (RUNNABLE, ("fantom5_activity_similarity", "spearman_all_pairs"), "Pearson over 1,829 libraries instead of 638 collapsed groups"),
    "fantom5_activity_cat_cell_lines": (RUNNABLE, ("fantom5_activity_similarity", "spearman_all_pairs"), "collapsed groups of category 'cell lines'"),
    "fantom5_activity_cat_primary_cells": (RUNNABLE, ("fantom5_activity_similarity", "spearman_all_pairs"), "collapsed groups of category 'primary cells'"),
    "fantom5_activity_cat_tissues": (RUNNABLE, ("fantom5_activity_similarity", "spearman_all_pairs"), "collapsed groups of category 'tissues'"),
}
NOT_RUNNABLE = {
    "fourdn_K_per_bin_pair": (NEW_ARTIFACT, "needs freeze_4dn_pairs with K != 20 (K values not specified by the protocol)"),
    "fantom5_chromatin_covariate_matching": (NEW_ARTIFACT, "needs a new matching (freeze_fantom5_matching) with chromatin covariates (not specified)"),
    "fourdn_distance_only_covariate_baseline": (AMBIGUOUS, "protocol: 'confronto con baseline di sola distanza/covariate (dichiarato)' - score/model/covariates undefined"),
    "rt_gc_cpgdensity_tss_residualisation": (AMBIGUOUS, "covariate table is frozen, but residualisation model/which variable is residualised/fit scope are not defined"),
    "rt_normal_vs_tumour_lines": (AMBIGUOUS, "normal/tumour assignment of the 15 UW lines is not frozen (hESC, LCL, K562, SK-N-SH ...)"),
}
ALREADY_DONE = (
    "loyfer: profile Spearman vs Pearson; per distance stratum; >=30 shared groups; marker kNN paper-lifted and U25-only",
    "microc: HFFc6 intra; per distance bin (H1); inter H1 (exploratory)",
    "fantom5: win500; tss5k pool; activity per intra/inter",
    "rt: per-line (15) Spearman/R2",
    "pmd probe (exploratory); E1 probes 100 kb H1/GM12878; delta-cosine",
)
FOLLOWUP_ARMS_REFERENCE = "regulatory_histone_dnase_v1"
