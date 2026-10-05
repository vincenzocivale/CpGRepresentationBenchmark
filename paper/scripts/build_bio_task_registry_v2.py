#!/usr/bin/env python
"""Registry v2 (DATA ONLY): apply the decisions of 2026-10-05 to the committed v1 registry.

Reads the committed v1 files in ``paper/registry/`` (never modified) and a few finished exp5 result files
(read-only), writes ``paper/registry/*_v2.*``. Nothing is trained, evaluated, probed or bootstrapped. The only
computation on result files is the DERIVED matched-pair concordance (a pure function of the stored out-of-fold
predictions); the stored pooled AUROC is re-derived from the same file as a consistency check.

Never reads outputs/**/loyfer_failure_audit, TCGA betas or external betas.

Run (low CPU):  nice -n 19 env OMP_NUM_THREADS=2 python paper/scripts/build_bio_task_registry_v2.py [--validate]
Deterministic: same inputs -> byte-identical outputs (no timestamps in CSVs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "paper" / "registry"
EXP5 = ROOT / "outputs" / "encode_atlas_v1" / "followup" / "exp5"
FORBIDDEN = "loyfer_failure_audit"
DECISIONS_DATE = "2026-10-05"

SUPPORT_MAIN, SUPPORT_EXPL, SUPPORT_INSUFF = 200, 100, 0   # rule fixed before any candidate result (decision 4)
TIER_MAIN, TIER_EXPL, TIER_INSUFF = "main-quality", "exploratory", "insufficient"

LEGACY_STORE_ID = "functional_pca_biovalidation_legacy_store"
LEGACY_STORE_FLAG = "HISTORICAL_NOT_FOR_PAPER"
LEGACY_STORE_MARK = "legacy_pca_native_genomewide_256"
FUNC_ID = "functional_annotations_pca"
FUNC_PAPER_USE = "SUPPLEMENTARY_LEGACY_FUNCTIONAL (campaign full SVD-256, outputs/encode_atlas_v1/embeddings/full_a18b869b.h5)"

NEW_REG_COLUMNS = [
    "paper_use", "proxy_flag", "support_tier", "support_tier_basis", "metric_role", "circularity_use_note",
    "registry_v2_note",
]
# metric hierarchy (decision 5)
METRIC_ROLE = {
    "AUROC": "primary_for_membership_probes",
    "matched_pair_concordance": "co_primary_matched_designs (DERIVED)",
    "AUPRC": "SENSITIVITY_ONLY (prevalence 0.5 under 1:1 matching is artificial; never primary)",
}
CIRC_NOTE_CONTEXT = ("characterization / source-retention only (decision 6): CpG context class is a functional-store input "
                     "and sequence-composition defined; NOT evidence of biological generalization")
CIRC_NOTE_TSS = ("characterization only (decision 6): TSS-proximity is annotation-coupled; not independent validation")
CIRC_NOTE_CCRE = ("cCRE classes derive from DNase/H3K27ac/H3K4me3/CTCF: too close to the candidate's own mark families "
                  "to be independent validation (decision 6); NOT built")
PLANNED_NOT_BUILT = [
    dict(task="promoter_class", use="representation characterization / source-retention", independent_validation=False),
    dict(task="gene_region_class", use="representation characterization / source-retention", independent_validation=False),
    dict(task="cpg_island_shore_shelf", use="representation characterization / source-retention", independent_validation=False,
         existing_rows="legacy_bio_validation/context/*, umap_neighborhood/{island,shore,shelf,open_sea}/*"),
    dict(task="ccre_class", use="characterization only; too close to DNase/H3K27ac/H3K4me3", independent_validation=False),
    dict(task="tss_distance", use="representation characterization / source-retention", independent_validation=False,
         existing_rows="v2_frozen/fantom5_membership__tss5k_pool (FANTOM5 sensitivity, unchanged)"),
]


def guard(p) -> None:
    if FORBIDDEN in str(p):
        raise RuntimeError(f"refusing to touch {p}")


def sha256(path: Path) -> str:
    guard(path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def rd(name: str) -> pd.DataFrame:
    return pd.read_csv(REG / name, dtype=str, keep_default_na=False)


def tier_of(n: float) -> str:
    if n >= SUPPORT_MAIN:
        return TIER_MAIN
    if n >= SUPPORT_EXPL:
        return TIER_EXPL
    return TIER_INSUFF


# ---------------------------------------------------------------------------------- exp5 derived quantities
def exp5_support() -> pd.DataFrame:
    """matched positives per set x variant (= pairs; positives == negatives under 1:1) from evaluable_sets.csv."""
    ev = pd.read_csv(EXP5 / "evaluable_sets.csv")
    out = ev.pivot(index="set", columns="variant", values="pairs_mean").rename(columns={"base": "matched_pos_base",
                                                                                         "probe": "matched_pos_probe"})
    out["tier_base"] = out.matched_pos_base.map(tier_of)
    out["tier_probe"] = out.matched_pos_probe.map(tier_of)
    # task tier: base variant is the reference; the probe variant must not change tiers (checked, reported)
    out["support_tier"] = out.tier_base
    out["tier_consistent_across_variants"] = out.tier_base == out.tier_probe
    return out.reset_index()


def concordance_table() -> pd.DataFrame:
    """DERIVED. matched-pair concordance per (variant, set, representation) from oof_predictions.parquet.

    For each matching seed and pair: 1 if score(positive) > score(matched negative), 0.5 if equal, else 0;
    averaged over pairs, then averaged over the matching seeds (same convention as the stored mean-over-seeds AUROC).
    Also re-derives the pooled AUROC from the same file (consistency check against summary_auc_delta.csv).
    """
    from scipy.stats import rankdata

    d = pd.read_parquet(EXP5 / "oof_predictions.parquet", columns=["variant", "set", "match_seed", "pair_id", "y_true",
                                                                    "representation", "oof_score"])
    rows = []
    for (v, s, rep, seed), g in d.groupby(["variant", "set", "representation", "match_seed"], observed=True, sort=True):
        y = g.y_true.to_numpy()
        pid = g.pair_id.to_numpy()
        sc = g.oof_score.to_numpy().astype(np.float64)
        pos = sc[y][np.argsort(pid[y], kind="stable")]
        neg = sc[~y][np.argsort(pid[~y], kind="stable")]
        n = len(pos)
        conc = float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / n)
        r = rankdata(np.concatenate([pos, neg]))
        auc = float((r[:n].sum() - n * (n + 1) / 2) / (n * n))
        rows.append((str(v), str(s), str(rep), int(seed), n, conc, auc, float((pos == neg).mean())))
    per = pd.DataFrame(rows, columns=["variant", "set", "representation", "match_seed", "n_pairs", "concordance",
                                      "auroc_rederived", "tie_fraction"])
    agg = per.groupby(["variant", "set", "representation"], sort=True).agg(
        n_seeds=("match_seed", "nunique"), n_pairs_mean=("n_pairs", "mean"),
        matched_pair_concordance=("concordance", "mean"), concordance_between_seed_sd=("concordance", "std"),
        auroc_rederived=("auroc_rederived", "mean"), tie_fraction_mean=("tie_fraction", "mean")).reset_index()
    sm = pd.read_csv(EXP5 / "summary_auc_delta.csv")
    sm = sm[sm.kind == "auc"][["variant", "set", "representation", "estimate"]].rename(columns={"estimate": "auroc_stored"})
    agg = agg.merge(sm, on=["variant", "set", "representation"], how="left")
    agg["abs_diff_auroc_rederived_vs_stored"] = (agg.auroc_rederived - agg.auroc_stored).abs()
    agg["flag"] = "DERIVED_from_oof_predictions.parquet (no training, no bootstrap); point estimate, CI = needs_run"
    return agg


# ---------------------------------------------------------------------------------- registry transforms
def classify(reg: pd.DataFrame) -> pd.DataFrame:
    rid, var, pf = reg.representation_id, reg.representation_variant, reg.proxy_for
    legacy_store = var.str.contains(LEGACY_STORE_MARK, regex=False)
    reg["representation_id_v1"] = rid
    reg.loc[legacy_store, "representation_id"] = LEGACY_STORE_ID
    paper_use = pd.Series("", index=reg.index)
    proxy = pd.Series("", index=reg.index)
    paper_use[reg.is_main_panel == "True"] = "MAIN_PANEL"
    func = (rid == FUNC_ID) & ~legacy_store
    paper_use[func] = FUNC_PAPER_USE
    paper_use[legacy_store] = LEGACY_STORE_FLAG
    proxy[legacy_store] = LEGACY_STORE_FLAG
    hist = rid == "other:histone_plus_context_campaign"
    proxy[hist] = "proxy_not_candidate"
    paper_use[hist] = "SUPPLEMENTARY_ONLY_proxy_not_candidate"
    rawf = rid == "other:raw_feature_subset"
    proxy[rawf & (pf == "regulatory_histone_dnase_v1")] = "proxy_not_candidate"
    proxy[rawf & (pf != "regulatory_histone_dnase_v1")] = "raw_features_not_embedding"
    paper_use[rawf] = "SUPPLEMENTARY_ONLY_" + proxy[rawf]
    compact = (pf.isin(["cpgpt_large_locus", "deepcpg_dna_locus"])) & (rid.isin(["cpgpt_large_locus", "deepcpg_dna_locus"]))
    proxy[compact] = "proxy_variant_not_main_panel"
    paper_use[compact] = "SUPPLEMENTARY_ONLY_proxy_variant_not_main_panel"
    ctx = rid == "other:context_only"
    proxy[ctx] = "control_arm_not_main_panel"
    paper_use[ctx] = "SUPPLEMENTARY_ONLY_control_arm"
    nonpanel = rid.isin(["other:cpgpt_locus_small", "other:deepcpg_dna_locus_hepg2", "other:methylgpt_locus",
                         "other:methylgpt_locus_medium"])
    proxy[nonpanel] = "non_panel_variant"
    paper_use[nonpanel] = "SUPPLEMENTARY_ONLY_non_panel_variant"
    reg["paper_use"], reg["proxy_flag"] = paper_use, proxy
    return reg


def set_name(task_id: str) -> str:
    t = task_id.split("/")
    if t[0] in ("encode_exp5_base", "encode_exp5_probe", "encode_ewas_matched_single_draw"):
        return t[1]
    if t[0] == "legacy_bio_validation" and len(t) > 2 and t[1] == "known_set":
        return t[2]
    if t[0] == "umap_neighborhood" and len(t) > 1 and (t[1].startswith("ewas_") or t[1].endswith("_clock")):
        return t[1]
    return ""


def circ_note(task_id: str) -> str:
    low = task_id.lower()
    if "/context/" in low or low.startswith("umap_neighborhood/") and any(
            k in low for k in ("/island/", "/shore/", "/shelf/", "/open_sea/")):
        return CIRC_NOTE_CONTEXT
    if "tss" in low or "promoter" in low or "gene_region" in low:
        return CIRC_NOTE_TSS
    if "ccre" in low:
        return CIRC_NOTE_CCRE
    return ""


def build() -> dict:
    sup = exp5_support()
    tier_by_set = dict(zip(sup["set"], sup.support_tier))
    pos_by_set = dict(zip(sup["set"], sup.matched_pos_base))
    conc = concordance_table()
    out: dict[str, pd.DataFrame] = {}

    # --- registry
    reg = classify(rd("bio_task_registry.csv"))
    sn = reg.task_id.map(set_name)
    reg["support_tier"] = sn.map(tier_by_set).fillna("")
    reg["support_tier_basis"] = np.where(sn != "", "exp5 base-variant matched positives (pairs): " + sn.map(
        lambda s: f"{pos_by_set[s]:.0f}" if s in pos_by_set else ""), "")
    reg.loc[sn == "", "support_tier_basis"] = ""
    reg["metric_role"] = reg.metric.map(lambda m: METRIC_ROLE.get(m, ""))
    reg.loc[~reg.stat_class.isin(["ewas", "clock"]) & ~reg.suite.isin(["encode_ewas_matched_single_draw", "legacy_bio_validation"]),
            "metric_role"] = ""
    reg["circularity_use_note"] = reg.task_id.map(circ_note)
    reg["registry_v2_note"] = ""
    reg.loc[reg.proxy_flag == LEGACY_STORE_FLAG, "registry_v2_note"] = (
        "old PCA store of the first bio-validation; HISTORICAL, must not appear in the manuscript as functional_annotations_pca")
    reg.loc[reg.suite == "encode_exp5_base", "registry_v2_note"] = "exp5 base: SUPPLEMENTARY_ONLY until candidate completion"
    reg.loc[reg.suite == "encode_exp5_probe", "registry_v2_note"] = "exp5 probe: SUPPLEMENTARY_ONLY until candidate completion"
    reg.loc[reg.suite == "legacy_bio_validation", "registry_v2_note"] = (
        reg.loc[reg.suite == "legacy_bio_validation", "registry_v2_note"].replace("", "legacy random-fold probe: PROBLEMATIC"))
    reg.loc[reg.suite == "v2_exploratory_posthoc", "registry_v2_note"] = (
        "Loyfer stays prominent: frozen negative primary stands; audit outcome (bug -> declared correction; geometry mismatch "
        "-> negative stays) is NOT recorded here (audit directory not read)")

    # --- DERIVED matched-pair concordance rows (exp5 arms, flagged)
    base_rows = reg[(reg.stat_name == "pooled_oof_auc_mean_over_seeds") & reg.suite.isin(["encode_exp5_base", "encode_exp5_probe"])]
    new = []
    for _, r in base_rows.iterrows():
        v = "base" if r.suite.endswith("base") else "probe"
        m = conc[(conc.variant == v) & (conc.set == r.task_name) & (conc.representation == r.representation_raw)]
        if len(m) != 1:
            continue
        m = m.iloc[0]
        nr = r.copy()
        nr["stat_name"] = "DERIVED_matched_pair_concordance_mean_over_seeds"
        nr["stat_class"] = "ewas" if r.stat_class == "ewas" else r.stat_class
        nr["is_headline"] = "False"
        nr["metric"] = "matched_pair_concordance"
        nr["score"] = f"{m.matched_pair_concordance:.10g}"
        nr["ci_lo"], nr["ci_hi"], nr["ci_definition"] = "", "", ""
        nr["source_file"] = "outputs/encode_atlas_v1/followup/exp5/oof_predictions.parquet"
        nr["source_sha256"] = sha256(EXP5 / "oof_predictions.parquet")
        nr["metric_role"] = METRIC_ROLE["matched_pair_concordance"]
        nr["registry_v2_note"] = ("DERIVED by paper/scripts/build_bio_task_registry_v2.py from stored OOF scores; fraction of "
                                  "matched pairs with score(EWAS CpG) > score(matched background), ties 0.5; CI needs_run")
        new.append(nr)
    reg = pd.concat([reg, pd.DataFrame(new)], ignore_index=True)
    reg["row_id_v1"] = reg.row_id
    reg["row_id"] = [str(i + 1) for i in range(len(reg))]
    out["bio_task_registry_v2.csv"] = reg

    # --- matrix
    mat = rd("bio_task_matrix.csv")
    sn = mat.task_id.map(set_name)
    mat["funcleg_store_id"] = np.where(mat.legacy_functional_variant.str.contains(LEGACY_STORE_MARK, regex=False),
                                       LEGACY_STORE_ID + f" [{LEGACY_STORE_FLAG}]", FUNC_ID + " [campaign full SVD-256; supplementary legacy]")
    mat["support_tier"] = sn.map(tier_by_set).fillna("")
    mat["matched_positives_exp5_base"] = sn.map(lambda s: f"{pos_by_set[s]:.0f}" if s in pos_by_set else "")
    mat["primary_metric_v2"] = np.where(mat.support_tier != "", "AUROC", "")
    mat["metric_hierarchy_v2"] = np.where(mat.support_tier != "",
                                          "primary=AUROC; co-primary=matched_pair_concordance (DERIVED); AUPRC=sensitivity only", "")
    mat["circularity_use_note"] = mat.task_id.map(circ_note)
    for arm_key, rep in [("cand", "regulatory_histone_dnase_v1"), ("cpgpt", "cpgpt_large_native512"),
                         ("deepcpg", "deepcpg_hcc_native128"), ("funcleg", "functional_256")]:
        vals, st = [], []
        for _, r in mat.iterrows():
            if r.suite in ("encode_exp5_base", "encode_exp5_probe"):
                v = "base" if r.suite.endswith("base") else "probe"
                s = r.task_id.split("/")[1]
                m = conc[(conc.variant == v) & (conc.set == s) & (conc.representation == rep)]
                if len(m):
                    vals.append(f"{m.iloc[0].matched_pair_concordance:.10g}")
                    st.append("DERIVED")
                else:
                    vals.append("")
                    st.append("needs_run")
            elif r.support_tier != "":
                vals.append("")
                st.append("needs_run" if r.suite == "encode_ewas_matched_single_draw" else "not_applicable_unmatched_design")
            else:
                vals.append("")
                st.append("")
        mat[f"{arm_key}__matched_pair_concordance"] = vals
        mat[f"{arm_key}__matched_pair_concordance__status"] = st
    out["bio_task_matrix_v2.csv"] = mat

    # --- coverage
    cov = rd("bio_task_coverage_matrix.csv")
    cov = cov.rename(columns={"functional_annotations_pca_legacy": "functional_annotations_pca_legacy_supp"})
    cov["regulatory_histone_dnase_v1__proxy_label"] = np.where(cov["regulatory_histone_dnase_v1__proxy"] != "", "proxy_not_candidate", "")
    cov["cpgpt_large_locus__proxy_label"] = np.where(cov["cpgpt_large_locus__proxy"] != "", "proxy_variant_not_main_panel", "")
    cov["deepcpg_dna_locus__proxy_label"] = np.where(cov["deepcpg_dna_locus__proxy"] != "", "proxy_variant_not_main_panel", "")
    cov["legacy_store_note"] = np.where(cov.task_id.str.startswith("legacy_bio_validation"),
                                        f"legacy-suite functional arm = {LEGACY_STORE_ID} [{LEGACY_STORE_FLAG}]", "")
    out["bio_task_coverage_matrix_v2.csv"] = cov

    # --- contrasts (content unchanged; paper_use label added)
    con = rd("bio_task_paired_contrasts.csv")
    con["paper_use_v2"] = np.where(con.comparator_id.str.startswith(FUNC_ID) | con.reference_id.str.startswith(FUNC_ID),
                                   "functional = campaign full SVD-256 (supplementary legacy)", "")
    out["bio_task_paired_contrasts_v2.csv"] = con

    # --- per-task audit
    aud = rd("per_task_audit.csv")
    sn = aud.task_id.map(set_name)
    aud["support_tier"] = sn.map(tier_by_set).fillna("")
    aud["matched_positives_exp5_base"] = sn.map(lambda s: f"{pos_by_set[s]:.0f}" if s in pos_by_set else "")
    # not-evaluable sets of exp5 are rows too (task_id may end with the set name)
    aud["metric_hierarchy_v2"] = np.where(aud.support_tier != "",
                                          "primary=AUROC; co-primary=matched_pair_concordance; AUPRC=sensitivity only", "")
    aud["circularity_use_note"] = aud.task_id.map(circ_note)
    aud["rigor_grade_v2"] = aud.rigor_grade   # unchanged: legacy PROBLEMATIC; exp5 SUPPLEMENTARY_ONLY until completion
    rigorous = aud.suite.isin(["encode_exp5_base", "encode_exp5_probe"])
    elig = np.where(aud.rigor_grade == "MAIN_QUALITY", "TRUE (already main-quality)",
                    np.where(rigorous & (aud.support_tier == TIER_MAIN), "TRUE_AFTER_CANDIDATE_COMPLETION",
                             np.where(rigorous & (aud.support_tier == TIER_EXPL), "FALSE_exploratory_support",
                                      np.where(rigorous & (aud.support_tier == TIER_INSUFF), "FALSE_insufficient_support", "FALSE"))))
    aud["eligible_for_main_after_completion"] = elig
    aud["eligibility_basis"] = np.where(
        aud.rigor_grade == "MAIN_QUALITY", "frozen/preregistered bioval-v2",
        np.where(rigorous, "matched 1:1, chromosome-blocked, fixed-C probe, CI; needs candidate arm; matching strata are partly functional-store "
                           "inputs (caveat); MAIN evidence route is EWAS-v2, exp5 completion = discovery",
                 np.where(aud.suite == "legacy_bio_validation", "legacy random-fold probe, unmatched negatives (PROBLEMATIC)",
                          np.where(aud.suite == "encode_ewas_matched_single_draw", "single draw, proxies only; superseded by exp5", "not applicable / not eligible"))))
    aud["current_status_note"] = np.where(aud.rigor_grade == "MAIN_QUALITY", "current status: MAIN_QUALITY",
                                          "current status unchanged: " + aud.rigor_grade)
    out["per_task_audit_v2.csv"] = aud

    # --- exp5 support + concordance + tiers
    sup_out = sup.copy()
    for c in ("matched_pos_base", "matched_pos_probe"):
        sup_out[c] = sup_out[c].round(1)
    out["exp5_support_tiers_v2.csv"] = sup_out
    out["exp5_matched_pair_concordance_v2.csv"] = conc.round(10)

    # --- tier listing for ALL EWAS/phenotype tasks (task-level)
    tl = aud[aud.support_tier != ""][["task_id", "suite", "task_name", "support_tier", "matched_positives_exp5_base",
                                      "rigor_grade", "eligible_for_main_after_completion"]]
    out["support_tier_task_list_v2.csv"] = tl.sort_values(["support_tier", "suite", "task_id"]).reset_index(drop=True)
    return out


def dictionary(files: list[str], inputs: dict[str, str]) -> dict:
    v1 = json.load(open(REG / "bio_task_registry.json"))
    d = dict(v1)
    d["registry_version"] = "v2 (decisions of 2026-10-05); v1 files are untouched and recoverable via git"
    d["registry_file"] = "bio_task_registry_v2.csv"
    cols = dict(v1["columns"])
    cols.update({
        "representation_id": "v2: legacy-suite old PCA store renamed functional_pca_biovalidation_legacy_store (HISTORICAL_NOT_FOR_PAPER); "
                             "functional_annotations_pca now denotes ONLY the campaign full SVD-256 store (supplementary legacy)",
        "representation_id_v1": "representation_id as in the committed v1 registry",
        "row_id_v1": "row_id in v1 (blank for DERIVED concordance rows appended in v2)",
        "paper_use": "MAIN_PANEL | SUPPLEMENTARY_LEGACY_FUNCTIONAL | HISTORICAL_NOT_FOR_PAPER | SUPPLEMENTARY_ONLY_<proxy label>",
        "proxy_flag": "proxy_not_candidate (histone+context arm; raw histone-only features) | proxy_variant_not_main_panel (compact CpGPT-256, "
                      "DeepCpG pad256) | control_arm_not_main_panel | non_panel_variant | raw_features_not_embedding | HISTORICAL_NOT_FOR_PAPER",
        "support_tier": f"EWAS/phenotype tasks only, from exp5 base-variant matched positives: main-quality >= {SUPPORT_MAIN}; exploratory "
                        f"{SUPPORT_EXPL}-{SUPPORT_MAIN-1}; insufficient < {SUPPORT_EXPL}. Rule fixed on N only, before any candidate result (decision 4)",
        "support_tier_basis": "matched positive count used",
        "metric_role": "primary_for_membership_probes (AUROC) | co_primary_matched_designs (matched_pair_concordance, DERIVED) | SENSITIVITY_ONLY (AUPRC)",
        "circularity_use_note": "decision 6: context/promoter/gene-region/island/cCRE/TSS targets = characterization / source retention only",
        "registry_v2_note": "free text",
        "metric": "v1 vocabulary + matched_pair_concordance",
    })
    d["columns"] = cols
    d["metrics_vocabulary_v2"] = list(v1.get("metrics", [])) + ["matched_pair_concordance"] if isinstance(v1.get("metrics"), list) else None
    d["matched_pair_concordance_definition"] = (
        "For each matched pair (EWAS CpG, matched background CpG) in one matching draw: 1 if OOF score(EWAS CpG) > OOF score(background), "
        "0.5 if equal, 0 otherwise; mean over pairs; then mean over the 20 matching draws. Pooled OOF scores from the SAME chromosome-blocked "
        "folds as the AUROC. 0.5 = chance, 1 = every EWAS CpG ranks above its partner. Differs from AUROC (which ranks across all pairs).")
    d["support_tier_rule"] = dict(main_quality=f">={SUPPORT_MAIN} matched positives", exploratory=f"{SUPPORT_EXPL}-{SUPPORT_MAIN-1}",
                                  insufficient=f"<{SUPPORT_EXPL}", all_may_appear_in_supplementary=True,
                                  fixed_before_candidate_results=True, decisions_date=DECISIONS_DATE)
    d["decisions_of_2026-10-05"] = [
        "1 add regulatory_histone_dnase_v1 to EWAS probes in two levels: exp5_candidate_completion (exact exp5 pairs/folds/bootstrap), then frozen EWAS-v2 for the main paper",
        "2 functional legacy = only campaign full SVD-256 store, supplementary; old PCA store HISTORICAL_NOT_FOR_PAPER",
        "3 histone+context campaign arm is not a main-paper proxy; supplementary only, label proxy_not_candidate",
        "4 EWAS support rule on N only (>=200 main-quality; 100-199 exploratory; <100 insufficient)",
        "5 primary AUROC; matched-pair concordance; AUPRC only as sensitivity",
        "6 promoter/gene region/island/cCRE/TSS: characterization / source-retention only; not built now",
        "7 Loyfer stays prominent (frozen negative primary stands); audit outcome handled separately",
    ]
    d["planned_not_built_characterization_targets"] = PLANNED_NOT_BUILT
    d["files_v2"] = files
    d["v2_inputs_sha256"] = inputs
    return d


def write_all() -> dict:
    out = build()
    for name, df in out.items():
        df.to_csv(REG / name, index=False, lineterminator="\n")
    inputs = {f"paper/registry/{n}": sha256(REG / n) for n in [
        "bio_task_registry.csv", "bio_task_matrix.csv", "per_task_audit.csv", "bio_task_coverage_matrix.csv",
        "bio_task_paired_contrasts.csv", "bio_task_registry.json"]}
    for n in ["oof_predictions.parquet", "evaluable_sets.csv", "summary_auc_delta.csv", "protocol.json", "matching_summary_per_seed.csv"]:
        inputs[f"outputs/encode_atlas_v1/followup/exp5/{n}"] = sha256(EXP5 / n)
    d = dictionary(sorted(out), inputs)
    (REG / "bio_task_registry_v2.json").write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
    outs = {f"paper/registry/{n}": sha256(REG / n) for n in sorted(out) + ["bio_task_registry_v2.json"]}
    (REG / "manifests" / "registry_v2_outputs_sha256.json").write_text(json.dumps(
        dict(inputs=inputs, outputs=outs), indent=1, sort_keys=True) + "\n")
    return out


def validate(out: dict) -> int:
    err = []
    reg, mat, aud = out["bio_task_registry_v2.csv"], out["bio_task_matrix_v2.csv"], out["per_task_audit_v2.csv"]
    v1 = rd("bio_task_registry.csv")
    if len(reg) != len(v1) + (reg.metric == "matched_pair_concordance").sum():
        err.append("row count mismatch vs v1 + derived")
    if reg.row_id.duplicated().any():
        err.append("duplicate row_id")
    # exactly one object called functional_annotations_pca
    f = reg[reg.representation_id == FUNC_ID]
    if f.representation_variant.str.contains(LEGACY_STORE_MARK, regex=False).any():
        err.append("old store still named functional_annotations_pca")
    if not reg[reg.representation_id == LEGACY_STORE_ID].proxy_flag.eq(LEGACY_STORE_FLAG).all():
        err.append("legacy store rows not flagged")
    hist = reg[reg.representation_id == "other:histone_plus_context_campaign"]
    if not hist.proxy_flag.eq("proxy_not_candidate").all():
        err.append("histone+context not flagged proxy_not_candidate")
    if (reg.circularity_use_note == "").all():
        err.append("no circularity notes")
    c = out["exp5_matched_pair_concordance_v2.csv"]
    if c.abs_diff_auroc_rederived_vs_stored.max() > 1e-4:  # OOF scores are stored as float32 (rank ties), so tiny drift is expected
        err.append(f"re-derived AUROC differs from stored (max {c.abs_diff_auroc_rederived_vs_stored.max():.2e})")
    if not ((c.matched_pair_concordance >= 0) & (c.matched_pair_concordance <= 1)).all():
        err.append("concordance out of range")
    if not out["exp5_support_tiers_v2.csv"].tier_consistent_across_variants.all():
        err.append("tier differs between exp5 variants (reported, rule uses base)")
    if len(aud) != len(rd("per_task_audit.csv")) or len(mat) != len(rd("bio_task_matrix.csv")):
        err.append("audit/matrix row count changed")
    for e in err:
        print("ERROR:", e)
    print(f"validate: {len(err)} errors; registry rows {len(reg)}; concordance rows {len(c)}; max |AUROC rederived - stored| = "
          f"{c.abs_diff_auroc_rederived_vs_stored.max():.2e}")
    return 1 if err else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--determinism", action="store_true", help="build twice in memory and compare CSV bytes")
    a = ap.parse_args()
    out = write_all()
    rc = validate(out) if a.validate else 0
    if a.determinism:
        h1 = {n: sha256(REG / n) for n in out}
        write_all()
        h2 = {n: sha256(REG / n) for n in out}
        print("determinism:", "PASS" if h1 == h2 else "FAIL")
        rc = rc or (0 if h1 == h2 else 1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
