#!/usr/bin/env python
"""Build the biological-task registry (DATA COLLECTION ONLY).

Reads existing, finished result summary files (CSV / JSON / tiny .npy annotation sets) and writes
``paper/registry/*``. It never trains, evaluates, probes or bootstraps anything, never modifies an existing
file, never reads the exploratory Loyfer failure-audit directory, never reads TCGA betas / predictions .npz.

Deterministic: same inputs -> byte-identical outputs (no timestamps in the CSVs, sorted iteration).

Run:  PYTHONPATH=$PWD/src python paper/scripts/build_bio_task_registry.py
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "registry"
FORBIDDEN = "loyfer_failure_audit"
V2 = "outputs/biological_validation_v2/bioval-v2-protocol-freeze-v1"
V2_TAG = "bioval-v2-protocol-freeze-v1"
ENC = "outputs/encode_atlas_v1"

# ----------------------------------------------------------------------------------------------- vocabularies
FAMILIES = [
    "genomic locus identity", "regulatory activity", "disease/pathology", "aging",
    "environmental exposure", "3D genome", "replication timing", "methylation program", "other",
]
STATUSES = ["frozen", "preregistered", "exploratory", "discovery-campaign", "legacy-unregistered"]
CIRC = ["INDEPENDENT", "PARTIALLY_RELATED", "SOURCE_RETENTION"]
GRADES = ["MAIN_QUALITY", "SUPPLEMENTARY_ONLY", "PROBLEMATIC"]
COVERAGE = ["HAVE", "PROXY_VARIANT_ONLY", "MISSING"]
MAIN_PANEL = ["regulatory_histone_dnase_v1", "cpgpt_large_locus", "deepcpg_dna_locus"]
METRICS = ["AUROC", "AUPRC", "Spearman", "Pearson", "R2", "MAE", "MSE", "accuracy", "F1", "precision", "recall",
           "enrichment", "neighbor_hit_rate", "neighbor_fraction", "random_fraction", "delta_cosine",
           "mean_neighbor_correlation", "mean_matched_correlation", "correlation_gap"]
MATRIX_METRICS = {"AUROC": "auroc", "AUPRC": "auprc", "Spearman": "spearman", "R2": "r2", "enrichment": "enrichment"}
ARMKEY = {"regulatory_histone_dnase_v1": "cand", "cpgpt_large_locus": "cpgpt", "deepcpg_dna_locus": "deepcpg",
          "functional_annotations_pca": "funcleg"}

REG_COLUMNS = [
    "row_id", "suite", "task_id", "task_name", "stat_name", "stat_class", "is_headline", "biological_family",
    "target_source", "target_description", "positive_n", "negative_n", "n_items", "background_sampling",
    "matching_covariates", "split_strategy", "representation_raw", "representation_id", "representation_variant",
    "representation_dim", "is_main_panel", "proxy_for", "classifier_probe", "metric", "score", "ci_lo", "ci_hi",
    "ci_definition", "seed", "source_file", "source_sha256", "aux_source_files", "git_commit_recorded",
    "protocol_tag", "status", "endpoint_class", "circularity_level", "circularity_vs_candidate",
    "circularity_vs_functional_legacy", "circularity_vs_sequence", "notes",
]
CONTRAST_COLUMNS = [
    "contrast_id", "suite", "task_id", "stat_name", "reference_raw", "reference_id", "comparator_raw", "comparator_id",
    "delta", "ci_lo", "ci_hi", "ci_definition", "p_raw", "p_adjusted", "in_holm_family", "sign_convention",
    "source_file", "source_sha256", "status", "notes",
]

_SHA: dict[str, str] = {}


def guard(path: Path | str) -> None:
    if FORBIDDEN in str(path):
        raise RuntimeError(f"refusing to touch {path}: excluded exploratory audit directory")


def sha(rel: str) -> str:
    guard(rel)
    if rel not in _SHA:
        h = hashlib.sha256()
        with open(ROOT / rel, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _SHA[rel] = h.hexdigest()
    return _SHA[rel]


def rjson(rel: str):
    guard(rel)
    with open(ROOT / rel) as fh:
        return json.load(fh)


def rcsv(rel: str, **kw) -> pd.DataFrame:
    guard(rel)
    return pd.read_csv(ROOT / rel, **kw)


def fin(x):
    """finite float or None"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def circ_word(code: str) -> str:
    return {"I": "INDEPENDENT", "P": "PARTIALLY_RELATED", "C": "SOURCE_RETENTION"}[code]


# ----------------------------------------------------------------------------------------------- representation map
def norm_arm(raw: str, ctx: str = "") -> dict:
    """Map an arm id as written in a results file to the paper panel.  ctx disambiguates stores."""
    A = dict
    if raw == "regulatory_histone_dnase_v1":
        return A(id=raw, variant="svd256_tracks_only (1,959 histone + 533 DNase tracks, no dense cols)", dim=256,
                 main=True, proxy_for="", ck="candidate")
    if raw in ("cpgpt_large_locus", "cpgpt_locus_large", "cpgpt_large_native512", "CpGPT-large native-512"):
        return A(id="cpgpt_large_locus", variant="native_512D", dim=512, main=True, proxy_for="", ck="seq")
    if raw in ("deepcpg_dna_locus", "deepcpg_hcc_native128", "DeepCpG native-128"):
        return A(id="deepcpg_dna_locus", variant="native_128D", dim=128, main=True, proxy_for="", ck="seq")
    if raw == "functional_annotations_pca":
        if ctx == "legacy_bio_validation":
            return A(id=raw, variant="legacy_pca_native_genomewide_256 (older PCA store, not the campaign 'full' SVD)",
                     dim=256, main=False, proxy_for="", ck="legacy")
        return A(id=raw, variant="campaign_full_svd256 (4,165 tracks + 23 dense cols)", dim=256, main=False,
                 proxy_for="", ck="legacy")
    if raw in ("full", "functional_256", "Functional-256"):
        return A(id="functional_annotations_pca", variant="campaign_full_svd256 (4,165 tracks + 23 dense cols)",
                 dim=256, main=False, proxy_for="", ck="legacy")
    if raw == "Functional-128":
        return A(id="functional_annotations_pca", variant="svd128 (exp1)", dim=128, main=False, proxy_for="", ck="legacy")
    if raw in ("context_only", "context_only_256", "Context-only"):
        return A(id="other:context_only", variant="campaign context_only_256 (18 dense context cols, SVD256)", dim=256,
                 main=False, proxy_for="", ck="legacy")
    if raw in ("add/assay/Histone ChIP-seq", "add_assay_histone_chip_seq"):
        return A(id="other:histone_plus_context_campaign",
                 variant="campaign add/assay/Histone ChIP-seq (18 dense context cols + 1,959 histone tracks, SVD256; "
                         "no DNase, includes context columns)", dim=256, main=False,
                 proxy_for="regulatory_histone_dnase_v1", ck="legacy")
    if raw in ("fm/cpgpt_locus_large", "fm_cpgpt_locus_large", "cpgpt_large_svd256", "CpGPT-large compacted-256"):
        return A(id="cpgpt_large_locus", variant="compact_svd256 (campaign fm/cpgpt_locus_large)", dim=256, main=False,
                 proxy_for="cpgpt_large_locus", ck="seq")
    if raw in ("fm/deepcpg_dna_locus", "deepcpg_hcc_pad256", "DeepCpG campaign-256 (std+pad)"):
        return A(id="deepcpg_dna_locus", variant="pad256_std (campaign fm/deepcpg_dna_locus)", dim=256, main=False,
                 proxy_for="deepcpg_dna_locus", ck="seq")
    if raw == "cpgpt_locus":
        return A(id="other:cpgpt_locus_small", variant="native_128D (small CpGPT)", dim=128, main=False, proxy_for="", ck="seq")
    if raw == "deepcpg_dna_locus_hepg2":
        return A(id="other:deepcpg_dna_locus_hepg2", variant="native_128D (HepG2-trained DeepCpG)", dim=128, main=False,
                 proxy_for="", ck="seq")
    if raw == "ntv3_pre":
        return A(id="other:ntv3_pre", variant="native_1536D (chr1-only atlas)", dim=1536, main=False, proxy_for="", ck="seq")
    if raw == "methylgpt_locus":
        return A(id="other:methylgpt_locus", variant="native_64D", dim=64, main=False, proxy_for="", ck="seq")
    if raw == "methylgpt_locus_medium":
        return A(id="other:methylgpt_locus_medium", variant="native_256D", dim=256, main=False, proxy_for="", ck="seq")
    if raw.startswith("raw::"):
        name = raw[5:]
        proxy = "regulatory_histone_dnase_v1" if name == "Histone | alone" else ""
        ck = "legacy"
        return A(id="other:raw_feature_subset", variant=f"raw (unreduced) feature subset: {name}", dim=None, main=False,
                 proxy_for=proxy, ck=ck)
    raise KeyError(f"unmapped representation id: {raw!r}")


# ----------------------------------------------------------------------------------------------- task metadata
CI_V2 = ("95% percentile bootstrap over 22 chromosome blocks (B=1000, seed 17); coarse by design "
         "(frozen protocol sec.0)")
CI_EXP5 = ("95% percentile of 10,000-replicate seed-mixture bootstrap (draw 1 of 20 matching seeds, resample matched "
           "pairs within chromosome); includes matching-draw variability")
CI_EXP2 = "95% percentile chromosome-block bootstrap (B recorded in exp2 script; see scripts/followup/exp2_decodability.py)"
SIGN_V2 = "delta = regulatory_histone_dnase_v1 - comparator (higher-is-better metric)"

V2_META = {
    "loyfer_profile": dict(
        family="methylation program", src="Loyfer et al. 2023 WGBS atlas (39 present cell groups, 205 samples; GRCh38)",
        desc="Rank agreement between embedding cosine and cell-type methylation-profile similarity of CpG pairs "
             "(pooled >1 Mb + interchromosomal strata, equal weight per stratum)",
        bg="CpG pairs sampled in distance strata (<1kb ... >1Mb, inter); no negatives (continuous pair score)",
        match="distance strata only (confounders handled as sensitivity); no embedding used in sampling",
        split="no training; frozen embeddings, cosine score; chromosome-block bootstrap",
        probe="none (cosine similarity of frozen embeddings)", circ=("I", "I", "I"),
        notes="Primary endpoint. Candidate and legacy functional are NEGATIVE (rank below zero) at the primary pooled stat; "
              "sequence arms are positive. cpgpt_large_locus is methylation-supervised (provenance_outcome="
              "DIRECT_METHYLATION_SUPERVISION in exploratory_posthoc_loyfer/interpretation_ABC.json). 849 all-zero "
              "candidate rows excluded for all arms (union rule)."),
    "loyfer_marker_knn": dict(
        family="methylation program", src="Loyfer UXM_deconv U25+U250 hypomethylated cell-type markers (hg38 GitHub set; provenance WEAK)",
        desc="kNN (k=10, cosine) enrichment of same-cell-type marker neighbours over permutation null",
        bg="all other candidate loci (n_candidates in file); permutation within chromosome", match="none for kNN; permutation null",
        split="no training; permutation p-value (n_perm=999)", probe="none (exact top-k cosine neighbours)", circ=("I", "I", "I"),
        notes="SECONDARY: marker provenance verdict WEAK; low power; not in Holm family. Enrichment is fold over null (not AUROC)."),
    "microc_H1_intra10kb": dict(
        family="3D genome", src="4DN H1-hESC Micro-C 4DNFI9GMP2J8 (intra-chromosomal, 10 kb)",
        desc="AUROC of embedding cosine for contact vs distance-matched non-contact CpG pairs",
        bg="matched non-contact pairs (same chromosome, distance bin +-1)",
        match="exact strata: chromosome x 80 log distance bins x coverage quintile x GC tercile x TSS-distance tercile; 1:1",
        split="no training; chromosome-block bootstrap", probe="none (cosine similarity of frozen embeddings)",
        circ=("P", "P", "I"),
        notes="PRIMARY. CTCF/RAD21/SMC3 ChIP tracks (input to candidate and legacy functional) are mechanistically linked to "
              "loop anchors (partially related, not corrected). Effect size secondary = paired delta cosine."),
    "microc_HFFc6_intra10kb": dict(
        family="3D genome", src="4DN HFFc6 Micro-C 4DNFI9FVHJZQ (intra-chromosomal, 10 kb)",
        desc="AUROC of embedding cosine, contact vs matched non-contact (second cell line)", bg="as H1", match="as H1",
        split="no training", probe="none", circ=("P", "P", "I"), notes="Sensitivity of the Micro-C primary (different depth)."),
    "microc_H1_inter1Mb": dict(
        family="3D genome", src="4DN H1 Micro-C, inter-chromosomal 1 Mb",
        desc="AUROC cosine, trans contact vs non-contact", bg="trans non-contact pool (thin)",
        match="chromosome-pair x coverage-decile group; weak balance (SMD 0.30)", split="no training",
        probe="none", circ=("I", "I", "I"),
        notes="EXPLORATORY class: 98.3% total pair drop, weak balance."),
    "compartment_E1_probe_H1_100kb": dict(
        family="3D genome", src="4DN H1 Micro-C E1 eigenvector (cooltools, 100 kb; sign oriented by GC)",
        desc="Spearman(E1, out-of-fold ridge prediction from embedding)", bg="continuous target (no negatives)",
        match="none", split="5-fold chromosome-blocked CV (seed 17)", probe="ridge, alpha grid 1e-2..1e4, inner CV on train chromosomes",
        circ=("P", "P", "P"),
        notes="SECONDARY. E1 sign is GC-oriented (sequence arms see GC); histone/DNase mark chromatin state."),
    "compartment_E1_probe_GM12878_100kb": dict(
        family="3D genome", src="4DN GM12878 in situ Hi-C E1 (100 kb)",
        desc="Spearman(E1, out-of-fold ridge prediction from embedding)", bg="continuous target", match="none",
        split="5-fold chromosome-blocked CV (seed 17)", probe="ridge, alpha grid, inner CV on train chromosomes",
        circ=("P", "P", "P"), notes="SECONDARY; compartments only (no contact table)."),
    "fantom5_membership": dict(
        family="regulatory activity", src="FANTOM5 hg38 permissive enhancers F5.hg38.enhancers.bed.gz (Zenodo 556775)",
        desc="AUROC: CpG inside FANTOM5 enhancer vs matched control CpG (pool bg_enh5k: >=5 kb from any enhancer)",
        bg="CpGs >=5 kb from any FANTOM5 enhancer (257,326), 1:1 nearest-neighbour matched, no replacement",
        match="exact chromosome x CpG context (island/shore/shelf/open sea); numeric caliper 0.25 SD: CpG density, hg38 CpG density, "
              "GC, log10 TSS distance; probe type NOT matched (reported)",
        split="5-fold chromosome-blocked CV (seed 17)",
        probe="logistic L2 (lbfgs); C grid chosen by inner CV on train chromosomes; zero-embedding rows kept",
        circ=("P", "P", "I"),
        notes="PRIMARY. H3K27ac/H3K4me1/DNase/TF ChIP and cCRE enhancers mark the same enhancers (partially related)."),
    "fantom5_activity_similarity": dict(
        family="regulatory activity", src="FANTOM5 CAGE enhancer activity matrix (638 collapsed groups)",
        desc="Spearman(cosine of embeddings, Pearson similarity of enhancer activity profiles) over 171,017 pairs",
        bg="intra (same-enhancer removed) and inter pairs", match="none", split="no training", probe="none",
        circ=("P", "P", "I"), notes="SECONDARY."),
    "rt_consensus": dict(
        family="replication timing", src="UW Repli-seq WaveSignal 15 lines (hg19 lifted to GRCh38), z-score consensus",
        desc="Spearman / R2 of out-of-fold ridge prediction of continuous consensus replication timing",
        bg="continuous target (no negatives)", match="none (GC / CpG density / TSS confounders as residualisation sensitivity)",
        split="5-fold chromosome-blocked CV (seed 17)", probe="ridge, alpha grid 1e-2..1e4, inner CV on train chromosomes",
        circ=("I", "I", "I"), notes="PRIMARY. Strongly GC / gene-density correlated target."),
    "pmd_probe": dict(
        family="replication timing", src="Decato 2020 partially methylated domains (hg19 -> GRCh38 liftover; methylation-derived)",
        desc="AUROC of out-of-fold ridge score for PMD-in-any-sample membership",
        bg="all non-PMD autosomal CpGs of the 408,399 universe (unmatched)", match="none",
        split="5-fold chromosome-blocked CV (seed 17)", probe="ridge on 0/1 label, alpha grid, inner CV on train chromosomes",
        circ=("I", "I", "I"),
        notes="EXPLORATORY class: methylation-derived target, correlated with GC/RT; prevalence 0.805 (328,894/408,399)."),
    "B3_profile_ridge": dict(
        family="methylation program", src="Loyfer WGBS group profiles (raw or centred), ridge from embedding",
        desc="R2 / correlation of out-of-fold ridge prediction of per-group methylation profile",
        bg="continuous multi-output target", match="none", split="5-fold chromosome-blocked (frozen map)",
        probe="ridge, alpha per target chosen by inner leave-one-fold-out CV", circ=("I", "I", "I"),
        notes="EXPLORATORY POST-HOC (registered Part B before running). No CI on absolute values; paired delta CIs in contrasts file."),
    "exploratory_similarity": dict(
        family="methylation program", src="Loyfer WGBS (same pair set as the primary)",
        desc="Spearman of cosine similarity vs profile similarity, per stratum / all pairs", bg="as primary", match="as primary",
        split="no training", probe="none", circ=("I", "I", "I"), notes="EXPLORATORY POST-HOC."),
}
V2_PARENT = {
    "loyfer_mask_lenient": "loyfer_profile", "loyfer_drop_single_sample_groups": "loyfer_profile",
    "compartment_E1_probe_H1_50kb": "compartment_E1_probe_H1_100kb",
    "compartment_E1_probe_GM12878_50kb": "compartment_E1_probe_GM12878_100kb",
    "fantom5_activity_cat_cell_lines": "fantom5_activity_similarity",
    "fantom5_activity_cat_primary_cells": "fantom5_activity_similarity",
    "fantom5_activity_cat_tissues": "fantom5_activity_similarity",
    "fantom5_activity_sample_level": "fantom5_activity_similarity",
}
for _n in (1, 3, 10, 20):
    V2_PARENT[f"fantom5_membership_N{_n}"] = "fantom5_membership"
EXTRA_TASK = {("fantom5_membership", "auroc_oof__win500"): "win500", ("fantom5_membership", "auroc_oof__tss5k_pool"): "tss5k_pool"}
EXTRA_HEADLINE_SAME_TASK = {("rt_consensus", "r2_oof_consensus"), ("compartment_E1_probe_H1_100kb", "r2_oof"),
                            ("compartment_E1_probe_GM12878_100kb", "r2_oof"),
                            ("compartment_E1_probe_H1_50kb", "r2_oof"), ("compartment_E1_probe_GM12878_50kb", "r2_oof")}


def metric_of(stat: str) -> str | None:
    s = stat.lower()
    if s.startswith("auroc"):
        return "AUROC"
    if s.startswith("spearman"):
        return "Spearman"
    if s.startswith("pearsoncorr") or s.startswith("pearson"):
        return "Pearson"
    if s.startswith("r2"):
        return "R2"
    if s.startswith("enrichment"):
        return "enrichment"
    if s.startswith("delta_cosine"):
        return "delta_cosine"
    return None


# ----------------------------------------------------------------------------------------------- registry building
class Reg:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.contrasts: list[dict] = []
        self.unscored: list[dict] = []

    def add(self, **kw) -> None:
        r = {c: "" for c in REG_COLUMNS}
        r.update(kw)
        self.rows.append(r)


def circ_cols(cells, ck):
    cand, leg, seq = (circ_word(c) for c in cells)
    level = {"candidate": cand, "legacy": leg, "seq": seq}[ck]
    return dict(circularity_vs_candidate=cand, circularity_vs_functional_legacy=leg, circularity_vs_sequence=seq,
                circularity_level=level)


def v2_commit(arm: str, sub: str) -> str:
    m = rjson(f"{V2}/{sub}{arm}/run_manifest.json")
    return str(m.get("head_commit", ""))


def build_v2(reg: Reg) -> None:
    arms = ["regulatory_histone_dnase_v1", "cpgpt_large_locus", "deepcpg_dna_locus", "functional_annotations_pca"]
    for sub, suite, status in [("", "v2_frozen", "frozen"), ("sensitivity/", "v2_sensitivity", "preregistered"),
                               ("exploratory_posthoc_loyfer/", "v2_exploratory_posthoc", "exploratory")]:
        for arm in arms:
            na = norm_arm(arm, "v2")
            mf = f"{V2}/{sub}{arm}/run_manifest.json"
            commit = v2_commit(arm, sub)
            epdir = ROOT / V2 / f"{sub}{arm}" / "endpoints"
            for jf in sorted(p.name for p in epdir.glob("*.json")):
                rel = f"{V2}/{sub}{arm}/endpoints/{jf}"
                d = rjson(rel)
                eid = d.get("endpoint_id", jf[:-5])
                if eid == "B3_profile_ridge" or "stats" not in d:
                    continue  # handled from CSV summaries
                parent = V2_PARENT.get(eid, eid)
                meta = V2_META[parent]
                ex = d.get("meta", {}).get("exclusion", {}) if isinstance(d.get("meta"), dict) else {}
                for sname, s in d["stats"].items():
                    met = metric_of(sname)
                    if met is None:
                        continue
                    task_suffix = EXTRA_TASK.get((eid, sname), "")
                    headline = (sname == (d.get("primary_stat") or ("spearman__cosine__all" if eid == "exploratory_similarity" else None))) or ((eid, sname) in EXTRA_HEADLINE_SAME_TASK) or bool(task_suffix)
                    task_id = f"{suite}/{eid}" + (f"__{task_suffix}" if task_suffix else "")
                    smeta = s.get("meta") or {}
                    pos = neg = ""
                    n_items = s.get("n", "")
                    if "n_pos" in smeta:
                        pos, neg = smeta["n_pos"], smeta["n_neg"]
                    elif met == "AUROC" and "n_label1_kept" in ex:
                        pos, neg = ex["n_label1_kept"], ex["n_label0_kept"]
                    elif eid == "pmd_probe":
                        pos, neg = d["meta"]["n_pos"], d["meta"]["n"] - d["meta"]["n_pos"]
                    elif eid == "loyfer_marker_knn":
                        pos, neg = s.get("n", ""), d["meta"].get("n_candidates", "")
                    elif met == "AUROC" and eid.startswith("fantom5_membership") and s.get("n"):
                        pos = neg = int(s["n"]) // 2
                    cls = s.get("class", d.get("class", ""))
                    notes = meta["notes"]
                    if met == "AUROC" and pos != "" and "n_pos" not in smeta and eid.startswith("fantom5_membership"):
                        notes += " positive_n/negative_n derived as n/2 (1:1 matched pairs)."
                    if suite == "v2_exploratory_posthoc":
                        notes = "EXPLORATORY POST-HOC (registered before running, no claim). " + meta["notes"]
                    reg.add(
                        suite=suite, task_id=task_id, task_name=eid, stat_name=sname, stat_class=cls,
                        is_headline=headline, biological_family=meta["family"], target_source=meta["src"],
                        target_description=meta["desc"], positive_n=pos, negative_n=neg, n_items=n_items,
                        background_sampling=meta["bg"], matching_covariates=meta["match"], split_strategy=meta["split"],
                        representation_raw=arm, representation_id=na["id"], representation_variant=na["variant"],
                        representation_dim=na["dim"], is_main_panel=na["main"], proxy_for=na["proxy_for"],
                        classifier_probe=meta["probe"], metric=met, score=fin(s["value"]), ci_lo=fin(s.get("ci_lo")),
                        ci_hi=fin(s.get("ci_hi")), ci_definition=CI_V2 if fin(s.get("ci_lo")) is not None else "",
                        seed=17, source_file=rel, source_sha256=sha(rel), aux_source_files=mf, git_commit_recorded=commit,
                        protocol_tag=V2_TAG, status=status, endpoint_class=cls, notes=notes,
                        **circ_cols(meta["circ"], na["ck"]),
                    )
    # ---- exploratory B3 ridge from CSV (no CI on absolute values)
    rel = f"{V2}/exploratory_posthoc_loyfer/ridge_summary_by_arm.csv"
    d = rcsv(rel)
    d = d[d.kind == "model"]
    meta = V2_META["B3_profile_ridge"]
    for _, r in d.sort_values(["arm", "target", "subset"]).iterrows():
        na = norm_arm(r["arm"], "v2")
        for col, met in [("r2_global", "R2"), ("r2_macro", "R2"), ("pearson_pooled", "Pearson")]:
            v = fin(r[col])
            if v is None:
                continue
            headline = col == "r2_global"
            reg.add(
                suite="v2_exploratory_posthoc", task_id=f"v2_exploratory_posthoc/B3_ridge__{r['target']}__{r['subset']}",
                task_name="B3_profile_ridge", stat_name=f"{r['target']}/{r['subset']}/{col}", stat_class="exploratory_posthoc",
                is_headline=headline, biological_family=meta["family"], target_source=meta["src"],
                target_description=meta["desc"] + f" [target={r['target']}, subset={r['subset']}]", background_sampling=meta["bg"],
                matching_covariates=meta["match"], split_strategy=meta["split"], representation_raw=r["arm"],
                representation_id=na["id"], representation_variant=na["variant"], representation_dim=na["dim"],
                is_main_panel=na["main"], proxy_for=na["proxy_for"], classifier_probe=meta["probe"], metric=met, score=v,
                ci_definition="", seed=17, source_file=rel, source_sha256=sha(rel),
                aux_source_files=f"{V2}/{r['arm']}/run_manifest.json".replace(f"{V2}/", f"{V2}/exploratory_posthoc_loyfer/"),
                git_commit_recorded=v2_commit(r["arm"], "exploratory_posthoc_loyfer/"), protocol_tag=V2_TAG,
                status="exploratory", endpoint_class="exploratory_posthoc",
                notes="EXPLORATORY POST-HOC; no CI on the absolute value (paired delta CI in contrasts file). " + meta["notes"],
                **circ_cols(meta["circ"], na["ck"]),
            )


def build_v2_contrasts(reg: Reg) -> None:
    ref = "regulatory_histone_dnase_v1"
    comps = ["cpgpt_large_locus", "deepcpg_dna_locus", "functional_annotations_pca"]
    n = 0

    def add(**kw):
        nonlocal n
        n += 1
        r = {c: "" for c in CONTRAST_COLUMNS}
        r.update(kw)
        r["contrast_id"] = n
        reg.contrasts.append(r)

    for comp in comps:
        # main frozen
        rel = f"{V2}/contrasts/{ref}__vs__{comp}.csv"
        d = rcsv(rel)
        for _, r in d.sort_values(["endpoint", "stat"]).iterrows():
            eid = r["endpoint"]
            tsk = f"v2_frozen/{eid}" + (f"__{EXTRA_TASK[(eid, r['stat'])]}" if (eid, r["stat"]) in EXTRA_TASK else "")
            add(suite="v2_frozen", task_id=tsk, stat_name=r["stat"], reference_raw=ref, reference_id=ref, comparator_raw=comp,
                comparator_id=comp, delta=fin(r["delta"]), ci_lo=fin(r["ci_lo"]), ci_hi=fin(r["ci_hi"]),
                ci_definition=CI_V2, p_raw=fin(r["p"]), p_adjusted=fin(r.get("holm_p_adjusted")),
                in_holm_family=r["in_holm_family"], sign_convention=SIGN_V2, source_file=rel, source_sha256=sha(rel),
                status="frozen", notes=f"stat_class={r['stat_class']}; supports_regulatory={r.get('supports_regulatory')}")
        # sensitivity
        rel = f"{V2}/sensitivity/contrasts/{ref}__vs__{comp}.csv"
        d = rcsv(rel)
        for _, r in d.sort_values(["item", "stat"]).iterrows():
            add(suite="v2_sensitivity", task_id=f"v2_sensitivity/{r['item']}", stat_name=r["stat"], reference_raw=ref,
                reference_id=ref, comparator_raw=comp, comparator_id=comp, delta=fin(r["delta"]), ci_lo=fin(r["ci_lo"]),
                ci_hi=fin(r["ci_hi"]), ci_definition=CI_V2, p_raw=fin(r["p"]), in_holm_family=False,
                sign_convention=SIGN_V2, source_file=rel, source_sha256=sha(rel), status="preregistered",
                notes="descriptive; raw p; no Holm")
        # exploratory similarity
        rel = f"{V2}/exploratory_posthoc_loyfer/contrasts/similarity__{ref}__vs__{comp}.csv"
        d = rcsv(rel)
        for _, r in d.sort_values(["stat"]).iterrows():
            add(suite="v2_exploratory_posthoc", task_id="v2_exploratory_posthoc/exploratory_similarity", stat_name=r["stat"],
                reference_raw=ref, reference_id=ref, comparator_raw=comp, comparator_id=comp, delta=fin(r["delta"]),
                ci_lo=fin(r["ci_lo"]), ci_hi=fin(r["ci_hi"]), ci_definition=CI_V2, p_raw=fin(r["p"]), in_holm_family=False,
                sign_convention=SIGN_V2, source_file=rel, source_sha256=sha(rel), status="exploratory",
                notes="EXPLORATORY POST-HOC; raw p; no Holm")
        # exploratory ridge
        rel = f"{V2}/exploratory_posthoc_loyfer/contrasts/ridge__{ref}__vs__{comp}.csv"
        d = rcsv(rel)
        d = d[d["group"].isna()]
        for _, r in d.sort_values(["target", "subset", "metric"]).iterrows():
            add(suite="v2_exploratory_posthoc", task_id=f"v2_exploratory_posthoc/B3_ridge__{r['target']}__{r['subset']}",
                stat_name=f"{r['target']}/{r['subset']}/{r['metric']}", reference_raw=ref, reference_id=ref,
                comparator_raw=comp, comparator_id=comp, delta=fin(r["delta"]), ci_lo=fin(r["ci_lo"]), ci_hi=fin(r["ci_hi"]),
                ci_definition=CI_V2, p_raw=fin(r["p"]), in_holm_family=False, sign_convention=SIGN_V2, source_file=rel,
                source_sha256=sha(rel), status="exploratory", notes="EXPLORATORY POST-HOC; raw p; no Holm")


# ---- legacy bio_validation ------------------------------------------------------------------------------
SET_FAMILY = {
    "ewas_catalog_age": ("aging", "EWAS Catalog bulk, traits Age/age/Ageing, p<1e-7 (25 studies)"),
    "ewas_catalog_bmi": ("environmental exposure", "EWAS Catalog bulk, trait BMI, p<1e-7"),
    "ewas_catalog_smoking": ("environmental exposure", "EWAS Catalog bulk, trait smoking, p<1e-7"),
    "ewas_catalog_rheumatoid_arthritis": ("disease/pathology", "EWAS Catalog bulk, RA traits, p<1e-7 (8 studies; includes GSE42861 Liu 2013 by design)"),
    "ewas_catalog_schizophrenia": ("disease/pathology", "EWAS Catalog bulk, schizophrenia, p<1e-7 (10 studies)"),
    "ewas_catalog_breast_cancer": ("disease/pathology", "EWAS Catalog bulk, breast cancer, p<1e-7 (7 studies)"),
    "ewas_catalog_lung_cancer": ("disease/pathology", "EWAS Catalog bulk, lung cancer, p<1e-7 (8 studies)"),
    "ewas_catalog_sle": ("disease/pathology", "Lanata 2022 Arthritis Rheumatol, Suppl. Table S3 (single study, CLUES cohort)"),
    "ewas_catalog_type2_diabetes": ("disease/pathology", "Cardona 2022 Diabetologia ESM Table 5 (meta-analysis 5 cohorts)"),
    "ewas_catalog_coronary_heart_disease": ("disease/pathology", "Agha 2019 Circulation Tables 2+3 (9-cohort meta-analysis)"),
    "ewas_atlas_colorectal_cancer": ("disease/pathology", "EWAS Atlas bulk, colorectal cancer (no p filter; as reported by source studies)"),
    "ewas_atlas_prostate_cancer": ("disease/pathology", "EWAS Atlas bulk, prostate cancer (no p filter)"),
    "ewas_atlas_ovarian_cancer": ("disease/pathology", "EWAS Atlas bulk, ovarian cancer (no p filter)"),
    "horvath_clock": ("aging", "Horvath 2013 pan-tissue clock CpGs (353; biolearn Horvath1.csv)"),
    "hannum_clock": ("aging", "Hannum 2013 blood clock CpGs (71; biolearn Hannum.csv)"),
    "phenoage_clock": ("aging", "Levine 2018 PhenoAge CpGs (513; biolearn PhenoAge.csv)"),
}


def load_sets():
    """Positive sets mapped to the 408,399 universe (canonical cpg_idx via loci.parquet + encode_many)."""
    from cpg_repr_benchmark.data.coordinates import encode_many  # local import: needs PYTHONPATH=src

    loci = rcsv(f"{ENC}/loci.parquet") if False else pd.read_parquet(ROOT / ENC / "loci.parquet")
    can = encode_many(loci.chr, loci.pos)
    sets, raw_len, raw_uniq = {}, {}, {}
    for f in sorted((ROOT / "data/bio_annotations/known_sets").glob("*.npy")):
        a = np.load(f, allow_pickle=False)
        raw_len[f.stem], raw_uniq[f.stem] = len(a), len(np.unique(a))
        sets[f.stem] = set(can[np.isin(can, a)].tolist())
    gc = pd.read_parquet(ROOT / "data/bio_annotations/genomic_context.parquet")
    gcm = gc[gc.cpg_idx.isin(can)]
    ctx_counts = gcm.context.value_counts().to_dict()
    coef = {}
    for nm in ["horvath", "hannum", "phenoage"]:
        p = pd.read_parquet(ROOT / f"data/bio_annotations/{nm}_clock_coefficients.parquet")
        coef[f"{nm}_clock"] = int(p.cpg_idx.isin(can).sum())
    p = pd.read_parquet(ROOT / "data/bio_annotations/phastcons100way_coefficients.parquet")
    coef["phastcons100way"] = int(p.cpg_idx.isin(can).sum())
    return dict(sets=sets, raw_len=raw_len, raw_uniq=raw_uniq, ctx=ctx_counts, ctx_n=len(gcm), coef=coef, universe=len(can))


def overlaps(sets: dict) -> dict:
    out = {}
    for a in sorted(sets):
        best = ("", 0, 0.0)
        for b in sorted(sets):
            if a == b:
                continue
            i = len(sets[a] & sets[b])
            j = i / max(1, len(sets[a] | sets[b]))
            if (i / max(1, len(sets[a]))) > (best[1] / max(1, len(sets[a]))):
                best = (b, i, j)
        out[a] = dict(partner=best[0], shared=best[1], frac_of_self=best[1] / max(1, len(sets[a])), jaccard=best[2])
    return out


LEG_SRC = "outputs/bio_validation/{arm}/native_frozen/seed_17/summary.json"
LEG_ARMS = ["cpgpt_locus", "cpgpt_locus_large", "deepcpg_dna_locus", "deepcpg_dna_locus_hepg2", "functional_annotations_pca",
            "methylgpt_locus", "methylgpt_locus_medium", "ntv3_pre"]
LEG_COMMON = dict(
    split="StratifiedKFold(5, shuffle, random_state=17): RANDOM locus split (neighbouring CpGs fall in train and test)",
    probe="LogisticRegressionCV (default inner CV, L2) on z-scored embedding; AUROC = mean of 5 fold AUROCs",
)


def build_legacy(reg: Reg, info: dict) -> None:
    for arm in LEG_ARMS:
        rel = LEG_SRC.format(arm=arm)
        d = rjson(rel)
        na = norm_arm(arm, "legacy_bio_validation")
        m = d["metrics"]
        base = dict(suite="legacy_bio_validation", representation_raw=arm, representation_id=na["id"],
                    representation_variant=na["variant"], representation_dim=d["embedding_source"]["dim"],
                    is_main_panel=na["main"], proxy_for=na["proxy_for"], seed=17, source_file=rel, source_sha256=sha(rel),
                    protocol_tag="", status="legacy-unregistered", endpoint_class="legacy_exploratory",
                    ci_definition="", git_commit_recorded="")
        # ---- genomic context (one-vs-rest island/shore/shelf/open_sea)
        for cat, v in sorted(m.get("genomic_context", {}).items()):
            if "error" in v:
                reg.unscored.append(dict(suite="legacy_bio_validation", arm=arm, task=f"context/{cat}", reason=v["error"]))
                continue
            pos = info["ctx"].get(cat, "")
            tot = v.get("n", "")
            neg = (tot - pos) if (pos != "" and tot != "") else ""
            for k, met in [("auc", "AUROC"), ("accuracy", "accuracy"), ("f1", "F1"), ("precision", "precision"), ("recall", "recall")]:
                sc = fin(v.get(k))
                if sc is None:
                    continue
                reg.add(
                    task_id=f"legacy_bio_validation/context/{cat}", task_name=f"genomic_context_{cat}", stat_name=k,
                    stat_class="legacy", is_headline=(k == "auc"), biological_family="genomic locus identity",
                    target_source="UCSC cpgIslandExt hg38 (island = inside; shore <=2 kb; shelf 2-4 kb; open sea >4 kb)",
                    target_description=f"one-vs-rest CpG context class '{cat}' of a CpG",
                    positive_n=pos, negative_n=neg, n_items=tot,
                    background_sampling="ALL other CpGs of the annotated universe (no sampling, unmatched)",
                    matching_covariates="none", split_strategy=LEG_COMMON["split"], classifier_probe=LEG_COMMON["probe"],
                    metric=met, score=sc,
                    notes="SOURCE-RETENTION check for functional_annotations_pca (dense cols 0-3 are its inputs); w.r.t. "
                          "regulatory_histone_dnase_v1 island is not an input (frozen table cell I) but is biologically coupled to "
                          "H3K4me3/DNase promoter marks. Island status is sequence-composition defined (PARTIAL for sequence arms). "
                          "Threshold metrics (accuracy/F1/precision/recall) are at p=0.5 under class imbalance. AUPRC not computed.",
                    **circ_cols(("I", "I", "P"), na["ck"]) if False else
                    dict(circularity_vs_candidate="INDEPENDENT", circularity_vs_functional_legacy="SOURCE_RETENTION",
                         circularity_vs_sequence="PARTIALLY_RELATED",
                         circularity_level={"candidate": "INDEPENDENT", "legacy": "SOURCE_RETENTION", "seq": "PARTIALLY_RELATED"}[na["ck"]]),
                    **base)
        # ---- known sets (membership)
        for sname, v in sorted(m.get("known_cpg_sets", {}).items()):
            if "error" in v:
                reg.unscored.append(dict(suite="legacy_bio_validation", arm=arm, task=f"known_set/{sname}", reason=v["error"]))
                continue
            if sname not in SET_FAMILY:
                continue
            fam, src = SET_FAMILY[sname]
            pos = len(info["sets"][sname]) if v.get("n") == info["universe"] else ""
            tot = v.get("n", "")
            neg = (tot - pos) if pos != "" else ""
            for k, met in [("auc", "AUROC"), ("accuracy", "accuracy"), ("f1", "F1"), ("precision", "precision"), ("recall", "recall")]:
                sc = fin(v.get(k))
                if sc is None:
                    continue
                reg.add(
                    task_id=f"legacy_bio_validation/known_set/{sname}", task_name=sname, stat_name=k, stat_class="legacy",
                    is_headline=(k == "auc"), biological_family=fam, target_source=src,
                    target_description="binary membership of a CpG in the literature set (positives = set ∩ 408,399 universe)",
                    positive_n=pos, negative_n=neg, n_items=tot,
                    background_sampling="ALL other universe CpGs (unlisted != unassociated; unmatched, extreme imbalance)",
                    matching_covariates="none", split_strategy=LEG_COMMON["split"], classifier_probe=LEG_COMMON["probe"],
                    metric=met, score=sc,
                    notes="Array-probe design is biased towards promoters/CpG islands (covariate, not circularity). AUPRC not "
                          "computed; baseline AUPRC would equal prevalence = positive_n/n_items. Threshold metrics at p=0.5.",
                    **circ_cols(("I", "I", "I"), na["ck"]), **base)
        # ---- clock coefficient regression & phastCons
        for cname, v in sorted(m.get("clock_coefficients", {}).items()):
            if "error" in v:
                reg.unscored.append(dict(suite="legacy_bio_validation", arm=arm, task=f"coef/{cname}", reason=v["error"]))
                continue
            is_p = cname == "phastcons100way"
            fam = "genomic locus identity" if is_p else "aging"
            src = ("phastCons 100-way conservation score per CpG (data/bio_annotations/phastcons100way_coefficients.parquet)"
                   if is_p else SET_FAMILY[cname][1] + " ; elastic-net coefficient per CpG")
            for k, met in [("r2", "R2"), ("pearson", "Pearson"), ("mae", "MAE")]:
                sc = fin(v.get(k))
                if sc is None:
                    continue
                reg.add(
                    task_id=f"legacy_bio_validation/coef/{cname}", task_name=f"{cname}_coefficient_regression", stat_name=k,
                    stat_class="legacy", is_headline=(k == "r2"), biological_family=fam, target_source=src,
                    target_description="continuous per-CpG weight (clock elastic-net coefficient or conservation score); only CpGs with a weight",
                    positive_n="", negative_n="", n_items=v.get("n", ""),
                    background_sampling="regression on the CpGs that have a weight (no negatives)", matching_covariates="none",
                    split_strategy="KFold(5, shuffle, random_state=17): RANDOM locus split",
                    classifier_probe="RidgeCV alphas 1e-2..1e4 (inner efficient LOO) on z-scored embedding; mean of 5 fold metrics",
                    metric=met, score=sc,
                    notes=("n=%s CpGs: " % v.get("n", "")) + ("very small n; negative R2 = worse than mean predictor. " if not is_p else
                           "large n but spatially autocorrelated target with random split. ") + "No CI.",
                    **circ_cols(("I", "I", "I"), na["ck"]), **base)
        # ---- locality
        v = m.get("locality") or {}
        if v and "error" not in v:
            for k, met in [("enrichment", "enrichment"), ("neighbor_hit_rate", "neighbor_hit_rate")]:
                sc = fin(v.get(k))
                if sc is None:
                    continue
                reg.add(
                    task_id="legacy_bio_validation/locality", task_name="locality_knn_genomic_proximity", stat_name=k,
                    stat_class="legacy", is_headline=(k == "enrichment"), biological_family="genomic locus identity",
                    target_source="CpG genomic coordinates (master_cpg_registry)",
                    target_description="fraction of 10 embedding nearest neighbours within 100 kb on the same chromosome vs random pairing",
                    positive_n="", negative_n="", n_items=v.get("n_sampled", ""),
                    background_sampling="random pairing null (null_hit_rate in file)", matching_covariates="none",
                    split_strategy="unsupervised; 2,000 randomly sampled query CpGs of the universe",
                    classifier_probe="exact kNN, Euclidean on z-scored dimensions (no training)", metric=met, score=sc,
                    notes="Unsupervised, no CI, 2,000 queries. Candidate/legacy tracks of neighbouring CpGs overlap the same peak "
                          "intervals, so locality is expected by construction (judgement; not in the frozen table).",
                    **dict(circularity_vs_candidate="PARTIALLY_RELATED", circularity_vs_functional_legacy="PARTIALLY_RELATED",
                           circularity_vs_sequence="INDEPENDENT",
                           circularity_level={"candidate": "PARTIALLY_RELATED", "legacy": "PARTIALLY_RELATED", "seq": "INDEPENDENT"}[na["ck"]]),
                    **base)
        if arm == "ntv3_pre" and not m.get("genomic_context"):
            reg.unscored.append(dict(suite="legacy_bio_validation", arm=arm, task="all", reason="all probes errored (chr1-only atlas; stale set names)"))


# ---- encode atlas ----------------------------------------------------------------------------------------
def build_ewas_matched(reg: Reg, info: dict) -> None:
    rel = f"{ENC}/ewas_matched_membership.csv"
    d = rcsv(rel)
    for _, r in d.iterrows():
        if r["status"] != "complete":
            reg.unscored.append(dict(suite="encode_ewas_matched_single_draw", arm=str(r["arm"]), task=str(r["set"]), reason=str(r["status"])))
            continue
        na = norm_arm(r["arm"])
        fam, src = SET_FAMILY[r["set"]]
        npairs = int(r["n"]) // 2
        for col, met in [("auc", "AUROC"), ("average_precision_matched_sample", "AUPRC")]:
            reg.add(
                suite="encode_ewas_matched_single_draw", task_id=f"encode_ewas_matched_single_draw/{r['set']}", task_name=r["set"],
                stat_name=col, stat_class="campaign", is_headline=(col == "auc"), biological_family=fam, target_source=src,
                target_description="listed vs matched unlisted CpG (single matching draw, <=2,000 pairs)",
                positive_n=npairs, negative_n=npairs, n_items=int(r["n"]),
                background_sampling="1 unlisted CpG per listed CpG, same stratum, no replacement (unlisted != unassociated)",
                matching_covariates="chromosome x CpG context x gene region x core-breadth decile (dense cols 0-7, 22 of the functional store)",
                split_strategy="GroupKFold(5) by chromosome; pooled out-of-fold score",
                representation_raw=r["arm"], representation_id=na["id"], representation_variant=na["variant"],
                representation_dim=na["dim"], is_main_panel=na["main"], proxy_for=na["proxy_for"],
                classifier_probe="StandardScaler (fold-wise) + LogisticRegression C=1 (fixed, no tuning)", metric=met,
                score=fin(r[col]), ci_definition="", seed="17001", source_file=rel, source_sha256=sha(rel), status="discovery-campaign",
                endpoint_class="campaign_single_draw",
                notes="Single matching draw, no CI. AUPRC is on the matched sample (prevalence 0.5 by construction; baseline 0.5). "
                      "Matching covariates are inputs of the functional store (gene region/context/breadth), not of regulatory_histone_dnase_v1.",
                **circ_cols(("I", "I", "I"), na["ck"]))


def build_exp5(reg: Reg, info: dict) -> None:
    base = f"{ENC}/followup/exp5"
    rel = f"{base}/summary_auc_delta.csv"
    ev = rcsv(f"{base}/evaluable_sets.csv")
    pairs = {(r["variant"], r["set"]): r["pairs_mean"] for _, r in ev.iterrows() if r["status"] == "EVALUABLE"}
    for _, r in ev[ev.status != "EVALUABLE"].iterrows():
        reg.unscored.append(dict(suite=f"encode_exp5_{r['variant']}", arm="all", task=r["set"],
                                 reason=f"NOT EVALUABLE (matched pairs {r['pairs_mean']:.0f} < 100 or <5 chromosomes)"))
    d = rcsv(rel)
    dd = d[d.kind == "auc"]
    for _, r in dd.sort_values(["variant", "set", "representation"]).iterrows():
        na = norm_arm(r["representation"])
        fam, src = SET_FAMILY[r["set"]]
        p = pairs[(r["variant"], r["set"])]
        v = r["variant"]
        reg.add(
            suite=f"encode_exp5_{v}", task_id=f"encode_exp5_{v}/{r['set']}", task_name=r["set"], stat_name="pooled_oof_auc_mean_over_seeds",
            stat_class=r["family"], is_headline=True, biological_family=fam, target_source=src,
            target_description="listed vs matched unlisted CpG, 20 independent matching draws",
            positive_n=int(p), negative_n=int(p), n_items=int(2 * p),
            background_sampling="1 unlisted CpG per listed CpG in the same stratum, no replacement, <=2,000 pairs/draw; unlisted != unassociated",
            matching_covariates=("chromosome x CpG context x gene region x core-breadth decile" +
                                 (" x Infinium probe type (I/II; loci with unknown type excluded)" if v == "probe" else
                                  "; Infinium probe type NOT matched (balance reported)") + "; GC / TSS distance NOT matched"),
            split_strategy="GroupKFold(5) by chromosome; pooled out-of-fold AUC",
            representation_raw=r["representation"], representation_id=na["id"], representation_variant=na["variant"],
            representation_dim=na["dim"], is_main_panel=na["main"], proxy_for=na["proxy_for"],
            classifier_probe="StandardScaler (fold-wise) + LogisticRegression C=1 (fixed, no tuning)", metric="AUROC",
            score=fin(r["estimate"]), ci_lo=fin(r["ci_lo"]), ci_hi=fin(r["ci_hi"]), ci_definition=CI_EXP5,
            seed="17001-17020 (20 matching draws)", source_file=rel, source_sha256=sha(rel),
            aux_source_files=f"{base}/protocol.json;{base}/evaluable_sets.csv", git_commit_recorded="",
            protocol_tag="", status="discovery-campaign", endpoint_class="campaign_robustness",
            notes=f"between_seed_sd={fin(r['between_seed_sd'])}; seed range [{fin(r['seed_min'])}, {fin(r['seed_max'])}]. "
                  "regulatory_histone_dnase_v1 and native-DNase/histone arms NOT included in this run. AUPRC not computed. "
                  "Matching uses functional-store dense columns (inputs of the legacy functional/context arms).",
            **circ_cols(("I", "I", "I"), na["ck"]))
    # contrasts (functional_256 - X) -> contrasts file
    for _, r in d[d.kind == "delta"].sort_values(["variant", "set", "comparison"]).iterrows():
        a, b = [s.strip() for s in r["comparison"].split(" - ")]
        reg.contrasts.append({**{c: "" for c in CONTRAST_COLUMNS},
            "contrast_id": 100000 + len(reg.contrasts), "suite": f"encode_exp5_{r['variant']}",
            "task_id": f"encode_exp5_{r['variant']}/{r['set']}", "stat_name": "pooled_oof_auc_delta", "reference_raw": a,
            "reference_id": norm_arm(a)["id"] + "|" + norm_arm(a)["variant"], "comparator_raw": b,
            "comparator_id": norm_arm(b)["id"] + "|" + norm_arm(b)["variant"], "delta": fin(r["estimate"]), "ci_lo": fin(r["ci_lo"]),
            "ci_hi": fin(r["ci_hi"]), "ci_definition": CI_EXP5, "p_raw": fin(r["p_boot"]), "p_adjusted": fin(r["q_bh"]),
            "in_holm_family": "BH within variant x comparison (not Holm)", "sign_convention": "delta = functional_256 - comparator (AUC)",
            "source_file": rel, "source_sha256": sha(rel), "status": "discovery-campaign",
            "notes": f"n_seeds_delta_positive={r['n_seeds_delta_positive']}/{r['n_seeds']}; reference is LEGACY functional, not the candidate"})


def build_exp2(reg: Reg) -> None:
    rel = f"{ENC}/followup/exp2/summary.csv"
    d = rcsv(rel)
    for _, r in d.iterrows():
        raw = r["name"] if r["kind"] == "rep" else f"raw::{r['name']}"
        na = norm_arm(raw)
        for col, met, lo, hi in [("r2", "R2", "r2_ci_lo", "r2_ci_hi"), ("pearson", "Pearson", None, None),
                                 ("spearman", "Spearman", None, None), ("mse", "MSE", "mse_ci_lo", "mse_ci_hi")]:
            reg.add(
                suite="encode_exp2_decodability", task_id="encode_exp2_decodability/mean_beta", task_name="mean_beta_decodability",
                stat_name=col, stat_class="campaign", is_headline=(col in ("r2", "spearman")), biological_family="methylation program",
                target_source="mean beta of each locus over TCGA discovery patients (count >= max(20, 50% of patients)); 11,929 loci",
                target_description="locus mean methylation across benchmark DISCOVERY patients (patient-derived label, not an input of any arm)",
                positive_n="", negative_n="", n_items=int(r["n"]), background_sampling="continuous target (no negatives)",
                matching_covariates="none", split_strategy="nested chromosome CV: outer GroupKFold(5) by chromosome, inner GroupKFold(3)",
                representation_raw=raw, representation_id=na["id"], representation_variant=na["variant"],
                representation_dim=na["dim"] if na["dim"] else (int(r["active_dims"]) if fin(r["active_dims"]) else ""),
                is_main_panel=na["main"], proxy_for=na["proxy_for"],
                classifier_probe="StandardScaler(with_mean=False)+Ridge(lsqr), alpha in {0.1,10,1000} chosen by inner chromosome CV",
                metric=met, score=fin(r[col]), ci_lo=fin(r[lo]) if lo else None, ci_hi=fin(r[hi]) if hi else None,
                ci_definition=CI_EXP2 if lo else "", seed="", source_file=rel, source_sha256=sha(rel), status="discovery-campaign",
                endpoint_class="campaign_followup",
                notes="Target is derived from benchmark training (discovery) patients: patient information enters the LABEL, not the representation. "
                      "Only 11,929 of 408,399 loci. Subset 'raw (unreduced) feature subset' rows are not SVD embeddings.",
                **circ_cols(("I", "I", "I"), na["ck"]))
    rel2 = f"{ENC}/followup/exp2/paired_vs_functional256.csv"
    pr = rcsv(rel2)
    for _, r in pr.iterrows():
        ci = json.loads(r["d_r2_ci"])
        reg.contrasts.append({**{c: "" for c in CONTRAST_COLUMNS}, "contrast_id": 200000 + len(reg.contrasts),
            "suite": "encode_exp2_decodability", "task_id": "encode_exp2_decodability/mean_beta", "stat_name": "r2_delta",
            "reference_raw": "Functional-256", "reference_id": "functional_annotations_pca|campaign_full_svd256",
            "comparator_raw": r["comparator"], "comparator_id": "see comparator_raw", "delta": fin(r["d_r2"]), "ci_lo": fin(ci[0]),
            "ci_hi": fin(ci[1]), "ci_definition": CI_EXP2, "p_raw": fin(r["p_boot_two_sided"]), "in_holm_family": "no",
            "sign_convention": "d_r2 = R2(Functional-256) - R2(comparator)", "source_file": rel2, "source_sha256": sha(rel2),
            "status": "discovery-campaign", "notes": f"chromosomes won by functional: {r['chrom_wins_functional']}; reference is LEGACY functional"})


def build_comethylation(reg: Reg) -> None:
    arms = [("full_a18b869b", "full"), ("context_only_2805a394", "context_only"),
            ("add_assay_histone_chip_seq_2d418bee", "add/assay/Histone ChIP-seq"),
            ("fm_cpgpt_locus_large_d3474a48", "fm/cpgpt_locus_large")]
    for stem, raw in arms:
        rel = f"{ENC}/co_methylation/confirm/{stem}.csv"
        meta_rel = f"{ENC}/co_methylation/confirm/{stem}.json"
        d = rcsv(rel)
        na = norm_arm(raw)
        n = int(d.neighbor_correlation.notna().sum())
        nm = int(d.matched_correlation.notna().sum())
        vals = {"mean_neighbor_correlation": d.neighbor_correlation.mean(), "mean_matched_correlation": d.matched_correlation.mean(),
                "correlation_gap": (d.neighbor_correlation - d.matched_correlation).mean()}
        for k, v in vals.items():
            reg.add(
                suite="encode_co_methylation", task_id="encode_co_methylation/far_neighbour_correlation",
                task_name="co_methylation_far_neighbours", stat_name=k, stat_class="campaign", is_headline=(k == "correlation_gap"),
                biological_family="methylation program",
                target_source="held-out-patient beta correlation between a query CpG and its 5 nearest far (>=1 Mb) embedding neighbours",
                target_description="mean Pearson correlation of beta (held-out patients) neighbour vs distance/context/variance-matched control",
                positive_n=n, negative_n=nm, n_items=len(d),
                background_sampling="one matched reference CpG per neighbour (distance within 2x, same context, same variance quintile)",
                matching_covariates="distance band / context / beta-SD quintile; correlations never used to choose matches",
                split_strategy="evaluation on held-out patients (patient list in the json); no training", representation_raw=raw,
                representation_id=na["id"], representation_variant=na["variant"], representation_dim=na["dim"],
                is_main_panel=na["main"], proxy_for=na["proxy_for"], classifier_probe="none (cosine nearest neighbours)",
                metric=k if k in METRICS else "correlation_gap", score=fin(v), ci_definition="", seed="", source_file=rel,
                source_sha256=sha(rel), aux_source_files=meta_rel, status="discovery-campaign", endpoint_class="campaign_confirm",
                notes="Score DERIVED by the registry script as the mean over finite query-neighbour pairs of the per-pair CSV columns (no CI in the "
                      "result file). Uses held-out patients' betas at evaluation time only (patients in the json); representation patient-agnostic. "
                      "positive_n = pairs with finite neighbour correlation; negative_n = pairs with finite matched correlation.",
                **circ_cols(("I", "I", "I"), na["ck"]))


def build_umap(reg: Reg) -> None:
    rel = "outputs/figures/umap_biology/neighborhood_metrics.csv"
    d = rcsv(rel)
    for _, r in d.iterrows():
        na = norm_arm(r["representation"], "v2")
        ann = r["annotation"]
        is_ctx = ann in ("island", "shore", "shelf", "open_sea")
        fam, src = (("genomic locus identity", "UCSC cpgIslandExt") if is_ctx else SET_FAMILY[ann])
        circs = ({"candidate": "INDEPENDENT", "legacy": "SOURCE_RETENTION", "seq": "PARTIALLY_RELATED"} if is_ctx else
                 {"candidate": "INDEPENDENT", "legacy": "INDEPENDENT", "seq": "INDEPENDENT"})
        for col, met in [("enrichment", "enrichment"), ("neighbor_fraction", "neighbor_fraction"), ("random_fraction", "random_fraction")]:
            reg.add(
                suite="umap_neighborhood", task_id=f"umap_neighborhood/{ann}/{r['space']}", task_name=f"neighborhood_{ann}",
                stat_name=f"{col}[space={r['space']},seed={'' if pd.isna(r['seed']) else int(r['seed'])}]", stat_class="descriptive",
                is_headline=(col == "enrichment" and r["space"] == "native"), biological_family=fam, target_source=src,
                target_description=f"fraction of k=30 neighbours (in {r['space']} space of a 12,000-point sample) that are '{ann}' positives vs random",
                positive_n=int(r["n_positive"]), negative_n="", n_items=12000, background_sampling="random fraction in the same sample",
                matching_covariates="none", split_strategy="none (descriptive neighbourhood statistic on a 12,000-point sample)",
                representation_raw=r["representation"], representation_id=na["id"], representation_variant=na["variant"] + f"; space={r['space']}",
                representation_dim=na["dim"], is_main_panel=na["main"], proxy_for=na["proxy_for"],
                classifier_probe="kNN (k=30) in native / PCA50 / UMAP space", metric=met, score=fin(r[col]), seed="" if pd.isna(r["seed"]) else int(r["seed"]),
                source_file=rel, source_sha256=sha(rel), aux_source_files="outputs/figures/umap_biology/manifest.json", status="legacy-unregistered",
                endpoint_class="legacy_descriptive",
                notes="Side output of scripts/plot_umap_biology.py; no regulatory_histone_dnase_v1; no CI; visualization-space values (pca/umap) are not "
                      "representation quality metrics.",
                circularity_vs_candidate=circs["candidate"], circularity_vs_functional_legacy=circs["legacy"],
                circularity_vs_sequence=circs["seq"], circularity_level=circs[na["ck"]])


# ----------------------------------------------------------------------------------------------- coverage
def coverage_table(rows: list[dict]) -> pd.DataFrame:
    """task_id x main-panel arm -> HAVE / PROXY_VARIANT_ONLY / MISSING (+ proxy description)."""
    df = pd.DataFrame(rows)
    out = []
    for tid, g in df.groupby("task_id", sort=True):
        rec = {"task_id": tid, "suite": g["suite"].iloc[0]}
        for arm in MAIN_PANEL:
            have = g[(g.representation_id == arm) & (g.is_main_panel.astype(bool))]
            if len(have):
                rec[arm] = "HAVE"
                rec[arm + "__proxy"] = ""
                continue
            prox = g[g.proxy_for == arm]
            if len(prox):
                rec[arm] = "PROXY_VARIANT_ONLY"
                rec[arm + "__proxy"] = ";".join(sorted(prox.representation_raw.unique()))
            else:
                rec[arm] = "MISSING"
                rec[arm + "__proxy"] = ""
        rec["functional_annotations_pca_legacy"] = "HAVE" if (g.representation_id == "functional_annotations_pca").any() else "MISSING"
        out.append(rec)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------------- matrix
def evidence_level(status: str, ec: str) -> str:
    if status == "frozen":
        return {"primary": "frozen primary", "secondary": "frozen secondary", "exploratory": "frozen exploratory",
                "sensitivity": "frozen sensitivity", "descriptive": "frozen descriptive"}.get(ec, "frozen " + ec)
    return {"preregistered": "preregistered sensitivity", "exploratory": "exploratory post-hoc (registered)",
            "discovery-campaign": "discovery-campaign", "legacy-unregistered": "legacy-unregistered"}[status]


def pref_sort(g: pd.DataFrame) -> pd.DataFrame:
    """deterministic choice among several variants of one arm: campaign_full 256D / native first."""
    g = g.copy()
    g["_k"] = (~g.representation_variant.astype(str).str.startswith(("campaign_full", "native"))).astype(int)
    return g.sort_values(["_k", "representation_raw"], kind="mergesort").drop(columns="_k")


def build_matrix(reg_df: pd.DataFrame, cov: pd.DataFrame, con_df: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    head = reg_df[reg_df.is_headline.astype(bool)]
    arms = list(ARMKEY.items())
    rows = []
    covi = cov.set_index("task_id")
    audi = audit.set_index("task_id")
    prio = {"AUROC": 0, "Spearman": 1, "enrichment": 2, "R2": 3, "AUPRC": 4}
    for tid, g in head.groupby("task_id", sort=True):
        g = g.assign(_p=g.metric.map(prio).fillna(9)).sort_values(["_p", "stat_name", "representation_raw"], kind="mergesort").drop(columns="_p")
        full = reg_df[reg_df.task_id == tid]
        r = {"task_id": tid, "task_name": g.task_name.iloc[0], "suite": g.suite.iloc[0], "biological_family": g.biological_family.iloc[0],
             "evidence_level": evidence_level(g.status.iloc[0], str(g.endpoint_class.iloc[0])), "status": g.status.iloc[0],
             "endpoint_class": g.endpoint_class.iloc[0],
             "circularity_vs_candidate": g.circularity_vs_candidate.iloc[0],
             "circularity_vs_functional_legacy": g.circularity_vs_functional_legacy.iloc[0],
             "circularity_vs_sequence": g.circularity_vs_sequence.iloc[0],
             "rigor_grade": audi.loc[tid, "rigor_grade"] if tid in audi.index else "",
             "positive_n": "", "negative_n": "", "n_items": ""}
        # N from the first main-panel / first row of the headline
        first = g.iloc[0]
        r["positive_n"], r["negative_n"], r["n_items"] = first.positive_n, first.negative_n, first.n_items
        # candidate if available else any
        for a, ak in arms:
            ga = g[(g.representation_id == a) & ((g.is_main_panel.astype(bool)) | (a == "functional_annotations_pca"))]
            for met, mk in MATRIX_METRICS.items():
                gm = ga[ga.metric == met]
                # AUPRC and headline-less metrics come from the same task (full set) when not flagged headline
                if gm.empty and met in ("AUPRC",):
                    gm = full[(full.representation_id == a) & (full.metric == met)]
                    if a != "functional_annotations_pca":
                        gm = gm[gm.is_main_panel.astype(bool)]
                gm = pref_sort(gm)
                r[f"{ak}__{mk}"] = gm.score.iloc[0] if len(gm) else ""
                if mk != "auprc":
                    r[f"{ak}__{mk}__ci_lo"] = gm.ci_lo.iloc[0] if len(gm) else ""
                    r[f"{ak}__{mk}__ci_hi"] = gm.ci_hi.iloc[0] if len(gm) else ""
        for met, mk in MATRIX_METRICS.items():
            if mk == "auprc":
                continue
            defs = sorted({x for x in g[g.metric == met].ci_definition.astype(str) if x})
            r[f"{mk}__ci_definition"] = defs[0] if defs else ""
        # coverage
        for arm in MAIN_PANEL:
            r[f"coverage_{ARMKEY[arm]}"] = covi.loc[tid, arm]
            r[f"proxy_arm_{ARMKEY[arm]}"] = covi.loc[tid, arm + "__proxy"]
        # proxy scores for missing main arms (headline metric only: first matrix metric present in headline)
        hmet = g.metric.iloc[0]
        for arm in MAIN_PANEL:
            ak = ARMKEY[arm]
            r[f"proxy_metric_{ak}"], r[f"proxy_value_{ak}"] = "", ""
            if covi.loc[tid, arm] == "PROXY_VARIANT_ONLY":
                px = full[(full.proxy_for == arm) & (full.metric == hmet) & (full.is_headline.astype(bool) | True)]
                px = px[px.stat_name == g.stat_name.iloc[0]] if (px.stat_name == g.stat_name.iloc[0]).any() else px
                px = pref_sort(px)
                if len(px):
                    r[f"proxy_metric_{ak}"], r[f"proxy_value_{ak}"] = hmet, px.score.iloc[0]
        r["headline_stat"], r["headline_metric"] = g.stat_name.iloc[0], g.metric.iloc[0]
        for a, ak in arms:
            ga = g[(g.representation_id == a) & ((g.is_main_panel.astype(bool)) | (a == "functional_annotations_pca")) & (g.stat_name == g.stat_name.iloc[0])]
            ga = pref_sort(ga)
            r[f"headline_value_{ak}"] = ga.score.iloc[0] if len(ga) else ""
        r["legacy_functional_variant"] = ";".join(sorted(g[g.representation_id == "functional_annotations_pca"].representation_variant.unique()))
        # deltas
        cc = con_df[(con_df.task_id == tid) & (con_df.reference_id == "regulatory_histone_dnase_v1")]
        have_any = False
        for met, mk in MATRIX_METRICS.items():
            if mk == "auprc":
                continue
            stat_names = g[g.metric == met].stat_name.unique()
            for comp, ck in [("cpgpt_large_locus", "cpgpt"), ("deepcpg_dna_locus", "deepcpg"), ("functional_annotations_pca", "funcleg")]:
                sel = cc[(cc.comparator_id == comp) & (cc.stat_name.isin(stat_names))]
                col = f"candidate_minus_{ck}__{mk}"
                if len(sel):
                    s = sel.sort_values("stat_name").iloc[0]
                    r[col], r[col + "__ci_lo"], r[col + "__ci_hi"] = s.delta, s.ci_lo, s.ci_hi
                    r[col + "__ci_definition"] = s.ci_definition
                    have_any = True
                else:
                    r[col], r[col + "__ci_lo"], r[col + "__ci_hi"], r[col + "__ci_definition"] = "", "", "", ""
        if have_any:
            r["delta_flag"] = "paired delta CI available (95% paired chromosome-block bootstrap)"
        else:
            r["delta_flag"] = "no paired CI available"
        r["notes"] = "; ".join(sorted({str(x) for x in g.notes if x})[:1])
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- audit
def audit_rows(reg_df: pd.DataFrame, info: dict, ov: dict, cov: pd.DataFrame, unscored: list[dict]) -> pd.DataFrame:
    out = []
    covi = cov.set_index("task_id")

    def reps(tid):
        g = reg_df[reg_df.task_id == tid]
        return ";".join(sorted(g.representation_raw.unique()))

    def covs(tid):
        return {f"coverage_{ARMKEY[a]}": covi.loc[tid, a] for a in MAIN_PANEL}

    def ciinfo(tid):
        g = reg_df[(reg_df.task_id == tid) & reg_df.ci_lo.astype(str).ne("")]
        d = sorted({x for x in g.ci_definition.astype(str) if x})
        return ("yes" if len(g) else "no"), (d[0] if d else "")

    for tid in sorted(reg_df.task_id.unique()):
        g = reg_df[reg_df.task_id == tid]
        suite = g.suite.iloc[0]
        name = g.task_name.iloc[0]
        h = g[g.is_headline.astype(bool)]
        h0 = h.iloc[0] if len(h) else g.iloc[0]
        has_ci, ci_def = ciinfo(tid)
        rec = dict(task_id=tid, task_name=name, suite=suite, biological_family=h0.biological_family, status=h0.status,
                   endpoint_class=h0.endpoint_class, representations_tested=reps(tid), headline_metric=h0.metric,
                   positive_n=h0.positive_n, negative_n=h0.negative_n, n_items=h0.n_items,
                   circularity_vs_candidate=h0.circularity_vs_candidate, circularity_vs_functional_legacy=h0.circularity_vs_functional_legacy,
                   circularity_vs_sequence=h0.circularity_vs_sequence, ci_present=has_ci, ci_definition=ci_def,
                   preregistered=("yes (frozen protocol tag " + V2_TAG + ")" if suite == "v2_frozen" else
                                  "yes (registered before running: sensitivity doc Part A)" if suite == "v2_sensitivity" else
                                  "registered Part B before running, but post-hoc exploratory" if suite == "v2_exploratory_posthoc" else "no"),
                   **covs(tid))
        try:
            pos, tot = float(h0.positive_n), float(h0.n_items)
            rec["prevalence_in_probe_set"] = round(pos / tot, 6) if (tot and h0.metric == "AUROC") else ""
        except (TypeError, ValueError):
            rec["prevalence_in_probe_set"] = ""
        rec["set_raw_size"], rec["set_fraction_in_universe"] = "", ""
        if name in info["sets"]:
            rec["set_raw_size"] = info["raw_len"][name]
            rec["set_fraction_in_universe"] = round(len(info["sets"][name]) / info["raw_len"][name], 5)
        auprc = g[g.metric == "AUPRC"]
        rec["auprc_reported"] = "yes (matched-sample prevalence 0.5)" if len(auprc) else "no"
        rec["problems"] = ""
        rec["probe_kind"] = ""
        rec["patient_derived_data_in_labels"] = "no (annotation/assay target)"
        rec["patient_derived_data_in_representation"] = "no for all panel arms (patient-agnostic stores; cpgpt/deepcpg pretrained externally)"
        rec["source_cohort_overlap_with_training"] = "not applicable (external assay target)"
        rec["representation_specific_advantages"] = ""
        rec["overlap_with_other_sets"] = ""
        rec["duplicates"] = ""
        rec["hyperparameter_selection"] = ""
        rec["tuning_on_eval_labels"] = ""
        rec["negatives"] = ""
        rec["chromosome_matching"] = ""
        rec["context_matching"] = ""
        rec["probe_type_matching"] = ""
        rec["gc_tss_matching"] = ""
        rec["split_strategy"] = h0.split_strategy
        rec["locus_leakage_checks"] = ""
        grade, why = "", []

        if suite in ("v2_frozen", "v2_sensitivity", "v2_exploratory_posthoc"):
            parent = V2_PARENT.get(name, name)
            ec = h0.endpoint_class
            rec["probe_kind"] = "locus-level assay target (non-phenotype)"
            rec["negatives"] = h0.background_sampling
            m = h0.matching_covariates
            rec["chromosome_matching"] = "yes" if ("chromosome" in m) else "n/a or no"
            rec["context_matching"] = "yes (island/shore/shelf/open sea)" if "context" in m and parent == "fantom5_membership" else "no"
            rec["probe_type_matching"] = "no (reported balance only)" if parent == "fantom5_membership" else "n/a"
            rec["gc_tss_matching"] = ("yes (GC, log10 TSS distance, CpG density caliper)" if parent == "fantom5_membership" else
                                      "yes (GC tercile, TSS tercile)" if parent.startswith("microc") else
                                      "no (confounders as sensitivity)" if parent in ("rt_consensus", "loyfer_profile") else "n/a")
            rec["hyperparameter_selection"] = "inner CV restricted to train chromosomes (never test chromosomes)" if "ridge" in h0.classifier_probe or "logistic" in h0.classifier_probe else "no tuning (cosine / kNN, no model)"
            rec["tuning_on_eval_labels"] = "no"
            rec["locus_leakage_checks"] = "chromosome-blocked folds (assert_no_leak); matching never uses embeddings; same item set for all arms (union zero-row exclusion for cosine endpoints)"
            rec["representation_specific_advantages"] = ("regulatory_histone_dnase_v1 has 849 all-zero rows: excluded from cosine endpoints for ALL arms (D2), kept in probes; "
                                                         "dimensions differ (256/512/128) - cosine is dimension-free but probes are not")
            rec["source_cohort_overlap_with_training"] = "none documented; external assays (Loyfer, 4DN, FANTOM5, Repli-seq, Decato PMD)"
            if parent in ("loyfer_profile", "loyfer_marker_knn", "B3_profile_ridge", "exploratory_similarity"):
                rec["representation_specific_advantages"] += "; cpgpt_large_locus is methylation-supervised (DIRECT_METHYLATION_SUPERVISION)"
            if suite == "v2_frozen" and ec == "primary":
                grade = "MAIN_QUALITY"; why = ["frozen primary, chromosome-blocked, matched/stratified design, CIs (coarse: 22 blocks, B=1000)"]
                if parent == "fantom5_membership":
                    why.append("circularity PARTIALLY_RELATED (shared enhancer chromatin evidence)")
                if parent == "microc_H1_intra10kb":
                    why.append("circularity PARTIALLY_RELATED (CTCF/cohesin tracks vs loop anchors); effect small (AUROC ~0.50-0.57)")
                if parent == "loyfer_profile":
                    why.append("primary result is NEGATIVE for candidate and legacy functional")
            elif suite == "v2_frozen" and ec == "secondary":
                grade = "MAIN_QUALITY" if parent != "loyfer_marker_knn" else "SUPPLEMENTARY_ONLY"
                why = ["frozen secondary, same rigour as primaries"] if grade == "MAIN_QUALITY" else ["marker provenance WEAK, low power (1,257 markers), not in Holm family"]
                if parent.startswith("compartment"):
                    why.append("E1 sign GC-oriented; circularity PARTIALLY_RELATED")
            elif suite == "v2_frozen" and ec in ("exploratory",):
                grade = "SUPPLEMENTARY_ONLY"
                why = ["declared EXPLORATORY in the frozen protocol" + ("; weak balance, 98.3% drop" if parent == "microc_H1_inter1Mb" else "; methylation-derived target, prevalence 0.805, GC/RT confounded")]
            elif suite == "v2_frozen" and ec == "sensitivity":
                grade = "MAIN_QUALITY"; why = ["frozen sensitivity of a primary (supplement-level)"]
            elif suite == "v2_sensitivity":
                grade = "MAIN_QUALITY"; why = ["preregistered sensitivity (registered before running), descriptive only, no Holm; supplement"]
            else:
                grade = "SUPPLEMENTARY_ONLY"; why = ["EXPLORATORY POST-HOC (Part B): no claim; absolute values lack CI" if name == "B3_profile_ridge" else "EXPLORATORY POST-HOC"]
            rec["problems"] = "; ".join(why) if grade != "MAIN_QUALITY" else ""
        elif suite == "legacy_bio_validation":
            sub = tid.split("/")[1]
            ov_ = ov.get(name, {})
            rec["split_strategy"] = h0.split_strategy
            rec["hyperparameter_selection"] = "LogisticRegressionCV / RidgeCV inner CV inside each training fold (no test labels); inner CV also random"
            rec["tuning_on_eval_labels"] = "no (inner CV on train fold)"
            rec["chromosome_matching"] = "no"; rec["context_matching"] = "no"; rec["probe_type_matching"] = "no"; rec["gc_tss_matching"] = "no"
            rec["negatives"] = h0.background_sampling
            rec["locus_leakage_checks"] = "NONE: random folds -> spatially/chromatin-correlated neighbouring CpGs in train and test"
            rec["representation_specific_advantages"] = ("regulatory_histone_dnase_v1 not evaluated; functional_annotations_pca here is the older PCA store; "
                                                         "dimension 64..1536 differs across arms; ntv3_pre errored for all probes")
            if sub == "known_set":
                fam = h0.biological_family
                rec["probe_kind"] = {"aging": "age/clock", "environmental exposure": "exposure (smoking/BMI)", "disease/pathology": "disease/EWAS"}[fam]
                rec["overlap_with_other_sets"] = (f"max overlap partner {ov_.get('partner')}: {ov_.get('shared')} CpGs "
                                                  f"({100*ov_.get('frac_of_self',0):.1f}% of this set; Jaccard {ov_.get('jaccard',0):.3f})") if ov_ else ""
                rec["duplicates"] = "no duplicate CpGs within set (verified); cross-set overlaps listed"
                rec["source_cohort_overlap_with_training"] = ("EWAS source cohorts not recorded per CpG; overlap with the TCGA training cohort NOT documented in the repo"
                                                              + ("; RA set includes GSE42861 (the benchmark's external disease cohort) by design (README)" if name == "ewas_catalog_rheumatoid_arthritis" else
                                                                 "; schizophrenia set mirrors GSE147221 by design (README)" if name == "ewas_catalog_schizophrenia" else ""))
                probs = ["random split with spatially correlated loci", "unmatched negatives (all other CpGs; unlisted != unassociated)",
                         "AUROC only, no AUPRC; prevalence %s" % rec["prevalence_in_probe_set"], "no CI",
                         "candidate regulatory_histone_dnase_v1 MISSING"]
                if h0.positive_n != "" and float(h0.positive_n) < 100:
                    probs.append("very few positives (%s)" % h0.positive_n)
                rec["problems"] = "; ".join(probs); grade = "PROBLEMATIC"
            elif sub == "context":
                rec["probe_kind"] = "genomic context (non-phenotype)"
                rec["problems"] = ("SOURCE_RETENTION for functional_annotations_pca (dense cols 0-3 are inputs); island is sequence-defined (partial for sequence arms); "
                                   "random split; unmatched; no CI; imbalance for shelf/shore with AUROC + threshold metrics")
                grade = "PROBLEMATIC"
            elif sub == "coef":
                is_p = name.startswith("phastcons")
                rec["probe_kind"] = "conservation regression" if is_p else "age/clock coefficient regression"
                key = name.replace("_coefficient_regression", "")
                o2 = ov.get(key, {})
                rec["overlap_with_other_sets"] = "" if is_p else (f"clock CpGs: max overlap partner {o2.get('partner')}: {o2.get('shared')} CpGs "
                                                                  f"({100*o2.get('frac_of_self',0):.1f}% of the clock set)")
                hr = g[(g.metric == "R2") & g.is_headline.astype(bool)]
                nneg = int((hr.score <= 0).sum())
                rec["problems"] = ("random KFold with spatially autocorrelated target; no CI" if is_p else
                                   "random KFold; tiny n (%s) -> R2 <= 0 for %d of %d arms; no CI; effectively uninformative" % (h0.n_items, nneg, len(hr))) + "; candidate MISSING"
                grade = "PROBLEMATIC"
            else:
                rec["probe_kind"] = "locality (unsupervised)"
                rec["problems"] = "unsupervised, 2,000 queries, no CI; locality expected by construction for track-based arms; candidate MISSING"
                grade = "SUPPLEMENTARY_ONLY"
        elif suite.startswith("encode_exp5") or suite == "encode_ewas_matched_single_draw":
            fam = h0.biological_family
            rec["probe_kind"] = {"aging": "age/clock", "environmental exposure": "exposure (smoking/BMI)", "disease/pathology": "disease/EWAS"}[fam]
            ov_ = ov.get(name, {})
            rec["negatives"] = "1:1 unlisted CpG from the same stratum (<=2,000 pairs); unlisted != unassociated; negatives may belong to other EWAS sets"
            rec["chromosome_matching"] = "yes (exact)"; rec["context_matching"] = "yes (island/shore/shelf/open sea; gene region; core-breadth decile)"
            rec["probe_type_matching"] = "yes (Infinium I/II)" if suite.endswith("probe") else "no (balance reported in probe_type_balance.csv)"
            rec["gc_tss_matching"] = "no (GC and TSS distance not matched; gene region + breadth decile only)"
            rec["hyperparameter_selection"] = "none: fixed LogisticRegression C=1, scaler fit on train folds only"
            rec["tuning_on_eval_labels"] = "no"
            rec["locus_leakage_checks"] = "GroupKFold by chromosome; pairs resampled within chromosome in the bootstrap; matching never uses embeddings"
            rec["overlap_with_other_sets"] = (f"max overlap partner {ov_.get('partner')}: {ov_.get('shared')} CpGs ({100*ov_.get('frac_of_self',0):.1f}% of this set)") if ov_ else ""
            rec["duplicates"] = "no duplicate CpGs within set (verified)"
            rec["source_cohort_overlap_with_training"] = ("EWAS source cohorts / TCGA overlap NOT documented; RA set includes GSE42861 (external disease cohort) by design"
                                                          if name == "ewas_catalog_rheumatoid_arthritis" else "EWAS source cohorts / TCGA overlap NOT documented")
            rec["representation_specific_advantages"] = ("matching strata (context, gene region, breadth) are inputs of the legacy functional store and context_only arm, "
                                                         "not of regulatory_histone_dnase_v1 or sequence arms; native dims 512/128 vs 256")
            probs = ["regulatory_histone_dnase_v1 (and any histone/DNase arm) NOT evaluated" if suite != "encode_ewas_matched_single_draw"
                     else "only campaign proxies (histone+context arm, compact CpGPT-256); native CpGPT-512 and DeepCpG missing"]
            if suite == "encode_ewas_matched_single_draw":
                probs += ["single matching draw, no CI"]
            if h0.positive_n != "" and float(h0.positive_n) < 150:
                probs.append("small matched set (%s pairs): wide CIs" % h0.positive_n)
            probs.append("AUPRC not computed" if suite != "encode_ewas_matched_single_draw" else "AUPRC on matched sample (baseline 0.5)")
            probs.append("GC / TSS distance unmatched")
            if name.endswith("clock"):
                probs.append("tiny positive set; clock CpGs selected by elastic net on blood/pan-tissue 450K data")
            rec["problems"] = "; ".join(probs)
            grade = "SUPPLEMENTARY_ONLY"
        elif suite == "encode_exp2_decodability":
            rec["probe_kind"] = "methylation program (patient-derived label)"
            rec["negatives"] = "n/a (regression)"
            rec["hyperparameter_selection"] = "ridge alpha by inner chromosome CV on train chromosomes"
            rec["tuning_on_eval_labels"] = "no"
            rec["locus_leakage_checks"] = "nested chromosome CV"
            rec["patient_derived_data_in_labels"] = "YES: target = mean beta over benchmark DISCOVERY (training) patients (label only)"
            rec["source_cohort_overlap_with_training"] = "target IS computed on the benchmark training cohort (TCGA discovery patients)"
            rec["representation_specific_advantages"] = "raw feature subsets and functional store contain track features correlated with methylation; 11,929 loci only"
            rec["problems"] = "regulatory_histone_dnase_v1 MISSING (only raw histone-only proxy); label derived from training patients; 11,929/408,399 loci"
            grade = "SUPPLEMENTARY_ONLY"
        elif suite == "encode_co_methylation":
            rec["probe_kind"] = "methylation program (held-out patients)"
            rec["negatives"] = "matched reference CpG per neighbour"
            rec["hyperparameter_selection"] = "none (no model)"
            rec["tuning_on_eval_labels"] = "no"
            rec["locus_leakage_checks"] = "held-out patients (independence recorded in json); far neighbours >=1 Mb or other chromosome"
            rec["patient_derived_data_in_labels"] = "yes at evaluation: held-out patients' betas define correlations (not inputs to representations)"
            rec["problems"] = "no CI; derived mean from per-pair CSV; only campaign arms (histone+context proxy, compact CpGPT-256); no DeepCpG, no native CpGPT-512"
            grade = "SUPPLEMENTARY_ONLY"
        elif suite == "umap_neighborhood":
            rec["probe_kind"] = "descriptive neighbourhood (context + 2 EWAS sets)"
            rec["negatives"] = "random fraction"
            rec["problems"] = "descriptive on 12,000-point sample; no CI; candidate MISSING; island/shore/shelf targets are SOURCE_RETENTION for legacy functional"
            grade = "PROBLEMATIC" if "island" in name or "shore" in name or "shelf" in name or "open_sea" in name else "SUPPLEMENTARY_ONLY"
        rec["rigor_grade"] = grade
        rec["rigor_reasons"] = "; ".join(why) if why else rec["problems"]
        out.append(rec)
    # rows for unscored tasks (NOT EVALUABLE etc.)
    seen = set()
    for u in unscored:
        key = (u["suite"], u["task"])
        if u["suite"].startswith("encode_exp5") and key not in seen:
            seen.add(key)
            out.append({"task_id": f"{u['suite']}/{u['task']}", "task_name": u["task"], "suite": u["suite"], "biological_family": SET_FAMILY[u["task"]][0],
                        "status": "discovery-campaign", "endpoint_class": "campaign_robustness", "representations_tested": "none (no scored result)",
                        "rigor_grade": "SUPPLEMENTARY_ONLY", "rigor_reasons": u["reason"], "problems": u["reason"],
                        "probe_kind": "disease/EWAS", "preregistered": "no", "ci_present": "no"})
    cols = ["task_id", "task_name", "suite", "probe_kind", "biological_family", "status", "endpoint_class", "preregistered", "representations_tested",
            "coverage_cand", "coverage_cpgpt", "coverage_deepcpg", "headline_metric", "positive_n", "negative_n", "n_items", "prevalence_in_probe_set",
            "auprc_reported", "set_raw_size", "set_fraction_in_universe", "negatives", "chromosome_matching", "context_matching", "probe_type_matching", "gc_tss_matching",
            "hyperparameter_selection", "tuning_on_eval_labels", "split_strategy", "locus_leakage_checks", "patient_derived_data_in_labels",
            "patient_derived_data_in_representation", "source_cohort_overlap_with_training", "overlap_with_other_sets", "duplicates",
            "representation_specific_advantages", "ci_present", "ci_definition", "circularity_vs_candidate", "circularity_vs_functional_legacy",
            "circularity_vs_sequence", "rigor_grade", "rigor_reasons", "problems"]
    return pd.DataFrame(out).reindex(columns=cols).fillna("")


# ----------------------------------------------------------------------------------------------- inventory
def dir_digest(rel: str, patterns=("*.json", "*.csv")) -> str:
    h = hashlib.sha256()
    for pat in patterns:
        for p in sorted((ROOT / rel).glob(pat)):
            h.update(p.name.encode())
            h.update(sha(str(p.relative_to(ROOT))).encode())
    return h.hexdigest()


def inventory() -> pd.DataFrame:
    def files(*rels):
        return ";".join(rels), ";".join(sha(r) for r in rels)

    items = []

    def add(eid, name, family, target, reps, metrics, status, in_reg, rels, notes):
        f, s = files(*rels) if rels else ("", "")
        items.append(dict(experiment_id=eid, name=name, family=family, target_source=target, representations_tested=reps,
                          metrics=metrics, status=status, in_registry=in_reg, files=f, sha256=s, notes=notes))

    add("v2_frozen", "bioval-v2 frozen endpoints (11 endpoint types x 4 arms)", "multiple (methylation program, 3D genome, regulatory activity, replication timing)",
        "Loyfer WGBS; 4DN Micro-C/Hi-C; FANTOM5; UW Repli-seq; Decato PMD", "regulatory_histone_dnase_v1; cpgpt_large_locus (512D); deepcpg_dna_locus (128D); functional_annotations_pca (legacy)",
        "Spearman; AUROC; R2; enrichment; delta-cosine", "frozen", "yes",
        [f"{V2}/report/summary_primary.csv", f"{V2}/report/summary_contrasts.csv"] + [f"{V2}/{a}/results.csv" for a in ["regulatory_histone_dnase_v1", "cpgpt_large_locus", "deepcpg_dna_locus", "functional_annotations_pca"]],
        "Registry rows read the per-endpoint JSONs (hashed per row) because they carry n_pos/n_neg; CSV summaries listed here for cross-check.")
    add("v2_sensitivity", "bioval-v2 registered sensitivities (12 endpoint files x 4 arms)", "same axes", "same targets", "same 4 arms",
        "Spearman; AUROC; R2", "preregistered", "yes", [f"{V2}/sensitivity/SUMMARY.csv"], "Sensitivity run at repo head 7d2153cf (primary at f0b00ac3).")
    add("v2_exploratory_posthoc", "bioval-v2 EXPLORATORY POST-HOC Loyfer (B3 ridge, similarity ranking, geometry)", "methylation program", "Loyfer WGBS",
        "same 4 arms", "R2; Pearson; Spearman", "exploratory", "yes",
        [f"{V2}/exploratory_posthoc_loyfer/ridge_summary_by_arm.csv", f"{V2}/exploratory_posthoc_loyfer/interpretation_ABC.json",
         f"{V2}/exploratory_posthoc_loyfer/geometry_within_arm_deltas.csv"],
        "geometry_within_arm_deltas.csv (within-arm geometry deltas) not row-ised: not an across-representation task result. loyfer_failure_audit is EXCLUDED and was not read.")
    add("legacy_bio_validation", "legacy bio_validation probes (8 arms)", "genomic locus identity; disease/pathology; aging; environmental exposure",
        "UCSC cpgIslandExt; EWAS Catalog/Atlas; Horvath/Hannum/PhenoAge; phastCons; genomic proximity",
        "cpgpt_locus; cpgpt_locus_large (512D); deepcpg_dna_locus (128D); deepcpg_dna_locus_hepg2; functional_annotations_pca (older PCA store); methylgpt_locus; methylgpt_locus_medium; ntv3_pre (all errored)",
        "AUROC (mean of folds); accuracy/F1/precision/recall; R2/Pearson/MAE; locality enrichment", "legacy-unregistered", "yes",
        [LEG_SRC.format(arm=a) for a in LEG_ARMS], "Random-fold probes, no CI; regulatory_histone_dnase_v1 never evaluated here.")
    add("umap_neighborhood", "UMAP biology neighbourhood metrics", "genomic locus identity; disease/pathology", "island/shore/shelf/open sea; colorectal + prostate EWAS Atlas",
        "functional_annotations_pca; deepcpg_dna_locus; cpgpt_locus_large", "neighbour-fraction enrichment", "legacy-unregistered", "yes",
        ["outputs/figures/umap_biology/neighborhood_metrics.csv", "outputs/figures/umap_biology/manifest.json"], "Side output of plot_umap_biology.py (figure folder); no CI.")
    add("encode_ewas_matched_single_draw", "ENCODE campaign matched EWAS/clock membership (single draw)", "disease/pathology; aging; environmental exposure",
        "EWAS Catalog/Atlas + clocks", "campaign arms: full; context_only; add/assay/Histone ChIP-seq; fm/cpgpt_locus_large", "AUROC; AUPRC (matched sample)", "discovery-campaign", "yes",
        [f"{ENC}/ewas_matched_membership.csv", f"{ENC}/ewas_protocol.json"], "3 sets lack matched support. No CI.")
    add("encode_exp5", "Follow-up exp5: repeated matched EWAS / clock probes (20 draws; base + probe-type variants)", "disease/pathology; aging; environmental exposure",
        "EWAS Catalog/Atlas + clocks", "functional_256 (campaign full); cpgpt_large_native512; deepcpg_hcc_native128; cpgpt_large_svd256; deepcpg_hcc_pad256; context_only_256",
        "AUROC (+ paired delta vs functional_256)", "discovery-campaign", "yes",
        [f"{ENC}/followup/exp5/summary_auc_delta.csv", f"{ENC}/followup/exp5/protocol.json", f"{ENC}/followup/exp5/evaluable_sets.csv",
         f"{ENC}/followup/exp5/matching_summary_per_seed.csv", f"{ENC}/followup/exp5/probe_type_balance.csv"],
        "oof_predictions.parquet (47 MB) NOT read (AUPRC could be recomputed from it; not done). regulatory_histone_dnase_v1 absent.")
    add("encode_exp2", "Follow-up exp2: decodability of training-patient mean beta", "methylation program", "mean beta over TCGA discovery patients (11,929 loci)",
        "Functional-256/128; CpGPT-large native-512/compact-256; DeepCpG native-128/pad-256; context-only; raw feature-family subsets", "R2; Pearson; Spearman; MSE", "discovery-campaign", "yes",
        [f"{ENC}/followup/exp2/summary.csv", f"{ENC}/followup/exp2/paired_vs_functional256.csv"], "Label derived from benchmark training patients.")
    add("encode_co_methylation", "Co-methylation far-neighbour test (confirmation stage)", "methylation program", "held-out patient beta correlations",
        "campaign arms: full; context_only; add/assay/Histone; fm/cpgpt_locus_large", "mean neighbour vs matched correlation", "discovery-campaign", "yes",
        [f"{ENC}/co_methylation/confirm/{s}.csv" for s in ["full_a18b869b", "context_only_2805a394", "add_assay_histone_chip_seq_2d418bee", "fm_cpgpt_locus_large_d3474a48"]],
        "Means derived by registry script from per-pair CSVs; no CI in source.")
    add("encode_biological_probes", "ENCODE campaign feature-group probes of mean/variance beta (3,988 rows)", "methylation program", "mean/variance beta over discovery patients",
        "feature groups (target/*, assay/*, dense/*), not embeddings", "MSE; R2; Pearson", "discovery-campaign", "no", [f"{ENC}/biological_probes.csv"],
        "Feature-attribution table; arms are raw feature subsets, not panel representations. Inventory only.")
    items[-1]["files"] = f"{ENC}/biological_probes.csv"
    add("encode_tissue", "ENCODE campaign per-cancer tissue-association probes (TCGA projects)", "disease/pathology", "TCGA per-cancer mean beta minus global mean (discovery)",
        "feature groups (organ-matched vs prevalence-matched), not embeddings", "MSE; R2; Pearson", "discovery-campaign", "no", [],
        f"60 files under {ENC}/tissue (TCGA label from training patients); directory digest (json+csv) = {dir_digest(ENC + '/tissue')}. Inventory only.")
    add("encode_exp3", "Follow-up exp3: ridge / 'proxy null' for feature subsets", "methylation program", "mean/variance beta", "feature subsets (real vs blind-matched)",
        "relative MSE delta %", "discovery-campaign", "no", [f"{ENC}/followup/exp3/tables/proxy_null_summary.csv", f"{ENC}/followup/exp3/tables/proxy_ridge_null.csv"],
        "Tests whether histone-vs-non-histone proxy gains exceed random feature subsets; not a representation panel comparison. Inventory only.")
    add("encode_exp4", "Follow-up exp4: chromosome-transfer (OOD) reconstruction", "other", "patient reconstruction on chr20-22", "campaign arms", "MSE / skill", "discovery-campaign", "no",
        [f"{ENC}/followup/exp4/unseen_absolute_recomputed.csv", f"{ENC}/followup/exp4/unseen_paired_recomputed.csv"], "Reconstruction of patient methylation, not a locus-property task. Excluded.")
    add("excluded_patient_level", "Patient-level downstream disease/age tasks (GSE42861/GSE147221/GSE40279; sparse_reconstruction; external_reconstruction_v1)", "other",
        "patient labels", "panel arms via reconstruction", "patient-level metrics", "n/a", "no", [], "Not locus-level (patient information is in labels by design). Excluded by scope.")
    add("excluded_reconstruction", "Masking benchmark / regulatory family screen / confirmation / selection / upscaling", "other", "methylation reconstruction", "all arms", "MSE",
        "n/a", "no", [], "Reconstruction benchmark, not locus biology; grep of outputs found no AUROC/AUPRC columns outside v2 and ewas_matched_membership.csv.")
    add("excluded_loyfer_failure_audit", "Loyfer primary failure audit (exploratory, in progress)", "methylation program", "Loyfer", "4 arms", "n/a", "exploratory", "no", [],
        "EXCLUDED ENTIRELY by instruction: directory not read, not hashed, not in registry.")
    return pd.DataFrame(items)


# ----------------------------------------------------------------------------------------------- main
def git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)
    reg = Reg()
    info = load_sets()
    ov = overlaps(info["sets"])
    build_v2(reg)
    build_v2_contrasts(reg)
    build_legacy(reg, info)
    build_ewas_matched(reg, info)
    build_exp5(reg, info)
    build_exp2(reg)
    build_comethylation(reg)
    build_umap(reg)

    df = pd.DataFrame(reg.rows, columns=REG_COLUMNS)
    df = df.sort_values(["suite", "task_id", "stat_name", "representation_raw", "metric"], kind="mergesort").reset_index(drop=True)
    df["row_id"] = np.arange(1, len(df) + 1)
    cov = coverage_table(df.to_dict("records"))
    con = pd.DataFrame(reg.contrasts, columns=CONTRAST_COLUMNS).sort_values(["suite", "task_id", "stat_name", "comparator_raw"], kind="mergesort").reset_index(drop=True)
    con["contrast_id"] = np.arange(1, len(con) + 1)
    audit = audit_rows(df, info, ov, cov, reg.unscored)
    matrix = build_matrix(df, cov, con, audit)

    df.to_csv(OUT / "bio_task_registry.csv", index=False, float_format="%.10g")
    con.to_csv(OUT / "bio_task_paired_contrasts.csv", index=False, float_format="%.10g")
    cov.to_csv(OUT / "bio_task_coverage_matrix.csv", index=False)
    audit.to_csv(OUT / "per_task_audit.csv", index=False)
    matrix.to_csv(OUT / "bio_task_matrix.csv", index=False, float_format="%.10g")
    inv = inventory()
    inv.to_csv(OUT / "bio_task_inventory.csv", index=False)
    pd.DataFrame(reg.unscored).drop_duplicates().sort_values(list(pd.DataFrame(reg.unscored).columns)).to_csv(OUT / "unscored_tasks.csv", index=False)

    # input manifest: every provenance file with sha256
    prov = sorted(set(df.source_file) | {p for s in df.aux_source_files for p in str(s).split(";") if p} | set(con.source_file))
    prov = [p for p in prov if (ROOT / p).exists()]
    man = {"repo_head_at_build": git_head(), "protocol_tag": V2_TAG, "n_provenance_files": len(prov),
           "files": {p: sha(p) for p in prov},
           "known_sets_n_positives_in_universe": {k: len(v) for k, v in sorted(info["sets"].items())},
           "known_sets_raw_len": info["raw_len"], "known_sets_unique": info["raw_uniq"],
           "set_overlap_max_partner": ov, "context_counts_in_universe": info["ctx"],
           "excluded_directory_not_read": "outputs/biological_validation_v2/**/loyfer_failure_audit"}
    with open(OUT / "manifests" / "input_files_sha256.json", "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True, default=str)
    write_schema(df, cov, matrix, audit, con)
    print(f"registry rows={len(df)} contrasts={len(con)} matrix_rows={len(matrix)} audit_rows={len(audit)} "
          f"coverage_rows={len(cov)} inventory={len(inv)} unscored={len(reg.unscored)} provenance_files={len(prov)}")
    return 0


DICT = {
    "row_id": "1-based row index (deterministic sort)", "suite": "result family (v2_frozen, v2_sensitivity, v2_exploratory_posthoc, legacy_bio_validation, umap_neighborhood, encode_ewas_matched_single_draw, encode_exp5_base, encode_exp5_probe, encode_exp2_decodability, encode_co_methylation)",
    "task_id": "suite/endpoint; unique task key used by the matrix, audit and coverage files", "task_name": "endpoint / set name as in the results file",
    "stat_name": "exact statistic name in the source file (or derived name)", "stat_class": "class attached to the stat in the source (primary/secondary/sensitivity/descriptive/...)",
    "is_headline": "true for the stat that represents the task in bio_task_matrix.csv (primary_stat or the AUROC/mean-AUC; r2 companions)",
    "biological_family": "vocabulary: " + " | ".join(FAMILIES), "target_source": "provenance of the target labels", "target_description": "what is predicted/retrieved",
    "positive_n": "number of positives (blank when not applicable/unknown)", "negative_n": "number of negatives/controls", "n_items": "items entering the statistic (as recorded)",
    "background_sampling": "how negatives/background were obtained", "matching_covariates": "covariates matched between positives and negatives",
    "split_strategy": "CV / blocking scheme", "representation_raw": "arm id exactly as in the results file", "representation_id": "normalized id: regulatory_histone_dnase_v1 | cpgpt_large_locus | deepcpg_dna_locus | functional_annotations_pca [legacy] | other:*",
    "representation_variant": "native 512D vs compact 256D, store variant, etc.", "representation_dim": "embedding width (blank for raw feature subsets)",
    "is_main_panel": "true only for the three main-panel arms in their native form (candidate 256D SVD, CpGPT-large native 512D, DeepCpG native 128D)",
    "proxy_for": "main-panel arm this non-panel variant approximates (blank if none)", "classifier_probe": "model, regularisation and tuning scheme",
    "metric": "vocabulary: " + " | ".join(METRICS), "score": "metric value as stored (never imputed; blank = undefined)", "ci_lo": "lower CI bound", "ci_hi": "upper CI bound",
    "ci_definition": "mandatory when a CI is present", "seed": "seed(s) recorded", "source_file": "repo-relative result file", "source_sha256": "sha256 of source_file at build time",
    "aux_source_files": "other files consulted (semicolon-separated)", "git_commit_recorded": "repo head recorded by the run manifest (v2 only)", "protocol_tag": "frozen protocol git tag if any",
    "status": "vocabulary: " + " | ".join(STATUSES), "endpoint_class": "class in source protocol (primary/secondary/sensitivity/exploratory/campaign_*/legacy_*)",
    "circularity_level": "circularity applicable to THIS row's representation: " + " | ".join(CIRC),
    "circularity_vs_candidate": "w.r.t. regulatory_histone_dnase_v1 inputs (histone + DNase tracks)", "circularity_vs_functional_legacy": "w.r.t. functional_annotations_pca inputs (tracks + 23 dense columns)",
    "circularity_vs_sequence": "w.r.t. sequence foundation-model inputs (reference sequence)", "notes": "caveats specific to the row",
}


def write_schema(df, cov, matrix, audit, con) -> None:
    schema = {
        "registry_file": "bio_task_registry.csv", "format": "long: one row per task x representation x metric (x stat)",
        "columns": {c: DICT.get(c, "") for c in REG_COLUMNS},
        "enumerations": {"biological_family": FAMILIES, "status": STATUSES, "circularity_level": CIRC, "circularity_vs_candidate": CIRC,
                         "circularity_vs_functional_legacy": CIRC, "circularity_vs_sequence": CIRC, "metric": METRICS,
                         "rigor_grade": GRADES, "coverage": COVERAGE, "main_panel": MAIN_PANEL},
        "family_mapping_rules": {"genomic locus identity": "CpG context classes, conservation, genomic proximity",
                                 "regulatory activity": "FANTOM5 enhancer membership/activity", "disease/pathology": "EWAS disease/cancer sets",
                                 "aging": "age EWAS and clock CpG sets", "environmental exposure": "smoking, BMI", "3D genome": "Micro-C contacts, compartments",
                                 "replication timing": "RT consensus, PMD", "methylation program": "Loyfer cell-type profiles/markers, mean beta, co-methylation",
                                 "other": "unused"},
        "files": {"bio_task_registry.csv": f"{len(df)} rows", "bio_task_paired_contrasts.csv": f"{len(con)} rows (paired deltas as published; never constructed here)",
                  "bio_task_coverage_matrix.csv": f"{len(cov)} rows task x main-panel coverage (HAVE/PROXY_VARIANT_ONLY/MISSING)",
                  "bio_task_matrix.csv": f"{len(matrix)} rows (headline stat per task; wide)",
                  "per_task_audit.csv": f"{len(audit)} rows fairness audit + rigor grade", "bio_task_inventory.csv": "experiment-level inventory",
                  "unscored_tasks.csv": "probes that errored / lacked support (no score)", "manifests/input_files_sha256.json": "sha256 of every provenance file"},
        "matrix_columns": {"{arm}__{metric}[__ci_lo|__ci_hi]": "arm in cand|cpgpt|deepcpg|funcleg; metric in auroc|auprc|spearman|r2|enrichment; AUPRC has no CI",
                           "{metric}__ci_definition": "CI definition for that metric's intervals",
                           "coverage_{cand|cpgpt|deepcpg}": "HAVE | PROXY_VARIANT_ONLY | MISSING", "proxy_arm_* / proxy_metric_* / proxy_value_*": "proxy variant used when the main arm is missing (headline metric only)",
                           "candidate_minus_{cpgpt|deepcpg|funcleg}__{metric}[__ci_lo|__ci_hi|__ci_definition]": "paired delta exactly as published (blank: none)",
                           "delta_flag": "'no paired CI available' when no published paired delta exists for the task"},
        "conventions": {"circularity_level": "per-row, for the row's own representation", "scores": "never invented; blanks where undefined or not computed",
                        "legacy_AUC": "legacy bio_validation AUROC is the mean of fold AUROCs", "legacy_functional_variants": "functional_annotations_pca appears in two store variants (campaign full SVD256 vs older PCA store)"},
        "excluded": "outputs/biological_validation_v2/**/loyfer_failure_audit (exploratory, in progress) not read",
    }
    with open(OUT / "bio_task_registry.json", "w") as fh:
        json.dump(schema, fh, indent=1, sort_keys=False)


if __name__ == "__main__":
    sys.exit(main())
