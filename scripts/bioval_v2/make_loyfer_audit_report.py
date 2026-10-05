# ruff: noqa: C410, PLC3002
"""Generate docs/LOYFER_PRIMARY_FAILURE_AUDIT.md MECHANICALLY from the audit outputs (read-only on every result file).

Every number in the report is read from a JSON/CSV/NPZ under outputs/biological_validation_v2/loyfer_failure_audit/ (or the registered
decision flags in report_inputs/REPORT_INPUTS.json); the only computations are formatting, sorting/ranking and paired bootstrap contrasts
from the STORED replicates. A manifest of the sha256 of every source file is written next to the report.
  python scripts/bioval_v2/make_loyfer_audit_report.py            # write the report + sources manifest
  python scripts/bioval_v2/make_loyfer_audit_report.py --check    # verify that the files on disk equal a fresh generation and the sources are unchanged
No frozen file is read or written (the frozen results are only referenced by the integrity snapshot).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import contrast

OUT = ROOT / "outputs/biological_validation_v2/loyfer_failure_audit"
DOC = ROOT / "docs/LOYFER_PRIMARY_FAILURE_AUDIT.md"
MANIFEST = ROOT / "docs/LOYFER_PRIMARY_FAILURE_AUDIT_sources.json"
ARMS = ("regulatory_histone_dnase_v1", "functional_annotations_pca", "cpgpt_large_locus", "deepcpg_dna_locus")
CAND, COMPS = ARMS[0], ARMS[2:]
LABEL = {"regulatory_histone_dnase_v1": "Histone+DNase (candidate)", "functional_annotations_pca": "functional PCA (legacy control)",
         "cpgpt_large_locus": "CpGPT-large", "deepcpg_dna_locus": "DeepCpG-DNA"}
STRATA = ("<1kb", "1-10kb", "10-100kb", "100kb-1Mb", ">1Mb", "interchromosomal")
ORIG_TAG = "g0.01_0.1_1_10_100_1000_10000"
SOURCES: set[Path] = set()


def J(rel):
    p = OUT / rel
    SOURCES.add(p)
    return json.loads(p.read_text())


def NPZ(rel):
    p = OUT / rel
    SOURCES.add(p)
    return np.load(p)


def tag(grid):
    return "g" + "_".join(f"{g:g}" for g in grid)


# ------------------------------------------------------------------------------------------------ formatting
def na(x):
    return x is None or (isinstance(x, float) and not math.isfinite(x))


def f(x, d=4, sign=False):
    if na(x):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def e(x):
    return "n/a" if na(x) else f"{x:.3g}"


def ci(s, d=4):
    if s is None:
        return "n/a"
    lo, hi = s.get("ci_lo"), s.get("ci_hi")
    v = f(s["value"], d, True)
    return v if na(lo) else f"{v} [{f(lo, d, True)}, {f(hi, d, True)}]"


def v(s, d=4):
    return "n/a" if s is None else f(s["value"], d, True)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def med(d, key="median", dd=3):
    return "n/a" if d is None else f"{d[key]:.{dd}f}"


def medq(x, dd=3):
    return "n/a" if x is None else f"{x['median']:.{dd}f} [{x['q025']:.{dd}f}, {x['q975']:.{dd}f}]"


def ordering(vals: dict, rev=True):
    return " > ".join(LABEL[k] for k, _ in sorted(vals.items(), key=lambda kv: -kv[1]))


# ------------------------------------------------------------------------------------------------ report
def build() -> str:
    SOURCES.clear()
    L = []
    P = L.append
    ver = J("step1_target_audit/VERDICT.json")
    ver0 = J("step1_target_audit_VERDICT_before_bugfix1.json")
    att1 = J("step1_target_audit_attempt1_FAIL/VERDICT.json")
    att1r = J("step1_target_audit_attempt1_FAIL/target_audit_results.json")
    res1 = J("step1_target_audit/target_audit_results.json")
    diag = J("step1_diag_f16_mismatch/diagnosis_attempt1.json")
    rel = J("step1_target_audit/reliability.json")
    sem = J("step3_4_semantics_pairs/semantics_pairs.json")
    bb = J("step5_baselines/baselines.json")
    base = {a: J(f"step5_baselines/arm_{a}.json") for a in ARMS}
    rv = J("step6_raw_vs_svd/raw_vs_svd.json")
    svd = {a: J(f"step7_svd_geometry/arm_{a}.json") for a in ARMS}
    ph = J("step9_probe_fairness/grid_history.json")
    final_grid = tuple(ph["final_grid"])
    oof = {a: J(f"step8_oof_similarity/arm_{a}_{ORIG_TAG}.json") for a in ARMS}
    ext = {a: J(f"step9_probe_fairness/arm_{a}_{tag(final_grid)}.json") for a in ARMS}
    dec = J("report_inputs/REPORT_INPUTS.json")
    cmp_ = J("integrity/COMPARE.json")
    man_after = J("integrity/SNAPSHOT_AFTER.json")
    man_before = J("integrity/SNAPSHOT_BEFORE.json")
    pc_manifest = {s: J(f"{s}/run_manifest.json") for s in ("step1_target_audit", "step3_4_semantics_pairs", "step5_baselines", "step6_raw_vs_svd",
                                                           "step7_svd_geometry", "step8_oof_similarity", "step9_probe_fairness", "report_inputs")}
    flags = dec["decision"]["flags"]
    classes = dec["decision"]["classes"]
    native = {a: base[a]["frozen_primary"] for a in ARMS}
    nullhi = {a: base[a]["null_hi"] for a in ARMS}
    # consistency assertions (the narrative below states these facts)
    assert ver["verdict"] == "PASS_LEVEL_A" and ver["raw_level_B_executed"] is False and att1["verdict"] == "FAIL"
    assert native[CAND] < 0 and native[COMPS[0]] > 0 and native[COMPS[1]] > 0
    assert cmp_["identical"] and cmp_["primary_fingerprint_unchanged"]
    # the narrative states these facts; the generator refuses to run if the data contradict them
    assert not flags["BUG"]["flag"] and not flags["TARGET_LIMITATION"]["flag"] and not flags["INFORMATION_ABSENCE"]["flag"] and not flags["COMPARATOR_ADVANTAGE"]["flag"]
    assert flags["COMPRESSION_GEOMETRY"]["flag"] and all(k.startswith("oof") for k in flags["COMPRESSION_GEOMETRY"]["sources_recovering"])
    assert all(x["level"] != "STRONG" for x in flags["COMPARATOR_ADVANTAGE"]["per_comparator"].values())
    _rv = rv["scores"]
    assert _rv["raw_binary_cosine"]["pooled_gt1Mb_inter_eqw"]["value"] < _rv["embedding256_cosine_native"]["pooled_gt1Mb_inter_eqw"]["value"]
    assert _rv["raw_jaccard"]["pooled_gt1Mb_inter_eqw"]["value"] < _rv["embedding256_cosine_native"]["pooled_gt1Mb_inter_eqw"]["value"]
    assert svd[CAND]["curve"]["centered_cosine"]["pooled_gt1Mb_inter_eqw"]["value"] > native[CAND]

    P("# Loyfer primary failure audit: REPORT (EXPLORATORY POST-HOC)\n")
    P("Generated MECHANICALLY by `scripts/bioval_v2/make_loyfer_audit_report.py` from the files under "
      "`outputs/biological_validation_v2/loyfer_failure_audit/` (sources and sha256 in `docs/LOYFER_PRIMARY_FAILURE_AUDIT_sources.json`; "
      "`--check` verifies them). Registration: `docs/LOYFER_PRIMARY_FAILURE_AUDIT_REGISTRATION.md` (Amendments 1-2, Bugfix log). Classification of section 11 = output of "
      "`decision.py` (`report_inputs/REPORT_INPUTS.json`), applied without ad hoc interpretation. Descriptive: raw chromosome-block bootstrap CIs "
      "(22 blocks of `cpg_i`, B = 1000, seed 17), no Holm, no claim. It does not replace or amend the frozen primary. 'n/a' = not available in the files.\n")

    P("## 0. Caveats that apply to everything below\n")
    P(f"1. **Level A only.** Verdict `{ver['verdict']}` (level {ver['level']}, raw_level_B_executed = {ver['raw_level_B_executed']}). It excludes bugs downstream of the per-sample counts "
      "(donor/group aggregation, masks, float16 storage, coordinates between artifacts, pairing, Pearson) but does NOT verify the per-sample counts nor the `.pat` parsing / CpG-index "
      "mapping against the raw files (Level B not executed, user decision D-L1). Any 'no bug' statement carries this limit.")
    P(f"2. **Attempt 1 of step 1 was FAIL** (`step1_target_audit_attempt1_FAIL/`, failed check: {att1['failed_checks']}): the a-priori threshold of 1e-5 on the float16 rounding-mismatch fraction was not met "
      f"(observed {e(att1r['group_beta_vs_frozen_float16']['frac_f16_rounding_mismatch'])}, {att1r['group_beta_vs_frozen_float16']['n_f16_rounding_mismatch']} of "
      f"{att1r['group_beta_vs_frozen_float16']['n_cells']} cells). `PASS_LEVEL_A` exists only thanks to **Amendment 2** (mechanism-based rule, see section 1). "
      f"A verdict computed before Bugfix 1 is kept in `step1_target_audit_VERDICT_before_bugfix1.json` (verdict {ver0['verdict']}; it was bound to the earlier registration hash).")
    P("3. **Bugfix 1** (`report-inputs` aggregation of step 5 from the per-arm files) changed no result.")
    P("4. Exploratory post hoc: conceived after the primary; thresholds fixed before any result; nothing here selects a new primary or a new representation; "
      "linear decodability is not alignment with a 'methylation program' and pretraining effects are not excluded (section 13).\n")

    # ------------------------------------------------------------------ summary + key table
    P("## Summary in plain language\n")
    pooled = {a: base[a]["native"]["pooled_gt1Mb_inter_eqw"] for a in ARMS}
    cen = svd[CAND]["curve"]
    oof1 = oof[CAND]["oof_similarity"]
    t1 = oof1["raw"]["all_complete_case"]["pooled_gt1Mb_inter_eqw"]
    t2 = oof1["centered"]["all_complete_case"]["pooled_gt1Mb_inter_eqw"]
    nat_same = oof1["native_cosine_same_pair_set"]
    cov = bb["covariate_baselines"]
    cov_pool = {k: x["pooled_gt1Mb_inter_eqw"]["value"] for k, x in cov.items() if x["pooled_gt1Mb_inter_eqw"]}
    P(f"* **What the registered rules return**: classes = {classes} (BUG {flags['BUG']['flag']}, TARGET_LIMITATION {flags['TARGET_LIMITATION']['flag']}, "
      f"COMPRESSION_GEOMETRY {flags['COMPRESSION_GEOMETRY']['flag']}, INFORMATION_ABSENCE {flags['INFORMATION_ABSENCE']['flag']}, COMPARATOR_ADVANTAGE {flags['COMPARATOR_ADVANTAGE']['flag']}).")
    P(f"* **The target is not the problem by the registered criteria**: no Level-A bug (with the Level-A limit), reliability-bounded ceiling for the primary "
      f"{f(flags['TARGET_LIMITATION']['evidence']['ceiling_primary'], 3)} (threshold 0.30), reliability in >1Mb / interchromosomal {f(flags['TARGET_LIMITATION']['evidence']['rel_gt1Mb'], 3)} / "
      f"{f(flags['TARGET_LIMITATION']['evidence']['rel_inter'], 3)} (threshold 0.50), the top 1% similarity band contains {f(flags['TARGET_LIMITATION']['evidence']['topband']['share_top'], 3)} near-constant/small-amplitude pairs.")
    rs_ = rel["donors_ge2"]["summary"]
    P(f"* **Reliability depends strongly on methylation variance** (not part of the registered rules): Spearman-Brown rho / ceiling by variance quartile of the pair (min variance) "
      f"Q1 {f(rs_['V_Q1']['rho_sb']['median'], 3)} / {f(rs_['V_Q1']['ceiling']['median'], 3)}, Q2 {f(rs_['V_Q2']['rho_sb']['median'], 3)} / {f(rs_['V_Q2']['ceiling']['median'], 3)}, "
      f"Q3 {f(rs_['V_Q3']['rho_sb']['median'], 3)} / {f(rs_['V_Q3']['ceiling']['median'], 3)}, Q4 {f(rs_['V_Q4']['rho_sb']['median'], 3)} / {f(rs_['V_Q4']['ceiling']['median'], 3)}, "
      f"top 10% {f(rs_['V_top10pct']['rho_sb']['median'], 3)} / {f(rs_['V_top10pct']['ceiling']['median'], 3)}; the lowest-variance quartile is close to noise.")
    P(f"* **Why the primary is {f(native[CAND], 3, True)} while the linear probe reads out the profile**: on the same frozen pairs, the cosine of the candidate's 256-d embedding is "
      f"{f(nat_same['value'], 3, True)} (complete-case pairs), whereas the Pearson similarity of the chromosome-held-out ridge-predicted 39-group profiles of the SAME embedding is "
      f"{f(t1['value'], 3, True)} (T1) / {f(t2['value'], 3, True)} (T2 centred); the registered rule COMPRESSION_GEOMETRY is satisfied through these OOF sources "
      f"(delta vs native {f(oof1['raw']['delta_vs_native_same_pairs']['delta'], 3, True)}, CI [{f(oof1['raw']['delta_vs_native_same_pairs']['ci_lo'], 3)}, {f(oof1['raw']['delta_vs_native_same_pairs']['ci_hi'], 3)}]). "
      f"The linear probe of the candidate has centred-target R2 (variable CpGs) {f(oof[CAND]['r2_global']['centered__variable_cpgs'], 3)}. "
      f"The raw 2,492-track space does NOT recover the target either (raw cosine {f(rv['scores']['raw_binary_cosine']['pooled_gt1Mb_inter_eqw']['value'], 3, True)}, Jaccard "
      f"{f(rv['scores']['raw_jaccard']['pooled_gt1Mb_inter_eqw']['value'], 3, True)}, both below the native 256-d {f(rv['scores']['embedding256_cosine_native']['pooled_gt1Mb_inter_eqw']['value'], 3, True)}), "
      "so the registered raw-space metrics show no recovery that the SVD would then have removed.")
    o_t1 = {a: oof[a]["oof_similarity"]["raw"]["all_complete_case"]["pooled_gt1Mb_inter_eqw"]["value"] for a in ARMS}
    r2c = {a: oof[a]["r2_global"]["centered__variable_cpgs"] for a in ARMS}
    assert o_t1[CAND] > max(o_t1[c] for c in COMPS) and r2c[CAND] > max(r2c[c] for c in COMPS)
    P(f"* **The arm ordering reverses under the linear read-out**: OOF-profile similarity (T1) {ordering(o_t1)} ({', '.join(f'{LABEL[a]} {f(o_t1[a], 3, True)}' for a in ARMS)}); "
      f"ridge R2 (centred target, variable CpGs) {ordering(r2c)} ({', '.join(f'{LABEL[a]} {f(r2c[a], 3)}' for a in ARMS)}); the native-cosine primary orders them {ordering(native)}.")
    P(f"* **Geometry operations change the primary-style number a lot for the candidate** (centred cosine {f(cen['centered_cosine']['pooled_gt1Mb_inter_eqw']['value'], 3, True)}, negative Euclidean "
      f"{f(cen['neg_euclidean']['pooled_gt1Mb_inter_eqw']['value'], 3, True)}, PC1 removed {f(cen['remove_pc1']['pooled_gt1Mb_inter_eqw']['value'], 3, True)}); PC1 explains "
      f"{f(svd[CAND]['explained_variance_ratio_first20'][0], 3)} of the embedding variance and is associated with the number of overlapped tracks (Spearman {f(svd[CAND]['assoc_pc_scores_spearman']['peak_total'][0], 3, True)}).")
    P(f"* **Is {f(native[COMPS[0]], 4)} (CpGPT) strong?** It is of the same size as simple covariate baselines on the same pairs (-|dM| {f(cov_pool['neg_abs_delta_methylation_mean'], 4)}, "
      f"-|d CpG density| {f(cov_pool['neg_abs_delta_cpg_density'], 3)}) and below the difference-in-peak-count baseline ({f(cov_pool['neg_abs_delta_peak_count_total'], 3)}); null controls are within +-{f(max(nullhi.values()), 3)}.")
    P(f"* **Comparator advantage is not 'STRONG'**: both sequence comparators lead on the native cosine but the lead disappears under most other registered fair evaluations "
      f"(CpGPT: {flags['COMPARATOR_ADVANTAGE']['per_comparator'][COMPS[0]]['level']}, DeepCpG: {flags['COMPARATOR_ADVANTAGE']['per_comparator'][COMPS[1]]['level']}); the pretraining provenance (section 10) is not excluded as a factor.\n")

    P("### Key numbers (primary-style statistic = weighted Spearman pooled over >1Mb + interchromosomal, equal stratum weight)\n")
    kt = [("Reliability ceiling of the primary (sqrt of Spearman-Brown rho), donors>=2 [q2.5, q97.5 over 20 splits]", medq(rel["donors_ge2"]["summary"]["pooled_gt1Mb_inter_eqw"]["ceiling"])),
          ("same, groups with >=4 donors", medq(rel["donors_ge4"]["summary"]["pooled_gt1Mb_inter_eqw"]["ceiling"])),
          ("Spearman-Brown reliability >1Mb / interchromosomal", f"{medq(rel['donors_ge2']['summary']['dist_>1Mb']['rho_sb'])} / {medq(rel['donors_ge2']['summary']['dist_interchromosomal']['rho_sb'])}")]
    for a in ARMS:
        kt.append((f"Native cosine primary, {LABEL[a]}", ci(pooled[a])))
    for a in ARMS:
        kt.append((f"NULL_HI (97.5% of 200 nulls), {LABEL[a]}", f(nullhi[a], 4)))
    for k in ("neg_abs_delta_peak_count_total", "neg_abs_delta_methylation_mean", "neg_abs_delta_cpg_density", "chromosome_only"):
        kt.append((f"Baseline {k}", ci(cov[k]["pooled_gt1Mb_inter_eqw"])))
    for k, nm in (("raw_binary_cosine", "Raw binary cosine (2,492 tracks)"), ("raw_jaccard", "Raw Jaccard"), ("embedding256_cosine_native", "Native 256-d cosine (same set)")):
        kt.append((nm, ci(rv["scores"][k]["pooled_gt1Mb_inter_eqw"])))
    kt += [("OOF-profile similarity T1 raw, candidate (complete-case pairs)", ci(t1)), ("OOF-profile similarity T2 centred, candidate", ci(t2)),
           ("Native cosine on the same complete-case pairs, candidate", ci(nat_same))]
    for n_ in ("centered_cosine", "remove_pc1", "remove_pc1_2", "remove_pc1_5", "remove_pc1_10", "standardized_cosine", "whitened_cosine", "neg_euclidean"):
        kt.append((f"Candidate geometry: {n_}", ci(cen[n_]["pooled_gt1Mb_inter_eqw"])))
    for a in ARMS:
        kt.append((f"Ridge R2 centred, variable CpGs: original grid / extended grid, {LABEL[a]}",
                   f"{f(oof[a]['r2_global']['centered__variable_cpgs'], 4)} / {f(ext[a]['r2_global']['centered__variable_cpgs'], 4)}"))
    P(table(["Quantity", "Value [95% CI]"], kt))
    P("")

    # ------------------------------------------------------------------ 1
    P("## 1. Target audit (section 1; Level A)\n")
    P(f"Verdict `{ver['verdict']}`, {ver['n_hard_checks']} hard checks, failed: {ver['failed_checks'] or 'none'}; registration sha256 `{ver['registration_sha256'][:16]}...`, code sha256 `{ver['code_sha256'][:16]}...`, HEAD `{ver['head_commit'][:10]}`, amendment `{ver['amendment']}`.\n")
    P(table(["Hard check", "Result"], [(k, "PASS" if x is True else ("FAIL" if x is False else "info")) for k, x in ver["checks"].items()]))
    mp = res1["mapping_vs_published_S1"]
    P(f"\n* **Sample-to-group mapping** vs published Table S1 (re-joined by Z-suffix): {mp['n_used']} used samples, {mp['n_s1']} S1 rows, group mismatches {len(mp['group_mismatch'])}, patient mismatches {len(mp['patient_mismatch'])}, unmatched {len(mp['unmatched_in_s1'])}, S1-not-used {len(mp['s1_not_in_used'])}.")
    pg = res1["per_group_counts"]
    P(f"* **Samples/donors per group**: 39 groups, {sum(r['n_samples'] for r in pg)} samples, per-group donors sum {sum(r['n_donors'] for r in pg)}; groups with N = 1: {', '.join(res1['groups_with_N_samples_1'])}; groups with >= 2 donors {sum(1 for r in pg if r['n_donors'] >= 2)}, >= 4 donors {sum(1 for r in pg if r['n_donors'] >= 4)}.")
    P("\n" + table(["Group", "Samples", "Donors"], [(r["group"], r["n_samples"], r["n_donors"]) for r in pg]))
    exc = res1["excluded_samples"]
    P(f"\n* **Excluded samples**: {len(exc)} ({', '.join(sorted({r['decision_type'] for r in exc}))}); e.g. {exc[0]['file_name']}.")
    stt = res1["structural"]
    P(f"* **Coordinates / duplicates / range**: pairs {stt['pairs_n']}, cpg_i<cpg_j {stt['pairs_cpg_i_lt_cpg_j']}, duplicate pairs {stt['pairs_duplicates']}, endpoints in universe {stt['pairs_endpoints_in_universe']}, universe cpg_idx unique {stt['universe_cpg_unique']}, (chrom,pos) unique {stt['universe_chrom_pos_unique']}; beta range [{stt['beta_min']}, {stt['beta_max']}]. Genome build: counts-h5 / beta-h5 coordinates equal the universe covariates (checks above); the FASTA CG check is part of Level B (not executed).")
    miss = stt["missing_primary_frac_per_group"]
    P(f"* **Missingness**: CpGs observed in all 39 groups {stt['cpgs_with_all_groups_primary']}, >= 20 groups {stt['cpgs_with_ge20_groups_primary']}, < 20 groups {stt['cpgs_with_lt20_groups_primary']}; per-group missing fraction (primary mask) min {f(min(miss.values()), 4)} / max {f(max(miss.values()), 4)} ({max(miss, key=miss.get)}).")
    P("* **Aggregation rule** (frozen, read not run): cov >= 10; donor = published PatientID; donor beta = mean of valid sample betas; group beta = mean over donors; mask_primary = >= min(2, n_donors) valid donors; stored float16.")
    pf16 = res1["pearson_float16_path_vs_frozen"]
    P(f"* **Target reproduction** (float16-rounded recomputation vs frozen `loyfer_pearson`, all {pf16['n_pairs']} pairs): max |diff| on pairs not touching a boundary cell {e(pf16['max_abs_diff_clean_pairs'])} (tolerance 1e-9); {pf16['n_pairs_touching_f16_boundary_cells']} pairs touch a rounding-boundary cell (max |diff| over all pairs {e(pf16['max_abs_diff_all_pairs'])}).")
    gb = res1["group_beta_vs_frozen_float16"]
    P(f"* **Float16 rounding-boundary cells (attempt 1 FAIL and Amendment 2)**: {gb['n_f16_rounding_mismatch']} of {gb['n_cells']} cells ({e(gb['frac_f16_rounding_mismatch'])}) have round_f16(beta_f64) != frozen float16 (superseded threshold 1e-5: would have passed = {res1['f16_mismatch_fraction_superseded_threshold']['would_have_passed']}). "
      f"Diagnosed mechanism (read-only diagnosis `step1_diag_f16_mismatch/diagnosis_attempt1.json`): all mismatch cells lie within {e(diag['mismatch_distance_to_midpoint_ulp']['max'])} float16 ULP (relative {e(diag['mismatch_distance_to_midpoint_relative']['max'])}) of a rounding midpoint, "
      f"i.e. within float32 precision, consistent with float32 accumulation in the frozen aggregation; {diag['n_distinct_cpgs']} distinct CpGs (max {diag['max_mismatches_in_one_cpg']} per CpG); a float32 re-implementation reproduces {diag['float32_accumulation_reproduces_frozen_cells']} of {diag['n_cells']} frozen cells and "
      f"{diag['mismatch_cells_explained_by_float32_accumulation']} of {diag['n_mismatch']} mismatch cells (hypothesis supported, not bit-exact). New hard rule: relative distance <= 64 x 2^-24 = 3.8e-06; observed max {e(gb['mismatch_midpoint_rel_max'])}; cells beyond tolerance {gb['n_mismatch_beyond_f32_tol']}. **PASS_LEVEL_A passes only with Amendment 2.** The case is classified NON-BUG.")
    un = res1["pearson_unrounded_vs_frozen"]
    P(f"* **Effect bound of the float16 storage on the target**: unrounded float64 Pearson vs frozen: max |diff| {f(un['max_abs_diff'], 5)}, median {e(un['median_abs_diff'])}, Spearman {f(un['spearman_unrounded_vs_frozen'], 7)}, fraction > 0.01: {un['frac_gt_0.01']}; "
      f"primary on the unrounded target changes by at most {e(max(abs(x) for x in flags['TARGET_LIMITATION']['evidence']['precision_deltas'].values()))} for any arm (ordering changed: {flags['TARGET_LIMITATION']['evidence']['precision_order_changed']}).")
    P(f"* Informational: per-sample mean beta (array universe) vs published S1 'CpG methylation' Spearman {f(res1['sample_mean_beta_vs_S1_CpG_methylation']['spearman'], 3)} (n = {res1['sample_mean_beta_vs_S1_CpG_methylation']['n']}).\n")

    # ------------------------------------------------------------------ 2
    P("## 2. Reliability / noise ceiling (section 2)\n")
    P("Donor-level split-half, 20 repetitions (`default_rng([17000, rep])`), equal halves (odd donor dropped), frozen aggregation rule per half; Spearman-Brown rho = 2r/(1+r); ceiling = sqrt(max(rho,0)) (descriptive attenuation bound). Median [2.5%, 97.5%] over repetitions.\n")
    for var, nm in (("donors_ge2", "groups with >= 2 donors"), ("donors_ge4", "groups with >= 4 donors")):
        r_ = rel[var]
        P(f"**{nm}**: {r_['n_groups_used']} groups, min shared groups {r_['min_shared']}, pairs defined per repetition median {r_['n_pairs_defined_per_rep']['median']:.0f}.\n")
        P(table(["Cell", "n (median)", "r (Spearman)", "rho (Spearman-Brown)", "ceiling"],
                [(k, f"{x['n']['median']:.0f}", medq(x["r_spearman"]), medq(x["rho_sb"]), medq(x["ceiling"])) for k, x in r_["summary"].items()]))
        P("")
    P(f"Single-sample / single-donor groups (not estimable by split-half): {', '.join(rel['single_sample_groups_not_estimable'])}. N = 1 groups are reported separately; every pair target includes them in all 39 columns.\n")

    # ------------------------------------------------------------------ 3
    P("## 3. Target semantics (section 3)\n")
    th = sem["thresholds_recorded"]
    P(f"Thresholds recorded at run time (before examples were drawn): R_small (25th pct of pair min-range) = {f(th['R_small_min_pair_range_q25'], 4)}; G_hi (75th pct of per-CpG R2 with the global group level) = {f(th['G_hi_r2_level_q75'], 4)}; "
      f"V edges {th['V_edges_registered']}; bands: top 1% >= {f(th['band_thresholds']['q99'], 4)}, bottom 1% <= {f(th['band_thresholds']['q01'], 4)}, median band [{f(th['band_thresholds']['q495'], 4)}, {f(th['band_thresholds']['q505'], 4)}].\n")
    bs_ = sem["band_summary"]
    rows = []
    for b in ("top1pct", "median", "bottom1pct", "all"):
        x = bs_[b]
        cs = x["class_shares"]
        rows.append((b, cs["n"], f(cs["NEAR_CONSTANT"], 3), f(cs["SMALL_AMPLITUDE"], 3), f(cs["GLOBAL_LEVEL"], 3), f(cs["SPECIFIC_PROGRAM"], 3),
                     med(x["pair_mean_beta"]), med(x["pair_min_variance"], dd=4), med(x["pair_min_range"]), med(x["n_informative_groups"], dd=0),
                     med(x["target_pearson"]), med(x["pearson_after_global_level_removed"]), f(x["frac_persist_after_global_level_removal"], 3)))
    P(table(["Band", "n", "NEAR_CONSTANT", "SMALL_AMPLITUDE", "GLOBAL_LEVEL", "SPECIFIC_PROGRAM", "median mean beta", "median min variance", "median min range",
             "median informative groups", "median target", "median Pearson after global-level removal", "fraction persisting"], rows))
    d_ = sem["topband_dominance"]
    P(f"\n**Dominance rule R2(c)**: share(near-constant u small-amplitude) in the top-1% band = {f(d_['share_top'], 3)}, among all pairs = {f(d_['share_all'], 3)}, enrichment {f(d_['enrichment'], 3)}; thresholds share >= 0.50 and enrichment >= 1.5; dominated = {d_['dominated']}.\n")
    P("Class shares per stratum:\n")
    P(table(["Stratum", "n"] + list(("NEAR_CONSTANT", "SMALL_AMPLITUDE", "GLOBAL_LEVEL", "SPECIFIC_PROGRAM")),
            [(s, x["n"], *[f(x[c], 3) for c in ("NEAR_CONSTANT", "SMALL_AMPLITUDE", "GLOBAL_LEVEL", "SPECIFIC_PROGRAM")]) for s, x in sem["class_shares_per_stratum"].items()]))
    P("\nDeterministic examples (6 per band, seeded `default_rng([17, band])`, no manual selection): profile vectors are tabulated in `step3_4_semantics_pairs/semantics_examples.csv`; no figures are produced in this report.\n")

    # ------------------------------------------------------------------ 4
    P("## 4. Pair-list audit per distance stratum (section 4)\n")
    cmpd = sem["per_stratum_vs_reference_(>1Mb+inter)"]
    keyv = [("methylation_mean__pair_mean", "pair mean beta"), ("methylation_variance__pair_mean", "pair mean variance"), ("amplitude_range__pair_mean", "pair mean range"),
            ("gc_content__pair_mean", "GC"), ("cpg_density_hg38__pair_mean", "CpG density (hg38)"), ("tss_dist_log10__pair_mean", "TSS distance (log10)"),
            ("peak_count_total__pair_mean", "peak count total (regulatory tracks overlapped)"), ("peak_count_histone__pair_mean", "peak count histone"),
            ("peak_count_dnase__pair_mean", "peak count DNase"), ("target_pearson", "target Pearson")]
    P("Mean per stratum (D2-kept pairs); pair-level value = mean of the two CpGs:\n")
    P(table(["Variable"] + list(STRATA), [(nm, *[f(cmpd[k]["per_stratum"][s]["mean"], 3) for s in STRATA]) for k, nm in keyv]))
    P("\nStandardised mean difference (Cohen d) / KS statistic of each stratum vs the reference set (pooled >1Mb + interchromosomal):\n")
    keyv2 = keyv[:9] + [("methylation_mean__abs_diff", "|d| mean beta"), ("gc_content__abs_diff", "|d| GC"), ("cpg_density_hg38__abs_diff", "|d| CpG density"),
                        ("peak_count_total__abs_diff", "|d| peak count total"), ("tss_dist_log10__abs_diff", "|d| TSS")]
    P(table(["Variable"] + list(STRATA), [(nm, *[f"{f(cmpd[k]['vs_reference'][s]['smd'], 3, True)} / {f(cmpd[k]['vs_reference'][s]['ks'], 3)}" for s in STRATA]) for k, nm in keyv2]))
    P("\nContext and chromosome composition, zero-row exclusion (candidate store: 849 all-zero loci = D2 union):\n")
    P(table(["Stratum", "same-context share", "island share (CpG i)", "JS distance of chromosome composition vs reference", "pairs excluded by D2 (of n)"],
            [(s, f(sem["context_per_stratum"][s]["same_context"], 3), f(sem["context_per_stratum"][s]["context_share_i"].get("island", float("nan")), 3),
              f(sem["chromosome_composition"]["js_distance_vs_reference"][s], 3),
              f"{sem['zero_row_exclusion_per_stratum'][s]['n_excluded_D2']} of {sem['zero_row_exclusion_per_stratum'][s]['n']}") for s in STRATA]))
    P("\n**Why the arm ranking differs between short and long range (only what the data show)**: per-stratum Spearman(target, cosine) of each arm and the partial Spearman controlling for covariate closeness (-|dM|, -|dGC|, -|d peak count|):\n")
    P(table(["Arm"] + list(STRATA), [(LABEL[a], *[f(base[a]["native"]["per_stratum"][s]["value"], 3, True) for s in STRATA]) for a in ARMS]))
    P("\nPartial Spearman given [-|dM|, -|dGC|, -|d peak total|]:\n")
    P(table(["Arm"] + list(STRATA), [(LABEL[a], *[f(base[a]["mediation_partial_spearman"][s]["partial_given_dM_dGC_dPeak"], 3, True) for s in STRATA]) for a in ARMS]))
    P("\nOrdering of the arms per stratum (by Spearman): " + "; ".join(f"{s}: {ordering({a: base[a]['native']['per_stratum'][s]['value'] for a in ARMS})}" for s in STRATA) + ".")
    P("\nSpearman of the cosine with -|d peak total| (candidate): " + ", ".join(f"{s} {f(base[CAND]['mediation_partial_spearman'][s]['spearman_cosine_vs_neg_abs_dPeak'], 3, True)}" for s in STRATA) +
      "; of the target with -|d peak total|: " + ", ".join(f"{s} {f(base[CAND]['mediation_partial_spearman'][s]['spearman_target_vs_neg_abs_dPeak'], 3, True)}" for s in STRATA) + ".\n")

    # ------------------------------------------------------------------ 5
    P("## 5. Baselines on the same frozen pair universe (section 5)\n")
    P("Native cosine, primary-style pooled statistic (>1Mb + inter, equal stratum weight) and per stratum, with CIs; reproduces the frozen primary of every arm (checked at run time to 1e-9):\n")
    P(table(["Arm", "dim", "pooled primary-style [CI]"] + list(STRATA), [(LABEL[a], base[a]["dim"], ci(pooled[a]), *[v(base[a]["native"]["per_stratum"][s], 3) for s in STRATA]) for a in ARMS]))
    P("\nNegative controls per arm (100 row-permutations of the embedding among CpGs, seeds 1000-1099; 100 random Gaussian embeddings of the arm dimension, seeds 2000-2099); pooled primary-style statistic:\n")
    P(table(["Arm", "shuffled mean (sd)", "shuffled [q2.5, q97.5]", "Gaussian mean (sd)", "Gaussian [q2.5, q97.5]", "NULL_HI", "native - NULL_HI"],
            [(LABEL[a], f"{f(base[a]['null_shuffled_pooled']['mean'], 4, True)} ({f(base[a]['null_shuffled_pooled']['sd'], 4)})",
              f"[{f(base[a]['null_shuffled_pooled']['q025'], 4)}, {f(base[a]['null_shuffled_pooled']['q975'], 4)}]",
              f"{f(base[a]['null_random_gaussian_pooled']['mean'], 4, True)} ({f(base[a]['null_random_gaussian_pooled']['sd'], 4)})",
              f"[{f(base[a]['null_random_gaussian_pooled']['q025'], 4)}, {f(base[a]['null_random_gaussian_pooled']['q975'], 4)}]", f(nullhi[a], 4),
              f(base[a]["native_minus_null_hi"], 4, True)) for a in ARMS]))
    P("\nChromosome-only, distance and simple covariate baselines (score vs target; pooled primary-style [CI] and per stratum; 'n/a' = undefined by construction: chromosome-only is constant within one stratum, distance is undefined for interchromosomal pairs):\n")
    rows = []
    for k, x in cov.items():
        rows.append((k, ci(x["pooled_gt1Mb_inter_eqw"]), *[v(x["per_stratum"][s], 3) for s in STRATA], v(x["all_pairs"], 3)))
    P(table(["Baseline", "pooled primary-style [CI]"] + list(STRATA) + ["all pairs (unweighted)"], rows))
    nl = cov["neg_log10_distance"]
    P(f"\nDistance baseline over the five intra strata with equal stratum weight: {ci(nl['intra_5strata_equal_weight'])}.")
    P(f"\n**CpGPT {f(native[COMPS[0]], 4)} vs simple controls**: baselines in the same range or above: " + ", ".join(f"{k} {f(x, 3, True)}" for k, x in sorted(cov_pool.items(), key=lambda kv: -kv[1]) if x >= 0.5 * native[COMPS[0]]) +
      f". Null controls reach at most {f(max(max(base[a]['null_shuffled_pooled']['max'], base[a]['null_random_gaussian_pooled']['max']) for a in ARMS), 4)}.")
    P("\n**Precision of the frozen (float16) target**: primary-style statistic on the UNROUNDED target per arm: " + "; ".join(
        f"{LABEL[a]} {f(base[a]['unrounded_target_pooled']['value'], 6, True)} (delta {e(base[a]['unrounded_minus_frozen'])})" for a in ARMS) +
      f"; ordering frozen: {ordering(native)}.\n")
    reps = {a: NPZ(f"step5_baselines/reps_{a}.npz")["native"] for a in ARMS}
    P("Paired contrasts of the native statistic (candidate - comparator, same draws): " + "; ".join(
        f"vs {LABEL[c]}: {f(contrast(native[CAND], reps[CAND], native[c], reps[c])['delta'], 4, True)} [{f(contrast(native[CAND], reps[CAND], native[c], reps[c])['ci_lo'], 4)}, {f(contrast(native[CAND], reps[CAND], native[c], reps[c])['ci_hi'], 4)}]"
        for c in ARMS[1:]) + ".\n")

    # ------------------------------------------------------------------ 6
    P("## 6. Raw regulatory features vs 256-d SVD embedding (section 6)\n")
    pr = rv["projection_reproduction"]
    P(f"Raw space: {rv['n_tracks']} binary tracks (histone + DNase, resolver `regulatory_histone_dnase`); CSR only. Compressor weights all 1: {rv['compressor_weights_all_one']} (the column-weighted cosine IS the binary cosine). "
      f"Raw all-zero loci {rv['n_raw_all_zero_loci']}; pairs undefined after D2: {rv['n_pairs_raw_undefined_after_D2']}; common pair set {rv['n_pairs_common_set']} (D2 set {rv['n_pairs_D2']}). "
      f"Projection reproduction on 200 rows: max |diff| {e(pr['max_abs_diff'])}, registered tolerance 1e-5, `ok` = {pr['ok']} (**tolerance not met; float32 matrix product, informational: no decision rule uses it**).\n")
    rows = []
    for k, nm in (("raw_binary_cosine", "Raw binary cosine"), ("raw_jaccard", "Raw Jaccard"), ("embedding256_cosine_native", "Native 256-d cosine")):
        x = rv["scores"][k]
        rows.append((nm, ci(x["pooled_gt1Mb_inter_eqw"]), v(x["all_pairs"], 3), v(x["per_stratum"][">1Mb"], 3), v(x["per_stratum"]["interchromosomal"], 3), v(x["pooled_V_top10pct"], 3), v(x["unweighted_V_top10pct"], 3)))
    P(table(["Score", "pooled primary-style [CI]", "all pairs", ">1Mb", "interchromosomal", "V_top10pct pooled", "V_top10pct unweighted"], rows))
    P("\nPer stratum Spearman: " + "; ".join(f"{k}: " + ", ".join(f"{s} {v(rv['scores'][k]['per_stratum'][s], 3)}" for s in STRATA) for k in ("raw_binary_cosine", "raw_jaccard", "embedding256_cosine_native")) + ".")
    P("\nPaired delta (raw - native 256-d, pooled): " + "; ".join(f"{k}: {f(x['delta'], 4, True)} [{f(x['ci_lo'], 4)}, {f(x['ci_hi'], 4)}]" for k, x in rv["delta_vs_native_embedding_pooled"].items()) +
      ". Reading under the registered rule: the raw sources do not recover the target (both are below NULL_HI + 0.03 and below the native cosine); the failure is present in the raw feature space.\n")

    # ------------------------------------------------------------------ 7
    P("## 7. SVD / embedding geometry (section 7; descriptive)\n")
    sc = svd[CAND]
    P(f"Embedding-space PCA of the candidate over {sc['universe_n_loci']} D2 universe loci. First 20 PCs: explained variance, Spearman of the PC score with covariates, eta^2 with CpG context, and the Spearman of the target with the PC's centred dot-product contribution (s_ik s_jk, pooled primary-style).\n")
    covs = list(sc["assoc_pc_scores_spearman"])
    rows = []
    for k in range(20):
        rows.append((k + 1, f(sc["explained_variance_ratio_first20"][k], 4), *[f(sc["assoc_pc_scores_spearman"][c][k], 2, True) for c in covs], f(sc["assoc_pc_scores_eta2_context"][k], 3),
                     f(sc["pc_dot_contribution_pooled_spearman"][k], 4, True)))
    P(table(["PC", "EV ratio"] + covs + ["eta2 context", "dot-contribution Spearman"], rows))
    P("\nRaw embedding coordinates 1-5 (the SVD component axes) vs covariates (Spearman): " + "; ".join(
        f"coord {k + 1}: " + ", ".join(f"{c} {f(sc['assoc_embedding_coordinates_1_20_spearman'][c][k], 2, True)}" for c in ("peak_total", "gc_content", "loyfer_mean_methylation")) for k in range(5)) + ".")
    P("\n**Diagnostic curve (primary-style pooled [CI]; V_top10pct pooled point estimate)**; centred = mean removed; PCk removed = project out the top-k axes of the centred embedding; whitening eps 1e-6 lambda_max:\n")
    names = list(sc["curve"])
    P(table(["Geometry"] + [LABEL[a] for a in ARMS], [(n_, *[ci(svd[a]["curve"][n_]["pooled_gt1Mb_inter_eqw"], 3) for a in ARMS]) for n_ in names]))
    P("\nSame curve, V_top10pct pooled (point estimate) and per stratum for the candidate:\n")
    P(table(["Geometry", "V_top10pct (candidate)"] + [f"{s} (candidate)" for s in STRATA],
            [(n_, v(sc["curve"][n_]["pooled_V_top10pct"], 3), *[v(sc["curve"][n_]["per_stratum"][s], 3) for s in STRATA]) for n_ in names]))
    dots = sc["pc_dot_contribution_pooled_spearman"]
    order = sorted(range(20), key=lambda i: dots[i])
    nonneg = all(x >= 0 for x in dots)
    P(f"\nDescriptive attribution (candidate): the centred dot-product contribution of each of the 20 PCs has Spearman with the target between {f(min(dots), 4, True)} and {f(max(dots), 4, True)} "
      f"(all non-negative: {nonneg}); the largest are " + ", ".join(f"PC{i + 1} {f(dots[i], 3, True)}" for i in order[::-1][:3]) +
      f". The uncentered native cosine is {f(cen['native_cosine']['pooled_gt1Mb_inter_eqw']['value'], 3, True)} whereas the centred cosine is {f(cen['centered_cosine']['pooled_gt1Mb_inter_eqw']['value'], 3, True)}: "
      f"subtracting the universe mean embedding alone changes the sign. PC1: EV {f(sc['explained_variance_ratio_first20'][0], 3)}, Spearman with peak count {f(sc['assoc_pc_scores_spearman']['peak_total'][0], 3, True)}, with Loyfer mean methylation {f(sc['assoc_pc_scores_spearman']['loyfer_mean_methylation'][0], 3, True)}. "
      "No claim is made on mechanism beyond these descriptive numbers.")
    cb = sc["consistency_with_B2"]
    P(f"\nConsistency with the earlier B2 output: centred {f(cb['centered_here'], 6)} vs {f(cb['centered_B2'], 6)}; PC1-removed {f(cb['pc1_removed_here'], 6)} vs {f(cb['pc1_removed_B2'], 6)}.\n")

    # ------------------------------------------------------------------ 8
    P("## 8. Linear decodability <-> pairwise geometry (section 8)\n")
    o = oof[CAND]
    P(f"Ridge = `ridge_blocked_cv` unchanged, frozen chromosome folds (seed 17), original alpha grid {o['alpha_grid']}; 382,041 complete-case CpGs; OOF predictions strictly out of fold (independent closed-form refit check max |diff| {e(max(oof[a]['refit_check_max_abs_diff'] for a in ARMS))}; stored B3 r2_global reproduced, max |diff| {e(max(oof[a]['B3_reproduction']['max_abs_diff'] for a in ARMS))}). "
      f"Pair set: both CpGs complete-case and D2-kept = {o['oof_similarity']['n_pairs_complete_case']} pairs (of {o['oof_similarity']['n_pairs_D2']}); same-fold pairs {o['oof_similarity']['n_pairs_same_fold']} (interchromosomal {o['oof_similarity']['n_same_fold_per_stratum']['interchromosomal']}). "
      "The two CpGs of a pair may be predicted by models of different folds (always the same fold for intra pairs).\n")
    rows = []
    for a in ARMS:
        oo = oof[a]["oof_similarity"]
        for t_, nm in (("raw", "T1 raw"), ("centered", "T2 centred")):
            x = oo[t_]
            rows.append((LABEL[a], nm, ci(x["all_complete_case"]["pooled_gt1Mb_inter_eqw"]), v(x["all_complete_case"]["per_stratum"][">1Mb"], 3), v(x["all_complete_case"]["per_stratum"]["interchromosomal"], 3),
                         v(x["all_pairs_unweighted"], 3), v(x["V_top10pct"], 3), ci(x["same_fold_only"]["pooled_gt1Mb_inter_eqw"], 3),
                         f"{f(x['delta_vs_native_same_pairs']['delta'], 3, True)} [{f(x['delta_vs_native_same_pairs']['ci_lo'], 3)}, {f(x['delta_vs_native_same_pairs']['ci_hi'], 3)}]"))
        rows.append((LABEL[a], "native cosine (same pairs)", ci(oo["native_cosine_same_pair_set"]), "", "", "", "", "", ""))
    P(table(["Arm", "Score", "pooled primary-style [CI]", ">1Mb", "inter", "all pairs (unweighted)", "V_top10pct", "same-fold only pooled", "delta vs native (same pairs)"], rows))
    P("\nPer stratum OOF T1 Spearman: " + "; ".join(f"{LABEL[a]}: " + ", ".join(f"{s} {v(oof[a]['oof_similarity']['raw']['all_complete_case']['per_stratum'][s], 3)}" for s in STRATA) for a in ARMS) + ".")
    P("\nLinear probe R2 (original grid; global, pooled over CpGs x groups): " + "; ".join(
        f"{LABEL[a]}: " + ", ".join(f"{k} {f(x, 3)}" for k, x in oof[a]["r2_global"].items()) for a in ARMS) + ".")
    r1 = {a: NPZ(f"step8_oof_similarity/reps_{a}_{ORIG_TAG}.npz") for a in ARMS}
    P("\nPaired contrasts on the OOF T1 statistic (candidate - comparator): " + "; ".join(
        f"vs {LABEL[c]}: " + (lambda d_: f"{f(d_['delta'], 4, True)} [{f(d_['ci_lo'], 4)}, {f(d_['ci_hi'], 4)}]")(
            contrast(oof[CAND]["oof_similarity"]["raw"]["all_complete_case"]["pooled_gt1Mb_inter_eqw"]["value"], r1[CAND][f"{CAND}_oof_raw"],
                     oof[c]["oof_similarity"]["raw"]["all_complete_case"]["pooled_gt1Mb_inter_eqw"]["value"], r1[c][f"{c}_oof_raw"])) for c in ARMS[1:]) + ".\n")

    # ------------------------------------------------------------------ 9
    P("## 9. Probe fairness: extended alpha grid (section 9; sensitivity only)\n")
    P(f"Registered grid 10^-2..10^8 (11 values); edge rule: an arm selects an edge if >= 10% of its (fold x target) alphas sit at the grid min/max; then ALL arms are refitted with the grid extended by 2 decades on that side, at most 2 extensions. "
      f"Executed: {ph['n_extensions']} extensions, final grid {[f'{g:g}' for g in final_grid]}.\n")
    rows = []
    for i, h in enumerate(ph["history"]):
        rows.append((i, len(h["grid"]), f"{h['grid'][0]:g} .. {h['grid'][-1]:g}", *[f"{f(h['edge_reports'][a]['frac_at_min'], 3)} / {f(h['edge_reports'][a]['frac_at_max'], 3)}" for a in ARMS]))
    P(table(["Round", "n alphas", "grid", *[f"{LABEL[a]} min / max edge fraction" for a in ARMS]], rows))
    P("\nAfter the maximum number of extensions the candidate still has " + f"{f(ph['history'][-1]['edge_reports'][CAND]['frac_at_min'], 3)} of its selections at the grid minimum (rule threshold 0.10): the edge condition is **not resolved** for that arm; selections at the minimum alpha mean the smallest registered regularisation was preferred.\n")
    rows = []
    for a in ARMS:
        rows.append((LABEL[a], f(oof[a]["alpha_edge_report"]["frac_at_min"], 3), f(oof[a]["alpha_edge_report"]["frac_at_max"], 3), f(ext[a]["alpha_edge_report"]["frac_at_min"], 3), f(ext[a]["alpha_edge_report"]["frac_at_max"], 3),
                     ", ".join(f"{x:g}" for x in oof[a]["alpha_median_per_fold_all_targets"]), ", ".join(f"{x:g}" for x in ext[a]["alpha_median_per_fold_all_targets"])))
    P(table(["Arm", "orig min frac", "orig max frac", "ext min frac", "ext max frac", "median alpha per fold (orig)", "median alpha per fold (ext)"], rows))
    P("\nr2_global (T1 raw / T2 centred / T3 level; all CpGs / variable CpGs), original -> extended grid:\n")
    keys = ["raw__all_cpgs", "raw__variable_cpgs", "centered__all_cpgs", "centered__variable_cpgs", "level__all_cpgs", "level__variable_cpgs"]
    P(table(["Arm"] + keys, [(LABEL[a], *[f"{f(oof[a]['r2_global'][k], 4)} -> {f(ext[a]['r2_global'][k], 4)}" for k in keys]) for a in ARMS]))
    rk = lambda src, k: ordering({a: src[a]["r2_global"][k] for a in ARMS})
    P(f"\nRanking of the arms by r2_global (centred, variable CpGs): original {rk(oof, 'centered__variable_cpgs')}; extended {rk(ext, 'centered__variable_cpgs')}. "
      f"Ranking T1 all CpGs: original {rk(oof, 'raw__all_cpgs')}; extended {rk(ext, 'raw__all_cpgs')}. Ranking changes: "
      f"{rk(oof, 'centered__variable_cpgs') != rk(ext, 'centered__variable_cpgs') or rk(oof, 'raw__all_cpgs') != rk(ext, 'raw__all_cpgs')}.")
    P("\nOOF T1 pooled statistic under the extended grid: " + "; ".join(f"{LABEL[a]}: {ci(ext[a]['oof_similarity']['raw']['all_complete_case']['pooled_gt1Mb_inter_eqw'], 3)}" for a in ARMS) +
      f"; ranking {ordering({a: ext[a]['oof_similarity']['raw']['all_complete_case']['pooled_gt1Mb_inter_eqw']['value'] for a in ARMS})}. The original ridge results are not replaced.\n")

    # ------------------------------------------------------------------ 10
    P("## 10. Pretraining provenance (section 10; documentary, reused)\n")
    P("Source: `docs/BIOLOGICAL_VALIDATION_V2_EXPLORATORY_POSTHOC_PROVENANCE_AUDIT.md` (labels: VERIFIED-IN-REPO, VERIFIED-LOCAL-SIBLING, VERIFIED-EXTERNAL fetched 2026-10-04, UNVERIFIED); no new web search was made for this report. The classification below restates that audit; it does not reinterpret the primary.\n")
    P(table(["Item", "CpGPT-large", "DeepCpG-DNA (HCC)"], [
        ("Pretraining objective", "methylation-value reconstruction / uncertainty, Wasserstein on beta distribution, contrastive and sample losses (VERIFIED-LOCAL-SIBLING config `large.yaml`)", "prediction of single-cell CpG methylation states from the DNA window (zoo/README statement VERIFIED-IN-REPO; paper text UNVERIFIED)"),
        ("Corpus", "CpGCorpus: Illumina arrays (450k/EPIC/EPICv2), >150,000 human bulk samples from >2,000 studies (VERIFIED-EXTERNAL, preprint summary and GitHub README)", "Hou et al. 2016 scRRBS, 25 human hepatocellular carcinoma cells (VERIFIED-IN-REPO README; VERIFIED-EXTERNAL zoo page)"),
        ("Direct supervision on methylation", "YES (VERIFIED)", "YES (VERIFIED as training fact)"),
        ("Overlap with Loyfer individuals/datasets (GSE186458, WGBS of sorted cell types)", "direct data overlap UNVERIFIED (arrays vs WGBS; whether Loyfer samples are in CpGCorpus not checked)", "no evidence (cancer cell-line scRRBS)"),
        ("Classification", "SAME_ASSAY_DOMAIN (methylation objective on array data at the benchmark loci); NO_EVIDENCE_OF_OVERLAP for direct data overlap (UNVERIFIED)", "SAME_ASSAY_DOMAIN (methylation states, different tissue/assay); NO_EVIDENCE_OF_OVERLAP"),
        ("Not verifiable here", "exact CpGCorpus composition; whether the 512-d locus vector retains methylation information beyond sequence", "training CpG set of the HCC DNA module; original paper text")]))
    P("\nDocumentary outcome (fixed in the registration, independent of any result): both sequence arms = DIRECT_METHYLATION_SUPERVISION. This audit cannot separate 'the ENCODE-feature arms lack the information' from 'the sequence arms were trained on methylation'.\n")

    # ------------------------------------------------------------------ 11
    P("## 11. Decision tree: mechanical application of the registered rules\n")
    P(f"Constants: M_REC = {dec['thresholds']['M_REC']}; NULL_HI (97.5th percentile of the 200 nulls) of the candidate = {f(dec['null_hi_candidate'], 5)}; 'recovers' = statistic >= NULL_HI + 0.03 = {f(dec['null_hi_candidate'] + dec['thresholds']['M_REC'], 5)}.\n")
    tl = flags["TARGET_LIMITATION"]
    ev = tl["evidence"]
    cg = flags["COMPRESSION_GEOMETRY"]
    ia = flags["INFORMATION_ABSENCE"]
    ca = flags["COMPARATOR_ADVANTAGE"]
    P(table(["Class", "Flag", "Evidence / margin to the registered threshold"], [
        ("BUG", flags["BUG"]["flag"], f"step-1 verdict {flags['BUG']['evidence']['step1_verdict']} (failed checks {flags['BUG']['evidence']['failed_checks']}); scope: {flags['BUG']['bug_absent_statement_scope']}"),
        ("TARGET_LIMITATION", tl["flag"], "; ".join([
            f"(a) ceiling {f(ev['ceiling_primary'], 3)} vs < 0.30 -> {tl['sub_rules']['a_ceiling_lt_0.30']} (margin {f(ev['ceiling_primary'] - 0.30, 3, True)})",
            f"(b) reliability >1Mb {f(ev['rel_gt1Mb'], 3)} / inter {f(ev['rel_inter'], 3)} vs < 0.50 -> {tl['sub_rules']['b_stratum_reliability_lt_0.5']} (margins {f(ev['rel_gt1Mb'] - 0.5, 3, True)} / {f(ev['rel_inter'] - 0.5, 3, True)})",
            f"(c) top-band share {f(ev['topband']['share_top'], 3)} (>= 0.50 needed), enrichment {f(ev['topband']['enrichment'], 3)} (>= 1.5 needed) -> {tl['sub_rules']['c_topband_dominated_by_near_constant_or_small_amplitude']}",
            f"(d) max |primary(unrounded) - primary(frozen)| {e(max(abs(x) for x in ev['precision_deltas'].values()))} vs > 0.02; ordering changed {ev['precision_order_changed']} -> {tl['sub_rules']['d_float16_precision_changes_primary']}"])),
        ("COMPRESSION_GEOMETRY", cg["flag"], f"native {f(cg['evidence']['native'], 4, True)} (recovers: {cg['native_recovers']}); sources recovering with delta-CI > 0: {cg['sources_recovering']}; "
                                             f"raw sources: " + ", ".join(f"{k} {f(x, 3, True)}" for k, x in cg["evidence"]["candidates"].items() if k.startswith("raw")) +
                                             "; OOF sources: " + ", ".join(f"{k} {f(x, 3, True)}" for k, x in cg["evidence"]["candidates"].items() if k.startswith("oof")) +
                                             f"; supporting PC-removal flag {cg['support_pc_removal']}"),
        ("INFORMATION_ABSENCE", ia["flag"], f"requires none of {len(ia['evidence']['candidate_values'])} sources >= {f(ia['evidence']['null_hi'] + ia['evidence']['margin'], 4)}; the OOF sources exceed it (max {f(max(ia['evidence']['candidate_values'].values()), 3)})"),
        ("COMPARATOR_ADVANTAGE", ca["flag"], "; ".join(f"{LABEL[c]}: {x['level']} ({sum(x['wins'].values())}/{len(x['wins'])} fair evaluations won)" for c, x in ca["per_comparator"].items()))]))
    P("\nCOMPRESSION_GEOMETRY delta (source - native on the same pairs) CIs: " + "; ".join(f"{k}: [{f(x[0], 3)}, {f(x[1], 3)}]" for k, x in cg["evidence"]["delta_ci"].items()) + ".")
    P("\nSupporting PC-removal flags (delta vs native >= 0.03 with CI > 0): " + ", ".join(f"{k} {x}" for k, x in cg["evidence"]["pc_removal_support"].items()) + ".")
    P("\nCOMPARATOR_ADVANTAGE, evaluation by evaluation (win = comparator - candidate CI > 0 and comparator >= its NULL_HI + 0.03):\n")
    evs = list(next(iter(ca["per_comparator"].values()))["wins"])
    P(table(["Evaluation"] + [LABEL[c] for c in COMPS], [(e_, *[ca["per_comparator"][c]["wins"][e_] for c in COMPS]) for e_ in evs]))
    P(f"\n**Resulting classes: {classes}.** Coexisting classes: {', '.join(classes) if len(classes) > 1 else 'none (single class)'}. INFORMATION_ABSENCE and COMPRESSION_GEOMETRY are mutually exclusive by construction. "
      "The raw-space evidence does not recover the target; the rule is satisfied through the OOF predicted-profile similarity only; BUG is limited by the Level-A scope.\n")

    # ------------------------------------------------------------------ 12
    P("## 12. Integrity\n")
    P(f"* Primary fingerprint (registered) `{cmp_['primary_fingerprint_before']}` before, `{cmp_['primary_fingerprint_after']}` after; unchanged = {cmp_['primary_fingerprint_unchanged']}.")
    P(f"* Snapshot of ALL files under the frozen tag directory: {man_before['n_files']} files before, {man_after['n_files']} after; identical = {cmp_['identical']} (added {len(cmp_['added'])}, removed {len(cmp_['removed'])}, changed {len(cmp_['changed'])}); audit root outside the fingerprinted tree = {cmp_['audit_root_outside_fingerprinted_tree']}.")
    P(f"* Protocol tag `{pc_manifest['report_inputs']['protocol_tag']}` -> `{pc_manifest['report_inputs']['protocol_commit']}`.")
    P(table(["Step", "HEAD at run", "registration sha256", "code sha256", "freeze modules imported", "TCGA referenced", "gate checks failing"],
            [(s, m["head_commit"][:10], m["registration_doc"]["sha256"][:12], m["code_sha256"][:12], m["freeze_modules_imported"] or "none", m["tcga_test_set_referenced"],
              [c["name"] for c in m["gate_checks"] if not c["ok"]] or "none") for s, m in pc_manifest.items()]))
    P("\n* Amendment 1 (Level A only; Level B disabled, gate refuses `--level-b`), Amendment 2 (float32-precision midpoint rule), Bugfix 1 (report-inputs aggregation): see the registration. Attempt 1 (FAIL) is archived in `step1_target_audit_attempt1_FAIL/`.")
    P("* Frozen commands unused (never imported): see the registration section 12; the run manifests record `freeze_modules_imported`. Limit: Level B (raw `.pat` counts and parsing) NOT executed.\n")

    # ------------------------------------------------------------------ 13
    P("## 13. What the results do NOT show\n")
    P("* Linear decodability of the profile from the embedding (ridge R2) is not the same as the embedding's cosine geometry being aligned with cell-type methylation programs; the OOF-profile similarity is a different pairwise object built from a supervised linear map trained on methylation profiles of other chromosomes.")
    P("* The Level-A verdict does not verify raw counts or `.pat` parsing; 'BUG absent' is limited to the downstream construction.")
    P("* The effect of methylation pretraining of the comparators is not excluded (section 10); the audit cannot attribute the comparators' native-cosine lead to supervision rather than to sequence information.")
    P("* The OOF null reuses NULL_HI from the cosine nulls (no refitted nulls); the OOF statistics are on complete-case pairs only (346,503); block bootstrap by the chromosome of `cpg_i` approximates interchromosomal pairs.")
    P(f"* The probe-fairness edge condition remains unresolved for the candidate after the maximum number of extensions ({f(ph['history'][-1]['edge_reports'][CAND]['frac_at_min'], 3)} of selections at the minimum alpha).")
    P("* Nothing here designates a new primary or a representation; all statistics are descriptive and exploratory.\n")
    return "\n".join(L) + "\n"


def sources_manifest() -> dict:
    return {"generator": "scripts/bioval_v2/make_loyfer_audit_report.py", "n_sources": len(SOURCES),
            "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = build()
    man = sources_manifest()
    if a.check:
        ok = DOC.is_file() and DOC.read_text() == text
        ok2 = MANIFEST.is_file() and json.loads(MANIFEST.read_text()) == man
        print(f"report equals a fresh generation: {ok}; sources manifest unchanged: {ok2}; {man['n_sources']} source files")
        raise SystemExit(0 if ok and ok2 else 1)
    DOC.write_text(text)
    MANIFEST.write_text(json.dumps(man, indent=1))
    print(f"wrote {DOC} ({len(text.splitlines())} lines) and {MANIFEST} ({man['n_sources']} sources)")


if __name__ == "__main__":
    main()
