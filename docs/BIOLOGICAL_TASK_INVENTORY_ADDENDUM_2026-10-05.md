# Biological task inventory: addendum of 2026-10-05 (registry v2, data only)

Companion to `docs/BIOLOGICAL_TASK_INVENTORY.md` (not edited). Nothing was run, trained, bootstrapped or frozen. The committed v1 registry
files are untouched; v2 is written next to them with the suffix `_v2` by `paper/scripts/build_bio_task_registry_v2.py`
(deterministic: two consecutive builds give identical CSV hashes; validation: 0 errors). The excluded `loyfer_failure_audit` directory was neither
read nor hashed.

## Decisions of 2026-10-05 (recorded verbatim in intent)

1. YES: add `regulatory_histone_dnase_v1` to the EWAS probes at TWO DISTINCT LEVELS. (i) `exp5_candidate_completion`: exactly the Exp5 pairs, folds and
   bootstrap (direct comparison with the work already done). (ii) A real, frozen EWAS-v2 for the MAIN paper: same background loci for ALL embeddings,
   chromosome-blocked CV, preprocessing only inside the train fold, native representations (256 / 512 / 128 D), a shared technical matching, and a more
   aggressively context-matched SENSITIVITY, so that "the embedding recognises EWAS hits" can be distinguished from "it still recognises them after controlling
   for coarse genomic context".
2. Functional legacy = ONLY the campaign `full` SVD-256 store (`outputs/encode_atlas_v1/embeddings/full_a18b869b.h5`, arm `functional_256` / `full`),
   supplementary only. The old PCA store of the first bio-validation is purely HISTORICAL. The manuscript must not contain two different objects called
   `functional_annotations_pca`.
3. `histone + context` (campaign arm `add/assay/Histone ChIP-seq`) is NOT an acceptable proxy in the main paper; supplementary only with the explicit label
   `proxy_not_candidate` (it lacks DNase and contains dense context the final candidate does not use).
4. EWAS task support rule, fixed BEFORE any candidate result and based ONLY on N: >= 200 matched positives = main-quality; 100-199 = exploratory;
   < 100 = insufficient support (all may appear in supplementary). Ovarian / CHD / schizophrenia (and any set below the thresholds) do not become evidence
   because the candidate might do well.
5. Exp5 primary metric: AUROC; plus MATCHED-PAIR CONCORDANCE; AUPRC only as a SENSITIVITY (prevalence 50% is artificial under 1:1 matching), never primary.
6. Promoter, gene region, CpG island, cCRE, TSS: representation characterization / source-retention only, NOT evidence of biological generalization; cCRE is too close to
   DNase / H3K27ac / H3K4me3 to be an independent validation. Not built now.
7. Loyfer stays prominent: the frozen negative primary stays. If the running audit finds a true bug it is corrected with a declaration; if it finds only a geometry mismatch,
   the frozen negative result stays part of the story. (No registry action beyond a note; the audit outcome is not recorded here.)

## What changed in the registry (v2)

| file | change |
|---|---|
| `bio_task_registry_v2.csv` | 2,286 rows = 2,130 v1 rows + 156 DERIVED `matched_pair_concordance` rows (exp5, 13 sets x 2 variants x 6 arms). New columns: `representation_id_v1`, `row_id_v1`, `paper_use`, `proxy_flag`, `support_tier`, `support_tier_basis`, `metric_role`, `circularity_use_note`, `registry_v2_note` |
| `bio_task_matrix_v2.csv` | v1 + `funcleg_store_id`, `support_tier`, `matched_positives_exp5_base`, `primary_metric_v2`, `metric_hierarchy_v2`, `circularity_use_note`, `{cand,cpgpt,deepcpg,funcleg}__matched_pair_concordance` and `__status` |
| `per_task_audit_v2.csv` | v1 + `support_tier`, `matched_positives_exp5_base`, `metric_hierarchy_v2`, `circularity_use_note`, `rigor_grade_v2` (= v1 grade), `eligible_for_main_after_completion`, `eligibility_basis`, `current_status_note` |
| `bio_task_coverage_matrix_v2.csv` | v1 + proxy labels (`proxy_not_candidate`, `proxy_variant_not_main_panel`), `legacy_store_note`; column renamed `functional_annotations_pca_legacy_supp` |
| `bio_task_paired_contrasts_v2.csv` | content unchanged (+ `paper_use_v2`); contrasts only ever used the campaign store |
| `exp5_matched_pair_concordance_v2.csv` | DERIVED table: per set / variant / arm, mean over 20 matching draws, between-draw SD, re-derived AUROC and stored AUROC |
| `exp5_support_tiers_v2.csv`, `support_tier_task_list_v2.csv` | matched positives per exp5 set and variant; tier per task across all suites |
| `bio_task_registry_v2.json`, `manifests/registry_v2_outputs_sha256.json` | data dictionary (decisions, tier rule, concordance definition, planned-not-built targets) and sha256 of inputs/outputs |

### (a) Disambiguation of `functional_annotations_pca`
- `functional_annotations_pca` now means ONLY the campaign `full` SVD-256 store (bioval-v2, exp5, exp2, single-draw, UMAP rows), `paper_use = SUPPLEMENTARY_LEGACY_FUNCTIONAL`.
- The old PCA store (`functional_annotations__pca_native_genomewide.h5`, `legacy_bio_validation` suite) is renamed `functional_pca_biovalidation_legacy_store`, `proxy_flag = paper_use =
  HISTORICAL_NOT_FOR_PAPER`: 88 registry rows, 25 matrix rows (`funcleg_store_id`). The v1 id is kept in `representation_id_v1`.

### (b) Proxy flags
| flag | rows | arms |
|---|---|---|
| `proxy_not_candidate` | 33 | histone+context campaign arm (29), raw histone-/DNase-only feature subsets of exp2 (4) |
| `proxy_variant_not_main_panel` | 141 | compact CpGPT-256 (`cpgpt_large_svd256`, `fm/cpgpt_locus_large`), DeepCpG pad256 |
| `control_arm_not_main_panel` | 85 | context-only 256 |
| `raw_features_not_embedding` | 52 | other raw feature subsets (exp2) |
| `non_panel_variant` | 316 | small CpGPT, HepG2 DeepCpG, MethylGPT (legacy suites) |
| `HISTORICAL_NOT_FOR_PAPER` | 88 | old PCA store |

### (c) Support tier (decision 4; computed from exp5 matched positives = pairs, base variant; the probe variant gives identical tiers for every set)
Tiers are attached to every EWAS/phenotype task of every suite by set name (legacy, single-draw, exp5 base/probe, UMAP neighbourhood) with the basis recorded. No candidate
result exists, so the assignment cannot depend on one. EWAS-v2 must recompute N from its own (different) matching; this table is the exp5-era reference.

| tier | sets (matched positives base / probe) |
|---|---|
| main-quality (>= 200) | colorectal (2000/2000), prostate (2000/2000), age (2000/1998), BMI (2000/2000), smoking (2000/2000), breast (502/479), PhenoAge clock (498/488), Horvath clock (344/336), lung (241/235; 93% of the set is smoking CpGs), RA (238/216) : 10 sets |
| exploratory (100-199) | SLE (139/128) : 1 set |
| insufficient (< 100) | type 2 diabetes (70/67; 92% in BMI), Hannum clock (64/62), CHD (49/45), ovarian (28/27), schizophrenia (28/24) : 5 sets |

Per-task rows (legacy 25-task suite, single-draw, exp5 base, exp5 probe, UMAP) are in `support_tier_task_list_v2.csv`. Not-evaluable exp5 sets (ovarian, CHD, schizophrenia) have no scores but keep the tier.

### (d) Metric hierarchy
- Primary: AUROC. Co-primary for 1:1 matched designs: matched-pair concordance. AUPRC: SENSITIVITY only (`metric_role`).
- Definition: for each matched pair and matching draw, 1 if OOF score(EWAS CpG) > OOF score(matched background CpG), 0.5 if equal, 0 otherwise; mean over pairs; then mean over the 20 matching
  draws. Same pooled OOF scores and chromosome-blocked folds as the AUROC. 0.5 = chance.
- Derivable from existing artifacts? YES for exp5: `oof_predictions.parquet` stores pair_id, y_true, match_seed and score for all six exp5 arms (functional_256, CpGPT-512 native, DeepCpG-128 native,
  CpGPT SVD-256, DeepCpG pad256, context-only). 156 DERIVED rows (13 sets x 2 variants x 6 arms), point estimates only, flagged `DERIVED`; CIs are `needs_run` (bootstrap). The pooled AUROC re-derived from the
  same file matches the stored `summary_auc_delta.csv` within 3.5e-5 (float32 score storage) for all 156 cells. The candidate is `needs_run`. NOT derivable (marked `needs_run` / `not_applicable_unmatched_design`):
  the single-draw table (no stored scores) and the legacy probes (unmatched all-negatives design, no pairs). AUPRC is also derivable from the same parquet but was not computed (not requested; sensitivity only).
- Observation on derived values (not interpretation): in exp5 base, functional-256 vs CpGPT-512 has the same sign under concordance and under AUROC for all 13 sets; concordance and AUROC differ by up to a few hundredths per cell (e.g. T2D functional 0.634 vs 0.623).

### (e) Circularity notes (decision 6)
Rows for island / shore / shelf / open-sea (legacy context probes, UMAP neighbourhood) and the FANTOM5 TSS5k pool carry `circularity_use_note`; no promoter / gene-region / cCRE row exists (none built; `planned_not_built_characterization_targets`
in the JSON dictionary lists the five target families with `independent_validation = false`). cCRE: too close to the candidate's own DNase / H3K27ac / H3K4me3 families.

### (f) Rigor grades and eligibility
Grades unchanged: legacy random-fold probes remain PROBLEMATIC; exp5 base/probe remain SUPPLEMENTARY_ONLY until the candidate is added (`rigor_grade_v2` = `rigor_grade`, plus `current_status_note`).
New, clearly separate: `eligible_for_main_after_completion` over the 122 audit rows:
`TRUE (already main-quality)` 22 (bioval-v2 frozen/preregistered); `TRUE_AFTER_CANDIDATE_COMPLETION` 20 (exp5 base/probe x 10 main-quality sets, design rigorous: 1:1 matched, chromosome-blocked, fixed-C, CI);
`FALSE_exploratory_support` 2 (SLE); `FALSE_insufficient_support` 10; `FALSE` 68 (legacy PROBLEMATIC, single-draw, descriptive/campaign). The main-paper route for these tasks is EWAS-v2; exp5 completion is a discovery completion
(caveat: exp5 matching strata partly equal inputs of the functional store).

### (g) Loyfer
Registry note only on the exploratory post-hoc rows: frozen negative primary stands; audit outcome not recorded.

## Reproduction
`PYTHONPATH=$PWD/src nice -n 19 python paper/scripts/build_bio_task_registry_v2.py --validate --determinism` (about one minute; reads the 47 MB exp5 OOF parquet).
Input and output sha256: `paper/registry/manifests/registry_v2_outputs_sha256.json`.
