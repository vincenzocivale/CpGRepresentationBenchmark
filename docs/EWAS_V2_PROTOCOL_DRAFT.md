# EWAS-v2 protocol: DRAFT (freezable, not frozen; nothing run)

Machine-readable spec: `configs/ewas_v2/ewas_v2_spec_draft.yaml`. Design primitives with synthetic tests: `scripts/ewas_v2/ewas_v2_design.py`, `tests/test_ewas_v2_design.py`.
State: DRAFT. Intended to become the MAIN-paper EWAS evidence (decisions of 2026-10-05, item 1(ii)); `exp5_candidate_completion` (see `docs/EXP5_CANDIDATE_COMPLETION_PLAN.md`) is the separate discovery completion on the old Exp5 design.

## 0. Question and principle
Distinguish (a) "the embedding recognises EWAS hits" (MAIN: technical matching only) from (b) "it still recognises them after controlling for coarse genomic context" (SENSITIVITY S1). Everything is decided without looking at any embedding score:
sets, backgrounds, folds, tiers, metrics, contrasts, draw table. No tuning on results; both directions are reported (CpGPT may win age / Horvath / PhenoAge, the candidate smoking / disease: all shown).

## 1. Decisions of 2026-10-05 that this protocol implements
1(ii) real EWAS-v2: same background loci for ALL embeddings, chromosome-blocked CV, preprocessing inside the train fold only, native representations (256/512/128 D), shared technical matching, aggressive context-matched sensitivity.
2 functional legacy = campaign `full` SVD-256 only, supplementary; the old PCA store never appears. 3 histone+context is not a proxy in the main paper. 4 support rule on N only (>=200 / 100-199 / <100). 5 AUROC primary + matched-pair concordance, AUPRC sensitivity only.
6 promoter / gene region / island / cCRE / TSS are characterization only, not part of this protocol. 7 Loyfer: not touched here.

## 2. Hit sets (all from the repo; `data/bio_annotations/known_sets/*.npy`, sha256 pinned in `configs/exp5_candidate_completion/exp5_pins.json`)
Universe: 408,399 array CpGs (chr1-22); N = members present in the universe; "probe-known" = members with a known Infinium type (matching requires it; 6,429 universe CpGs have none).

| set | family | source / version | N universe | N probe-known | expected tier |
|---|---|---|---|---|---|
| smoking | exposure | EWAS Catalog bulk, p<1e-7 | 7,889 | 7,781 | main-quality |
| BMI | exposure | EWAS Catalog bulk, p<1e-7 | 24,190 | 23,816 | main-quality |
| age | aging | EWAS Catalog bulk, p<1e-7 | 2,194 | 2,169 | main-quality |
| colorectal cancer | disease | EWAS Atlas curated (no p cutoff) | 7,830 | 7,708 | main-quality |
| prostate cancer | disease | EWAS Atlas curated | 7,663 | 7,560 | main-quality |
| breast cancer | disease | EWAS Catalog bulk | 502 | 498 | main-quality |
| lung cancer | disease | EWAS Catalog bulk | 242 | 240 | main-quality, non_independent (93% in smoking) |
| rheumatoid arthritis | disease | EWAS Catalog bulk | 238 | 236 | main-quality, non_independent (62% in age) |
| SLE | disease | Lanata 2022, Suppl. Table S3 (single study) | 139 | 134 | exploratory |
| type 2 diabetes | disease | Cardona 2022, ESM Table 5 | 71 | 68 | insufficient, non_independent (92% in BMI) |
| CHD | disease | Agha 2019, Tables 2+3 | 49 | 49 | insufficient |
| ovarian cancer | disease | EWAS Atlas curated | 28 | 27 | insufficient |
| schizophrenia | disease | EWAS Catalog bulk | 28 | 28 | insufficient |
| Horvath clock | aging clock | Horvath 2013 via biolearn | 344 | 336 | main-quality |
| PhenoAge clock | aging clock | Levine 2018 via biolearn | 498 | 492 | main-quality |
| Hannum clock | aging clock | Hannum 2013 via biolearn | 64 | 63 | insufficient |

Counts of the probe-known column were read from existing files (set npy x covariates table) for this draft; the frozen tiers are recomputed from the frozen backgrounds. Not available in the repo: alcohol, sex, GrimAge, tissue-specific EWAS; adding any requires a frozen source before the freeze.
Version caveat: download dates of EWAS Catalog / Atlas are not recorded (UNVERIFIED). Clocks form a separate aging-clock family (not EWAS hits per se: they are selected predictors of age).

Tier rule (decision 4): matched positives >= 200 main-quality; 100-199 exploratory; < 100 insufficient. Computed on the frozen MAIN matching, embedding-free. Exploratory and insufficient tasks appear in supplementary only and never become evidence because of an embedding's result.

Overlap rule (a priori, deterministic, embedding-free): a set is `non_independent` if >= 50% of its universe members are in a larger set. Flagged: lung (225/242 = 93% in smoking), T2D (65/71 = 92% in BMI), RA (147/238 = 62% in age). Handling: report both, tag the smaller set with its partner, count the pair once in family summaries.
Set-difference sensitivity only if the difference has >= 100 members: lung - smoking 17, T2D - BMI 6, RA - age 91, so none qualifies (the tag is the only handling). Other overlaps (smoking-BMI 2,082; colorectal-prostate 676; clocks 6-40) are reported, not flagged.

## 3. Background (shared, embedding-free)
- ONE background per task for ALL arms, frozen as pair files before any embedding is loaded. 1:1 (k = 1) primary. Candidate backgrounds exclude members of ANY panel set (union 46,092 CpGs, 11.3% of the universe) so that "unlisted" is as clean as the data allow (negatives are still unlisted, not proven unassociated).
- MAIN matching = technical covariates only: exact chromosome x Infinium type (I/II). Justification: chromosome keeps each pair in one CV fold and one bootstrap block; probe type is a measurement-design covariate that affects EWAS detectability and is not an embedding input. Deliberately NOT matched in main: CpG context, gene region, GC, CpG density, TSS distance, any ENCODE-derived value: each is an explicit or implicit input of a compared representation (sequence models see GC/density/context; functional/candidate tracks encode promoter/island/breadth), so matching on them would neutralise the arms unequally.
- Deterministic draw: strata sorted by (chromosome, type), positives sorted, `default_rng(17)`, without replacement; positives without a known probe type are dropped and counted; no cap on pairs (Exp5 capped at 2,000) [open question]. 10 further frozen draws (seeds 18-27) only as a stability SD (supplementary); draws are not replicates.
- Zero rows: the 849 all-zero candidate rows are kept (same backgrounds); per-task counts of zero rows among positives/backgrounds are reported.
- Covariates table: `data/derived/bioval_v2/covariates/universe_covariates.parquet` (context, cpg_density, gc_content, tss_dist, probe_type; 98.4% probe coverage). Its row order differs from `outputs/encode_atlas_v1/loci.parquet`: always join by `cpg_idx`.

SENSITIVITY S1 (context-matched): within exact chromosome x probe type x island/shore/shelf/open-sea, nearest neighbour without replacement on standardized (GC, CpG density hg38, log10 TSS distance) with caliper 0.25 SD; unmatched positives dropped and counted; its own N/tier reported (main tier unchanged).
S2 (diagnostic only): S1 + core-breadth decile: circular for the candidate and functional, never a conclusion. S3 (beta-variance matching) not proposed by default (uses TCGA-discovery patient betas on the label side).
Reading: per task and arm report AUROC(main), AUROC(S1) and the difference; sign agreement of candidate - CpGPT between main and S1; no pre-set attenuation threshold. Sequence arms are also neutralised by S1 (they encode context from sequence).

## 4. Probe
- Arms: candidate 256 D, CpGPT-large 512 D, DeepCpG 128 D (native), plus functional_annotations_pca (campaign full SVD-256) in supplementary. No padding or re-compaction.
- Outer folds: the frozen chromosome->fold map `chromosome_blocked_folds(22 chromosomes, 5, seed 17)` (as in bioval-v2); a pair is always in one fold. Preprocessing: `StandardScaler` fitted on the outer-train fold only; nothing else.
- Classifier family: logistic regression, L2, lbfgs, max_iter 2000, C in {1e-4 ... 1e2} (7 values), selected by inner leave-one-frozen-fold-out CV on the outer-train folds only (pooled inner AUROC; ties -> smaller C); the same grid and rule for every arm. (Differs from Exp5's fixed C = 1: a fixed C favours arms whose scale suits C = 1.)
- Scores: pooled out-of-fold, stored; AUROC (pooled across pairs), matched-pair concordance (share of pairs with score(EWAS) > score(background), ties 0.5), AUPRC (sensitivity only; prevalence 0.5 under 1:1 is artificial).

## 5. Uncertainty and contrasts
- Paired chromosome-block bootstrap: ONE shared draw table (B = 1000, seed 17, 22 blocks, `default_rng(17).integers(0, 22, (1000, 22))`), same resampled chromosomes for every arm, task and metric (as bioval-v2); 95% percentile CIs; raw two-sided p with (1+k)/(1+B) floor. Coarse by construction (22 blocks); small sets have blocks with 0-2 pairs.
- Contrasts (candidate minus comparator, positive = candidate better): primary vs CpGPT-512, secondary vs DeepCpG-128, supplementary vs functional_256 (legacy). Per task; no pooling across tasks; NO aggregate score.
- Family-level descriptive summary (main-quality tasks only; exploratory listed separately; non_independent counted with its partner): per family x contrast x metric the number of tasks with CI > 0, CI < 0, CI containing 0. Families: disease/pathology, exposure (smoking, BMI), aging (age EWAS), aging clocks.
- Multiplicity: descriptive (CI and raw p); Holm within family x contrast for AUROC given as a supplementary column; no claim is gated on it [open question].

## 6. Freeze state machine and gate
DRAFT -> DESIGN_ARTIFACTS_BUILT (set list, tiers, backgrounds, S1 pairs, fold map, draw table, all hashed; NO embedding read) -> FROZEN (registration doc committed clean, tag `ewas-v2-protocol-freeze-v1`) -> EXECUTED -> REPORTED.
Gate conditions at FROZEN and again at EXECUTED: candidate frozen manifest status `candidate_frozen`; candidate store sha256 equals the manifest (`fa3a8163...`); CpGPT/DeepCpG/functional store sha256 recorded; set npy, pair-file, fold-map and draw-table sha256 recorded; registration committed clean; tree clean in `src/`, `scripts/ewas_v2/`, `configs/ewas_v2/`; no write outside `outputs/ewas_v2/`; `loyfer_failure_audit` untouched. "No tuning on results": after EXECUTED, no design parameter changes; deviations are declared amendments (as in the bioval-v2 registrations).
Ordering risk: the exp5 completion will show candidate results on the same sets. To avoid design tuning, freeze the design-only part (sets, matching, tiers, metrics, contrasts) BEFORE inspecting those results, or state in the registration that they were seen.

## 7. Outputs
`outputs/ewas_v2/<freeze_tag>/{sets, backgrounds, folds, arms/<arm>/<task>/oof.parquet, endpoints/<task>.json, contrasts.csv, family_summary.csv, sensitivity_S1/, manifest.json}`; one JSON per task with N, tier, flags, zero-row counts, point estimates, CIs for each arm; contrasts with CI and raw p.

## 8. Reporting plan
Main: per-task table (AUROC, concordance with CIs; arms side by side) for main-quality tasks, contrasts with CIs, family counts, S1 next to main. Supplementary: exploratory and insufficient tasks (labelled), functional legacy, AUPRC, draw-stability SD, S2 diagnostic, non_independent tags. Interpretation sentences must name the design (technical-matched vs context-matched).

## 9. Caveats (to be stated in the paper)
EWAS probes come from array content biased towards promoters/CpG islands (shared by positives and backgrounds; the main matching does not control it, S1 does); EWAS cohort overlap with GEO/TCGA benchmark cohorts UNVERIFIED (RA includes GSE42861, SZ mirrors GSE147221 by design); CpGPT is methylation-pretrained (direct methylation supervision) so a CpGPT advantage on blood EWAS may reflect that; negatives are unlisted, not proven unassociated; sets overlap (see 2); EWAS Catalog sets use p<1e-7 while Atlas sets use no uniform cutoff; clocks are selected predictors, not association hits; small N tasks have coarse CIs; 22-block bootstrap is coarse.

## 10. Open questions for the user
1. Is matching on chromosome + Infinium type ONLY acceptable for the main design (with context left to S1)? Add platform presence (450K vs EPIC) or exclusion of cross-reactive/polymorphic probes (no mask file located in the repo)?
2. Background ratio: 1:1 (default, prevents imbalance but makes AUPRC artificial) or k:1 (e.g. 5:1, more stable negatives, AUPRC meaningful)?
3. Cap on pairs for large sets (Exp5 used 2,000; BMI has ~23,800 probe-known positives): no cap (default) or a fixed cap (e.g. 5,000) to equalise precision?
4. Background exclusion of ALL panel-set members (default, union 46,092 CpGs) or only the task's own set (as Exp5)?
5. Clocks: keep Horvath / PhenoAge / Hannum as a separate aging-clock family (default)? Add other clocks (GrimAge weights not public; other biolearn clocks) before the freeze?
6. Overlap rule: containment >= 0.5 tagging only (default), or additionally drop/merge lung, T2D and RA from the main table?
7. Classifier: nested-CV C grid (default) or keep Exp5's fixed C = 1 for direct comparability?
8. S1 definition (island class + GC + CpG density + TSS distance with caliper 0.25 SD) acceptable? Include S2 (ENCODE breadth, circular) as a labelled diagnostic? Include S3 (beta variance from TCGA discovery patients) at all?
9. Multiplicity: purely descriptive (default) or Holm within family x contrast reported as secondary?
10. Bootstrap B = 1000 with 22 blocks (as bioval-v2) or larger B (the CI granularity is limited by the 22 blocks, not B)?
11. Sequencing: freeze the EWAS-v2 design BEFORE the exp5 completion results are inspected (recommended), or after (then declare)?
12. Is the 450K array universe (408,399 CpGs) the right background universe given EPIC-only EWAS hits are outside it (age 300,698 -> 2,194 in universe)? Supplementing the universe is a new representation-coverage problem, not a design choice.
13. Provide or authorise a check of EWAS source cohorts against GEO/TCGA cohorts (currently UNVERIFIED).
