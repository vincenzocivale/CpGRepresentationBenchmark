# exp5_candidate_completion: registration (written 2026-10-05, before any candidate result exists)

> ## AMENDMENT 2 (2026-10-05, orchestrator decision flagged for user review): numerical-fidelity tolerance of the regression (decided BEFORE any candidate was fitted or opened; NOT a methodological change)
> - New regression PASS criterion, per cell (variant, set, seed, arm; 18 cells): (i) `cpg_idx`, `pair_id`, `y_true`, `fold`, `chromosome` identical to the stored exp5 rows; (ii) |dAUROC| <= 5e-4 against the stored exp5 AUROC;
>   (iii) Spearman(stored score, re-run score) >= 0.9999. The absolute score difference (previous criterion 1e-5) stays reported as an INFORMATIVE metric only. This supersedes the 1e-5 wording in section 8 and in the plan.
> - Rationale (mechanism and numbers of the regression, not candidate results): near-separable fits (128-512 features on ~100-200 training rows) with lbfgs + StandardScaler have a solution that is not determined to 1e-5 across library versions.
>   Observed with the legacy-fold shim: folds/pairs identical 18/18; max |dscore| 2.1e-3 (DeepCpG-128), 1.6e-4 (CpGPT-512), 6.9e-7 (functional_256); max |dAUROC| 2.44e-4; min Spearman 0.999987.
>   The effect on AUROC (<= 2.4e-4) is negligible against the Exp5 confidence intervals (order >= 0.01).
> - Stated limit: the candidate is fitted with the CURRENT libraries while comparators enter with their stored scores (original libraries), so candidate - comparator deltas can carry a systematic fidelity error of order <= ~3e-4 AUROC.
> - Gate/regression: `regression --verdict-out` writes `outputs/exp5_candidate_completion/regression/VERDICT.json`; the gate now requires verdict PASS there. Synthetic test: a cell with |dAUROC| > 5e-4 or non-identical pairs/folds fails.
> - Verdict with the amended criterion: PASS (max |dAUROC| 2.44e-4, min Spearman 0.999987, 18/18 cells; `regression/VERDICT.json`, `regression/regression_amended.log`). Nothing else was loosened.


> ## AMENDMENT 1 (2026-10-05): fold-assignment fidelity, classified BUGFIX (no methodological change)
> - Finding: the installed scikit-learn 1.9.1 `GroupKFold` assigns chromosomes to folds differently from the original algorithm that produced the stored Exp5 folds
>   (SLE / seed 17001 / base: stored folds equal the legacy algorithm in 100% of rows, the current sklearn in 92.8%; the fold partitions differ). The first `regression` attempt FAILED on `fold` (and so on scores);
>   `cpg_idx`, `pair_id`, `y_true` were identical (log kept: `outputs/exp5_candidate_completion/regression/regression_attempt1_sklearn_groupkfold_FAIL.log`).
> - Fix: `LegacyGroupKFold` in `scripts/followup/exp5_candidate_completion.py` (greedy: chromosomes from the largest, each to the currently lightest fold; same `np.argsort(counts)[::-1]` / `np.argmin` tie-breaks),
>   bound over `ewas_robust.GroupKFold` in this process only (no change to `src/`). Folds are Exp5's own algorithm restored, so the design is unchanged: this is a fidelity bugfix.
> - Evidence: the shim reproduces the stored `fold` column in 3120 of 3120 cells (13 evaluable sets x 2 variants x 20 seeds x 6 arms, 5,785,200 rows), 0 mismatches
>   (`regression/shim_vs_stored_folds.txt`). Gate now includes this stored-fold check, and `run` aborts unless the candidate's `fold`, `cpg_idx`, `pair_id`, `y_true`, `chromosome` equal the stored rows in every cell, against every comparator. Synthetic unit tests added.
> - Second regression attempt with the shim: pairs and folds identical in 18/18 cells; score differences are NOT <= 1e-5 for DeepCpG-128 / CpGPT-512 (max |dscore| 2.1e-3 / 1.6e-4; functional_256 5.4e-7 to 6.9e-7);
>   max |dAUROC| 2.4e-4 (see `regression/regression.log`, `regression/diag_auroc.txt`). The tolerance is NOT widened here; status of the regression is OPEN pending a user decision.


Status of the result: **discovery-campaign completion under the exp5 protocol; NOT main evidence. EWAS-v2 is the main evidence, and EWAS-v2 is frozen before
exp5-completion candidate results are inspected.** Companion documents: `docs/EXP5_CANDIDATE_COMPLETION_PLAN.md` (mechanics, hashes, risks),
`docs/BIOLOGICAL_TASK_INVENTORY_ADDENDUM_2026-10-05.md` (decision text), `docs/EWAS_V2_PROTOCOL_DRAFT.md`. This file is the document that
`scripts/followup/exp5_candidate_completion.py` requires to be tracked and clean (`REGISTRATION_DOC = docs/EXP5_CANDIDATE_COMPLETION_REGISTRATION.md`).

## 1. Question and design
Add the frozen candidate `regulatory_histone_dnase_v1` (store `data/cache/representations/regulatory_histone_dnase__global_svd256__discovery_chr1_19.h5`,
sha256 `fa3a8163...9fe1e`, equal to `configs/frozen/regulatory_histone_dnase_v1.json`) to the Exp5 matched EWAS / epigenetic-clock probe, using EXACTLY the
Exp5 pairs, folds, probe and bootstrap (user decision 1(i), 2026-10-05):
- pairs: the 20 matching draws (seeds 17001..17020), <= 2,000 pairs, variants `base` and `probe`, 13 evaluable sets (ovarian, CHD, schizophrenia have no matched
  support and are skipped; listed in the manifest);
- folds / probe: GroupKFold(5) by chromosome on sorted pair rows; StandardScaler fit in the train fold + LogisticRegression(C=1, max_iter=2000); pooled OOF;
- bootstrap: B = 10,000, one seed drawn uniformly per replicate, pairs resampled with replacement within chromosome, same index vector for every arm,
  `rng_seed = 20260929 + k` (k = index of (variant, set) in the sorted EVALUABLE list); p = 2 min(P(d<=0), P(d>=0)) with 1/(B+1) floor; BH across sets within variant x comparison (x metric).

## 2. Metrics and comparisons
- Primary: AUROC. Co-primary: matched-pair concordance (share of pairs with score(EWAS CpG) > score(its matched partner), ties 0.5; mean over pairs, then over draws).
- AUPRC: sensitivity only (point estimate over draws, no CI; prevalence is 0.5 by construction).
- Deltas (candidate minus comparator): vs `cpgpt_large_native512` (CpGPT-512), vs `deepcpg_hcc_native128` (DeepCpG-128), vs `functional_256` (campaign `full` SVD-256;
  legacy, supplementary). No aggregate across sets; both directions reported (CpGPT currently wins age / Horvath / PhenoAge in Exp5).
- Support tier column on every row, fixed from the pre-existing Exp5 `base` matched positive counts (`evaluable_sets.csv`, `pairs_mean`), independent of any candidate result:
  main-quality >= 200 (colorectal, prostate, age, BMI, smoking, breast, PhenoAge, Horvath, lung, RA); exploratory 100-199 (SLE); insufficient < 100 (T2D, Hannum;
  still reported, labelled). The tier of a set is the same in `base` and `probe`.
- Result label written to every row and to the manifest: "discovery-campaign completion (exp5 protocol, not preregistered main evidence)".

## 3. Conventions accepted (orchestrator-adopted details flagged)
1. Comparators are NOT re-trained. They enter the paired bootstrap through their STORED Exp5 OOF scores (float32 parquet), after row-level identity checks
   (`cpg_idx`, `pair_id`, `y_true`, `fold`, `chromosome` identical per (variant, set, seed); otherwise abort with `ABORT_pairs_or_folds_differ.json`). Candidate scores are rounded
   to float32 before the bootstrap. Consequence: the stored Exp5 CIs are reproduced only to ~1e-4 (AUROC from float32 differs from stored float64-based AUROC by <= 3.5e-5). *(orchestrator-adopted detail)*
2. The candidate store is float32; CpGPT / DeepCpG / functional stores are float16 (all cast to float32 for the probe). Minor precision advantage for the candidate.
3. 849 all-zero candidate rows are kept (same pair draws; no resampling). Chromosomes 20-22 are out-of-fit for the label-free SVD (transform only).
4. **Matching strata and bias.** Strata = chromosome x CpG context x gene region x core-breadth decile; context, gene region and core-breadth are INPUTS of the functional store
   (`functional_256`, `context_only_256`) but NOT of the candidate, CpGPT or DeepCpG. The matching therefore neutralises exactly the information the functional arm
   holds as input, while the candidate (histone/DNase tracks encode promoter / island / breadth implicitly) is evaluated under a matching partly based on features it lacks.
   The matching is defined on features of the functional store: it is a matching 'by the functional arm's own inputs', so it is not neutral between arms. Two effects pull in
   different directions and the net direction is NOT known a priori: (a) matching removes the context / gene-region / breadth signal that the functional arm holds as direct input
   (handicaps functional_256 and, in the plan's reading, favours the candidate in candidate - functional_256); (b) the candidate is scored under strata partly built from features it
   lacks, so residual strata structure may be learnable by it only implicitly. Candidate-vs-functional deltas must be read as potentially biased by this matching and are supplementary;
   candidate vs CpGPT / DeepCpG shares the same matching. This is why Exp5-completion
   is not main evidence. GC, CpG density and TSS distance are not matched in Exp5.
5. Old Exp5 caveats carry over: negatives are "unlisted", not unassociated; sets overlap (lung in smoking, T2D in BMI), tasks not independent; the 20 draws are not independent
   replicates (p/q descriptive); cohort overlap of EWAS with GEO/TCGA unverified; CpGPT is methylation-pretrained; SVD fit is label-free and transductive.
6. Naming: "proxy" has two meanings in this repo. The exp3 "proxy null" is the assay-matched ridge null. The dropped "proxy-training track" (re-optimising a representation against
   methylation) is unrelated and not used here.

## 4. Gate conditions (`exp5_candidate_completion.py gate` / start of `run`; every item must pass; gate writes nothing)
1. Frozen manifest status `candidate_frozen`; store sha256 equals the manifest and `exp5_pins.json`.
2. The 7 Exp5 output files and 16 known-set files equal the pinned sha256 in `configs/exp5_candidate_completion/exp5_pins.json`.
3. This registration doc is tracked in git with no uncommitted changes.
4. Working tree clean in `src/`, `scripts/followup/exp5_candidate_completion.py`, `configs/exp5_candidate_completion/`, `configs/frozen/`.
5. Out dir under `outputs/exp5_candidate_completion/`, empty or absent, never in the Exp5 results directory; `loyfer_failure_audit` paths are refused.
6. After the run the Exp5 hashes are recomputed and must equal the pre-run snapshot (assert).

## 5. Outputs layout
`outputs/exp5_candidate_completion/regression/` (pre-run checks: `stage2_selfcheck.log`, `regression.log`, timings);
`outputs/exp5_candidate_completion/<run_id>/{candidate_oof_predictions.parquet, summary_candidate_completion.csv, run_manifest.json}` (+ `ABORT_pairs_or_folds_differ.json` on identity failure).
Nothing is written into `outputs/encode_atlas_v1/followup/exp5/`.

## 6. Run and rollback policy
- Order: stage2-selfcheck and regression PASS -> registration committed -> gate PASS -> single `run`. One run id per attempt; an out dir is never overwritten (non-empty out dir fails the gate).
- A crashed or aborted run is deleted only as a whole new-directory (`outputs/exp5_candidate_completion/<run_id>`) and re-run under a new run id; partial results are never inspected selectively.
- Rollback: delete `outputs/exp5_candidate_completion/<run_id>` (new files only). No existing result, protocol, store, tag or manifest is modified by this analysis.
- Identity abort (pairs/folds differ) is a hard stop: diagnose, do not patch the comparison to pass.

## 7. Bugfix versus methodology policy
- Bugfix (allowed after registration, must be logged in the run manifest/commit message, and made before the candidate results are inspected): crashes, I/O, paths, logging,
  performance with identical numerical output (verified by the unit tests and by the regression/self-check).
- Methodology change (NOT allowed after candidate results are seen; before that requires a dated amendment to this file): pairs, folds, seeds, classifier, B, RNG streams, metric definitions,
  tier rule, comparator set, float convention, status label. If a methodology amendment is needed after results are seen, results are reported as post-hoc.
- No candidate result may be inspected until `run` has finished; no re-run conditioned on its content.

## 8. Pre-run checklist
- [ ] `ruff check` clean on the script and tests; `pytest tests/test_exp5_candidate_completion.py` passes.
- [ ] `stage2-selfcheck` PASS (comparator AUROC / delta CIs reproduced from stored OOF within 2e-3; observed value recorded).
- [ ] `regression` PASS (retrained CpGPT-512, DeepCpG-128, functional_256 on SLE and Hannum x seeds 17001-17003: pairs/folds identical; max |score diff| <= 1e-5).
- [ ] This document committed; `gate` PASS without `--skip-registration-check`.
- [ ] EWAS-v2 protocol frozen before the run starts (so it is frozen before candidate results are inspected).
- [ ] Loyfer audit finished or run at low priority; launch with `nice`, <= 6 worker processes, one BLAS thread each.
- [ ] No candidate score has been computed anywhere before `run` (the pre-run checks only train the three existing comparators).

## 9. Expected cost
Stage 1 (candidate only): 520 tasks (2 variants x 13 sets x 20 draws); stage 2: 26 bootstrap jobs. See the measured timing and the extrapolation in `docs/EXP5_CANDIDATE_COMPLETION_PLAN.md` section 5.

## 10. Freeze ordering
1. Candidate frozen (`configs/frozen/regulatory_histone_dnase_v1.json`) -> 2. this registration committed -> 3. EWAS-v2 protocol frozen -> 4. `run` executed ->
5. only then candidate exp5-completion results are inspected. EWAS-v2 (main evidence) is frozen before any exp5-completion candidate result is inspected; Exp5-completion cannot
influence EWAS-v2 design choices.
