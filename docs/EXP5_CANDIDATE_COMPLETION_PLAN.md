# exp5_candidate_completion: plan and scaffold (PREPARATION ONLY, nothing run)

Status when run: **discovery-campaign completion (exp5 protocol, not preregistered main evidence)**. Decisions of 2026-10-05: item 1(i) (exactly the Exp5
pairs, folds and bootstrap), item 5 (AUROC primary; matched-pair concordance; AUPRC sensitivity only), item 4 (support tiers on N only), item 2 (functional =
campaign `full` SVD-256, supplementary). See `docs/BIOLOGICAL_TASK_INVENTORY_ADDENDUM_2026-10-05.md` for the text of the decisions.

Files created: `scripts/followup/exp5_candidate_completion.py` (scaffold), `configs/exp5_candidate_completion/exp5_candidate_completion.yaml`,
`configs/exp5_candidate_completion/exp5_pins.json` (hash pins), `tests/test_exp5_candidate_completion.py` (synthetic, 9 tests pass).
No existing file was edited.

## 1. How Exp5 works (read from `src/cpg_repr_benchmark/encode_atlas/ewas_robust.py`, `scripts/followup/ewas_robust.py`, `outputs/encode_atlas_v1/followup/exp5/`)

- Universe: `outputs/encode_atlas_v1/loci.parquet` (408,399 CpGs, same order as every store). Positives: members of a `data/bio_annotations/known_sets/*.npy` set
  (16 sets: 13 EWAS Catalog / EWAS Atlas / clock sets evaluable + ovarian, CHD, schizophrenia without matched support), mapped via `encode_many(chr, pos)`.
- Strata: `chromosome : CpG context (argmax dense[:, :4]) : gene region (argmax dense[:, 4:8]) : core-breadth decile (floor(dense[:, 22]*10))`, where `dense` is the
  functional feature store `data/derived/functional_annotations/reference_functional_features_tcga_array_all_v1_cpgidx_fixed.h5`. The `probe` variant appends the Infinium type (I/II) and restricts to loci with a known type
  (type codes stored in `probe_type_annotation.parquet`, 92.6% coverage).
- Pairs (`match_pairs`): for each stratum in sorted order draw `min(#pos, #neg)` positives and negatives without replacement (`default_rng(seed)`), negatives = eligible CpGs not in the set; then subsample to <= 2,000
  pairs (same rng). 20 independent draws, seeds 17001..17020 (`mask_seed` 17001 + i). Ovarian / CHD / schizophrenia have 28 / 49 / 28 pairs (<100 rows) and are `insufficient_matched_support`.
- Probe (`pooled_oof`): rows sorted by CpG row index; `StandardScaler` fit in the train fold only + `LogisticRegression(C=1, max_iter=2000)`; `GroupKFold(5)` over chromosomes (deterministic given y and chromosome);
  pooled out-of-fold scores; pooled AUROC (`fast_auc`, all positives vs all negatives). Embeddings are cast to float32 for every arm.
- Arms (`REPRESENTATIONS`, module-level dict): `functional_256` (campaign `full`), `cpgpt_large_native512`, `deepcpg_hcc_native128`, `cpgpt_large_svd256`, `deepcpg_hcc_pad256`, `context_only_256`.
- Variants: `base` and `probe`. Outputs: `oof_predictions.parquet` (5,785,200 rows: variant, set, match_seed, cpg_idx, chromosome, pair_id, y_true, fold, representation, oof_score[float32]),
  `summary_auc_delta.csv`, `matching_summary_per_seed.csv`, `evaluable_sets.csv`, `probe_type_balance.csv`, `probe_type_annotation.parquet`, `protocol.json`.
- Bootstrap (`_stage2`): per (variant, set) one `default_rng(20260929 + k)` with `k` = index of the (variant, set) in the sorted list of EVALUABLE entries; B = 10,000; each replicate draws one matching seed uniformly,
  then resamples that seed's pairs with replacement within chromosome (one index vector applied to every arm: exact paired bootstrap); point estimate = mean over matching seeds; percentile 95% CI; two-sided p = 2 min(P(d<=0), P(d>=0)) with a 1/(B+1)
  floor; BH across sets within variant x comparison. Exp5 deltas are `functional_256 - X`. Matching draws are not independent biological replicates (stated in the code).

### What "exactly the pairs, folds and bootstrap" means in files
Pairs and folds are not separate files: they are a deterministic function of (known-set npy, functional dense features, seeds 17001..17020, probe-type codes) and are stored row by row in `oof_predictions.parquet`
(`cpg_idx`, `pair_id`, `match_seed`, `fold`). The draws of the bootstrap are fixed by the RNG seeds above. The scaffold therefore (i) pins the hashes of all inputs and exp5 outputs, and (ii) checks, for
every (variant, set, seed), that the candidate's regenerated `cpg_idx`, `pair_id`, `y_true`, `fold`, `chromosome` arrays are IDENTICAL to the stored comparator rows; any difference aborts the run.

Hashes (sha256; also in `configs/exp5_candidate_completion/exp5_pins.json`, along with the 16 known-set files):

| file | sha256 |
|---|---|
| exp5/oof_predictions.parquet (pairs, folds, scores) | ba9ee74b1a49944a377a1494ecaaa6430404162fbed5d0a10f7edc8a5806cf81 |
| exp5/protocol.json (seeds, B, classifier, folds) | 201e01c3b94e40ff65f2da8966493c5a61e0c9be0fa0abd4c989d4f7dab9aae1 |
| exp5/matching_summary_per_seed.csv (draws: n_pairs, probe-type counts) | a64c5b3bb54bf880b59c5bd20d02805527732b6504f4faf3d457f34fb7ccb33b |
| exp5/evaluable_sets.csv | 7c2504ba104cdb1a085ca38978b2085f85b4a58a367c3382ea946cd7decc5350 |
| exp5/summary_auc_delta.csv | ec213c25e26888c93578976f219428e27e14e75c2ab0e7c1ed00bfbee23e5662 |
| exp5/probe_type_annotation.parquet | f26a3338524ce96e1d40b08c66c042be9d3fa7d3eed5b32830919c9e37207d96 |
| exp5/probe_type_balance.csv | d4484a0f69b902f9055e1f6229dbb5dab4f7ed4eeb87234dcaf0020fba68ef8b |
| configs/frozen/regulatory_histone_dnase_v1.json | 0a6d9c4f2b6f2a05f47669489705e9cad5dd0d17e813f2d9b1383fe959277a70 |
| candidate store `regulatory_histone_dnase__global_svd256__discovery_chr1_19.h5` | fa3a816338ce2c2ce5669b4be6f5334a44470e2b149395901168dbe673a9fe1e (equals the frozen manifest; verified by `gate`) |

## 2. Can the candidate be added without editing code?
- The CLI `scripts/followup/ewas_robust.py` and `run()` cannot take an arm list (arms are the module-level `REPRESENTATIONS`; `run()` always re-trains all six arms, writes into the given outdir and would
  compare against `functional_256` only). Running it unchanged would re-train five other arms (hours) and produce no candidate comparisons.
- The internals can be reused read-only: `_stage1` (matching + OOF), `match_pairs`, `pooled_oof`, `fast_auc`, `_strata`, `_index_strata`, `_load_embeddings`, `_G`. A process-local rebinding
  `ewas_robust.REPRESENTATIONS = {candidate: store path}` (the store loader accepts a plain path) makes `_stage1` train ONLY the candidate with the identical pairs and folds. The fork-pool workers inherit the rebinding.
  The probe-type codes come from the stored `probe_type_annotation.parquet` (hash-pinned) so the EPICv2 manifest outside the repo is not needed.
- Decision: NEW script `scripts/followup/exp5_candidate_completion.py` (done). Proposed (NOT applied) upstream improvement: let `run()` take `representations` / `comparisons` arguments, which would remove the monkey-patch.
- Comparators (CpGPT-512, DeepCpG-128, functional_256) are NOT re-trained: their stored OOF scores enter the same paired bootstrap, after the row-level identity check. Candidate scores are rounded to float32 before the bootstrap,
  like the stored scores. Consequence: AUROC from the float32 parquet differs from the stored float64-based AUROC by <= 3.5e-5 (verified over all 156 cells), so the stored exp5 CIs are reproduced only up to that rounding.
- Float formats: the candidate store is float32; CpGPT, DeepCpG and functional stores are float16 (all cast to float32 for the probe, as in Exp5). The candidate therefore carries more precision than its comparators (minor; note in the report).
- Rows: 849 candidate rows are all-zero (kept; same pair draws). Chromosomes 20-22 are out-of-fit for the label-free SVD (transformed only); the SVD fit (chr1-19) is label-free and transductive, as for the other compacted arms (`compaction_scope` in the Exp5 protocol).

## 3. Scaffold: `scripts/followup/exp5_candidate_completion.py`
Subcommands: `gate`, `regression`, `stage2-selfcheck`, `run` (none executed). Pure functions (`matched_pair_concordance`, `stage2_extended`, `per_seed_from_oof`, `verify_same_pairs`, `stage2_k_index`, `bh_by_comparison`,
`support_tier`) are unit-tested on synthetic data. In particular `stage2_extended` is asserted equal to `ewas_robust._stage2` (AUROC estimates, CIs, delta CI, p) on synthetic input and its RNG stream is asserted independent of the arm list.

Pre-run gate (`check_gate`; every item must pass, no writes): (1) frozen manifest status `candidate_frozen` and store sha256 equal to the manifest and to the pins file; (2) the 7 exp5 output files and 16 known-set files equal the pinned hashes;
(3) `docs/EXP5_CANDIDATE_COMPLETION_REGISTRATION.md` exists, is tracked and has no uncommitted changes (the registration doc is NOT written yet; it must be committed before the run); (4) working tree clean in `src/`, this script, its configs and `configs/frozen/`;
(5) out dir is under `outputs/exp5_candidate_completion/`, empty or absent, and never inside the exp5 results directory; (6) git HEAD and exp5 mtimes recorded. After the run the exp5 file hashes are recomputed and must equal the pre-run snapshot (assert). The `loyfer_failure_audit` path is rejected by `guard`.
`gate --skip-registration-check` (dry gate before the registration exists) was run: PASS (frozen/store/exp5/known-set hashes OK).

Arms: candidate trained; comparators read from stored OOF (no re-training). Comparisons (delta = candidate - comparator): vs `cpgpt_large_native512`, vs `deepcpg_hcc_native128`, vs `functional_256` (the Exp5 reference). The remaining Exp5 arms (compact CpGPT, pad256, context-only) are not needed.
Sets: the 13 evaluable sets x {base, probe}; ovarian / CHD / schizophrenia are skipped (no matched support, listed in the manifest). Support tier column (decision 4) from the exp5 base matched positives:
main-quality: colorectal, prostate, age, BMI, smoking, breast, PhenoAge, Horvath, lung, RA; exploratory: SLE; insufficient (still reported in supplementary): T2D, Hannum.

Metrics per set/variant/arm: AUROC (mean over 20 draws, between-draw SD, bootstrap CI), matched-pair concordance (same, ties 0.5), AUPRC (point estimate over draws, sensitivity only; no CI). Paired deltas for AUROC and concordance with exp5's bootstrap and BH within variant x comparison x metric.
Concordance is a different statistic from AUROC: it compares each EWAS CpG only with its own matched partner (the natural metric for 1:1 designs).

Outputs (new): `outputs/exp5_candidate_completion/<run_id>/{candidate_oof_predictions.parquet, summary_candidate_completion.csv, run_manifest.json}`; abort marker `ABORT_pairs_or_folds_differ.json` if identity fails.
The registry can ingest the summary with an additional `registry_v2`-style builder step (not done).

## 4. Regression check (planned; not run on real data)
1. `stage2-selfcheck` (no training, bootstrap only): recompute AUROC CIs and `functional - CpGPT / DeepCpG` deltas for 2 small sets (default `base/ewas_catalog_sle`, `base/hannum_clock`) from the stored parquet with `stage2_extended`, same RNG seeds, B = 10,000,
   and compare to `summary_auc_delta.csv` (tolerance 2e-3, expected ~1e-4 from float32 storage; tighten after the first run).
2. `regression` (training of a small subset): re-train CpGPT-512, DeepCpG-128, functional_256 through the same `_stage1` for `base` x {SLE (139 pairs), Hannum (64 pairs)} x seeds 17001-17003 (18 small fits, ~1-2 min plus ~1 min loading) and require: `cpg_idx`, `pair_id`, `y_true`, `fold`, `chromosome` identical to the stored rows
   and `max |oof_score - stored| <= 1e-5`. Optional deep case: `--sets ewas_catalog_smoking --seeds 17001` (2,000 pairs; 3 fits). Failure (e.g. library drift since 2026-09-29) blocks the run until understood.
3. Full run only after (1) and (2) pass and the registration doc is committed.

## 5. Cost, time and launch plan (MEASURED 2026-10-05 on comparators, candidate never fitted)
- Measured: `stage2-selfcheck` (2 small sets, B = 10,000, 3 arms) 15.8 s wall, 0.9 GB RSS; `regression` (18 small fits, 4 workers) 6.5-7.2 s wall, 1.6 GB RSS (mostly store loading); one stage-1 task, 1 arm, 256-d: 0.18 s (2,000 pairs), 0.07 s (502 pairs), 0.04 s (139 pairs), single thread;
  stage 2 for a 2,000-pair set: 2.5 s per B = 1,000 with 3 arms (B = 10,000 with 4 arms about 35-40 s, scaled; small sets a few seconds).
- Extrapolation: stage 1, 520 candidate tasks (10 of 26 cells have 2,000 pairs): about 1-3 CPU-minutes. Stage 2 (26 cells x B = 10,000 x 4 arms x 2 metrics + AUPRC): about 8-12 CPU-minutes (10 big cells ~40 s each), i.e. ~2-3 min wall with 6 workers (jobs are unequal). Loading (parquet comparators 0.5 GB + candidate store 0.42 GB) 20-60 s.
  Expected total wall: ~5 min with 6 workers (earlier estimate of 1-4 h was ~30x too pessimistic). Memory ~2-4 GB total across workers (fork shares); disk < 0.5 GB.
- Launch (after the registration is committed and the gate passes):
  `cd /data2/home/vcivale/projects/methylation/CpGRepresentationBenchmark && PYTHONPATH=$PWD/src nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/miniconda3/envs/cpg-repr-benchmark/bin/python scripts/followup/exp5_candidate_completion.py gate && PYTHONPATH=$PWD/src nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/miniconda3/envs/cpg-repr-benchmark/bin/python scripts/followup/exp5_candidate_completion.py run --out outputs/exp5_candidate_completion/run_001 --workers 6`
- Regression and self-check have been run (see registration Amendments 1-2): VERDICT PASS (Amendment 2 criterion).

## 6. Risks and implications
1. **Matching strata use inputs of the functional store.** Context, gene region and core-breadth decile (dense columns 0-7 and 22) are inputs of `functional_256` and `context_only`, but not of the candidate (`dense_columns = 0`), CpGPT or DeepCpG. The matching therefore neutralises exactly the information the functional arm holds
   as input while leaving it free for the candidate; the candidate's histone/DNase tracks encode promoter/island/breadth information implicitly (not matched out). Expected direction: candidate - functional_256 is biased towards the candidate; candidate vs CpGPT/DeepCpG is exposed to the same matching but those arms have no direct access to the dense inputs. This is why exp5 completion is discovery only and the main
   evidence needs EWAS-v2 (matching on technical covariates only). GC, CpG density, TSS distance are not matched in exp5 (FANTOM5 does match them).
2. Negatives are "unlisted", not unassociated; may belong to other EWAS sets. Sets overlap (lung 93% in smoking, T2D 92% in BMI): tasks are not independent.
3. The candidate sees more numeric precision than float16 comparators; 849 zero rows kept (exp5 pairs may include some; the count of zero-row loci per set should be added to the run manifest: not yet implemented).
4. Library drift since 2026-09-29 (sklearn/numpy) could change fits: covered by the regression check. Float32 storage limits exact reproduction of stored CIs (see 2).
5. Exp5 AUPRC is artificial (prevalence 0.5): sensitivity only. Concordance CI uses the same seed-mixture bootstrap, which mixes matching-draw and pair variability.
6. The 20 matching draws are not independent replicates; p-values/q-values are descriptive. Not preregistered main evidence: no claim stronger than "discovery completion".
7. EWAS cohort overlap with GEO/TCGA remains UNVERIFIED; CpGPT is methylation-pretrained.
8. Report both directions: CpGPT currently wins age / Horvath / PhenoAge in exp5; the candidate's results are shown whatever they are, with tiers and no aggregate.

## 7. Decisions needed from the user
1. Approve using the new script rather than a change to `ewas_robust.run` (no code edit needed).
2. Write and commit `docs/EXP5_CANDIDATE_COMPLETION_REGISTRATION.md` (suggested content: this plan's sections 1-4, tier rule, metric hierarchy, report-both-directions rule); the gate requires it.
3. Accept the stage-2 float32 convention (comparators are read from the stored float32 parquet), or request a re-run of comparators in float64 (adds ~1-2 CPU-hours per arm).
4. Run the regression check (items 4.1 and 4.2) first, and when (the CPU-heavy audit is running).
5. Include the candidate in the `probe` variant as well as `base` (default: both) and the insufficient-tier sets in the supplementary (default: Hannum and T2D reported with the `insufficient` label; ovarian / CHD / schizophrenia skipped).
6. Whether to add AUPRC bootstrap CIs (default: point estimates only).
