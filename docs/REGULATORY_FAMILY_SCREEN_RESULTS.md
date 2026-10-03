# Regulatory family screen: results

Screening only. Validation patients only (918 patients, `evaluation.require_patient_view: validation`; test patients never read), 50% masking,
global SVD-256 embeddings fit on `discovery_chr1_19`, one decoder seed (17). Criterion and protocol: `docs/REGULATORY_FAMILY_SCREEN.md` (fixed before running).
Reproduce: `python scripts/analyze_regulatory_family_screen.py` (tables in `outputs/regulatory_family_screen_v1/analysis/`).

Caveats that apply to everything below:
- Deviation from the ENCODE campaign: batch 32 and 8 loader workers (campaign: 8 and 0), lr/wd/epochs not rescaled. Absolute MSE is NOT comparable with campaign numbers; only the within-screen paired comparison is meaningful.
- Single decoder seed: seed-to-seed variance is not estimated. The bootstrap (patient x 1 Mb block, 2000 replicates, seed 17) covers patient and locus-block sampling only, so the CIs are optimistic about reproducibility.
- Validation patients are used twice: `best.pt` selection (50% validation MSE) and the reported score. The bias is small (best epoch vs last epoch differ by <= 1e-5 MSE) and shared across arms, but the numbers are not untouched held-out estimates.
- Nothing here is promoted to a final representation.

## Per-arm results (validation, 50% masking, `best.pt`)

| Arm | Tracks | MSE | MAE | MAS-PCC | MAC-PCC | Best epoch (0-based, of 30) | Last-epoch MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| regulatory_histone | 1,959 | 0.016925 | 0.081786 | 0.93623 | 0.30297 | 28 | 0.016935 |
| regulatory_histone_dnase | 2,492 | 0.016680 | 0.080397 | 0.93704 | 0.31645 | 29 | 0.016680 |
| regulatory_histone_tf | 3,618 | 0.016798 | 0.081035 | 0.93670 | 0.30963 | 28 | 0.016807 |
| regulatory_clean | 4,151 | 0.016633 | 0.080402 | 0.93722 | 0.31451 | 29 | 0.016633 |

Prior-only MSE (empirical train-patient prior, identical for all arms, verified from `prior_prediction`): 0.023642. Skill vs prior: 0.2841 / 0.2945 / 0.2895 / 0.2964 (histone / +DNase / +TF / clean).

Training time (one A100, sequential; from run-directory file timestamps): training about 1,335 / 1,308 / 1,354 / 1,342 s (22 to 23 min); training plus 50% evaluation 1,397 / 1,377 / 1,427 / 1,407 s; whole run including embedding/dataset load about 1,800 s each.
Totals: training 5,339 s (1.48 h), training plus evaluation 5,608 s (1.56 h), wall-clock 7,203 s (2.0 h). Cost is essentially independent of the number of tracks (the run is loader bound; the embedding is a fixed 256-d store). The extra cost of the larger sets is only the offline SVD fit (Clean: 976 s).

## Paired contrasts (2000 replicates, seed 17; delta = alternative minus reference, negative favours the alternative)

Pairing check: sample_index, target_matrix_column (hence patients, panel loci and mask), panel_repeat, targets and prior_prediction are bit-identical across the four arms (918 patients x 1024 masked loci = 940,032 pairs, 2,687 blocks); `paired_bootstrap` additionally enforces one-to-one pairing.

| Reference -> alternative | dMSE | Rel. dMSE | 95% CI dMSE | 95% CI rel. | Frac. replicates favouring alt | dMAE (rel.) [95% CI] |
| --- | ---: | ---: | --- | --- | ---: | --- |
| histone -> +DNase | -0.000245 | -1.45% | [-0.000290, -0.000201] | [-1.71%, -1.19%] | 1.000 | -0.001389 (-1.70%) [-0.001517, -0.001264] |
| histone -> +TF | -0.000127 | -0.75% | [-0.000155, -0.000099] | [-0.92%, -0.59%] | 1.000 | -0.000751 (-0.92%) [-0.000825, -0.000679] |
| histone -> clean | -0.000292 | -1.72% | [-0.000339, -0.000243] | [-2.00%, -1.45%] | 1.000 | -0.001384 (-1.69%) [-0.001515, -0.001260] |
| +DNase -> clean (context) | -0.000047 | -0.28% | [-0.000073, -0.000020] | [-0.44%, -0.12%] | 1.000 | +0.000005 (+0.01%) [-0.000064, +0.000072] |
| +TF -> clean (context) | -0.000165 | -0.98% | [-0.000210, -0.000120] | [-1.24%, -0.72%] | 1.000 | -0.000633 (-0.78%) [-0.000753, -0.000512] |
| +DNase -> +TF (context) | +0.000118 | +0.71% | [+0.000072, +0.000165] | [+0.43%, +0.99%] | 0.000 | +0.000637 (+0.79%) [+0.000509, +0.000761] |

## Criterion (applied literally: rel. MSE <= -1%, MSE CI entirely below 0, MAE same sign)

| Arm | Rel. dMSE vs histone | >= 1% | MSE CI < 0 | MAE same sign | Passes |
| --- | ---: | :-: | :-: | :-: | :-: |
| histone_dnase | -1.45% | yes | yes | yes | PASS |
| histone_tf | -0.75% | no | yes | yes | FAIL (direction is consistent, magnitude below the 1% threshold) |
| clean | -1.72% | yes | yes | yes | PASS |

Clean's advantage over histone_dnase is small (-0.28% MSE) and MAE-neutral, so most of the clean gain is the DNase part; TF alone is weaker than DNase alone despite adding three times as many tracks (1,659 vs 533).

## Diagnostics (descriptive only; CSR track store only, no `/dense`, no methylation beyond existing predictions/targets)

(a) All-zero embedding rows (identical to loci with no track overlapping in the arm's feature set; verified): universe 1,166 / 849 / 971 / 733 loci of 408,399 (histone / DNase / TF / clean), of which 1,000 / 732 / 838 / 636 are in the evaluation panel (2,521 / 1,848 / 2,120 / 1,610 of 940,032 pairs, 0.2%). They have lower MSE than average (0.0137 / 0.0132 / 0.0137 / 0.0134 vs 0.0169 / 0.0167 / 0.0168 / 0.0166 for non-zero rows), so these loci are not hard cases; the gain on them is not distinguishable from 0 (CIs include 0, few pairs). The overall gain is carried by non-zero rows (-1.45% / -0.75% / -1.72%, CIs below 0). Adding tracks reduces the number of uncovered loci by 17% to 37%, which is not what drives the gain.

(b) Quartiles of track coverage (`quartile_definitions.csv`, `diagnostic_strata_contrasts.csv`). Rel. dMSE vs histone by quartile of the arm's own track count (Q1 lowest to Q4 highest):

| Arm | Q1 | Q2 | Q3 | Q4 |
| --- | ---: | ---: | ---: | ---: |
| +DNase | -0.83% | -2.12% | -1.53% | -0.83% |
| +TF | -0.23% (CI touches 0) | -0.90% | -0.93% | -1.18% |
| clean | -0.93% | -2.27% | -1.93% | -1.46% |

All 12 cells are negative with CIs below 0 (except TF Q1, upper bound 0.000000). The same holds when stratifying by histone-track count alone, a stratifier common to all arms (all negative, CIs below 0). The gain is not confined to dense-peak loci: it is largest at mid density (Q2-Q3) for DNase/clean and grows with density only for TF. Absolute MSE falls strongly with density (Q4 about 0.006 vs 0.019 to 0.024 for Q1 to Q3: CpG-island/dense loci are easier), so the quartile MSEs are confounded with methylation variance and only the paired deltas are interpretable. Mean track count: 306 / 372 / 388 / 454 per locus.

(c) In-fit (chr1-19; 893,935 pairs, 348,755 loci) vs out-of-fit (chr20-22; 46,097 pairs, 17,836 loci), rel. dMSE vs histone:

| Arm | In-fit | Out-of-fit |
| --- | ---: | ---: |
| +DNase | -1.47% [CI < 0] | -0.98% (CI includes 0) |
| +TF | -0.76% [CI < 0] | -0.54% (CI includes 0) |
| clean | -1.74% [CI < 0] | -1.26% [CI < 0] |

Absolute MSE on chr20-22 is lower than on chr1-19 for all arms (0.0149 to 0.0151 vs 0.0167 to 0.0170; different loci, not comparable as a quality measure). The out-of-fit gain has the same sign and similar magnitude for all arms (only clean's out-of-fit CI excludes 0), i.e. no evidence that the gain depends on the embedding having been fit on the locus; with 5% of the pairs the out-of-fit CIs are wide and the effect cannot be called equal either.

## Interpretation: what is and is not supported

Supported (single seed, validation, 50% masking):
- All three additions reduce validation MSE and MAE versus histone alone, in every paired replicate, with consistent MAS-PCC and MAC-PCC improvements (MAC-PCC +0.007 TF, +0.014 DNase, +0.012 clean).
- DNase and clean pass the pre-fixed screen (about -1.5% and -1.7% MSE). TF alone does not (-0.75%, below the 1% threshold although the direction is consistent).
- DNase carries more information per track than TF: +DNase beats +TF by 0.71% MSE with a CI excluding 0. Clean is only 0.28% better than DNase, and not better in MAE.
- The gains are broadly distributed over coverage quartiles, are present on loci with non-zero embedding, and have the same sign on the chr20-22 loci that the embedding was not fit on.

Not supported / not shown:
- "Substantial" information beyond histone: the effects are small in absolute terms (about 1 to 2% of MSE, about 0.0003 in MSE against a prior-only MSE of 0.0236 and a model MSE of 0.0169). The additions close a small fraction of the model-to-prior gap (skill vs prior 0.284 -> 0.296 at best).
- Robustness to decoder seed: untested. Model-to-model variability from decoder initialisation could be of the same order as the TF effect and a sizeable fraction of the DNase effect; the bootstrap CIs do not include it.
- Any statement at other masking fractions, on test patients, on external data, or on a final ranking of representations.
- Causal attribution (DNase vs TF vs a mere increase of input dimension before SVD): not tested; clean is not an additive combination (SVD-256 is a fixed budget shared by all tracks).

Flags:
- Best epoch is 28 or 29 of 0..29 for every arm and DNase/clean end exactly at the best epoch: validation MSE was still decreasing at the end of the 30-epoch budget (under-training relative to convergence). The budget is fixed across arms by design, but the ranking could shift if arms differ in convergence speed; the arms with slightly lower MSE are not shown to have converged either.
- The differences are small relative to what one decoder seed could change; treat them as a go/no-go for a multi-seed confirmation, not as a result.
- Batch 32 (about 230 steps/epoch) rather than 8 (campaign) means fewer optimizer steps; this likely contributes to the under-training flag.
- Bootstrap fraction favouring is 1.000 for all histone contrasts at 2000 replicates, i.e. the sampling uncertainty over patients and blocks is small; the dominant uncertainty is seed variance.

Suggested next step if pursued: confirmation of histone vs histone+DNase vs clean with at least 3 decoder seeds (and, if budget allows, a longer schedule), since TF alone failed the screen.

## Decision (closing record of feature engineering)

Evidence: this screen (1 seed, batch 32, 30 epochs, validation only) and the 3-seed confirmation (`docs/REGULATORY_CONFIRMATION_PROTOCOL.md`,
`docs/REGULATORY_CONFIRMATION_RESULTS.md`; batch 8, 80 epochs, validation only).

**Selected: `regulatory_histone_dnase`** (1,959 Histone ChIP-seq + 533 DNase-seq = 2,492 tracks; global SVD 256D; frozen as
`regulatory_histone_dnase_v1`, status `candidate_frozen`, see `docs/REGULATORY_REPRESENTATION.md`, section Frozen candidate). The choice is made
for parsimony and interpretability, on these grounds (3 seeds, paired bootstrap per seed):

- Adding DNase to Histone improves validation MSE by about 1.75% on average (seed range -1.62% to -1.87%), in 3/3 seeds, with a consistent MAE
  gain (-0.99% to -1.49%) and a per-seed bootstrap CI excluding 0 for both MSE and MAE in 3/3 seeds.
- Regulatory-Clean versus Histone+DNase improves MSE by only about 0.51% on average (-0.46% to -0.54%; seed-wise just above or below 0.5%).
- Clean's small advantage is less stable on MAE: seed 42 has delta MAE of about 0.00% with a CI that includes 0, and the MAE CI excludes 0 in
  2/3 seeds (MSE CI excludes 0 in 3/3).
- Clean needs 1,659 additional TF/CTCF tracks (4,151 versus 2,492), which also include the methylation-linked TF tracks handled separately in section 2 of
  `docs/REGULATORY_REPRESENTATION.md`.

Histone+DNase is NOT statistically equivalent to Clean. Clean's MSE gain is statistically clear within the bootstrap and reproducible in sign in
every seed. The decision is a parsimony/interpretability choice that leaves a small, MSE-reproducible advantage of Clean on the table. Whether
that advantage is biologically non-negligible is a judgement of the author; no numeric threshold for it was preregistered and none is used. An
earlier 1.5%-of-gap bar that appeared in the confirmation analysis was chosen post hoc by the assistant and was never preregistered by the user: it has
been removed from the decision and from the analysis code. The quantity "gain as a percentage of the prior-to-model gap" (about 0.90% for
H+D to Clean, about 3.2% for H to H+D, seed mean) is kept as a descriptive number only.

Training regime of the confirmation: a **fixed-budget selection protocol** (80 epochs, identical for all arms and seeds), not convergence. Early
stopping never triggered in any of the 9 runs and best epochs were 77 to 79, so curves were still improving at the end of the budget (mean
slope about -0.05% of MSE per epoch). Conclusions concern the ranking of arms at an equal, fixed budget.

Limits:

- The three decoder seeds share the mask seed, sample order and validation panels, so they are not independent samples of masking; n = 3 seeds
  supports sign counts and spread, not a formal test across seeds. The bootstrap covers patient and 1 Mb block sampling only.
- Fixed budget, not converged: a longer schedule could change the size (and in principle the ordering) of small gaps.
- Validation patients were used for checkpoint selection (`best.pt`) as well as for the reported score (small bias shared across arms).
- TCGA validation only; one dataset, 50% masking only.

Archive statement:

- Histone-only: mechanistic / sufficiency control.
- Regulatory-Clean: upper / reference feature-set control.
- The TF/CTCF block is NOT carried forward in the primary representation.
- Block-SVD and equal-group-mass weighting remain sensitivity analyses and are not selected.
- No further feature engineering will use TCGA validation. The test set is untouched.

