# Approved EWAS-v3 execution

Approval: user instruction “Procedi con gli esperimenti mancanti”, recorded against the exact frozen manifest SHA256 in `configs/ewas_v3/user_approval.json`.

Scope: the 13 frozen main tasks, three frozen arms, both p-value thresholds and both frozen matching designs. Supplementary and inventory-only tasks are not added. The primary paper table has 39 rows, with 39 paired AUROC contrasts. Sensitivities are separate tables with the same task/source panel. Support status never changes task selection.

The new runner reads and verifies the panel and every paired artifact; verifies registered store SHA256, native dimensions and all 408,399 locus IDs before fitting; retains zero rows; runs two independent endpoint processes at four threads each. Checkpoints retain each completed outer fold. It uses the frozen L2/lbfgs probe, C grid and max_iter; StandardScaler is fitted inside EACH inner training partition and each outer training partition, as the frozen EWAS-v3 scaler contract requires. The old EWAS-v2 implementation fits one scaler on the outer train and reuses it inside the inner CV, so that function is not reused. Strict pair concordance gives ties zero, as the EWAS-v3 definition requires.

Preparation artifact clarification: the frozen file named `chromosome_bootstrap_multiplicities.csv` contains sampled block indices, not multiplicities. Its integer matrix is exactly the 1,000 seed-17 draws. The runner interprets these draws using sorted chromosome block names, as specified in the frozen uncertainty contract, reconstructs multiplicities and checks equality before fitting. The panel/artifacts are unchanged.

Endpoints lacking all five frozen outer folds are reported as not_evaluable, without inventing a different CV design or selecting replacement sources. Non-converged fits prevent paired contrasts. Every representation uses exactly the same pair order/folds/bootstrap draws. Results cannot change selection.

Command: `nice -n 10 /data2/home/vcivale/miniconda3/envs/cpg-repr-benchmark/bin/python scripts/ewas_v3/run_experiments.py`.

Outputs: `outputs/ewas_v3/ewas-v3-main-panel-freeze-v1/` with primary tidy results, paired contrasts, all sensitivity rows, OOF scores/checkpoints, execution provenance, and main absolute-AUROC PDF/PNG. The figure marks traits whose primary positive sets share >50% containment. No category average or category win counts are generated.

After the runner exits successfully, run `python scripts/ewas_v3/finalize_results.py` in the same environment. This exports versioned results/figures and checks every endpoint against the exact shared pairs. It also repairs the inherited AP tie bug: the old utility ranks tied rows individually, favoring positive-first row order; standard average precision must aggregate equal-score groups. The repair is applied to every evaluated endpoint from existing OOF predictions, with legacy values retained as diagnostics. No probe, score, AUROC, CI, matching pair, source or panel selection changes. Output/log hashes are finalized after stdout closes.
