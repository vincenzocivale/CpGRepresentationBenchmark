# External reconstruction confirmation: runner, gate, checkpoint policy, Experiment B (implementation)

Implements protocol v1.1 (`docs/EXTERNAL_RECONSTRUCTION_PROTOCOL.md`, AMENDMENT 1; tags `external-recon-protocol-freeze-v1` / `-v1.1`).
The protocol document, the freeze manifest and all frozen data are NOT modified by this implementation. Nothing described here was
run on real data when it was written (tests are synthetic; the external test split is locked).

## How to run (after the implementation is committed)

```bash
export PYTHONPATH=$PWD/src
python scripts/run_external_confirmation.py generate            # 12 configs -> configs/experiments/external_confirmation/ (already tracked)
python scripts/run_external_confirmation.py validate            # configs == builder; diff vs TCGA only in declared fields
python scripts/run_external_confirmation.py audit [--json outputs/external_reconstruction_v1/gate/gate_validation.json]
python scripts/run_external_confirmation.py run --phase A --dry-run         # lists exactly the 12 runs in order, launches nothing
python scripts/run_external_confirmation.py run --phase A                   # ORCHESTRATOR ONLY; gate must be green
python scripts/run_external_confirmation.py run --phase A --seeds 17 --only regulatory_histone_dnase   # subsets (e.g. 3 concurrent seed processes)
python scripts/run_external_confirmation.py status
python scripts/analyze_external_confirmation.py [--experiment A|B_strict|B_recalibrated]   # read-only, validation by default
python scripts/run_external_transfer.py --mode both --dry-run                # Experiment B (secondary)
```

Order is seed-major (17, 42, 97), arms `regulatory_histone_dnase` (the manifest/matrix id of `regulatory_histone_dnase_v1`),
`functional_annotations_pca`, `cpgpt_large_locus`, `deepcpg_dna_locus`. Resumable: a finished run writes
`outputs/external_reconstruction_v1/logs/<arm>__seed<seed>.done` (JSON with its run dir); finished runs are skipped. A crashed run leaves
an unmarked run directory (never reused, never read by the analysis); rerunning starts a new run dir. `--test` / `--split test` are refused by `run`
(exit 2); the test evaluation is the separate `test-eval` subcommand, refused unless `freeze_state: final` AND `test_set_authorized: true`
AND every gate check is green.

## Budget verified by the runner itself (N_train = 524 from the frozen patient protocol)

`ceil(524 / 8) = 66` updates per epoch (65 batches of 8 + one of 4), `66 x 120 = 7,920` updates, 120 validation points
(validation MSE @ mask 0.50 at the end of every epoch), `max_updates` absent/null, `early_stopping: false`. The gate recomputes this from
the protocol file; after each run `build_status` asserts from the run's own `history.json` (opt-in `updates_in_epoch` / `updates_done` keys) that
`epochs_run == 120`, every epoch had 66 updates and `updates_done == 7920`, and that `best.pt` holds the epoch of the first history minimum.

## Gate (hard refusal; `run` exits 2, `audit` exits 1; code `src/cpg_repr_benchmark/external/gate.py`)

| check | refuses when |
| --- | --- |
| `freeze_verify` | any of the 50 `freeze.verify` checks fails (hashes, counts, partition, budget, amendment recorded, checkpoints) |
| `protocol_doc_sha256`, `sha256:*` | protocol doc / patient protocol / locus protocol / matrix h5 / mapping manifest / split csv differ from the manifest |
| `store_sha256:<arm>` | a representation store sha256 differs from manifest (= matrix.yaml) |
| `coverage_100pct:<arm>`, `universe_*` | a store covers less than 100% of the 408,017-locus universe, wrong dim, universe != matrix axis |
| `configs_12` | a config is missing/stale (differs from the builder), epochs != 120, early_stopping not false, `max_updates` set, batch != 8, updates per epoch (from N_train) != 66, `save_epoch_checkpoints` not false, no `record_update_counts`, no validation-only guard, layout not `split_dirs`, output root unsafe |
| `freeze_state_valid`, `state_consistent`, `validation_phase_test_locked`, `test_authorized_and_final` | state not draft/final; `test_set_authorized: true` without `final`; validation phase with test authorized; test requested without authorized + final |
| `protocol_tag_present`, `amendment_commit_is_tag`, `amendment_commit_ancestor_of_HEAD` | tag `external-recon-protocol-freeze-v1.1` missing / not commit `da02b78` / not an ancestor of HEAD |
| `frozen_files_unchanged_since_tag` | protocol doc, manifest, `configs/external`, `configs/frozen`, TCGA matrix dir differ from the tag or are dirty |
| `implementation_committed_clean` | a file of `REQUIRED_COMMITTED` is untracked or modified (list below) |
| `output_root_safe` / `assert_writable` | output root outside `outputs/external_reconstruction_v1`, or inside `data/derived/external_gse40279_v1`, `configs/external`, `configs/frozen`, stores dir, `outputs/{encode_atlas_v1,regulatory_confirmation_v1,biological_validation_v2}`, `.../audit`, `docs` |
| `no_evaluation_test_dir` | any `evaluation/test` directory exists under the external output root (validation phase) |

Validation-phase test protection: every config carries `patient_view: validation` + `require_patient_view: validation` (`guards.enforce_patient_view`
fails on `test` or an omitted view); the runner reads beta values only for `patient_split[patient_view]` (the protocol npz with test row ids is only
used to build the partition; no test beta is read); `evaluation/validation/` and `evaluation/test/` are separate (`output_layout: split_dirs`,
`allow_overwrite_split_dir: false`), so a test evaluation can never overwrite validation outputs and a used split directory is never reused.
The external test authorization is parametrised by the opt-in config key `evaluation.test_authorization: external_gse40279_v1`
(`eval_layout.authorizer_from_cfg`; default None keeps the TCGA-matrix authorization, behaviour unchanged for TCGA callers).

### Files that must be committed (clean) before launch (`gate.REQUIRED_COMMITTED`)

```
src/cpg_repr_benchmark/external/{gate,runner,transfer}.py
src/cpg_repr_benchmark/training/engine.py                (modified, opt-in keys)
src/cpg_repr_benchmark/experiments/eval_layout.py        (modified, authorizer_from_cfg)
scripts/run_masking_benchmark.py                         (modified, passes opt-in keys + authorizer)
scripts/{run_external_confirmation,run_external_transfer,analyze_external_confirmation}.py
configs/experiments/external_confirmation/               (12 configs)
tests/{test_external_runner,test_external_transfer,test_external_analysis,test_checkpoint_policy}.py
docs/EXTERNAL_RECONSTRUCTION_RUNNER.md
```

## Run configs and the allowed differences vs the TCGA confirmation configs

Built by the TCGA builder (`confirmation_matrix.build_config`, same template `configs/experiments/masking/functional_pca_genomewide.yaml`) and then
overridden ONLY at these dotted paths (`runner.ALLOWED_DIFF_VS_TCGA`; a test proves that exactly these differ for all 12 configs and `generate`
refuses anything else):

| field | TCGA | external |
| --- | --- | --- |
| `experiment.name` | `regulatory_confirmation_matrix_<arm>__seed<s>` | `external_confirmation_<arm>__seed<s>` |
| `experiment.output_root` | `outputs/regulatory_confirmation_v1/benchmark` | `outputs/external_reconstruction_v1/benchmark` |
| `experiment.locus_split.protocol_path` | `outputs/encode_atlas_v1/loci_seen.npz` | `data/derived/external_gse40279_v1/loci_seen_external_gse40279_v1.npz` |
| `dataset.name` | `tcga_array_genomewide` | `external_gse40279_v1` |
| `dataset.methylation_h5` | `data/methylation/tcga_array_official_full.h5` | `data/derived/external_gse40279_v1/methylation.h5` |
| `dataset.patient_protocol` | `outputs/encode_atlas_v1/patients.npz` | `data/derived/external_gse40279_v1/patients_external_gse40279_v1.npz` |
| `training.save_epoch_checkpoints` | (absent = true) | `false` |
| `training.save_last_checkpoint` | (absent = true) | `false` (see below) |
| `training.record_update_counts` | (absent = false) | `true` |

Everything else is identical: epochs 120, early_stopping false, batch 8 (train and eval), lr 1e-4, wd 1e-4, fp16, panel 2048, panel_repeats 1,
training mask fractions, evaluation fractions 0.15/0.30/0.50/0.70/0.90, `selection_mask_fraction` 0.5, mask seed 17001, `num_workers` 8 / 0,
model widths, representation block (store, provenance, sha256), patient split seed 20260925, `heldout_fraction` 0.0, `patient_view`/`require_patient_view`
validation, `split_dirs`, `allow_overwrite_split_dir` false. Per-arm configs differ only in `experiment.name`, `training.seed` and `representation.*`.

## Checkpoint policy (opt-in, default off = TCGA behaviour byte-for-byte)

`training.save_epoch_checkpoints` (default true), `training.save_last_checkpoint` (default true), `training.record_update_counts` (default false), passed by
`run_masking_benchmark.py` to `engine.train_model`. External runs: no `epoch_XXXX.pt`, `last.pt` OFF, `best.pt` always, `history.json` complete (120 epochs
with `updates_done`), resolved config, `experiment.json`, `prior_logit.npy`, split files, `confirmation_status.json` (hashes, best epoch/update, best val MSE,
wall clock, `test_read: false`, protocol tag/version). `best.pt` is overwritten only on STRICT validation-MSE improvement (`<`), so ties keep the earliest
epoch. `generate --keep-last-checkpoint` writes `save_last_checkpoint: true` (audit/debug; ~14 MB per run); the gate accepts either value but all 12 configs
must equal the builder output for the value they carry. Regression tests: `tests/test_checkpoint_policy.py` (default unchanged, no epoch files, strict
improvement/ties, 120 epochs with 7,920 counted updates, runner smoke on synthetic CPU tensors).

## Experiment B (secondary, exploratory): TCGA -> GSE40279 transfer

`scripts/run_external_transfer.py --mode B_strict|B_recalibrated|both [--split validation]`, module `external/transfer.py`. For each of the 12 frozen phase-A
checkpoints: sha256 checked against the manifest, `MaskedMethylomeReconstructor(raw_locus_dim=store.dim, 256, 256, 256, 512)` loaded from `state["model"]` only
(`eval()`, `requires_grad_(False)`, no optimizer; a test asserts the weights are bit-identical after evaluation), the arm's own store mapped to the external
matrix columns by cpg_idx, same patients/locus pool/order, panel 2048, mask seed 17001 and fractions as Experiment A (identical masks, so A and B predictions are paired).

* `B_strict`: prior = the checkpoint's own TCGA run `prior_logit.npy` (it IS stored in every TCGA run dir; sha256 recorded), re-indexed to the external columns by global cpg_idx.
  Provenance asserted without reading any TCGA beta: `patient_split.npz` train rows == frozen TCGA train rows (`outputs/encode_atlas_v1/patients.npz`) and == `experiment.json`
  `n_train_patients_rows`. An optional heavy check (`recompute_tcga_prior_check`, reads TCGA TRAIN betas only, never TCGA test rows) recomputes the prior and asserts equality.
* `B_recalibrated`: prior recomputed with `compute_leakage_safe_priors` from the 524 external TRAIN rows x all 408,017 universe columns; the function asserts
  the exact rows (== frozen train, disjoint from validation/test) and columns before reading; one prior shared by all arms, saved as `prior_external_train.npy` + json.
* Outputs: `outputs/external_reconstruction_v1/transfer/{B_strict,B_recalibrated}/<arm>/seed_<s>/evaluation/validation/seen/mask_<f>/...`, `transfer_manifest.json`
  (checkpoint/store/protocol/matrix sha256, prior info, `weights_updated: false`, `test_read`), `transfer.done`. Modes never share a directory. Test split: refused
  unless final + authorized + green gate (scope B adds checkpoint sha256 checks, no config checks).
* Locus-agnostic check: the model has no locus-indexed parameter (`locus_adapter` is `LayerNorm + Linear` on the raw dimension; the decoder sees adapter outputs and the
  prior logit); the TCGA universe (408,399) is a superset of the external one (408,017), so the 382 TCGA-only prior entries are simply unused (re-indexing requires every
  external id to exist on the TCGA axis and raises otherwise). The prior shift tumour -> blood is the declared caveat of B (protocol section 16).
* Analysis: `analyze_external_confirmation.py --experiment B_strict|B_recalibrated` (same contrasts, written to a separate directory; convergence block not available).

## Analysis scaffold

`scripts/analyze_external_confirmation.py`: per seed and across seeds, delta = comparator - candidate (positive = candidate lower error), absolute and relative, paired
95% CI via `encode_atlas/statistics.paired_bootstrap` (patients x 1 Mb blocks, 2,000 replicates, seed 17), all five fractions (primary: MSE@0.50), MSE/MAE with CIs and
MAS-PCC/MAC-PCC differences, per-arm metrics, convergence (best epoch, best in last 10, last-10 slope), neutral reading of each CI (protocol section 9). Three primary
contrasts only (no multiplicity adjustment); no threshold is applied to the functional comparison. `deepcpg_dna_locus - cpgpt_large_locus` is written to separate files
and a separate report section ("outside the primary family"). Works with partial results (marked PROVISIONAL). Block labels use the external registry reindexed to the
external matrix columns. `--split test` is refused unless authorized + final + green gate.

## Cost (protocol section 17, unchanged)

About 10-15 min per run (7,920 updates at 12.9-20.9 updates/s = 6-10 min + setup/priors 2-5 min); 12 runs sequential about 2-3 h, about 1-1.5 h with three concurrent
seed processes (assumption). Disk without epoch checkpoints: `best.pt` about 14 MB (+ 14 MB `last.pt` only with `--keep-last-checkpoint`), `prior_logit.npy` 1.6 MB,
`locus_split.npz` about 2 MB, predictions (5 fractions x 66 patients x 2,048 values) a few MB: well under 100 MB per run, about 1 GB for 12. Experiment B: no training,
12 checkpoints x 2 modes x 5 fractions of 66 patients, minutes per run on GPU.
