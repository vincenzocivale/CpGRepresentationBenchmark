# Regulatory confirmation matrix (frozen spec; nothing executed)

`matrix.yaml` freezes the final comparison of the frozen candidate `regulatory_histone_dnase_v1` against the legacy functional PCA baseline,
CpGPT-large, DeepCpG-DNA and two modern sequence FMs (still `pending`). Protocol: `docs/REGULATORY_CONFIRMATION_PROTOCOL.md` (AMENDMENT 2: exactly 120
epochs, no early stopping, `best.pt` by validation MSE @ 0.50, never the last epoch automatically). Fairness: `docs/REGULATORY_CONFIRMATION_FAIRNESS.md`.

## Arms
Main (native dimension, primary result): candidate (256D), `functional_annotations_pca` (256D, canonical store = campaign `full_a18b869b.h5`,
OPEN decision flagged in matrix.yaml), `cpgpt_large_locus` (512D), `deepcpg_dna_locus` (128D), `modern_sequence_fm`, `modern_sequence_fm_alt` (pending).
Sensitivity only (`role: sensitivity`, never main comparators): `cpgpt_large_locus_256_compact`, `deepcpg_dna_locus_hepg2`.
Seeds 17, 42, 97; evaluation fractions 0.15/0.30/0.50/0.70/0.90 from the SAME `best.pt`; shared mask seed 17001.

## Commands (all CPU/metadata only unless `run` is invoked by the user)
```bash
python scripts/run_regulatory_confirmation_matrix.py generate   # 18 configs in runs/ (12 main + 6 sensitivity)
python scripts/run_regulatory_confirmation_matrix.py validate
python scripts/audit_confirmation_matrix.py [--list]            # exit 1 + failure list unless fully green
python scripts/run_regulatory_confirmation_matrix.py run --dry-run
python scripts/audit_confirmation_matrix.py --phase A           # also reports "phase-A arms ready"; overall audit still fails (pending FMs)
python scripts/run_regulatory_confirmation_matrix.py run --phase A --dry-run   # exactly the 12 Phase A runs, launches nothing
python scripts/run_regulatory_confirmation_matrix.py run --phase A --split validation   # PHASE A: 12 validation-only runs
python scripts/run_regulatory_confirmation_matrix.py run [--only ARM...] [--seeds S...] [--include-sensitivity]
python scripts/analyze_confirmation_matrix.py                   # validation split; --split test is refused unless authorized
```
`run` without `--phase` refuses while the audit fails. `--phase A` (validation only) launches the 4 fully registered main arms when they pass the audit; only the pending
modern-FM failures are tolerated and they still block `final`/test. The legacy `--allow-incomplete-validation-only` flag is an equivalent escape hatch for all available arms. Neither permits `--split test`. `--split test` additionally needs
`test_set_authorized: true`, `freeze_state: final` and an all-green audit, and test execution is not implemented in this scaffold (the `split_dirs` layout already separates
`evaluation/test` from `evaluation/validation`).

## Registering a modern sequence FM
Copy `sequence_fm_registration.template.yaml` to `sequence_fm_registration_slot1.yaml` / `_slot2.yaml`, fill EVERY field, materialize the store
(canonical `/cpg_idx` int64 + `/embedding`), add its catalog entry, record its sha256 and flip the arm to `status: not_run`. No choice may be optimized
with TCGA methylation reconstruction. An incomplete registration fails the audit.

## State machine
`freeze_state: draft -> final` (final requires an all-green audit); `test_set_authorized` may be true only when final + green. Both are enforced by the audit.

## Cost (estimate; shared machine)
Per run ~7 min setup + 120 x ~49 s + ~3 min evaluation (5 fractions) = ~1.8 h; CpGPT-large 512D ~2.1 h (conservative: larger locus-embedding
reads per batch, loader bound). Main 12 runs: 9 x 1.8 + 3 x 2.1 = ~22.5 h; sensitivity 6 runs: ~11 h; the two future modern FMs add 6 runs (~11-13 h).
Disk ~1.1 GB per run (every-epoch checkpoints, ~14 MB/epoch).

## Phase A outputs and analysis
Runs live under `outputs/regulatory_confirmation_v1/benchmark/...`, logs/`.done` markers in `outputs/regulatory_confirmation_v1/logs/`, each run has
`confirmation_status.json` and the aggregate is `outputs/regulatory_confirmation_v1/phase_A_status.json`. Results are under `<run>/evaluation/validation/`.
After completion: `python scripts/analyze_confirmation_matrix.py --split validation` (paired bootstrap, comparator minus candidate, writes to `<runs-root>/analysis`).
