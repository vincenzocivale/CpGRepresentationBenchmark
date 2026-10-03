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
python scripts/run_regulatory_confirmation_matrix.py run [--only ARM...] [--seeds S...] [--include-sensitivity]
python scripts/analyze_confirmation_matrix.py                   # validation split; --split test is refused unless authorized
```
`run` refuses while the audit fails. `--allow-incomplete-validation-only` is the only escape hatch: it tolerates only the pending modern-FM failures,
launches validation-split runs of available arms only, is not the final benchmark, and never permits `--split test`. `--split test` additionally needs
`test_set_authorized: true`, `freeze_state: complete` and an all-green audit, and test execution is not implemented in this scaffold (it needs a separate
evaluation output directory so the validation results are not overwritten).

## Registering a modern sequence FM
Copy `sequence_fm_registration.template.yaml` to `sequence_fm_registration_slot1.yaml` / `_slot2.yaml`, fill EVERY field, materialize the store
(canonical `/cpg_idx` int64 + `/embedding`), add its catalog entry, record its sha256 and flip the arm to `status: not_run`. No choice may be optimized
with TCGA methylation reconstruction. An incomplete registration fails the audit.

## State machine
`freeze_state: draft -> complete` (complete requires an all-green audit); `test_set_authorized` may be true only when complete + green. Both are enforced by the audit.

## Cost (estimate; shared machine)
Per run ~7 min setup + 120 x ~49 s + ~3 min evaluation (5 fractions) = ~1.8 h; CpGPT-large 512D ~2.1 h (conservative: larger locus-embedding
reads per batch, loader bound). Main 12 runs: 9 x 1.8 + 3 x 2.1 = ~22.5 h; sensitivity 6 runs: ~11 h; the two future modern FMs add 6 runs (~11-13 h).
Disk ~1.1 GB per run (every-epoch checkpoints, ~14 MB/epoch).
