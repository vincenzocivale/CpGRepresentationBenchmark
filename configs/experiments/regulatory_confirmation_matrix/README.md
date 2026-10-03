# Regulatory confirmation matrix (scaffold)

Nothing in this directory has been executed. It only specifies the arms (`matrix.yaml`) of a future confirmation comparison of the frozen
candidate `regulatory_histone_dnase_v1` against the legacy functional-annotation PCA baseline and sequence/foundation-model embeddings.

- No per-run configs are provided and no runner reads this directory (runner globs target `configs/experiments/masking/` and the
  `regulatory_*` experiment folders only). Any per-run config added later must be named `*.template` until launch is authorized.
- The evaluation protocol is not defined yet ("to be defined"). The test patients are NOT used, and the test set remains untouched, until
  the user explicitly authorizes it. `seeds` is an empty placeholder.
- `modern_sequence_fm*` entries are placeholders (status `pending`, no store).
- Store files are referenced by path only; none was created or modified by this scaffold.
