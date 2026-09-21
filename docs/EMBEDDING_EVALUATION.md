# Embedding-centric evaluation

This is the primary evaluation path after the alignment with the methylation-FM collaborator:
we score representation quality through the **patient embedding** and the **CpG-locus
embedding** the model produces, not through the reconstructed methylation profile. The
reconstruction-centric pipeline (`scripts/run_reconstruction_downstream.py`) and masking
benchmark (`scripts/run_sparse_reconstruction.py`, `scripts/run_masking_benchmark.py`) remain
in the repo — masking stays a first-class representation-quality axis — but they are no longer
the primary downstream-phenotype evaluation.

## Pipeline

1. **Extract** — `scripts/extract_embeddings.py` loads a fitted `sparse_reconstruction`
   checkpoint and runs `encode_patient()` once per patient in a downstream-task cohort, using
   every finite CpG that patient shares with the representation's universe (not a sparse
   sample — the whole observed set). Output: a `.npz` following the fixed contract in
   `cpg_repr_benchmark.embedding.store` (`patient_ids`, `embedding`) plus a `.meta.json` with
   provenance (checkpoint, representation, track, architecture, pooling, patient split).

2. **Probe** — `scripts/run_embedding_probe.py` fits a frozen linear probe (`RidgeCV` for
   regression tasks like age, `LogisticRegressionCV` for binary tasks like disease) on the
   embedding, train-only standardized and alpha-selected, evaluated once on a held-out patient
   split. This is the `mode: frozen_pretrained` case from the plan.

3. **Fine-tune** (optional) — `cpg_repr_benchmark.probing.finetune.fine_tune_encoder` unlocks
   `encode_patient` plus a 1-layer head and trains end-to-end on the downstream task's train
   split (`mode: fine_tuned`). It never touches the reconstruction decoder or the original
   sparse-reconstruction training loop.

4. **Bio-validation** — `scripts/run_bio_validation.py` probes the CpG-locus embedding itself
   (a representation's raw `/embedding` array, not the patient embedding) against external
   annotations that are not part of the ENCODE input features: genomic context
   (island/shore/shelf, gene relationship) and literature-curated known CpG sets (e.g. the
   Horvath clock). See `data/bio_annotations/README.md` for the expected file formats — none
   are bundled; provenance must be recorded before use.

## Result format

Every new script writes a `summary.json` through
`cpg_repr_benchmark.experiments.result_schema.embedding_summary_payload`, under the existing
content-addressed layout from `experiments/run_store.py`:

```
outputs/embedding_probe/<dataset>/<representation>/<track>/seed_<seed>/summary.json
outputs/bio_validation/<representation>/<track>/seed_<seed>/summary.json
```

Common shape:

```json
{
  "schema_version": 1,
  "task": "embedding_linear_probe",
  "dataset": "gse40279_age",
  "representation": "functional_annotations_pca",
  "track": "native_frozen",
  "mode": "frozen_pretrained",
  "embedding_source": {"type": "patient_embedding", "dim": 256, "checkpoint": "..."},
  "metrics": {"train": {...}, "test": {...}},
  "n_patients": {"train": 400, "test": 100}
}
```

`bio_validation` uses `embedding_source.type: "cpg_locus_embedding"` and nests `metrics` under
`genomic_context`/`known_cpg_sets`.

## Plugging in the methylation-FM collaborator's embeddings

No adapter code is required here. Any embedding source that can be exported as an `.npz`
following `cpg_repr_benchmark.embedding.store`'s contract (`patient_ids` + `embedding` for
patient-level, or a canonical `/cpg_idx` + `/embedding` HDF5 per
`docs/ADDING_REPRESENTATIONS.md` for locus-level) plugs directly into
`run_embedding_probe.py` / `run_bio_validation.py` — the collaborator only needs to produce
that file, not integrate with this codebase.
