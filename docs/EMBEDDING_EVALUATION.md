# Bio-validation of CpG-locus embeddings

`scripts/run_bio_validation.py` (or the catalog-driven `scripts/run_bio_validation_all.py`) probes a
representation's raw CpG-locus embedding (the `/embedding` array of its canonical HDF5 store, not a
patient embedding) against external annotations that are not part of the ENCODE input features used to
build `functional_annotations_pca`: genomic context (island/shore/shelf, gene relationship),
literature-curated known CpG sets (binary membership; e.g. Horvath/Hannum/PhenoAge clocks, EWAS
Catalog traits), and — for the three clocks that publish a full per-CpG weight table — a stricter
`clock_coefficients` regression probe (`--coefficients-dir`) that predicts the actual elastic-net
coefficient rather than just set membership. See `data/bio_annotations/README.md` for the expected
file formats — none are bundled; provenance must be recorded before use.

This is the primary biological-validity axis for the paper, alongside the masking-reconstruction
benchmark (`docs/BENCHMARK_V2.md`), which is the computational-justification axis. Downstream
phenotype-task probing (age/disease prediction from a patient embedding) was dropped from this repo;
see git history if that pipeline is ever needed again.

## Running

```bash
# one representation
python scripts/run_bio_validation.py \
  --representation-store data/cache/representations/<repr>.h5 \
  --representation-name <repr> \
  --genomic-context-parquet data/bio_annotations/genomic_context.parquet \
  --known-sets-dir data/bio_annotations/known_sets \
  --coefficients-dir data/bio_annotations

# whole catalog, skipping representations already evaluated
python scripts/run_bio_validation_all.py
python scripts/run_bio_validation_all.py --only cpgpt_locus --force
```

## Result format

Writes `summary.json` through `cpg_repr_benchmark.experiments.result_schema.embedding_summary_payload`:

```
outputs/bio_validation/<representation>/<track>/seed_<seed>/summary.json
```

```json
{
  "schema_version": 1,
  "task": "bio_validation",
  "dataset": "cpg_locus_annotations",
  "representation": "functional_annotations_pca",
  "track": "native_frozen",
  "mode": "frozen_pretrained",
  "embedding_source": {"type": "cpg_locus_embedding", "dim": 256, "checkpoint": "..."},
  "metrics": {"genomic_context": {...}, "known_cpg_sets": {...}, "clock_coefficients": {...}},
  "n_patients": null
}
```

## Plugging in a new representation

No adapter code is required. Any embedding source exported as a canonical `/cpg_idx` + `/embedding`
HDF5 per `docs/ADDING_REPRESENTATIONS.md` plugs directly into `run_bio_validation.py` — the
representation only needs to produce that file and be registered in
`configs/representations/future_models.yaml`.
