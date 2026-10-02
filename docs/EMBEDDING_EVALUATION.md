# Bio-validation of CpG-locus embeddings

**Source audit correction:** genomic context, gene region, cCRE and TSS information are
already functional inputs. Their recoverability is not independent biological validation.
The older random-fold probes below remain exploratory. For chromosome-blocked, context-controlled
attribution and source-feature ablations, use [ENCODE_ATTRIBUTION.md](ENCODE_ATTRIBUTION.md).

`scripts/run_bio_validation.py` (or the catalog-driven `scripts/run_bio_validation_all.py`) probes a
representation's raw CpG-locus embedding (the `/embedding` array of its canonical HDF5 store, not a
patient embedding) against annotations with different levels of independence from the functional
inputs: source-derived genomic context (island/shore/shelf, gene relationship; a
**source-information retention / sanity check** for `functional_annotations_pca`, not independent validation),
literature-curated known CpG sets (binary membership; e.g. Horvath/Hannum/PhenoAge clocks, EWAS
Catalog traits), and — for the three clocks that publish a full per-CpG weight table — a stricter
`clock_coefficients` regression probe (`--coefficients-dir`) that predicts the actual elastic-net
coefficient rather than just set membership. See `data/bio_annotations/README.md` for the expected
file formats — none are bundled; provenance must be recorded before use.

Only the probes independent of the functional inputs (phastCons, EWAS sets, clock coefficients,
etc.) count as biological evidence; genomic context, gene region, cCRE class and TSS distance are
retention checks for `functional_annotations_pca`. Together with the masking-reconstruction
benchmark (`docs/BENCHMARK_V2.md`, the computational axis), these exploratory probes are
supporting evidence; chromosome-blocked, context-controlled attribution lives in `ENCODE_ATTRIBUTION.md`. Downstream
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
  "metrics": {"genomic_context": {... source-retention check ...}, "known_cpg_sets": {...}, "clock_coefficients": {...}},
  "n_patients": null
}
```

## Plugging in a new representation

No adapter code is required. Any embedding source exported as a canonical `/cpg_idx` + `/embedding`
HDF5 per `docs/ADDING_REPRESENTATIONS.md` plugs directly into `run_bio_validation.py` — the
representation only needs to produce that file and be registered in
`configs/representations/future_models.yaml`.
