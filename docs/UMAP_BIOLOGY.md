# Comparable CpG UMAP figures

Run in an environment with the project's `viz` dependencies:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 \
  python scripts/plot_umap_biology.py
```

The default comparison uses Functional PCA, DeepCpG DNA (HCC), and CpGPT large.
All three cover the same 408,399 loci in the current stores. A uniform sample of
12,000 shared GRCh38 coordinates is drawn once, without using biological labels.
Every representation receives centered PCA to 50 components, without whitening
or per-feature standardization, and cosine UMAP with 30 neighbors, min_dist 0.1.
The sample and PCA seed are fixed at 17; all three UMAP seeds (17, 42, 97) are
reported, rather than selecting a favorable map. UMAP axes are arbitrary and
are not aligned between representations. Distances between separated clusters
and cluster area should not be used to infer a quantitative biological effect.

Each figure reuses the exact same coordinates for genomic context and both EWAS
overlays. Background CpGs are drawn first and highlighted CpGs last; no EWAS
oversampling is used. Missing context annotations remain `unknown`.

Outputs in `outputs/figures/umap_biology/`:

- `biology_seed_*.png` / `.pdf`: maps for every seed.
- `ewas_neighborhood_enrichment.png` / `.pdf`: native, PCA and 2D neighborhood
  enrichment; the 2D error bars are the full seed range, not confidence intervals.
- `sample.csv`: sampled coordinates and labels, shared by all arms and seeds.
- `<representation>_seed_<seed>.csv`: UMAP coordinates with explicit locus IDs.
- `neighborhood_metrics.csv`: per-class neighbor fractions and fold enrichment.
- `manifest.json`: parameters, package versions, source catalog entries, file
  size/mtime, coverage and PCA variance retained. File metadata is not a content hash.

Enrichment is the mean fraction of same-label neighbors around positive CpGs,
divided by `(number_of_positives - 1) / (sample_size - 1)`. Self-neighbors are
excluded, including when embeddings coincide. Native and PCA neighbors use the
configured metric; 2D neighbors use Euclidean distance. Enrichment is undefined
for fewer than two positives. It is a descriptive statistic, not cross-validated
prediction, a significance test, or a correction for spatial correlation.

## Interpretation and source audit

**Genomic context is already present in the functional source features.** The
source cache manifest at `data/external/functional/locus_features_v1/manifest.json`
declares `annotation_core_v1` and UCSC hg38 CpG islands. The sibling source code
`MehylPredictor/src/methylation_predictor/locus_features/annotations.py`, function
`encode_annotation_core`, explicitly one-hot encodes island/shore/shelf/open_sea
as the first four dense features. Its `storage.py` concatenates those annotations
with regulatory breadth, and `scripts/export_functional_feature_store.py` exports
that dense block for `scripts/build_functional_pca_embedding.py`.

Consequently, context separation illustrates retention of source information;
it is not independent biological validation. This source audit qualifies the
claim of annotation independence currently made in `EMBEDDING_EVALUATION.md`.
The EWAS overlays are more relevant for exploring external associations, but
their enrichment can also reflect genomic context and correlated nearby loci.
A stronger follow-up is a context-matched background or chromosome-blocked
prediction, plus the separate masking/reconstruction benchmark.

Defaults use the two larger individual cancer sets already available locally
(colorectal and prostate). This is an exploratory choice, not a preregistered
evaluation or evidence that all traits favor one representation. Small clock or
disease sets may contribute too few positives under uniform sampling; counts
are printed in the maps and saved in the metrics.

To examine native geometry without PCA, apply the change to every arm:

```bash
PYTHONPATH=src python scripts/plot_umap_biology.py --pca-components 0 \
  --out-dir outputs/figures/umap_biology_native
```

Adding NTv3 restricts the shared universe to chr1; adding MethylGPT restricts it
to its covered probe vocabulary. Coverage is recorded explicitly. Those would
be different comparison universes and should be labeled as such.

To restyle existing results without recomputing UMAP:

```bash
PYTHONPATH=src python scripts/plot_umap_biology.py --render-only
```

## Generated comparison (2026-09-23)

The default sample contains 217 colorectal and 221 prostate EWAS CpGs, with 15
shared between the two sets. Native cosine-neighborhood enrichment is:

| Representation | Colorectal | Prostate |
|---|---:|---:|
| Functional PCA | 4.488 | 4.278 |
| DeepCpG DNA (HCC) | 1.280 | 1.423 |
| CpGPT (large) | 3.345 | 2.402 |

Functional PCA also has the largest enrichment after PCA and in every tested
UMAP seed. Its 2D enrichment ranges from 3.055 to 3.260 for colorectal and from
2.912 to 3.060 for prostate. This supports stronger local grouping for these
two sets on this sample; it does not establish a context-independent effect or
general superiority across traits, samples or UMAP hyperparameters.

Validation included the full three-representation/three-seed run, inspection
of rendered maps and the diagnostic figure, Ruff, and synthetic checks of
coordinate-to-row alignment, self-exclusion with tied embeddings, the random
mixing baseline, and undefined enrichment for a singleton set.
