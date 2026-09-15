# Experiment matrix

## Axis 1 — representation source

- **Functional annotations**: reference-genome functional tracks + dense locus covariates.
- **Genomic foundation models**: patient-agnostic sequence-derived CpG embeddings.
- **Methylation foundation models**: only their patient-agnostic CpG/locus component, before
  methylation-value or other patient-context integration.

## Axis 2 — representation training regime

| Track | Question | Allowed supervision |
|---|---|---|
| `native_frozen` | How informative is the representation as published? | only the source model's original training |
| `proxy_aligned` | How informative can this source become with our methylation-aware proxy? | train-patient mean beta on persistent train CpGs only |

Do not mix these tracks in one ranking.

## Axis 3 — downstream tasks

1. masked methylation reconstruction;
2. age prediction / age-bin classification;
3. mortality classification;
4. disease classification.

For age/mortality/disease, evaluate robustness under the same CpG masking sweep:
`0%, 15%, 30%, 50%, 70%, 90%`.

## Publication matrix

Every row below uses the same patient/CpG protocol within a task.

| Representation | Family | Track | Masking | Age | Mortality | Disease | Sparse eval |
|---|---|---|---:|---:|---:|---:|---:|
| Functional annotation encoder | functional | proxy_aligned | yes | yes | yes | yes | yes |
| NTv3-pre | genomic FM | native_frozen | yes | yes | yes | yes | yes |
| NTv3-pre + proxy | genomic FM | proxy_aligned | yes | yes | yes | yes | yes |
| other genomic FM | genomic FM | native_frozen | yes | yes | yes | yes | yes |
| other genomic FM + proxy | genomic FM | proxy_aligned | yes | yes | yes | yes | yes |
| CpGPT locus component | methylation FM | native_frozen | yes | yes | yes | yes | yes |
| CpGPT locus + proxy | methylation FM | proxy_aligned | yes | yes | yes | yes | yes |
| MethylGPT locus component | methylation FM | native_frozen | yes | yes | yes | yes | yes |
| MethylGPT locus + proxy | methylation FM | proxy_aligned | yes | yes | yes | yes | yes |

## Controls required before the main paper table

- multiple seeds;
- persistent patient and CpG protocols;
- common CpG vocabulary (no silent representation-specific intersection);
- fixed embedding width control (proxy output is 256-D; native raw embeddings should also get a
  label-free 256-D PCA/IncrementalPCA sensitivity analysis);
- functional-track leakage audit;
- strict proxy train-locus-only fitting;
- exact extraction manifest for every external model: checkpoint, layer, pooling, sequence window,
  genome build and whether the extracted tensor is before patient/value integration;
- native full-model masking results in a separate complete-method table only.
