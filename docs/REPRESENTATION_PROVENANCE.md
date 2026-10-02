# Representation provenance audit

Documentation-only audit (2026-10-02). No code, config or representation was changed and no experiment was run.
Only configs, docs, scripts, small JSON/TSV sidecars and HDF5 attributes were read. Statements are tagged
**FACT** (verified in a local file, cited), **EXTERNAL** (general knowledge about a published resource, not verified
in this repo) or **UNKNOWN**.

Terminology: "label-free" = no methylation value, patient data or task label enters the construction of the locus
embedding. "Methylation supervision" = whether the encoder that produced the vectors was trained against methylation
targets (by its authors or by us).

## 1. Table of current representation arms

Common to all arms: the locus universe is the 408,399 array CpGs of `data/cpg/registries/array_cpg_map.parquet`
(Illumina-array probe design; see section 3) unless noted. The catalog `provenance.supervision` field is quoted
verbatim in the "catalog says" column because, for several arms, it is **not** equivalent to "encoder never saw methylation".

| Arm | Family | Source data | Patient-specific | Label-free | Methylation supervision (facts) | Checkpoint / provenance | Locus-fit scope | Evidence |
|---|---|---|---|---|---|---|---|---|
| `functional_annotations_pca` (and ENCODE atlas `full`) | functional (reference) | 4,165 ENCODE binary peak-overlap tracks (Histone ChIP-seq 1959, TF ChIP-seq 1673, DNase-seq 533) + 23 dense features (4 CpG context from UCSC cpgIslandExt, 4 gene region from GENCODE v50, 9 cCRE classes from cCRE Registry V4, log10 TSS distance, 5 breadth summaries) | No | Yes (TruncatedSVD, 256 comp., seed 17; variance retained 0.707) | None in the SVD. Inputs contain no methylation assay (section 3). | Store built from `reference_functional_features_tcga_array_all_v1_cpgidx_fixed.h5`; origin is the sibling MehylPredictor cache (export script), see section 2 | Fit on all 408,399 loci (transductive, unsupervised) | `configs/representations/future_models.yaml:5-20`; `scripts/build_functional_pca_embedding.py:1-70`; `data/cache/representations/functional_annotations__pca_native_genomewide.h5.json`; `outputs/encode_atlas_v1/embeddings/full_a18b869b.json`; `data/derived/functional_annotations/*cpgidx_fixed.h5` attrs (`patient_specific=False`, `source: reference-genome functional annotation cache`) |
| `functional_annotations_pca_450k_fit` | functional (reference) | Same feature store (paired 450K/EPIC universe, 843,855 loci) | No | Yes | None | SVD fit on 439,760 450K-shared loci, projected to 404,095 EPIC-only loci | Fit subset only (unsupervised) | `future_models.yaml:22-36`; `scripts/build_functional_pca_projection.py:1-61`; `functional_annotations__pca_450k_fit_paired_v1.h5.json` |
| ENCODE atlas ablation arms: `drop_assay_*`, `add_assay_*`, `drop/add_biosample_class_*`, `add/only_target_h3k4me3`, `drop_dense_{ccre,context,breadth}`, `balanced_assays_{17,42,97}`, `cluster_representatives`, `control_(non_)histone_prevalence_matched`, `drop_preserve_breadth_*`, `*_ood` variants | functional subsets | Subsets of the same 4,165 tracks / 23 dense columns | No | Yes (per-arm SVD, width 256) | None | Same store as above; spec in sidecar JSON | Fit on 408,399 loci (`*_ood`: 388,599 loci, chr20-22 held out of the fit) | `src/cpg_repr_benchmark/encode_atlas/campaign.py:109-139,274-288`; `extensions.py:41-56`; `outputs/encode_atlas_v1/embeddings/*.json` (36 sidecars); `configs/encode_atlas.yaml` |
| `context_only` | annotation baseline | Dense columns 0-17 only (context, gene region, cCRE, TSS distance) | No | Yes (no SVD; `variance_retained: null`) | None | -- | Reference-only | `campaign.py:111`; `outputs/.../context_only_2805a394.json` |
| `breadth_only`, `null`, `tracks_only` | controls | Breadth summaries only / nothing / tracks without dense | No | Yes | None | -- | Reference-only | `campaign.py:110-114`; `breadth_only_9fde35b4.json` |
| `panel/selected/N`, `panel/random_*/N` | compact panels | Track subsets chosen by discovery screen (selected) or randomly | No | Selection used attribution results on methylation reconstruction (discovery runs), so "selected" panels are not label-free in selection, only in construction | Selection step is indirectly methylation-informed (UNKNOWN whether it matters; flag) | `campaign.py:282-288`; `configs/encode_atlas.yaml` (`panel_sizes`) | Reference-only construction | `campaign.py:282-288` |
| `fm/cpgpt_locus_large`, `fm_cpgpt_locus_large_d3474a48` | frozen FM compressed to 256-D | CpGPT large 512-D DNA-sequence-encoder output, SVD to 256 (var. 0.838) | No | Compression yes; the source model is not (next row) | See CpGPT rows | `external_sha256 c08138cd...` | `n_fit_loci` 408,399 (`_ood`: 388,599) | `outputs/encode_atlas_v1/embeddings/fm_cpgpt_locus_large_d3474a48.json` |
| `fm/deepcpg_dna_locus` | frozen FM compressed to 256-D | DeepCpG HCC DNA-module 128-D (width capped by source) | No | Compression yes; source not label-free | See DeepCpG rows | `external_sha256 e7280d5f...` | 408,399 | `fm_deepcpg_dna_locus_1facb46c.json` |
| `add_fm/*` (fm_complementarity) | functional + FM concatenation | Union of an ENCODE group and a frozen FM store | No | Same caveats as the FM component | Same as the FM component | -- | -- | `campaign.py:275-278`; `extensions.py:56` |
| `cpgpt_locus_native` | methylation FM (sequence branch) | CpGPT small, DNA-sequence-encoder projection per CpG, 128-D, no positional encoding | No | Extraction uses no methylation input; the encoder is part of a model pretrained on methylation | **Catalog says `none`. Pretraining of CpGPT used methylation targets (EXTERNAL; not verifiable locally). Whether the DNA branch weights were shaped by methylation loss: UNKNOWN locally.** | `small.ckpt` symlinked from sibling `methylation-fm-benchmark/data/cpgpt/` (path in `scripts/cpgpt/README.md`) | `external_pretrained` (training CpGs/species/cohorts UNKNOWN) | `future_models.yaml:62-80`; `scripts/cpgpt/README.md`; `data/cache/representations/cpgpt_locus.h5.json` (`representation_mode: sequence`) |
| `cpgpt_locus_large_native` | methylation FM (sequence branch) | CpGPT large, 512-D | No | As above | As above | `large` | `external_pretrained` | `future_models.yaml:82-98`; `cpgpt_locus_large.h5.json` |
| `methylgpt_locus_native` / `_medium_native` | methylation FM (token table) | MethylGPT `encoder.embedding.weight` looked up by Illumina probe ID; no forward pass | No | Extraction yes; table was learned from sample methylation (EXTERNAL) | **Catalog says `none`, but the CpG token table is by construction a product of methylation-value pretraining (EXTERNAL; local README only says no sample values are consumed at extraction).** Checkpoint widths 64 (`tiny_epoch10`) and 256 (local "medium" folder, args.json layer_size=256; official identity unverified) | `future_models.yaml:100-135` | `external_pretrained`; coverage only 43,578 of 408,399 loci (364,813 out of vocabulary), so it cannot span the universe | `scripts/methylgpt/README.md`; `methylgpt_locus.h5.json` |
| `deepcpg_dna_locus_native` (HCC) / `_hepg2_native` | sequence CNN, trained on methylation | DeepCpG DNA module stem, 1001 bp window, output head stripped | No | Extraction yes; the DNA module was trained to predict single-cell methylation states | **Catalog says `none`. DNA module was trained against methylation targets of the HCC / HepG2 datasets (the README states the weights come from the DeepCpG model zoo and that the pretraining output head is removed; the targets being methylation is EXTERNAL knowledge of DeepCpG).** Training CpG set vs benchmark loci: UNKNOWN | `future_models.yaml:137-165`; `scripts/deepcpg/README.md` | `external_pretrained` | same files; `deepcpg_dna_locus*.h5.json` |
| `ntv3_pre_native` | genomic FM (sequence) | NTv3 650M pre-trained checkpoint (`InstaDeepAI/NTv3_650M_pre`), 32,768 bp window, 1536-D central-CG pooling (chr1-5 store: 131,757 loci) | No | Yes for the pre-trained checkpoint as used | `genomic_pretraining` (catalog); `posttrained: false` in the materialization sidecar. Pretraining data content beyond DNA sequence: UNKNOWN locally (the NTv3 "pre" name implies a pre-trained, not post-trained checkpoint; EXTERNAL) | HF `InstaDeepAI/NTv3_650M_pre` | `external_pretrained`; store covers chr1 (legacy atlas) or chr1-5 only, not genome-wide | `future_models.yaml:38-60`; `providers/ntv3_pre.py:1-30,199-200`; `ntv3_pre_chr1to5_native_v1.h5.json` |
| `control_random_stable` | control | Gaussian vector seeded by CpG id | No | Yes | None | -- | none | `representations/control_providers.py:34-42`; `configs/experiments/upscale_450k_epic/random_stable.yaml:19` |
| `control_constant` | control | Same constant vector for all loci | No | Yes | None | -- | none | `control_providers.py:44-51`; `constant.yaml:18` |
| `baseline_prior_mean` / `baseline_ridge_pca` | baselines (no locus representation) | Train-patient mean prior / PCA-ridge on observed shared probes | Not a locus embedding; ridge uses patient methylation by design | n/a | Uses methylation by design (they are baselines, not representations) | -- | none | `configs/experiments/upscale_450k_epic/baseline_*.yaml` |
| `functional_annotations_chr1` (legacy encoder store) | functional, learned | Same 4,165 tracks + 23 dense, passed through the legacy MehylPredictor functional encoder | No | **No** | **Yes** (section 2) | `data/external/functional/checkpoints/functional_encoder.pt` -> MehylPredictor `encode-wgbs-seed17-b128-c65536/checkpoints/last.pt` | UNKNOWN (train CpG split of that run vs this repo's protocols not recorded here) | `data/cache/representations/functional_annotations_chr1.h5.json`; `scripts/build_functional_embeddings.py:103,129`; section 2 |
| Proxy-track arms (`functional_annotations__proxy_*`, `ntv3_pre` proxy) | dropped | Same features, re-optimized to predict train-patient mean beta | Uses patient methylation | No | Yes (`objective: train_patient_mean_beta`) | `configs/proxy/*.yaml` | train protocol | `configs/proxy/functional_mean_beta_chr1.yaml:25-30`; `providers/functional_proxy.py:81`. Track dropped per CLAUDE.md; no current arm. |

Other bio-validation targets (phastCons100way, clock coefficients, EWAS sets) are evaluation targets, not representations
(`scripts/build_phastcons_coefficients.py`, `docs/EMBEDDING_EVALUATION.md`).

## 2. Methylation supervision in legacy encoders and checkpoints

### Facts

1. The "legacy functional encoder" (README:217-219 refers to its "methylation-supervised fitting") is the checkpoint
   `data/external/functional/checkpoints/functional_encoder.pt`. It is a symlink to
   `MehylPredictor/local_methyl_data/runs/encode_wgbs/runs/rna_methylation/genomewide/encode-wgbs-seed17-b128-c65536/checkpoints/last.pt`
   (verified with `ls -la`; also `scripts/bootstrap_local_data.py:66-72`, `data/README.md:21`).
2. The run's `metadata.json` and `config.resolved.yaml` (read-only) show: model `rna_methylation`, architecture
   `methylpredictor_final`, loss `beta_mse_weight: 1.0` + `locus_pearson_weight: 0.15`, `train_cpg_split: train`,
   `validation_cpg_split: validation`, run folder named `encode_wgbs`, 80 epochs, seed 17. So the checkpoint was trained
   with a methylation-value (beta) regression loss; the run name indicates ENCODE WGBS data. The full model is RNA +
   locus -> DNAm, the benchmark extracts only the functional-locus branch (`functional_provider.py:11-79`, prefixes
   `track_embedding.`, `dense_encoder.`, `locus_norm.`, `functional_ffn.`).
3. The functional branch is therefore NOT label-free: its weights were shaped by methylation targets on the train CpGs
   (and the RNA path of the full model). This contradicts a literal reading of `docs/BENCHMARK_V2.md:21-29` ("Only its CpG
   branch is migrated") if that text is applied to the produced embedding, and is why README:217-219 requires an audit.
   `docs/BENCHMARK_V2.md:3` and `docs/REFACTOR_AUDIT.md` already label the encoder as legacy with open audit status.
4. The benchmark-facing `functional_annotations_pca` does NOT use this checkpoint: `build_functional_pca_embedding.py`
   reads only `/dense`, `/track_indices`, `/track_indptr` of the raw store (docstring: "no supervised training against any
   methylation objective"). The `functional_annotations_chr1.h5` store built from the checkpoint still exists in
   `data/cache/representations/` and is used by the legacy smoke path (`docs/REPRESENTATION_MATERIALIZATION.md:42-92`),
   which that doc says is not valid for strict external-locus claims.
5. `scripts/export_functional_feature_store.py` copies the raw track/dense features out of the MehylPredictor cache;
   these are reference annotations (the HDF5 attrs say `patient_specific=False`). The cache build command and source
   hashes are recorded in `data/external/functional/locus_features_v1/manifest.json` (UCSC cpgIslandExt, GENCODE v50,
   cCRE Registry V4, 4,165 ENCODE regulatory tracks).
6. Proxy-trained checkpoints (`configs/proxy/*.yaml`) were methylation-supervised (train-patient mean beta) but the proxy
   track has been dropped; no current arm uses them (`CLAUDE.md`).
7. Methylation-FM-derived arms (CpGPT, MethylGPT, DeepCpG) carry catalog `supervision: none` while the underlying
   published models were trained on methylation. The catalog field therefore means "no further supervision applied in
   this repo", not "encoder never saw methylation". The sibling-pathway note in `configs/representations/README.md`
   lists the allowed values `none | masked_methylation | other` and has no value for "pretrained with methylation targets".

### Unknown

- Whether `functional_encoder.pt` was trained on CpGs that overlap this repo's train/held-out CpG protocols, and on which
  cohorts (the metadata records `train_cpg_split` but not the CpG identities in a file reviewed here).
- Whether any figure, table or result currently cited in README/docs was produced from the checkpoint-derived
  functional store rather than from the PCA store (the atlas arms demonstrably use the raw store; the masking configs
  use the PCA store). Older age/disease protocols referenced `functional_legacy.h5` but their config templates are
  noted as missing in `docs/REFACTOR_AUDIT.md`.
- The pretraining objective composition and data of CpGPT, MethylGPT, DeepCpG DNA modules as locally verified: the local
  READMEs only describe the extraction step. Treat statements here about their methylation pretraining as EXTERNAL.
- NTv3 pre-training data beyond DNA sequence.
- The identity of the local MethylGPT "medium" checkpoint (documented in the YAML as architecturally "large").

## 3. Feature families that can be directly or indirectly methylation-derived

What the shared feature store actually contains (FACT, from `track_contract.tsv` and `encode_atlas/features.py:15-24`):
4,165 ENCODE GRCh38 BED peak files; `encode_assay` values are exactly Histone ChIP-seq (1959), TF ChIP-seq (1673),
DNase-seq (533); `functional_block` values histone / tf_binding / accessibility / ctcf; no row has an assay name
matching WGBS, RRBS, bisulfite, or DNAme. 635 distinct targets; ENCODE release dates 2010-12-16 to 2018-07-12; output
types are peak calls (IDR/replicated/pseudoreplicated/plain peaks). Dense columns: 4 CpG context, 4 gene region, 9 cCRE
classes, 1 TSS distance, 5 breadth summaries.

| Family | In the store? | Direct methylation-derived? | Indirect link (what is verified, what is not) |
|---|---|---|---|
| ENCODE WGBS / RRBS / methylation-array tracks | No (assay list above) | Not present | If a future store adds them they would be direct leaks. The ENCODE accessions were not cross-checked against ENCODE metadata beyond the local `encode_assay` column. |
| Histone ChIP-seq peaks (H3K4me3, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K9me3, ...) | Yes, 1959 tracks | No (immunoprecipitation of histone marks) | Strong biological coupling with DNA methylation (e.g. H3K4me3 anti-correlated, H3K9me3/H3K36me3 correlated; EXTERNAL). This is a genuine association, not a data leak, but it is the likely mechanism of any reconstruction gain and must be described as such. |
| DNase-seq accessibility peaks | Yes, 533 tracks | No | Accessibility is coupled with hypomethylation (EXTERNAL). |
| TF ChIP-seq (1472 tf_binding + 201 CTCF) | Yes | No | Includes methylation-reader or methylation-machinery targets: ZBTB33 (8 tracks), MBD1 (2), MBD2 (2), DNMT1 (1), DNMT3B (1) (FACT, from `encode_target`). Their peaks are binding maps (ChIP), not methylation measurements, but their binding is methylation-dependent by biology; report as indirect. |
| cCRE Registry V4 classes (PLS, pELS, dELS, CA, TF, CA-CTCF, CA-H3K4me3, CA-TF; 9 one-hot incl. `none`) | Yes, dense columns 8-16 | UNKNOWN. cCRE definitions are, to my knowledge, built from DNase hypersensitivity plus H3K4me3, H3K27ac and CTCF ChIP-seq (EXTERNAL; not verifiable from the BED hash alone). The V4 methodology was not read locally. | The same marks are also present as raw tracks, so cCRE labels and tracks are partly redundant (the campaign already ablates `drop_dense_ccre`, `docs/ENCODE_ATTRIBUTION.md:75-77`). Whether V4 used any DNA-methylation input: UNKNOWN. |
| UCSC CpG islands (island/shore/shelf/open sea) | Yes, dense columns 0-3 (manifest: UCSC hg38 cpgIslandExt) | No: island calls are sequence-derived (GC content and observed/expected CpG), EXTERNAL | Island/shore/shelf are strongly associated with methylation state by definition of the field; recovery of these labels is a retention check (`docs/EMBEDDING_EVALUATION.md:3-20`, `docs/UMAP_BIOLOGY.md:20-24,45-60`). |
| GENCODE v50 gene region, TSS distance | Yes, dense columns 4-7, 17 | No | Promoter proximity correlates with methylation; not a leak. |
| Breadth (5 summaries of the retained tracks) | Yes, dense columns 18-22 | No (derived from the same peak tracks) | Rank-1 summary of the peak tracks; `drop_preserve_breadth` arms separate it. |
| ChromHMM / segmentation states | No (no such track name or `functional_block`) | -- | Not applicable. If added: ChromHMM models trained on histone marks may be methylation-free, but some published segmentations (e.g. Roadmap, WGBS-informed) use methylation: check before adding. |
| Conservation (phastCons) | Not in the store | -- | Only used as an evaluation target (`scripts/build_phastcons_coefficients.py`). |
| Array probe design (450K/EPIC/TCGA array loci) | Defines the locus universe for every arm (408,399 or 843,855 CpGs) | Not methylation values, but selection of loci is array-design driven (probes were chosen around promoters, CpG islands, known genes; EXTERNAL) | All arms inherit the same bias, so it does not differentiate arms but limits generality claims (genome-wide in name only). |
| CpG density / local sequence | Implicit in sequence FMs (CpGPT, DeepCpG, NTv3) and in island labels | No | Sequence FMs can learn CpG density and island structure directly from the DNA; an advantage in explaining methylation that is unrelated to ENCODE data. |
| Array-derived annotations (probe type I/II, SNP-at-probe, cross-reactive) | Not found in the store | -- | `probe_id` is only used as a join key for MethylGPT (`illumina_probe_grch38.parquet`). |

## 4. Open questions

1. Which of the legacy encoder checkpoint's training CpGs overlap the benchmark train/held-out protocols, and which
   cohorts (ENCODE WGBS samples vs others) were used? Needed to decide `downstream-unseen` vs `strict representation-OOD`.
2. Is any result in the paper draft or in `docs/ENCODE_RESULTS_*.md` generated from the checkpoint-derived store
   (`functional_annotations_chr1.h5`) rather than from the PCA store?
3. Should the catalog `supervision` vocabulary be extended (e.g. `pretrained_with_methylation_targets`) so that CpGPT,
   MethylGPT and DeepCpG are not all `none`?
4. Does cCRE Registry V4 depend on any DNA-methylation data, and are all ENCODE sources in the 4,165 tracks free of
   methylation-derived peak calls (e.g. IDR peaks from ChIP of methylation-related targets are binding data, but should be
   described separately)?
5. Should the ZBTB33/MBD1/MBD2/DNMT1/DNMT3B tracks (14 total) be ablated in a "methylation-machinery-free" arm to bound
   indirect coupling? No such arm exists today (check `arms.json` before concluding).
6. Do the upstream cCRE and island sources share the 450K/EPIC probe-design bias, and how does this limit the genome-wide
   claim?
7. What are the real pretraining data/objectives of the CpGPT sequence branch, the DeepCpG HCC/HepG2 DNA modules and
   NTv3 pre, as confirmed from the primary publications (not from local README summaries)?
8. Is the local MethylGPT "medium" checkpoint the official large model? And since MethylGPT covers only 43,578 loci,
   should it be dropped from the comparison (consistent with the no-intersection policy)?
9. For selected compact panels, the track selection used discovery-run attribution against methylation reconstruction;
   how is leakage of discovery into confirmation controlled (the docs say historical outputs are not confirmation)?
10. Is the PCA fitted on all loci (including held-out and `*_ood` chromosomes for the main arm) acceptable for the
    claim being made? Only the `*_ood` and 450K-fit variants restrict the fit set.
