# Regulatory representation: design contract and candidate arms

Documentation only (2026-10-02). No embedding, SVD, feature selection, training or code change was made. Tags:
**[V]** verified in a local file (cited), **[A]** assumption or external knowledge not verified in this repo.

## Purpose and scientific objective

Build a patient-independent, label-free, interpretable CpG-locus representation grounded only in ENCODE experimental
regulatory evidence, and test whether it captures methylation and regulatory properties better than sequence-based and
foundation-model locus embeddings. The masking benchmark is the evaluation instrument. The current discovery baseline
(`functional_annotations_pca`) mixes experimental tracks with handcrafted annotations and track-derived summaries; this
document defines cleaner candidate contracts so that annotation-recovery checks are no longer circular.
Related: `docs/REPRESENTATION_PROVENANCE.md`, `docs/ENCODE_ATTRIBUTION.md`, `docs/REFACTOR_AUDIT.md`.

## 0. Verified feature-store facts

Source: `data/derived/functional_annotations/reference_functional_features_tcga_array_all_v1_cpgidx_fixed.h5`
(attrs `patient_specific=False`, `n_cpg=408399`, `n_tracks=4165`, `dense_dim=23`) and the track catalog
`data/external/functional/locus_features_v1/regulatory/track_contract.tsv` (4,165 rows + header, sha256 in
`locus_features_v1/manifest.json`). **[V]**

- Structure **[V]**: datasets `cpg_idx`, `source_cpg_idx`, `dense` (408399 x 23, float16), and a CSR binary peak-overlap matrix
  `track_indptr` / `track_indices` (185,489,393 non-zeros; values implicitly 1). The tracks and the 23 dense columns are
  physically separate datasets: tracks are never inside `/dense`.
- **CSR column c is catalog row c (0-based row order), not `track_index`** (`track_index` is non-contiguous, 222..5828;
  `encode_atlas/features.py:load_catalog` inserts `feature_column = arange(n)`). Any column mask must use `feature_column`.
- Tracks are **binary peak-overlap** (ENCODE narrowPeak/bed peak calls intersected with each CpG), not fold-change or
  -log10 p signal. Output types: optimal IDR 1362, pseudoreplicated 1034, replicated 925, DNase "peaks" 533,
  pseudoreplicated IDR 222, conservative IDR 81, IDR 8. All GRCh38, released, 4,165 distinct experiments.
- No assay in the catalog is WGBS/RRBS/bisulfite/array methylation (assay values are exactly 3). A regex for "methyl" hits 4
  descriptions, all "dimethyl sulfoxide" treatments (ECC-1 DNase, CTCF, ESR1, GATA3/FOXA2 entry), not methylation data.
- Release dates 2010-12-16 to 2018-07-12; 635 distinct targets (DNase rows have no target); 344 distinct biosamples;
  biosample class: cell line 1993, tissue 1360, primary cell 541, in vitro differentiated 271.

### Counts per block (verified)

| Contract block | Definition (catalog) | Tracks |
|---|---|---|
| `histone` | `encode_assay == "Histone ChIP-seq"` (= `functional_block histone`) | **1,959** |
| `tf` | `encode_assay == "TF ChIP-seq"` (= `functional_block tf_binding` 1,472 + `ctcf` 201) | **1,673** |
| `accessibility` | `encode_assay == "DNase-seq"` (= `functional_block accessibility`) | **533** |
| Total | | **4,165** (1959 + 1673 + 533) |

Note: the repo's `functional_block` has four values; the contract's `tf` block merges `tf_binding` and `ctcf`. Decide and
record this merge explicitly (CTCF is 201 tracks, one target).

### The 23 dense columns (store order) **[V]** (`outputs/encode_atlas_v1/dense_features.json`, `encode_atlas/features.py`)

| Cols | Names | Origin | Class |
|---|---|---|---|
| 0-3 | `context:island`, `context:shore`, `context:shelf`, `context:open_sea` | UCSC cpgIslandExt hg38 | handcrafted genomic annotation |
| 4-7 | `gene_region:promoter`, `gene_region:exon`, `gene_region:intron`, `gene_region:intergenic` | GENCODE v50 | handcrafted genomic annotation |
| 8-16 | `ccre:none`, `ccre:PLS`, `ccre:pELS`, `ccre:dELS`, `ccre:CA`, `ccre:TF`, `ccre:CA-CTCF`, `ccre:CA-H3K4me3`, `ccre:CA-TF` | cCRE Registry V4 | ENCODE-derived annotation |
| 17 | `log10_tss_distance` | GENCODE v50 TSS | handcrafted |
| 18-22 | `breadth:accessibility`, `breadth:histone`, `breadth:ctcf`, `breadth:tf_binding`, `breadth:core` | linear summaries of the tracks (`breadth_matrix`; verified at load by `tracks @ breadth_matrix`, atol 5e-4) | derived from the tracks |

Implementation hint: a contract is a pair (boolean mask over 4,165 catalog rows, subset of dense columns); `FeatureData.matrix(tracks, dense)`
already takes both. For a tracks-only contract use `dense=[]`. Breadth must never be included (it is a function of the tracks).

## 1. Representation design contract

1. Inputs: only ENCODE experimental tracks (the 4,165 binary peak-overlap tracks above).
2. Patient-independent: reference annotation only (store attr `patient_specific=False`).
3. No methylation values, no beta, no RNA, no task tokens at any stage; label-free (no supervised fitting; unsupervised
   compaction only if declared).
4. No handcrafted genomic annotations as input.
5. Blocks: `histone` 1,959; `tf` 1,673; `accessibility` 533.
6. **Excluded from the main representation**: CpG island/shore/shelf/open-sea (dense 0-3), gene-region annotations (4-7), cCRE classes
   (8-16), TSS distance (17), and all derived breadth features (18-22): all 23 dense columns listed in section 0. They become
   validation targets or controls (section 4), not inputs.
7. A frozen-store checkpoint (`functional_encoder.pt`, methylation-supervised) is never used; only the raw store is.

## 2. Methylation-linked TF tracks and contracts

### The 14 tracks **[V]** (count verified: 14; rule `encode_target in {ZBTB33, MBD1, MBD2, DNMT1, DNMT3B}` on `track_contract.tsv`; per target ZBTB33 8, MBD1 2, MBD2 2, DNMT1 1, DNMT3B 1)

All are `TF ChIP-seq`, block `tf_binding`, output type optimal IDR thresholded peaks. CSV with CSR column index:
`scratchpad/methylation_linked_tracks_14.csv` (CSR column = catalog row index).

| CSR col | track_id (= experiment) | file | target | biosample | class | assay |
|---|---|---|---|---|---|---|
| 75 | ENCSR000BPZ | ENCFF593ZJA | ZBTB33 | A549 | cell line | TF ChIP-seq |
| 342 | ENCSR000BHC | ENCFF773OQL | ZBTB33 | GM12878 | cell line | TF ChIP-seq |
| 343 | ENCSR542FLV | ENCFF475DID | ZBTB33 | GM12878 | cell line | TF ChIP-seq |
| 883 | ENCSR156CWW | ENCFF186YGH | DNMT3B | HepG2 | cell line | TF ChIP-seq |
| 954 | ENCSR260UJI | ENCFF960FKQ | MBD1 | HepG2 | cell line | TF ChIP-seq |
| 955 | ENCSR396QWK | ENCFF515DUD | MBD1 | HepG2 | cell line | TF ChIP-seq |
| 1046 | ENCSR000BNA | ENCFF943WRA | ZBTB33 | HepG2 | cell line | TF ChIP-seq |
| 1205 | ENCSR987PBI | ENCFF549TVW | DNMT1 | K562 | cell line | TF ChIP-seq |
| 1307 | ENCSR221GAN | ENCFF617QSK | MBD2 | K562 | cell line | TF ChIP-seq |
| 1494 | ENCSR876GXA | ENCFF556STK | ZBTB33 | K562 | cell line | TF ChIP-seq |
| 1679 | ENCSR940MHE | ENCFF464QAL | MBD2 | MCF-7 | cell line | TF ChIP-seq |
| 1719 | ENCSR231YFE | ENCFF780WLS | ZBTB33 | MCF-7 | cell line | TF ChIP-seq |
| 3413 | ENCSR345YWJ | ENCFF882UHR | ZBTB33 | liver | tissue | TF ChIP-seq |
| 3414 | ENCSR516HUP | ENCFF727ZIT | ZBTB33 | liver | tissue | TF ChIP-seq |

(Catalog `track_index` values for the same rows: 301, 593, 594, 1236, 1307, 1308, 1399, 1606, 1708, 1895, 2175, 2215, 4662, 4663.
Do not confuse them with CSR columns.)

Their peaks are ChIP binding maps, not methylation measurements; the proteins read or write DNA methylation, so the
binding pattern is methylation-dependent by biology **[A]**. Adjacent targets **not in the 14 (not reclassified)** **[V]**
counts from the catalog: CTCF 201 (methylation-sensitive binding **[A]**), SETDB1 2, EHMT2 3, KDM1A 7, KDM5A 3, KDM5B 1,
KDM4A 1, KDM4B 2, KDM3A 1, KDM6A 1, EZH2 18, SUZ12 7, HDAC1 5, HDAC2 9, SIRT6 2, TRIM28 4, ZNF274 5, NRF1 8, SP1 9,
ZBTB7A 3 and other ZBTB-family targets. Absent from the store: DNMT3A, DNMT3L, TET1/2/3, UHRF1/2, MECP2, MBD3-6, ZFP57, CXXC1.

### Candidate contracts (no track is deleted from the store; contracts are column masks)

- `regulatory_clean` (**PRIMARY candidate**): 4,165 - 14 = 4,151 tracks. Claim: "experimental regulatory evidence without explicit methylation machinery".
- `regulatory_all_experimental`: all 4,165 tracks, no dense columns.

Neither includes any dense column. The choice between them is open; both should be reported (the difference bounds the effect of
direct methylation-machinery evidence).

## 3. Candidate representations

| Name | Block set | Dense | Exclusions | Status |
|---|---|---|---|---|
| Functional-All-Legacy (`functional_annotations_pca`) | 4,165 tracks | all 23 | none | existing discovery baseline (SVD 256, seed 17) |
| Regulatory-AllExperimental | histone + tf + accessibility = 4,165 | none | dense 0-22 | candidate, not built |
| **Regulatory-Clean** | histone + tf + accessibility, minus rule below = 4,151 | none | (a) catalog rows with `encode_target in {ZBTB33, MBD1, MBD2, DNMT1, DNMT3B}` (14 rows; by rule on `encode_target`, not by ids); (b) all 23 dense columns (context, gene region, cCRE, TSS, breadth) | candidate (primary), not built |
| Regulatory-Histone | `histone` only, 1,959 tracks | none | all non-histone tracks, all dense | mechanistic/sufficiency arm, not built |
| Regulatory-Compact (future) | same information as Regulatory-Clean | none | label-free redundancy pruning only (e.g. track-track correlation or duplicate removal from the tracks themselves); must NOT select features using methylation reconstruction or any methylation performance | NOT implemented |

Implied block counts for Regulatory-Clean (all 14 are in `tf`): histone 1,959; tf 1,673 - 14 = 1,659; accessibility 533.
Fit scope of any compaction (SVD) must be declared (transductive on all 408,399 loci vs fit subset), as in `functional_annotations_pca`.
Note `outputs/encode_atlas_v1` panels (`panel/selected/N`) used methylation-informed selection and are not a basis for Regulatory-Compact.

## 4. Validation separation

1. **Representation design**: only the contract above.
2. **Source-retention controls** (check that the encoding retains the information of its source; not independent evidence): CpG
   island/shore/shelf/open-sea, gene region, TSS distance, cCRE classes, breadth. The cCRE registry is ENCODE-derived and its
   definition uses DNase and H3K4me3/H3K27ac/CTCF ChIP **[A]**, which are also input tracks, so cCRE recovery is partly circular. CpG
   island labels are sequence-derived **[A]**.
3. **Independent biological validation (future)**: external DNA-methylation programs (EWAS sets, clocks), 3D genome/compartments,
   replication timing, and other annotations not used as input (phastCons exists as a target, `scripts/build_phastcons_coefficients.py`).
   Caveat: any ENCODE-derived resource (cCRE, ENCODE-based segmentations) overlaps in source with the inputs and cannot be called independent.

## 5. Residual provenance ambiguities

1. Other methylation-adjacent targets stay in `regulatory_clean`: CTCF (201), SETDB1, EHMT2, KDM family, EZH2/SUZ12, HDACs, TRIM28, ZNF274 and
   many ZBTB-family factors **[V]** present; their coupling to methylation is biological **[A]**. Histone marks (H3K4me3, H3K9me3, H3K36me3,
   H3K27me3) are strongly coupled to methylation **[A]**; this is a genuine association, not a leak, and probably the mechanism of any gain.
2. cCRE and any ChromHMM-type derivation: not in the contract; whether cCRE V4 used methylation input is unknown **[A/unknown]**.
3. ENCODE biosample overlap with methylation cohorts: biosamples are mostly cell lines (1,993) and tissues (1,360); some (e.g. K562, GM12878,
   HepG2, liver) may coincide with samples in methylation datasets used for evaluation. Overlap with the TCGA/GSE evaluation cohorts was not checked.
   Tracks are patient-independent in the sense of not using benchmark patients **[V by `patient_specific=False`]**, but donor identity of ENCODE
   samples is unknown.
4. Array probe design bias: the 408,399-locus universe comes from the Illumina array registry (probes concentrated at promoters/islands **[A]**),
   while ChIP coverage is genome-wide; locus-level signal may be confounded by probe selection.
5. Checkpoint-derived store: `functional_annotations_chr1.h5` (legacy encoder `last.pt`, beta-MSE loss) is methylation-supervised
   (`REPRESENTATION_PROVENANCE.md` section 2); the raw store used here is not. The raw store was exported from the sibling MehylPredictor cache
   (`manifest.json` build command, git commit `613492c`); the peak files are verified (4165/4165) but their ENCODE accessions were not re-queried.
6. Peak-overlap tracks are binary and prevalence-dependent: tracks with broad peaks (many histone marks, DNase) correlate with CpG density and
   GC content (CpG-rich regions are accessible and H3K4me3-marked) **[A]**; CpG-density confounds were not measured. Output types are heterogeneous
   (IDR, pseudoreplicated, plain DNase peaks), so peak width and sensitivity differ across tracks **[V]** counts above.
7. Catalog dated 2010-2018 on GRCh38; DNase tracks have no target; one DNase track (spleen, ENCSR850YHJ) is `bed3+`, others narrowPeak; 8 tracks carry `fallback_peak`.
8. Redundant experiments: several biosamples/targets are replicated experiments (e.g. two ZBTB33 GM12878 experiments, two MBD1 HepG2 experiments); breadth of one factor
   across multiple experiments inflates its weight in an SVD.
9. No existing campaign arm removes the 14 methylation-linked tracks: `outputs/encode_atlas_v1/arms.json` (102 arms) has none; only screening groups
   `target/ZBTB33`, `target/MBD1`, `target/MBD2`, `target/DNMT1`, `target/DNMT3B` exist in `groups.json` **[V]**.
10. The `tf` block merge of `ctcf` and `tf_binding` is a naming choice (see section 0).

## 6. Not done

No embedding built, no SVD or PCA fit, no feature selection or redundancy pruning, no training or evaluation, no change to the store, catalog,
configs, code or tests, no registration in `configs/representations/`, no git commit. The 14 tracks were not removed from any file. ENCODE
accessions were not re-validated online. Overlap between ENCODE biosamples and evaluation cohorts, CpG-density confounds and the cCRE methylation
dependency were not measured. Regulatory-Compact is not designed beyond the rule above.
