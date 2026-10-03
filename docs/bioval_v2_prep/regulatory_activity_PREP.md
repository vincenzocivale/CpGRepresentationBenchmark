# regulatory_activity (FANTOM5 enhancer activity) - data prep report

## 0. Pre-declaration (written before any CpG x enhancer overlap was computed)

Written 2026-10-02, after downloading the data and inspecting only file formats / enhancer counts, before intersecting with benchmark CpGs.

- Primary target enhancer set: FANTOM5 hg38 reprocessed transcribed enhancers (`F5.hg38.enhancers.bed.gz`, Dalby/Rennie/Andersson 2017, Zenodo 10.5281/zenodo.556775), phase 1+2, 63,285 bidirectional-CAGE loci (this is the "permissive" set re-called on hg38). Justification: it is the only enhancer set distributed natively on GRCh38 (no liftover). The Andersson 2014 "robust" (~38k) subset is not distributed as a hg38 file under /5/datafiles/ (the `latest/extra/Enhancers/` and `phase1.3/extra/Enhancers/` directories contain only permissive hg19 BEDs). No robust-set lift was fabricated. Robustness is instead handled by the expression-breadth threshold below.
- Normalization: RLE-normalized TPM as provided (`F5.hg38.enhancers.expression.tpm.matrix.gz`); activity = log2(TPM + 1).
- Expressed-in-sample threshold: TPM >= 1 (per enhancer, per library).
- "Informative profile" threshold (primary): enhancer expressed (TPM >= 1) in >= 5 libraries; counts also reported for N = 1, 3, 10, 20.
- Collapsing: libraries -> FANTOM5 sample category (tissues / primary cells / cell lines; time-course and fractionation/perturbation libraries are kept at sample level only) -> biological-replicate-collapsed label (strip donor/pool/biol_rep/rep suffix); value = mean of log2(TPM+1) over member libraries.
- Primary CpG-enhancer mapping: 0-based cytosine (pos-1) inside the BED interval [start, end).
- Sensitivity windows: |cytosine - enhancer midpoint| <= 500 bp and <= 1000 bp (midpoint = BED thickStart, the inferred mid position).
- Background candidate pool: CpGs >= 5 kb from any FANTOM5 enhancer interval (flag `bg_enh5k`), and additionally >= 5 kb from any GENCODE v50 transcript TSS (flag `bg_enh5k_tss5k`). Distances to nearest GENCODE TSS and nearest FANTOM5 CAGE-peak TSS provided as matching covariates. ENCODE cCRE is not used to define any target.

## 1. Input-circularity check (representation inputs)

Checked `data/external/functional/locus_features_v1/manifest.json` (source_versions: UCSC cpgIslandExt hg38, GENCODE v50, ENCODE cCRE Registry V4; annotation_schema `annotation_core_v1`), `regulatory/manifest.json`, `regulatory/track_contract.tsv` (4,165 ENCODE experiments: 1,959 histone ChIP-seq, 1,673 TF ChIP-seq, 533 DNase-seq; zero rows mention FANTOM/CAGE/eRNA/enhancer-RNA) and the chr1 catalog parquet schema (locus_key, chrom, position, chrom_code only). No grep hit for fantom/cage in src/ python.
- No FANTOM5 / CAGE / eRNA track is an input to `functional_annotations_pca` or the regulatory ENCODE-track embeddings. No direct source retention.
- Indirect, non-FANTOM overlap exists and is flagged: (a) ENCODE cCRE V4 (includes enhancer-like classes: pELS/dELS, CTCF-bound) is in `functional_annotations` inputs; (b) H3K27ac/H3K4me1/DNase/TF ChIP tracks (regulatory family inputs) mark active enhancers, which FANTOM5 bidirectional CAGE loci enrich for; (c) GENCODE v50 TSS/gene annotations are an input and also used here as a TSS covariate. Therefore a representation-vs-FANTOM5 signal is expected to be partly mediated by shared ENCODE chromatin evidence; FANTOM5 is an independent assay (RNA 5' capture, not ChIP/DNase) and no FANTOM-derived target uses cCRE, but this is not full independence. Matching on TSS distance (provided) and, later, on cCRE/ chromatin-derived covariates is advisable.

## 2. Sources (all natively GRCh38; no liftover performed, no tool needed; chain/pyliftover unused)

| file | source URL | note |
|---|---|---|
| F5.hg38.enhancers.bed.gz | fantom.gsc.riken.jp/5/datafiles/reprocessed/hg38_latest/extra/enhancer/ | BED12, 63,285 loci, chr1-22,X,Y |
| F5.hg38.enhancers.expression.tpm.matrix.gz | same | 63,285 x 1,829 libraries, RLE TPM (87.6 MB) |
| Human.sample_name2library_id.txt | fantom.gsc.riken.jp/5/datafiles/latest/extra/Enhancers/ | name <-> CNhs (1,829) |
| HumanSamples2.0.sdrf.xlsx | .../reprocessed/hg38_latest/basic/ | category (tissues/primary cells/cell lines/time courses/fractionations) via description-name join |
| hg38_fair+new_CAGE_peaks_phase1and2.bed.gz | .../hg38_latest/extra/CAGE_peaks/ | CAGE-peak representative TSS (thickStart), 209,374 peaks on chr1-22,X,Y |
| gencode.v50.annotation.gtf.gz | ftp.ebi.ac.uk GENCODE release_50 | 388,474 distinct (chrom, transcript TSS) |
| ontology obo, READMEs, 00MD5SUM.txt | FANTOM5 | auxiliary |

Retrieval date 2026-10-02; sha256 in `data/derived/bioval_v2/regulatory_activity/MANIFEST.json` (written with the sibling `provenance.write_manifest`; universe via sibling `load_benchmark_universe()`; sibling `harmonize`/`overlap_audit` not needed since sources are native hg38 and the audit is interval-based).

## 3. Derived files (data/derived/bioval_v2/regulatory_activity/)

- `enhancers.parquet` (63,285): chrom, start0, end, pos1_start, pos1_end, enhancer_id, mid0, width, score, n_samples_expressed (TPM>=1), build.
- `activity_sample_log2tpm.npy` float16 [63285 x 1829] + `activity_sample_index.parquet` (cnhs, name, category, ff_ontology, group_label, collapsed_key).
- `activity_collapsed_log2tpm.npy` float32 [63285 x 638] + index (249 cell-line, 230 primary-cell, 159 tissue groups; mean of log2(TPM+1)).
- `cpg_enhancer_map.parquet` (408,399 rows): cpg_idx, chrom, pos, enhancer_id (nearest by interval), dist_to_enhancer_bp (0 if inside), dist_to_enhancer_mid_bp, in_enhancer, in_window500, in_window1000, dist_to_gencode_tss_bp, dist_to_cage_peak_tss_bp, bg_enh5k, bg_enh5k_tss5k.
- `audit.json`, `audit_per_chrom.csv`, `audit_expressed_thresholds.csv`, `MANIFEST.json`.
- Helper `enhancer_profile_similarity(pairs)` in `scripts/bioval_v2/prepare_fantom5.py` (cpg_idx pairs -> Pearson over collapsed log2 profiles; NaN if not in an enhancer, enhancer expressed in < 5 libraries, or zero variance; options level='sample', categories=[...], enhancer_ids=True). Smoke-tested on 2,000 random in-enhancer pairs only for NaN handling; no embedding statistics computed. Mapping verified against an O(nm) brute force on chr21 (exact agreement for in_enhancer, distance, both windows).

## 4. Overlap audit (universe 408,399; chrX/chrY absent from universe)

| | CpGs | fraction | enhancers covered |
|---|---|---|---|
| inside enhancer interval (primary) | 12,551 | 3.07% | 8,245 |
| within 500 bp of midpoint | 29,184 | 7.15% | 15,609 |
| within 1 kb of midpoint | 52,953 | 12.97% | 21,797 |
| background >=5 kb from any enhancer | 257,326 | 63.0% | - |
| background >=5 kb from enhancer and GENCODE TSS | 59,315 | 14.5% | - |

Per-chromosome table: `audit_per_chrom.csv` (primary fraction 2.4% chr16 to 3.6% chr10/chr13/chr17/chr21).
Enhancer width (as provided): median 282 bp, mean 295, IQR 183-375, min 2, max 2,804 (so ~300 bp, but wide tail).
Informative profile (TPM>=1 in >=N libraries), CpGs inside an enhancer: N=1 12,504; N=3 12,113; N=5 (primary) 11,490 (7,394 enhancers); N=10 9,894; N=20 7,708 (4,638 enhancers).
Sparsity: 90.8% of matrix entries have TPM = 0; 1.39% have TPM>=1. Enhancer n_samples_expressed (TPM>=1): median 9, IQR 4-24, p90 57, max 1,788; 1,751 enhancers never reach TPM>=1; 44,484 enhancers expressed in >=5 libraries.
Median distance to nearest GENCODE TSS: 969 bp for in-enhancer CpGs vs 564 bp for others (descriptive; shows a TSS-distance covariate imbalance that matching must handle). Median distance to nearest enhancer over all CpGs 9.6 kb.

## 5. Problems / caveats

- Genome build: none (native hg38). The enhancer set is the reprocessed permissive-equivalent set; the Andersson 2014 robust (~38k) list is not available as hg38 and was not lifted.
- Sample ontology: no CNhs column in the sdrf; mapping is CNhs -> name (name2library file) -> sdrf description -> category. 27 of 1,829 libraries did not match (category `unmapped`, sample-level only, excluded from collapsed level). Time-course (780) and fractionation/perturbation (21) libraries are retained at sample level only. Collapsing is by regex suffix stripping; 638 groups, many with a single library (median group size 1, max 7), so replicate-level collapse is partial and group counts are not independent tissue counts.
- Tissue category is heterogeneous (pooled adult tissue RNA, fetal, cell lines); no fine ontology (UBERON/CL) collapse was done (obo file downloaded but unused).
- Mapping uses nearest enhancer interval; overlapping enhancer intervals resolve to nearest midpoint. Distances are cytosine (0-based pos-1) to interval [start0, end).
- Enhancer score column is max pooled expression of TCs, unused.
- License: FANTOM5 data are released under CC BY 4.0 (verified 2026-10-02 from the FANTOM data page footer and the Zenodo record metadata, saved copies and sha256 in `CITATIONS.md`; the statement is on the web page, not inside the downloaded files; cite Andersson et al. 2014 Nature 507:455, Dalby/Rennie/Andersson Zenodo 10.5281/zenodo.556775, full list in `CITATIONS.md`). GENCODE: open (EMBL-EBI terms).
- Downloads from fantom.gsc.riken.jp were slow/interrupted; the 87.6 MB TPM matrix was resumed with curl -C and verified with `gzip -t` and a complete parse (63,285 x 1,829).

## 6. Frozen matching policy and lists (written/produced before any embedding metric)

Script `scripts/bioval_v2/freeze_fantom5_matching.py`; outputs and sha256 in `data/derived/bioval_v2/regulatory_activity/FROZEN.json` (each file built twice, byte-identical sha256). License verified as CC BY 4.0 (see `CITATIONS.md`).

- Positives: all 12,551 CpGs inside a FANTOM5 enhancer interval (sensitivity: 29,184 within 500 bp of midpoint).
- Pool choice (deviation from the "proposal" in BIOLOGICAL_VALIDATION_V2.md, justified by feasibility, decided from covariates only): the main pool is `bg_enh5k` (257,326), NOT `bg_enh5k_tss5k`. Reason: only 22% of positives lie >= 5 kb from a TSS (median 969 bp), so matching positives to the TSS-filtered pool on tss_dist keeps only 3,349 pairs (73.3% dropped) with residual tss_dist SMD -0.117 (>0.1) - a selected, unrepresentative subset. With `bg_enh5k` the tss_dist imbalance is removed by matching. The tss5k-pool design is frozen as a sensitivity list (`..._tss5kpool_seed17`), with the above drop/balance caveat.
- Matching (`match_controls`): exact chromosome x genomic context; numeric calipers = 0.25 x SD over the 408,399-CpG universe on cpg_density (1.037), cpg_density_hg38 (7.54), gc_content (0.0245), tss_dist (0.258 log10 units); nearest on pool-standardized distance; 1:1 without replacement; seed 17. Distance to nearest enhancer is not matched. probe_type is not matched (6,429 NaN kept, reported as an 'unknown' level; no CpG dropped for it).
- Folds: `chromosome_blocked_folds(n_folds=5, seed=17)` on the 22 universe chromosomes.
- Results (primary): 11,575 matched pairs; 976 positives dropped (7.78%) before analysis, identical for every representation (list in `..._dropped_positives.parquet`). |SMD| after: numeric <= 0.006, probe_I 0.053, probe_unknown -0.013, context 0 (exact). SMD before: cpg_density -0.256, cpg_density_hg38 0.026, gc 0.122, tss_dist 0.191, shore 0.124. All < 0.1 after.
- Window500 sensitivity: 26,880 pairs, 2,304 dropped (7.9%), max |SMD| 0.020. tss5k-pool sensitivity: 3,349 pairs, 73.3% dropped, tss_dist SMD -0.117 (fails <0.1; reported as caveat).
- Activity-similarity pairs (secondary; informative CpGs = in enhancer expressed TPM>=1 in >=5 libraries: 11,490 CpGs, 7,394 enhancers): `sample_pairs(seed=17)` 100k intra (equal over the 6 distance bins; bins 0 and 1 are only feasible in part: 9,760 and 4,599 sampled) + 100k inter = 179,507 sampled; pairs within the same enhancer dropped (8,490, similarity trivially 1) -> 171,017 final (100,000 inter; intra by bin 1,452 / 4,417 / 15,149 / 16,667 / 16,666 / 16,666); none NaN. Profile similarity = Pearson on the 638 collapsed log2(TPM+1) profiles, precomputed (mean 0.098, SD 0.167). Files: `frozen_activity_pairs_seed17.parquet`.
- Frozen files: `frozen_enhancer_membership_seed17.parquet` (primary: 23,150 rows = 11,575 + 11,575; columns cpg_idx, chrom, pos, label, matched_to, enhancer_id, fold, context, covariates, probe_type), `..._win500_seed17`, `..._tss5kpool_seed17`.
