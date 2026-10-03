# external_methylation_programs: Loyfer 2023 WGBS atlas preparation (facts only)

> **UPDATE (provenance revision): sections 1-5 below describe v1 (filename-rule mapping, sample-level donors, 203 samples). They are SUPERSEDED where they conflict with section 6 (published Table S1 mapping, patient-level donors, 205 samples, verdict on markers, frozen pairs).** v1 files kept as `*_v1*`.

Code: `scripts/bioval_v2/{loyfer_cpg_index,loyfer_common,loyfer_extract,loyfer_aggregate,prepare_loyfer,loyfer_markers,loyfer_profile_similarity,loyfer_manifest}.py`.
Outputs: `data/derived/bioval_v2/external_methylation_programs/` (MANIFEST.json written with the sibling package's `write_manifest`/`overlap_audit`; `load_benchmark_universe()` set verified identical to the universe used here).
No GPU, no embeddings loaded, nothing under outputs/. Extraction used 16 worker processes (~3 min wall).

## 1. Inventory and coordinate convention
- 253 `hg38.pat.gz` files (93 GB) in `data/external/wgbs_atlas/hg38_pat/`; **only .pat.gz were downloaded locally** (no .beta; filelist.txt lists 506 beta/pat = hg19+hg38 pairs). Counts derived directly from pat.
- 207 atlas samples (GSE186458, GSM5652176...) + 46 `CNVS-NORM-cfDNA/WBC` samples (GSM6810003...; 23 cfDNA + 23 WBC, matched-donor controls). The 46 are extracted (counts file) but excluded from cell-type matrices. Donor id = Z-suffix of the file title (sample-level id; GEO "patient id" only exists for the cfDNA/WBC pairs). Table: `sample_inventory.tsv` (gsm, label, sample_id, group, flag, tissue, age, sex).
- 82 fine labels (tissue-celltype). Mapped to the published atlas groups (column names of UXM_deconv `Atlas.U25.l4.hg38.full.tsv`): **39 groups present locally with 203 samples**; the published atlas has 40 columns, the extra one (Megakaryocytes) has no local samples. 4 atlas samples unassigned (3 Endometrium-Epithelial, 1 Lung-Pleura; no corresponding atlas group). **The sample-to-group mapping is ours (file names/GEO metadata), not a published table**; ambiguous: Endocrine (colon/gastric/small-int -> organ epithelium) and Kidney podocytes (-> Kidney-Ep). Sanity check below supports it (marker target mean beta 0.17 vs 0.89 elsewhere).
- Pat convention: column 2 = **global 1-based wgbstools CpG index** over chr1..22,X,Y,M (confirmed: indices are monotone across chromosomes, chr1 max 2,375,158 < chr2 min 2,375,161). wgbstools `CpG.bed.gz` was not available locally; **regenerated from hg38.fa.gz**: 29,401,795 CpGs (pat max index 29,401,794; consistent with ~29.4M), coordinate = 1-based cytosine of CG. chr1 2,375,159 CpGs. Validation: for 100% of 9,520 + 979 published marker blocks, block start == our pos(startCpG), chrom matches, end == pos(endCpG-1)+1 (endCpG exclusive, lenCpG == endCpG-startCpG). 
- **Universe**: `data/cpg/registries/array_cpg_map.parquet` = 408,399 CpGs (chr1-22 only, no X/Y), identical (cpg_idx, chrom, pos) to `load_benchmark_universe()`. Its cpg_idx is the benchmark id (the master registry uses a different id `chr*1e9+pos`; all 408,399 are in master by chr,pos). **100.0% (408,399/408,399) of universe positions are C of a CG in hg38.fa** => 1-based cytosine convention confirmed, no off-by-one.

## 2. Matrices (`loyfer_celltype_beta.h5`, `loyfer_sample_counts.h5`)
- All 408,399 rows kept (cpg_idx, chrom, pos, in universe row order). Per-sample: meth_count, cov (uint16, no clipping needed beyond 65535 check), sample_beta (float16). Per-group (39): beta float16 with NaN, n_valid_donors, mean_cov, mask_primary, mask_lenient.
- Pre-declared rules: sample-CpG valid iff cov >= 10. Group beta = unweighted mean over donors' valid betas. **mask_primary = n_valid_donors >= min(2, n_samples_in_group)** (>=2 donors; 1 for single-sample groups); mask_lenient = >=1 valid donor (the task's "or >=1 donor with cov>=10" literal reading equals this).
- Raw mean genome-wide covered CpGs per sample: 27.89M of 29.4M; median per-sample CpG depth 36x.
- Overall: mean fraction of sample-CpG valid (cov>=10) 0.964; CpGs observed (primary) in all 39 groups: 0.897 (lenient 0.903); in >=20 groups: 0.986; in >=1 group (lenient): 0.9988; median groups per CpG 39.
- Per cell type coverage of the 408,399 and markers (markers = published U25 atlas union U250 candidates, all in universe-or-not counts):

```
                            group  n_samples  frac_lenient  frac_primary  median_mean_cov  blocks  cpgs  in_universe
                       Adipocytes        3.0        0.9905        0.9839             43.0     115   743           11
                       Bladder-Ep        5.0        0.9913        0.9848             34.0     250  1840           19
                          Blood-B        5.0        0.9910        0.9784             31.0     250  2152           23
                     Blood-Granul        3.0        0.9949        0.9880             42.0     250  1967           38
                 Blood-Mono+Macro       11.0        0.9937        0.9907             40.0     250  1708           22
                         Blood-NK        3.0        0.9922        0.9828             34.0     250  2050           30
                          Blood-T       22.0        0.9935        0.9905             32.0     250  1977           54
                      Bone-Osteob        1.0        0.9809        0.9809             39.0     250  1939           29
                  Breast-Basal-Ep        4.0        0.9891        0.9811             30.0     250  2433           62
                Breast-Luminal-Ep        3.0        0.9949        0.9886             36.0     250  1737           24
                         Colon-Ep        8.0        0.9927        0.9886             34.0     250  1959           27
                      Colon-Fibro        2.0        0.9887        0.9757             42.0       4    53            1
                     Dermal-Fibro        1.0        0.9707        0.9707             35.0     250  1952           20
                         Endothel       19.0        0.9964        0.9944             38.0     151  1070           25
                       Epid-Kerat        1.0        0.9702        0.9702             36.0     250  2028           25
                       Eryth-prog        3.0        0.9967        0.9919             48.0     250  2072           28
                     Fallopian-Ep        3.0        0.9844        0.9708             37.0     250  1705           22
                      Gallbladder        1.0        0.9753        0.9753             30.0     250  2516           69
                       Gastric-Ep       11.0        0.9948        0.9919             40.0     250  1834           17
                     Head-Neck-Ep       13.0        0.9945        0.9919             36.0     250  1759           16
                     Heart-Cardio        6.0        0.9937        0.9893             29.0     250  2294           51
                      Heart-Fibro        4.0        0.9849        0.9748             32.0     250  1818           32
                        Kidney-Ep        8.0        0.9940        0.9905             39.0     250  2036           50
                        Liver-Hep        6.0        0.9932        0.9892             43.0     250  2329           57
                    Lung-Ep-Alveo        3.0        0.9936        0.9890             50.0     250  1991           18
                     Lung-Ep-Bron        3.0        0.9950        0.9887             50.0     250  1707           11
Megakaryocytes (no local samples)        NaN           NaN           NaN              NaN     250  2067           39
                           Neuron       10.0        0.9921        0.9883             36.0     250  2125           76
                        Oligodend        4.0        0.9899        0.9827             35.0     250  2259           62
                         Ovary-Ep        1.0        0.9117        0.9117             22.0     250  1896           30
                  Pancreas-Acinar        4.0        0.9953        0.9885             43.0     250  2869           67
                   Pancreas-Alpha        3.0        0.9885        0.9780             35.0     250  1707           22
                    Pancreas-Beta        3.0        0.9872        0.9768             32.0     250  1863           40
                   Pancreas-Delta        3.0        0.9863        0.9721             31.0     250  1596           16
                    Pancreas-Duct        4.0        0.9907        0.9815             32.0     250  1761           21
                      Prostate-Ep        4.0        0.9919        0.9851             39.0     250  1689           15
                    Skeletal-Musc        2.0        0.9750        0.9527             38.0     250  1727           45
                     Small-Int-Ep        5.0        0.9917        0.9851             38.0     250  2038           36
                      Smooth-Musc        5.0        0.9876        0.9799             35.0     250  1539            9
                       Thyroid-Ep        3.0        0.9853        0.9751             35.0     250  2353           37
```

## 3. Markers (original published source)
- Source: nloyfer/UXM_deconv `supplemental/` (Loyfer et al. 2023, hg38 files, no liftover needed): `Atlas.U25.l4.hg38.full.tsv` (published atlas: 25 hypomethylated blocks per cell type, wgbstools find_markers with >= 4 CpGs per block, "l4") and `Markers.U250.hg38.tsv` (top-250 candidate blocks per cell type with find_markers stats tg_mean, bg_mean, delta_means, delta_quants, delta_maxmin, ttest). Exact find_markers thresholds beyond "l4" and U25/U250 are not stated in the files and were not independently verified. Also not fetched: the Nature supplementary tables (the GitHub atlas is the same published atlas; provenance = GitHub, retrieval date in MANIFEST).
- All published markers are direction U (hypomethylated in target); no hyper markers exist in these files. Note Colon-Fibro has only 4 blocks and Endothel 151 in the U250 files.
- Blocks expanded to CpGs with our index: 75,158 marker CpG rows (9,520 blocks, 40 cell types; megakaryocyte included), **1,296 fall in the benchmark universe** (1.72%, consistent with the 1.4% universe density), 164 of them from the 25-per-type published atlas. Per cell type 8-64 (median 24) universe CpGs from U250; published U25 atlas alone has 0-14 per type (median 4; Pancreas-Duct, Prostate-Ep, Smooth-Musc have 0).
- Sanity (not selection): for the 1,255 universe marker CpGs with valid data, mean beta in the target group 0.171 vs 0.890 in the other groups; 98.8% have target < others - 0.2.
- Files: `loyfer_markers_universe.parquet` (cpg_idx, chrom, pos, cell_type, direction, effect_size_delta_means, tg_mean, bg_mean, source, block, in_published_atlas_U25), `loyfer_markers_all_cpgs.parquet` (all CpGs, in_universe flag), `loyfer_markers_per_celltype.tsv`. effect size = delta_means (target vs background mean, from the U250 file; for published blocks present in it).

## 4. Pair-prep (`loyfer_profile_similarity.py`)
`profile_similarity(cpg_a, cpg_b, min_shared=20)` -> n_shared, pearson, spearman (average ranks), distance_bp, stratum; profiles use mask_primary; constant profiles -> NaN. Test (seed 17, 200,000 intra + 200,000 inter pairs): the intra set is distance-stratified (40,000 per stratum) because uniform intra pairs are 96.9% '>1Mb' (uniform: 11,219 same-chromosome pairs out of 200k draws; 1 pair <1kb). No embeddings touched.

```
         stratum  n_pairs  n_survive  median_n_shared  pearson_mean  pearson_p05  pearson_p25  pearson_median  pearson_p75  pearson_p95  spearman_median  frac_survive
            <1kb    40000      38602             39.0         0.322       -0.210        0.041           0.291        0.614        0.901            0.245         0.965
          1-10kb    40000      39040             39.0         0.127       -0.256       -0.046           0.105        0.279        0.600            0.109         0.976
        10-100kb    40000      38977             39.0         0.090       -0.272       -0.072           0.077        0.231        0.519            0.082         0.974
       100kb-1Mb    40000      38837             39.0         0.064       -0.275       -0.082           0.056        0.200        0.437            0.062         0.971
            >1Mb    40000      38922             39.0         0.054       -0.283       -0.086           0.051        0.187        0.410            0.055         0.973
interchromosomal   200000     194273             39.0         0.055       -0.281       -0.086           0.051        0.189        0.410            0.055         0.971
```

Survival: 96.5-97.6% of pairs have >=20 shared cell types and non-constant profiles (median shared 39). Pearson profile similarity decays with distance (median 0.29 / 0.11 / 0.08 / 0.06 / 0.05; inter 0.05).

## 5. Estimated experiment size and problems
- Markers: 1,296 CpGs (157 published-atlas U25 CpGs in present groups); marker-based tests are small (about 4 CpGs per type for the published atlas, ~24 per type with U250) and several types are very sparse; consider U250 as primary with the U25 subset as sensitivity.
- Pairs: up to the 200k+200k tested (about 387k usable), arbitrarily scalable; 98.6% of universe CpGs have >=20 groups.
- Problems/caveats: (1) pat only, no beta files (derived from pat; fine). (2) CpG.bed regenerated (validated on markers, but wgbstools' own file was not used). (3) group mapping is ours; 4 samples unassigned; Megakaryocytes absent locally. (4) Universe is chr1-22 only. (5) The Loyfer atlas methylation is of sorted primary cell types, donors are not independent across fine labels within a tissue; donor ids are sample ids only. (6) Hypomethylated-only markers; the atlas sampling favours blood/epithelial groups (Blood-T n=22, Endothel n=19, several groups n=1). (7) A stray background wait loop from my own shell (pgrep of itself, harmless sleep) could not be killed (kill was denied); it holds no resources and stops by itself after 30 min.
- Tool versions: python 3.10, numpy 2.2.6, pandas 2.3.3, numba 0.67.0, h5py 3.16.0, scipy 1.15.3.


## 6. Provenance revision (supersedes v1 statements above)
Code: `scripts/bioval_v2/{loyfer_provenance,loyfer_aggregate,loyfer_markers_reconcile,loyfer_freeze_pairs,loyfer_manifest}.py`.

### 6.1 Sample -> group provenance (`sample_provenance.csv/.parquet`, `manual_decisions.csv`, `sample_inventory.tsv`)
- **A published sample->group table exists**: Nature 2023 (s41586-022-05580-6) Supplementary Table S1 (`41586_2022_5580_MOESM4_ESM.xlsx`, sheet "Table S1", 205 samples; saved as `data/external/wgbs_atlas/meta/Loyfer2023_Nature_SuppTables_MOESM4.xlsx`, sha256 in MANIFEST). Columns used: `Group` (39 groups; `Refined group` is finer), `PatientID` (donor), joined on `Sample name` == GEO title (GEO writes `-Epithelial-`/`-Endothel-` where S1 writes `-Epithelium-`/`-Endothelium-`). GEO (GSE186458) carries only tissue/cell-type/age/sex characteristics (local `geo_sample_metadata.tsv`), no group. The UXM_deconv GitHub supplemental has no sample list.
- Result: 253 files = **205 `published_metadata`** (derived_group = S1 `Group`; two published names harmonised to atlas column names: Endothelium->Endothel, "Ovary+Endom-Ep"->Ovary-Ep) + **48 `manual_decision`** (46 cfDNA/WBC controls excluded; 2 Heart-Cardiomyocyte samples GSM5652214/15 on GEO but not in S1 -> excluded by conservative default). No `filename_rule` assignment remains in the final mapping (the v1 rule is kept only as column `filename_rule_group_v1`).
- Items previously "ambiguous"/"unmatched" are all resolved by the published table: Endocrine (colon x3, gastric x2, small-int x1) -> Colon-Ep / Gastric-Ep / Small-Int-Ep; podocytes x3 -> Kidney-Ep; the 3 Endometrium-Epithelial -> published "Ovary+Endom-Ep" (= atlas column Ovary-Ep); Lung-Pleura -> Lung-Ep-Alveo. The only residual assumption is the name harmonisation Ovary+Endom-Ep -> Ovary-Ep (S3 "Ovary / Endometrial Epithelium"; S4A lists markers under Ovary-Ep). Megakaryocytes: absent from S1/S3 (39 groups) and locally; the 40th column of the GitHub atlas was added after the paper.
- Donors: the published `PatientID` collapses 205 samples to 168 donors (Blood-T 22 samples/6 donors, Kidney-Ep 8/3, Gastric-Ep 11/6, Blood-Mono+Macro 11/8, Neuron 10/7, Head-Neck-Ep 13/11, Endothel 19/18, Bladder-Ep 5/4, Small-Int-Ep 5/4). v1 treated each sample as a donor; v2 aggregates at donor level.

### 6.2 Matrices rebuilt (same rules: cov>=10; mask_primary = >=min(2, n_donors) valid donors; donor beta = mean of its valid samples; group beta = mean over donors)
Before (v1) -> after (v2): samples in groups 203 -> 205; groups 39 -> 39; donors (v1: = samples) 203 -> 168; Heart-Cardio 6 -> 4 samples; Lung-Ep-Alveo 3 -> 4; Ovary-Ep 1 -> 4; mean sample-CpG valid 0.964 -> 0.968; CpGs observed in all 39 groups (primary) 0.897 -> 0.935 (lenient 0.903 -> 0.951); >=20 groups 0.9860 -> 0.9859; >=1 group 0.9988 -> 0.9988; median groups per CpG 39 -> 39. Per-group coverage: `celltype_coverage_audit.tsv` (v1: `_v1`). Per-group primary-coverage range 0.953 (Skeletal-Musc) to 0.994 (Endothel); all 39 groups >= 0.95.
Samples/donors per group (v2): Adipocytes 3/3, Bladder-Ep 5/4, Blood-B 5/5, Blood-Granul 3/3, Blood-Mono+Macro 11/8, Blood-NK 3/3, Blood-T 22/6, Bone-Osteob 1/1, Breast-Basal-Ep 4/4, Breast-Luminal-Ep 3/3, Colon-Ep 8/8, Colon-Fibro 2/2, Dermal-Fibro 1/1, Endothel 19/18, Epid-Kerat 1/1, Eryth-prog 3/3, Fallopian-Ep 3/3, Gallbladder 1/1, Gastric-Ep 11/6, Head-Neck-Ep 13/11, Heart-Cardio 4/4, Heart-Fibro 4/4, Kidney-Ep 8/3, Liver-Hep 6/6, Lung-Ep-Alveo 4/4, Lung-Ep-Bron 3/3, Neuron 10/7, Oligodend 4/4, Ovary-Ep 4/4, Pancreas-Acinar 4/4, -Alpha 3/3, -Beta 3/3, -Delta 3/3, -Duct 4/4, Prostate-Ep 4/4, Skeletal-Musc 2/2, Small-Int-Ep 5/4, Smooth-Musc 5/5, Thyroid-Ep 3/3. Groups with a single donor (primary mask = >=1): Bone-Osteob, Dermal-Fibro, Epid-Kerat, Gallbladder.

### 6.3 Marker count reconciliation (164 vs 157) and provenance (`marker_reconciliation.json`, `loyfer_markers_reconciled.parquet`, `marker_provenance_verdict.json`)
- 164 = GitHub hg38 U25 marker CpGs in the universe over **40** atlas columns; **7 of them belong to Megakaryocytes** (no local samples, not in the paper's 39 groups) -> **157 in the 39 present groups**. Not a double-count: all 164 / 157 are unique cpg_idx, no CpG is assigned to two cell types in U25. Present-group totals (GitHub hg38): U25 157 + U250-only 1,100 = 1,257 unique CpGs (1,296 with Megakaryocytes: 7 + 32).
- `loyfer_markers_reconciled.parquet` columns: cpg_idx, chrom, pos, cell_type, source_table (U25|U250; U250 = candidate not in U25 for GitHub rows), marker_source, provenance_strength, in_present_group. Per-type counts: `loyfer_markers_reconciled_per_celltype.tsv` (present groups, median 4 U25 / 24 U250 per type; zero U25 markers in universe for Pancreas-Duct, Prostate-Ep, Smooth-Musc; Colon-Fibro has 4 blocks only).
- **Verdict: WEAK for the hg38 GitHub files used.** Facts: (a) the paper's Supplementary Tables S4A/S4B (top-25: 953 single-type + 293 combination markers; top-250: 9,288 single-type + 2,425 combination rows) are **hg19 only**; (b) the GitHub hg19 files equal the paper exactly (953/953 U25, 9,288/9,288 U250 single-type blocks, identical coordinates), plus 25/250 Megakaryocyte blocks not in the paper; (c) the hg38 files are **not a liftover**: only 67% of hg38 U25 blocks (69% excluding Megakaryocytes) and 74% of hg38 U250 blocks overlap a same-type paper block lifted hg19->hg38 (UCSC chain, pyliftover); atlas values differ from hg19, i.e. markers recomputed on hg38 with an undocumented procedure; (d) per-row strength: 959 of the 1,257 present-group GitHub marker CpGs sit in blocks overlapping a lifted paper block (MODERATE), 298 do not (WEAK). Min block size 5 CpGs (>=4 satisfied), 25 per type except Colon-Fibro (4 in hg38, 3 in hg19). GitHub repo nloyfer/UXM_deconv commit `8d0bb456742ece21802bc8e651e080869c67c944` (2023-01-04, HEAD of main on 2026-10-02); URLs and sha256 of every file in the verdict JSON/MANIFEST (git blob SHAs equal the GitHub API values).
- **Strong-provenance alternative built**: paper S4A/S4B single-type blocks lifted hg19->hg38 and expanded to universe CpGs (`marker_source == paper_Suppl_Table_S4A_S4B_hg19_lifted_to_hg38`, blocks in `loyfer_markers_paper_lifted_blocks.parquet`): **164 U25 and 1,288 U250 unique universe CpGs** (all present groups; 4 of 953 U25 and 36 of 9,288 U250 blocks fail to lift; 1,000 blocks hold >=1 universe CpG). Identity of markers is STRONG (published table), the lift is ours. Pre-declared user rule (applied by the doc agent): marker set not sufficiently strong -> WSBS profile similarity PRIMARY, marker-kNN SECONDARY.

### 6.4 Frozen pair list (`frozen_pairs_seed17.parquet`, `frozen_pairs_seed17_counts.tsv`; sha256 in MANIFEST params.frozen_pairs_sha256 and `frozen_pairs_seed17_sha256.json`)
Frozen before any embedding was seen; seed 17 via `loyfer_profile_similarity.sample_pairs`; cpg_i < cpg_j; unordered duplicates dropped; only n_shared_groups >= 20 kept; columns cpg_i, cpg_j, stratum, distance_bp, n_shared_groups, loyfer_pearson, loyfer_spearman, pearson_defined, draw_order. Total **386,268** pairs (sha256 `fbb9440e...82ce7`).
```
stratum            generated  after_dedup  survive(>=20 shared)
<1kb                 40000      38696      37315
1-10kb               40000      39120      38167
10-100kb             40000      39816      38785
100kb-1Mb            40000      39987      38819
>1Mb                 40000      39999      38914
interchromosomal    200000     200000     194268
```
All surviving pairs have a defined Pearson. The `<1kb` stratum loses 3.3% to duplicate (a,b)/(b,a) draws (close CpGs). `loyfer_profile_similarity_test_*` files were regenerated on the v2 matrix (not the frozen list).
