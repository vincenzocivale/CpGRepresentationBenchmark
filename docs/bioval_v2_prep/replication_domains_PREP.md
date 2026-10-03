# replication_domains PREP (facts only)

Code: `scripts/bioval_v2/prepare_replication_timing.py`, `scripts/bioval_v2/prepare_pmd_decato2020.py`.
Raw: `data/external/bioval_v2/replication_domains/` (RT), `data/external/bioval_v2/pmd_decato2020/` (PMD).
Derived: `data/derived/bioval_v2/replication_domains/` (RT + merged `MANIFEST.json`), `data/derived/bioval_v2/pmd_decato2020/` (PMD).
Coordinate/liftover helpers: `biological_validation_v2.coordinates` (`harmonize`, `load_benchmark_universe`); pyliftover 0.4.1, chain `data/raw/liftover/hg19ToHg38.over.chain.gz`. `provenance.write_manifest` was not used (not present when the scripts were run); manifests are written directly.

## Independence check (done before choosing sources)
The representation input catalog (`locus_features_v1`, `manifest.json`, `docs/REGULATORY_REPRESENTATION.md`) has exactly 3 assay values: Histone ChIP-seq, TF ChIP-seq, DNase (4,165 tracks, all GRCh38). No Repli-seq / replication timing / Repli-chip / lamin / LAD tracks are inputs. ENCODE has 104 released Repli-seq experiments, none in the catalog. Dense features (cCRE classes, CpG contexts, gene regions, TSS distance, breadths) contain no RT. Source-retention is therefore not triggered by Repli-seq. Residual note: the UW Repli-seq data are ENCODE-consortium data (same programme, different lab/assay, not an input).

## 1. Candidate inventory
| Source | Build | Lines | Access | Status |
|---|---|---|---|---|
| **UW Repli-seq WaveSignal (Hansen 2010 PNAS; ENCODE Mar 2012 freeze, lab UW/Stam)** | hg19 | 15 (BG02ES, BJ, GM06990, GM12801, GM12812, GM12813, GM12878, HeLa-S3, HepG2, HUVEC, IMR90, K562, MCF7, NHEK, SK-N-SH) | https://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeUwRepliSeq/ ; GEO e.g. GSM923453 | **PRIMARY (used)** |
| Koren 2012 AJHG RT | hg19 | LCLs (many) | AJHG supplement / GEO; not retrieved | not used |
| 4DN Repli-seq (data.4dnucleome.org) | GRCh38 | many | portal; needs login/accession curation, not retrieved | not used (candidate for a replication) |
| Hiratani/Ryba TimeEvolution RT atlas, Repli-Atlas (Zhao 2020) | hg19/hg38 | many | not retrieved | not used |
| ENCODE Repli-seq (encodeproject.org, GRCh38) | GRCh38 | ~104 expts | same consortium as inputs; not used | not used |
| **Decato 2020 PMDs (MethPipe `pmd`)** | hg19 (native; inferred, see below) | 157 human samples | https://github.com/bdecato/PMD_Paper_Scripts (AllPMDs.tar.bz2); license GPL-3.0 repo, paper CC-BY | **PMD source (used)** |
| Salhab 2018 (195 methylomes, MethylSeekR) | hg19 | 195 | only HepG2 track deposited (GSM3105137); PMD calls not downloadable in bulk | not used |
| Zhou 2018, Berman 2012 | hg19/hg18 | - | no bulk PMD calls located | not used |

Pre-declared primary RT source (declared before any CpG overlap was computed): UW Repli-seq WaveSignal. Justification: 15 cell lines (>=3 so consensus is allowed), single lab/pipeline, directly downloadable with md5 in files.txt, hg19 with a UCSC chain available, not an input of the representation. Data-use restriction: README says use restricted until the restriction date; files.txt gives `dateUnrestricted` 2012-07-27/2012-07-31 (passed). No explicit license text beyond the ENCODE data policy (genome.ucsc.edu/ENCODE/terms.html). Citation: Hansen RS et al. PNAS 2010;107:139-144 (UW Repli-seq method); ENCODE Project Consortium 2012 Nature 489:57.

## 2. RT preparation
- Signal: WaveSignal (wavelet-smoothed, percentage-normalised S-phase signal; range -8..91 in K562; HIGHER = EARLIER replication). It is not an early/late log ratio.
- Resolution: 1,000 bp bins (step 1,000, offset 499), 3,035,165 bins/line, grid identical across all 15 lines (verified; no mismatches). bigWig 0-based half-open.
- Liftover of the shared bin grid (both ends lifted): 3,035,165 -> 2,820,279 bins (200,661 unmapped, 13,209 dropped by interval rule: different chrom/reversed/size change >2x, 0 ambiguous). Both builds are stored (`start_hg19_0based`, `end_hg19`).
- Long bin table: `rt_bins_grch38_long.parquet`, 42,304,185 rows = 15 lines x 2,820,279 bins; columns chrom,start,end (GRCh38 1-based inclusive),rt,cell_line,source,build_in(hg19),start_hg19_0based,end_hg19.
- CpG table `cpg_rt_universe.parquet`: 408,399 rows (all universe CpGs kept), cpg_idx, chrom, pos, `rt_<LINE>` x15, `rt_consensus_z`, `n_lines_with_rt`, `bin_hg19_start0`. Rule (pre-declared): value of the bin containing the CpG; NaN if none; no extrapolation/interpolation. Consensus = mean over available lines of per-line z-scores (z using each line's genome-wide mean/sd over its hg19 bins, params in manifest); computed only where >=3 lines available (here all-or-none).
- Coverage (overlap audit, `coverage_per_chrom.csv`): 408,345 / 408,399 CpGs (99.987%) have RT in all 15 lines; 54 have none (all lines share the same grid). Per chromosome 99.95-100% (lowest chr7 24,861/24,873). Universe contains chr1-22 only (no chrX/Y).
- Lines cross-correlate strongly (e.g. GM12878 vs GM06990 r=0.97, GM12878 vs K562 0.64-0.67 at CpG level); consensus z at CpGs mean 0.76, sd 0.62 (array CpGs are skewed to early-replicating regions).

## 3. PMD table (separate)
- Source: Decato et al. 2020 Epigenetics & Chromatin 13:42 (doi 10.1186/s13072-020-00363-7); MethPipe `pmd` segmentation of public WGBS; 157 human samples (110 non-TCGA tissues/cell lines e.g. IMR90, HUES6, Roadmap tissues, Lister brain; 39 TCGA tumours; 8 TCGA matched-healthy). Source only ships samples with PMDs (>=5% genome, mean segment >=50 kb) so absence of a sample is not a "no PMD" call.
- Build: hg19, VERIFIED empirically (`pmd_decato2020/BUILD_CHECK.json`): max PMD end <= hg19 chromosome length on 22/22 autosomes and > hg38 length on 14/22; 96.3% of interval starts are the C of a CG in hg38.fa after hg19->GRCh38 liftover vs 2.2% if read as GRCh38 (random 1.6%). Explicit per-sample liftover via `harmonize(build='hg19', end_col=...)` (both ends lifted; interval rule). Totals over 157 samples: 397,664 intervals in; 4,749 unmapped, 0 ambiguous, 4,995 dropped by interval rule (2.45% total); 387,252 out. (The earlier build already lifted; the rerun reproduces identical tables.)
- STATUS: PMD is a SECONDARY/EXPLORATORY endpoint (derived from methylation); flag recorded in `MANIFEST_pmd.json`.
- Lift effect: CpGs in >=1 PMD sample 328,894 (80.5% of 408,399) lifted vs 293,847 (72.0%) had coordinates wrongly been read as GRCh38.
- Outputs: `pmd_decato2020/cpg_pmd_membership_per_sample.parquet` (cpg_idx x 157 int8: 1 in PMD, 0 not in a called PMD, -1 not assessable), `cpg_pmd_summary_universe.parquet` (pmd_frac_all_samples, pmd_frac_non_tumor_samples, n_samples_assessed, pmd_in_any), `pmd_samples.csv` (sample, category, intervals in/out, n CpG in PMD, sha256).
- Result: 328,894 / 408,399 CpGs in a PMD in >=1 sample; mean per-CpG PMD fraction by chromosome 0.096 (chr22) to 0.178 (chr18). All 408,399 assessable (PMD files contain autosomes only).
- DEPENDENCY FLAG: PMDs are derived from WGBS methylation, so they are confounded with methylation-prediction tasks and with the Loyfer-type methylation axis; they are independent of the ENCODE inputs. Some samples (IMR90, HUES6 etc.) overlap in biosample with RT lines (IMR90). Non-PMD = outside called PMDs, including unsegmented/low-coverage regions.

## 3b. Frozen RT endpoint
`data/derived/bioval_v2/replication_domains/FROZEN_ENDPOINT.json`: primary = consensus RT z (mean of per-line z over 15 UW lines; NaN if <3 lines), 408,345 CpGs after QC (54 excluded for all representations), per-line RT and covariates (GC, CpG density, TSS distance; `covariates/universe_covariates.parquet`, built afterwards: sha256 466428137dcc...e3997, see `MANIFEST_covariates.json`) as sensitivity, 5-fold chromosome-blocked folds (seed 17) stored chrom->fold.

## 4. Experiment size and confounders
- n CpGs with RT: 408,345 (all 15 lines); PMD: 408,399.
- 122,492 distinct 1 kb bins hold the 408,345 CpGs (K562 grid; ~3.3 CpG/bin) so CpG-level observations are not independent.
- Autocorrelation of the 1 kb wavelet series (autosomes): K562 rho = 0.98 at 100 kb, 0.93 at 250 kb, 0.83 at 500 kb, 0.65 at 1 Mb, 0.45 at 2 Mb (GM12878 0.99/0.94/0.85/0.66/0.44; IMR90 0.98/0.92/0.79/0.58/0.39). The e-folding length is about 2-2.5 Mb. Approximate effective independent blocks ~ autosomal length (~2.88 Gb) / (2 x ~2.5 Mb) ~ 600 (order 500-1,000; rough, not a formal estimate). Natural blocking: 22 chromosomes (leave-chromosome-out / chromosome-blocked folds); smaller blocks must be >= ~5 Mb to be roughly independent.
- Known confounders (only described; pre-declared sensitivity = partial correlation / residualisation controlling for them): GC content, CpG density/CpG-island status, gene density/TSS distance, (and for PMD: replication timing itself). RT is strongly correlated with GC and gene density (early = GC-rich, gene-dense). Array-CpG sampling is itself skewed to early RT. Cell-line identity (cancer lines vs normal) differs between sources.

## 5. Problems / caveats
- All sources are hg19; lift loses 6.6% of bins (unmapped) and drops 0.4% by the interval rule; 54 CpGs not covered.
- WaveSignal is not a log ratio; sign: higher = earlier; per-line scaling differs (z-scoring used for consensus).
- Only Rep1 used (BJ is the only line with a Rep2 on UCSC; checked in a scratch copy: BJ Rep1 vs Rep2 r=0.989 on 20,000 universe-CpG bins; no other line has Rep2). Restriction text in README is stale (dates passed).
- Decato build not stated in files; GPL-3.0 licence applies to the repo; underlying WGBS from many studies with their own terms (TCGA samples are tumour, not healthy tissue).
- Lifted bin overlaps are resolved by last-start-<=pos rule (rare).
- Not done: independent replication source (4DN GRCh38, Koren 2012) not retrieved.
