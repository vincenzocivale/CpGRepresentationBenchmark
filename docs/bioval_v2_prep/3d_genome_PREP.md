# 3d_genome axis: preparation report (bioval_v2)

## 0. Pre-declaration (written before any contact value was read)

Date written: 2026-10-02, before any mcool content was downloaded or inspected.

### Independence check (representation inputs)
- `data/external/functional/locus_features_v1/regulatory/track_contract.tsv` lists 4,165 tracks: Histone ChIP-seq 1,959; TF ChIP-seq 1,673; DNase-seq 533. Functional blocks: histone 1,959, tf_binding 1,472, accessibility 533, ctcf 201.
- No Hi-C, Micro-C, ChIA-PET, compartment, TAD, or loop track is among the inputs (grep of assay field and of manifest.json: 0 hits).
- Related-but-not-identical inputs: 201 CTCF ChIP-seq tracks (plus 14 RAD21 and 6 SMC3 ChIP-seq tracks inside tf_binding). CTCF/cohesin occupancy is mechanistically linked to loops/TAD boundaries, so loop-based labels are not strictly mechanism-independent of the inputs; compartments (A/B) are closer to histone/DNase/replication-timing signal. Both facts are reported, not corrected for.
- The `annotation_core_v1` schema (UCSC cpgIslandExt, GENCODE v50, cCRE Registry V4) also contributes; none are 3D-contact assays.

### System selection (all public, GRCh38, 4DN-processed merged mcool; open-data S3 bucket, no auth)
Primary: **H1-hESC Micro-C, 4DNFI9GMP2J8** (experiment set 4DNES21D8SP8, Tier 1, Dekker lab, merged replicates, 13.1 GB mcool, GRCh38). Reason: highest-depth merged Micro-C for a 4DN Tier 1 line (nucleosome-resolution, no restriction-enzyme site bias).
Secondary: **HFFc6 Micro-C, 4DNFI9FVHJZQ** (4DNESWST3UBH, Tier 1, merged, 11.1 GB, GRCh38).
Tertiary (compartments/E1 only if time permits): **GM12878 in situ Hi-C, 4DNFIXP4QG5B** (4DNES3JX38V5, merged, 27.4 GB, GRCh38; DpnII in situ Hi-C: restriction-enzyme bias applies).
Only the needed resolutions are fetched by HTTP range requests (not the full file).

### Resolutions
- Compartments: 100 kb primary, 50 kb secondary.
- Intra contacts: 10 kb.
- Inter contacts: 1 Mb.

### Thresholds (fixed now)
- Balanced values: ICE weights stored in the 4DN mcool (not recomputed). Bins with NaN weight = masked/invalid.
- E1: cooltools eigs_cis per whole chromosome (n_eigs=3), phasing track = GC fraction of hg38 per bin (from data/external/reference/hg38.fa.gz); E1 sign set so that Spearman(E1, GC) > 0 ("A" = positive). Per-chromosome |Spearman| < 0.2 flagged low-confidence (kept, flagged). Compartment = A if E1 > 0, B if E1 < 0.
- Intra contact (10 kb): both bins valid; distance 20 kb to 2 Mb (2 to 200 bins); raw count >= 8; obs/exp >= 2, with exp = cooltools expected_cis (balanced, per chromosome, per diagonal, mean over valid bins), no smoothing.
- Intra non-contact pool: both bins valid; both bins have a cis raw-count coverage >= the 25th percentile of valid bins of the same chromosome; both bins contain at least one benchmark CpG; pixel absent (raw=0) or obs/exp <= 0.5; same chromosome; same distance bin (matching step may use +-1 distance bin). Pool is capped per (chromosome, distance bin) at max(10 x number of contacts there with CpG on both sides, 200), random with seed 17.
- Inter contact (1 Mb): both bins valid; raw count >= 50; balanced obs/exp >= 2 versus the mean balanced value of the same chromosome pair (cooltools expected_trans-style).
- Inter non-contact pool: same chromosome pair; both bins valid; raw count == 0 or obs/exp <= 0.5; bins in same trans-coverage decile as contact bins is left to the matching step, so deciles are stored per bin; capped at 10 x number of contacts per chromosome pair.
- CpG->bin: floor((pos-1)/binsize) with 1-based GRCh38 cytosine positions; no representation-specific filtering of the universe.
- CpG pair projection: all CpGs in bin i x all in bin j, deterministic subsample with seed 17, max K per bin pair.

(Section 0 was not altered after the contact values were read. The only change after first execution: duplicate-pixel summation, see section 4. GM12878 was restricted to compartments at 50/100 kb; its 10 kb and 1 Mb were not fetched, a scope decision taken before any GM12878 value was read.)

## 1. What was done (facts)

Code: `scripts/bioval_v2/prepare_4dn.py` (subcommands fetch, compartments, intra, inter, audit, manifest; also `project_to_cpg_pairs`), `scripts/bioval_v2/fourdn_rangefile.py` (sparse local mirror filled by parallel HTTP range requests), `scripts/bioval_v2/gc_track_hg38.py` (GC per 10 kb bin from hg38.fa.gz).
Tools: cooler 0.10.4, cooltools 0.7.1. Universe loaded with `biological_validation_v2.coordinates.load_benchmark_universe()` (408,399 CpGs, no chrY/chrM CpGs). `harmonize()` was not needed (all sources are native GRCh38). `overlap_audit()` takes (chrom,pos) tables and was not applicable; the bin-based audit in section 3 is equivalent simple logic. `match_pairs`/`sample_pairs` were not run (no matching done here); `project_to_cpg_pairs` outputs the columns i, j, chrom_i, chrom_j, dist that `match_pairs` expects.

Download: single-resolution coolers were extracted from the remote mcool by HTTP range requests (S3 open-data URL, no auth). Resolutions extracted: H1 and HFFc6 10/50/100 kb and 1 Mb; GM12878 50/100 kb. Raw data in `data/external/bioval_v2/3d_genome/` (about 8-11 GB after the sparse mirrors were deleted). Full-mcool md5 (from the 4DN portal) is recorded but could not be verified, because the full file was not downloaded; sha256 of each extracted cooler is in MANIFEST.json.

Outputs in `data/derived/bioval_v2/3d_genome/` (per system prefix H1_MicroC, HFFc6_MicroC, GM12878_HiC):
- `*_compartment_bins_{100000,50000}.parquet`: chrom, bin, start, end, GC, E1, E2, E3, compartment, bin_valid.
- `*_cpg_compartment_{100000,50000}.parquet`: cpg_idx, chrom, pos, bin, E1, compartment, bin_valid for all 408,399 CpGs (one row per CpG; NaN where the bin is masked).
- `*_compartment_qc.csv`: per chromosome Spearman(E1, GC), valid bins, fraction of positive E1.
- `*_intra_contacts_10000.parquet`: chrom, bin_i, bin_j, dist_bins, dist_bp, raw, balanced, expected, obs_exp, n_cpg_i, n_cpg_j (n_cpg = number of benchmark CpGs in the bin).
- `*_intra_noncontact_pool_10000.parquet`: same keys plus raw, obs_exp, cov_i, cov_j (cis raw coverage), n_cpg_i, n_cpg_j. INTRA only.
- `*_inter_contacts_1000000.parquet` and `*_inter_noncontact_pool_1000000.parquet`: chrom_i, chrom_j, bin_i, bin_j, raw, balanced, expected, obs_exp, n_cpg_i, n_cpg_j, tcov_decile_i, tcov_decile_j. INTER only, separate files.
- `*_intra_stats.csv`, `*_inter_stats.csv`, `*_audit.json`, `hg38_gc_10kb.parquet`, `MANIFEST.json`.
- Projection to CpG pairs: `project_to_cpg_pairs(bin_pairs, universe, res, K, seed=17)`: all benchmark CpGs in bin i x all in bin j, at most K per bin pair, sampled by `default_rng([17, crc32(chrom pair), bin_i, bin_j])`; verified identical across two calls. No CpG pair table is materialized.

## 2. Results per system

### H1-hESC Micro-C (4DNFI9GMP2J8)
- Compartments (100 kb, whole chromosome, 23 chromosomes chr1-22 + X): Spearman(E1, GC) per chromosome median 0.78, min 0.67 (chr15); 0 chromosomes below the 0.2 flag. 50 kb: median 0.75, min 0.47 (chr21).
- CpGs in valid bins / with E1: 99.88% (100 kb; 496 CpGs without E1), 99.86% (50 kb; 573). A: 342,745, B: 65,158 CpGs at 100 kb.
- Intra 10 kb: 91.5% of bins valid (non-NaN ICE weight); 91,268 of the 10 kb bins contain at least one benchmark CpG (99.90% of CpGs lie in a valid bin). Candidate pixels examined: 47.5 M (distance 20 kb-2 Mb, both bins valid). Contact bin pairs: 2,348,800; with at least one benchmark CpG on both sides: 325,733; estimated CpG pairs: 12,766,143 (all) or 3,078,680 (K=20 per bin pair). Contacts with CpG both sides by distance: <=100 kb 13,853; 100-500 kb 111,378; 0.5-1 Mb 103,408; 1-2 Mb 97,094.
- Intra non-contact pool: 1,677,949 bin pairs (all have CpG on both sides by construction; max obs/exp 0.4999997); estimated CpG pairs 47,931,119 (all) / 15,245,886 (K=20). By distance: <=100 kb 13,407; 100-500 kb 290,742; 0.5-1 Mb 544,371; 1-2 Mb 829,429. The pool is shorter than the contact set at <=100 kb (13,407 vs 13,853 bin pairs), so distance matching there is tight; the pool cap per (chrom, distance bin) is max(10 x contacts, 200).
- Inter 1 Mb: 92.1% of bins valid; 99.67% of CpGs in valid bins; 2,691 bins contain CpGs. Contact bin pairs 33,167 (30,019 with CpG both sides; estimated CpG pairs 6,670,853,492 all, 600,155 at K=20). Non-contact pool: 7,074 bin pairs (estimated 78.9 M CpG pairs all, 141,022 at K=20). The pool is limited by availability, not by the cap: only 8.7% of the 253 chromosome pairs have a pool at least 10x the number of contacts with CpG both sides.

### HFFc6 Micro-C (4DNFI9FVHJZQ)
- Compartments 100 kb: Spearman(E1, GC) median 0.52, min 0.38 (chr15, chr22 0.38); 0 below the flag. 50 kb: median 0.50, min 0.33. CpGs with E1: 99.96% (100 kb; 176 without), 99.89% (50 kb; 433). A: 322,024, B: 86,199 (100 kb).
- Intra 10 kb: 91.9% bins valid; 99.92% of CpGs in valid bins. Contact bin pairs 3,046,709; with CpG both sides 436,843; estimated CpG pairs 15,620,508 (all) / 4,124,875 (K=20). Pool 2,183,192 bin pairs (estimated 65.6 M CpG pairs all / 19,959,083 at K=20).
- Inter 1 Mb: contact bin pairs 5,970 (4,776 with CpG both sides; 95,378 CpG pairs at K=20); non-contact pool only 178 bin pairs (median 0 per chromosome pair). Inter is not usable at these thresholds for HFFc6.

### GM12878 in situ Hi-C (4DNFIXP4QG5B), compartments only
- 100 kb: Spearman(E1, GC) median 0.63, min 0.34; 0 chromosomes below 0.2. CpGs with E1: 99.97% (124 without). A: 300,171, B: 108,104. 50 kb: median 0.59, min 0.31; CpGs with E1 99.92% (326 without); A 299,636, B 108,437.
- No contact tables were built for GM12878 (scope decision above). DpnII in situ Hi-C: restriction-site bias applies. Tier 3 / secondary; not a Micro-C system.

File sizes (rows, MB) for H1 and HFFc6 are in `*_audit.json` -> `files`. Largest: intra contacts 2.35 M rows 33.3 MB (H1), 3.05 M rows 41.9 MB (HFFc6); intra pool 1.68 M rows 22.3 MB (H1), 2.18 M rows 28.3 MB (HFFc6); CpG compartment tables 408,399 rows 2.7-2.8 MB each. Total derived about 140-200 MB.

## 3. Independence relative to the representation inputs
See Section 0. No contact, compartment, TAD or loop assay is among the 4,165 input tracks. CTCF (201 tracks), RAD21 (14) and SMC3 (6) ChIP-seq are inputs and relate mechanistically to loop anchors, so contact labels are not mechanism-independent of those inputs. H3K27ac/H3K9me3/H3K27me3 histone tracks and DNase correlate with compartments by biology.

## 4. Problems and caveats
- Cell-system mismatch: H1-hESC, HFFc6 and GM12878 are cell lines; the representation is patient/tissue independent, but these contact maps come from one system each and compartments change across cell types. No tissue-matched Hi-C was used.
- Duplicate pixels: the 4DN source coolers contain a few repeated (bin1,bin2) keys (10 kb: 5,806 key duplicates summed in the chr1-22,X analysis for H1 (of 642,428,012 stored pixels, 6,189 duplicate keys genome-wide), 10,134 for HFFc6). First run did not deduplicate and left pool rows with obs/exp above 0.5 (119 of 1.68 M); after summation duplicates the pool maximum is 0.4999997. All reported numbers are from the deduplicated run.
- Genome build: the coolers' `genome-assembly` attribute is "unknown"; chromosome lengths and 10 kb bin counts of all 23 chromosomes match hg38.fa.gz (chr1 = 248,956,422). Treated as GRCh38 (4DN lists genome_assembly GRCh38 for these files).
- chrY and chrM are excluded from contact maps (no benchmark CpGs on them anyway).
- Masking: bins with NaN ICE weight are masked (about 8.5% of 10 kb bins; centromeres, gaps, low-mappability); they hold 0.10% (10 kb) of benchmark CpGs. Contact calls depend on depth: HFFc6 and H1 differ in the number of calls (3.05 M vs 2.35 M) for the same thresholds, so depth is a batch effect; thresholds are not depth-normalized.
- Restriction-enzyme bias: Micro-C uses MNase (no site bias but a nucleosome-positioning/GC-related bias). The GM12878 in situ Hi-C uses DpnII (site-density/mappability bias); ICE balancing corrects it only partly. Both are reported as caveats; not corrected.
- Compartment sign relies on GC (E1 oriented by GC correlation). GC correlates with CpG density, so E1 is not independent of sequence composition; HFFc6 has weaker E1-GC coupling (median 0.52) than H1 (0.78).
- Inter-chromosomal contacts at 1 Mb are weak: few contact bin pairs and a very thin non-contact pool (7,074 for H1, 178 for HFFc6); coverage-decile matching marginals are stored (tcov_decile) but the pool is too small for stratified matching on decile.
- Intra non-contact pool is eligible only for bins with at least one benchmark CpG and coverage >= 25th percentile; contacts are not so restricted (the table keeps all contacts, with n_cpg columns). Matching must use the same eligibility on both sides.
- Spatial structure: contacts at 20 kb-2 Mb overlap strongly with distance and local chromatin state; CpG pairs from the same bin pair are not independent (K per bin pair, chromosome-blocked splitting recommended).

## 5. Freeze (written before any embedding metric; no embedding or run result was read)

Roles (user decision): H1 Micro-C 4DNFI9GMP2J8 = PRIMARY; HFFc6 Micro-C 4DNFI9FVHJZQ = SENSITIVITY only; GM12878 Hi-C 4DNFIXP4QG5B = SECONDARY (compartments only). This supersedes the primary/secondary wording in section 0. Primary metric: AUROC contact vs matched non-contact with cosine similarity as pair score (paired delta-cosine secondary). Folds: `splits.chromosome_blocked_folds(chroms with CpGs, n_folds=5, seed=17)` (22 autosomes; the 408,399-CpG universe has no chrX CpGs). All pairs are fixed from the 408,399 universe; no representation-specific intersection.

Assembly: `assembly_validation.json` (script `validate_4dn_assembly.py`). Cooler attribute is "unknown"; verdict GRCh38 (inferred+validated): 23/23 chromosome lengths equal hg38.fa.gz in all three systems and 0/23 equal hg19; 4DN portal API `genome_assembly` = GRCh38 for all three files; 408,399/408,399 universe positions are a C of a CG dinucleotide in hg38.fa; E1-GC Spearman (100 kb) median 0.78 / 0.52 / 0.63 (H1/HFFc6/GM12878).

Code: `scripts/bioval_v2/freeze_4dn_pairs.py` (cov, intra, inter, compartments, freeze). Per-bin cis coverage (`*_intra_bin_cov_10000.parquet`, same rule as the pool) is used to apply the same eligibility (valid bin, cov >= chromosome q25, CpG present) to positives; this removes 26,596 H1 contact bin pairs (325,733 -> 299,137).

Intra matching (10 kb): CpG pairs from contact bin pairs (K=20, seed 17) vs CpG pairs from the non-contact pool (K=20). Exact strata = chromosome x 80 log-spaced distance bins (1e4-2.02e6 bp) x quintile of pair-min bin coverage x tercile of pair-mean gc_content x tercile of pair-mean tss_dist (covariates from `data/derived/bioval_v2/covariates/universe_covariates.parquet`, present at freeze; no embeddings). Within a stratum a deterministic random subset of the larger side is cut to the smaller, then pairs are formed by distance rank (1:1, no replacement). Unmatched positives are dropped and counted. Columns: pair_id, cpg_i, cpg_j (i<j), chrom_i, chrom_j, distance, label, bin_pair_id, bin_i, bin_j, fold, matched_to, match_id.

| System (role) | projected positives | matched pos = controls | dropped | drop rate |
|---|---|---|---|---|
| H1 intra (primary) | 2,911,439 | 2,689,921 | 221,518 | 7.6% |
| HFFc6 intra (sensitivity) | 3,919,850 | 3,724,445 | 195,405 | 5.0% |

H1 balance (SMD before -> after): distance -0.51 -> -0.0002; log10 distance -0.60 -> -0.0003; log10 min bin coverage 0.25 -> 0.055; pair-mean GC 0.078 -> 0.016; TSS distance 0.075 -> 0.017; CpG density (hg38, not a stratum) 0.100 -> 0.100; CpG density (universe, not a stratum) 0.008 -> 0.048. Median |distance difference| within a matched pair 2.1 kb (max 113 kb). Full reports: `frozen_pairs_*.report.json`. Caveat: controls come from 946,594 pool bin pairs vs 296,796 contact bin pairs (K=20 each), so control CpG pairs are more spatially diverse; CpG pairs sharing a bin pair are not independent (use chromosome-blocked folds and bin-pair-level resampling).

Inter (H1 1 Mb, exploratory/secondary): 33,167 contact bin pairs (30,019 with CpGs both sides) -> 600,155 projected CpG pairs; pool 7,074 bin pairs -> 141,022. Strata = unordered chromosome pair x 3 groups of mean trans-coverage decile. Unmatched positives dropped BEFORE analysis: 538,022 (89.6%). Matched 62,133 pos/control pairs (8,347 bin pairs). With `pair_split_by_chrom(policy='drop')` 52,202 matched pairs straddle folds and have fold=-1 (rows kept in the file; filter fold>=0), leaving 9,931 pos + 9,931 controls (total drop 98.3% of projected positives). Residual balance of mean coverage decile SMD 2.00 -> 0.30. HFFc6 inter not frozen (pool 178 bin pairs). Treat inter as exploratory only.

Compartments (100 kb E1, A/B, chromosome fold): `frozen_compartments_{H1,HFFc6,GM12878}_100kb.parquet`; CpGs with E1 of 408,399: H1 407,903 (496 without), HFFc6 408,223 (176), GM12878 408,275 (124).

`FROZEN.json` holds sha256 of every frozen file, counts, params and roles. MANIFEST.json was regenerated afterwards. Do not re-run the freeze subcommands after embedding metrics exist.
