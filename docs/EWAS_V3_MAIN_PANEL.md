# EWAS-v3: main-paper panel freeze

Status: panel frozen; explicit user approval pending. No representation has been run.

Official EWAS Catalog Zenodo fallback. Coverage of the complete current Catalog remains incomplete. Catalog StudyIDs are unavailable; source analysis IDs, original filenames, Zenodo records, PMIDs and source checksums are retained.

Selection is prospective and uses only the prescribed category/support/matching/N/source-ID hierarchy. Candidate audit includes every compatible inventoried source analysis with an allowed endpoint label. No embedding or representation results are read.

Primary: p < 1e-4, EWAS Catalog-associated CpGs. Sensitivity: p < 1e-7, high-confidence EWAS-associated CpGs. The same 13 source analyses are retained at both cutoffs. Support tiers use intersected positives, before matching.

## Exact main panel

| Category | Trait / selected endpoint | PMID | N | Tissue | Array | Raw positives | Universe positives | Main matched pairs | p<1e-7 universe / matched |
|---|---|---|---:|---|---|---:|---:|---:|---:|
| Cancer | Cutaneous melanoma versus nevi | 30529013.0 | 162 | Skin lesions | Illumina HumanMethylation450 | 419 | 321 | 320 | 2 / 2 |
| Cancer | Oral or pharyngeal squamous cell carcinoma | 26635906.0 | 223 | Saliva | Illumina HumanMethylation450 | 324 | 289 | 284 | 10 / 9 |
| Cancer | Prevalent Breast Cancer (Self-report) | 37410739.0 | 10650 | Whole blood | Illumina HumanMethylationEPIC | 930 | 414 | 414 | 17 / 17 |
| Cancer | Prevalent Colorectal Cancer (Self-report) | 37410739.0 | 18055 | Whole blood | Illumina HumanMethylationEPIC | 887 | 360 | 360 | 5 / 5 |
| Cancer | Incident Lung Cancer | 37410739.0 | 18364 | Whole blood | Illumina HumanMethylationEPIC | 4012 | 1976 | 1976 | 235 / 235 |
| Immune/inflammatory | Crohn's disease | 27886173.0 | 161 | Whole blood | Illumina HumanMethylation450 | 5418 | 4040 | 3994 | 473 / 468 |
| Immune/inflammatory | Rheumatoid arthritis | 23334450.0 | 689 | Whole blood | Illumina HumanMethylation450 | 815 | 676 | 670 | 20 / 20 |
| Immune/inflammatory | Psoriasis | 26743604.0 | 219 | Skin | Illumina HumanMethylation450 | 367 | 321 | 319 | 4 / 4 |
| Immune/inflammatory | Graves' disease | 26459776.0 | 135 | T cells | Illumina HumanMethylation450 | 295 | 281 | 274 | 13 / 13 |
| Cardiometabolic | Body mass index | 39706196.0 | 17304 | Whole blood | EPIC | 123691 | 48071 | 48071 | 22119 / 22119 |
| Cardiometabolic | Incident Type 2 Diabetes | 37410739.0 | 17749 | Whole blood | Illumina HumanMethylationEPIC | 30199 | 10269 | 10269 | 1979 / 1979 |
| Cardiometabolic | HDL cholesterol | 39706196.0 | 18225 | Whole blood | EPIC | 113156 | 48936 | 48936 | 14439 / 14439 |
| Cardiometabolic | Incident Ischemic Heart Disease | 37410739.0 | 17483 | Whole blood | Illumina HumanMethylationEPIC | 2627 | 936 | 936 | 26 / 26 |

Counts: Cancer **5**, immune/inflammatory **4**, cardiometabolic **4**; **13** main tasks. All 13 are main_eligible at the primary cutoff. Main matching supplies **116,823** pairs (summed task memberships, not distinct loci); **67** positives have unknown probe type and are excluded from matching. No positives with known probe type lack a technical background match.

The EWAS-v2 protocol has no pair cap, so no 2,000-pair limit is applied. Seed 17, one background per positive, exact chromosome × Infinium type, no replacement within a task. Context sensitivity additionally matches island class and GC/CpG density/log10 TSS distance with the existing 0.25-SD Euclidean caliper. Both designs are generated at both thresholds. All arms must use exactly these same frozen pairs. Background excludes the 351 inventory primary positive sets and all 16 old EWAS-v2 sets at both thresholds; exclusion remains independent of representation.

At p < 1e-7, 5 tasks retain >=200 intersected CpGs; the other 8 have <100. Sources are not substituted, and small sensitivity sets must be reported as insufficient_support rather than used as strong independent evidence.

## Redundancy before evaluation

Five primary pairs exceed 50% containment (shared / smaller positive set):

- adiposity_bmi / glycemic_t2d: 88.4% (9076/10269).
- adiposity_bmi / cardiovascular_ihd: 78.4% (734/936).
- glycemic_t2d / lipid_hdl: 69.2% (7103/10269).
- glycemic_t2d / cardiovascular_ihd: 55.7% (521/936).
- lipid_hdl / cardiovascular_ihd: 65.2% (610/936).

The strict-threshold audit flags six pairs. All matrices and pair lists are available in `data/registry/ewas_v3/`. Highly overlapping tasks remain selected but cannot be interpreted as independent replications. The future main figure must annotate these relationships. No category average or category win count is defined.

## Frozen future evaluation

Three arms only: regulatory_histone_dnase_v1 (256D), cpgpt_large_locus (native 512D), deepcpg_dna_locus (native 128D). Identical locus universe, pairs, 5 chromosome folds and 1,000 paired chromosome bootstrap draws, seed 17. Logistic L2/lbfgs probe and nested C grid inherit EWAS-v2; every scaler is fitted only to its own training fold. Pooled OOF AUROC primary; AUPRC descriptive given artificial 50% prevalence; strict matched-pair concordance secondary. All three pairwise AUROC contrasts are prespecified.

Future main output: 39 category × trait × representation rows, plus 39 task × contrast rows per primary design. Absolute AUROC figure grouped by the three biological families. No results table is populated before approval.

Supplementary registry prespecifies one representative each for glucose, total cholesterol, body-fat percentage, CKD, stroke and waist-hip ratio. Other cardiometabolic inventory traits need a new prospective supplementary freeze before evaluation. Environmental/behavioral and psychiatric are inventory only.

## Freeze and execution gate

Manifest: `configs/ewas_v3/main_trait_panel.yaml`; local freeze tag: `ewas-v3-main-panel-freeze-v1`. Raw source integrity was verified using official MD5 plus SHA256 and gzip/ZIP checks in the inventory phase. Source SHA256 values and all set/pair/input hashes are pinned. Large raw and derived files remain local caches; matched-pair CSVs and redundancy/audit tables are committed.

`python scripts/ewas_v3/check_freeze.py` checks the committed tag and all pinned inputs/artifacts, and refuses execution until an actual user approval is recorded against this exact manifest digest. Approval has not been recorded. Model execution is outside the current stop point.
