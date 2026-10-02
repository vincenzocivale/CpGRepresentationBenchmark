# ENCODE attribution: results and paper decisions (26 September 2026)

This is a dated snapshot of `outputs/encode_atlas_v1`, not a final results section.
The [campaign protocol](ENCODE_ATTRIBUTION.md) defines the analyses and their inputs.
Run-level values below come from completed `jobs/*.json` and their `summary.json`;
screening values come from `screen/*.json`. Keep this snapshot unchanged when later
results arrive, and create a new dated update.

## Question and present status

The historical representation-controlled masking benchmark motivates the question:
Functional PCA had MSE 0.015668 at 50% masking, versus 0.017551 for CpGPT large,
0.018957 for CpGPT small and 0.019565 for DeepCpG DNA. These are **historical**
results on another patient split; they cannot be compared numerically with the new
attribution runs. Functional PCA compresses a reference-based *functional annotation*
matrix, including experimental ENCODE tracks; it is not a sequence-only embedding.
The FM arms use frozen locus embeddings, not the complete native FM inference systems.

The new campaign has audited 408,399 CpGs, 4,165 peak-overlap tracks and 23 dense
features. Its 997-group biological screen, redundancy analysis and 30-cancer tissue
screen are complete. Of 97 distinct discovery decoder arms, 22 full-length runs are
complete and one is running. The only `full` result is a one-epoch, small-panel **pilot**;
the comparable 30-epoch full reference, confirmation, chromosome transfer,
co-methylation and EWAS stages are pending. No paired effect interval or necessity
claim is available yet.

`context_only` below contains the first 18 dense features: CpG context, gene region,
cCRE class and TSS distance. It does **not** mean DNA sequence alone. Assay-addition
arms use those same 18 features plus tracks of one assay; full includes all tracks and
five additional breadth features. This distinction matters for both interpretation
and the controls selected below.

## Biological screen: what is associated with methylation?

The screening endpoint is mean beta or inter-patient beta variance at each CpG,
computed only from discovery patients. Ridge probes use nested chromosome folds on
11,929 eligible loci from a 12,000-locus chr1–19 sample. The table reports R² with
the 18-feature context block, except the first row. The assay sets have unequal
track counts, so these values are evidence of information content, not per-track
importance or an assay-ranking test.

| Features | Tracks | Mean-beta R² | Variance R² |
| --- | ---: | ---: | ---: |
| Context block | 0 | 0.524 | 0.107 |
| Context + histone ChIP-seq | 1,959 | 0.849 | 0.414 |
| Context + H3K4me3 | 321 | 0.805 | 0.350 |
| Context + TF ChIP-seq | 1,673 | 0.735 | 0.247 |
| Context + DNase-seq | 533 | 0.674 | 0.256 |
| Context + assay breadth | 0 | 0.758 | 0.200 |

Histone annotations are the clearest *descriptive* signal in this screen, with
H3K4me3 a plausible compact contributor. The association is consistent with known
links between chromatin state and DNA methylation, but the screen alone cannot show
that the decoder uses these annotations or that histone marks cause methylation
differences. Because the probes use methylation-derived targets, this is also not
an independent external biological validation.

## Discovery decoder: what has been observed?

These completed runs share the new patient partition, validation view, mask seed
17001, 50% masking, 30 epochs, one evaluation panel per patient and decoder seed 17.
Each has 940,032 evaluated pairs. Lower MSE is better. Values are descriptive until
the comparable full reference and paired patient/genomic-block intervals exist.

| Arm | MSE |
| --- | ---: |
| `context_only` | 0.020203 |
| `breadth_only` | 0.019619 |
| `add/assay/DNase-seq` | 0.018053 |
| `add/assay/TF ChIP-seq` | 0.017959 |
| `add/assay/Histone ChIP-seq` | 0.016186 |
| `drop/assay/Histone ChIP-seq` | 0.017378 |
| `drop/assay/DNase-seq` | 0.016045 |
| `drop/assay/TF ChIP-seq` | 0.015898 |
| `cluster_representatives` | 0.015840 |
| `drop/dense/ccre` | 0.015838 |
| `drop/dense/breadth` | 0.015855 |

Adding histone tracks to the context block lowers observed MSE by 0.004017
(19.9% relative to `context_only`); adding DNase or TF tracks gives smaller
observed improvements. Removing histone tracks from an otherwise broad annotation
set yields a higher MSE than removing either DNase or TF tracks. The latter is **not**
yet a drop-from-full effect: it requires a completed 30-epoch `full` run and paired
comparison on identical observations. Likewise, the apparent small changes for
cCRE and breadth removal cannot be called dispensability before that comparison.
Feature count, compression rank, assay coverage and annotation correlation can
explain part of the differences.

The `full` pilot has MSE 0.020754, but used one epoch and a 128-locus panel (58,752
evaluated pairs). Do not put it in a results table beside the runs above. Three
balanced-assay discovery controls yielded MSE 0.016274, 0.016338 and 0.016245;
their similar values are reassuring about that particular sampling control, not
decoder-seed replication (all three used decoder seed 17).

## Tissue specificity and negative results

Thirty cancer groups were profiled; only 13 had exact organ-matched ENCODE tissue
tracks with assay- and prevalence-matched other-tissue controls. The organ-matched
probe had higher R² in nine of 13. Examples: THYM 0.229 versus 0.102, STAD 0.205
versus 0.186, while LIHC was 0.091 versus 0.100 and TGCT 0.081 versus 0.126.
These are chromosome-fold probes of a cancer-minus-global methylation contrast,
not reconstructed methylation MSE. The mixed directions do not support a general
claim that tissue matching explains the representation advantage. Coverage is
uneven; BRCA, for example, had no eligible exact-match control in this analysis.

## Focused experiments for the paper

Keep the complete broad screen, including negative results, as the hypothesis
generation phase. Confirmation should focus on a small prespecified set rather
than every selected pair, FM combination, panel size and subgroup. The following
priority order states which result would change the paper's claim.

| Priority | Comparison or analysis | Decision it supports |
| --- | --- | --- |
| Required | Finish `full` at 30 epochs; pair it with `context_only`, `add/assay/Histone ChIP-seq`, `drop/assay/Histone ChIP-seq` and `drop_preserve_breadth/assay/Histone ChIP-seq` on the same patients/loci. | Is histone signal sufficient, and is it necessary after accounting for breadth? |
| Required | Compare `full` with `fm/cpgpt_locus_large` and `fm/deepcpg_dna_locus` using the *new* split, decoder protocol and frozen locus vectors. | Does the original FM gap persist in a directly comparable setting? |
| Required control | Complete `balanced_assays/17`, `cluster_representatives` and histone/non-histone subsets matched within peak-prevalence bins; check effective PCA/SVD rank and peak coverage. | Is the effect explained by track number, redundancy or the 256-dimensional compression? |
| Targeted biology | Test `only/target/H3K4me3` and `add/target/H3K4me3` if H3K4me3 passes the frozen selection rule, with context and histone-assay comparators. Localize paired errors by gene region, CpG context and CpG-island/promoter strata. | Is the signal concentrated at promoter-related loci rather than being a diffuse feature-count effect? |
| Robustness | Repeat the short primary arm set over decoder seeds 17/42/97; run chromosome-held-out transfer for `full`, `context_only`, histone-addition and FM comparator only. | Are effects stable across optimization seeds and unseen loci? |
| Secondary validation | Co-methylation and matched EWAS only for `full`, `context_only`, histone-addition and the strongest FM comparator, with explicit label provenance and negative controls. | Does the reconstruction advantage correspond to an independent biological endpoint? |

At the current cost of roughly one hour per full decoder run on the shared GPU,
confirmation of all roughly 260 generated arms across three seeds and then all
chromosome-transfer repeats would take many weeks. The focused confirmation set
above comprises 13 primary arms (including two matched-subset controls)
and four chromosome-transfer arms: about 51 seed-specific decoder runs, before CPU
follow-up and GPU waiting time. The arm list is frozen from the completed biological
screen, before any confirmation/test evaluation. The default `pipeline` now implements
this focused selection and writes its exact arm list to `focused_plan.json`.
This snapshot's observed results remain those available on 26 September 2026.
The frozen screen selects H3K4me3 among the four histone targets. The matched control
has 1,232 tracks per assay set; 727 of the 1,959 histone tracks lack a non-histone
match in the same 1%-wide prevalence bin. Its result therefore addresses only the
shared-prevalence portion of the annotation catalog.
Retain all discovery and screening outputs in supplements. Do not promote a subgroup
because it looks best on confirmation patients.

Context, cCRE and TSS-related strata are already encoded in some input arms. Their
error patterns localize an effect but do not independently validate a mechanism;
matched external EWAS or other genuinely separate annotations are needed for that.

Suggested paper claim **if** the required comparisons hold: reference functional
annotations provide chromatin-state information at CpG loci that improves masked
methylation reconstruction beyond the tested frozen FM locus embeddings; histone
tracks contribute materially, with H3K4me3 as a plausible compact component.
If the equal-count control removes the advantage, revise the claim to annotation
coverage/feature richness. If matched tissue tests remain mixed, present them as
a boundary of the result rather than a mechanism. These are decision rules, not
conclusions already established.

## External biological context

ENCODE describes chromatin, histone and TF assays across cell and tissue contexts
([ENCODE Project Consortium, 2020](https://www.nature.com/articles/s41586-020-2493-4)).
The Roadmap Epigenomics analysis relates histone states, accessibility and methylation
across reference epigenomes
([Roadmap Epigenomics Consortium, 2015](https://www.nature.com/articles/nature14248)).
These references motivate the hypotheses; neither validates the dataset-specific
effects reported above.
