# ENCODE attribution: completed focused campaign (29 September 2026)

The [2 October follow-up audit](ENCODE_FOLLOWUP_RESULTS_2026-10-02.md) documents subsequent native-dimension, decodability, assay-specificity, chromosome-transfer and EWAS robustness analyses, including incomplete runs. The results below describe the frozen focused campaign and are unchanged.

The focused campaign has completed 13/13 discovery arms, 39/39 three-seed test-set
confirmation runs, 12/12 chromosome-transfer runs, co-methylation, matched EWAS and
paired reporting. There were no failed decoder jobs. The frozen arm list is in
`outputs/encode_atlas_v1/focused_plan.json`; the analysis tables and exact intervals
are in `outputs/encode_atlas_v1/FOCUSED_REPORT.md`, `paired_comparisons.csv`,
`focused_contrasts.csv`, `reconstruction_metrics.csv`, `context_gains.csv` and
`ewas_matched_membership.csv`. The [protocol](ENCODE_ATTRIBUTION.md) describes the
patient split, masks, controls and uncertainty calculation. The earlier
[discovery snapshot](ENCODE_RESULTS_2026-09-26.md) remains a dated record.

## Main result: test-patient masked reconstruction

The primary endpoint is MSE at 50% masking on the frozen test patients. All arms
use identical evaluation panels, a fixed decoder architecture and seeds 17/42/97.
Lower is better. The table shows the range across seeds, not a confidence interval.

| Arm | MSE range across seeds | Reading |
| --- | ---: | --- |
| Full functional annotations | 0.015606–0.015610 | Reference |
| Context block | 0.019933–0.019940 | CpG-island context, gene region, cCRE and TSS distance; not sequence-only |
| Context + histone ChIP-seq | 0.015951–0.015986 | Nearly recovers the full gain |
| Full minus histone ChIP-seq | 0.017043–0.017059 | Histone information is consequential |
| Full minus histones, original breadth retained | 0.016982–0.017017 | Breadth recomputation does not explain the drop |
| Frozen CpGPT large locus embedding | 0.017825–0.017840 | Tested FM comparator |
| Frozen DeepCpG DNA locus embedding | 0.019606–0.019609 | Tested FM comparator |
| Correlation-cluster representatives | 0.015584–0.015592 | Most track redundancy is unnecessary |

Relative to full, CpGPT large has **14.2–14.3% higher paired MSE** (95% bootstrap
intervals across seeds span 13.5–15.0%), and DeepCpG DNA has **25.6% higher**
(intervals span 24.4–26.8%). Removing histones increases paired MSE by **9.2–9.3%**
(intervals span 8.7–9.8%); preserving original breadth still increases it by
**8.8–9.0%** (intervals span 8.3–9.5%). Context plus histones reduces MSE versus
context alone by **19.8–20.0%** (intervals span reductions of 19.1–20.7%). By raw
MSE difference, histone addition recovers about **92%** of the context-to-full gain.
These are predictive ablations, not perturbational evidence that histone marks cause
methylation differences. The bootstrap resamples patient identities and 1 Mb genomic
blocks; optimization seeds are stability checks, not independent biological samples.
The bootstrap p-value resolution is 1/1001, so q-values near 0.001 should not be
interpreted as more precise evidence than the intervals and design support.

## Biological identity, H3K4me3 and redundancy

The histone and non-histone common-support controls contain **1,232 tracks each**, with
equal counts in every 1%-wide peak-prevalence bin. Histone subsets achieve MSE
0.016635–0.016648 versus 0.017163–0.017197 for non-histone subsets. Their direct
paired comparison favors histones by **3.0–3.3%** (95% intervals span reductions of
2.5–3.8%). This narrows a feature-count/coverage explanation, but **727 of 1,959**
histone tracks lack a non-histone prevalence-bin match and are excluded from this
particular control.

H3K4me3 was selected by the pre-test chromosome-blocked biological screen.
Context + H3K4me3 achieves MSE 0.017388–0.017406, a **12.7–12.8%** paired
improvement over context alone (95% intervals span reductions of 12.1–13.4%).
H3K4me3 alone achieves 0.017493–0.017515. It is a plausible compact component,
but does not reproduce the full histone-assay result. Correlation-cluster
representatives achieve a very small approximately 0.1% lower MSE than full; this
supports redundancy, not a claim that pruning generally improves performance.

## Transfer to excluded chromosomes

Compression, decoder training and locus-specific priors excluded chromosomes 20–22.
At held-out loci, MSE ranges are 0.123681–0.124140 for full, 0.129945–0.130178
for CpGPT large, 0.133629–0.134066 for context, and 0.124645–0.125334 for context
plus histones. Paired MSE is **4.8–5.1% higher** for CpGPT large than full, with
95% intervals spanning 4.4–5.5%; context is **7.6–8.4% higher**, with intervals
spanning 6.6–9.3%. The relative ordering persists, while absolute errors are
roughly eight times the seen-locus MSE. This is a serious boundary for the model's
genomic transfer. External FM pretraining was not proven to exclude these chromosomes.

## Secondary biological endpoints and limits

Co-methylation uses distant held-out-patient neighbors and matched controls. Mean
neighbor-minus-control correlation is 0.112 for full, 0.107 for context plus
histones, 0.068 for CpGPT large and 0.012 for context alone (roughly 2,500 query
pairs per arm). This is supportive but descriptive and uses methylation from the
same TCGA resource.

Matched EWAS/clock probing evaluated 13 usable external sets for four arms; three additional sets
had insufficient matched support. Full has higher AUC than context on 10/13 sets,
but exceeds CpGPT large on only **7/13 by AUC and 5/13 by average precision**.
For example, the colorectal set yields AUC 0.762 for full versus 0.652 for CpGPT
large, whereas several age and disease sets favor CpGPT. EWAS labels come
from external studies, but unlisted CpGs are not proven negatives; matched-set
prevalence is artificial. The data do **not** support a universal external-biological
superiority claim. The exact-organ tissue probes remain mixed (9/13 favor matched
tissue) and should not be used as the main explanation.

## Suggested paper story and figure order

The supported claim is limited to the tested protocol and frozen locus embeddings:
**reference functional annotations improve masked methylation reconstruction, with
histone-track information providing most of the measured gain beyond the existing
context/cCRE/TSS block**. Track-count/prevalence matching, preserved-breadth
ablation, three decoder seeds and chromosome transfer all support this explanation.
H3K4me3 contributes but does not account for the whole histone signal. The result is
predictive association, not a causal chromatin mechanism or a ranking of complete FM
inference systems.

Use the matched reconstruction result as the main figure, histone sufficiency and
necessity plus the common-support control as the mechanism figure, and chromosome
transfer as a robustness figure. Put H3K4me3, redundancy, co-methylation, EWAS and
mixed tissue results in secondary or supplementary panels. Retain all negative
results and support counts. A compact-panel equivalence claim is unavailable because
compact panels were deliberately excluded from focused confirmation.
