# Experiment matrix

## Phase 1 — current executable comparison

Shared scope: `TCGA array × chr1`, seed 17, same persistent CpG protocol.

| Arm | Input representation | Status | Store |
|---|---|---|---|
| Functional | functional annotations -> trained 256-D locus encoder | ready | generated cache `data/cache/representations/functional_annotations_chr1.h5` |
| NTv3-pre | sequence-derived NTv3 pre-training locus embedding | ready if preflight coverage is 100% | `data/derived/ntv3_pre_chr1_atlas/chr1_ntv3_pretrain_atlas_v1.h5` |

Evaluation views for both arms: `seen` and `unseen_locus` at mask fractions 0.15/0.30/0.50/0.70/0.90.

## Phase 2 — representation expansion

After the chr1 protocol is validated end-to-end:

1. expand NTv3-pre to a genome-wide/common benchmark universe;
2. materialize the functional representation over the same universe;
3. add CpGPT, MethylGPT, DNAmBERT, MethylProphet and additional genomic FMs;
4. define explicit common-vocabulary protocols when an encoder cannot cover the master universe.

Do not use a representation-specific intersection silently.

## Phase 3 — publication controls

- multiple random seeds;
- fixed 256-D PCA/IncrementalPCA control fit on train loci only;
- functional-track leakage audit;
- strict representation-OOD functional encoder fit if making that claim;
- genomic/function-context stratification;
- native-model masked-inference comparison as a secondary complete-method table.
