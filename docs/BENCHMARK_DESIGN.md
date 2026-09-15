# Benchmark design

## Research hypothesis

Current methylation and genomic foundation models frequently characterize a CpG from sequence context, learned CpG identity, or a model-specific token embedding. This project tests a broader hypothesis: **a patient-agnostic CpG representation built from reference-genome functional annotations can be more useful than sequence-centric representations for methylome reconstruction and, later, methylation downstream tasks.**

The initial evidence came from MehylPredictor, where a functional locus representation derived from ENCODE/reference annotations was markedly stronger than an NTv3-pre sequence embedding for predicting methylation-related locus behavior. This repository turns that observation into an independent representation benchmark.

## Two benchmark tracks

### Track A — representation-controlled (primary)

Keep the methylome reconstruction architecture, data splits, masking schedule, optimizer and training budget fixed. Swap only the patient-agnostic CpG representation. This is the experiment that supports claims about **representation quality**.

Initial arms:

- functional annotation representation;
- NTv3-pre;
- later: locus representations extracted from CpGPT, MethylGPT, DNAmBERT, MethylProphet and additional genomic FMs.

### Track B — native-model masking (secondary)

Evaluate CpGPT/MethylGPT/etc. with their own native masking/inference heads on the same patient/CpG protocol. This answers which **complete method** reconstructs methylation best, but it does not isolate the locus representation. The old `MethylomeReconstruction` repo already contains useful CpGPT/MethylGPT wrapper logic that can be ported later.

## Evaluation views

### Seen-locus masking

- Train patients: train split only.
- Train targets: train CpGs only.
- Test patients: held-out patients.
- Test targets: train CpGs only.

This measures conventional masked reconstruction on loci that the network has optimized against.

### Unseen-locus masking

- Train patients: train split only.
- Train context and train targets: train CpGs only.
- Test context: train CpGs only.
- Test targets: held-out CpGs only.

A held-out CpG never appears in any training context or training target. This directly tests whether a representation allows the decoder to generalize to a new genomic locus.

## Prior leakage rule

The old reconstruction prototype predicts residuals around a per-CpG methylation prior. That is valid for seen loci, but it leaks target information in the unseen-locus setting if the held-out CpG prior was estimated from train-patient methylation.

This repo therefore computes:

- **train loci:** empirical per-CpG prior from train patients;
- **held-out loci:** a single global beta prior estimated only from train patients × train loci.

The code never reads train-patient methylation values at held-out CpGs while constructing the prior. Tests enforce this invariant.

## Mask fractions

A sampled panel of `P` CpGs is split according to mask fraction `m`:

- `target_count = round(P * m)`;
- `observed_count = P - target_count`.

A single fraction is used within a batch, avoiding padding. The default sweep is 15%, 30%, 50%, 70% and 90% masking.

## Representation dimensionality

Raw representation dimensions differ (for example 256-D functional embeddings vs much wider genomic FM embeddings). V1 uses one shared learned adapter per arm to map every representation to a common `locus_latent_dim`, after which the entire network is identical.

This is adequate for pipeline deployment, but the strict paper benchmark should add a **capacity-control arm** using a label-free fixed-dimensional adapter such as train-locus PCA/IncrementalPCA. That removes the residual difference in parameter count caused by the raw input dimension.

## Functional annotation leakage audit

Before using the functional arm for the primary biological claim, export a manifest of every annotation track and classify whether any track directly measures DNA methylation or is derived from methylation labels. The clean primary arm should exclude direct methylation-derived tracks; any version retaining them should be reported separately.

## What “unseen locus” means

The implemented `unseen_locus` view guarantees that the **downstream reconstruction model** never receives the held-out CpGs as training context or targets, and that held-out methylation labels do not enter its prior.

For a stronger claim — *the representation itself has never been supervised on those loci* — provenance must also show one of the following:

- the representation is label-free/reference-only and therefore has no methylation-label fit scope; or
- its supervised/proxy training was restricted to the persistent train-CpG protocol.

A functional embedding produced by a checkpoint trained with methylation supervision over the held-out CpGs should be reported as `downstream-unseen`, not `strict representation-OOD`. This distinction should be encoded in each representation's `provenance` fields (for example `supervision` and `locus_fit_scope`) and carried into the paper table.
