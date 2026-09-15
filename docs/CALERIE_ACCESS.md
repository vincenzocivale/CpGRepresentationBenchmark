# CALERIE intervention dataset: access status

## What CALERIE is

CALERIE (Comprehensive Assessment of Long-term Effects of Reducing Intake of Energy,
`ClinicalTrials.gov` NCT00427193) is the randomized controlled trial of sustained caloric
restriction (CR) vs. ad libitum (AL) diet in healthy, non-obese adults. It is the intended
`intervention` evaluation task for this benchmark: predicting/discriminating CR vs. AL (and,
longitudinally, response over time) from patient-agnostic CpG locus representations, holding the
same reconstruction/masking infrastructure fixed as every other task.

The genomic data resource is described in Ideraabdullah et al., *Nature Aging* 2024,
["The CALERIE Genomic Data Resource"](https://www.nature.com/articles/s43587-024-00775-0)
(PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11956694/).

**This dataset is not on GEO and cannot be fetched by URL like GSE40279/GSE42861.** It is
controlled-access. Do not confuse it with `GSE286313` — that GEO series includes 22 CALERIE
subjects only as part of an unrelated EPICv1-vs-EPICv2 cross-platform concordance study, carries
no treatment-arm or timepoint metadata, and per direct instruction is **not** to be used as the
CALERIE intervention benchmark (it may only be considered later, separately, as a technical
cross-platform QC benchmark, if ever).

## Data available, per the Genomic Data Resource paper

Illumina EPIC v1 DNA methylation, three-timepoint-longitudinal (baseline, 12-month, 24-month),
across blood, skeletal muscle, and adipose tissue:

| Tissue  | CR  | AL | Timepoints             |
|---------|-----|----|--------------------------|
| Blood   | 142 | 74 | baseline, 12mo, 24mo    |
| Muscle  | 59  | 34 | baseline, 12mo, 24mo    |
| Adipose | 60  | 31 | baseline, 12mo, 24mo    |

Also available: whole-genome SNP genotyping (Illumina GSA-24 v3.0, blood, baseline only),
small-RNA-seq (all three tissues), mRNA-seq (muscle + adipose only), and precomputed derived
variables (Horvath/Hannum/PhenoAge/GrimAge/DunedinPACE clocks, PC-based clock variants, immune
cell-type proportions, control-probe PCs).

For our purposes the relevant arm is **blood DNAm**: 216 participants (142 CR / 74 AL),
3 timepoints — this is the tissue consistent with GSE40279 and GSE42861, so it is the one that
composes cleanly with the rest of this benchmark's dataset registry.

## Access mechanism (verified 2026-09-15)

1. **Repository**: [Aging Research Biobank](https://agingresearchbiobank.nia.nih.gov/studies/calerie/)
   — a controlled-access web platform, not an open FTP/HTTP drop like GEO.
2. **Eligibility**: requester must register as a P.I./senior researcher — "a permanent employee
   of their institution(s) at a level equivalent to, at a minimum, a tenure-track professor."
3. **Required documents for the request**:
   - research protocol / brief summary of the research question and intended analysis
   - IRB approval (or waiver) for the project — required for human-subjects genomic data
   - collaborator attestation
   - CV/resume
   - proof of funding (biospecimen requests only, not needed for data-only access)
4. **Data Use Agreement**: on approval, an "Outgoing Human Data Transfer Agreement" (OHDTA) is
   signed by the requester's institutional signing official (not the requester alone). The
   agreement expires after 12 months, renewable.
5. **Turnaround**: the biobank states review "can take several weeks"; the DUA signing step adds
   further time.
6. **Cost**: not stated for data-only (non-biospecimen) access in the materials reviewed.
7. **Alternative contact for raw data**: the paper also names the Belsky Lab
   (`cpc_geroscience@cumc.columbia.edu`) as a source for original raw data beyond the processed
   summaries.
8. **Open, non-restricted companion resource**: Dryad
   (https://doi.org/10.5061/dryad.pzgmsbcxh) hosts a **molecular data summary only** — per-tissue
   probe lists (probes with complete data at each timepoint / across all timepoints) and
   probe-level summary statistics (n missing, mean, sd, median, min, max beta). This is
   aggregate/QC-level, **not** a per-sample beta matrix, and its file descriptions do not confirm
   inclusion of treatment-arm labels. It is not sufficient on its own to build the per-patient
   `methylation.h5` + `phenotypes.parquet` pair this benchmark requires; it is useful only as a
   locus-coverage sanity check once real per-sample data is in hand.

## What this means for integration

Every dataset currently in this repo (`GSE40279`, `GSE42861`) was prepared from openly
downloadable per-sample beta matrices. CALERIE blood DNAm is not open: obtaining it requires an
institutional PI to submit a request with IRB approval to the Aging Research Biobank and sign a
DUA, a process outside what can be automated or completed as part of this session.

**Nothing has been prepared or downloaded for CALERIE.** No `configs/datasets/calerie.yaml`, no
`scripts/data/prepare_calerie.py`, and no `data/processed/CALERIE/` exist yet, since writing a
parser against a file format I have not seen (exact per-sample export format — idat vs. processed
beta CSV, phenotype/covariate column names, sample ID scheme) would be guesswork and likely wrong.

## Next steps (once access is granted)

1. Submit the Aging Research Biobank request for the blood-DNAm CALERIE collection (PI-level
   account, IRB approval, research protocol) — this step needs to happen outside this repo/session.
2. Once the processed export is in hand, share its exact file layout (file names, column headers
   for the beta matrix and phenotype table, ID scheme linking `participant_id` x `timepoint` x
   `tissue` to sample rows) so `scripts/data/prepare_calerie.py` can be written against the real
   schema, following the `prepare_gse40279.py` / `prepare_gse42861.py` pattern: canonical
   `chr:pos` → `cpg_idx` crosswalk (same Illumina pickled DB, since CALERIE is also on an Illumina
   array), `/cpg_idx` + `/embedding`-style HDF5 beta store, and a `phenotypes.parquet` exposing
   `treatment_arm` (CR/AL), `timepoint` (baseline/12mo/24mo), `tissue`, and `participant_id` (for
   the patient-disjoint split — critical here since each participant contributes multiple
   longitudinal samples, so the split must be participant-level, not sample-level, to avoid
   leakage across timepoints).
3. Register the new dataset in `configs/datasets/calerie.yaml` and add it to the master CpG
   registry (`build_master_cpg_registry.py`) and a `tcga_array` transfer-locus protocol
   (`build_transfer_locus_protocols.py`), exactly as done for GSE42861 in
   `docs/GSE42861_DISEASE_PROTOCOL.md`.
4. Decide the task framing before writing the classification config: the natural binary target is
   `treatment_arm` (CR vs AL), evaluated per timepoint and/or on the 24-month sample only (the
   trial's primary endpoint), with `participant_id` used for the disjoint split so the same
   individual never appears in both train and held-out.
