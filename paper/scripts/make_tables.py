"""Tables 1 and 2 (CSV + booktabs LaTeX)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa: F401,F403,E402

O = ROOT / "outputs"
BV = O / "biological_validation_v2/bioval-v2-protocol-freeze-v1"
# =========================================================== Table 1
b1 = Build("table1_datasets", __file__, kind="table", tags=["bioval-v2-protocol-freeze-v1", "external-recon-test-authorization-v1"], title="Datasets and evaluation protocols")
fz = b1.json(ROOT / "configs/external/gse40279_v1_freeze_manifest.json")
sp = b1.json(ROOT / "configs/external/gse40279_v1_split_manifest.json")
mp = b1.json(ROOT / "configs/external/gse40279_v1_mapping_manifest.json")
rmf = b1.json(BV / "regulatory_histone_dnase_v1/run_manifest.json")
cov = b1.json(ROOT / "data/derived/bioval_v2/external_methylation_programs/coverage_summary.json")
rt = b1.json(BV / "regulatory_histone_dnase_v1/endpoints/rt_consensus.json")
pe = rmf["exclusion_counts"]["per_endpoint"]
pv = b1.csv(O / "regulatory_confirmation_v1/analysis/paired_contrasts.csv", usecols=["n_patients"])
n_tr, n_va = int(fz["budget"]["tcga_n_train"]), int(pv.n_patients.iloc[0])
univ = int(mp["universe"]["n_benchmark"])
f = lambda x: f"{int(x):,}"  # noqa: E731
rows = [
    dict(Dataset="TCGA (MethylProphet)", Tissue="Pan-cancer tumours", N=f"{n_tr + n_va + 918:,}", Platform="Illumina 450K", **{"Train/val/test": f"{f(n_tr)}/{f(n_va)}/918"},
         **{"CpG overlap": f"{f(univ)} benchmark loci (genome-wide, no held-out loci)"}, Role="Development, discovery and Phase A reconstruction benchmark",
         **{"Test status": "Phase A: validation only; test patients not evaluated here"}, Source="train N, val N, universe: machine-readable; test N 918 and platform: doc"),
    dict(Dataset="GSE40279", Tissue="Whole blood", N=f(sum(sp["counts"].values())), Platform="Illumina 450K", **{"Train/val/test": f"{sp['counts']['train']}/{sp['counts']['validation']}/{sp['counts']['test']}"},
         **{"CpG overlap": f"{f(mp['universe']['n_mapped'])} of {f(univ)} loci mapped"}, Role="Independent external reconstruction confirmation",
         **{"Test status": "One-shot test evaluated once after authorization (external-recon-test-authorization-v1)"}, Source="machine-readable (split/mapping manifests); platform: doc"),
    dict(Dataset="Loyfer WGBS atlas", Tissue=f"{cov['n_groups']} cell-type groups", N=f"{cov['n_atlas_samples_used']} samples / {cov['n_donors_used']} donors", Platform="WGBS",
         **{"Train/val/test": "n/a (no training)"}, **{"CpG overlap": f"{pe['loyfer_profile']['n_total']:,} frozen CpG pairs ({pe['loyfer_profile']['n_kept']:,} kept); {100*cov['frac_cpg_all_groups_primary']:.1f}% of loci covered in all groups"},
         Role="Biological validation: long-range methylation program (primary D)", **{"Test status": "Frozen primary endpoint (no held-out test)"}, Source="machine-readable (coverage_summary.json, run_manifest.json); platform: doc"),
    dict(Dataset="4DN H1 Micro-C", Tissue="H1 hESC", N=f"{pe['microc_H1_intra10kb']['n_total']:,} pairs", Platform="Micro-C",
         **{"Train/val/test": "n/a (no training)"}, **{"CpG overlap": f"{pe['microc_H1_intra10kb']['n_kept']:,} pairs kept after zero-embedding exclusion"},
         Role="Biological validation: 3D contact (primary A)", **{"Test status": "Frozen primary endpoint (no held-out test)"}, Source="machine-readable (run_manifest.json); platform: doc"),
    dict(Dataset="FANTOM5", Tissue="CAGE enhancer atlas (multi-tissue)", N="n/a in result files (activity similarity: " + f"{pe['fantom5_activity_similarity']['n_total']:,} pairs)", Platform="CAGE",
         **{"Train/val/test": "blocked CV (probe)"}, **{"CpG overlap": "n/a in result files"}, Role="Biological validation: enhancer membership (primary B)",
         **{"Test status": "Frozen primary endpoint (no held-out test)"}, Source="activity pair count: machine-readable; membership N/overlap: source: doc (not in result files)"),
    dict(Dataset="Replication timing (consensus)", Tissue="Repli-seq cell lines", N=f"{rt['meta']['n_eligible']:,} eligible loci", Platform="Repli-seq",
         **{"Train/val/test": "5-fold out-of-fold probe"}, **{"CpG overlap": f"{rt['meta']['n_eligible']:,} of {rt['meta']['n_universe']:,} loci"},
         Role="Biological validation: replication timing (primary C)", **{"Test status": "Frozen primary endpoint (no held-out test)"}, Source="machine-readable (endpoint json meta); platform/tissue: doc"),
]
T1 = pd.DataFrame(rows)
cols1 = {c: c for c in T1.columns}; cols1["Source"] = "provenance of the values: machine-readable file vs 'doc' (taken from docs because no machine-readable source exists)"
write_table(b1, "table1_datasets", T1.drop(columns=["Source"]), "Datasets and evaluation protocols. Values with a documentation-only source are flagged in the manifest (source: doc).", "tab:datasets", cols1,
            colfmt="p{2.0cm}p{1.8cm}p{1.6cm}p{1.4cm}p{1.8cm}p{2.6cm}p{2.4cm}p{2.4cm}")
T1.to_csv(TAB / "table1_datasets_with_source.csv", index=False)
b1.outputs.append(rel(TAB / "table1_datasets_with_source.csv"))
b1.note("source: doc values: TCGA test N (918) and 450K platform; GSE40279 platform; Loyfer/Micro-C/FANTOM5/RT platform and tissue descriptors; FANTOM5 membership N/overlap (no machine-readable source found).")
b1.check("TCGA N train+val+test", 9178, n_tr + n_va + 918, 0, "docs 7342/918/918")
b1.finish(filters=["see script"], caption_stats=["counts only; no inference"])

# =========================================================== Table 2
b2 = Build("table2_main_results", __file__, kind="table", tags=["external-recon-protocol-freeze-v1.2", "external-recon-test-authorization-v1"], title="Main reconstruction results")
tc = O / "regulatory_confirmation_v1/analysis"; ex = O / "external_reconstruction_v1/analysis/A_v1.2/test"
tpr = b2.csv(tc / "per_run_metrics.csv"); tsm = b2.csv(tc / "across_seeds_summary.csv")
epr = b2.csv(ex / "per_run_metrics.csv")
ecp = b2.csv(ex / "primary_contrast_across_seeds.csv"); edc = b2.csv(ex / "secondary_contrast_across_seeds.csv"); elg = b2.csv(ex / "legacy_sensitivity_control_contrast_across_seeds.csv")
ARMS = MAIN_ARMS + ["functional_annotations_pca"]
def ms(v, nd): return f"{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f}"
out = []
for cohort, pr_, nd, rels in [("TCGA validation (Phase A)", tpr, 6, None), ("GSE40279 independent test", epr, 7, None)]:
    for a in ARMS:
        d = pr_[(pr_.arm == a) & (pr_.mask_fraction == 0.5)]
        assert len(d) == 3
        if cohort.startswith("TCGA"):
            r = 0.0 if a == MAIN_ARMS[0] else 100 * float(tsm[(tsm.comparator == a) & (tsm.mask_fraction == 0.5) & (tsm.metric == "mse")].mean_relative.iloc[0])
        else:
            src = {MAIN_ARMS[1]: ecp, MAIN_ARMS[2]: edc, "functional_annotations_pca": elg}.get(a)
            r = 0.0 if src is None else 100 * float(src[(src.comparator == a) & (src.mask_fraction == 0.5) & (src.metric == "mse")].mean_relative.iloc[0])
        out.append({"Cohort": cohort, "Block": "main" if a in MAIN_ARMS else "legacy sensitivity control", "Representation": ARM_LABEL[a],
                    "MSE": ms(d.mse, nd), "MAE": ms(d.mae, 5), "MAS-PCC": ms(d.mas_pcc, 5), "MAC-PCC": ms(d.mac_pcc, 4),
                    "Rel. ΔMSE vs candidate (%)": "ref." if a == MAIN_ARMS[0] else f"{r:+.2f}"})
T2 = pd.DataFrame(out)
cols2 = {"Cohort": "cohort/split (never averaged together)", "Block": "main comparators vs legacy sensitivity control (separate block; excluded from main claims)", "Representation": "arm",
         "MSE": "mean ± sample SD over seeds 17/42/97 at 50% masking", "MAE": "same", "MAS-PCC": "same (patient-level Pearson over masked loci)", "MAC-PCC": "same (locus-level Pearson)",
         "Rel. ΔMSE vs candidate (%)": "100*mean over seeds of per-seed paired (arm - candidate)/candidate MSE@0.50 (across_seeds files)"}
write_table(b2, "table2_main_results", T2, "Main reconstruction results at 50% masking (mean $\\pm$ SD over 3 seeds). TCGA: Phase A validation patients. GSE40279: one-shot independent test. Cohorts are not averaged. Relative $\\Delta$MSE = (arm $-$ candidate)/candidate; the legacy Functional PCA control is shown in a separate block.", "tab:main", cols2, colfmt="llllllll")
b2.note("Block option chosen: legacy Functional PCA is a clearly marked separate block ('legacy sensitivity control') within each cohort, so main comparators and the control are never mixed in a single block; legacy row is excluded from main-claim figures.")
b2.check("TCGA candidate MSE@0.5", 0.014784, tpr[(tpr.arm == MAIN_ARMS[0]) & (tpr.mask_fraction == 0.5)].mse.mean(), 1e-5)
b2.finish(filters=["TCGA per_run_metrics mask_fraction=0.5; across_seeds_summary mse@0.5", "A_v1.2/test per_run_metrics mask 0.5; primary/secondary/legacy contrast across_seeds mse@0.5"],
          caption_stats=["mean +- sample SD (ddof=1) over 3 seeds", "relative delta definition as in column description"])
