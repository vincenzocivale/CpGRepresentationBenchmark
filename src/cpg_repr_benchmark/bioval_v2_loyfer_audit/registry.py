"""Registered constants of the Loyfer primary failure audit (fixed BEFORE any real computation; see the registration doc)."""
from __future__ import annotations

from cpg_repr_benchmark.bioval_v2_followup import registry as frg

PROTOCOL_TAG = frg.PROTOCOL_TAG
TAG_DIR = frg.BASE_OUT                                   # frozen primary tree (read-only input, fingerprinted)
AUDIT_OUT = "outputs/biological_validation_v2/loyfer_failure_audit"   # NEW sibling of the tag dir (not covered by the fingerprint)
REGISTRATION_DOC = "docs/LOYFER_PRIMARY_FAILURE_AUDIT_REGISTRATION.md"
REPORT_DOC = "docs/LOYFER_PRIMARY_FAILURE_AUDIT.md"      # written in phase 2 only
FROZEN_DOCS_GLOB = "docs/BIOLOGICAL_VALIDATION_V2*.md"
FROZEN_RESULTS_SHA256 = "38b9da12700a8fc83dc08f5a01aaa76b61f0d5097bffe395d8eb5225f1f02094"

PKG = "src/cpg_repr_benchmark/bioval_v2_loyfer_audit"
CODE_FILES = (
    f"{PKG}/__init__.py", f"{PKG}/registry.py", f"{PKG}/gate.py", f"{PKG}/common.py", f"{PKG}/target_audit.py",
    f"{PKG}/reliability.py", f"{PKG}/semantics.py", f"{PKG}/baselines.py", f"{PKG}/raw_space.py",
    f"{PKG}/svd_geometry.py", f"{PKG}/oof_pairs.py", f"{PKG}/decision.py", f"{PKG}/drivers.py",
    "scripts/bioval_v2/run_loyfer_audit.py", "tests/test_bioval_v2_loyfer_audit.py",
)

STEPS = ("target-audit", "semantics-pairs", "baselines", "raw-vs-svd", "svd-geometry", "oof-similarity", "probe-fairness",
         "report-inputs", "final-integrity")
GATED_AFTER_STEP1 = tuple(s for s in STEPS if s not in ("target-audit", "final-integrity"))
STEP_DIRS = {
    "target-audit": "step1_target_audit", "semantics-pairs": "step3_4_semantics_pairs", "baselines": "step5_baselines",
    "raw-vs-svd": "step6_raw_vs_svd", "svd-geometry": "step7_svd_geometry", "oof-similarity": "step8_oof_similarity",
    "probe-fairness": "step9_probe_fairness", "report-inputs": "report_inputs", "final-integrity": "integrity",
}
VERDICT_FILE = f"{STEP_DIRS['target-audit']}/VERDICT.json"
# Amendment 1 (2026-10-05, user decision D-L1): registered runs execute LEVEL A ONLY; the raw .pat.gz scan (Level B) is not executed.
VERDICT_PASS = "PASS_LEVEL_A"
VERDICT_PASS_WITH_B = "PASS_LEVEL_A_AND_B"       # only reachable with the unregistered --level-b flag; the gate refuses such runs
ACCEPTED_VERDICTS = (VERDICT_PASS,)
VERDICT_LEVEL = "A"
VERDICT_CAVEAT = ("PASS_LEVEL_A excludes bugs in the target construction downstream of the per-sample counts (donor/group aggregation, masks, float16, "
                  "coordinates between artifacts, pairing, Pearson) but does NOT verify the per-sample counts nor the .pat parsing against the raw files "
                  "(Level B not executed, user decision D-L1, 2026-10-05).")
SNAPSHOT_BEFORE = "integrity/SNAPSHOT_BEFORE.json"
SNAPSHOT_AFTER = "integrity/SNAPSHOT_AFTER.json"

# ---- frozen inputs (relative to data/derived/bioval_v2) -----------------------------------------------------------
BV = "data/derived/bioval_v2"
F = {
    "pairs": "external_methylation_programs/frozen_pairs_seed17.parquet",
    "beta": "external_methylation_programs/loyfer_celltype_beta.h5",
    "counts": "external_methylation_programs/loyfer_sample_counts.h5",
    "provenance": "external_methylation_programs/sample_provenance.csv",
    "covariates": "covariates/universe_covariates.parquet",
    "gc10kb": "3d_genome/hg38_gc_10kb.parquet",
    "rt_frozen": "replication_domains/FROZEN_ENDPOINT.json",
    "rt_universe": "replication_domains/cpg_rt_universe.parquet",
}
S1_XLSX = "data/external/wgbs_atlas/meta/Loyfer2023_Nature_SuppTables_MOESM4.xlsx"
PAT_DIR = "data/external/wgbs_atlas/hg38_pat"
FASTA = "data/external/reference/hg38.fa"
FASTA_FAI = "data/external/reference/hg38.fa.fai"
CATALOG = "data/external/functional/locus_features_v1/regulatory/track_contract.tsv"
TRACK_STORE = "data/derived/functional_annotations/reference_functional_features_tcga_array_all_v1_cpgidx_fixed.h5"
COMPRESSOR_NPZ = "data/cache/representations/regulatory_histone_dnase__global_svd256__discovery_chr1_19.compressor.npz"
FEATURE_SET = "regulatory_histone_dnase"
N_FROZEN_PAIRS = 386_268
N_UNIVERSE = 408_399
N_UNIVERSE_D2 = 407_550          # universe minus the 849 union-zero loci (as in B2)
STRATA = frg.ALL_STRATA
POOLED_STRATA = frg.POOLED_STRATA
CHROMS = tuple(f"chr{i}" for i in range(1, 23))
ARMS = ("regulatory_histone_dnase_v1", "functional_annotations_pca", "cpgpt_large_locus", "deepcpg_dna_locus")
CANDIDATE = "regulatory_histone_dnase_v1"
COMPARATORS = ("cpgpt_large_locus", "deepcpg_dna_locus")       # sequence comparators of rule R5
LEGACY_CONTROL = "functional_annotations_pca"
ARM_DIMS = {"regulatory_histone_dnase_v1": 256, "functional_annotations_pca": 256, "cpgpt_large_locus": 512, "deepcpg_dna_locus": 128}

# ---- Part 1: target audit -----------------------------------------------------------------------------------------
SEED = 17
SUBSET_PAIRS_PER_STRATUM = 1000          # deterministic subset: 1,000 frozen pairs per stratum (6,000), rng = default_rng(17)
SUBSET_SEED = 17
MIN_COV = 10                             # frozen rule: sample-CpG valid iff cov >= 10
MIN_SHARED = frg.MIN_SHARED              # 20
TOL_EXACT = 1e-9                         # Pearson reproduction tolerance on the float16-rounded path
TOL_BETA_ULP_FRAC = 0.5                  # |beta_f64 - beta_f16_frozen| <= 0.5 float16 ulp + 1e-6
MAX_F16_MISMATCH_FRAC = 1e-5             # cells where round_f16(beta_f64) != frozen f16 (rounding-boundary only)
TOL_UNROUNDED_MAX_FRAC_GT = 0.05         # informational: fraction of pairs with |r_unrounded - r_frozen| > 0.05 (reported, not a BUG)
FASTA_CHROMS = tuple(f"chr{i}" for i in range(1, 23)) + ("chrX", "chrY", "chrM")
N_GROUPS = 39
EXPECTED_S1_SAMPLES = 205
EXPECTED_DONORS = 168

# ---- Part 2: reliability ------------------------------------------------------------------------------------------
SPLIT_REPS = 20
SPLIT_SEED_BASE = 17000                  # rep r uses default_rng([SPLIT_SEED_BASE, r])
MIN_SHARED_FRACTION = 20 / 39            # min shared groups in a half = ceil(G_used * 20/39)
HIGH_N_MIN_DONORS = 4                    # sensitivity: groups with >= 4 donors (each half >= 2)

# ---- Part 3: semantics --------------------------------------------------------------------------------------------
V_EDGES = frg.V_EDGES                    # (0.000369276, 0.00133439, 0.00841199), registered in the follow-up (B1)
V_TOP10 = frg.V_TOP10
M_EDGES = frg.M_EDGES
BAND_TOP = 0.99                          # top 1 percent of target Pearson
BAND_BOTTOM = 0.01
BAND_MEDIAN = (0.495, 0.505)             # +/- 0.5 percentile around the median
EXAMPLES_PER_BAND = 6
SEMANTICS_SEED = 17
SMALL_AMPLITUDE_QUANTILE = 0.25          # R_small = 25th percentile of per-pair min range over the D2-kept frozen pairs (run-time value recorded)
GLOBAL_LEVEL_QUANTILE = 0.75             # G_hi = 75th percentile of per-CpG R2 w.r.t. the global group level L (run-time value recorded)
RESIDUAL_PERSIST = 0.5                   # a pair "persists" if Pearson after regressing out L >= 0.5 * original and >= 0
TOPBAND_DOMINANCE_SHARE = 0.50           # R2(c): share(near-constant U small-amplitude) in top-1% band >= 0.50 ...
TOPBAND_DOMINANCE_ENRICH = 1.5           # ... and >= 1.5 x its share among all pairs

# ---- Part 5: baselines --------------------------------------------------------------------------------------------
N_SHUFFLE = 100
SHUFFLE_SEED_BASE = 1000
N_RANDOM = 100
RANDOM_SEED_BASE = 2000

# ---- Part 7: SVD geometry -----------------------------------------------------------------------------------------
N_PCS = 20
PC_REMOVE_KS = (1, 2, 5, 10)
WHITEN_EPS_REL = 1e-6                    # whitening: (lambda + EPS_REL * lambda_max)^(-1/2)

# ---- Part 8/9: ridge -----------------------------------------------------------------------------------------------
ALPHA_GRID_ORIGINAL = (1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4)
ALPHA_GRID_EXTENDED = tuple(10.0 ** k for k in range(-2, 9))        # 1e-2 ... 1e8, 11 decades
EDGE_FRACTION = 0.10                     # an arm "selects an edge" if >= 10% of its (fold, target) alphas sit at the grid min or max
EDGE_EXTEND_DECADES = 2
EDGE_MAX_EXTENSIONS = 2
OOF_TARGETS = ("raw", "centered")        # T1 and T2 (T3 level fitted as in B3 for the R2 tables)
B3_FIT_TARGETS = ("raw", "centered", "level")

# ---- Part 11: decision rules --------------------------------------------------------------------------------------
M_REC = 0.03                             # "recovers the target": statistic >= NULL_HI + 0.03 (Spearman units)
NULL_HI_QUANTILE = 0.975                 # upper quantile of the 200 null draws (shuffle + random)
CEILING_MIN = 0.30                       # R2(a): empirical ceiling for the primary below this
REL_STRATUM_MIN = 0.50                   # R2(b): SB reliability in >1Mb or interchromosomal below this
PRECISION_DELTA = 0.02                   # R2(d): |primary(unrounded target) - primary(frozen)| for any arm above this
