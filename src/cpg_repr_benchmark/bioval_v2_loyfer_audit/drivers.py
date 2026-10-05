# ruff: noqa: RUF059
"""Step drivers of the Loyfer failure audit. Heavy I/O glue; the logic lives in the kernel modules (unit-tested on synthetic data).
Every driver reads frozen files only, writes only through AuditWriter, never imports/runs a freeze command, never reads TCGA/protocol paths."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from cpg_repr_benchmark.biological_validation_v2.splits import chromosome_blocked_folds
from cpg_repr_benchmark.bioval_v2_followup import targets as tg
from cpg_repr_benchmark.bioval_v2_launch.embedding_eval import (
    EmbeddingStore,
    assert_readable,
    exclusion_mask,
    load_arm_specs,
    pair_cosine,
    union_zero_loci,
)

from . import baselines as bl
from . import decision as dc
from . import gate as gt
from . import oof_pairs as op
from . import raw_space as rs
from . import registry as rg
from . import reliability as rel
from . import semantics as sm
from . import svd_geometry as sg
from . import target_audit as ta
from .common import (
    block_of,
    make_boot,
    paired_delta,
    percentile_summary,
    read_xlsx_sheet,
    sha256_array,
    sha256_file,
    spearman_fast,
    stat_dict,
    stat_pooled,
    stat_unweighted,
)

PRIMARY_STAT_NAME = "spearman_pooled_gt1Mb_inter_eqw"

COST_NOTES = {
    "target-audit": "Level A ONLY (amendment 1; the 75 GB raw .pat.gz scan is NOT executed): ~10-20 min plus ~1 min float32 diagnostic (counts h5 205x408k, donor aggregation, 386k pairs; RAM ~6 GB); "
                    "section 2: 20 split-half repetitions x (2 x 386k pair Pearson) ~1-2 min each, RAM ~4 GB.",
    "semantics-pairs": "CSR row counts over the histone+DNase columns (reads track_indptr/track_indices only, ~1.5 GB) ~3-6 min; per-pair tables ~5 min; RAM ~6 GB.",
    "baselines": "per arm: load embeddings (<=0.8 GB), 100 shuffled + 100 Gaussian cosines on 384,675 pairs + 7 Spearman each ~25-45 min per arm (512-d slowest); "
                 "covariate baselines + bootstrap CIs (B=1000) ~10 min; RAM ~4 GB per arm.",
    "raw-vs-svd": "TrackStore CSR load (1.5-3 GB) + restriction to 2,492 columns + pair metrics ~15-25 min; bootstrap CIs ~10 min; RAM ~8 GB.",
    "svd-geometry": "per arm: universe covariance/eigh + 20 PC scores + ~14 geometry variants x CIs ~25-40 min; RAM ~4 GB.",
    "oof-similarity": "per arm: ridge_blocked_cv on 382,041 x d with 117 targets (same cost as the earlier B3, ~10-25 min) + OOF pair similarity/CIs ~10 min; "
                      "RAM ~6-8 GB; OOF float32 ~60 MB per target set per arm.",
    "probe-fairness": "extended grid: up to 3 rounds x 4 arms of the same ridge fit (~10-25 min each) -> ~2-5 h sequential; RAM ~6-8 GB per process.",
    "report-inputs": "reads only audit outputs; < 5 min (plots need matplotlib: use the cpgpt env python).",
    "final-integrity": "sha256 of ~100 files; < 1 min.",
}


def log(*a):
    print(*a, flush=True)


class Ctx:
    def __init__(self, root, out_root=None, threads=2, logf=log):
        self.root = Path(root)
        self.bv = self.root / rg.BV
        self.threads = threads
        self.log = logf
        self.out_root = Path(out_root) if out_root else gt.audit_root(self.root)
        self._w = None
        self._pairs = None
        self._universe = None
        self.specs = load_arm_specs(self.root, rg.ARMS)

    # ---- io
    @property
    def w(self) -> gt.AuditWriter:
        if self._w is None:
            self._w = gt.AuditWriter(self.root, self.out_root)
        return self._w

    def fp(self, key):
        p = self.bv / rg.F[key]
        assert_readable(p)
        return p

    def rp(self, rel):
        p = self.root / rel
        if rel == rg.TRACK_STORE:
            # The ONLY whitelisted path: the patient-independent ENCODE track store (the frozen candidate's recorded source_store). Its file
            # name contains 'tcga' (array universe provenance) and would trip the substring guard; exact-path whitelist, CSR datasets only.
            return p
        assert_readable(p)
        return p

    def pairs(self) -> pd.DataFrame:
        if self._pairs is None:
            self._pairs = pd.read_parquet(self.fp("pairs"))
            if len(self._pairs) != rg.N_FROZEN_PAIRS:
                raise AssertionError("frozen pair table size changed")
        return self._pairs

    def universe(self) -> pd.DataFrame:
        if self._universe is None:
            u = pd.read_parquet(self.fp("covariates")).sort_values("cpg_idx").reset_index(drop=True)
            if len(u) != rg.N_UNIVERSE:
                raise AssertionError("universe size changed")
            self._universe = u
        return self._universe

    def urow(self, ids):
        u = self.universe().cpg_idx.to_numpy()
        ids = np.asarray(ids, np.int64)
        pos = np.searchsorted(u, ids)
        if (pos >= len(u)).any() or (u[np.minimum(pos, len(u) - 1)] != ids).any():
            raise ValueError("cpg id outside the universe")
        return pos

    # ---- zero-row union (cached under the audit root)
    def recorded_zero_counts(self):
        m = json.loads((self.root / rg.TAG_DIR / rg.CANDIDATE / "run_manifest.json").read_text())
        return {k: int(v) for k, v in m["exclusion_counts"]["all_zero_rows_per_store"].items()}

    def zero_union(self, verify_sha=False, full=False):
        """Union of all-zero loci. Default (light): scan ONLY the stores whose frozen run manifest records > 0 all-zero rows
        (the other stores' zero count 0 is the frozen record, checked by ``full=True``, used once in the real step 1)."""
        cache = self.out_root / "common" / "zero_union.json"
        if cache.is_file():
            d = json.loads(cache.read_text())
            return np.asarray(d["union"], np.int64), d["per_store"]
        rec = self.recorded_zero_counts()
        zero, per = {}, {}
        for a, s in self.specs.items():
            if not full and rec.get(a, 1) == 0:
                per[a] = 0
                continue
            store = EmbeddingStore(s, self.root, verify_sha=verify_sha)
            zero[a] = store.zero_row_ids()
            per[a] = len(zero[a])
            if per[a] != rec.get(a, per[a]):
                raise AssertionError(f"{a}: zero rows {per[a]} differ from the frozen record {rec[a]}")
        union = union_zero_loci(zero)
        return union, per

    def zero_union_cached(self, verify_sha=False, full=False):
        union, per = self.zero_union(verify_sha, full)
        cache = self.out_root / "common" / "zero_union.json"
        if not cache.is_file():
            self.w.write_json("common/zero_union.json", {"union": union.tolist(), "per_store": per, "full_scan": full})
        return union, per

    def d2_keep(self, union=None):
        if union is None:
            union, _ = self.zero_union_cached()
        p = self.pairs()
        return exclusion_mask([p.cpg_i.to_numpy(), p.cpg_j.to_numpy()], union)

    def c2f(self):
        fz = json.loads(self.fp("rt_frozen").read_text())["folds"]["chrom_to_fold"]
        rec = chromosome_blocked_folds(list(rg.CHROMS), 5, seed=17)
        if {k: int(v) for k, v in fz.items()} != rec:
            raise AssertionError("frozen chrom_to_fold differs from chromosome_blocked_folds(22, 5, 17)")
        return rec

    def frozen_primary(self, arm):
        p = self.root / rg.TAG_DIR / arm / "endpoints" / "loyfer_profile.json"
        return json.loads(p.read_text())["stats"][PRIMARY_STAT_NAME]["value"]

    def store(self, arm, verify_sha=False):
        return EmbeddingStore(self.specs[arm], self.root, verify_sha=verify_sha)


def load_prof(ctx) -> tg.LoyferProfiles:
    return tg.LoyferProfiles(ctx.fp("beta"))


def read_counts(ctx, files_wanted):
    """Counts h5 -> (M, C) uint16 arrays for the wanted sample files in the order ``files_wanted`` (+ coordinates)."""
    with h5py.File(ctx.fp("counts"), "r") as h:
        files = [x.decode() for x in h["sample_file"][:]]
        pos = {f: i for i, f in enumerate(files)}
        ix = np.array([pos[f] for f in files_wanted])
        order = np.argsort(ix)
        M = np.empty((len(ix), h["meth_count"].shape[1]), np.uint16)
        C = np.empty_like(M)
        M[order] = h["meth_count"][ix[order]]
        C[order] = h["cov"][ix[order]]
        meta = {"cpg_idx": h["cpg_idx"][:], "chrom": h["chrom"][:].astype(str), "pos": h["pos"][:],
                "clipped_cells": int(h.attrs.get("clipped_cells_cov_gt65535", -1)), "files_in_h5": len(files)}
    return M, C, meta


# ================================================================================================= dry-run counts
def dry_counts(ctx: Ctx, with_raw: bool = True) -> dict:
    """Read-only counts only (no audit statistic): pairs per stratum, complete-case CpGs, N=1 groups, source availability."""
    out = {}
    p = ctx.pairs()
    union, per = ctx.zero_union()
    keep = ctx.d2_keep(union)
    out["zero_loci_per_store"] = per
    out["zero_union"] = len(union)
    out["pairs_total"] = len(p)
    out["pairs_per_stratum"] = {s: int((p.stratum == s).sum()) for s in rg.STRATA}
    out["pairs_per_stratum_after_D2"] = {s: int(((p.stratum == s).to_numpy() & keep).sum()) for s in rg.STRATA}
    out["pairs_after_D2"] = int(keep.sum())
    prof = load_prof(ctx)
    P = prof.matrix("primary")
    full = ~np.isnan(P).any(1)
    out["cpgs_universe"] = len(prof.cpg)
    out["cpgs_full_39_group_profiles_primary_mask"] = int(full.sum())
    out["cpgs_ge20_groups_primary"] = int((np.sum(~np.isnan(P), 1) >= rg.MIN_SHARED).sum())
    ri, rj = prof.rows(p.cpg_i.to_numpy()), prof.rows(p.cpg_j.to_numpy())
    both = full[ri] & full[rj]
    out["pairs_both_cpgs_full_profile"] = int(both.sum())
    out["pairs_both_full_and_D2"] = int((both & keep).sum())
    out["pairs_both_full_and_D2_per_stratum"] = {s: int((both & keep & (p.stratum == s).to_numpy()).sum()) for s in rg.STRATA}
    c2f = ctx.c2f()
    fold_row = np.array([c2f[c] for c in prof.chrom[full]])
    out["probe_rows_per_fold"] = {int(f): int((fold_row == f).sum()) for f in range(5)}
    fi = pd.Series(prof.chrom[ri]).map(c2f).to_numpy()
    fj = pd.Series(prof.chrom[rj]).map(c2f).to_numpy()
    out["pairs_same_fold_among_full_D2"] = int((both & keep & (fi == fj)).sum())
    out["groups_with_N_samples_1"] = [g for g, n in zip(prof.groups, prof.n_samples, strict=True) if n == 1]
    prov = pd.read_csv(ctx.fp("provenance"))
    tab = ta.group_donor_table(prov)
    ng = ta.n_per_group(tab)
    out["samples_used"], out["donors_used"] = len(tab), int(tab.donor.nunique())
    out["groups_with_1_donor"] = ng.index[ng.n_donors == 1].tolist()
    out["groups_with_ge2_donors"] = int((ng.n_donors >= 2).sum())
    out["groups_with_ge4_donors"] = int((ng.n_donors >= rg.HIGH_N_MIN_DONORS).sum())
    out["samples_excluded_from_groups"] = int(prov.derived_group.isna().sum())
    pat_dir = ctx.root / rg.PAT_DIR
    present = [(pat_dir / f).is_file() for f in tab.file]
    out["source_files"] = {
        "pat_gz_used_samples_present": f"{int(sum(present))}/{len(present)}",
        "pat_gz_total_in_dir": len(list(pat_dir.glob("*.hg38.pat.gz"))),
        "pat_gz_used_total_bytes": int(sum((pat_dir / f).stat().st_size for f in tab.file if (pat_dir / f).is_file())),
        "fasta_hg38": (ctx.root / rg.FASTA).is_file(), "fasta_fai": (ctx.root / rg.FASTA_FAI).is_file(),
        "S1_xlsx": (ctx.root / rg.S1_XLSX).is_file(), "sample_counts_h5": ctx.fp("counts").is_file(),
        "celltype_beta_h5": ctx.fp("beta").is_file(), "sample_provenance_csv": ctx.fp("provenance").is_file(),
        "covariates_parquet": ctx.fp("covariates").is_file(), "track_store_h5": (ctx.root / rg.TRACK_STORE).is_file(),
        "compressor_npz": (ctx.root / rg.COMPRESSOR_NPZ).is_file(), "catalog_tsv": (ctx.root / rg.CATALOG).is_file(),
    }
    cov = pd.read_parquet(ctx.fp("covariates"))
    out["covariate_columns"] = [c for c in cov.columns if c not in ("cpg_idx", "chrom", "pos")]
    out["covariate_nan_counts"] = {c: int(cov[c].isna().sum()) for c in out["covariate_columns"]}
    sub = ta.subset_indices(p)
    out["section1_subset"] = {"n_pairs": len(sub),
                              "n_unique_cpgs": len(np.unique(np.r_[p.cpg_i.to_numpy()[sub], p.cpg_j.to_numpy()[sub]])),
                              "per_stratum": {s: int((p.stratum.to_numpy()[sub] == s).sum()) for s in rg.STRATA}}
    out["existing_oof_predictions_saved"] = _existing_oof(ctx)
    if with_raw:
        out["raw_feature_space"] = raw_counts_readonly(ctx)
    return out


def _existing_oof(ctx):
    base = ctx.root / rg.TAG_DIR / "exploratory_posthoc_loyfer"
    found = [str(f.relative_to(ctx.root)) for f in base.rglob("*") if f.is_file() and ("oof" in f.name.lower() or "pred" in f.name.lower())]
    return {"files_matching_oof_or_pred": found, "reusable": bool(found),
            "note": "B3_profile_ridge.json/.replicates.npz hold metrics and per-fold median alphas only; OOF predictions were NOT saved "
                    "-> section 8 recomputes them with the identical protocol (and reproduces the stored B3 metrics as a check)"}


def feature_columns(ctx):
    from cpg_repr_benchmark.encode_atlas.feature_sets import annotate, read_catalog, resolve_feature_set
    fs = resolve_feature_set(rg.FEATURE_SET, ctx.rp(rg.CATALOG), check_expected=True)
    frame = read_catalog(ctx.rp(rg.CATALOG))
    ann = annotate(frame, fs.spec)
    assay = ann.set_index("feature_column").encode_assay
    cols = np.asarray(fs.feature_columns, np.int64)
    return fs, cols, assay.loc[cols].to_numpy()


def raw_counts_readonly(ctx):
    """Count-only: resolve the feature set and count loci with an all-zero raw vector (reads track_indptr/track_indices only)."""
    fs, cols, assay = feature_columns(ctx)
    tot, _hist, _dn, ids = row_peak_counts(ctx, cols, assay)
    return {"n_selected_tracks": len(cols), "per_assay": fs.counts_per_assay, "n_loci": len(ids),
            "n_loci_all_zero_raw_vector": int((tot == 0).sum()), "datasets_read": ["cpg_idx", "track_indptr", "track_indices"]}


def row_peak_counts(ctx, cols, assay, chunk_rows=200_000):
    """Per-locus number of overlapped selected tracks (total / histone / DNase), store order. CSR only; /dense is never opened."""
    with h5py.File(ctx.rp(rg.TRACK_STORE), "r") as h:
        ids = h["cpg_idx"][:].astype(np.int64)
        indptr = h["track_indptr"][:].astype(np.int64)
        n_tracks = int(h.attrs["n_tracks"])
        sel_h = np.zeros(n_tracks, bool)
        sel_d = np.zeros(n_tracks, bool)
        sel_h[cols[assay == "Histone ChIP-seq"]] = True
        sel_d[cols[assay == "DNase-seq"]] = True
        dsi = h["track_indices"]
        th, td = np.zeros(len(ids), np.int32), np.zeros(len(ids), np.int32)
        for a in range(0, len(ids), chunk_rows):
            b = min(a + chunk_rows, len(ids))
            lo, hi = indptr[a], indptr[b]
            ind = dsi[lo:hi]
            for sel, outv in ((sel_h, th), (sel_d, td)):
                cs = np.concatenate([[0], np.cumsum(sel[ind], dtype=np.int64)])
                outv[a:b] = cs[indptr[a + 1:b + 1] - lo] - cs[indptr[a:b] - lo]
    return th + td, th, td, ids


# ================================================================================================= STEP 1 (+2): target audit
def _scan_one(args):
    path, targets, G = args
    meth, cov, nlines = ta.pat_counts(path, targets, G)
    return meth, cov, nlines


def level_a(ctx, log=log):
    """Counts -> donors -> group betas -> pair Pearson; compared with the frozen artifacts. Returns (checks, results, state)."""
    from cpg_repr_benchmark.bioval_v2_followup import strata as st
    checks, res = {}, {}
    prov = pd.read_csv(ctx.fp("provenance"))
    s1 = read_xlsx_sheet_s1(ctx)
    mp = ta.verify_mapping(prov, s1)
    res["mapping_vs_published_S1"] = {k: v for k, v in mp.items()}
    checks["mapping_matches_published_S1"] = bool(mp["ok"])
    tab = ta.group_donor_table(prov)
    ng = ta.n_per_group(tab)
    res["per_group_counts"] = ng.reset_index().to_dict("records")
    res["groups_with_N_samples_1"] = ng.index[ng.n_samples == 1].tolist()
    res["groups_with_1_donor"] = ng.index[ng.n_donors == 1].tolist()
    checks["n_samples_used_eq_205"] = len(tab) == rg.EXPECTED_S1_SAMPLES
    checks["n_donors_summed_over_groups_eq_168"] = int(ng.n_donors.sum()) == rg.EXPECTED_DONORS
    checks["n_groups_eq_39"] = len(ng) == rg.N_GROUPS
    res["excluded_samples"] = prov[prov.derived_group.isna()][["GSM", "file_name", "decision_type", "notes"]].to_dict("records")
    prof = load_prof(ctx)
    checks["groups_identical_to_frozen"] = list(prof.groups) == list(ng.index)
    checks["n_samples_per_group_identical"] = bool((prof.n_samples == ng.n_samples.loc[prof.groups].to_numpy()).all())
    with h5py.File(ctx.fp("beta"), "r") as h:
        nd_frozen = h["n_donors_per_group"][:]
        n_valid_frozen = h["n_valid_donors"][:]
    checks["n_donors_per_group_identical"] = bool((nd_frozen == ng.n_donors.loc[prof.groups].to_numpy()).all())
    log("  loading counts")
    M, C, meta = read_counts(ctx, list(tab.file))
    checks["counts_not_clipped"] = meta["clipped_cells"] == 0
    checks["counts_row_order_identical_to_frozen_beta"] = bool((meta["cpg_idx"] == prof.cpg).all())
    uni = ctx.universe().set_index("cpg_idx")
    chrom_u, pos_u = uni.chrom.loc[meta["cpg_idx"]].to_numpy().astype(str), uni.pos.loc[meta["cpg_idx"]].to_numpy()
    checks["coordinates_counts_h5_equal_universe_covariates"] = bool((chrom_u == meta["chrom"]).all() and (pos_u == meta["pos"]).all())
    checks["coordinates_beta_h5_equal_universe_covariates"] = bool((prof.chrom == chrom_u).all())
    log("  donor aggregation")
    D, V, donors = ta.donor_betas(M, C, list(tab.file), tab)
    beta64, nval, groups = ta.group_betas(D, V, donors)
    nd = ng.n_donors.loc[groups].to_numpy()
    mask_p = ta.primary_mask(nval, nd)
    checks["mask_primary_identical"] = bool((mask_p == prof.mask["primary"]).all())
    checks["mask_lenient_identical"] = bool(((nval >= 1) == prof.mask["lenient"]).all())
    checks["n_valid_donors_identical"] = bool((nval == n_valid_frozen).all())
    cb = ta.compare_beta(beta64, prof.beta)
    res["group_beta_vs_frozen_float16"] = cb
    checks["group_beta_nan_pattern_identical"] = cb["nan_pattern_mismatch"] == 0
    checks["group_beta_within_half_ulp"] = cb["n_beyond_half_ulp"] == 0
    # Amendment 2: the old fraction rule (1e-5, SUPERSEDED) is informational; the hard rule is mechanism-based
    checks["group_beta_f16_mismatch_within_float32_precision_of_midpoint"] = cb["n_mismatch_beyond_f32_tol"] == 0
    checks["info_f16_mismatch_fraction_vs_superseded_1e-5"] = None
    res["f16_mismatch_fraction_superseded_threshold"] = {"observed": cb["frac_f16_rounding_mismatch"], "superseded_threshold": rg.MAX_F16_MISMATCH_FRAC,
                                                         "would_have_passed": cb["frac_f16_rounding_mismatch"] <= rg.MAX_F16_MISMATCH_FRAC}
    try:
        b32 = ta.frozen_style_group_beta_float32(M, C, list(tab.file), tab)
        res["f16_mismatch_float32_accumulation_diagnostic"] = ta.diagnose_f16_mismatch(beta64, prof.beta, b32)
        del b32
    except Exception as e:  # noqa: BLE001
        res["f16_mismatch_float32_accumulation_diagnostic"] = {"error": str(e)}
    pairs = ctx.pairs()
    sc = ta.structural_checks(pairs, ctx.universe(), prof.beta, prof.mask["primary"], prof.groups)
    res["structural"] = sc
    for k in ("pairs_cpg_i_lt_cpg_j", "pairs_endpoints_in_universe", "universe_cpg_unique", "universe_chrom_pos_unique", "beta_in_0_1"):
        checks[f"structural_{k}"] = bool(sc[k])
    checks["structural_pairs_no_duplicates"] = sc["pairs_duplicates"] == 0
    ia, ib = prof.rows(pairs.cpg_i.to_numpy()), prof.rows(pairs.cpg_j.to_numpy())
    # float16-rounded path: must equal the frozen loyfer_pearson
    b16 = beta64.astype(np.float16).astype(np.float64)
    bad_cell = np.where(~np.isnan(beta64), b16 != prof.beta, False).any(1)   # CpGs with a rounding-boundary mismatch
    P16 = np.where(mask_p, b16, np.nan)
    ns16, r16 = tg.pair_pearson(P16, ia, ib)
    fz = pairs.loyfer_pearson.to_numpy()
    clean = ~(bad_cell[ia] | bad_cell[ib])
    checks["n_shared_groups_identical"] = bool((ns16 == pairs.n_shared_groups.to_numpy()).all())
    checks["pearson_defined_pattern_identical"] = bool((np.isnan(r16) == np.isnan(fz)).all())
    d16 = np.abs(r16 - fz)
    res["pearson_float16_path_vs_frozen"] = {"n_pairs": len(fz), "n_pairs_touching_f16_boundary_cells": int((~clean).sum()),
                                             "max_abs_diff_clean_pairs": float(np.nanmax(d16[clean])),
                                             "max_abs_diff_all_pairs": float(np.nanmax(d16))}
    checks["pearson_reproduces_frozen_float16_path"] = res["pearson_float16_path_vs_frozen"]["max_abs_diff_clean_pairs"] <= rg.TOL_EXACT
    # unrounded path: informational (precision of the frozen target)
    P64 = np.where(mask_p, beta64, np.nan)
    ns64, r64 = tg.pair_pearson(P64, ia, ib)
    vmin, mmean = st.pair_strata_values(pairs, prof)
    cells = st.cells(pairs.stratum.to_numpy(), vmin, mmean)
    d64 = np.abs(r64 - fz)
    res["pearson_unrounded_vs_frozen"] = {
        "max_abs_diff": float(np.nanmax(d64)), "median_abs_diff": float(np.nanmedian(d64)),
        "frac_gt_0.01": float(np.nanmean(d64 > 0.01)), "frac_gt_0.05": float(np.nanmean(d64 > rg.TOL_UNROUNDED_MAX_FRAC_GT)),
        "spearman_unrounded_vs_frozen": spearman_fast(r64, fz), "n_defined_unrounded_not_frozen": int((np.isfinite(r64) & ~np.isfinite(fz)).sum()),
        "n_defined_frozen_not_unrounded": int((~np.isfinite(r64) & np.isfinite(fz)).sum()),
        "per_cell": {k: {"n": int(m.sum()), "median_abs_diff": float(np.nanmedian(d64[m])), "q99_abs_diff": float(np.nanquantile(d64[m], 0.99)),
                         "max_abs_diff": float(np.nanmax(d64[m]))} for k, m in cells.items()
                     if k.startswith(("V_", "dist_")) and m.any()}}
    ctx.w.save_npy(f"{rg.STEP_DIRS['target-audit']}/pair_pearson_unrounded.npy", r64)
    # missingness tables
    miss = pd.DataFrame({"group": groups, "n_samples": ng.n_samples.loc[groups].to_numpy(), "n_donors": nd,
                         "frac_cpgs_observed_primary": mask_p.mean(0), "frac_cpgs_observed_lenient": (nval >= 1).mean(0)})
    ctx.w.write_csv(f"{rg.STEP_DIRS['target-audit']}/missingness_per_group.csv", miss)
    ngr = mask_p.sum(1)
    res["missingness_per_cpg"] = {"hist_n_groups_observed": {int(k): int(v) for k, v in zip(*np.unique(ngr, return_counts=True), strict=True)}}
    # beta range / sample QC (informational)
    res["sample_mean_beta_vs_S1_CpG_methylation"] = _s1_qc(M, C, tab, s1)
    state = {"D": D, "V": V, "donors": donors, "prof": prof, "pairs": pairs, "ia": ia, "ib": ib, "P64": P64, "P16": P16,
             "M": M, "C": C, "meta": meta, "tab": tab, "mask_p": mask_p, "beta64": beta64, "nval": nval, "fz": fz}
    return checks, res, state


def read_xlsx_sheet_s1(ctx):
    return read_xlsx_sheet(ctx.rp(rg.S1_XLSX), "Table S1", "Sample name")


def _s1_qc(M, C, tab, s1):
    try:
        ok = C >= rg.MIN_COV
        mean_beta = np.array([np.nanmean(np.where(ok[i], M[i] / np.maximum(C[i], 1), np.nan)) for i in range(len(tab))])
        s1z = s1.copy()
        s1z["z"] = s1z["Sample name"].map(ta.zsuffix)
        s1z = s1z.set_index("z")
        tz = tab.file.map(ta.zsuffix)
        v = pd.to_numeric(s1z.loc[tz, "CpG methylation"], errors="coerce").to_numpy()
        return {"spearman": spearman_fast(mean_beta, v), "n": int(np.isfinite(v).sum()),
                "note": "informational: array-universe mean beta vs published genome-wide CpG methylation of the sample"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def level_b(ctx, st_a, threads, log=log):
    """Raw .pat.gz (+ fasta) -> counts for the deterministic subset; compared with the counts h5 and the frozen target."""
    from multiprocessing import Pool
    checks, res = {}, {}
    pairs, prof, tab, M, C, meta = st_a["pairs"], st_a["prof"], st_a["tab"], st_a["M"], st_a["C"], st_a["meta"]
    log("  fasta CpG index (independent of loyfer_cpg_index.py)")
    gidx, ncg = ta.genome_cpg_index(ctx.rp(rg.FASTA), ctx.rp(rg.FASTA_FAI))
    res["fasta_cpg_total_chr1_22_X_Y_M"] = int(ncg)
    uni = pd.DataFrame({"chrom": meta["chrom"], "pos": meta["pos"]})
    glob = ta.universe_global_index(uni, gidx)
    checks["all_universe_positions_are_C_of_CG_in_hg38_fasta"] = bool((glob > 0).all())
    res["n_universe_positions_not_CG"] = int((glob <= 0).sum())
    z = np.load(ctx.fp("provenance").parent / "hg38_CpG_index.npz")
    chroms = list(z["chroms"])
    gk = z["chrom_id"].astype(np.int64) * 10**9 + z["pos"].astype(np.int64)
    uk = pd.Series(meta["chrom"]).map({c: i for i, c in enumerate(chroms)}).to_numpy(np.int64) * 10**9 + meta["pos"].astype(np.int64)
    j = np.clip(np.searchsorted(gk, uk), 0, len(gk) - 1)
    checks["global_cpg_index_equals_frozen_hg38_CpG_index_npz"] = bool(((gk[j] == uk) & (j + 1 == glob)).all())
    sub = ta.subset_indices(pairs)
    sub_ids = np.unique(np.r_[pairs.cpg_i.to_numpy()[sub], pairs.cpg_j.to_numpy()[sub]])
    rows = prof.rows(sub_ids)                      # same row space as the counts h5 (asserted identical order)
    targets = np.unique(glob[rows])
    files = list(tab.file)
    cache = ctx.out_root / rg.STEP_DIRS["target-audit"] / "pat_subset_counts.npz"
    key = sha256_array(targets) + "|" + "|".join(files)
    meth = cov = None
    if cache.is_file():
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            meth, cov = z["meth"], z["cov"]
            log("  reusing cached raw .pat subset counts")
    if meth is None:
        log(f"  scanning {len(files)} .pat.gz files with {threads} processes (targets {len(targets)})")
        args = [(str(ctx.rp(rg.PAT_DIR) / f), targets, int(ncg)) for f in files]
        with Pool(threads) as pool:
            out = []
            for i, r in enumerate(pool.imap(_scan_one, args)):
                out.append(r)
                if (i + 1) % 10 == 0:
                    log(f"    {i + 1}/{len(files)} files")
        meth = np.stack([o[0] for o in out])
        cov = np.stack([o[1] for o in out])
        ctx.w.path(rg.STEP_DIRS["target-audit"], "pat_subset_counts.npz")
        np.savez_compressed(cache, meth=meth, cov=cov, targets=targets, key=np.array(key))
    pos = np.searchsorted(targets, glob[rows])
    Mp, Cp = meth[:, pos], cov[:, pos]
    eqm, eqc = bool((Mp == M[:, rows]).all()), bool((Cp == C[:, rows]).all())
    checks["raw_pat_methylated_counts_equal_counts_h5"] = eqm
    checks["raw_pat_coverage_equal_counts_h5"] = eqc
    res["raw_pat_vs_counts_h5"] = {"n_files": len(files), "n_cpgs": len(rows), "n_cells": int(Mp.size),
                                   "n_meth_cells_different": int((Mp != M[:, rows]).sum()), "n_cov_cells_different": int((Cp != C[:, rows]).sum())}
    # target recomputed from the raw-.pat counts only, per-pair loop (independent of pair_pearson)
    D, V, donors = ta.donor_betas(Mp, Cp, files, tab)
    beta, nval, groups = ta.group_betas(D, V, donors)
    nd = tab.groupby("group").donor.nunique().loc[groups].to_numpy()
    mask = ta.primary_mask(nval, nd)
    P = np.where(mask, beta.astype(np.float16).astype(np.float64), np.nan)
    loc = {c: i for i, c in enumerate(sub_ids)}
    ia = np.array([loc[c] for c in pairs.cpg_i.to_numpy()[sub]])
    ib = np.array([loc[c] for c in pairs.cpg_j.to_numpy()[sub]])
    ns, r = ta.pearson_loop(P, ia, ib)
    fz = pairs.loyfer_pearson.to_numpy()[sub]
    frozen_beta16 = prof.beta[rows]
    b_mism = np.where(~np.isnan(beta), beta.astype(np.float16).astype(np.float64) != frozen_beta16, False).any(1)
    cl = ~(b_mism[ia] | b_mism[ib])
    d = np.abs(r - fz)
    checks["raw_pat_subset_n_shared_identical"] = bool((ns == pairs.n_shared_groups.to_numpy()[sub]).all())
    checks["raw_pat_subset_pearson_defined_pattern_identical"] = bool((np.isnan(r) == np.isnan(fz)).all())
    res["raw_pat_subset_pearson_vs_frozen"] = {"n_pairs": len(sub), "n_pairs_touching_f16_boundary_cells": int((~cl).sum()),
                                               "max_abs_diff_clean_pairs": float(np.nanmax(d[cl])), "max_abs_diff_all_pairs": float(np.nanmax(d)),
                                               "subset_seed": rg.SUBSET_SEED, "per_stratum": {s: int((pairs.stratum.to_numpy()[sub] == s).sum()) for s in rg.STRATA}}
    checks["raw_pat_subset_pearson_reproduces_frozen"] = res["raw_pat_subset_pearson_vs_frozen"]["max_abs_diff_clean_pairs"] <= rg.TOL_EXACT
    # per-pair loop equals the vectorised kernel (kernel validity)
    P64 = st_a["P64"]
    _, rl = ta.pearson_loop(P64, st_a["ia"][sub], st_a["ib"][sub])
    rv = tg.pair_pearson(P64, st_a["ia"][sub], st_a["ib"][sub])[1]
    checks["vectorised_kernel_equals_loop_on_subset"] = bool(np.nanmax(np.abs(rl - rv)) <= 1e-12 and (np.isnan(rl) == np.isnan(rv)).all())
    return checks, res


def step_target_audit(ctx, threads, sections="all", log=log, with_level_b=False):
    from cpg_repr_benchmark.bioval_v2_followup import strata as st  # noqa: F401
    d = rg.STEP_DIRS["target-audit"]
    ctx.zero_union_cached(full=True)
    log("[step1] Level A (counts -> target)")
    checks, res, state = level_a(ctx, log)
    if with_level_b:
        log("[step1] Level B (fasta + raw .pat.gz)  [NOT a registered run: the gate refuses this flag]")
        cb, rb = level_b(ctx, state, threads, log)
        checks.update(cb)
        res.update(rb)
    else:
        log("[step1] Level B NOT executed (user decision D-L1, amendment 1): raw .pat counts/parsing are not verified")
        res["raw_level_B"] = {"executed": False, "reason": "user decision D-L1 (cost); see registration amendment 1"}
    verdict = ta.build_verdict(checks, level_b_executed=with_level_b)
    verdict.update({"checks": checks, "amendment": "1+2 (2026-10-05)", "timestamp_utc": gt.datetime.now(gt.timezone.utc).isoformat(),
                    "registration_sha256": sha256_file(ctx.root / rg.REGISTRATION_DOC), "code_sha256": gt.current_code_sha256(ctx.root),
                    "head_commit": gt.head_commit(ctx.root)})
    ctx.w.write_json(f"{d}/target_audit_results.json", res)
    ctx.w.write_json(rg.VERDICT_FILE, verdict)
    log(f"[step1] VERDICT: {verdict['verdict']} (level {verdict['level']}, raw_level_B_executed={verdict['raw_level_B_executed']})  failed: {verdict['failed_checks']}")
    if verdict["verdict"] not in rg.ACCEPTED_VERDICTS:
        log("[step1] MISMATCH FOUND: possible BUG. Report BEFORE any other analysis. Section 2 and all later steps are refused.")
        return verdict
    if sections in ("all", "2"):
        log("[step2] split-half reliability")
        r2 = step_reliability(ctx, state, log)
        ctx.w.write_json(f"{d}/reliability.json", r2)
    return verdict


# ================================================================================================= STEP 2: reliability
def step_reliability(ctx, state, log=log):
    from cpg_repr_benchmark.bioval_v2_followup import strata as st
    pairs, prof, D, V, donors = state["pairs"], state["prof"], state["D"], state["V"], state["donors"]
    ia, ib = state["ia"], state["ib"]
    keep = ctx.d2_keep()
    vmin, mmean = st.pair_strata_values(pairs, prof)
    strat = pairs.stratum.to_numpy()
    out = {"unit": "donor", "n_reps": rg.SPLIT_REPS, "seeds": f"default_rng([{rg.SPLIT_SEED_BASE}, rep])", "pair_set": "frozen pairs minus D2, defined in both halves"}
    for variant, min_donors in (("donors_ge2", 2), ("donors_ge4", rg.HIGH_N_MIN_DONORS)):
        per_rep, n_defined = [], []
        for r in range(rg.SPLIT_REPS):
            half, used = rel.make_halves(donors, r, min_donors)
            BA, BB = rel.half_group_betas(D, V, donors, half, used)
            ta_, tb_ = rel.half_similarities(BA, BB, ia, ib, len(used))
            per_rep.append(rel.reliability_table(ta_, tb_, strat, vmin, mmean, keep))
            n_defined.append(int((np.isfinite(ta_) & np.isfinite(tb_) & keep).sum()))
            log(f"    {variant} rep {r + 1}/{rg.SPLIT_REPS}: pooled r={per_rep[-1].get('pooled_gt1Mb_inter_eqw', {}).get('r_spearman')}")
        out[variant] = {"groups_used": used, "n_groups_used": len(used), "min_shared": rel.min_shared_for(len(used)),
                        "n_pairs_defined_per_rep": percentile_summary(n_defined), "summary": rel.summarize_reps(per_rep)}
    out["single_sample_groups_not_estimable"] = [g for g in prof.groups if (donors.group == g).sum() < 2]
    out["formulas"] = {"spearman_brown": "rho = 2 r / (1 + r)", "ceiling": "sqrt(max(rho, 0)) (classical attenuation bound; descriptive)"}
    return out


# ================================================================================================= STEPS 3-4: semantics + pair-list audit
COV_COLS = ("gc_content", "cpg_density", "cpg_density_hg38", "tss_dist")


def build_cpg_table(ctx, prof, P, union):
    mean, var, rng_, n = sm.per_cpg_profile_stats(P)
    L = sm.global_group_level(P)
    r2 = sm.r2_with_level(P, L)
    uni = ctx.universe().set_index("cpg_idx").loc[prof.cpg]
    fs, cols, assay = feature_columns(ctx)
    tot, hist, dn, ids = row_peak_counts(ctx, cols, assay)
    order = prof.rows(ids)
    peaks = {k: np.full(len(prof.cpg), -1, np.int32) for k in ("peak_total", "peak_hist", "peak_dnase")}
    for k, v in zip(peaks, (tot, hist, dn), strict=True):
        peaks[k][order] = v
    t = pd.DataFrame({"cpg_idx": prof.cpg, "chrom": prof.chrom, "pos": uni.pos.to_numpy(), "M": mean, "V": var, "R": rng_, "n_groups": n, "r2_level": r2,
                      "context": uni.context.to_numpy(), "d2_zero_union": np.isin(prof.cpg, union), **peaks})
    for c in COV_COLS:
        t[c] = uni[c].to_numpy()
    return t, L, {"n_features": len(cols), "feature_set": fs.name, "feature_set_manifest_hash": fs.manifest_hash}


def step_semantics(ctx, log=log):
    from cpg_repr_benchmark.bioval_v2_followup import strata as st  # noqa: F401
    d = rg.STEP_DIRS["semantics-pairs"]
    prof = load_prof(ctx)
    P = prof.matrix("primary")
    pairs = ctx.pairs()
    union, _ = ctx.zero_union_cached()
    keep = ctx.d2_keep(union)
    log("  per-CpG table (profile stats, covariates, CSR peak counts)")
    ct, L, feat = build_cpg_table(ctx, prof, P, union)
    if (ct[["peak_total"]] < 0).any().any():
        raise AssertionError("track store does not cover the whole universe")
    ct.to_parquet(ctx.w.path(d, "cpg_table.parquet"), index=False)
    ia, ib = prof.rows(pairs.cpg_i.to_numpy()), prof.rows(pairs.cpg_j.to_numpy())
    tgt = np.where(keep, pairs.loyfer_pearson.to_numpy(), np.nan)
    vmin = np.minimum(ct.V.to_numpy()[ia], ct.V.to_numpy()[ib])
    rmin = np.minimum(ct.R.to_numpy()[ia], ct.R.to_numpy()[ib])
    mmean = (ct.M.to_numpy()[ia] + ct.M.to_numpy()[ib]) / 2
    r2c = ct.r2_level.to_numpy()
    r_small = float(np.quantile(rmin[keep & np.isfinite(rmin)], rg.SMALL_AMPLITUDE_QUANTILE))
    g_hi = float(np.nanquantile(r2c[ct.n_groups.to_numpy() >= rg.MIN_SHARED], rg.GLOBAL_LEVEL_QUANTILE))
    cls = sm.classify_pairs(vmin, rmin, r2c[ia], r2c[ib], r_small, g_hi)
    resid = sm.pair_residual_pearson(P, ia, ib, L)
    bands, th = sm.band_masks(tgt)
    strat = pairs.stratum.to_numpy()
    values = {"pair_mean_beta": mmean, "pair_min_variance": vmin, "pair_min_range": rmin, "n_informative_groups": pairs.n_shared_groups.to_numpy().astype(float),
              "pair_max_variance": np.maximum(ct.V.to_numpy()[ia], ct.V.to_numpy()[ib]), "target_pearson": tgt, "pearson_after_global_level_removed": resid}
    summ = sm.band_summary(values, bands, cls, resid, tgt)
    shares_all = summ["all"]["class_shares"]
    dom = sm.topband_dominance(summ["top1pct"]["class_shares"], shares_all)
    # per-stratum x class
    per_stratum_class = {s: sm.class_shares(cls, keep & (strat == s)) for s in rg.STRATA}
    # deterministic examples (profiles tabulated)
    rows = []
    for bi, b in enumerate(("top1pct", "median", "bottom1pct")):
        for k in sm.select_examples(bands[b], tag=bi):
            rows.append({"band": b, "pair_row": int(k), "cpg_i": int(pairs.cpg_i.iat[k]), "cpg_j": int(pairs.cpg_j.iat[k]), "stratum": strat[k],
                         "target_pearson": float(tgt[k]), "pearson_after_global_level_removed": float(resid[k]), "class": sm.CLASSES[int(cls[k])],
                         "n_shared": int(pairs.n_shared_groups.iat[k]), "mean_i": float(ct.M.iat[ia[k]]), "mean_j": float(ct.M.iat[ib[k]]),
                         "var_i": float(ct.V.iat[ia[k]]), "var_j": float(ct.V.iat[ib[k]]), "range_i": float(ct.R.iat[ia[k]]), "range_j": float(ct.R.iat[ib[k]]),
                         "profile_i": json.dumps([None if np.isnan(x) else round(float(x), 5) for x in P[ia[k]]]),
                         "profile_j": json.dumps([None if np.isnan(x) else round(float(x), 5) for x in P[ib[k]]])})
    ctx.w.write_csv(f"{d}/semantics_examples.csv", pd.DataFrame(rows))
    ctx.w.write_json(f"{d}/groups.json", prof.groups)
    # ---------------- part 4
    cov = {}
    for name, arr in (("methylation_mean", ct.M), ("methylation_variance", ct.V), ("amplitude_range", ct.R), ("gc_content", ct.gc_content),
                      ("cpg_density", ct.cpg_density), ("cpg_density_hg38", ct.cpg_density_hg38), ("tss_dist_log10", ct.tss_dist),
                      ("peak_count_total", ct.peak_total), ("peak_count_histone", ct.peak_hist), ("peak_count_dnase", ct.peak_dnase)):
        a = arr.to_numpy().astype(float)
        m_, ad = sm.pair_level(a[ia], a[ib])
        cov[f"{name}__pair_mean"], cov[f"{name}__abs_diff"] = m_[keep], ad[keep]
    cov["target_pearson"] = tgt[keep]
    ref = np.isin(strat[keep], rg.POOLED_STRATA)
    cmpr = sm.compare_strata(cov, strat[keep], ref)
    ctxs = ct.context.to_numpy()
    cat = {s: {"same_context": float((ctxs[ia] == ctxs[ib])[keep & (strat == s)].mean()),
               "context_share_i": pd.Series(ctxs[ia][keep & (strat == s)]).value_counts(normalize=True).to_dict()} for s in rg.STRATA}
    comp = sm.chrom_composition(ct.chrom.to_numpy()[ia][keep], strat[keep])
    excl = {s: {"n": int((strat == s).sum()), "n_excluded_D2": int(((strat == s) & ~keep).sum())} for s in rg.STRATA}
    out = {"EXPLORATORY_POST_HOC": True, "thresholds_recorded": {"R_small_min_pair_range_q25": r_small, "G_hi_r2_level_q75": g_hi,
                                                                   "V_edges_registered": list(rg.V_EDGES), "band_thresholds": th,
                                                                   "global_group_level_L": dict(zip(prof.groups, L.tolist(), strict=True))},
           "band_summary": summ, "topband_dominance": dom, "class_shares_per_stratum": per_stratum_class,
           "per_stratum_vs_reference_(>1Mb+inter)": cmpr, "context_per_stratum": cat, "chromosome_composition": comp, "zero_row_exclusion_per_stratum": excl,
           "feature_space": feat, "n_pairs_kept": int(keep.sum())}
    ctx.w.write_json(f"{d}/semantics_pairs.json", out)
    pt = pd.DataFrame({"row": np.arange(len(pairs)), "cpg_i": pairs.cpg_i, "cpg_j": pairs.cpg_j, "stratum": strat, "distance_bp": pairs.distance_bp,
                       "target": pairs.loyfer_pearson, "n_shared": pairs.n_shared_groups, "d2_keep": keep, "row_i": ia, "row_j": ib,
                       "chrom_i": ct.chrom.to_numpy()[ia], "chrom_j": ct.chrom.to_numpy()[ib], "vmin": vmin, "rmin": rmin, "mmean": mmean,
                       "class": cls, "resid_pearson": resid})
    for c in ("M", "V", "R", "gc_content", "cpg_density", "cpg_density_hg38", "tss_dist", "peak_total", "peak_hist", "peak_dnase"):
        a = ct[c].to_numpy()
        pt[f"{c}_i"], pt[f"{c}_j"] = a[ia], a[ib]
    pt["context_i"], pt["context_j"] = ctxs[ia], ctxs[ib]
    pt.to_parquet(ctx.w.path(d, "pair_table.parquet"), index=False)
    log(f"  topband dominance: {dom}")
    return out


# ================================================================================================= STEP 5: baselines (+ precision of the frozen target)
def load_pair_table(ctx) -> pd.DataFrame:
    p = ctx.out_root / rg.STEP_DIRS["semantics-pairs"] / "pair_table.parquet"
    if not p.is_file():
        raise SystemExit(f"{p} missing: run `semantics-pairs` first")
    return pd.read_parquet(p)


def kept_pairs(ctx):
    pt = load_pair_table(ctx)
    k = pt.d2_keep.to_numpy()
    return pt, k, pt[k].reset_index(drop=True)


def score_stats(boot, name, x, y, strat, blk, ci=True, reps=None):
    """primary-style pooled statistic + the six per-stratum Spearmans (+ all pairs, point) of the score ``y`` against target ``x``.
    If ``reps`` (dict) is given the bootstrap replicates of the pooled statistic are stored under ``name``."""
    sp = stat_pooled(boot, f"{name}__pooled", x, y, strat, blk, ci=ci)
    if reps is not None and sp is not None and sp.rep is not None:
        reps[name] = sp.rep
    out = {"pooled_gt1Mb_inter_eqw": stat_dict(sp), "per_stratum": {}}
    for s in rg.STRATA:
        out["per_stratum"][s] = stat_dict(stat_unweighted(boot, f"{name}__{s}", x, y, blk, mask=(strat == s), ci=ci))
    out["all_pairs"] = stat_dict(stat_unweighted(boot, f"{name}__all", x, y, blk, ci=False))
    return out


def covariate_scores(d: pd.DataFrame) -> dict:
    sc = {"chromosome_only": bl.same_chromosome(d.chrom_i, d.chrom_j), "neg_log10_distance": bl.neg_log10_distance(d.distance_bp.to_numpy()),
          "same_context": bl.same_category(d.context_i, d.context_j)}
    for c, label in (("M", "methylation_mean"), ("V", "methylation_variance"), ("R", "amplitude_range"), ("gc_content", "gc"),
                     ("cpg_density", "cpg_density"), ("cpg_density_hg38", "cpg_density_hg38"), ("tss_dist", "tss_dist_log10"),
                     ("peak_total", "peak_count_total"), ("peak_hist", "peak_count_histone"), ("peak_dnase", "peak_count_dnase")):
        sc[f"neg_abs_delta_{label}"] = bl.neg_abs_diff(d[f"{c}_i"], d[f"{c}_j"])
    return sc


def step_baselines(ctx, arms, log=log):
    dname = rg.STEP_DIRS["baselines"]
    pt, keep, d = kept_pairs(ctx)
    boot = make_boot()
    x = d.target.to_numpy()
    strat = d.stratum.to_numpy()
    blk = block_of(boot, d.chrom_i.to_numpy())
    unr_all = np.load(ctx.out_root / rg.STEP_DIRS["target-audit"] / "pair_pearson_unrounded.npy")
    unr = unr_all[keep]
    res = {"EXPLORATORY_POST_HOC": True, "n_pairs": len(d), "seeds": {"shuffle": [rg.SHUFFLE_SEED_BASE, rg.N_SHUFFLE], "random": [rg.RANDOM_SEED_BASE, rg.N_RANDOM]}}
    # ---- covariate / chromosome / distance baselines (arm independent)
    log("  covariate baselines")
    res["covariate_baselines"] = {}
    for name, y in covariate_scores(d).items():
        r = score_stats(boot, name, x, y, strat, blk)
        if name == "neg_log10_distance":
            r["intra_5strata_equal_weight"] = stat_dict(stat_pooled(boot, name + "__intra5", x, y, strat, blk, pooled=rg.STRATA[:5]))
        res["covariate_baselines"][name] = r
        log(f"    {name}: pooled {r['pooled_gt1Mb_inter_eqw']}")
    ids_u = ctx.universe().cpg_idx.to_numpy()
    ia_u, ib_u = ctx.urow(d.cpg_i.to_numpy()), ctx.urow(d.cpg_j.to_numpy())
    res["arms"] = {}
    stats_native = {}
    for arm in arms:
        log(f"  arm {arm}")
        store = ctx.store(arm)
        E = store.load(ids_u)
        cos = pair_cosine(E, ia_u, ib_u)
        if not np.isfinite(cos).all():
            raise AssertionError("non-finite cosine on D2-kept pairs")
        a = {"dim": int(E.shape[1])}
        nat = score_stats(boot, f"native_{arm}", x, cos, strat, blk)
        s_pool = stat_pooled(boot, f"native_{arm}__pooled", x, cos, strat, blk)
        stats_native[arm] = s_pool
        fv = ctx.frozen_primary(arm)
        if abs(s_pool.value - fv) > 1e-9:
            raise AssertionError(f"{arm}: native pooled statistic {s_pool.value} does not reproduce the frozen primary {fv}")
        a["native"], a["frozen_primary"] = nat, fv
        ctx.w.save_npz(f"{dname}/reps_{arm}.npz", native=s_pool.rep)
        ctx.w.save_npy(f"{dname}/native_cosine_{arm}.npy", cos.astype(np.float64))
        # precision of the frozen (float16) target: primary on the unrounded target
        u = stat_pooled(boot, f"native_{arm}__pooled_unrounded", unr, cos, strat, blk, ci=False)
        a["unrounded_target_pooled"] = stat_dict(u)
        a["unrounded_minus_frozen"] = (u.value - fv) if u is not None else None
        # nulls
        shuf, rnd = [], []
        for k in range(rg.N_SHUFFLE):
            y = bl.shuffled_cosine(E, ia_u, ib_u, rg.SHUFFLE_SEED_BASE + k)
            s = stat_pooled(boot, "n", x, y, strat, blk, ci=False)
            shuf.append(None if s is None else s.value)
        del E
        for k in range(rg.N_RANDOM):
            y = bl.random_gaussian_cosine(len(ids_u), rg.ARM_DIMS[arm], rg.RANDOM_SEED_BASE + k, ia_u, ib_u)
            s = stat_pooled(boot, "n", x, y, strat, blk, ci=False)
            rnd.append(None if s is None else s.value)
        a["null_shuffled_pooled"], a["null_random_gaussian_pooled"] = bl.null_summary(shuf), bl.null_summary(rnd)
        a["null_hi"] = bl.null_hi(shuf, rnd)
        a["null_values"] = {"shuffled": shuf, "random": rnd}
        a["native_minus_null_hi"] = s_pool.value - a["null_hi"]
        # per-stratum null (first 20 draws of each family; point estimates) for the per-stratum reading
        a["mediation_partial_spearman"] = _mediation(x, cos, strat, d)
        res["arms"][arm] = a
        ctx.w.write_json(f"{dname}/arm_{arm}.json", a)
    # ---- paired contrasts of the native statistic (candidate - comparator) + ordering under the unrounded target
    if all(a in stats_native for a in rg.ARMS):
        res["contrasts_native"] = {c: paired_delta(stats_native[rg.CANDIDATE], stats_native[c]) for c in rg.ARMS if c != rg.CANDIDATE}
        fz = {a: res["arms"][a]["frozen_primary"] for a in rg.ARMS}
        un = {a: res["arms"][a]["unrounded_target_pooled"]["value"] for a in rg.ARMS}
        res["ordering_frozen_target"] = sorted(fz, key=lambda k: -fz[k])
        res["ordering_unrounded_target"] = sorted(un, key=lambda k: -un[k])
        res["precision_order_changed"] = res["ordering_frozen_target"] != res["ordering_unrounded_target"]
    ctx.w.write_json(f"{dname}/baselines.json", res)
    return res


def _mediation(x, cos, strat, d):
    """Per stratum: Spearman(target, cosine) and partial Spearman controlling for covariate closeness (-|dM|, -|dGC|, -|dpeak|, [-log10 d])."""
    out = {}
    base = {"neg_abs_dM": -np.abs(d.M_i - d.M_j).to_numpy(), "neg_abs_dGC": -np.abs(d.gc_content_i - d.gc_content_j).to_numpy(),
            "neg_abs_dPeak": -np.abs(d.peak_total_i - d.peak_total_j).to_numpy().astype(float)}
    ld = bl.neg_log10_distance(d.distance_bp.to_numpy())
    for s in rg.STRATA:
        m = strat == s
        Z = np.column_stack([v[m] for v in base.values()])
        Zd = np.column_stack([Z, ld[m]]) if s != "interchromosomal" else Z
        out[s] = {"spearman": spearman_fast(x[m], cos[m]), "partial_given_dM_dGC_dPeak": sm.partial_spearman(x[m], cos[m], Z),
                  "partial_given_dM_dGC_dPeak_logdist": sm.partial_spearman(x[m], cos[m], Zd),
                  "spearman_cosine_vs_neg_abs_dGC": spearman_fast(cos[m], base["neg_abs_dGC"][m]),
                  "spearman_target_vs_neg_abs_dGC": spearman_fast(x[m], base["neg_abs_dGC"][m]),
                  "spearman_cosine_vs_neg_abs_dPeak": spearman_fast(cos[m], base["neg_abs_dPeak"][m]),
                  "spearman_target_vs_neg_abs_dPeak": spearman_fast(x[m], base["neg_abs_dPeak"][m])}
    return out


# ================================================================================================= STEP 6: raw regulatory space vs 256-d embedding
def step_raw_vs_svd(ctx, log=log):
    from cpg_repr_benchmark.bioval_v2_followup import strata as st  # noqa: F401
    from cpg_repr_benchmark.encode_atlas.compression import TrackStore
    dname = rg.STEP_DIRS["raw-vs-svd"]
    pt, keep, d = kept_pairs(ctx)
    boot = make_boot()
    x, strat = d.target.to_numpy(), d.stratum.to_numpy()
    blk = block_of(boot, d.chrom_i.to_numpy())
    fs, cols, assay = feature_columns(ctx)
    log("  loading CSR track store (cpg_idx, track_indptr, track_indices only)")
    store = TrackStore(ctx.rp(rg.TRACK_STORE))
    if "dense" in store.datasets_read:
        raise AssertionError("/dense was read")
    ids_u = ctx.universe().cpg_idx.to_numpy()
    X = store.matrix[store.rows_for(ids_u)][:, cols].tocsr()
    z = np.load(ctx.rp(rg.COMPRESSOR_NPZ), allow_pickle=False)
    weights = z["weights"]
    w_is_one = bool(np.all(weights == 1.0))
    if not np.array_equal(z["columns"], cols):
        raise AssertionError("compressor columns differ from the resolved feature set")
    est = ctx.store(rg.CANDIDATE)
    rng = np.random.default_rng(rg.SEED)
    samp = np.sort(rng.choice(len(ids_u), 200, replace=False))
    proj = rs.verify_projection(X[samp], z["projection"], est.load(ids_u[samp]))
    ia_u, ib_u = ctx.urow(d.cpg_i.to_numpy()), ctx.urow(d.cpg_j.to_numpy())
    zero_raw = rs.all_zero_rows(X)
    met = rs.pair_binary_metrics(X, ia_u, ib_u, weights=None if w_is_one else weights)
    E = est.load(ids_u)
    native = pair_cosine(E, ia_u, ib_u)
    common = np.isfinite(met["binary_cosine"]) & np.isfinite(met["jaccard"])
    res = {"EXPLORATORY_POST_HOC": True, "projection_reproduction": proj, "compressor_weights_all_one": w_is_one,
           "weighted_cosine_note": "weights are all 1 (replicate_weighting none): the column-weighted cosine IS the binary cosine; not reported separately"
                                   if w_is_one else "weights != 1: weighted cosine reported",
           "n_raw_all_zero_loci": int(zero_raw.sum()), "n_pairs_D2": len(d), "n_pairs_raw_undefined_after_D2": int((~common).sum()),
           "n_pairs_common_set": int(common.sum()), "n_tracks": len(cols)}
    vtop = d.vmin.to_numpy() >= rg.V_TOP10
    reps = {}
    scores = {"raw_binary_cosine": met["binary_cosine"], "raw_jaccard": met["jaccard"], "embedding256_cosine_native": native}
    if "weighted_cosine" in met:
        scores["raw_weighted_cosine"] = met["weighted_cosine"]
    res["scores"], stats = {}, {}
    for name, y in scores.items():
        c = common
        r = score_stats(boot, name, x[c], y[c], strat[c], blk[c], reps=reps)
        r["pooled_V_top10pct"] = stat_dict(stat_pooled(boot, name + "__vtop", x, y, strat, blk, mask=common & vtop))
        r["unweighted_V_top10pct"] = stat_dict(stat_unweighted(boot, name + "__vtop_unw", x, y, blk, mask=common & vtop))
        res["scores"][name] = r
        stats[name] = stat_pooled(boot, name, x, y, strat, blk, mask=common)
        log(f"    {name}: pooled {r['pooled_gt1Mb_inter_eqw']}")
    res["delta_vs_native_embedding_pooled"] = {n: paired_delta(stats[n], stats["embedding256_cosine_native"]) for n in scores if n != "embedding256_cosine_native"}
    res["on_D2_set_embedding_native_pooled"] = stat_dict(stat_pooled(boot, "nat_d2", x, native, strat, blk))
    ctx.w.save_npz(f"{dname}/reps.npz", **reps)
    ctx.w.write_json(f"{dname}/raw_vs_svd.json", res)
    return res


# ================================================================================================= STEP 7: SVD geometry
def step_svd_geometry(ctx, arms, log=log):
    dname = rg.STEP_DIRS["svd-geometry"]
    pt, keep, d = kept_pairs(ctx)
    boot = make_boot()
    x, strat = d.target.to_numpy(), d.stratum.to_numpy()
    blk = block_of(boot, d.chrom_i.to_numpy())
    union, _ = ctx.zero_union_cached()
    ids_u = ctx.universe().cpg_idx.to_numpy()
    d2 = ~np.isin(ids_u, union)
    if int(d2.sum()) != rg.N_UNIVERSE_D2:
        raise AssertionError("D2 universe size differs from 407,550")
    ct = pd.read_parquet(ctx.out_root / rg.STEP_DIRS["semantics-pairs"] / "cpg_table.parquet").set_index("cpg_idx").loc[ids_u]
    ia_u, ib_u = ctx.urow(d.cpg_i.to_numpy()), ctx.urow(d.cpg_j.to_numpy())
    covs = {"peak_total": ct.peak_total, "peak_histone": ct.peak_hist, "peak_dnase": ct.peak_dnase, "gc_content": ct.gc_content,
            "cpg_density": ct.cpg_density, "cpg_density_hg38": ct.cpg_density_hg38, "tss_dist_log10": ct.tss_dist,
            "loyfer_mean_methylation": ct.M, "loyfer_methylation_variance": ct.V, "loyfer_amplitude_range": ct.R}
    covs = {k: v.to_numpy().astype(float) for k, v in covs.items()}
    out_all = {}
    for arm in arms:
        log(f"  arm {arm}")
        E = ctx.store(arm).load(ids_u)
        mu, lam, V = sg.universe_pca(E[d2])
        sd = sg.coord_sd(E[d2])
        S = sg.pc_scores(E, mu, V, rg.N_PCS)
        a = {"dim": int(E.shape[1]), "explained_variance_ratio_first20": (lam[:rg.N_PCS] / lam.sum()).tolist(), "mu_sha256": sha256_array(mu),
             "universe_n_loci": int(d2.sum())}
        sel = d2
        a["assoc_pc_scores_spearman"] = sg.spearman_assoc(S[sel], {k: v[sel] for k, v in covs.items()})
        a["assoc_pc_scores_eta2_context"] = sg.eta_squared(S[sel], ct.context.to_numpy()[sel])
        a["assoc_embedding_coordinates_1_20_spearman"] = sg.spearman_assoc(np.asarray(E[sel][:, :rg.N_PCS], np.float64), {k: v[sel] for k, v in covs.items()})
        curve = {"native_cosine": pair_cosine(E, ia_u, ib_u), "centered_cosine": sg.remove_pcs_cosine(E, ia_u, ib_u, mu, V, 0)}
        for k in rg.PC_REMOVE_KS:
            curve[f"remove_pc1_{k}" if k > 1 else "remove_pc1"] = sg.remove_pcs_cosine(E, ia_u, ib_u, mu, V, k)
        curve["standardized_cosine"] = sg.standardized_cosine(E, ia_u, ib_u, mu, sd)
        curve["whitened_cosine"] = sg.whitened_cosine(E, ia_u, ib_u, mu, lam, V)
        curve["neg_euclidean"] = sg.neg_euclidean_pairs(E, ia_u, ib_u)
        reps = {}
        a["curve"] = {}
        for name, y in curve.items():
            a["curve"][name] = score_stats(boot, f"{arm}_{name}", x, y, strat, blk, reps=reps)
            a["curve"][name]["pooled_V_top10pct"] = stat_dict(stat_pooled(boot, name + "_vtop", x, y, strat, blk, mask=d.vmin.to_numpy() >= rg.V_TOP10, ci=False))
            log(f"    {name}: {a['curve'][name]['pooled_gt1Mb_inter_eqw']}")
        a["pc_dot_contribution_pooled_spearman"] = []
        for k in range(rg.N_PCS):
            s = stat_pooled(boot, "pc", x, sg.pc_dot_contribution(S, ia_u, ib_u, k), strat, blk, ci=False)
            a["pc_dot_contribution_pooled_spearman"].append(None if s is None else s.value)
        ref = ctx.root / rg.TAG_DIR / "exploratory_posthoc_loyfer" / arm / "endpoints" / "exploratory_similarity.json"
        if ref.is_file():
            b2 = json.loads(ref.read_text())["stats"]
            a["consistency_with_B2"] = {"pc1_removed_B2": b2["spearman_pooled_eqw__pc1_removed_cosine__all"]["value"],
                                        "pc1_removed_here": a["curve"]["remove_pc1"]["pooled_gt1Mb_inter_eqw"]["value"],
                                        "centered_B2": b2["spearman_pooled_eqw__centered_cosine__all"]["value"],
                                        "centered_here": a["curve"]["centered_cosine"]["pooled_gt1Mb_inter_eqw"]["value"],
                                        "tolerance_informational": 5e-4}
        ctx.w.save_npz(f"{dname}/reps_{arm}.npz", **reps)
        ctx.w.write_json(f"{dname}/arm_{arm}.json", a)
        ctx.w.save_npy(f"{dname}/pc_scores_{arm}.npy", S.astype(np.float32))
        out_all[arm] = a
        del E
    return out_all


# ================================================================================================= STEPS 8-9: OOF profiles / probe fairness
def grid_tag(grid):
    return "g" + "_".join(f"{g:g}" for g in grid)


def fit_arm(ctx, arm, grid, log=log):
    """Chromosome-blocked multi-output ridge (cp.ridge_blocked_cv unchanged) for T1/T2/T3 with the given alpha grid; cached."""
    from cpg_repr_benchmark.bioval_v2_followup import profile_ridge as pr
    from cpg_repr_benchmark.bioval_v2_launch import chrom_probes as cp
    base = ctx.out_root / "fits" / grid_tag(grid) / arm
    if (base / "alpha.npy").is_file():
        return load_fit(base)
    prof = load_prof(ctx)
    P = prof.matrix("primary")
    full = ~np.isnan(P).any(1)
    Pf, ids, chrom = P[full], prof.cpg[full], prof.chrom[full]
    c2f = ctx.c2f()
    fold = pd.Series(chrom).map(c2f).to_numpy().astype(int)
    op.assert_train_excludes_test(chrom, fold, c2f)
    X = ctx.store(arm).load(ids)
    T = pr.assemble_targets(Pf)
    Y = np.concatenate([T[k] for k in rg.B3_FIT_TARGETS], axis=1)
    log(f"  ridge {arm} grid {grid_tag(grid)}: X {X.shape}, Y {Y.shape}")
    fit = cp.ridge_blocked_cv(X, Y, fold, alphas=grid)
    chk = [op.refit_check(X, Y, fold, fit["oof"], fit["alpha"][fi], int(f), (0, 45, 78)) for fi, f in enumerate(fit["folds"])]
    if max(c["max_abs_diff"] for c in chk) > 1e-6:
        raise AssertionError(f"independent refit check disagrees with the OOF predictions: {chk}")
    ctx.w.path("fits", grid_tag(grid), arm, "x")
    base.mkdir(parents=True, exist_ok=True)
    np.save(base / "oof.npy", fit["oof"].astype(np.float32))
    np.save(base / "alpha.npy", fit["alpha"])
    np.save(base / "ids.npy", ids)
    np.save(base / "fold.npy", fold)
    (base / "refit_check.json").write_text(json.dumps(chk))
    return {"oof": fit["oof"].astype(np.float32), "alpha": fit["alpha"], "ids": ids, "fold": fold, "refit_check": chk, "chrom": chrom, "Y": Y}


def load_fit(base):
    prof_fold = np.load(base / "fold.npy")
    return {"oof": np.load(base / "oof.npy"), "alpha": np.load(base / "alpha.npy"), "ids": np.load(base / "ids.npy"), "fold": prof_fold,
            "refit_check": json.loads((base / "refit_check.json").read_text())}


def target_blocks(oof):
    return {"raw": oof[:, :39], "centered": oof[:, 39:78], "level": oof[:, 78:79]}


def oof_eval(ctx, arm, fit, boot, native_cos, d, reps=None):
    """OOF predicted-profile pair similarity (T1 raw, T2 centred) on the frozen pairs with both CpGs complete-case."""
    ids = fit["ids"]
    prof = load_prof(ctx)
    pos_full = np.full(len(prof.cpg), -1, np.int64)
    pos_full[prof.rows(ids)] = np.arange(len(ids))
    pi, pj = pos_full[d.row_i.to_numpy()], pos_full[d.row_j.to_numpy()]
    sel = (pi >= 0) & (pj >= 0)
    if not (d.n_shared.to_numpy()[sel] == rg.N_GROUPS).all():
        raise AssertionError("complete-case pairs must share all 39 groups")
    fold_i, fold_j = fit["fold"][np.maximum(pi, 0)], fit["fold"][np.maximum(pj, 0)]
    same = sel & op.same_fold_mask(fold_i, fold_j)
    x, strat = d.target.to_numpy(), d.stratum.to_numpy()
    blk = block_of(boot, d.chrom_i.to_numpy())
    out = {"n_pairs_D2": len(d), "n_pairs_complete_case": int(sel.sum()), "n_pairs_same_fold": int(same.sum()),
           "n_same_fold_per_stratum": {s: int((same & (strat == s)).sum()) for s in rg.STRATA}}
    tb = target_blocks(fit["oof"])
    nat = stat_pooled(boot, "nat_sel", x, native_cos, strat, blk, mask=sel)
    out["native_cosine_same_pair_set"] = stat_dict(nat)
    vtop = d.vmin.to_numpy() >= rg.V_TOP10
    for t in rg.OOF_TARGETS:
        sim = np.full(len(d), np.nan)
        sim[sel] = op.oof_pair_pearson(tb[t], pi[sel], pj[sel])
        o = {"n_undefined_predicted_profile": int((sel & ~np.isfinite(sim)).sum())}
        o["all_complete_case"] = _cell_stats(boot, f"{arm}_oof_{t}", x, sim, strat, blk, sel, reps)
        o["same_fold_only"] = _cell_stats(boot, f"{arm}_oof_{t}_samefold", x, sim, strat, blk, same, None)
        o["V_top10pct"] = stat_dict(stat_pooled(boot, "v", x, sim, strat, blk, mask=sel & vtop))
        o["all_pairs_unweighted"] = stat_dict(stat_unweighted(boot, "a", x, sim, blk, mask=sel))
        s = stat_pooled(boot, "p", x, sim, strat, blk, mask=sel)
        o["delta_vs_native_same_pairs"] = paired_delta(s, nat)
        out[t] = o
    return out


def _cell_stats(boot, name, x, y, strat, blk, mask, reps):
    sp = stat_pooled(boot, name, x, y, strat, blk, mask=mask)
    if reps is not None and sp is not None and sp.rep is not None:
        reps[name] = sp.rep
    r = {"pooled_gt1Mb_inter_eqw": stat_dict(sp), "per_stratum": {}}
    for s in rg.STRATA:
        r["per_stratum"][s] = stat_dict(stat_unweighted(boot, f"{name}_{s}", x, y, blk, mask=mask & (strat == s)))
    return r


def b3_reproduction(ctx, arm, fit):
    """Reproduce the stored B3 r2_global (model) of the original grid (raw/centered/level x all/variable) -- diagnostic of the identical protocol."""
    from cpg_repr_benchmark.bioval_v2_followup import profile_ridge as pr
    from cpg_repr_benchmark.bioval_v2_followup import strata as st
    prof = load_prof(ctx)
    P = prof.matrix("primary")
    full = ~np.isnan(P).any(1)
    Pf = P[full]
    _, cvar, _ = st.per_cpg_stats(Pf)
    T = pr.assemble_targets(Pf)
    tb = target_blocks(fit["oof"])
    nb = 22
    boot = make_boot()
    blk = block_of(boot, prof.chrom[full])
    ref = json.loads((ctx.root / rg.TAG_DIR / "exploratory_posthoc_loyfer" / arm / "endpoints" / "B3_profile_ridge.json").read_text())
    out = {}
    for t in rg.B3_FIT_TARGETS:
        for sname, sm_ in (("all_cpgs", np.ones(len(cvar), bool)), ("variable_cpgs", cvar >= rg.V_EDGES[2])):
            S = pr.SuffStats(T[t][sm_], tb[t][sm_], blk[sm_], nb)
            v = S.metrics(np.ones(nb))["r2_global"]
            out[f"{t}__{sname}"] = {"here": v, "stored_B3": ref["targets"][t]["subsets"][sname]["model"]["r2_global"]}
    out["max_abs_diff"] = max(abs(o["here"] - o["stored_B3"]) for o in out.values())
    return out


def r2_reps(ctx, arm, fit):
    from cpg_repr_benchmark.bioval_v2_followup import profile_ridge as pr
    from cpg_repr_benchmark.bioval_v2_followup import strata as st
    prof = load_prof(ctx)
    P = prof.matrix("primary")
    full = ~np.isnan(P).any(1)
    Pf = P[full]
    _, cvar, _ = st.per_cpg_stats(Pf)
    T = pr.assemble_targets(Pf)
    tb = target_blocks(fit["oof"])
    boot = make_boot()
    blk = block_of(boot, prof.chrom[full])
    res, reps = {}, {}
    for t in rg.B3_FIT_TARGETS:
        for sname, sm_ in (("all_cpgs", np.ones(len(cvar), bool)), ("variable_cpgs", cvar >= rg.V_EDGES[2])):
            S = pr.SuffStats(T[t][sm_], tb[t][sm_], blk[sm_], 22)
            rep = pr.bootstrap_metrics(S, boot.mult)["r2_global"]
            res[f"{t}__{sname}"] = S.metrics(np.ones(22))["r2_global"]
            reps[f"{t}__{sname}"] = rep
    return res, reps


def step_oof_similarity(ctx, arms, log=log, grid=rg.ALPHA_GRID_ORIGINAL, step="oof-similarity"):
    dname = rg.STEP_DIRS[step]
    pt, keep, d = kept_pairs(ctx)
    boot = make_boot()
    out = {}
    for arm in arms:
        fit = fit_arm(ctx, arm, grid, log)
        native = np.load(ctx.out_root / rg.STEP_DIRS["baselines"] / f"native_cosine_{arm}.npy")
        reps = {}
        a = {"alpha_grid": list(grid), "alpha_edge_report": op.alpha_edge_report(fit["alpha"], grid),
             "alpha_median_per_fold_all_targets": np.median(fit["alpha"], axis=1).tolist(), "refit_check_max_abs_diff": max(c["max_abs_diff"] for c in fit["refit_check"]),
             "oof_similarity": oof_eval(ctx, arm, fit, boot, native, d, reps)}
        a["r2_global"], r2rep = r2_reps(ctx, arm, fit)
        reps.update({f"r2__{k}": v for k, v in r2rep.items()})
        if grid == rg.ALPHA_GRID_ORIGINAL:
            a["B3_reproduction"] = b3_reproduction(ctx, arm, fit)
        ctx.w.save_npz(f"{dname}/reps_{arm}_{grid_tag(grid)}.npz", **reps)
        ctx.w.write_json(f"{dname}/arm_{arm}_{grid_tag(grid)}.json", a)
        out[arm] = a
        log(f"  {arm}: T1 pooled {a['oof_similarity']['raw']['all_complete_case']['pooled_gt1Mb_inter_eqw']}")
    return out


def step_probe_fairness(ctx, arms, log=log):
    """Extended alpha grid with the registered edge rule (identical for all arms), then the same evaluation as step 8."""
    grid, n_ext, history = rg.ALPHA_GRID_EXTENDED, 0, []
    while True:
        reps_edge = {}
        for arm in arms:
            reps_edge[arm] = op.alpha_edge_report(fit_arm(ctx, arm, grid, log)["alpha"], grid)
        history.append({"grid": list(grid), "edge_reports": reps_edge})
        new, ext = op.next_grid(grid, reps_edge, n_ext)
        if not ext:
            break
        grid, n_ext = new, n_ext + 1
    res = step_oof_similarity(ctx, arms, log, grid=grid, step="probe-fairness")
    ctx.w.write_json(f"{rg.STEP_DIRS['probe-fairness']}/grid_history.json", {"history": history, "final_grid": list(grid), "n_extensions": n_ext})
    return {"final_grid": list(grid), "n_extensions": n_ext, "arms": res}


# ================================================================================================= REPORT INPUTS + integrity
def _j(ctx, rel):
    p = ctx.out_root / rel
    if not p.is_file():
        raise SystemExit(f"missing prerequisite output {p}")
    return json.loads(p.read_text())


def _contrast_from_reps(ctx, step_dir, fn_a, key_a, fn_b, key_b, sign=1):
    ra, rb = np.load(ctx.out_root / step_dir / fn_a), np.load(ctx.out_root / step_dir / fn_b)
    return ra[key_a], rb[key_b]


def step_report_inputs(ctx, log=log):
    from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import contrast
    s1 = _j(ctx, rg.VERDICT_FILE)
    relj = _j(ctx, f"{rg.STEP_DIRS['target-audit']}/reliability.json")["donors_ge2"]["summary"]
    sem = _j(ctx, f"{rg.STEP_DIRS['semantics-pairs']}/semantics_pairs.json")
    bs = _j(ctx, f"{rg.STEP_DIRS['baselines']}/baselines.json")
    rv = _j(ctx, f"{rg.STEP_DIRS['raw-vs-svd']}/raw_vs_svd.json")
    cand = rg.CANDIDATE
    null_hi = bs["arms"][cand]["null_hi"]
    native = bs["arms"][cand]["frozen_primary"]
    g = lambda dct, *k: _get(dct, *k)
    precision = {a: bs["arms"][a]["unrounded_minus_frozen"] for a in rg.ARMS}
    flags = {"BUG": dc.rule_bug(s1)}
    flags["TARGET_LIMITATION"] = dc.rule_target_limitation(
        g(relj, "pooled_gt1Mb_inter_eqw", "ceiling", "median"), g(relj, "dist_>1Mb", "rho_sb", "median"),
        g(relj, "dist_interchromosomal", "rho_sb", "median"), sem["topband_dominance"], precision, bool(bs["precision_order_changed"]))
    rawv = {k: rv["scores"][k]["pooled_gt1Mb_inter_eqw"]["value"] for k in ("raw_binary_cosine", "raw_jaccard")}
    raw_nat = rv["scores"]["embedding256_cosine_native"]["pooled_gt1Mb_inter_eqw"]["value"]
    oof_orig = _j(ctx, f"{rg.STEP_DIRS['oof-similarity']}/arm_{cand}_{grid_tag(rg.ALPHA_GRID_ORIGINAL)}.json")
    pf = _j(ctx, f"{rg.STEP_DIRS['probe-fairness']}/grid_history.json")
    final_grid = tuple(pf["final_grid"])
    oof_ext = _j(ctx, f"{rg.STEP_DIRS['probe-fairness']}/arm_{cand}_{grid_tag(final_grid)}.json")
    oofv, delta_ci = {}, {}
    for tag, o in (("orig", oof_orig), ("ext", oof_ext)):
        for t in rg.OOF_TARGETS:
            k = f"{t}_{tag}"
            oofv[k] = o["oof_similarity"][t]["all_complete_case"]["pooled_gt1Mb_inter_eqw"]["value"]
            dd = o["oof_similarity"][t]["delta_vs_native_same_pairs"]
            delta_ci[f"oof:{k}"] = (dd["ci_lo"], dd["ci_hi"]) if dd else None
    for k, dd in rv["delta_vs_native_embedding_pooled"].items():
        if k in rawv:
            delta_ci[f"raw:{k}"] = (dd["ci_lo"], dd["ci_hi"]) if dd else None
    sv = _j(ctx, f"{rg.STEP_DIRS['svd-geometry']}/arm_{cand}.json")
    pc_support = {}
    nat_rep = np.load(ctx.out_root / rg.STEP_DIRS["svd-geometry"] / f"reps_{cand}.npz")
    for name in sv["curve"]:
        if name == "native_cosine":
            continue
        v = sv["curve"][name]["pooled_gt1Mb_inter_eqw"]["value"]
        c = contrast(v, nat_rep[f"{cand}_{name}"], sv["curve"]["native_cosine"]["pooled_gt1Mb_inter_eqw"]["value"], nat_rep[f"{cand}_native_cosine"])
        pc_support[name] = bool(c["delta"] >= rg.M_REC and c["ci_lo"] > 0)
    flags["COMPRESSION_GEOMETRY"] = dc.rule_compression_geometry(native, null_hi, rawv, {k: v for k, v in oofv.items()}, delta_ci, pc_support)
    flags["INFORMATION_ABSENCE"] = dc.rule_information_absence(null_hi, {**{f"raw:{k}": v for k, v in rawv.items()}, **{f"oof:{k}": v for k, v in oofv.items()}})
    # comparator advantage (paired, comparator - candidate) over the fair evaluations
    comp_delta, comp_vals, nh = {}, {}, {}
    curve_names = ("native_cosine", "centered_cosine", "remove_pc1", "remove_pc1_10", "whitened_cosine", "neg_euclidean")
    for c in rg.COMPARATORS:
        svc = _j(ctx, f"{rg.STEP_DIRS['svd-geometry']}/arm_{c}.json")
        repc = np.load(ctx.out_root / rg.STEP_DIRS["svd-geometry"] / f"reps_{c}.npz")
        ev, vals = {}, {}
        for n in curve_names:
            vc, vk = svc["curve"][n]["pooled_gt1Mb_inter_eqw"]["value"], sv["curve"][n]["pooled_gt1Mb_inter_eqw"]["value"]
            dd = contrast(vc, repc[f"{c}_{n}"], vk, nat_rep[f"{cand}_{n}"])
            ev[n], vals[n] = (dd["delta"], dd["ci_lo"], dd["ci_hi"]), vc
        for tag, grid, step in (("orig", rg.ALPHA_GRID_ORIGINAL, "oof-similarity"), ("ext", final_grid, "probe-fairness")):
            fn = f"arm_{{}}_{grid_tag(grid)}.json"
            oc = _j(ctx, f"{rg.STEP_DIRS[step]}/{fn.format(c)}")
            rc = np.load(ctx.out_root / rg.STEP_DIRS[step] / f"reps_{c}_{grid_tag(grid)}.npz")
            rk = np.load(ctx.out_root / rg.STEP_DIRS[step] / f"reps_{cand}_{grid_tag(grid)}.npz")
            key = f"{c}_oof_raw"
            kk = f"{cand}_oof_raw"
            vc = oc["oof_similarity"]["raw"]["all_complete_case"]["pooled_gt1Mb_inter_eqw"]["value"]
            vk = oofv[f"raw_{tag}"]
            dd = contrast(vc, rc[key], vk, rk[kk])
            ev[f"oof_T1_{tag}"], vals[f"oof_T1_{tag}"] = (dd["delta"], dd["ci_lo"], dd["ci_hi"]), vc
        comp_delta[c], comp_vals[c], nh[c] = ev, vals, bs["arms"][c]["null_hi"]
    flags["COMPARATOR_ADVANTAGE"] = dc.rule_comparator_advantage(comp_delta, nh, comp_vals)
    final = dc.classify(flags)
    ctx.w.write_json(f"{rg.STEP_DIRS['report-inputs']}/REPORT_INPUTS.json", {"EXPLORATORY_POST_HOC": True, "decision": final, "thresholds": {
        "M_REC": rg.M_REC, "CEILING_MIN": rg.CEILING_MIN, "REL_STRATUM_MIN": rg.REL_STRATUM_MIN, "PRECISION_DELTA": rg.PRECISION_DELTA},
        "null_hi_candidate": null_hi, "raw_native_same_set": raw_nat})
    log(f"  classes: {final['classes']}")
    return final


def _get(d, *keys):
    for k in keys:
        if d is None:
            return float("nan")
        d = d.get(k) if isinstance(d, dict) else None
    return float("nan") if d is None else d


def step_integrity(ctx, mode):
    w = ctx.w
    if mode == "snapshot":
        snap = gt.snapshot_tree(ctx.root)
        if snap["primary_fingerprint"] != rg.FROZEN_RESULTS_SHA256:
            raise SystemExit("primary fingerprint differs from the registered value; refusing to snapshot")
        if (ctx.out_root / rg.SNAPSHOT_BEFORE).is_file():
            raise SystemExit("SNAPSHOT_BEFORE already exists; it is written once, before the first real step")
        w.write_json(rg.SNAPSHOT_BEFORE, snap)
        return {"n_files": snap["n_files"], "tree_sha256": snap["tree_sha256"]}
    before = json.loads((ctx.out_root / rg.SNAPSHOT_BEFORE).read_text())
    after = gt.snapshot_tree(ctx.root)
    w.write_json(rg.SNAPSHOT_AFTER, after)
    cmpd = gt.compare_snapshots(before, after)
    cmpd["audit_root_outside_fingerprinted_tree"] = gt.fingerprint_excludes_audit(ctx.root)
    w.write_json("integrity/COMPARE.json", cmpd)
    return cmpd
