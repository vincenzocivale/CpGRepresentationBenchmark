# ruff: noqa: C408
"""EWAS-v2 design: sets, tiers, overlap rule, shared backgrounds (main + S1), folds, nested-CV plan, draw table.

Embedding-free by construction: inputs are the set .npy files, the universe registry, the covariates table (probe type,
island class, GC, CpG density, TSS distance) and nothing else. Deterministic: same inputs -> byte-identical artifacts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 17
N_BOOT = 1000
N_FOLDS = 5
CHROMS = tuple(f"chr{i}" for i in range(1, 23))
SUPPORT_MAIN, SUPPORT_EXPLORATORY = 200, 100
CONTAINMENT_THRESHOLD = 0.5
S1_CALIPER = 0.25
S1_COLUMNS = ("gc_content", "cpg_density_hg38", "tss_dist")   # tss_dist is already log10-scale in the covariates table
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
MAX_ITER = 10000

LOCI = "outputs/encode_atlas_v1/loci.parquet"
COVARIATES = "data/derived/bioval_v2/covariates/universe_covariates.parquet"
KNOWN_SETS_DIR = "data/bio_annotations/known_sets"
PINS = "configs/exp5_candidate_completion/exp5_pins.json"
EXP5_TIERS = "paper/registry/exp5_support_tiers_v2.csv"
DESIGN_DIR = "data/derived/ewas_v2_v1"
MANIFEST = "configs/ewas_v2/manifest_v1.json"

# task id -> family (aging family = age EWAS + clocks; decision 4)
TASK_FAMILY = {
    "ewas_catalog_smoking": "exposure", "ewas_catalog_bmi": "exposure",
    "ewas_catalog_age": "aging", "horvath_clock": "aging", "phenoage_clock": "aging", "hannum_clock": "aging",
    "ewas_atlas_colorectal_cancer": "disease", "ewas_atlas_prostate_cancer": "disease",
    "ewas_catalog_breast_cancer": "disease", "ewas_catalog_lung_cancer": "disease",
    "ewas_catalog_rheumatoid_arthritis": "disease", "ewas_catalog_sle": "disease",
    "ewas_catalog_type2_diabetes": "disease", "ewas_catalog_coronary_heart_disease": "disease",
    "ewas_atlas_ovarian_cancer": "disease", "ewas_catalog_schizophrenia": "disease",
}
TASKS = tuple(sorted(TASK_FAMILY))


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def support_tier(n_matched_positives: int) -> str:
    """Decision 4 (N only): >=200 main-quality; 100-199 exploratory; <100 insufficient."""
    if n_matched_positives >= SUPPORT_MAIN:
        return "main-quality"
    if n_matched_positives >= SUPPORT_EXPLORATORY:
        return "exploratory"
    return "insufficient"


def containment_flags(sets: dict[str, np.ndarray], threshold: float = CONTAINMENT_THRESHOLD) -> list[dict]:
    """small set is `non_independent` of a LARGER set when >= threshold of its members lie in it (size ties: id order).
    Returns every qualifying (small, large) row; ``partner_of`` resolution is done by ``non_independent_map``."""
    sets = {k: np.unique(np.asarray(v)) for k, v in sets.items()}
    rows = []
    for a in sorted(sets):
        for b in sorted(sets):
            if a == b or (len(sets[a]), a) > (len(sets[b]), b):
                continue
            shared = int(np.intersect1d(sets[a], sets[b]).size)
            c = shared / len(sets[a]) if len(sets[a]) else 0.0
            if c >= threshold:
                rows.append(dict(small=a, large=b, shared=shared, n_small=len(sets[a]), n_large=len(sets[b]),
                                 containment=c, n_set_difference=len(sets[a]) - shared))
    return rows


def non_independent_map(rows: list[dict]) -> dict[str, str]:
    """small set -> partner (the qualifying larger set with the highest containment; ties: larger set, then id)."""
    best: dict[str, tuple] = {}
    for r in rows:
        key = (-r["containment"], -r["n_large"], r["large"])
        if r["small"] not in best or key < best[r["small"]][0]:
            best[r["small"]] = (key, r["large"])
    return {k: v[1] for k, v in best.items()}


def technical_background(positive_idx, chrom, probe_type, excluded, seed: int = SEED, k: int = 1) -> dict:
    """ONE background per task, exact strata chromosome x Infinium type, independent of every embedding.

    positive_idx: universe row indices of the positives; chrom, probe_type per universe row (probe_type 1/2; 0/NaN unknown);
    excluded: bool per universe row, True = never a background candidate (member of ANY panel set).
    Strata sorted by (chrom, type); positives in sorted row order; background rows drawn without replacement with
    default_rng(seed) from the sorted eligible rows. Positives without known probe type are dropped (counted); positives in
    strata with fewer eligible rows than positives are `unmatched` (counted)."""
    pt = np.nan_to_num(np.asarray(probe_type, float), nan=0).astype(int)
    chrom = np.asarray(chrom).astype(str)
    pos = np.sort(np.unique(np.asarray(positive_idx, dtype=np.int64)))
    pos_known = pos[pt[pos] > 0]
    is_pos = np.zeros(len(chrom), bool)
    is_pos[pos] = True
    eligible = ~np.asarray(excluded, bool) & ~is_pos & (pt > 0)
    rng = np.random.default_rng(seed)
    pairs, unmatched = [], 0
    for c, t in sorted({(chrom[i], pt[i]) for i in pos_known}):
        p = pos_known[(chrom[pos_known] == c) & (pt[pos_known] == t)]
        pool = np.flatnonzero(eligible & (chrom == c) & (pt == t))
        take = min(len(p) * k, len(pool))
        neg = rng.choice(pool, take, replace=False) if take else np.empty(0, int)
        n_ok = take // k
        unmatched += len(p) - n_ok
        for j in range(n_ok):
            pairs.append((int(p[j]), [int(x) for x in neg[j * k:(j + 1) * k]]))
    return dict(pairs=pairs, n_dropped_unknown_probe_type=int(len(pos) - len(pos_known)), n_unmatched=int(unmatched),
                n_matched_positives=len(pairs))


def context_matched_background(positive_idx, chrom, probe_type, island, Z, excluded, seed: int = SEED,
                               caliper: float = S1_CALIPER) -> dict:
    """S1: exact chromosome x probe type x island class; nearest neighbour (Euclidean in standardized Z) without replacement
    with Euclidean caliper `caliper` SD; positives of a stratum processed in a seeded random order (default_rng(seed),
    strata sorted); ties broken by smaller row index. Unmatched positives are dropped and counted."""
    from scipy.spatial import cKDTree
    pt = np.nan_to_num(np.asarray(probe_type, float), nan=0).astype(int)
    chrom = np.asarray(chrom).astype(str)
    island = np.asarray(island).astype(str)
    pos = np.sort(np.unique(np.asarray(positive_idx, dtype=np.int64)))
    pos_known = pos[pt[pos] > 0]
    is_pos = np.zeros(len(chrom), bool)
    is_pos[pos] = True
    eligible = ~np.asarray(excluded, bool) & ~is_pos & (pt > 0)
    rng = np.random.default_rng(seed)
    pairs, dists, unmatched = [], [], 0
    for c, t, isl in sorted({(chrom[i], pt[i], island[i]) for i in pos_known}):
        p = pos_known[(chrom[pos_known] == c) & (pt[pos_known] == t) & (island[pos_known] == isl)]
        pool = np.flatnonzero(eligible & (chrom == c) & (pt == t) & (island == isl))
        order = rng.permutation(len(p))
        if len(pool) == 0:
            unmatched += len(p)
            continue
        tree = cKDTree(Z[pool])
        used = np.zeros(len(pool), bool)
        got = {}
        for oi in order:
            q = Z[p[oi]]
            K = 8
            while True:
                kk = min(K, len(pool))
                d, ix = tree.query(q, k=kk, distance_upper_bound=caliper)
                d, ix = np.atleast_1d(d), np.atleast_1d(ix)
                ok = np.isfinite(d) & (ix < len(pool))
                cand = sorted((float(dd), int(pool[ii]), int(ii)) for dd, ii in zip(d[ok], ix[ok], strict=True) if not used[ii])
                if cand:
                    dd, row, ii = cand[0]
                    used[ii] = True
                    got[int(oi)] = (row, dd)
                    break
                if ok.sum() < kk or kk >= len(pool):
                    break
                K *= 4
        for oi in sorted(got):                      # emit in sorted-positive order
            pairs.append((int(p[oi]), [got[oi][0]]))
            dists.append(got[oi][1])
        unmatched += len(p) - len(got)
    srt = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    return dict(pairs=[pairs[i] for i in srt], dist=[dists[i] for i in srt],
                n_dropped_unknown_probe_type=int(len(pos) - len(pos_known)), n_unmatched=int(unmatched),
                n_matched_positives=len(pairs))


def pairs_digest(pairs) -> str:
    h = hashlib.sha256()
    for p, ns in pairs:
        h.update(f"{p}:{','.join(map(str, ns))};".encode())
    return h.hexdigest()


def chrom_to_fold(chroms=CHROMS, n_folds: int = N_FOLDS, seed: int = SEED) -> dict[str, int]:
    from cpg_repr_benchmark.biological_validation_v2.splits import chromosome_blocked_folds
    return chromosome_blocked_folds(list(chroms), n_folds, seed=seed)


def nested_cv_plan(c2f: dict[str, int], grid=C_GRID, max_iter: int = MAX_ITER) -> dict:
    """Explicit outer/inner structure: outer test fold f; train folds = the other folds; inner = leave-one-train-fold-out;
    scaler fit on the outer-train rows only; C by mean inner-fold AUROC (ties -> smaller C); refit on all outer-train rows."""
    folds = sorted(set(c2f.values()))
    outer = []
    for f in folds:
        tr = [g for g in folds if g != f]
        outer.append(dict(test_fold=f, train_folds=tr, test_chroms=sorted(c for c, g in c2f.items() if g == f),
                          inner=[dict(inner_val_fold=g, inner_train_folds=[h for h in tr if h != g]) for g in tr]))
    n_fits = len(folds) * (len(grid) * (len(folds) - 1) + 1)
    return dict(grid_C=list(grid), max_iter=max_iter, penalty="l2", solver="lbfgs", class_weight=None,
                C_selection="mean inner-fold AUROC over leave-one-outer-train-fold-out; ties -> smaller C",
                scaler="StandardScaler (ddof=0) fit on outer-train rows only", n_fits_per_task=n_fits, outer=outer)


def draw_table(names, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, len(names), size=(n_boot, len(names))).astype(np.int64)


# ----------------------------------------------------------------------------------------------------- full build
def load_universe(root: Path) -> pd.DataFrame:
    """Universe in registry order (= store row order), covariates joined by cpg_idx (row order of the two tables differs)."""
    loci = pd.read_parquet(Path(root) / LOCI, columns=["cpg_idx", "chr", "pos"])
    cov = pd.read_parquet(Path(root) / COVARIATES).set_index("cpg_idx")
    u = loci.rename(columns={"chr": "chrom"})
    cv = cov.loc[u["cpg_idx"].to_numpy()]
    if not (cv["chrom"].to_numpy() == u["chrom"].to_numpy()).all() or not (cv["pos"].to_numpy() == u["pos"].to_numpy()).all():
        raise AssertionError("covariates / registry coordinates disagree after joining by cpg_idx")
    for c in ("context", "cpg_density_hg38", "gc_content", "tss_dist"):
        u[c] = cv[c].to_numpy()
    u["probe_type"] = cv["probe_type"].map({"I": 1, "II": 2}).fillna(0).astype(int).to_numpy()
    return u


def load_sets(root: Path, check_pins: bool = True) -> dict[str, np.ndarray]:
    pins = json.loads((Path(root) / PINS).read_text())["known_sets_sha256"]
    out = {}
    for t in TASKS:
        rel = f"{KNOWN_SETS_DIR}/{t}.npy"
        if check_pins and sha256_file(Path(root) / rel) != pins[rel]:
            raise AssertionError(f"set file {rel} differs from the pinned sha256")
        out[t] = np.unique(np.load(Path(root) / rel, allow_pickle=False).astype(np.int64))
    return out


def build_design(root: Path) -> dict:
    """Compute every design object in memory (no I/O besides reading inputs). Returns dict of frames/arrays/dicts."""
    root = Path(root)
    u = load_universe(root)
    sets = load_sets(root)
    n = len(u)
    from cpg_repr_benchmark.data.coordinates import encode_many
    cpg = u["cpg_idx"].to_numpy()                      # store id (/cpg_idx of every representation store)
    canonical = np.asarray(encode_many(u["chrom"], u["pos"]), dtype=np.int64)   # id space of the known-set .npy files
    chrom = u["chrom"].to_numpy()
    pt = u["probe_type"].to_numpy()
    rows_of = {}
    for t, ids in sets.items():
        rows_of[t] = np.flatnonzero(np.isin(canonical, ids))
    excluded = np.zeros(n, bool)
    for r in rows_of.values():
        excluded[r] = True
    uni_sets = {t: canonical[r] for t, r in rows_of.items()}
    ov = containment_flags(uni_sets)
    ni = non_independent_map(ov)
    c2f = chrom_to_fold(CHROMS)
    fold_of_row = np.array([c2f[c] for c in chrom], dtype=np.int64)
    # S1 covariates (standardized once over probe-known universe rows)
    X = u[list(S1_COLUMNS)].to_numpy(float)
    known = pt > 0
    mu, sd = X[known].mean(0), X[known].std(0, ddof=0)
    Z = (X - mu) / sd
    island = u["context"].to_numpy()
    main_pairs, s1_pairs, member_rows, summary = [], [], [], []
    for t in TASKS:
        pos = rows_of[t]
        bg = technical_background(pos, chrom, pt, excluded, SEED)
        s1 = context_matched_background(pos, chrom, pt, island, Z, excluded, SEED)
        matched = {p for p, _ in bg["pairs"]}
        s1m = {p for p, _ in s1["pairs"]}
        for r in pos:
            st = "unknown_probe_type" if pt[r] == 0 else ("matched" if r in matched else "unmatched")
            member_rows.append((t, int(cpg[r]), int(r), int(pt[r]), st, "matched" if r in s1m else
                                ("unknown_probe_type" if pt[r] == 0 else "unmatched")))
        for i, (p, (q,)) in enumerate(bg["pairs"]):
            main_pairs.append((t, i, p, int(cpg[p]), q, int(cpg[q]), chrom[p], int(pt[p]), int(fold_of_row[p])))
        for i, ((p, (q,)), d) in enumerate(zip(s1["pairs"], s1["dist"], strict=True)):
            s1_pairs.append((t, i, p, int(cpg[p]), q, int(cpg[q]), chrom[p], int(pt[p]), int(fold_of_row[p]), island[p], repr(float(d))))
        summary.append(dict(task=t, family=TASK_FAMILY[t], n_set_ids=len(sets[t]), n_universe=len(pos),
                            n_not_in_universe=int(len(sets[t]) - len(pos)),
                            n_dropped_unknown_probe_type=bg["n_dropped_unknown_probe_type"], n_unmatched=bg["n_unmatched"],
                            n_matched_pairs=bg["n_matched_positives"], tier=support_tier(bg["n_matched_positives"]),
                            s1_n_unmatched=s1["n_unmatched"], s1_n_matched_pairs=s1["n_matched_positives"],
                            s1_tier=support_tier(s1["n_matched_positives"]), non_independent_of=ni.get(t, ""),
                            pairs_sha256=pairs_digest(bg["pairs"]), s1_pairs_sha256=pairs_digest(s1["pairs"])))
    cols = ["task", "pair_id", "pos_row", "pos_cpg_idx", "neg_row", "neg_cpg_idx", "chrom", "probe_type", "fold"]
    return dict(
        universe=u, n_universe=n, n_excluded_union=int(excluded.sum()), set_rows=rows_of,
        sets_membership=pd.DataFrame(member_rows, columns=["task", "cpg_idx", "row", "probe_type", "main_status", "s1_status"]),
        pairs_main=pd.DataFrame(main_pairs, columns=cols),
        pairs_s1=pd.DataFrame(s1_pairs, columns=cols + ["island", "dist"]),
        overlap=pd.DataFrame(ov), non_independent=ni, summary=pd.DataFrame(summary), chrom_to_fold=c2f,
        plan=nested_cv_plan(c2f), blocks=sorted(CHROMS), draw=draw_table(sorted(CHROMS)),
        s1_standardization=dict(columns=list(S1_COLUMNS), mean=[float(x) for x in mu], sd=[float(x) for x in sd]))


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, lineterminator="\n").encode()


def write_design(des: dict, out_dir: Path) -> dict[str, str]:
    """Write artifacts to out_dir; returns {filename: sha256}. Byte-deterministic (no timestamps, fixed float repr)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "sets_membership.csv": _csv_bytes(des["sets_membership"]),
        "pairs_main.csv": _csv_bytes(des["pairs_main"]),
        "pairs_s1.csv": _csv_bytes(des["pairs_s1"]),
        "overlap.csv": _csv_bytes(des["overlap"]),
        "design_summary.csv": _csv_bytes(des["summary"]),
        "chrom_to_fold.json": json.dumps(dict(sorted(des["chrom_to_fold"].items())), indent=1).encode(),
        "nested_cv_plan.json": json.dumps(des["plan"], indent=1, sort_keys=True).encode(),
        "s1_standardization.json": json.dumps(des["s1_standardization"], indent=1, sort_keys=True).encode(),
    }
    for name, b in files.items():
        (out_dir / name).write_bytes(b)
    np.save(out_dir / "draw_table.npy", des["draw"])
    (out_dir / "draw_table_blocks.json").write_text(json.dumps(des["blocks"]))
    return {p.name: sha256_file(p) for p in sorted(out_dir.iterdir()) if p.is_file()}


def load_pairs(design_dir: Path, task: str, design: str = "main") -> pd.DataFrame:
    f = "pairs_main.csv" if design == "main" else "pairs_s1.csv"
    df = pd.read_csv(Path(design_dir) / f)
    return df[df["task"] == task].sort_values("pair_id").reset_index(drop=True)
