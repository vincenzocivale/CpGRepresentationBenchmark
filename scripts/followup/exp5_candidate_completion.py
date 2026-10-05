"""exp5_candidate_completion: add ``regulatory_histone_dnase_v1`` to the Exp5 matched EWAS/clock probe.

STATUS: SCAFFOLD (decisions of 2026-10-05). NOT RUN. Result status when run:
"discovery-campaign completion (exp5 protocol, not preregistered main evidence)".

What it does (see docs/EXP5_CANDIDATE_COMPLETION_PLAN.md)
  * re-uses the Exp5 internals READ-ONLY (``cpg_repr_benchmark.encode_atlas.ewas_robust``: ``_stage1``, ``match_pairs``,
    ``pooled_oof``, ``fast_auc``); nothing in the repo is edited. The module-level ``REPRESENTATIONS`` dict is
    swapped at runtime in THIS process only.
  * the candidate gets EXACTLY the Exp5 pairs (same strata/draw order/seeds 17001..17020), folds (GroupKFold(5) by
    chromosome on the sorted rows), classifier (StandardScaler fold-wise + LogisticRegression(C=1, max_iter=2000)) and
    seed-mixture bootstrap (same RNG stream: rng_seed = 20260929 + k, same draws, same within-chromosome pair resampling).
  * comparator arms are NOT re-trained: their stored OOF scores are read from the Exp5 parquet (hash-pinned) and put in
    the SAME paired bootstrap. The candidate's pairs/folds are verified identical to the stored ones, row by row.
  * metrics: AUROC (primary), matched-pair concordance (co-primary, ties = 0.5), AUPRC (sensitivity, point estimate only).

Subcommands
  gate        verify every pre-run condition; writes nothing; exit 0 only if all pass.
  regression  re-run comparator arms through the SAME code on a small deterministic subset and compare with the stored
              Exp5 OOF scores (pairs/folds exact; scores atol). Writes only to --out (default a scratch dir).
  stage2-selfcheck  recompute the stored arms' AUROC CIs / deltas from the stored OOF parquet with the extended
              bootstrap and compare to summary_auc_delta.csv (no training; bootstrap only).
  run         gate + candidate stage 1 + merge + extended stage 2 + outputs under outputs/exp5_candidate_completion/.

Nothing here is executed at import time. Heavy imports are lazy so the pure functions are unit-testable on synthetic data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP5_DIR = ROOT / "outputs" / "encode_atlas_v1" / "followup" / "exp5"
PINS = ROOT / "configs" / "exp5_candidate_completion" / "exp5_pins.json"
FROZEN = ROOT / "configs" / "frozen" / "regulatory_histone_dnase_v1.json"
REGISTRATION_DOC = "docs/EXP5_CANDIDATE_COMPLETION_REGISTRATION.md"   # to be written + committed by the user BEFORE the run
OUT_ROOT = ROOT / "outputs" / "exp5_candidate_completion"
FORBIDDEN = "loyfer_failure_audit"

CANDIDATE = "regulatory_histone_dnase_v1"
COMPARATORS = ["cpgpt_large_native512", "deepcpg_hcc_native128", "functional_256"]
COMPARISONS = [(CANDIDATE, c) for c in COMPARATORS]       # delta = candidate - comparator
BASE_SEED, N_SEEDS, BOOT = 17001, 20, 10000
STAGE2_RNG_BASE = 20260929                                # exp5: rng_seed = 20260929 + k, k = index in sorted (variant,set)
TIERS = ((200, "main-quality"), (100, "exploratory"), (0, "insufficient"))
REGRESSION_DEFAULT = dict(variant="base", sets=["ewas_catalog_sle", "hannum_clock"], seeds=[17001, 17002, 17003])


def guard(p) -> None:
    if FORBIDDEN in str(p):
        raise RuntimeError(f"refusing to touch {p}: excluded audit directory")


def sha256(path) -> str:
    guard(path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def support_tier(n_matched_positives: float) -> str:
    """Decision 4 (fixed on N only, before any candidate result)."""
    for thr, name in TIERS:
        if n_matched_positives >= thr:
            return name
    return "insufficient"


# ------------------------------------------------------------------------------------------------ pure metric code
def matched_pair_concordance(pos: np.ndarray, neg: np.ndarray) -> float:
    """Fraction of matched pairs with score(EWAS CpG) > score(matched background); ties count 0.5.

    ``pos[i]`` and ``neg[i]`` are the OOF scores of the two members of pair i (same matching draw).
    """
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / len(pos))


def auroc(pos, neg) -> float:
    from scipy.stats import rankdata
    n = len(pos)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n].sum() - n * (n + 1) / 2) / (n * n))


def auprc(pos, neg) -> float:
    from sklearn.metrics import average_precision_score
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(average_precision_score(y, np.r_[pos, neg]))


def per_seed_from_oof(df: pd.DataFrame, arms: list[str]) -> dict:
    """{(variant,set): {seed: {'chrom': per-pair chromosome, arm: (pos_scores, neg_scores) ordered by pair_id}}}.

    Same ordering convention as ``ewas_robust._stage1`` (pairs ordered by pair_id; positive/negative arrays aligned).
    """
    out: dict = {}
    for (v, s, seed), g in df[df.representation.isin(arms)].groupby(["variant", "set", "match_seed"], observed=True, sort=True):
        d: dict = {}
        for arm, ga in g.groupby("representation", observed=True):
            y = ga.y_true.to_numpy()
            pid = ga.pair_id.to_numpy()
            ip = np.flatnonzero(y)[np.argsort(pid[y], kind="stable")]
            ineg = np.flatnonzero(~y)[np.argsort(pid[~y], kind="stable")]
            sc = ga.oof_score.to_numpy()
            d[arm] = (sc[ip].astype(np.float64), sc[ineg].astype(np.float64))
            if "chrom" not in d:
                d["chrom"] = ga.chromosome.astype(str).to_numpy()[ip]
        out.setdefault((str(v), str(s)), {})[int(seed)] = d
    return out


def stage2_extended(variant, name, per_seed, B, rng_seed, arms, comparisons, with_auprc=True):
    """Seed-mixture bootstrap IDENTICAL in RNG consumption to ``ewas_robust._stage2`` (same draws, same within-chromosome
    pair resampling), but for any arm list and with a second paired statistic (matched-pair concordance).

    Rows: kind in {auc, concordance, auprc} per arm (auprc: point estimate over seeds, no CI) and kind
    {delta_auc, delta_concordance} per comparison (delta = first - second).
    """
    seeds = sorted(per_seed)
    groups = {s: [np.flatnonzero(per_seed[s]["chrom"] == c) for c in np.unique(per_seed[s]["chrom"])] for s in seeds}
    stat = {"auc": auroc, "concordance": matched_pair_concordance}
    seed_val = {m: {a: np.array([f(*per_seed[s][a]) for s in seeds]) for a in arms} for m, f in stat.items()}
    rng = np.random.default_rng(rng_seed)
    draws = rng.integers(0, len(seeds), B)
    boot = {m: {a: np.empty(B) for a in arms} for m in stat}
    for b in range(B):
        s = seeds[draws[b]]
        ix = np.concatenate([rng.choice(gr, len(gr)) for gr in groups[s]])
        for a in arms:
            pos, neg = per_seed[s][a]
            pp, nn = pos[ix], neg[ix]
            boot["auc"][a][b] = auroc(pp, nn)
            boot["concordance"][a][b] = matched_pair_concordance(pp, nn)
    rows = []
    for m in stat:
        for a in arms:
            lo, hi = np.percentile(boot[m][a], [2.5, 97.5])
            v = seed_val[m][a]
            rows.append(dict(variant=variant, set=name, kind=m, representation=a, comparison="", estimate=v.mean(),
                             between_seed_sd=v.std(ddof=1), ci_lo=lo, ci_hi=hi, seed_min=v.min(), seed_max=v.max()))
    if with_auprc:
        for a in arms:
            v = np.array([auprc(*per_seed[s][a]) for s in seeds])
            rows.append(dict(variant=variant, set=name, kind="auprc_sensitivity", representation=a, comparison="",
                             estimate=v.mean(), between_seed_sd=v.std(ddof=1), seed_min=v.min(), seed_max=v.max()))
    for m in stat:
        for a, b_ in comparisons:
            dseed = seed_val[m][a] - seed_val[m][b_]
            d = boot[m][a] - boot[m][b_]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = min(1.0, 2 * (min((d <= 0).mean(), (d >= 0).mean()) + 1 / (len(d) + 1)))   # same formula as exp5
            rows.append(dict(variant=variant, set=name, kind=f"delta_{m}", representation=a, comparison=f"{a} - {b_}",
                             estimate=dseed.mean(), between_seed_sd=dseed.std(ddof=1), ci_lo=lo, ci_hi=hi,
                             seed_min=dseed.min(), seed_max=dseed.max(), p_boot=p,
                             n_seeds_delta_positive=int((dseed > 0).sum()), n_seeds=len(seeds)))
    return rows


class LegacyGroupKFold:
    """Original (scikit-learn <= 1.5) GroupKFold assignment, used by Exp5 to produce the stored folds.

    Groups are visited from the largest to the smallest (``np.argsort(counts)[::-1]``, default quicksort tie-break as in the
    legacy sklearn code) and each goes to the currently lightest fold (``np.argmin``, first on ties). scikit-learn 1.9.1
    changed this assignment (AMENDMENT 1 of the registration). Deterministic; no randomness.
    """

    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def fold_of(self, groups) -> np.ndarray:
        _, inv = np.unique(np.asarray(groups), return_inverse=True)
        cnt = np.bincount(inv)
        if self.n_splits > len(cnt):
            raise ValueError("more splits than groups")
        load = np.zeros(self.n_splits)
        asg = np.zeros(len(cnt), dtype=int)
        for gi in np.argsort(cnt)[::-1]:
            k = int(np.argmin(load))
            load[k] += cnt[gi]
            asg[gi] = k
        return asg[inv]

    def split(self, X, y=None, groups=None):
        f = self.fold_of(groups)
        for k in range(self.n_splits):
            yield np.flatnonzero(f != k), np.flatnonzero(f == k)


def check_stored_folds(stored: pd.DataFrame) -> list:
    """Read-only: the legacy shim must reproduce the stored ``fold`` column row by row for every (variant,set,seed,arm) cell."""
    bad = []
    for key, g in stored.groupby(["variant", "set", "match_seed", "representation"], observed=True, sort=True):
        f = LegacyGroupKFold(5).fold_of(g.chromosome.astype(str).to_numpy())
        if not np.array_equal(f, g.fold.to_numpy()):
            bad.append(key)
    return bad


def regression_cell_verdict(stored: pd.DataFrame, new: pd.DataFrame, bad_cols: list, max_dauc: float, min_rho: float):
    """Amendment 2 criterion for one regression cell: pairs/folds identical AND |dAUROC| <= max_dauc AND Spearman >= min_rho.

    ``stored``/``new``: one (variant,set,seed,arm) each, same row order. Absolute score difference is informative only.
    """
    from scipy.stats import spearmanr
    y = stored.y_true.to_numpy()
    so, sn = stored.oof_score.to_numpy().astype(np.float64), new.oof_score.to_numpy().astype(np.float64)
    dauc = abs(auroc(so[y], so[~y]) - auroc(sn[y], sn[~y]))
    rho = float(spearmanr(so, sn)[0]) if not np.array_equal(so, sn) else 1.0
    ok = (not bad_cols) and dauc <= max_dauc and rho >= min_rho
    return bool(ok), dict(abs_dauroc=float(dauc), spearman=rho)


def stage2_k_index(evaluable: pd.DataFrame) -> dict:
    """exp5 used rng_seed = 20260929 + k with k = position in sorted((variant,set)) of the EVALUABLE entries."""
    keys = sorted((r.variant, r.set) for r in evaluable.itertuples() if r.status == "EVALUABLE")
    return {k: i for i, k in enumerate(keys)}


def verify_same_pairs(stored: dict, new: dict) -> list[str]:
    """Row-level identity of pairs/folds between the stored comparator OOF table and a fresh stage-1 frame.

    ``stored``/``new``: DataFrames of ONE (variant,set,seed,representation) with columns cpg_idx, pair_id, y_true, fold,
    chromosome (rows in the same sorted order).
    """
    bad = []
    for col in ("cpg_idx", "pair_id", "y_true", "fold"):
        if not np.array_equal(stored[col].to_numpy(), new[col].to_numpy()):
            bad.append(col)
    if not np.array_equal(stored["chromosome"].astype(str).to_numpy(), new["chromosome"].astype(str).to_numpy()):
        bad.append("chromosome")
    return bad


def bh_by_comparison(res: pd.DataFrame) -> pd.DataFrame:
    """BH across evaluable sets within variant x comparison (as exp5), separately for the two delta metrics."""
    from scipy.stats import false_discovery_control
    res = res.copy()
    res["q_bh"] = np.nan
    for _, idx in res[res.kind.str.startswith("delta_")].groupby(["variant", "comparison", "kind"]).groups.items():
        res.loc[idx, "q_bh"] = false_discovery_control(res.loc[idx, "p_boot"].to_numpy())
    return res


# ------------------------------------------------------------------------------------------------ gate
def _git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def check_gate(out: Path, store: Path | None = None, require_registration: bool = True) -> tuple[list[str], dict]:
    """Return (failures, evidence). Never writes."""
    fail, ev = [], {}
    guard(out)
    pins = json.load(open(PINS))
    fz = json.load(open(FROZEN))
    ev["frozen_manifest_sha256"] = sha256(FROZEN)
    store = store or ROOT / fz["path_convention"]["store_h5"]
    if fz.get("status") != "candidate_frozen":
        fail.append(f"frozen manifest status is {fz.get('status')!r}")
    ev["store_sha256"] = sha256(store)
    if ev["store_sha256"] != fz["file_sha256"]["store_h5"]:
        fail.append("store sha256 differs from the frozen manifest")
    if fz["file_sha256"]["store_h5"] != pins["candidate"]["store_h5_sha256"]:
        fail.append("frozen manifest store sha differs from the pins file")
    for rel, want in pins["exp5_files_sha256"].items():
        got = sha256(ROOT / rel)
        ev[rel] = got
        if got != want:
            fail.append(f"exp5 file changed: {rel}")
    for rel, want in pins["known_sets_sha256"].items():
        if sha256(ROOT / rel) != want:
            fail.append(f"known set changed: {rel}")
    try:
        out_r = out.resolve()
        if str(out_r).startswith(str(EXP5_DIR.resolve())) or "followup/exp5" in str(out_r):
            fail.append("out dir is inside the exp5 results directory")
        if not str(out_r).startswith(str(OUT_ROOT.resolve())) and out_r.name != "scratch":
            fail.append("out dir must be under outputs/exp5_candidate_completion/")
        if out_r.exists() and any(out_r.iterdir()):
            fail.append("out dir exists and is not empty")
    except OSError as e:
        fail.append(f"out dir check failed: {e}")
    if require_registration:
        try:
            _git("ls-files", "--error-unmatch", REGISTRATION_DOC)
            if _git("status", "--porcelain", "--", REGISTRATION_DOC):
                fail.append("registration doc has uncommitted changes")
        except subprocess.CalledProcessError:
            fail.append(f"registration doc {REGISTRATION_DOC} is not committed")
        dirty = _git("status", "--porcelain", "--", "src", "scripts/followup/exp5_candidate_completion.py",
                     "configs/exp5_candidate_completion", "configs/frozen")
        if dirty:
            fail.append("working tree dirty in src/, this script, its configs or configs/frozen")
    try:    # fold fidelity: the legacy shim must reproduce the stored exp5 folds in every cell (read-only)
        stored_f = pd.read_parquet(EXP5_DIR / "oof_predictions.parquet",
                                   columns=["variant", "set", "match_seed", "representation", "chromosome", "fold"])
        badf = check_stored_folds(stored_f)
        ev["stored_fold_cells_mismatching_shim"] = len(badf)
        if badf:
            fail.append(f"legacy GroupKFold shim does not reproduce stored folds in {len(badf)} cells")
    except Exception as e:  # noqa: BLE001
        fail.append(f"stored-fold check failed: {e}")
    vf = OUT_ROOT / "regression" / "VERDICT.json"      # Amendment 2: comparator regression must have passed
    try:
        vd = json.load(open(vf))
        ev["regression_verdict"] = vd.get("verdict")
        if vd.get("verdict") != "PASS":
            fail.append("comparator regression verdict is not PASS")
    except (OSError, ValueError):
        fail.append("regression VERDICT.json missing or unreadable")
    ev["git_head"] = _git("rev-parse", "HEAD")
    ev["exp5_dir_mtimes"] = {p.name: p.stat().st_mtime for p in sorted(EXP5_DIR.glob("*"))}
    return fail, ev


# ------------------------------------------------------------------------------------------------ real-data plumbing
def _setup(arms: dict, sets: list[str] | None):
    """Mirror ``ewas_robust.run`` global set-up for the given arms (read-only on every repo file)."""
    import h5py
    import yaml

    from cpg_repr_benchmark.data.coordinates import encode_many
    from cpg_repr_benchmark.encode_atlas import ewas_robust as E

    cfg = yaml.safe_load(open(ROOT / "configs/encode_atlas.yaml"))
    out = ROOT / cfg.get("output", "outputs/encode_atlas_v1")
    registry = pd.read_parquet(out / "loci.parquet")
    canonical = encode_many(registry.chr, registry.pos)
    with h5py.File(ROOT / cfg["features"]) as h:
        dense = h["dense"][:].astype(np.float32)
    strata = E._strata(registry, dense)
    probe_df = pd.read_parquet(EXP5_DIR / "probe_type_annotation.parquet")        # stored codes (hash-pinned)
    assert np.array_equal(probe_df.cpg_idx.to_numpy(), registry.cpg_idx.to_numpy())
    probe = probe_df.probe_type.to_numpy().astype(np.int8)
    paths = sorted((ROOT / "data/bio_annotations/known_sets").glob("*.npy"))
    if sets:
        paths = [p for p in paths if p.stem in sets]
    labels = {p.stem: np.isin(canonical, np.load(p, allow_pickle=False)) for p in paths}
    E.REPRESENTATIONS = dict(arms)            # process-local swap; the module file is untouched
    E.GroupKFold = LegacyGroupKFold           # restore the original Exp5 fold assignment (sklearn 1.9.1 changed GroupKFold)
    E._G.update(labels=labels, probe=probe, chr=registry.chr.to_numpy(), cpg_idx=registry.cpg_idx.to_numpy(),
                index_base=E._index_strata(strata), index_probe=E._index_strata(strata + ":" + probe.astype(str)),
                emb=E._load_embeddings(out, registry))
    return E, cfg, registry


def cmd_gate(a) -> int:
    fail, ev = check_gate(Path(a.out), require_registration=not a.skip_registration_check)
    print(json.dumps({k: v for k, v in ev.items() if k in ("store_sha256", "git_head", "frozen_manifest_sha256")}, indent=1))
    for f in fail:
        print("GATE FAIL:", f)
    print("GATE", "PASS" if not fail else "FAIL")
    return 1 if fail else 0


def cmd_regression(a) -> int:
    """Stage-1 regression: comparator arms re-run through the same code vs stored OOF (small deterministic subset)."""
    import multiprocessing as mp
    arms_all = json.load(open(EXP5_DIR / "protocol.json"))["representations"]
    arms = {k: v for k, v in arms_all.items() if k in COMPARATORS}
    E, cfg, _ = _setup(arms, a.sets)
    stored = pd.read_parquet(EXP5_DIR / "oof_predictions.parquet")
    ctx = mp.get_context("fork")
    tasks = [(a.variant, s, sd) for s in a.sets for sd in a.seeds]
    with ctx.Pool(a.workers) as pool:
        res = pool.map(E._stage1, tasks, chunksize=1)
    worst, worst_dauc, min_rho, ok = 0.0, 0.0, 1.0, True
    verdict_rows = []
    for (info, long, _), (v, s, sd) in zip(res, tasks):
        for arm in arms:
            st = stored[(stored.variant == v) & (stored.set == s) & (stored.match_seed == sd) & (stored.representation == arm)]
            nw = long[long.representation == arm]
            st = st.reset_index(drop=True)
            nw = nw.reset_index(drop=True)
            bad = verify_same_pairs(st, nw)
            diff = float(np.max(np.abs(st.oof_score.to_numpy() - nw.oof_score.to_numpy())))
            cell_ok, m = regression_cell_verdict(st, nw, bad, a.max_dauc, a.min_rho)
            worst = max(worst, diff)
            worst_dauc = max(worst_dauc, m["abs_dauroc"])
            min_rho = min(min_rho, m["spearman"])
            ok &= cell_ok
            verdict_rows.append(dict(variant=v, set=s, seed=sd, arm=arm, pairs_folds_identical=not bad, max_abs_dscore_informative=diff,
                                     **m, cell_pass=cell_ok))
            print(f"{v}/{s}/{sd}/{arm}: pairs_folds_identical={not bad} |dAUROC|={m['abs_dauroc']:.2e} spearman={m['spearman']:.6f} "
                  f"max|dscore|(info)={diff:.2e} {'ok' if cell_ok else 'FAIL'}")
    verdict = "PASS" if ok else "FAIL"
    print("REGRESSION", verdict, f"max|dAUROC|={worst_dauc:.2e} (<= {a.max_dauc}) min_spearman={min_rho:.6f} (>= {a.min_rho}) "
          f"max|dscore|={worst:.2e} (informative; Amendment 2)")
    if a.verdict_out:
        vo = Path(a.verdict_out)
        guard(vo)
        vo.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(verdict_rows).to_csv(vo / "regression_cells.csv", index=False)
        (vo / "VERDICT.json").write_text(json.dumps(dict(
            verdict=verdict, criterion="pairs/folds identical AND |dAUROC|<=max_dauc AND spearman>=min_rho in every cell (Amendment 2)",
            max_dauc=a.max_dauc, min_rho=a.min_rho, observed_max_dauroc=worst_dauc, observed_min_spearman=min_rho,
            observed_max_abs_dscore_informative=worst, n_cells=len(verdict_rows), n_cells_failed=int(sum(not r["cell_pass"] for r in verdict_rows))), indent=1))
    return 0 if ok else 1


def cmd_stage2_selfcheck(a) -> int:
    """Recompute stored arms' AUROC CIs/deltas with stage2_extended and compare to summary_auc_delta.csv (no training)."""
    stored = pd.read_parquet(EXP5_DIR / "oof_predictions.parquet")
    ev = pd.read_csv(EXP5_DIR / "evaluable_sets.csv")
    kidx = stage2_k_index(ev)
    summ = pd.read_csv(EXP5_DIR / "summary_auc_delta.csv")
    arms = ["functional_256", "cpgpt_large_native512", "deepcpg_hcc_native128"]
    comps = [("functional_256", "cpgpt_large_native512"), ("functional_256", "deepcpg_hcc_native128")]
    per = per_seed_from_oof(stored, arms)
    worst = 0.0
    for key in a.keys:
        v, s = key.split("/")
        # NOTE exp5 resampled with the six-arm per_seed dict; RNG consumption does not depend on the arm list.
        rows = stage2_extended(v, s, per[(v, s)], a.boot, STAGE2_RNG_BASE + kidx[(v, s)], arms, comps, with_auprc=False)
        for r in rows:
            if r["kind"] not in ("auc", "delta_auc"):
                continue
            m = summ[(summ.variant == v) & (summ.set == s) & (summ.kind == ("auc" if r["kind"] == "auc" else "delta"))
                     & (summ.representation == r["representation"]) & (summ.comparison.fillna("") == r["comparison"])]
            if len(m) == 1 and a.boot == BOOT:
                worst = max(worst, abs(m.iloc[0].ci_lo - r["ci_lo"]), abs(m.iloc[0].ci_hi - r["ci_hi"]),
                            abs(m.iloc[0].estimate - r["estimate"]))
    # tolerance: the stored parquet holds float32 OOF scores while exp5 bootstrapped float64 scores (rank ties / rounding)
    print("STAGE2 SELFCHECK", "PASS" if worst < 2e-3 else "FAIL", f"worst={worst:.2e} (tol 2e-3; meaningful only with --boot {BOOT})")
    return 0 if (a.boot != BOOT or worst < 2e-3) else 1


def cmd_run(a) -> int:
    import multiprocessing as mp
    out = Path(a.out)
    fail, ev = check_gate(out, require_registration=True)
    if fail:
        for f in fail:
            print("GATE FAIL:", f)
        return 1
    pins = json.load(open(PINS))
    snapshot = {rel: sha256(ROOT / rel) for rel in pins["exp5_files_sha256"]}
    fz = json.load(open(FROZEN))
    store = fz["path_convention"]["store_h5"]
    E, cfg, registry = _setup({CANDIDATE: store}, None)
    evset = pd.read_csv(EXP5_DIR / "evaluable_sets.csv")
    todo = sorted((r.variant, r.set) for r in evset.itertuples() if r.status == "EVALUABLE")
    tasks = [(v, s, BASE_SEED + i) for v, s in todo for i in range(N_SEEDS)]
    out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers) as pool:
        stage1 = pool.map(E._stage1, tasks, chunksize=1)
    stored = pd.read_parquet(EXP5_DIR / "oof_predictions.parquet",
                             filters=[("representation", "in", COMPARATORS)])
    # row-level identity of pairs/folds with the stored arms
    cand_long = pd.concat([s[1] for s in stage1])
    bad_all = []
    for (info, long, _), (v, s, sd) in zip(stage1, tasks):
        for comp in COMPARATORS:      # candidate folds/pairs must equal the stored ones in every cell, against every comparator
            st = stored[(stored.variant == v) & (stored.set == s) & (stored.match_seed == sd) & (stored.representation == comp)]
            bad = verify_same_pairs(st.reset_index(drop=True), long.reset_index(drop=True))
            if bad:
                bad_all.append((v, s, sd, comp, bad))
    if bad_all:
        (out / "ABORT_pairs_or_folds_differ.json").write_text(json.dumps(bad_all[:50], indent=1))
        print("ABORT: candidate pairs/folds differ from exp5 for", len(bad_all), "tasks")
        return 2
    cand_long.to_parquet(out / "candidate_oof_predictions.parquet", index=False)
    per_cand: dict = {}
    for (info, _, arr), (v, s, sd) in zip(stage1, tasks):
        per_cand.setdefault((v, s), {})[sd] = arr
    per_cmp = per_seed_from_oof(stored, COMPARATORS)
    arms = [CANDIDATE] + COMPARATORS
    kidx = stage2_k_index(evset)
    jobs = []
    for key in todo:
        merged = {}
        for sd, arr in per_cand[key].items():
            m = dict(per_cmp[key][sd])
            assert np.array_equal(m["chrom"].astype(str), np.asarray(arr["chrom"]).astype(str)), "chromosome per pair differs"
            # same float32 rounding as the stored comparator scores (the stored parquet keeps float32 OOF scores)
            m[CANDIDATE] = tuple(np.asarray(x, np.float32).astype(np.float64) for x in arr[CANDIDATE])
            merged[sd] = m
        jobs.append((key[0], key[1], merged, BOOT, STAGE2_RNG_BASE + kidx[key], arms, COMPARISONS))
    with ctx.Pool(min(a.workers, len(jobs))) as pool:
        rows = [r for part in pool.starmap(stage2_extended, jobs, chunksize=1) for r in part]
    res = bh_by_comparison(pd.DataFrame(rows))
    res["matched_positives"] = res.set.map(dict(zip(evset[evset.variant == "base"].set, evset[evset.variant == "base"].pairs_mean)))
    res["support_tier"] = res.matched_positives.map(support_tier)
    res["status_label"] = "discovery-campaign completion (exp5 protocol, not preregistered main evidence)"
    res.to_csv(out / "summary_candidate_completion.csv", index=False)
    after = {rel: sha256(ROOT / rel) for rel in pins["exp5_files_sha256"]}
    assert after == snapshot, "exp5 results changed during the run"
    manifest = dict(evidence=ev, exp5_snapshot_before=snapshot, exp5_snapshot_after=after, arms=arms, comparisons=COMPARISONS,
                    seeds=[BASE_SEED + i for i in range(N_SEEDS)], boot=BOOT, classifier="StandardScaler(fold-wise)+LogReg(C=1,max_iter=2000)",
                    folds="GroupKFold(5) by chromosome", not_evaluable_sets_skipped=sorted(set(evset[evset.status != "EVALUABLE"].set)),
                    metric_hierarchy="AUROC primary; matched_pair_concordance co-primary; AUPRC sensitivity only",
                    status="discovery-campaign completion (exp5 protocol, not preregistered main evidence)")
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print("done ->", out)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--out", default=str(OUT_ROOT / "run_001"))
    g.add_argument("--skip-registration-check", action="store_true", help="dry gate before the registration doc exists")
    r = sub.add_parser("regression")
    r.add_argument("--variant", default=REGRESSION_DEFAULT["variant"])
    r.add_argument("--sets", nargs="+", default=REGRESSION_DEFAULT["sets"])
    r.add_argument("--seeds", nargs="+", type=int, default=REGRESSION_DEFAULT["seeds"])
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--max-dauc", dest="max_dauc", type=float, default=5e-4, help="Amendment 2: max |dAUROC| per cell")
    r.add_argument("--min-rho", dest="min_rho", type=float, default=0.9999, help="Amendment 2: min Spearman per cell")
    r.add_argument("--verdict-out", dest="verdict_out", default=None)
    s2 = sub.add_parser("stage2-selfcheck")
    s2.add_argument("--keys", nargs="+", default=["base/ewas_catalog_sle", "base/hannum_clock"])
    s2.add_argument("--boot", type=int, default=BOOT)
    u = sub.add_parser("run")
    u.add_argument("--out", default=str(OUT_ROOT / "run_001"))
    u.add_argument("--workers", type=int, default=2)
    a = p.parse_args(argv)
    return {"gate": cmd_gate, "regression": cmd_regression, "run": cmd_run, "stage2-selfcheck": cmd_stage2_selfcheck}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
