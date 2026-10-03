"""Analysis of the 3-seed regulatory confirmation (docs/REGULATORY_CONFIRMATION_PROTOCOL.md).

Validation patients only (asserted from every resolved config and metrics.json), 50% masking, `best.pt`.
Reads per run: resolved_config.yaml, history.json, early_stopping.json, evaluation/seen/mask_0.50/{metrics.json,
predictions.npz}, file timestamps.  The methylation file is opened for the `sample_name` axis only.
Never writes into run directories.  Works with PARTIAL results (only runs with a .done marker are used; the verdict is
then labelled PROVISIONAL) and is idempotent (deterministic outputs, explicit bootstrap seed).

    python scripts/analyze_regulatory_confirm.py [--replicates 2000] [--seed 17]
        [--out-dir outputs/regulatory_confirm_v1/analysis] [--report docs/REGULATORY_CONFIRMATION_RESULTS.md]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cpg_repr_benchmark.encode_atlas.statistics import paired_bootstrap, prediction_table

_spec = importlib.util.spec_from_file_location("_screen", ROOT / "scripts/analyze_regulatory_family_screen.py")
_screen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_screen)  # reuse abs_error_view / replicate_estimates (no side effects at import)
abs_error_view, replicate_estimates = _screen.abs_error_view, _screen.replicate_estimates

RUNS_ROOT = ROOT / "outputs/regulatory_confirm_v1"
CAMPAIGN = ROOT / "outputs/encode_atlas_v1"
ARMS = ["regulatory_histone", "regulatory_histone_dnase", "regulatory_clean"]
SHORT = {"regulatory_histone": "H", "regulatory_histone_dnase": "H+D", "regulatory_clean": "Clean"}
H, HD, CLEAN = ARMS
SEEDS = [17, 42, 97]
CONTRASTS = [(H, HD), (HD, CLEAN), (H, CLEAN)]  # primary rule contrasts first, context last
MASK_SEED = 17001
CONV_WINDOW = 10
CURVE_EPOCHS = (30, 50, 70)  # 1-based epoch counts
# Decision-rule constants (docs/REGULATORY_CONFIRMATION_PROTOCOL.md, rules i-iii; do not change here)
STRONG_REL = -0.01
CLEAN_MIN_REL = -0.005
# NOTE: no numeric 'biologically non-negligible' bar is coded. An earlier 1.5%-of-gap bar was chosen post hoc, never
# preregistered by the user, and has been removed from the decision (gain as % of gap is descriptive only).
NON_NEGLIGIBILITY_NOTE = "judged by the author, not coded"
PAIR_KEYS = ("sample_index", "target_matrix_column", "panel_repeat", "target", "prior_prediction")


# ------------------------------------------------------------------------------------------------ run discovery
def run_dir_from_log(runs_root: Path, arm: str, seed: int) -> Path | None:
    """Run dir of a FINISHED run (.done marker present), from the last RUN_DIR= line of its log; else None."""
    if not (runs_root / "logs" / f"{arm}__seed{seed}.done").exists():
        return None
    log = runs_root / "logs" / f"{arm}__seed{seed}.log"
    lines = [x for x in log.read_text().splitlines() if x.startswith("RUN_DIR=")]
    return Path(lines[-1].split("=", 1)[1]) if lines else None


def discover(runs_root: Path, arms=ARMS, seeds=SEEDS):
    done, missing = {}, []
    for arm in arms:
        for seed in seeds:
            d = run_dir_from_log(runs_root, arm, seed)
            if d is None:
                missing.append((arm, seed))
            else:
                done[(arm, seed)] = d
    return done, missing


# ------------------------------------------------------------------------------------------------ per-run metrics
def assert_validation_only(cfg: dict, metrics: dict, where: str):
    ev = cfg["evaluation"]
    if ev.get("patient_view") != "validation" or ev.get("require_patient_view") != "validation":
        raise PermissionError(f"{where}: resolved config is not validation-only")
    if metrics.get("patient_view") != "validation":
        raise PermissionError(f"{where}: metrics.json patient_view != validation")
    if abs(float(metrics["mask_fraction"]) - 0.5) > 1e-12:
        raise ValueError(f"{where}: evaluation mask fraction is not 0.5")


def convergence_flag(best_epoch: int, epochs_run: int, window: int = CONV_WINDOW) -> bool:
    """'not converged' iff the best (0-based) epoch lies within the last `window` epochs of the run."""
    return best_epoch >= epochs_run - window


def curve_stats(mse: np.ndarray, window: int = CONV_WINDOW) -> dict:
    n = len(mse)
    out = {f"val_mse_epoch{e}": (float(mse[e - 1]) if n >= e else np.nan) for e in CURVE_EPOCHS}
    out["val_mse_final"] = float(mse[-1])
    w = mse[-window:]
    slope = np.polyfit(np.arange(len(w)), w, 1)[0] if len(w) >= 3 else np.nan
    out["slope_last10_pct_per_epoch"] = float(slope / w.mean() * 100) if len(w) >= 3 else np.nan
    return out


def load_run(arm: str, seed: int, d: Path) -> dict:
    cfg = yaml.safe_load((d / "resolved_config.yaml").read_text())
    metrics = json.loads((d / "evaluation/seen/mask_0.50/metrics.json").read_text())
    assert_validation_only(cfg, metrics, f"{arm}/seed{seed}")
    if cfg["training"]["seed"] != seed:
        raise ValueError(f"{arm}/seed{seed}: resolved training.seed = {cfg['training']['seed']}")
    if cfg["evaluation"]["mask_seed"] != MASK_SEED:
        raise ValueError(f"{arm}/seed{seed}: mask_seed != {MASK_SEED}")
    hist = json.loads((d / "history.json").read_text())
    es = json.loads((d / "early_stopping.json").read_text())
    mse = np.array([r["validation_mse"] for r in hist])
    best = int(np.argmin(mse))
    if es["best_epoch"] != best or es["epochs_run"] != len(hist):
        raise ValueError(f"{arm}/seed{seed}: early_stopping.json disagrees with history.json")
    with np.load(d / "evaluation/seen/mask_0.50/predictions.npz") as h:
        prior_mse = float(np.mean((h["prior_prediction"].astype(np.float64) - h["target"].astype(np.float64)) ** 2))
    t0, t1, t2 = (d / "experiment.json").stat().st_mtime, (d / "history.json").stat().st_mtime, \
        (d / "summary.json").stat().st_mtime
    row = {"arm": arm, "seed": seed, "run_dir": d.name, "n_pairs": metrics["n_pairs"], "mse": metrics["mse"],
           "mae": metrics["mae"], "mas_pcc": metrics["mas_pcc"], "mac_pcc": metrics["mac_pcc"],
           "prior_mse": prior_mse, "skill_vs_prior": 1 - metrics["mse"] / prior_mse,
           "best_epoch": best, "last_epoch": len(hist) - 1, "epochs_run": len(hist), "max_epochs": es["max_epochs"],
           "stopped_early": bool(es["stopped_early"]), "stopped_epoch": es["stopped_epoch"],
           "not_converged": convergence_flag(best, len(hist)),
           **curve_stats(mse), "train_seconds": t1 - t0, "train_plus_eval_seconds": t2 - t0}
    curve = pd.DataFrame([{"arm": arm, "seed": seed, **{k: v for k, v in r.items()
                                                       if k in ("epoch", "train_mse", "validation_mse", "validation_mae",
                                                                "validation_mas_pcc", "validation_mac_pcc")}}
                          for r in hist])
    return {"row": row, "curve": curve, "outputs_path": d / "evaluation/seen/mask_0.50/predictions.npz"}


def last10_gaps(curves: pd.DataFrame, seed: int, ref: str, alt: str, window: int = CONV_WINDOW) -> dict:
    """Mean over the last `window` epochs common to both runs of (alt - ref) validation MSE / MAE (abs and % of ref)."""
    a = curves[(curves.arm == ref) & (curves.seed == seed)].set_index("epoch")
    b = curves[(curves.arm == alt) & (curves.seed == seed)].set_index("epoch")
    common = a.index.intersection(b.index)[-window:]
    out = {}
    for m in ("validation_mse", "validation_mae"):
        gap = b.loc[common, m] - a.loc[common, m]
        out[m] = {"gap": float(gap.mean()), "rel": float(gap.mean() / a.loc[common, m].mean()),
                  "sign_stable": bool((gap < 0).all() or (gap > 0).all()), "n_epochs": len(common)}
    return out


# ------------------------------------------------------------------------------------------------ paired contrasts
def boot(ref, alt, seed, reps):
    r = paired_bootstrap(ref, alt, seed=seed, replicates=reps)
    est = replicate_estimates(ref, alt, seed, reps)
    assert abs(np.quantile(est, .025) - r["ci95"][0]) < 1e-12 and abs(np.quantile(est, .975) - r["ci95"][1]) < 1e-12
    base = r["delta_mse"] / r["relative_delta_mse"]
    return {"ref_value": base, "alt_value": base + r["delta_mse"], "delta": r["delta_mse"],
            "relative": r["relative_delta_mse"], "ci_lo": r["ci95"][0], "ci_hi": r["ci95"][1],
            "rel_ci_lo": r["relative_ci95"][0], "rel_ci_hi": r["relative_ci95"][1],
            "frac_replicates_favoring_alt": float(np.mean(est < 0)), "n_pairs": r["n_pairs"],
            "n_patients": r["n_patients"], "n_blocks": r["n_blocks"]}


def pairing_check(outs: dict[str, dict], ref_arm: str) -> dict:
    res = {}
    for arm, o in outs.items():
        if arm != ref_arm:
            res[arm] = {k: bool(np.array_equal(o[k], outs[ref_arm][k])) for k in PAIR_KEYS}
    res["all_identical"] = all(v for a in res.values() if isinstance(a, dict) for v in a.values())
    return res


# ------------------------------------------------------------------------------------------------ decision rule
def _sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def apply_decision_rule(seeds_i: list[dict], seeds_iii: list[dict], *, n_runs_done: int, n_runs_total: int = 9,
                        any_not_converged: bool = False, point_only: bool = False) -> dict:
    """Rules (i)-(iv) of the protocol, MSE/MAE only (PCC metrics are deliberately not an argument).

    seeds_i   : per-seed dicts for Histone -> Histone+DNase, seeds_iii for Histone+DNase -> Clean, each with
                rel_mse, d_mse, d_mae, mse_ci_hi, mse_ci_lo, mae_ci_hi (CI fields may be None when point_only),
                gap_prior (seeds_iii only).
    point_only: sensitivity mode, per-seed CI clauses are not assessed (treated as satisfied).
    """
    ci_ok = (lambda s: True) if point_only else (lambda s: s["mse_ci_hi"] < 0)
    # ---- rule (i)
    ci = {}
    if seeds_i:
        sign_mse = all(s["rel_mse"] < 0 for s in seeds_i)
        sign_mae = all(s["d_mae"] < 0 for s in seeds_i)
        ci3 = all(ci_ok(s) for s in seeds_i)
        mean_rel = float(np.mean([s["rel_mse"] for s in seeds_i]))
        no_gain = all(s["rel_mse"] >= 0 for s in seeds_i) and all(s["d_mae"] >= 0 for s in seeds_i)
        confirmed = sign_mse and sign_mae and ci3
        ci = {"n_seeds": len(seeds_i), "sign_mse_all": sign_mse, "sign_mae_all": sign_mae, "ci_excludes0_mse_all": ci3,
              "mean_rel_mse": mean_rel, "ge_1pct_flag": mean_rel <= STRONG_REL, "PASS": confirmed,
              "label": ("STRONG confirmation" if mean_rel <= STRONG_REL else "SUPPORTED-SMALL (mean > -1%)")
              if confirmed else "NOT CONFIRMED", "no_gain_in_any_seed_or_metric": no_gain}
    # ---- rules (ii)/(iii)
    cl = {}
    if seeds_iii:
        mean_rel = float(np.mean([s["rel_mse"] for s in seeds_iii]))
        sign_mse = all(s["rel_mse"] < 0 for s in seeds_iii)
        sign_mae = all(s["d_mae"] < 0 for s in seeds_iii)
        ci3 = all(ci_ok(s) for s in seeds_iii)
        mae_ci_side = True if point_only else bool(np.mean([s["mae_ci_hi"] for s in seeds_iii]) < 0)
        gains = [-s["d_mse"] / s["gap_prior"] for s in seeds_iii]
        mean_gain = float(np.mean(gains))
        stable = sign_mse and sign_mae and ci3
        small = mean_rel > CLEAN_MIN_REL
        # Statistical clauses of (iii). Biological non-negligibility is NOT coded (author judgement).
        stat_all = (mean_rel <= CLEAN_MIN_REL) and sign_mse and ci3 and sign_mae and mae_ci_side
        cl = {"n_seeds": len(seeds_iii), "mean_rel_mse": mean_rel, "improvement_lt_0p5pct": small,
              "sign_mse_all": sign_mse, "sign_mae_all": sign_mae, "ci_excludes0_mse_all": ci3,
              "seed_mean_mae_ci_hi_below0": mae_ci_side, "stable": stable, "prefer_HD_by_rule_ii": small or not stable,
              "gain_fraction_of_gap_prior_mean": mean_gain, "gain_fractions_per_seed": gains,
              "gain_fraction_is_descriptive_only": True,
              "biologically_non_negligible": NON_NEGLIGIBILITY_NOTE, "STAT_CLAUSES_ALL": stat_all}
    # ---- verdict
    if not seeds_i and not seeds_iii:
        verdict, why = "not evaluable", "no complete contrast available yet"
    elif cl.get("STAT_CLAUSES_ALL") and ci.get("PASS"):
        verdict = "Histone+DNase preferred by parsimony pending author judgement on non-negligibility"
        why = ("rule (i) satisfied; all statistical clauses of (iii) hold (Clean advantage reproducible on MSE), but "
               "biological non-negligibility is not coded: author decision")
    elif cl.get("STAT_CLAUSES_ALL"):
        verdict = "Clean advantage reproducible on MSE; non-negligibility not coded - author decision"
        why = "all statistical clauses of (iii) hold and rule (i) is not satisfied; no automatic promotion"
    elif ci.get("PASS"):
        verdict = "Histone+DNase preferred"
        why = "rule (i) satisfied and Clean not promoted by (ii)/(iii)" if cl else \
            "rule (i) satisfied; Clean vs Histone+DNase not yet evaluable"
    elif ci and ci["no_gain_in_any_seed_or_metric"] and cl and not cl["STAT_CLAUSES_ALL"]:
        verdict, why = "Histone sufficient", "Histone+DNase gives no gain in any seed/metric; Clean not promoted"
    else:
        verdict, why = "inconclusive", "rule (i) neither confirmed nor cleanly rejected (mixed signs or CI including 0)"
    provisional = n_runs_done < n_runs_total or any_not_converged
    reasons = []
    if n_runs_done < n_runs_total:
        reasons.append(f"{n_runs_done}/{n_runs_total} runs done")
    if any_not_converged:
        reasons.append("at least one run's best epoch is in the last 10 epochs (fixed budget reached while still improving: "
                       "not converged)")
    return {"verdict": verdict, "reason": why, "provisional": provisional, "provisional_reasons": reasons,
            "rule_i": ci, "rule_ii_iii": cl, "point_only": point_only,
            "note_pcc": "MAS-PCC / MAC-PCC are reported only; they are not inputs of this function (rule iv)."}


# ------------------------------------------------------------------------------------------------ plot
def plot_curves(curves: pd.DataFrame, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    colours = {H: "#4C78A8", HD: "#E45756", CLEAN: "#54A24B"}
    styles = {17: "-", 42: "--", 97: ":"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for (arm, seed), g in curves.groupby(["arm", "seed"]):
        g = g[g.epoch >= 9]
        ax.plot(g.epoch + 1, g.validation_mse, styles.get(seed, "-"), color=colours[arm], alpha=.9, lw=1.5)
        ax.annotate(f"{SHORT[arm]} s{seed}", (g.epoch.iloc[-1] + 1, g.validation_mse.iloc[-1]), fontsize=7,
                    color=colours[arm], xytext=(3, 0), textcoords="offset points", va="center")
        if arm != H:
            r = curves[(curves.arm == H) & (curves.seed == seed)].set_index("epoch").validation_mse
            gg = g.set_index("epoch").validation_mse
            idx = gg.index.intersection(r.index)
            ax2.plot(idx + 1, (gg[idx] / r[idx] - 1) * 100, styles.get(seed, "-"), color=colours[arm], lw=1.5)
    ax.set(xlabel="epoch", ylabel="validation MSE (50% masking)", title="Validation MSE (epochs >= 10; line style = seed)")
    ax2.axhline(0, color="grey", lw=.8)
    ax2.set(xlabel="epoch", ylabel="MSE relative to Histone, same seed (%)",
            title="Arm gap vs Histone (red = H+D, green = Clean)")
    ax.margins(x=.12)
    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ------------------------------------------------------------------------------------------------ report
def pct(x, d=2, sign=True):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def build_report(res: dict, tables: dict, missing, args, plot_ok: bool) -> str:
    L = []
    v = res["verdict"]
    prov = res["provisional"]
    L.append("# Regulatory confirmation: results\n")
    L.append("GENERATED by `scripts/analyze_regulatory_confirm.py` from `" + str(args.out_dir_rel) + "` (do not edit by hand).")
    L.append("Protocol and decision rule (fixed before runs): `docs/REGULATORY_CONFIRMATION_PROTOCOL.md`. Validation patients only "
             "(asserted from every resolved config and metrics.json: `patient_view=validation`, 50% masking); test patients never read.\n")
    if prov:
        L.append(f"> **PROVISIONAL**: {'; '.join(res['provisional_reasons'])}. Numbers below are preliminary and the verdict "
                 "is not final.\n")
    else:
        L.append("> FINAL: all 9 runs done and no run's best epoch is in the last 10 epochs.\n")
    L.append("**Training regime: FIXED-BUDGET selection protocol** (80 epochs, identical for all arms and seeds), NOT convergence. "
             "Early stopping (patience 10) never triggered in any run; the best epoch lies in the last epochs of the budget, so "
             "curves were still improving (see slope column). Conclusions are about the ranking of arms at equal budget.\n")
    L.append(f"## Verdict{' (PROVISIONAL)' if prov else ''}: **{v}**\n")
    L.append(f"Reason: {res['reason']}.\n")
    L.append("No numeric 'biologically non-negligible' threshold is used anywhere in the verdict: an earlier 1.5%-of-gap bar was "
             "NOT preregistered by the user and has been removed. 'Gain as % of the prior-to-model gap' is reported as a descriptive "
             "number only. Non-negligibility of Clean's advantage is judged by the author, not coded.\n")
    L.append("Recommendation only (rule vi): nothing is promoted to the README, catalog or paper without separate approval.\n")
    L.append("Caveats: the three decoder seeds share mask seed 17001, sample order and validation panels, so per-seed "
             "bootstraps (patient x 1 Mb block, "
             f"{args.replicates} replicates, seed {args.seed}) are NOT independent samples of masking; n = 3 seeds supports "
             "only sign counts and spread, not a formal test across seeds.\n")
    L.append(f"Runs done: {res['n_runs_done']}/9. Missing/incomplete: "
             + (", ".join(f"{SHORT[a]} s{s}" for a, s in missing) or "none") + ".\n")
    ri, rc = res["rule_i"], res["rule_ii_iii"]
    L.append("## Rule application\n")
    if ri:
        L.append(f"**(i) Histone+DNase vs Histone** (seeds used: {ri['n_seeds']}/3): MSE sign favours H+D in all used seeds: "
                 f"{ri['sign_mse_all']}; MAE sign: {ri['sign_mae_all']}; per-seed MSE CI excludes 0: {ri['ci_excludes0_mse_all']}; "
                 f"seed-mean rel. MSE {pct(ri['mean_rel_mse'])}; >=1% flag: {ri['ge_1pct_flag']}. "
                 f"Outcome: **{'PASS' if ri['PASS'] else 'FAIL'}** ({ri['label']}).\n")
    else:
        L.append("**(i)** not evaluable yet (no seed with both Histone and Histone+DNase done).\n")
    if rc:
        L.append(f"**(ii) Parsimony, Clean vs Histone+DNase** (seeds used: {rc['n_seeds']}/3): seed-mean rel. MSE "
                 f"{pct(rc['mean_rel_mse'])} (improvement < 0.5%: {rc['improvement_lt_0p5pct']}); stable (signs MSE+MAE and MSE CI "
                 f"in all used seeds): {rc['stable']} -> prefer H+D by rule (ii): {rc['prefer_HD_by_rule_ii']}.\n")
        L.append(f"**(iii) Promote Clean**: seed-mean rel. MSE <= -0.5%: {not rc['improvement_lt_0p5pct']}; MSE CI excludes 0 with same sign "
                 f"in all used seeds: {rc['ci_excludes0_mse_all'] and rc['sign_mse_all']}; MAE same sign: {rc['sign_mae_all']}; "
                 f"seed-mean MAE CI upper bound < 0: {rc['seed_mean_mae_ci_hi_below0']}; gain as fraction of gap_prior (seed-mean) "
                 f"{pct(rc['gain_fraction_of_gap_prior_mean'], sign=False)} (DESCRIPTIVE ONLY, not a criterion; no numeric bar was "
                 f"preregistered and none is used). Biological non-negligibility: {rc['biologically_non_negligible']}. "
                 f"Statistical clauses of (iii) all hold: **{rc['STAT_CLAUSES_ALL']}**. No automatic promotion of Clean is coded.\n")
    else:
        L.append("**(ii)/(iii)** not evaluable yet (no seed with both Histone+DNase and Clean done).\n")
    L.append("**(iv)** MAS-PCC and MAC-PCC are reported in the tables but are not used by the verdict code.\n")
    sens = res.get("sensitivity")
    if sens:
        L.append("### Sensitivity (non-converged runs): rule applied to last-10-epoch average validation gaps "
                 "(point estimates only; CI clauses not assessed)\n")
        L.append(f"Verdict under sensitivity: **{sens['verdict']}** ({sens['reason']}). The primary verdict above follows the "
                 "protocol (best.pt evaluation with CIs).\n")
    L.append("## Per-seed contrasts (alternative - reference; negative favours the alternative)\n")
    pc = tables["paired_contrasts"]
    L.append("| Seed | Contrast | Metric | delta | rel. | 95% CI delta | frac. replicates favouring alt |")
    L.append("| --- | --- | --- | ---: | ---: | --- | ---: |")
    for _, r in pc.iterrows():
        L.append(f"| {r.seed} | {SHORT[r.reference]} -> {SHORT[r.alternative]} | {r.metric.upper()} | {r.delta:+.6f} | "
                 f"{pct(r.relative)} | [{r.ci_lo:+.6f}, {r.ci_hi:+.6f}] | {r.frac_replicates_favoring_alt:.3f} |")
    L.append("")
    L.append("## Protocol table (rel. MSE [95% CI], dMAE, best epochs)\n")
    L.append("| Seed | relMSE H->H+D [95% CI] | dMAE H->H+D | relMSE H+D->Clean [95% CI] | dMAE H+D->Clean | best epochs (H / H+D / Clean) |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    pm = tables["per_run_metrics"]

    def cell(seed, a, b, metric):
        s = pc[(pc.seed == seed) & (pc.reference == a) & (pc.alternative == b) & (pc.metric == metric)]
        return s.iloc[0] if len(s) else None

    for seed in SEEDS:
        m1, a1, m2, a2 = cell(seed, H, HD, "mse"), cell(seed, H, HD, "mae"), cell(seed, HD, CLEAN, "mse"), cell(seed, HD, CLEAN, "mae")
        be = " / ".join(str(int(pm[(pm.arm == a) & (pm.seed == seed)].best_epoch.iloc[0]))
                        if len(pm[(pm.arm == a) & (pm.seed == seed)]) else "-" for a in ARMS)
        f1 = f"{pct(m1.relative)} [{pct(m1.rel_ci_lo)}, {pct(m1.rel_ci_hi)}]" if m1 is not None else "incomplete"
        f2 = f"{pct(m2.relative)} [{pct(m2.rel_ci_lo)}, {pct(m2.rel_ci_hi)}]" if m2 is not None else "incomplete"
        L.append(f"| {seed} | {f1} | {f'{a1.delta:+.6f}' if a1 is not None else '-'} | {f2} | "
                 f"{f'{a2.delta:+.6f}' if a2 is not None else '-'} | {be} |")
    sm = tables["across_seeds"]
    for (a, b), lab in (((H, HD), "H->H+D"), ((HD, CLEAN), "H+D->Clean")):
        r = sm[(sm.reference == a) & (sm.alternative == b) & (sm.metric == "mse")]
        if len(r):
            r = r.iloc[0]
            L.append(f"| mean {lab} (min, max, SD; k={int(r.n_seeds)}) | {pct(r.mean_relative)} ({pct(r.min_relative)}, "
                     f"{pct(r.max_relative)}, {pct(r.sd_relative)}) | | | | |")
    L.append("")
    L.append("## Across seeds\n")
    L.append("| Contrast | Metric | k seeds | mean delta | mean rel. | min / max rel. | SD rel. | sign favours alt | CI excl. 0 favouring alt | "
             "delta as % of gap_prior (mean) | d skill vs prior (mean) |")
    L.append("| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for _, r in sm.iterrows():
        L.append(f"| {SHORT[r.reference]} -> {SHORT[r.alternative]} | {r.metric.upper()} | {int(r.n_seeds)} | {r.mean_delta:+.6f} | "
                 f"{pct(r.mean_relative)} | {pct(r.min_relative)} / {pct(r.max_relative)} | {pct(r.sd_relative)} | "
                 f"{int(r.k_sign_favours_alt)}/{int(r.n_seeds)} | {int(r.k_ci_excludes0_favours_alt)}/{int(r.n_seeds)} | "
                 f"{pct(r.delta_fraction_of_gap_prior, sign=False) if r.metric == 'mse' else ''} | " + (f"{r.mean_delta_skill:+.4f}" if r.metric == "mse" else "") + " |")
    L.append("")
    L.append("## Per run (best.pt, 50% masking, validation)\n")
    L.append("| Arm | Seed | MSE | MAE | MAS-PCC | MAC-PCC | skill vs prior | best ep (0-based) | last ep | epochs run | early stop | "
             "best epoch in last 10 (budget-limited) | train h |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |")
    for _, r in pm.iterrows():
        L.append(f"| {SHORT[r.arm]} | {r.seed} | {r.mse:.6f} | {r.mae:.6f} | {r.mas_pcc:.5f} | {r.mac_pcc:.5f} | {r.skill_vs_prior:.4f} | "
                 f"{r.best_epoch} | {r.last_epoch} | {r.epochs_run} | {'yes' if r.stopped_early else 'no'} | "
                 f"{'**YES**' if r.not_converged else 'no'} | {r.train_seconds / 3600:.2f} |")
    L.append("")
    L.append("Learning curve summary (validation MSE at 1-based epochs 30/50/70/final; slope = mean linear slope over the last 10 epochs "
             "in % of MSE per epoch; negative = still improving):\n")
    L.append("| Arm | Seed | MSE@30 | MSE@50 | MSE@70 | MSE final | slope last10 (%/ep) |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r in pm.iterrows():
        f = lambda x: "n/a" if not np.isfinite(x) else f"{x:.6f}"
        L.append(f"| {SHORT[r.arm]} | {r.seed} | {f(r.val_mse_epoch30)} | {f(r.val_mse_epoch50)} | {f(r.val_mse_epoch70)} | "
                 f"{f(r.val_mse_final)} | {r.slope_last10_pct_per_epoch:+.3f} |")
    L.append("")
    gp = tables["last10_gaps"]
    if len(gp):
        L.append("Last-10-epoch average gap vs Histone (same seed; negative = alternative better; sign stable = same sign in all 10 epochs):\n")
        L.append("| Seed | Alternative | gap MSE | rel. | sign stable | gap MAE | rel. | sign stable |")
        L.append("| ---: | --- | ---: | ---: | --- | ---: | ---: | --- |")
        for _, r in gp.iterrows():
            L.append(f"| {r.seed} | {SHORT[r.alternative]} | {r.mse_gap:+.6f} | {pct(r.mse_rel)} | {r.mse_sign_stable} | "
                     f"{r.mae_gap:+.6f} | {pct(r.mae_rel)} | {r.mae_sign_stable} |")
        L.append("")
    L.append("## Pairing and provenance\n")
    pr = tables["pairing"]
    L.append("Pairing check (sample_index, target_matrix_column, panel_repeat, target, prior_prediction bit-identical across arms within "
             "seed): " + "; ".join(f"seed {s}: {'identical' if p.get('all_identical') else 'DIFFERENT'}" for s, p in pr.items()) + ".\n")
    L.append("Files: `per_run_metrics.csv`, `validation_curves_all_runs.csv`, `paired_contrasts.csv`, `across_seeds_summary.csv`, "
             "`last10_gaps.csv`, `decision.json`, `pairing_checks.json`" + (", `validation_curves.png`" if plot_ok else
                                                                         " (plot skipped: matplotlib not installed)") + ".\n")
    L.append("Reproduce: `nice -n 10 python scripts/analyze_regulatory_confirm.py`.")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ main pipeline
def analyze(runs_root: Path, out_dir: Path, report_path: Path | None, registry: pd.DataFrame, names, *,
            replicates: int, seed: int, args=None) -> dict:
    done, missing = discover(runs_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = {k: load_run(k[0], k[1], d) for k, d in done.items()}
    rows = [loaded[k]["row"] for k in sorted(loaded, key=lambda k: (k[1], ARMS.index(k[0])))]
    pm = pd.DataFrame(rows)
    curves = pd.concat([loaded[k]["curve"] for k in sorted(loaded, key=lambda k: (k[1], ARMS.index(k[0])))], ignore_index=True) \
        if loaded else pd.DataFrame(columns=["arm", "seed", "epoch"])
    # last-10 gaps vs histone
    gap_rows = []
    for s in SEEDS:
        for alt in (HD, CLEAN):
            if (H, s) in loaded and (alt, s) in loaded:
                g = last10_gaps(curves, s, H, alt)
                gap_rows.append({"seed": s, "reference": H, "alternative": alt,
                                 "mse_gap": g["validation_mse"]["gap"], "mse_rel": g["validation_mse"]["rel"],
                                 "mse_sign_stable": g["validation_mse"]["sign_stable"],
                                 "mae_gap": g["validation_mae"]["gap"], "mae_rel": g["validation_mae"]["rel"],
                                 "mae_sign_stable": g["validation_mae"]["sign_stable"],
                                 "n_epochs": g["validation_mse"]["n_epochs"]})
    gaps = pd.DataFrame(gap_rows)
    # contrasts per seed, grouped by seeds having at least two arms
    comp, pairing = [], {}
    for s in SEEDS:
        avail = [a for a in ARMS if (a, s) in loaded]
        if len(avail) < 2:
            continue
        raw, tsq, tabs = {}, {}, {}
        for a in avail:
            with np.load(loaded[(a, s)]["outputs_path"]) as h:
                o = {k: h[k] for k in h}
            raw[a] = o
            tsq[a] = prediction_table(o, registry, names)
            tabs[a] = prediction_table(abs_error_view(o), registry, names)
        pairing[str(s)] = pairing_check(raw, avail[0])
        if not pairing[str(s)]["all_identical"]:
            raise SystemExit(f"pairing differs across arms for seed {s}")
        for a, b in CONTRASTS:
            if a in raw and b in raw:
                pa, pb = loaded[(a, s)]["row"], loaded[(b, s)]["row"]
                for metric, t in (("mse", tsq), ("mae", tabs)):
                    r = boot(t[a], t[b], seed, replicates)
                    comp.append({"seed": s, "reference": a, "alternative": b, "metric": metric, **r,
                                 "d_mas_pcc": pb["mas_pcc"] - pa["mas_pcc"], "d_mac_pcc": pb["mac_pcc"] - pa["mac_pcc"],
                                 "prior_mse": pa["prior_mse"], "gap_prior": pa["prior_mse"] - pa["mse"],
                                 "d_skill_vs_prior": pb["skill_vs_prior"] - pa["skill_vs_prior"],
                                 "replicates": replicates, "bootstrap_seed": seed})
        del raw, tsq, tabs
    pc = pd.DataFrame(comp)
    # across-seed summary
    sm = []
    for a, b in CONTRASTS:
        for metric in ("mse", "mae"):
            d = pc[(pc.reference == a) & (pc.alternative == b) & (pc.metric == metric)] if len(pc) else pc
            if not len(d):
                continue
            rel = d.relative.to_numpy()
            gp = d.gap_prior.to_numpy()
            sm.append({"reference": a, "alternative": b, "metric": metric, "n_seeds": len(d), "seeds": ",".join(map(str, d.seed)),
                       "mean_delta": float(d.delta.mean()), "mean_relative": float(rel.mean()), "min_relative": float(rel.min()),
                       "max_relative": float(rel.max()), "sd_relative": _sd(rel), "median_relative": float(np.median(rel)),
                       "k_sign_favours_alt": int((d.delta < 0).sum()), "k_ci_excludes0_favours_alt": int((d.ci_hi < 0).sum()),
                       "k_ci_excludes0_any": int(((d.ci_hi < 0) | (d.ci_lo > 0)).sum()),
                       "mean_ci_hi": float(d.ci_hi.mean()),
                       "delta_fraction_of_gap_prior": float(np.mean(-d.delta.to_numpy() / gp)) if metric == "mse" else np.nan,
                       "mean_delta_skill": float(d.d_skill_vs_prior.mean()) if metric == "mse" else np.nan})
    sm = pd.DataFrame(sm)

    def seed_records(a, b, source: str):
        recs = []
        for s in SEEDS:
            if len(pc) == 0:
                break
            m = pc[(pc.seed == s) & (pc.reference == a) & (pc.alternative == b) & (pc.metric == "mse")]
            e = pc[(pc.seed == s) & (pc.reference == a) & (pc.alternative == b) & (pc.metric == "mae")]
            if not len(m):
                continue
            m, e = m.iloc[0], e.iloc[0]
            if source == "primary":
                recs.append({"seed": s, "rel_mse": m.relative, "d_mse": m.delta, "d_mae": e.delta, "mse_ci_hi": m.ci_hi,
                             "mse_ci_lo": m.ci_lo, "mae_ci_hi": e.ci_hi, "gap_prior": m.gap_prior})
            else:
                g = gaps[(gaps.seed == s) & (gaps.reference == a) & (gaps.alternative == b)] if a == H else None
                if g is None:  # H+D -> Clean from the two curves directly
                    ca, cb = curves[(curves.arm == a) & (curves.seed == s)].set_index("epoch"), \
                        curves[(curves.arm == b) & (curves.seed == s)].set_index("epoch")
                    com = ca.index.intersection(cb.index)[-CONV_WINDOW:]
                    dm = float((cb.loc[com, "validation_mse"] - ca.loc[com, "validation_mse"]).mean())
                    dae = float((cb.loc[com, "validation_mae"] - ca.loc[com, "validation_mae"]).mean())
                    rel = dm / float(ca.loc[com, "validation_mse"].mean())
                else:
                    dm, dae, rel = float(g.mse_gap.iloc[0]), float(g.mae_gap.iloc[0]), float(g.mse_rel.iloc[0])
                recs.append({"seed": s, "rel_mse": rel, "d_mse": dm, "d_mae": dae, "mse_ci_hi": None, "mse_ci_lo": None,
                             "mae_ci_hi": None, "gap_prior": m.gap_prior})
        return recs

    n_done = len(loaded)
    any_nc = bool(pm.not_converged.any()) if len(pm) else False
    res = apply_decision_rule(seed_records(H, HD, "primary"), seed_records(HD, CLEAN, "primary"),
                              n_runs_done=n_done, any_not_converged=any_nc)
    res["n_runs_done"] = n_done
    res["missing_runs"] = [f"{a}__seed{s}" for a, s in missing]
    res["bootstrap"] = {"replicates": replicates, "seed": seed}
    if any_nc and len(pc):
        res["sensitivity"] = apply_decision_rule(seed_records(H, HD, "last10"), seed_records(HD, CLEAN, "last10"),
                                                 n_runs_done=n_done, any_not_converged=any_nc, point_only=True)
    # outputs
    pm.to_csv(out_dir / "per_run_metrics.csv", index=False)
    curves.to_csv(out_dir / "validation_curves_all_runs.csv", index=False)
    pc.to_csv(out_dir / "paired_contrasts.csv", index=False)
    sm.to_csv(out_dir / "across_seeds_summary.csv", index=False)
    gaps.to_csv(out_dir / "last10_gaps.csv", index=False)
    (out_dir / "pairing_checks.json").write_text(json.dumps(pairing, indent=2))
    (out_dir / "decision.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    plot_ok = plot_curves(curves, out_dir / "validation_curves.png") if len(curves) else False
    if report_path is not None:
        class _A:  # minimal args holder for the report
            pass
        a = args or _A()
        for k, v in (("replicates", replicates), ("seed", seed), ("out_dir_rel", out_dir)):
            if not hasattr(a, k):
                setattr(a, k, v)
        tables = {"paired_contrasts": pc, "per_run_metrics": pm, "across_seeds": sm, "last10_gaps": gaps, "pairing": pairing}
        report_path.write_text(build_report(res, tables, missing, a, plot_ok))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--out-dir", type=Path, default=RUNS_ROOT / "analysis")
    ap.add_argument("--report", type=Path, default=ROOT / "docs/REGULATORY_CONFIRMATION_RESULTS.md")
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17, help="bootstrap seed")
    args = ap.parse_args()
    os.nice(10)
    from cpg_repr_benchmark.data.methylation import read_axis
    registry = pd.read_parquet(CAMPAIGN / "loci.parquet")
    done, _ = discover(args.runs_root)
    if not done:
        raise SystemExit("no finished runs yet")
    cfg0 = yaml.safe_load((next(iter(done.values())) / "resolved_config.yaml").read_text())
    _, names = read_axis(ROOT / cfg0["dataset"]["methylation_h5"])  # sample_name axis only
    args.out_dir_rel = args.out_dir
    res = analyze(args.runs_root, args.out_dir, args.report, registry, names, replicates=args.replicates, seed=args.seed, args=args)
    print(json.dumps({k: res[k] for k in ("verdict", "reason", "provisional", "provisional_reasons", "n_runs_done")}, indent=2))


if __name__ == "__main__":
    main()
