#!/usr/bin/env python3
"""Read-only analysis scaffold of the external GSE40279 confirmation (protocol v1.1, section 9).

PRIMARY contrasts (the whole inferential family, MSE at mask 0.50, 3 contrasts, no multiplicity adjustment, raw 95% CIs):
    functional_annotations_pca - candidate, cpgpt_large_locus - candidate, deepcpg_dna_locus - candidate
SIGN CONVENTION (same as the TCGA analysis): delta = comparator - candidate on MSE, so a POSITIVE delta means the candidate
(regulatory_histone_dnase) has the LOWER error. Relative delta = delta / candidate value. Uncertainty = the repo's
`encode_atlas.statistics.paired_bootstrap` (crossed patients x 1 Mb genomic blocks, 2,000 replicates, bootstrap seed 17), per seed;
pairing (identical patients, loci, targets, observation counts) is a hard precondition.
The functional comparison reports absolute and relative delta MSE with the paired CI only: no decision threshold is defined.
DESCRIPTIVE SECONDARY block (outside the family): deepcpg_dna_locus - cpgpt_large_locus.
Per-arm convergence (best epoch, best epoch in the last 10 epochs, last-10 slope) is descriptive. Experiment B (B_strict,
B_recalibrated) is analysed separately with `--experiment`; it is never pooled with A.

    python scripts/analyze_external_confirmation.py [--experiment A|B_strict|B_recalibrated] [--split validation|test]
        [--replicates 2000] [--seed 17] [--out-dir DIR] [--runs-root DIR]

Validation split by default; `--split test` is refused unless the manifest is final + test_set_authorized + gate green. Never writes
into run directories. Works with partial results (provisional).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cpg_repr_benchmark.encode_atlas.statistics import prediction_table
from cpg_repr_benchmark.external import gate as G
from cpg_repr_benchmark.external import runner as R
from cpg_repr_benchmark.external import transfer as T

_spec = importlib.util.spec_from_file_location("_confirm", ROOT / "scripts/analyze_regulatory_confirm.py")
_confirm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_confirm)  # boot / pairing_check / abs_error_view / curve_stats / convergence_flag (no import side effects)
boot, pairing_check, abs_error_view = _confirm.boot, _confirm.pairing_check, _confirm.abs_error_view

CANDIDATE = "regulatory_histone_dnase"
PRIMARY_COMPARATORS = ("functional_annotations_pca", "cpgpt_large_locus", "deepcpg_dna_locus")
DESCRIPTIVE_PAIR = {"reference": "cpgpt_large_locus", "alternative": "deepcpg_dna_locus"}   # delta = deepcpg - cpgpt
ARMS = (CANDIDATE, *PRIMARY_COMPARATORS)
SEEDS = (17, 42, 97)
FRACTIONS = (0.15, 0.30, 0.50, 0.70, 0.90)
PRIMARY_FRACTION = 0.50
SIGN_CONVENTION = ("delta = comparator - candidate on the error metric (positive: the candidate regulatory_histone_dnase has the "
                   "lower error); relative delta = delta / candidate value")
DESCRIPTIVE_SIGN = "delta = deepcpg_dna_locus - cpgpt_large_locus (positive: cpgpt_large_locus has the lower error)"
METRICS = ("mse", "mae", "mas_pcc", "mac_pcc")
EXPERIMENTS = ("A", "B_strict", "B_recalibrated")


# ----------------------------------------------------------------------------- guards / discovery
def check_split_allowed(manifest: dict, split: str, repo: Path = ROOT) -> None:
    if split == "validation":
        return
    if split != "test":
        raise ValueError(f"unknown split {split!r}")
    from cpg_repr_benchmark.experiments.guards import require_test_authorization
    require_test_authorization(manifest)   # authorized AND final, else PermissionError
    G.authorize_external_test(repo)        # and an all-green gate


def discover(experiment: str, repo: Path, runs_root: Path | None = None) -> tuple[dict, list]:
    """{(arm, seed): run_dir} of FINISHED runs, [(arm, seed)] missing."""
    done, missing = {}, []
    for arm in ARMS:
        for seed in SEEDS:
            if experiment == "A":
                mk = R.read_marker(runs_root or repo, arm, seed) if runs_root is None else _marker_at(runs_root, arm, seed)
                rd = Path(mk["run_dir"]) if mk else None
            else:
                base = (runs_root or (repo / T.TRANSFER_ROOT)) / experiment / arm / f"seed_{seed}"
                rd = base if (base / "transfer.done").is_file() else None
            if rd is None:
                missing.append((arm, seed))
            else:
                done[(arm, seed)] = rd
    return done, missing


def _marker_at(runs_root: Path, arm: str, seed: int):
    p = Path(runs_root) / "logs" / f"{arm}__seed{seed}.done"
    return json.loads(p.read_text()) if p.is_file() else None


# ----------------------------------------------------------------------------- loading
def load_run(run_dir: Path, experiment: str, arm: str, seed: int, split: str = "validation", fractions=FRACTIONS) -> dict:
    """Layout: <run>/evaluation/<split>/seen/mask_<f>/{metrics.json,predictions.npz} (split_dirs)."""
    run_dir = Path(run_dir)
    out = {"experiment": experiment, "arm": arm, "seed": seed, "metrics": {}, "paths": {}, "history": None}
    for f in fractions:
        mdir = run_dir / "evaluation" / split / "seen" / f"mask_{f:.2f}"
        m = json.loads((mdir / "metrics.json").read_text())
        if m.get("patient_view") != split:
            raise PermissionError(f"{arm}/seed{seed}/mask_{f:.2f}: metrics patient_view {m.get('patient_view')!r} != {split!r}")
        out["metrics"][f] = m
        out["paths"][f] = mdir / "predictions.npz"
    if experiment == "A":
        hist = json.loads((run_dir / "history.json").read_text())
        if len(hist) != 120:
            raise ValueError(f"{arm}/seed{seed}: {len(hist)} epochs in history.json, expected 120")
        out["history"] = hist
    return out


def convergence_row(run: dict) -> dict | None:
    hist = run["history"]
    if hist is None:
        return None
    mse = np.array([r["validation_mse"] for r in hist], dtype=float)
    best = int(np.argmin(mse))       # first minimum: ties -> earliest epoch
    cs = _confirm.curve_stats(mse)
    return {"arm": run["arm"], "seed": run["seed"], "best_epoch_0based": best, "epochs_run": len(mse),
            "best_epoch_in_last_10": bool(_confirm.convergence_flag(best, len(mse))), "val_mse_best": float(mse[best]),
            "val_mse_final": cs["val_mse_final"], "slope_last10_pct_per_epoch": cs["slope_last10_pct_per_epoch"]}


# ----------------------------------------------------------------------------- reading rule (neutral, protocol section 9)
def reading(ci_lo: float, ci_hi: float, delta: float) -> str:
    if ci_lo > 0:
        return "delta > 0 with CI above 0 (comparator higher error than candidate)"
    if ci_hi < 0:
        return "delta < 0 with CI below 0 (comparator lower error than candidate)"
    return "CI includes 0"


def _bootrow(ref_t, alt_t, seed, replicates):
    b = boot(ref_t, alt_t, seed, replicates)
    return b, reading(b["ci_lo"], b["ci_hi"], b["delta"])


# ----------------------------------------------------------------------------- analysis
def analyze(loaded: dict, registry: pd.DataFrame, names, out_dir: Path, *, replicates: int = 2000, seed: int = 17,
            split: str = "validation", experiment: str = "A", fractions=FRACTIONS, missing=()) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_rows = [{"arm": a, "seed": s, "mask_fraction": f, **{k: r["metrics"][f][k] for k in METRICS}, "n_pairs": r["metrics"][f].get("n_pairs")}
                for (a, s), r in loaded.items() for f in fractions]
    conv = [c for c in (convergence_row(r) for r in loaded.values()) if c]
    prim, desc, pairing = [], [], {}
    seeds = sorted({s for _, s in loaded})
    for s in seeds:
        have = [a for a in ARMS if (a, s) in loaded]
        for f in fractions:
            tsq, tabs, raw = {}, {}, {}
            for a in have:
                with np.load(loaded[(a, s)]["paths"][f]) as h:
                    raw[a] = {k: h[k] for k in h}
                tsq[a] = prediction_table(raw[a], registry, names)
                tabs[a] = prediction_table(abs_error_view(raw[a]), registry, names)
            if CANDIDATE in have and len(have) > 1:
                pairing[f"{s}/{f:.2f}"] = pairing_check(raw, CANDIDATE)
                if not pairing[f"{s}/{f:.2f}"]["all_identical"]:
                    raise SystemExit(f"pairing differs across arms for seed {s}, fraction {f}")
            if CANDIDATE in have:
                for comp in PRIMARY_COMPARATORS:
                    if comp not in have:
                        continue
                    pa, pb = loaded[(CANDIDATE, s)]["metrics"][f], loaded[(comp, s)]["metrics"][f]
                    for metric, t in (("mse", tsq), ("mae", tabs)):
                        b, rd = _bootrow(t[CANDIDATE], t[comp], seed, replicates)
                        prim.append({"seed": s, "mask_fraction": f, "candidate": CANDIDATE, "comparator": comp, "metric": metric,
                                     "primary": bool(metric == "mse" and abs(f - PRIMARY_FRACTION) < 1e-12),
                                     "d_mas_pcc": pb["mas_pcc"] - pa["mas_pcc"], "d_mac_pcc": pb["mac_pcc"] - pa["mac_pcc"],
                                     "reading": rd, "replicates": replicates, "bootstrap_seed": seed, **b})
            ref, alt = DESCRIPTIVE_PAIR["reference"], DESCRIPTIVE_PAIR["alternative"]
            if ref in have and alt in have:
                b, rd = _bootrow(tsq[ref], tsq[alt], seed, replicates)
                desc.append({"seed": s, "mask_fraction": f, "reference": ref, "alternative": alt, "metric": "mse",
                             "block": "descriptive_secondary_not_in_family", "reading": rd, "replicates": replicates,
                             "bootstrap_seed": seed, **b})
            del raw, tsq, tabs
    pp, pd_, pm, pc = pd.DataFrame(prim), pd.DataFrame(desc), pd.DataFrame(run_rows), pd.DataFrame(conv)
    across = _across_seeds(pp, group=["comparator", "mask_fraction", "metric"]) if len(pp) else pd.DataFrame()
    across_desc = _across_seeds(pd_, group=["alternative", "mask_fraction", "metric"]) if len(pd_) else pd.DataFrame()
    res = {"experiment": experiment, "split": split, "n_runs_done": len(loaded), "n_runs_expected": len(ARMS) * len(SEEDS),
           "provisional": bool(missing), "missing_runs": [f"{a}__seed{s}" for a, s in missing], "candidate": CANDIDATE,
           "primary_family": [f"{c} - {CANDIDATE}" for c in PRIMARY_COMPARATORS], "primary_endpoint": "mse@0.50",
           "multiplicity_adjustment": "none (3 contrasts, raw CIs)", "sign_convention": SIGN_CONVENTION,
           "descriptive_secondary": {"contrast": "deepcpg_dna_locus - cpgpt_large_locus", "sign_convention": DESCRIPTIVE_SIGN,
                                     "in_inferential_family": False},
           "bootstrap": {"replicates": replicates, "seed": seed, "unit": "patients x 1 Mb genomic blocks"}}
    pm.to_csv(out_dir / "per_run_metrics.csv", index=False)
    pc.to_csv(out_dir / "convergence_descriptive.csv", index=False)
    pp.to_csv(out_dir / "primary_contrasts_per_seed.csv", index=False)
    across.to_csv(out_dir / "primary_contrasts_across_seeds.csv", index=False)
    pd_.to_csv(out_dir / "descriptive_secondary_deepcpg_minus_cpgpt_per_seed.csv", index=False)
    across_desc.to_csv(out_dir / "descriptive_secondary_deepcpg_minus_cpgpt_across_seeds.csv", index=False)
    (out_dir / "pairing_checks.json").write_text(json.dumps(pairing, indent=2))
    (out_dir / "analysis.json").write_text(json.dumps(res, indent=2))
    (out_dir / "report.md").write_text(render_report(res, pm, pp, across, pd_, across_desc, pc))
    return res


def _across_seeds(df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    rows = []
    for key, d in df.groupby(group):
        rel = d.relative.to_numpy()
        rows.append({**dict(zip(group, key)), "n_seeds": len(d), "per_seed_delta": json.dumps([float(x) for x in d.delta]),
                     "mean_delta": float(d.delta.mean()), "min_delta": float(d.delta.min()), "max_delta": float(d.delta.max()),
                     "mean_relative": float(rel.mean()), "min_relative": float(rel.min()), "max_relative": float(rel.max()),
                     "n_seeds_ci_above_0": int((d.ci_lo > 0).sum()), "n_seeds_ci_below_0": int((d.ci_hi < 0).sum()),
                     "n_seeds_ci_includes_0": int(((d.ci_lo <= 0) & (d.ci_hi >= 0)).sum()),
                     "primary": bool("metric" in d and (d.metric == "mse").all() and (abs(d.mask_fraction - PRIMARY_FRACTION) < 1e-12).all())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- report
def _fmt(x, p=5):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}g}"


def render_report(res: dict, pm: pd.DataFrame, pp: pd.DataFrame, across: pd.DataFrame, pdesc: pd.DataFrame,
                  across_desc: pd.DataFrame, conv: pd.DataFrame) -> str:
    L = [f"# External confirmation, experiment {res['experiment']}, split {res['split']}"
         + (" (PROVISIONAL: partial results)" if res["provisional"] else ""), "",
         f"Runs done: {res['n_runs_done']}/{res['n_runs_expected']}. Candidate: `{CANDIDATE}`. Primary endpoint: MSE at mask 0.50.",
         f"Sign convention: {res['sign_convention']}.",
         "The three primary contrasts form the whole inferential family; raw paired 95% CIs (patients x 1 Mb block bootstrap, "
         f"{res['bootstrap']['replicates']} replicates, seed {res['bootstrap']['seed']}), no multiplicity adjustment.",
         "The functional comparison is reported as absolute and relative differences with the paired CI only; no decision threshold "
         "is applied to it.", ""]
    L += ["## Primary contrasts at MSE@0.50 (per seed)", "",
          "| comparator | seed | candidate MSE | comparator MSE | delta | relative delta | 95% CI | reading |", "|---|---|---|---|---|---|---|---|"]
    if len(pp):
        for _, r in pp[pp.primary].sort_values(["comparator", "seed"]).iterrows():
            L.append(f"| {r.comparator} | {r.seed} | {_fmt(r.ref_value)} | {_fmt(r.alt_value)} | {_fmt(r.delta)} | {_fmt(r.relative)} | "
                     f"[{_fmt(r.ci_lo)}, {_fmt(r.ci_hi)}] | {r.reading} |")
    L += ["", "## Primary contrasts across seeds (MSE@0.50)", "",
          "| comparator | per-seed delta | min | max | seeds CI above 0 | seeds CI below 0 | seeds CI includes 0 |", "|---|---|---|---|---|---|---|"]
    if len(across):
        for _, r in across[across.primary].iterrows():
            L.append(f"| {r.comparator} | {r.per_seed_delta} | {_fmt(r.min_delta)} | {_fmt(r.max_delta)} | {r.n_seeds_ci_above_0} | "
                     f"{r.n_seeds_ci_below_0} | {r.n_seeds_ci_includes_0} |")
    L += ["", "## All mask fractions (exploratory; MSE delta, mean over seeds)", "", "| comparator | fraction | mean delta | mean relative |", "|---|---|---|---|"]
    if len(across):
        for _, r in across[across.metric == "mse"].sort_values(["comparator", "mask_fraction"]).iterrows():
            L.append(f"| {r.comparator} | {r.mask_fraction:.2f} | {_fmt(r.mean_delta)} | {_fmt(r.mean_relative)} |")
    L += ["", "## Per-arm metrics (all seeds)", "", "| arm | seed | fraction | MSE | MAE | MAS-PCC | MAC-PCC |", "|---|---|---|---|---|---|---|"]
    for _, r in pm.sort_values(["arm", "seed", "mask_fraction"]).iterrows():
        L.append(f"| {r.arm} | {r.seed} | {r.mask_fraction:.2f} | {_fmt(r.mse)} | {_fmt(r.mae)} | {_fmt(r.mas_pcc, 4)} | {_fmt(r.mac_pcc, 4)} |")
    if len(conv):
        L += ["", "## Convergence (descriptive)", "", "| arm | seed | best epoch (0-based) | best in last 10 | last-10 slope (%/epoch) |", "|---|---|---|---|---|"]
        for _, r in conv.sort_values(["arm", "seed"]).iterrows():
            L.append(f"| {r.arm} | {r.seed} | {r.best_epoch_0based} | {r.best_epoch_in_last_10} | {_fmt(r.slope_last10_pct_per_epoch)} |")
    L += ["", "## Descriptive secondary block (outside the primary family): deepcpg_dna_locus - cpgpt_large_locus", "",
          f"Sign convention: {DESCRIPTIVE_SIGN}. Not part of any inferential family.", "",
          "| seed | cpgpt MSE | deepcpg MSE | delta | 95% CI | reading |", "|---|---|---|---|---|---|"]
    if len(pdesc):
        for _, r in pdesc[abs(pdesc.mask_fraction - PRIMARY_FRACTION) < 1e-12].sort_values("seed").iterrows():
            L.append(f"| {r.seed} | {_fmt(r.ref_value)} | {_fmt(r.alt_value)} | {_fmt(r.delta)} | [{_fmt(r.ci_lo)}, {_fmt(r.ci_hi)}] | {r.reading} |")
    L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------------------- external registry (blocks by chr:Mb)
def external_registry(repo: Path, manifest: dict) -> tuple[pd.DataFrame, list[str]]:
    from cpg_repr_benchmark.data.methylation import read_axis
    ids, names = read_axis(Path(repo) / manifest["files"]["dataset_h5"])      # axes only
    table = pd.read_parquet(Path(repo) / "data/cpg/registries/array_cpg_map.parquet", columns=["cpg_idx", "chr", "pos"])
    reg = table.set_index("cpg_idx").reindex(ids)
    if reg.chr.isna().any():
        raise ValueError("registry is missing external CpGs")
    return reg.reset_index(), names


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=EXPERIMENTS, default="A")
    ap.add_argument("--split", choices=("validation", "test"), default="validation")
    ap.add_argument("--runs-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17, help="bootstrap seed")
    a = ap.parse_args(argv)
    manifest = json.loads((ROOT / "configs/external/gse40279_v1_freeze_manifest.json").read_text())
    try:
        check_split_allowed(manifest, a.split)
    except PermissionError as exc:
        print(f"REFUSED: {exc}")
        return 2
    os.nice(10)
    done, missing = discover(a.experiment, ROOT, a.runs_root)
    if not done:
        print("no finished runs yet")
        return 1
    out_dir = a.out_dir or ROOT / G.OUTPUT_ROOT_REL / "analysis" / a.experiment / a.split
    G.assert_writable(ROOT, out_dir)
    registry, names = external_registry(ROOT, manifest)
    loaded = {k: load_run(d, a.experiment, k[0], k[1], a.split) for k, d in done.items()}
    res = analyze(loaded, registry, names, out_dir, replicates=a.replicates, seed=a.seed, split=a.split, experiment=a.experiment, missing=missing)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
