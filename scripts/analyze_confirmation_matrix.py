#!/usr/bin/env python3
"""Analysis scaffold of the frozen confirmation matrix (spec: matrix.yaml; protocol: AMENDMENT 2).

Paired contrasts of every comparator against the frozen candidate (regulatory_histone_dnase): delta = comparator - candidate, so a
POSITIVE delta means the candidate is better. Per seed and across seeds, at all five mask fractions (primary endpoint: MSE at 0.50),
secondary MAE, MAS-PCC, MAC-PCC. Uncertainty = `encode_atlas.statistics.paired_bootstrap` (crossed patient x 1 Mb block bootstrap,
identical to the validated confirmation analysis; it is imported, not re-implemented).

The same `best.pt` (selected by validation MSE @ 0.50) is evaluated at all five fractions: every fraction is read from ONE run directory.
Split: `validation` by default. `--split test` is refused unless test_set_authorized is true AND freeze_state is complete AND the
audit is green (guards.require_test_authorization); even then this scaffold has no test result directory defined and refuses.
Never writes into run directories. Works with partial results.

    python scripts/analyze_confirmation_matrix.py [--runs-root DIR] [--out-dir DIR] [--replicates 2000] [--seed 17]
        [--include-sensitivity] [--split validation|test]
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
from cpg_repr_benchmark.encode_atlas.statistics import prediction_table
from cpg_repr_benchmark.experiments import confirmation_matrix as cm
from cpg_repr_benchmark.experiments.guards import require_test_authorization

_spec = importlib.util.spec_from_file_location("_confirm", ROOT / "scripts/analyze_regulatory_confirm.py")
_confirm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_confirm)  # reuse boot / pairing_check / abs_error_view (no side effects at import)
boot, pairing_check, abs_error_view = _confirm.boot, _confirm.pairing_check, _confirm.abs_error_view

RUNS_ROOT = ROOT / cm.OUTPUT_ROOT


def check_split_allowed(spec: dict, split: str, audit_results=None) -> None:
    """validation: always allowed. test: refuse unless authorized + complete + all-green audit (and then still refuse: no test layout)."""
    if split == "validation":
        return
    if split != "test":
        raise ValueError(f"unknown split {split!r}")
    require_test_authorization(spec)
    fails = cm.failures(audit_results if audit_results is not None else cm.audit(spec, ROOT))
    if fails:
        raise PermissionError(f"GUARD: audit has {len(fails)} failure(s); the TEST split stays locked")
    raise PermissionError("GUARD: test analysis is not implemented in this scaffold (no test result directory is defined yet)")


def discover(runs_root: Path, arm_ids, seeds):
    done, missing = {}, []
    for a in arm_ids:
        for s in seeds:
            marker = runs_root / "logs" / f"{a}__seed{s}.done"
            log = runs_root / "logs" / f"{a}__seed{s}.log"
            if not marker.exists() or not log.exists():
                missing.append((a, s))
                continue
            lines = [x for x in log.read_text().splitlines() if x.startswith("RUN_DIR=")]
            if not lines:
                missing.append((a, s))
            else:
                done[(a, s)] = Path(lines[-1].split("=", 1)[1])
    return done, missing


def load_run(arm: str, seed: int, d: Path, spec: dict, split: str) -> dict:
    p = spec["protocol"]
    cfg = yaml.safe_load((d / "resolved_config.yaml").read_text())
    ev = cfg["evaluation"]
    if split == "validation" and (ev.get("patient_view") != "validation" or ev.get("require_patient_view") != "validation"):
        raise PermissionError(f"{arm}/seed{seed}: resolved config is not validation-only")
    if cfg["training"]["seed"] != seed or ev["mask_seed"] != p["mask_seed"]:
        raise ValueError(f"{arm}/seed{seed}: seed / mask_seed mismatch in the resolved config")
    if cfg["training"]["epochs"] != p["max_epochs"] or cfg["training"].get("early_stopping") is not False:
        raise ValueError(f"{arm}/seed{seed}: not the frozen 120-epoch, no-early-stopping protocol")
    hist = json.loads((d / "history.json").read_text())
    if len(hist) != p["max_epochs"]:
        raise ValueError(f"{arm}/seed{seed}: {len(hist)} epochs run, expected exactly {p['max_epochs']}")
    mse = np.array([r["validation_mse"] for r in hist])
    out = {"best_epoch": int(np.argmin(mse)), "epochs_run": len(hist), "metrics": {}, "paths": {}}
    for f in spec["mask_fractions"]:
        key = f"mask_{f:.2f}"
        m = json.loads((d / "evaluation/seen" / key / "metrics.json").read_text())
        if m.get("patient_view") != split:
            raise PermissionError(f"{arm}/seed{seed}/{key}: metrics patient_view {m.get('patient_view')!r} != {split!r}")
        out["metrics"][f] = m
        out["paths"][f] = d / "evaluation/seen" / key / "predictions.npz"
    return out


def _sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def analyze(spec: dict, runs_root: Path, out_dir: Path, registry: pd.DataFrame, names, *, replicates: int, seed: int,
            include_sensitivity: bool = False, split: str = "validation", candidate: str = cm.CANDIDATE_ID) -> dict:
    arm_ids = [a["id"] for a in cm.job_arms(spec, include_sensitivity=include_sensitivity)]
    seeds, fractions = spec["seeds"], spec["mask_fractions"]
    done, missing = discover(runs_root, arm_ids, seeds)
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = {k: load_run(k[0], k[1], d, spec, split) for k, d in done.items()}
    run_rows, comp, pairing = [], [], {}
    for (a, s), r in loaded.items():
        for f in fractions:
            m = r["metrics"][f]
            run_rows.append({"arm": a, "seed": s, "mask_fraction": f, "mse": m["mse"], "mae": m["mae"], "mas_pcc": m["mas_pcc"],
                             "mac_pcc": m["mac_pcc"], "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"]})
    for s in seeds:
        if (candidate, s) not in loaded:
            continue
        comps = [a for a in arm_ids if a != candidate and (a, s) in loaded]
        for f in fractions:
            raw, tsq, tabs = {}, {}, {}
            for a in [candidate, *comps]:
                with np.load(loaded[(a, s)]["paths"][f]) as h:
                    raw[a] = {k: h[k] for k in h}
                tsq[a] = prediction_table(raw[a], registry, names)
                tabs[a] = prediction_table(abs_error_view(raw[a]), registry, names)
            pairing[f"{s}/{f:.2f}"] = pairing_check(raw, candidate)
            if comps and not pairing[f"{s}/{f:.2f}"]["all_identical"]:
                raise SystemExit(f"pairing differs across arms for seed {s}, fraction {f}")
            for a in comps:
                pa, pb = loaded[(candidate, s)]["metrics"][f], loaded[(a, s)]["metrics"][f]
                for metric, t in (("mse", tsq), ("mae", tabs)):
                    b = boot(t[candidate], t[a], seed, replicates)
                    comp.append({"seed": s, "mask_fraction": f, "reference": candidate, "comparator": a, "metric": metric,
                                 "primary": bool(metric == "mse" and abs(f - 0.5) < 1e-12),
                                 "d_mas_pcc": pb["mas_pcc"] - pa["mas_pcc"], "d_mac_pcc": pb["mac_pcc"] - pa["mac_pcc"],
                                 "replicates": replicates, "bootstrap_seed": seed, **b})
            del raw, tsq, tabs
    pc, pm = pd.DataFrame(comp), pd.DataFrame(run_rows)
    summ = []
    if len(pc):
        for (a, f, metric), d in pc.groupby(["comparator", "mask_fraction", "metric"]):
            rel = d.relative.to_numpy()
            summ.append({"comparator": a, "mask_fraction": f, "metric": metric, "n_seeds": len(d), "mean_delta": float(d.delta.mean()),
                         "mean_relative": float(rel.mean()), "min_relative": float(rel.min()), "max_relative": float(rel.max()),
                         "sd_relative": _sd(rel), "k_candidate_better": int((d.delta > 0).sum()),
                         "k_ci_excludes0_candidate_better": int((d.ci_lo > 0).sum()),
                         "k_ci_excludes0_comparator_better": int((d.ci_hi < 0).sum()),
                         "mean_d_mas_pcc": float(d.d_mas_pcc.mean()), "mean_d_mac_pcc": float(d.d_mac_pcc.mean()),
                         "primary": bool(metric == "mse" and abs(f - 0.5) < 1e-12)})
    sm = pd.DataFrame(summ)
    res = {"split": split, "n_runs_done": len(loaded), "n_runs_expected": len(arm_ids) * len(seeds), "provisional": bool(missing),
           "missing_runs": [f"{a}__seed{s}" for a, s in missing], "candidate": candidate, "sign_convention":
           "delta = comparator - candidate (positive: candidate better)", "bootstrap": {"replicates": replicates, "seed": seed}}
    pm.to_csv(out_dir / "per_run_metrics.csv", index=False)
    pc.to_csv(out_dir / "paired_contrasts.csv", index=False)
    sm.to_csv(out_dir / "across_seeds_summary.csv", index=False)
    (out_dir / "pairing_checks.json").write_text(json.dumps(pairing, indent=2))
    (out_dir / "analysis.json").write_text(json.dumps(res, indent=2))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, default=ROOT / cm.MATRIX_PATH)
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--split", choices=("validation", "test"), default="validation")
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17, help="bootstrap seed")
    ap.add_argument("--include-sensitivity", action="store_true")
    args = ap.parse_args(argv)
    spec = cm.load_matrix(args.matrix)
    try:
        check_split_allowed(spec, args.split)
    except PermissionError as exc:
        print(f"REFUSED: {exc}")
        return 2
    os.nice(10)
    from cpg_repr_benchmark.data.methylation import read_axis
    registry = pd.read_parquet(ROOT / "outputs/encode_atlas_v1/loci.parquet")
    done, _ = discover(args.runs_root, [a["id"] for a in cm.job_arms(spec, include_sensitivity=args.include_sensitivity)], spec["seeds"])
    if not done:
        print("no finished runs yet")
        return 1
    cfg0 = yaml.safe_load((next(iter(done.values())) / "resolved_config.yaml").read_text())
    _, names = read_axis(ROOT / cfg0["dataset"]["methylation_h5"])  # sample_name axis only
    res = analyze(spec, args.runs_root, args.out_dir or args.runs_root / "analysis", registry, names, replicates=args.replicates,
                  seed=args.seed, include_sensitivity=args.include_sensitivity, split=args.split)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
