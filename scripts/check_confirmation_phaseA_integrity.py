#!/usr/bin/env python3
"""Read-only integrity check of the Phase A confirmation runs (validation only; never reads test data).

Rebuilds the aggregate with the repo helper, then verifies per run: confirmation_status.json protocol checks, best.pt path + sha256 (recomputed),
config hash, store sha256 (representation_manifest.json provenance vs matrix.yaml), no evaluation/test directory, stray run directories.
Writes JSON to --out (default outputs/regulatory_confirmation_v1/analysis/integrity.json).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cpg_repr_benchmark.experiments import confirmation_matrix as cm


def check(spec: dict, runs_root: Path, rehash: bool = True, rebuild: bool = False) -> dict:
    if rebuild:
        cm.write_phase_a_status(spec, runs_root)
    agg = json.loads((runs_root / cm.PHASE_A_STATUS_FILE).read_text())
    rows, anomalies = [], []
    for arm, seed in cm.jobs(spec, phase="A"):
        rd = cm.run_dir_from_log(runs_root / "logs" / f"{arm['id']}__seed{seed}.log")
        row = {"arm": arm["id"], "seed": seed, "run_dir": str(rd)}
        if rd is None or not (rd / cm.STATUS_FILE).is_file():
            anomalies.append(f"{arm['id']}#{seed}: no run dir / status file")
            rows.append(row)
            continue
        st = json.loads((rd / cm.STATUS_FILE).read_text())
        cfg = yaml.safe_load((rd / "resolved_config.yaml").read_text())
        man = json.loads((rd / "representation_manifest.json").read_text())
        best = rd / "checkpoints" / "best.pt"
        sib = sorted(p.name for p in rd.parent.iterdir())
        row.update(status=st["status"], protocol_ok=st["protocol_check"]["ok"], all_checks_true=all(
            v for k, v in st["protocol_check"].items() if k not in ("epochs_run", "epochs_expected")),
            epochs=st["epochs_run"], early_stopping_false=cfg["training"].get("early_stopping") is False,
            test_read=st["test_read"], no_test_dir=not (rd / "evaluation" / "test").exists(),
            best_pt_path_matches=st["best_pt_path"] == str(best), in_aggregate=any(
                e["arm"] == arm["id"] and e["seed"] == seed and e.get("run_dir") == str(rd) and e["status"] == st["status"]
                for e in agg["runs"]),
            store_sha_manifest=man["provenance"].get("sha256"), store_sha_matrix=arm["sha256"],
            store_path_ok=Path(man["store_h5"]).resolve() == (ROOT / arm["store_h5"]).resolve(),
            stray_run_dirs=[s for s in sib if s != rd.name], config_hash=st["config_hash"])
        row["store_sha_ok"] = row["store_sha_manifest"] == row["store_sha_matrix"]
        row["config_hash_ok"] = st["config_hash"] == rd.name.split("-")[-1]
        row["best_pt_sha256_recorded"] = st["best_pt_sha256"]
        if rehash:
            row["best_pt_sha256_disk"] = cm.sha256_file(best)
            row["best_pt_sha_ok"] = row["best_pt_sha256_disk"] == st["best_pt_sha256"]
        rows.append(row)
        for k in ("protocol_ok", "all_checks_true", "early_stopping_false", "no_test_dir", "best_pt_path_matches", "in_aggregate",
                  "store_path_ok", "store_sha_ok", "config_hash_ok"):
            if not row[k]:
                anomalies.append(f"{arm['id']}#{seed}: {k} false")
        if rehash and not row["best_pt_sha_ok"]:
            anomalies.append(f"{arm['id']}#{seed}: best.pt sha256 mismatch")
        if row["test_read"] is not False or row["stray_run_dirs"]:
            anomalies.append(f"{arm['id']}#{seed}: test_read={row['test_read']} stray={row['stray_run_dirs']}")
    test_paths = [str(p) for p in (runs_root / "benchmark").rglob("*") if p.name == "test" or p.name.startswith("test_")]
    return {"n_expected": agg["n_expected"], "n_trained_validation_frozen": agg["n_trained_validation_frozen"],
            "aggregate_test_read": agg["test_read"], "evaluation_test_paths_found": test_paths, "anomalies": anomalies,
            "ok": not anomalies and not test_paths and agg["n_trained_validation_frozen"] == agg["n_expected"] == 12, "runs": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, default=ROOT / cm.OUTPUT_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-rehash", action="store_true")
    ap.add_argument("--rebuild-aggregate", action="store_true")
    a = ap.parse_args(argv)
    os.nice(10)
    res = check(cm.load_matrix(ROOT / cm.MATRIX_PATH), a.runs_root, rehash=not a.no_rehash, rebuild=a.rebuild_aggregate)
    out = a.out or a.runs_root / "analysis" / "integrity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=2))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
