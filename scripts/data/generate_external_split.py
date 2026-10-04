#!/usr/bin/env python3
"""Generate the external GSE40279 patient protocol (80/10/10 by individual, stratified sex x age tercile).

Writes NEW files only: patient protocol npz (format of outputs/encode_atlas_v1/patients.npz: train, validation, test,
sample_names, seed), split CSV (tracked, subject ids per split), split manifest JSON (tracked). Test beta values are
never read here (only phenotypes metadata).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cpg_repr_benchmark.encode_atlas.protocol import load_patient_protocol
from cpg_repr_benchmark.external.io import sha256_file, write_json_new
from cpg_repr_benchmark.external.split import age_tercile_thresholds, age_terciles, stratified_split

REPO = Path(__file__).resolve().parents[2]
SEED = 20260925
SPLITS = ("train", "validation", "test")


def build_split_table(pheno: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, tuple[float, float]]:
    thr = age_tercile_thresholds(pheno["age"].to_numpy())
    t = pheno.copy()
    t["age_tercile"] = age_terciles(t["age"].to_numpy(), thr)
    t["stratum"] = t["sex"].astype(str) + "_T" + t["age_tercile"].astype(str)
    t["split"] = stratified_split(t["subject_id"].astype(str).tolist(), t["stratum"].tolist(), seed)
    return t, thr


def diagnostics(t: pd.DataFrame) -> dict:
    out = {"strata_table": {}, "age": {}, "sex": {}, "site": {}, "plate": {}, "age_smd_vs_train": {}}
    for s, g in t.groupby("stratum"):
        out["strata_table"][s] = {"total": len(g), **{sp: int((g["split"] == sp).sum()) for sp in SPLITS}}
    tr = t[t["split"] == "train"]["age"]
    for sp in SPLITS:
        g = t[t["split"] == sp]
        out["age"][sp] = {"n": len(g), "mean": float(g["age"].mean()), "sd": float(g["age"].std(ddof=1)),
                          "min": float(g["age"].min()), "max": float(g["age"].max())}
        out["sex"][sp] = {k: int(v) for k, v in g["sex"].value_counts().sort_index().items()}
        out["site"][sp] = {k: int(v) for k, v in g["site"].value_counts().sort_index().items()}
        out["plate"][sp] = {k: int(v) for k, v in g["plate"].value_counts().sort_index().items()}
        pooled = np.sqrt((g["age"].var(ddof=1) + tr.var(ddof=1)) / 2)
        out["age_smd_vs_train"][sp] = float((g["age"].mean() - tr.mean()) / pooled)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/derived/external_gse40279_v1")
    ap.add_argument("--csv", type=Path, default=REPO / "configs/external/gse40279_v1_split.csv")
    ap.add_argument("--manifest", type=Path, default=REPO / "configs/external/gse40279_v1_split_manifest.json")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    npz = args.data_dir / "patients_external_gse40279_v1.npz"
    for p in (npz, args.csv, args.manifest):
        if p.exists():
            raise SystemExit(f"{p} exists; refusing to overwrite")
    pheno = pd.read_parquet(args.data_dir / "phenotypes.parquet")
    t, thr = build_split_table(pheno, args.seed)
    counts = {sp: int((t["split"] == sp).sum()) for sp in SPLITS}
    assert counts == {"train": 524, "validation": 66, "test": 66}, counts
    names = t["sample_name"].astype(str).to_numpy()
    rows = {sp: np.sort(t.index[t["split"] == sp].to_numpy().astype(np.int64)) for sp in SPLITS}
    assert (t["row"].to_numpy() == np.arange(len(t))).all()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **rows, sample_names=names.astype(str), seed=np.asarray(args.seed, dtype=np.int64))
    loaded = load_patient_protocol(npz, names.tolist())  # the runner's own validator
    assert all(np.array_equal(loaded[k], rows[k]) for k in SPLITS)
    t[["row", "sample_name", "subject_id", "split", "sex", "age", "age_tercile", "stratum", "site", "plate"]].to_csv(
        args.csv, index=False)
    manifest = {
        "schema": "external_gse40279_v1_split_manifest/1", "seed": args.seed, "unit": "individual (656 unique subjects)",
        "fractions": [0.8, 0.1, 0.1], "stratification": "sex x age tercile (6 strata)",
        "age_tercile_rule": {"thresholds_q1_q2": list(thr),
                             "definition": "tercile = (age > q1) + (age > q2), q = numpy linear quantiles 1/3, 2/3 on the FULL cohort; tied ages are never separated",
                             "tercile_sizes": {int(k): int(v) for k, v in t["age_tercile"].value_counts().sort_index().items()}},
        "allocation": "n_test = n_val = floor(0.1*N+0.5) = 66; per-stratum quotas by largest remainder (ties: stratum key order); inside a stratum subjects sorted by subject_id, permuted by default_rng([seed, stratum_rank]); first test, then validation, rest train",
        "counts": counts, "diagnostics": diagnostics(t),
        "validation": {"partition_and_leakage_checked_by": "cpg_repr_benchmark.encode_atlas.protocol.load_patient_protocol",
                       "overlap_between_splits": 0},
        "test_policy": "test row ids are in the protocol file as the loader requires; test beta values are not read before test authorisation (phase 2 runner enforces validation-only views)",
        "sha256": {"patient_protocol_npz": sha256_file(npz), "split_csv": sha256_file(args.csv)},
        "paths": {"patient_protocol_npz": str(npz.relative_to(REPO)) if npz.is_relative_to(REPO) else str(npz)},
    }
    write_json_new(args.manifest, manifest)
    import json
    print(json.dumps({k: manifest[k] for k in ("counts", "age_tercile_rule", "sha256")}, indent=1))
    print(json.dumps(manifest["diagnostics"], indent=1))


if __name__ == "__main__":
    main()
