#!/usr/bin/env python
"""Validate paper/registry/* (schema, enumerations, provenance sha256, CI definitions, coverage consistency).

Read-only except with --write-output-manifest (writes paper/registry/manifests/registry_outputs_sha256.json).
Usage: python paper/scripts/validate_registry.py [--write-output-manifest] [--determinism]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REGDIR = ROOT / "paper" / "registry"
spec = importlib.util.spec_from_file_location("build_reg", ROOT / "paper/scripts/build_bio_task_registry.py")
B = importlib.util.module_from_spec(spec)
sys.modules["build_reg"] = B
spec.loader.exec_module(B)

ERR: list[str] = []
WARN: list[str] = []


def err(msg):
    ERR.append(msg)


def sha_file(rel):
    h = hashlib.sha256()
    with open(ROOT / rel, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def blank(x):
    return x is None or (isinstance(x, float) and np.isnan(x)) or str(x) == ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-output-manifest", action="store_true")
    ap.add_argument("--determinism", action="store_true")
    a = ap.parse_args()
    reg = pd.read_csv(REGDIR / "bio_task_registry.csv", low_memory=False)
    con = pd.read_csv(REGDIR / "bio_task_paired_contrasts.csv", low_memory=False)
    cov = pd.read_csv(REGDIR / "bio_task_coverage_matrix.csv")
    mat = pd.read_csv(REGDIR / "bio_task_matrix.csv", low_memory=False)
    aud = pd.read_csv(REGDIR / "per_task_audit.csv", low_memory=False)
    sch = json.load(open(REGDIR / "bio_task_registry.json"))

    # 1 schema
    if list(reg.columns) != B.REG_COLUMNS:
        err("registry columns differ from REG_COLUMNS")
    if set(sch["columns"]) != set(B.REG_COLUMNS):
        err("data dictionary does not cover exactly the registry columns")
    if list(con.columns) != B.CONTRAST_COLUMNS:
        err("contrast columns differ")
    if reg.row_id.tolist() != list(range(1, len(reg) + 1)):
        err("row_id not 1..N")
    if reg.duplicated(["task_id", "stat_name", "representation_raw", "metric", "seed", "representation_variant"]).any():
        err("duplicate (task, stat, representation, metric) rows")

    # 2 enumerations
    for col, allowed in [("biological_family", B.FAMILIES), ("status", B.STATUSES), ("circularity_level", B.CIRC),
                         ("circularity_vs_candidate", B.CIRC), ("circularity_vs_functional_legacy", B.CIRC),
                         ("circularity_vs_sequence", B.CIRC), ("metric", B.METRICS)]:
        bad = set(reg[col].dropna().unique()) - set(allowed)
        if bad or reg[col].isna().any():
            err(f"{col}: invalid/missing values {sorted(bad)}")
    if not set(reg.is_main_panel.astype(str).unique()) <= {"True", "False"}:
        err("is_main_panel not boolean")
    mp = reg[reg.is_main_panel.astype(bool)]
    if not set(mp.representation_id) <= set(B.MAIN_PANEL):
        err("is_main_panel True for a non-panel representation id")
    for arm, dim in [("regulatory_histone_dnase_v1", 256), ("cpgpt_large_locus", 512), ("deepcpg_dna_locus", 128)]:
        d = mp[mp.representation_id == arm].representation_dim.dropna().unique()
        if len(d) and set(d) != {dim}:
            err(f"main-panel {arm} has dims {sorted(d)} (expected {dim})")
    if ((reg.proxy_for.fillna("") != "") & reg.is_main_panel.astype(bool)).any():
        err("a main-panel row has proxy_for set")
    if not set(reg.proxy_for.dropna()) <= set(B.MAIN_PANEL):
        err("proxy_for outside main panel")

    # 3 provenance sha for scored rows
    scored = reg[reg.score.notna()]
    files = scored[["source_file", "source_sha256"]].drop_duplicates()
    for _, r in files.iterrows():
        if B.FORBIDDEN in str(r.source_file):
            err(f"forbidden path {r.source_file}")
            continue
        if not (ROOT / r.source_file).exists():
            err(f"missing provenance file {r.source_file}")
        elif sha_file(r.source_file) != r.source_sha256:
            err(f"sha256 mismatch {r.source_file}")
    for _, r in con[["source_file", "source_sha256"]].drop_duplicates().iterrows():
        if not (ROOT / r.source_file).exists() or sha_file(r.source_file) != r.source_sha256:
            err(f"contrast provenance problem {r.source_file}")
    if reg.score.isna().any():
        err(f"{int(reg.score.isna().sum())} registry rows without a score (rows must be scored)")

    # 4 CI definitions
    for name, df in [("registry", reg), ("contrasts", con)]:
        has = df.ci_lo.notna() | df.ci_hi.notna()
        if (has & df.ci_definition.isna()).any():
            err(f"{name}: CI present without ci_definition")
        if (df.ci_lo.notna() != df.ci_hi.notna()).any():
            err(f"{name}: only one CI bound present")
    inv = reg[reg.ci_lo.notna() & (reg.ci_lo > reg.ci_hi)]
    if len(inv):
        err("ci_lo > ci_hi in registry")
    outside = reg[reg.ci_lo.notna() & ((reg.score < reg.ci_lo - 1e-9) | (reg.score > reg.ci_hi + 1e-9))]
    if len(outside):
        WARN.append(f"{len(outside)} registry rows with point estimate outside its CI (bootstrap-percentile / seed-mixture)")

    # 5 coverage consistency
    cov2 = B.coverage_table(reg.assign(is_main_panel=reg.is_main_panel.astype(bool)).fillna({"proxy_for": ""}).to_dict("records"))
    cov_a = cov.fillna("").set_index("task_id")
    cov_b = cov2.fillna("").set_index("task_id")
    if sorted(cov_a.index) != sorted(cov_b.index):
        err("coverage task set differs from registry")
    else:
        for arm in B.MAIN_PANEL:
            if not (cov_a.loc[cov_b.index, arm] == cov_b[arm]).all():
                err(f"coverage mismatch for {arm}")
    if not set(mat.task_id) <= set(cov.task_id):
        err("matrix task not in coverage matrix")
    for arm in B.MAIN_PANEL:
        col = f"coverage_{B.ARMKEY[arm]}"
        m = mat.set_index("task_id")[col]
        if not (m == cov_a.loc[m.index, arm]).all():
            err(f"matrix {col} inconsistent with coverage matrix")
        # HAVE => the matrix arm headline value present for HAVE tasks with a headline stat
        have = mat[(mat[col] == "HAVE")]
        if have[f"headline_value_{B.ARMKEY[arm]}"].isna().any():
            err(f"matrix: coverage HAVE but empty headline for {arm}")
        miss = mat[(mat[col] == "MISSING")]
        if miss[f"headline_value_{B.ARMKEY[arm]}"].notna().any():
            err(f"matrix: coverage MISSING but headline value present for {arm}")
        prox = mat[mat[col] == "PROXY_VARIANT_ONLY"]
        if prox[f"proxy_arm_{B.ARMKEY[arm]}"].isna().any():
            err(f"matrix: PROXY flag without named proxy arm for {arm}")
    if not set(mat.coverage_cand) | set(mat.coverage_cpgpt) | set(mat.coverage_deepcpg) <= set(B.COVERAGE):
        err("matrix coverage flag outside vocabulary")
    # deltas
    dcols = [c for c in mat.columns if c.startswith("candidate_minus_") and c.count("__") == 1]
    anyd = mat[dcols].notna().any(axis=1)
    if (mat.loc[~anyd, "delta_flag"] != "no paired CI available").any():
        err("delta_flag missing for tasks without paired delta")
    if (mat.loc[anyd, "delta_flag"] == "no paired CI available").any():
        err("delta_flag says none but delta present")
    for c in dcols:
        if (mat[c + "__ci_lo"].notna() & mat[c + "__ci_definition"].isna()).any():
            err(f"{c}: CI without definition")
    for c in [c for c in mat.columns if c.endswith("__ci_lo")]:
        base = c[:-7]
        if (mat[c].notna() & mat[base].isna()).any():
            err(f"{c}: CI without score")
    for c in ["circularity_vs_candidate", "circularity_vs_functional_legacy", "circularity_vs_sequence"]:
        if not set(mat[c]) <= set(B.CIRC):
            err(f"matrix {c} vocab")
    # 7 audit
    if not set(aud.rigor_grade.dropna()) <= set(B.GRADES) or aud.rigor_grade.isna().any():
        err("audit rigor_grade invalid/missing")
    missing = set(reg.task_id) - set(aud.task_id)
    if missing:
        err(f"tasks without audit row: {sorted(missing)[:5]}")
    mq = aud[aud.rigor_grade == "MAIN_QUALITY"]
    if (mq.ci_present != "yes").any():
        err("MAIN_QUALITY without CI")
    if (mq.coverage_cand != "HAVE").any():
        err("MAIN_QUALITY task lacks the candidate (policy: main quality requires full main-panel coverage)")
    if aud[(aud.rigor_grade != "MAIN_QUALITY") & aud.problems.isna() & aud.rigor_reasons.isna()].shape[0]:
        err("non-main grade without stated reasons")
    # 8 forbidden dir anywhere
    for nm, df in [("registry", reg), ("contrasts", con), ("matrix", mat), ("audit", aud)]:
        if df.astype(str).apply(lambda s: s.str.contains(B.FORBIDDEN)).any().any():
            err(f"{nm} mentions the excluded audit directory")
    # 9 optional determinism
    if a.determinism:
        names = ["bio_task_registry.csv", "bio_task_matrix.csv", "per_task_audit.csv", "bio_task_paired_contrasts.csv",
                 "bio_task_coverage_matrix.csv", "bio_task_registry.json", "bio_task_inventory.csv", "unscored_tasks.csv"]
        before = {n: sha_file(f"paper/registry/{n}") for n in names}
        subprocess.run([sys.executable, str(ROOT / "paper/scripts/build_bio_task_registry.py")], check=True, cwd=ROOT,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}, stdout=subprocess.DEVNULL)
        after = {n: sha_file(f"paper/registry/{n}") for n in names}
        for n in names:
            if before[n] != after[n]:
                err(f"non-deterministic output {n}")
    if a.write_output_manifest and not ERR:
        out = {}
        for n in ["bio_task_registry.csv", "bio_task_registry.json", "bio_task_matrix.csv", "per_task_audit.csv", "bio_task_paired_contrasts.csv",
                  "bio_task_coverage_matrix.csv", "bio_task_inventory.csv", "unscored_tasks.csv", "manifests/input_files_sha256.json"]:
            out[f"paper/registry/{n}"] = sha_file(f"paper/registry/{n}")
        out["paper/scripts/build_bio_task_registry.py"] = sha_file("paper/scripts/build_bio_task_registry.py")
        out["paper/scripts/validate_registry.py"] = sha_file("paper/scripts/validate_registry.py")
        with open(REGDIR / "manifests" / "registry_outputs_sha256.json", "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
    for w in WARN:
        print("WARN:", w)
    for e in ERR:
        print("ERROR:", e)
    print(f"validate: {len(reg)} registry rows, {len(con)} contrasts, {len(mat)} matrix rows, {len(aud)} audit rows, "
          f"{len(ERR)} errors, {len(WARN)} warnings")
    return 1 if ERR else 0


if __name__ == "__main__":
    sys.exit(main())
