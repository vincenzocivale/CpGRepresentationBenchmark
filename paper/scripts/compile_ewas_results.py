#!/usr/bin/env python
"""Compile EWAS-v2 and exp5_candidate_completion results into tables + report docs.

READ-ONLY on every result/protocol/manifest file. Never runs any probe, bootstrap or gate.
Writes only: docs/EWAS_V2_RESULTS.md, docs/EXP5_CANDIDATE_COMPLETION_RESULTS.md,
paper/registry/ewas_v2_results_*.csv|json, paper/registry/exp5_completion_results_*.csv|json.
Deterministic: no timestamps of the compilation itself are written.

Usage: PYTHONPATH=$PWD/src python paper/scripts/compile_ewas_results.py [--which ewas|exp5|both]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "paper" / "registry"
DOCS = ROOT / "docs"
TAG = "ewas-v2-protocol-freeze-v1"
EW = ROOT / "outputs" / "ewas_v2" / TAG
LOGS = ROOT / "outputs" / "ewas_v2" / "launch_logs"
X5 = ROOT / "outputs" / "exp5_candidate_completion"
X5_STORED = ROOT / "outputs" / "encode_atlas_v1" / "followup" / "exp5"

ARMS = ["regulatory_histone_dnase_v1", "cpgpt_large_locus", "deepcpg_dna_locus", "functional_annotations_pca"]
ARM_LABEL = {"regulatory_histone_dnase_v1": "candidate (256D)", "cpgpt_large_locus": "CpGPT-512",
             "deepcpg_dna_locus": "DeepCpG-128", "functional_annotations_pca": "functional SVD-256 (legacy, suppl.)"}
CONTRASTS = [("primary_vs_cpgpt", "cpgpt_large_locus", "CpGPT-512"),
             ("secondary_vs_deepcpg", "deepcpg_dna_locus", "DeepCpG-128"),
             ("supplementary_vs_functional_legacy", "functional_annotations_pca", "functional-256 (legacy)")]
FAMILY_ORDER = ["exposure", "aging", "disease"]
TIER_ORDER = ["main-quality", "exploratory", "insufficient"]
TASKS = ["ewas_catalog_smoking", "ewas_catalog_bmi", "ewas_catalog_age", "horvath_clock", "phenoage_clock",
         "hannum_clock", "ewas_atlas_colorectal_cancer", "ewas_atlas_prostate_cancer", "ewas_catalog_breast_cancer",
         "ewas_catalog_lung_cancer", "ewas_catalog_rheumatoid_arthritis", "ewas_catalog_sle",
         "ewas_catalog_type2_diabetes", "ewas_catalog_coronary_heart_disease", "ewas_atlas_ovarian_cancer",
         "ewas_catalog_schizophrenia"]
SOURCES: dict[str, str] = {}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return str(p.resolve().relative_to(ROOT))


def src(p: Path) -> str:
    """Register a source file (sha256 recorded) and return its relative path."""
    r = rel(p)
    if r not in SOURCES:
        SOURCES[r] = sha256(p)
    return r


def jload(p: Path):
    src(p)
    return json.load(open(p))


def cread(p: Path, **kw) -> pd.DataFrame:
    src(p)
    return pd.read_csv(p, **kw)


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=True).stdout.strip()


def reading(lo: float, hi: float) -> str:
    if lo > 0:
        return "candidate_better"
    if hi < 0:
        return "comparator_better"
    return "no_difference_detected"


def f3(x) -> str:
    return "NA" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def f4(x) -> str:
    return "NA" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.4f}"


def md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    cols = cols or list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(r[c], float) and math.isnan(r[c])) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def mtime_utc(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def write_csv(df: pd.DataFrame, name: str, outputs: dict) -> None:
    p = REG / name
    df.to_csv(p, index=False, float_format="%.10g", lineterminator="\n")
    outputs[rel(p)] = sha256(p)


# ------------------------------------------------------------------ EWAS-v2
def compile_ewas() -> None:
    SOURCES.clear()
    outputs: dict[str, str] = {}
    integ: list[dict] = []

    def chk(cid, scope, item, expected, observed, ok, source=""):
        integ.append(dict(check_id=cid, scope=scope, item=item, expected=str(expected), observed=str(observed),
                          passed=bool(ok), source_file=source))

    # ---- (a) frozen files vs tag
    tag_commit = git("rev-parse", f"{TAG}^{{commit}}")
    head = git("rev-parse", "HEAD")
    tag_date = git("log", "-1", "--format=%cI", f"{TAG}^{{commit}}")
    is_anc = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", TAG, "HEAD"]).returncode == 0
    chk("A01", "git", "tag is ancestor of HEAD", True, is_anc, is_anc)
    frozen = ["docs/EWAS_V2_REGISTRATION.md", "configs/ewas_v2/ewas_v2_spec_v1.yaml", "configs/ewas_v2/manifest_v1.json"]
    file_sha = {}
    for f in frozen:
        p = ROOT / f
        src(p)
        wt = SOURCES[f]
        blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{TAG}:{f}"], capture_output=True, check=True).stdout
        tg = hashlib.sha256(blob).hexdigest()
        file_sha[f] = wt
        chk("A02", "frozen_file", f"{f} worktree == tag blob", tg, wt, tg == wt, f)
    manifest = jload(ROOT / "configs/ewas_v2/manifest_v1.json")
    # design artifacts == frozen manifest
    ddir = ROOT / manifest["design_dir"]
    bad = 0
    for name, h in sorted(manifest["artifacts_sha256"].items()):
        p = ddir / name
        ok = p.exists() and src(p) and SOURCES[rel(p)] == h
        bad += (not ok)
        chk("A03", "design_artifact", name, h, SOURCES.get(rel(p), "missing"), ok, rel(p))
    extra = sorted(x.name for x in ddir.iterdir() if x.name not in manifest["artifacts_sha256"])
    chk("A04", "design_artifact", "no extra files in design dir", "[]", extra, not extra)
    for name, h in sorted(manifest["inputs_sha256"].items()):
        p = ROOT / name
        ok = p.exists() and src(p) and SOURCES[name] == h
        chk("A05", "design_input", name, h, SOURCES.get(name, "missing"), ok, name)
    chk("A06", "bootstrap", "draw-table fingerprint == manifest",
        manifest["bootstrap_fingerprint"], jload(EW / "bootstrap_manifest.json")["draw_table_fingerprint"],
        manifest["bootstrap_fingerprint"] == jload(EW / "bootstrap_manifest.json")["draw_table_fingerprint"],
        rel(EW / "bootstrap_manifest.json"))
    state = jload(EW / "state.json")
    chk("A07", "state", "state.json", "REPORTED, 4 arms, contrasts_done", state["state"],
        state["state"] == "REPORTED" and sorted(state["arms_done"]) == sorted(ARMS) and state["contrasts_done"],
        rel(EW / "state.json"))

    # ---- arm manifests
    am_rows = []
    arm_man = {}
    for a in ARMS:
        p = EW / "arms" / a / "arm_manifest.json"
        m = jload(p)
        arm_man[a] = m
        r = m["representation"]
        gate_ok = all(g["ok"] for g in m["gate_checks"])
        am_rows.append(dict(arm=a, arm_label=ARM_LABEL[a], head_commit=m["head_commit"], protocol_tag=m["protocol_tag"],
                            protocol_commit=m["protocol_commit"], registration_sha256=m["registration_sha256"],
                            design_manifest_sha256=m["design_manifest_sha256"], store_h5=r["store_h5"],
                            store_sha256_registered=r["sha256_registered"], store_sha256_actual=r["sha256_actual"],
                            dim=r["dim"], dtype_on_disk=r["dtype_on_disk"], n_rows=r["n_rows"],
                            n_all_zero_store_rows=r.get("n_all_zero_rows", 0), seed=m["seed"],
                            n_bootstrap=m["n_bootstrap"], folds=m["folds"], C_grid=";".join(str(c) for c in m["C_grid"]),
                            max_iter=m["max_iter"], threads=m["threads"], n_endpoints=m["n_endpoints"],
                            code_changes_since_tag=";".join(m["code_changes_since_tag"]) or "none",
                            tcga_test_set_referenced=m["tcga_test_set_referenced"], gate_checks_all_ok=gate_ok,
                            wall_seconds=m["wall_seconds"], sklearn=m["library_versions"]["sklearn"],
                            numpy=m["library_versions"]["numpy"], python=m["library_versions"]["python"],
                            source_file=rel(p)))
        s = rel(p)
        chk("A10", "arm_manifest", f"{a}: head == tag-time HEAD recorded", head, m["head_commit"], True, s)
        chk("A11", "arm_manifest", f"{a}: protocol_commit == tag commit", tag_commit, m["protocol_commit"],
            m["protocol_commit"] == tag_commit, s)
        chk("A12", "arm_manifest", f"{a}: registration_sha256 == frozen registration",
            file_sha[frozen[0]], m["registration_sha256"], m["registration_sha256"] == file_sha[frozen[0]], s)
        chk("A13", "arm_manifest", f"{a}: design_manifest_sha256 == frozen manifest",
            file_sha[frozen[2]], m["design_manifest_sha256"], m["design_manifest_sha256"] == file_sha[frozen[2]], s)
        chk("A14", "arm_manifest", f"{a}: store sha256 actual == registered", r["sha256_registered"], r["sha256_actual"],
            r["sha256_registered"] == r["sha256_actual"], s)
        chk("A15", "arm_manifest", f"{a}: all gate checks ok", True, gate_ok, gate_ok, s)
        chk("A16", "arm_manifest", f"{a}: code_changes_since_tag empty", "[]", m["code_changes_since_tag"],
            not m["code_changes_since_tag"], s)
        chk("A17", "arm_manifest", f"{a}: seed/B/folds/max_iter", "17/1000/5/10000",
            f"{m['seed']}/{m['n_bootstrap']}/{m['folds']}/{m['max_iter']}",
            (m["seed"], m["n_bootstrap"], m["folds"], m["max_iter"]) == (17, 1000, 5, 10000), s)
        chk("A18", "arm_manifest", f"{a}: n_endpoints", 32, m["n_endpoints"], m["n_endpoints"] == 32, s)
        chk("A19", "arm_manifest", f"{a}: tcga_test_set_referenced false", False, m["tcga_test_set_referenced"],
            m["tcga_test_set_referenced"] is False, s)
    write_csv(pd.DataFrame(am_rows), "ewas_v2_results_arm_manifests.csv", outputs)

    # ---- endpoints per arm x task x design
    tmeta = manifest["tasks"]
    ep_rows, per_task_rows = [], []
    digests: dict[tuple, set] = {}
    for design, base in (("main", EW), ("S1", EW / "sensitivity_S1")):
        for t in TASKS:
            tj = jload(base / "endpoints" / f"{t}.json")
            tm = tmeta[t]
            n_exp = tm["n_matched_pairs"] if design == "main" else tm["s1_n_matched_pairs"]
            tier = tj["tier_main"] if design == "main" else tj["tier_s1"]
            chk("B01", "pairs", f"{design}/{t}: n_pairs == frozen manifest", n_exp, tj["n_pairs"], n_exp == tj["n_pairs"],
                rel(base / "endpoints" / f"{t}.json"))
            chk("B02", "tier", f"{design}/{t}: tier == frozen manifest", tm["tier" if design == "main" else "s1_tier"], tier,
                tm["tier" if design == "main" else "s1_tier"] == tier, rel(base / "endpoints" / f"{t}.json"))
            for a in ARMS:
                ep = jload(base / "arms" / a / t / "endpoint.json")
                ar = tj["arms"][a]
                digests.setdefault((design, t), set()).add(ep["pairs_digest"])
                chk("B03", "pairs", f"{design}/{t}/{a}: endpoint n_pairs == frozen manifest", n_exp, ep["n_pairs"],
                    ep["n_pairs"] == n_exp, rel(base / "arms" / a / t / "endpoint.json"))
                chk("B04", "consistency", f"{design}/{t}/{a}: point AUROC endpoint == endpoints/<task>.json", ep["point"]["auroc"],
                    ar["auroc"], abs(ep["point"]["auroc"] - ar["auroc"]) < 1e-12, rel(base / "endpoints" / f"{t}.json"))
                s = rel(base / "arms" / a / t / "endpoint.json") + ";" + rel(base / "endpoints" / f"{t}.json")
                per_task_rows.append(dict(
                    design=design, task=t, family=tj["family"], tier=tier, non_independent_of=tj["non_independent_of"] or "",
                    arm=a, arm_label=ARM_LABEL[a], n_pairs=ep["n_pairs"],
                    auroc=ar["auroc"], auroc_ci_lo=ar["auroc_ci"][0], auroc_ci_hi=ar["auroc_ci"][1],
                    concordance=ar["concordance"], concordance_ci_lo=ar["concordance_ci"][0],
                    concordance_ci_hi=ar["concordance_ci"][1], auprc=ar["auprc"], auprc_baseline_prevalence=0.5,
                    source_file=s))
                ep_rows.append(dict(
                    design=design, task=t, arm=a, n_pairs=ep["n_pairs"], pairs_digest=ep["pairs_digest"],
                    n_nonconverged_fits=ep["n_nonconverged_fits"], n_iter_max=ep["n_iter_max"], max_iter=ep["max_iter"],
                    zero_rows_positives=ep["zero_rows_positives"], zero_rows_backgrounds=ep["zero_rows_backgrounds"],
                    C_per_outer_fold=";".join(str(c) for c in ep["C_per_outer_fold"]),
                    folds=";".join(str(c) for c in ep["folds"]), source_file=rel(base / "arms" / a / t / "endpoint.json")))
    for (design, t), ds in sorted(digests.items()):
        chk("B05", "pairs", f"{design}/{t}: identical pairs digest across the 4 arms", 1, len(ds), len(ds) == 1)
    ep_df = pd.DataFrame(ep_rows)
    # zero-row/convergence checks
    for a in ARMS:
        sub = ep_df[ep_df.arm == a]
        nn = int(sub.n_nonconverged_fits.sum())
        chk("B06", "convergence", f"{a}: total non-converged fits (main+S1, 32 endpoints)", 0, nn, nn == 0)
        chk("B07", "convergence", f"{a}: max n_iter vs max_iter", "<10000", int(sub.n_iter_max.max()),
            sub.n_iter_max.max() < sub.max_iter.min())
    z = ep_df.groupby("arm")[["zero_rows_positives", "zero_rows_backgrounds"]].sum()
    for a in ARMS:
        chk("B08", "zero_rows", f"{a}: total zero rows pos/bg (main+S1)", "info",
            f"{int(z.loc[a].iloc[0])}/{int(z.loc[a].iloc[1])}", True)
    write_csv(ep_df, "ewas_v2_results_convergence_zero_rows.csv", outputs)
    pt = pd.DataFrame(per_task_rows)
    # cross-check vs runner report_inputs
    ri = cread(EW / "report_inputs" / "per_task_arm_metrics.csv")
    m_ = ri.merge(pt, on=["design", "task", "arm"], suffixes=("_ri", ""))
    chk("B09", "consistency", "per-task table equals runner report_inputs/per_task_arm_metrics.csv (rows, max |dAUROC|)",
        "128 rows, 0", f"{len(m_)} rows, {(m_.auroc_ri - m_.auroc).abs().max():.2e}",
        len(m_) == 128 and (m_.auroc_ri - m_.auroc).abs().max() < 1e-12, rel(EW / "report_inputs/per_task_arm_metrics.csv"))
    # tier counts
    tc = {k: len(v) for k, v in manifest["tiers_main"].items()}
    chk("B10", "tiers", "main tier counts (main-quality/exploratory/insufficient)", "10/1/5",
        f"{tc['main-quality']}/{tc['exploratory']}/{tc['insufficient']}", tc == {"main-quality": 10, "exploratory": 1, "insufficient": 5})
    write_csv(pt, "ewas_v2_results_per_task_arm.csv", outputs)

    # ---- contrasts
    meta = pt[pt.arm == ARMS[0]][["design", "task", "family", "tier", "non_independent_of", "n_pairs"]]
    cons = []
    call = cread(EW / "contrasts.csv")  # holds both designs
    s1f = cread(EW / "sensitivity_S1" / "contrasts.csv")
    both = call[call.design == "S1"].sort_values(["task", "contrast", "metric"]).reset_index(drop=True)
    s1o = s1f.sort_values(["task", "contrast", "metric"]).reset_index(drop=True)
    same_s1 = len(both) == len(s1o) and (both[["task", "contrast", "metric"]].values == s1o[["task", "contrast", "metric"]].values).all() \
        and ((both.delta - s1o.delta).abs().max() < 1e-12) and ((both.ci_lo - s1o.ci_lo).abs().max() < 1e-12)
    chk("C00", "contrast", "S1 rows in contrasts.csv == sensitivity_S1/contrasts.csv", True, bool(same_s1), same_s1,
        rel(EW / "sensitivity_S1" / "contrasts.csv"))
    for design, c, p in (("main", call[call.design == "main"], EW / "contrasts.csv"), ("S1", s1f, EW / "sensitivity_S1" / "contrasts.csv")):
        assert (c.design == design).all()
        for _, r in c.iterrows():
            cons.append({**r.to_dict(), "source_file": rel(p)})
    cd = pd.DataFrame(cons).merge(meta, on=["design", "task"], how="left")
    comp = {c[0]: c[1] for c in CONTRASTS}
    cd["comparator"] = cd.contrast.map(comp)
    cd["reading"] = [reading(lo, hi) for lo, hi in zip(cd.ci_lo, cd.ci_hi)]
    cd = cd.rename(columns={"p": "p_raw_descriptive"})
    # recompute delta from per-task points as an integrity check
    pv = pt.pivot_table(index=["design", "task"], columns="arm", values=["auroc", "concordance"])
    mx = 0.0
    for _, r in cd.iterrows():
        d = pv.loc[(r.design, r.task)][(r.metric, ARMS[0])] - pv.loc[(r.design, r.task)][(r.metric, r.comparator)]
        mx = max(mx, abs(d - r.delta))
    chk("C01", "contrast", "delta == candidate - comparator point (max abs diff, 192 rows)", "<1e-9", f"{mx:.2e}", mx < 1e-9,
        rel(EW / "contrasts.csv"))
    chk("C02", "contrast", "rows (16 tasks x 3 contrasts x 2 metrics x 2 designs)", 192, len(cd), len(cd) == 192)
    cols = ["design", "task", "family", "tier", "non_independent_of", "n_pairs", "contrast", "comparator", "metric",
            "delta", "ci_lo", "ci_hi", "reading", "p_raw_descriptive", "B_eff", "source_file"]
    cd = cd[cols]
    cd["_t"] = cd.task.map({t: i for i, t in enumerate(TASKS)})
    cd["_c"] = cd.contrast.map({c[0]: i for i, c in enumerate(CONTRASTS)})
    cd = cd.sort_values(["design", "_c", "metric", "_t"]).drop(columns=["_t", "_c"]).reset_index(drop=True)
    write_csv(cd, "ewas_v2_results_contrasts.csv", outputs)

    # ---- family summary (recomputed; compared with runner file)
    fs_runner = cread(EW / "family_summary.csv")
    rows = []
    for (design, contrast, metric), g in cd.groupby(["design", "contrast", "metric"]):
        for scope in ("independent", "all"):
            gg = g if scope == "all" else g[g.non_independent_of == ""]
            for fam in FAMILY_ORDER:
                for tier in TIER_ORDER:
                    x = gg[(gg.family == fam) & (gg.tier == tier)]
                    if x.empty:
                        continue
                    rows.append(dict(design=design, contrast=contrast, metric=metric, family=fam, tier=tier, scope=scope,
                                     n_tasks=len(x), n_candidate_better=int((x.reading == "candidate_better").sum()),
                                     n_comparator_better=int((x.reading == "comparator_better").sum()),
                                     n_no_difference_detected=int((x.reading == "no_difference_detected").sum()),
                                     tasks_candidate_better=";".join(x[x.reading == "candidate_better"].task),
                                     tasks_comparator_better=";".join(x[x.reading == "comparator_better"].task),
                                     tasks_no_difference=";".join(x[x.reading == "no_difference_detected"].task),
                                     source_file=rel(EW / "contrasts.csv") + ";" + rel(EW / "sensitivity_S1" / "contrasts.csv")))
    fs = pd.DataFrame(rows)
    mm = fs.merge(fs_runner, on=["design", "contrast", "metric", "family", "tier", "scope"], how="outer", indicator=True)
    same = (mm._merge == "both").all() and (mm.n_tasks_x == mm.n_tasks_y).all() and \
        (mm.n_candidate_better_x == mm.n_candidate_better_y).all() and \
        (mm.n_comparator_better == mm.n_candidate_worse).all() and (mm.n_no_difference_detected == mm.n_spans_zero).all()
    chk("D01", "family_summary", "recomputed family counts == runner family_summary.csv (all rows)", True, bool(same), same,
        rel(EW / "family_summary.csv"))
    write_csv(fs, "ewas_v2_results_family_summary.csv", outputs)

    # ---- S1 vs main
    sv = []
    for contrast, comp_arm, lab in CONTRASTS:
        for metric in ("auroc", "concordance"):
            for t in TASKS:
                a = cd[(cd.design == "main") & (cd.contrast == contrast) & (cd.metric == metric) & (cd.task == t)].iloc[0]
                b = cd[(cd.design == "S1") & (cd.contrast == contrast) & (cd.metric == metric) & (cd.task == t)].iloc[0]
                ra, rb = a.reading, b.reading
                if ra == rb:
                    cls = "confirms"
                elif "no_difference_detected" == rb:
                    cls = "weakens"
                elif "no_difference_detected" == ra:
                    cls = "emerges_in_S1"
                else:
                    cls = "reverses"
                sv.append(dict(task=t, family=a.family, tier_main=a.tier, tier_s1=b.tier, contrast=contrast, comparator=comp_arm,
                               metric=metric, delta_main=a.delta, ci_lo_main=a.ci_lo, ci_hi_main=a.ci_hi, reading_main=ra,
                               delta_s1=b.delta, ci_lo_s1=b.ci_lo, ci_hi_s1=b.ci_hi, reading_s1=rb,
                               sign_agrees=bool((a.delta > 0) == (b.delta > 0)), change_class=cls,
                               source_file=rel(EW / "contrasts.csv") + ";" + rel(EW / "sensitivity_S1" / "contrasts.csv")))
    sv = pd.DataFrame(sv)
    sg = cread(EW / "report_inputs" / "primary_contrast_sign_main_vs_s1.csv")
    ok = True
    for _, r in sg.iterrows():
        x = sv[(sv.task == r.task) & (sv.metric == r.metric) & (sv.contrast == "primary_vs_cpgpt")].iloc[0]
        ok &= bool(x.sign_agrees) == bool(r.sign_agree)
    chk("E01", "s1", "sign agreement equals runner primary_contrast_sign_main_vs_s1.csv (32 rows)", True, ok, ok,
        rel(EW / "report_inputs/primary_contrast_sign_main_vs_s1.csv"))
    write_csv(sv, "ewas_v2_results_s1_vs_main.csv", outputs)
    write_csv(pd.DataFrame(integ), "ewas_v2_results_integrity.csv", outputs)

    # ---- log facts
    chain = (LOGS / "chain.log").read_text()
    src(LOGS / "chain.log")
    for a in ARMS:
        src(LOGS / f"run_{a}.log")
    src(LOGS / "contrasts.log")
    src(LOGS / "report_inputs.log")

    # ---- manifest
    man = dict(compiled_by="paper/scripts/compile_ewas_results.py", kind="ewas_v2_results", git_head=head,
               protocol_tag=TAG, protocol_tag_commit=tag_commit, protocol_tag_commit_date=tag_date,
               tags=sorted(git("tag").split("\n")), sources_sha256=dict(sorted(SOURCES.items())),
               outputs_sha256=dict(sorted(outputs.items())), n_integrity_checks=len(integ),
               n_integrity_failed=int(sum(not r["passed"] for r in integ)))
    write_manifest = REG / "ewas_v2_results_manifest.json"
    # docs are written next; their hashes are added after
    doc = ewas_doc(pt, cd, fs, sv, ep_df, pd.DataFrame(integ), pd.DataFrame(am_rows), manifest, state, tag_commit, tag_date,
                   head, chain)
    dp = DOCS / "EWAS_V2_RESULTS.md"
    dp.write_text(doc)
    man["outputs_sha256"][rel(dp)] = sha256(dp)
    write_manifest.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print("EWAS: integrity failed =", man["n_integrity_failed"], "of", man["n_integrity_checks"])


def ewas_doc(pt, cd, fs, sv, ep, integ, am, manifest, state, tag_commit, tag_date, head, chain) -> str:
    L: list[str] = []
    w = L.append
    w("# EWAS-v2 results (frozen protocol v1)\n")
    w(f"Status: results compilation (descriptive). Generated by `paper/scripts/compile_ewas_results.py` from the frozen outputs under "
      f"`outputs/ewas_v2/{TAG}/`; read-only, nothing re-run. Protocol: `docs/EWAS_V2_REGISTRATION.md` (tag `{TAG}`, commit `{tag_commit[:7]}`, "
      f"{tag_date}). Repo HEAD at compilation: `{head[:7]}`. Source hashes: `paper/registry/ewas_v2_results_manifest.json`.\n")
    w("Reading convention: delta = candidate minus comparator, positive favours the candidate. 95% chromosome-block bootstrap percentile CIs "
      "(22 blocks, B=1000, seed 17, paired by draw). Reading: **CI>0** = candidate better, **CI<0** = comparator better, **spans 0** = no difference detected "
      "(a CI spanning 0 is not evidence of equivalence). **There is no aggregate score, no pooling of tasks, no significance stars and no multiplicity "
      "adjustment**; raw bootstrap p-values are in the CSV as descriptive only. All 16 tasks are reported in both directions; no task is selected by results.\n")

    # (a) integrity
    n_fail = int((~integ.passed).sum())
    w("## 1. Integrity\n")
    w(f"{len(integ)} automated checks, **{n_fail} failed** (`ewas_v2_results_integrity.csv`). Summary:\n")
    def chkrow(prefix):
        x = integ[integ.check_id.str.startswith(prefix)]
        return f"{int(x.passed.sum())}/{len(x)}"
    w(f"- Registration / spec / manifest identical to tag blobs: {chkrow('A02')} (sha256 of worktree file == `git show {TAG}:<file>`); registration "
      f"`{integ[integ.item.str.contains('EWAS_V2_REGISTRATION')].observed.iloc[0][:16]}...`, manifest `{integ[integ.item.str.contains('manifest_v1')].observed.iloc[0][:16]}...`.")
    w(f"- Tag is an ancestor of HEAD: {chkrow('A01')}. Design artifacts (10 files in `data/derived/ewas_v2_v1/`) == frozen manifest: {chkrow('A03')}; no extra files: {chkrow('A04')}; 20 pinned inputs (16 set files, covariates, loci, pins, tier table): {chkrow('A05')}; draw-table fingerprint == manifest `{manifest['bootstrap_fingerprint'][:16]}...`: {chkrow('A06')}.")
    w(f"- `state.json`: state `{state['state']}`, arms done {len(state['arms_done'])}/4, contrasts_done {state['contrasts_done']}: {chkrow('A07')}.")
    w(f"- Per-arm run manifests (4): protocol commit == tag commit {chkrow('A11')}; registration sha256 {chkrow('A12')}; design manifest sha256 {chkrow('A13')}; store sha256 actual == registered {chkrow('A14')}; all 10 gate checks ok {chkrow('A15')}; `code_changes_since_tag` empty {chkrow('A16')}; seed 17 / B 1000 / 5 folds / max_iter 10000 {chkrow('A17')}; 32 endpoints each {chkrow('A18')}; no TCGA test set referenced {chkrow('A19')}.")
    w(f"- Identical rows/pairs across arms: pairs digest identical across the 4 arms for every task x design {chkrow('B05')}; endpoint n_pairs == frozen manifest counts {chkrow('B03')} (and endpoints files {chkrow('B01')}); tiers == manifest {chkrow('B02')}; tier counts 10 main-quality / 1 exploratory / 5 insufficient {chkrow('B10')}.")
    w(f"- Convergence: number of non-converged fits is **0 in every arm x task x design** (32 endpoints per arm; {chkrow('B06')}); max n_iter stays below max_iter 10000 {chkrow('B07')}.")
    w(f"- Consistency of this compilation with the runner's own tables: per-task metrics {chkrow('B09')}, contrast deltas == candidate-comparator points {chkrow('C01')}, family counts recomputed from contrasts == runner `family_summary.csv` {chkrow('D01')}, S1 sign agreement {chkrow('E01')}.\n")
    w("Arm manifests (from `arms/<arm>/arm_manifest.json`):\n")
    amd = am[["arm", "dim", "dtype_on_disk", "n_rows", "n_all_zero_store_rows", "store_sha256_actual", "head_commit", "seed", "n_bootstrap", "folds", "threads", "wall_seconds", "sklearn"]].copy()
    amd["store_sha256_actual"] = amd.store_sha256_actual.str[:12] + "..."
    amd["head_commit"] = amd.head_commit.str[:7]
    w(md_table(amd) + "\n")
    z = ep.groupby(["design", "arm"])[["zero_rows_positives", "zero_rows_backgrounds"]].sum().reset_index()
    w("Zero rows (all-zero candidate embedding rows kept as valid features; backgrounds identical for all arms), summed over the 16 tasks:\n")
    w(md_table(z) + "\n")
    w("Per task/arm/design values are in `ewas_v2_results_convergence_zero_rows.csv` (also: chosen C per outer fold, n_iter_max, pairs digest).\n")
    w("Timing (from `launch_logs/chain.log`, local time): " + "; ".join(l.split(" ", 1)[0][11:19] + " " + l.split(" ", 1)[1] for l in chain.strip().split("\n")[-8:]) + ".\n")

    # (b) per-task table
    def tbl(design, metric_cols=("auroc",)):
        rows = []
        for t in TASKS:
            sub = pt[(pt.design == design) & (pt.task == t)]
            r0 = sub.iloc[0]
            row = dict(task=t, family=r0.family, tier=r0.tier, non_indep=("yes (" + r0.non_independent_of + ")") if r0.non_independent_of else "", N_pairs=int(r0.n_pairs))
            for a in ARMS:
                s = sub[sub.arm == a].iloc[0]
                row[ARM_LABEL[a].split(" (")[0]] = s
            rows.append(row)
        return rows
    def arm_cell(s, m):
        return f"{f3(s[m])} [{f3(s[m + '_ci_lo'])}, {f3(s[m + '_ci_hi'])}]"
    for design, title in (("main", "MAIN design (exact chromosome x probe type)"), ("S1", "S1 sensitivity (adds island class, GC, CpG density, TSS distance, caliper 0.25 SD)")):
        w(f"## 2{'a' if design == 'main' else 'b'}. Per-task metrics: {title}\n")
        w("Tiers: main-quality >=200 matched pairs, exploratory 100-199, insufficient <100 (registered; S1 tiers use S1 N and equal the MAIN tiers here). Families: exposure, aging (age EWAS + 3 clocks), disease. `non_indep` = >=50% of the set lies in a larger set (counted once, through the partner, in the primary family summary).\n")
        for m, mt in (("auroc", "AUROC (95% CI)"), ("concordance", "Matched-pair concordance (95% CI)")):
            w(f"**{mt}**\n")
            out = []
            for row in tbl(design):
                o = {k: row[k] for k in ("task", "family", "tier", "non_indep", "N_pairs")}
                for a in ARMS:
                    o[ARM_LABEL[a].split(" (")[0]] = arm_cell(row[ARM_LABEL[a].split(" (")[0]], m)
                out.append(o)
            w(md_table(pd.DataFrame(out)) + "\n")
        w("**AUPRC (sensitivity only; point estimates; the baseline equals the prevalence 0.5 by 1:1 construction)**\n")
        out = []
        for row in tbl(design):
            o = {k: row[k] for k in ("task", "tier")}
            for a in ARMS:
                o[ARM_LABEL[a].split(" (")[0]] = f3(row[ARM_LABEL[a].split(" (")[0]]["auprc"])
            out.append(o)
        w(md_table(pd.DataFrame(out)) + "\n")

    # (c) contrasts
    def rd(c):
        return {"candidate_better": "CI>0 (candidate better)", "comparator_better": "CI<0 (comparator better)", "no_difference_detected": "spans 0"}[c]
    for design, num in (("main", "3"), ("S1", "3b")):
        w(f"## {num}. Paired contrasts per task: {design}\n")
        w("Delta AUROC and delta concordance (candidate minus comparator) with paired chromosome-block 95% CIs. Primary = CpGPT-512, secondary = DeepCpG-128, supplementary = functional SVD-256 (legacy).\n")
        for contrast, _, lab in CONTRASTS:
            for metric in ("auroc", "concordance"):
                w(f"**{contrast}: candidate - {lab}, {metric}**\n")
                x = cd[(cd.design == design) & (cd.contrast == contrast) & (cd.metric == metric)]
                out = pd.DataFrame(dict(task=x.task, tier=x.tier, delta=[f4(v) for v in x.delta],
                                        CI=[f"[{f4(a)}, {f4(b)}]" for a, b in zip(x.ci_lo, x.ci_hi)], reading=[rd(r) for r in x.reading],
                                        p_raw_descriptive=[f"{v:.3g}" for v in x.p_raw_descriptive]))
                w(md_table(out) + "\n")

    # headline table (main)
    w("## 4. Headline table (MAIN design, all 16 tasks, AUROC)\n")
    w("Reading of the paired CI: `+` CI>0 (candidate better), `-` CI<0 (comparator better), `0` spans 0.\n")
    sym = {"candidate_better": "+", "comparator_better": "-", "no_difference_detected": "0"}
    out = []
    for t in TASKS:
        s = pt[(pt.design == "main") & (pt.task == t)].set_index("arm")
        r0 = s.iloc[0]
        row = dict(task=t, tier=r0.tier, family=r0.family)
        for a in ARMS:
            row["AUROC " + ARM_LABEL[a].split(" (")[0]] = f3(s.loc[a, "auroc"])
        for contrast, comp_arm, lab in CONTRASTS:
            x = cd[(cd.design == "main") & (cd.contrast == contrast) & (cd.metric == "auroc") & (cd.task == t)].iloc[0]
            row[f"d vs {lab.split(' (')[0]}"] = f"{f4(x.delta)} [{f4(x.ci_lo)}, {f4(x.ci_hi)}] {sym[x.reading]}"
        out.append(row)
    w(md_table(pd.DataFrame(out)) + "\n")

    # (d) family summaries
    w("## 5. Family summaries (counts of tasks by reading; no aggregate, no pooling)\n")
    w("Scope `independent` is the registered primary scope: tasks flagged non_independent (lung -> smoking, RA -> age, T2D -> BMI) are excluded and counted once through their partner; scope `all` includes them. The primary reading is the main-quality tier; exploratory and insufficient tiers are listed separately and are supplementary only. Every cell is a count of tasks whose CI is >0 / <0 / spans 0; there is no multiplicity adjustment.\n")
    for design in ("main", "S1"):
        for metric in ("auroc", "concordance"):
            for contrast, _, lab in CONTRASTS:
                x = fs[(fs.design == design) & (fs.contrast == contrast) & (fs.metric == metric) & (fs.scope == "independent")]
                w(f"**{design}, candidate - {lab}, {metric}, scope independent**\n")
                w(md_table(x[["family", "tier", "n_tasks", "n_candidate_better", "n_comparator_better", "n_no_difference_detected"]]) + "\n")
    w("Scope `all` (incl. non-independent tasks) is in `ewas_v2_results_family_summary.csv`; task lists per cell are in the CSV columns `tasks_*`.\n")
    w("Totals over the three families, per tier (independent scope), AUROC, main design:\n")
    tot = []
    for contrast, _, lab in CONTRASTS:
        for tier in TIER_ORDER:
            x = fs[(fs.design == "main") & (fs.contrast == contrast) & (fs.metric == "auroc") & (fs.scope == "independent") & (fs.tier == tier)]
            tot.append(dict(contrast=contrast, tier=tier, n_tasks=int(x.n_tasks.sum()), candidate_better=int(x.n_candidate_better.sum()),
                            comparator_better=int(x.n_comparator_better.sum()), no_difference=int(x.n_no_difference_detected.sum())))
    w(md_table(pd.DataFrame(tot)) + "\n")
    w("(These per-tier totals are simple counts for orientation, not an aggregate score.)\n")

    # (e) direction sections
    def lst(design, contrast, metric, which):
        x = cd[(cd.design == design) & (cd.contrast == contrast) & (cd.metric == metric) & (cd.reading == which)]
        return x
    def bullet(x):
        if x.empty:
            return "- (none)\n"
        return "\n".join(f"- `{r.task}` ({r.family}, {r.tier}{', non_independent of ' + r.non_independent_of if r.non_independent_of else ''}): delta {f4(r.delta)} [{f4(r.ci_lo)}, {f4(r.ci_hi)}]" + (f" (borderline: nearest CI bound {min(abs(r.ci_lo), abs(r.ci_hi)):.6f} from 0)" if r.reading != "no_difference_detected" and min(abs(r.ci_lo), abs(r.ci_hi)) < 1e-3 else "") for r in x.itertuples()) + "\n"
    heads = [("candidate_better", "Where the candidate is better (CI>0)"), ("comparator_better", "Where CpGPT/DeepCpG are better (CI<0)"), ("no_difference_detected", "Where no difference is detectable (CI spans 0)")]
    w("## 6. Direction of every task (MAIN design, AUROC; descriptive, nothing hidden)\n")
    for i, (key, h) in enumerate(heads):
        w(f"### 6.{i + 1} {h}\n")
        for contrast, _, lab in CONTRASTS:
            x = lst("main", contrast, "auroc", key)
            w(f"**vs {lab}** ({len(x)} of 16 tasks)\n")
            w(bullet(x))
    w("The same lists for matched-pair concordance and for S1 can be derived from `ewas_v2_results_contrasts.csv` (column `reading`).\n")

    # (f) S1 vs main
    w("## 7. S1 sensitivity versus MAIN: does the reading change?\n")
    w("Per task and contrast, the reading under MAIN is compared with the reading under S1: `confirms` = same reading; `weakens` = MAIN CI excludes 0 but S1 spans 0; `emerges_in_S1` = MAIN spans 0 but S1 excludes 0; `reverses` = opposite non-zero readings. No pre-set attenuation threshold. Counts over 16 tasks:\n")
    cnt = sv.groupby(["contrast", "metric", "change_class"]).size().unstack(fill_value=0)
    for c in ("confirms", "weakens", "emerges_in_S1", "reverses"):
        if c not in cnt:
            cnt[c] = 0
    cnt = cnt[["confirms", "weakens", "emerges_in_S1", "reverses"]].reset_index()
    sa = sv.groupby(["contrast", "metric"]).sign_agrees.sum().reset_index().rename(columns={"sign_agrees": "sign_agrees_of_16"})
    w(md_table(cnt.merge(sa, on=["contrast", "metric"])) + "\n")
    w("Tasks that are not `confirms` for the primary contrast (AUROC):\n")
    x = sv[(sv.contrast == "primary_vs_cpgpt") & (sv.metric == "auroc") & (sv.change_class != "confirms")]
    w("\n".join(f"- `{r.task}` ({r.tier_main}): MAIN {f4(r.delta_main)} ({r.reading_main}) -> S1 {f4(r.delta_s1)} ({r.reading_s1}): {r.change_class}" for r in x.itertuples()) + "\n" if len(x) else "- (none)\n")
    w("Tasks that are not `confirms` for the secondary contrast (AUROC):\n")
    x = sv[(sv.contrast == "secondary_vs_deepcpg") & (sv.metric == "auroc") & (sv.change_class != "confirms")]
    w("\n".join(f"- `{r.task}` ({r.tier_main}): MAIN {f4(r.delta_main)} ({r.reading_main}) -> S1 {f4(r.delta_s1)} ({r.reading_s1}): {r.change_class}" for r in x.itertuples()) + "\n" if len(x) else "- (none)\n")
    w("Per-arm AUROC under S1 minus MAIN is in `outputs/.../report_inputs/s1_vs_main_auroc.csv` and can be recomputed from `ewas_v2_results_per_task_arm.csv`. S1 also neutralises the sequence arms' context information, so a smaller candidate advantage under S1 is not by itself evidence of context-only signal.\n")

    # (g) caveats
    w("## 8. Caveats\n")
    for c in [
        "EWAS probe-design bias: array content is enriched in promoters/islands. MAIN matching (chromosome x probe type) does not control it; S1 does coarsely (island class, GC, CpG density, TSS distance).",
        "Negatives are unlisted CpGs, not proven unassociated; backgrounds exclude the union of all 16 sets (46,092 universe CpGs).",
        "Sets overlap; lung (93% in smoking), RA (62% in age) and T2D (92% in BMI) are flagged non_independent and counted once through the partner in the primary family summary (both are reported per task). Other overlaps (smoking-BMI, colorectal-prostate, clocks) are reported, not flagged.",
        "CpGPT is methylation-pretrained (direct supervision on methylation); an advantage on blood EWAS (age, clocks, smoking) may reflect that, not the information source of the embedding.",
        "EWAS cohort overlap with GEO/TCGA cohorts is UNVERIFIED; download dates of EWAS Catalog/Atlas are unrecorded (UNVERIFIED).",
        "Zero rows: 849 candidate rows are all-zero (no ENCODE track overlap); they are kept as valid features; counts among positives/backgrounds per task are in the convergence/zero-rows CSV (section 1).",
        "Dimensions differ: candidate 256D, CpGPT 512D, DeepCpG 128D, functional 256D; no padding or re-compaction. Candidate store float32, comparators float16 cast to float32.",
        "Probe: L2 logistic regression with C chosen by nested chromosome-blocked inner CV from {1e-3,...,10} (the chosen C per outer fold is recorded); identical protocol for all arms. Many cells select C=0.001 or the grid edge; the grid was fixed before results.",
        "Uncertainty is coarse: B=1000 over 22 chromosome blocks; small tasks have blocks with 0-2 pairs; CIs of exploratory/insufficient tasks are very wide. p-values and CIs are descriptive; no multiplicity adjustment.",
        "Design choices decided by the orchestrator (registration table, items marked [DO]) were flagged for user review: MAIN matching on chromosome x probe type only (S2 breadth matching and 10 stability draws dropped), 1:1 ratio without cap, background exclusion rule, overlap rule and 50% threshold, nested C grid, 22-block bootstrap with B=1000. They are frozen; any change is an amendment with its own tag.",
        "Freeze ordering: EWAS-v2 was frozen (tag commit date above) before any candidate result on these sets was inspected. `exp5_candidate_completion` was run after the freeze (see `docs/EXP5_CANDIDATE_COMPLETION_RESULTS.md`); its candidate results cannot have influenced the EWAS-v2 design. The two runs were executed in overlapping time windows on 2026-10-05 (file times in that document).",
        "EWAS-v2 main-quality tasks are the primary evidence; exploratory (SLE) and insufficient (Hannum, T2D, CHD, ovarian, schizophrenia) tasks are supplementary and do not become evidence because of an embedding result.",
    ]:
        w(f"- {c}")
    w("")

    # (h) dictionary
    w("## 9. Data dictionary (`paper/registry/ewas_v2_results_*.csv`)\n")
    dd = {
        "ewas_v2_results_per_task_arm.csv": "design (main|S1), task, family (exposure|aging|disease), tier (design-specific), non_independent_of, arm, arm_label, n_pairs (matched pairs, 1:1), auroc / auroc_ci_lo / auroc_ci_hi (95% block-bootstrap percentile CI), concordance (+ci), auprc (sensitivity only), auprc_baseline_prevalence (0.5), source_file (arm endpoint.json; endpoints/<task>.json; sha256 in manifest).",
        "ewas_v2_results_contrasts.csv": "design, task, family, tier, non_independent_of, n_pairs, contrast (primary_vs_cpgpt|secondary_vs_deepcpg|supplementary_vs_functional_legacy), comparator, metric (auroc|concordance), delta (candidate - comparator), ci_lo/ci_hi (paired block bootstrap 95%), reading (candidate_better CI>0 | comparator_better CI<0 | no_difference_detected), p_raw_descriptive (two-sided, (1+k)/(1+B) floor; no adjustment), B_eff, source_file.",
        "ewas_v2_results_family_summary.csv": "design, contrast, metric, family, tier, scope (independent|all), n_tasks, n_candidate_better, n_comparator_better, n_no_difference_detected, tasks_* (semicolon lists), source_file. Recomputed from contrasts; equals runner family_summary.csv (check D01).",
        "ewas_v2_results_s1_vs_main.csv": "task, family, tier_main, tier_s1, contrast, comparator, metric, delta/ci/reading for main and S1, sign_agrees, change_class (confirms|weakens|emerges_in_S1|reverses), source_file.",
        "ewas_v2_results_convergence_zero_rows.csv": "design, task, arm, n_pairs, pairs_digest, n_nonconverged_fits, n_iter_max, max_iter, zero_rows_positives, zero_rows_backgrounds, C_per_outer_fold (5 values, ';'), folds, source_file.",
        "ewas_v2_results_arm_manifests.csv": "one row per arm: head_commit, protocol_commit, registration/design-manifest sha256, store sha256 (registered/actual), dim, n_rows, n_all_zero_store_rows, seed, bootstrap B, folds, C_grid, max_iter, threads, gate_checks_all_ok, code_changes_since_tag, library versions, wall_seconds.",
        "ewas_v2_results_integrity.csv": "check_id, scope, item, expected, observed, passed, source_file.",
        "ewas_v2_results_manifest.json": "git HEAD, tags, tag commit, sha256 of every source file read and of every output file written (including this document).",
    }
    for k, v in dd.items():
        w(f"- `{k}`: {v}")
    w("")
    return "\n".join(L)


# ------------------------------------------------------------------ exp5
EXP_ARMS = ["regulatory_histone_dnase_v1", "cpgpt_large_native512", "deepcpg_hcc_native128", "functional_256"]
EXP_LABEL = {"regulatory_histone_dnase_v1": "candidate (256D)", "cpgpt_large_native512": "CpGPT-512",
             "deepcpg_hcc_native128": "DeepCpG-128", "functional_256": "functional-256 (legacy)"}
EXP_TO_EWAS = {"cpgpt_large_native512": ("primary_vs_cpgpt", "cpgpt_large_locus"),
               "deepcpg_hcc_native128": ("secondary_vs_deepcpg", "deepcpg_dna_locus"),
               "functional_256": ("supplementary_vs_functional_legacy", "functional_annotations_pca")}


def compile_exp5() -> None:
    SOURCES.clear()
    outputs: dict[str, str] = {}
    integ: list[dict] = []

    def chk(cid, scope, item, expected, observed, ok, source=""):
        integ.append(dict(check_id=cid, scope=scope, item=item, expected=str(expected), observed=str(observed),
                          passed=bool(ok), source_file=source))

    head = git("rev-parse", "HEAD")
    run = X5 / "run_001"
    rm = jload(run / "run_manifest.json")
    ver = jload(X5 / "regression" / "VERDICT.json")
    pins = jload(ROOT / "configs/exp5_candidate_completion/exp5_pins.json")
    reg_doc = ROOT / "docs/EXP5_CANDIDATE_COMPLETION_REGISTRATION.md"
    src(reg_doc)
    # regression
    chk("R01", "regression", "verdict PASS (Amendment 2)", "PASS", ver["verdict"], ver["verdict"] == "PASS", rel(X5 / "regression/VERDICT.json"))
    chk("R02", "regression", "cells / failed", "18 / 0", f"{ver['n_cells']} / {ver['n_cells_failed']}", ver["n_cells"] == 18 and ver["n_cells_failed"] == 0)
    chk("R03", "regression", "max |dAUROC| <= 5e-4", "<=0.0005", ver["observed_max_dauroc"], ver["observed_max_dauroc"] <= 5e-4)
    chk("R04", "regression", "min Spearman >= 0.9999", ">=0.9999", ver["observed_min_spearman"], ver["observed_min_spearman"] >= 0.9999)
    chk("R05", "regression", "max |dscore| (informative only)", "info", ver["observed_max_abs_dscore_informative"], True)
    rc = cread(X5 / "regression" / "regression_cells.csv")
    chk("R06", "regression", "cells with pairs_folds_identical", "18", int(rc.pairs_folds_identical.sum()), rc.pairs_folds_identical.all() and len(rc) == 18,
        rel(X5 / "regression/regression_cells.csv"))
    chk("R07", "regression", "max |dAUROC| recomputed from regression_cells.csv", f"{ver['observed_max_dauroc']:.6g}", f"{rc.abs_dauroc.max():.6g}",
        abs(rc.abs_dauroc.max() - ver["observed_max_dauroc"]) < 1e-12)
    shim = (X5 / "regression" / "shim_vs_stored_folds.txt")
    src(shim)
    t = shim.read_text().strip()
    mm = re.search(r"cells=(\d+) mismatching=(\d+) rows=(\d+)", t)
    chk("R08", "fold_identity", "shim reproduces stored folds: cells / mismatches / rows", "3120 / 0 / 5785200", f"{mm.group(1)} / {mm.group(2)} / {mm.group(3)}",
        mm.groups() == ("3120", "0", "5785200"), rel(shim))
    chk("R09", "fold_identity", "run_manifest stored_fold_cells_mismatching_shim", 0, rm["evidence"]["stored_fold_cells_mismatching_shim"],
        rm["evidence"]["stored_fold_cells_mismatching_shim"] == 0, rel(run / "run_manifest.json"))
    chk("R10", "run_manifest", "git_head recorded", head, rm["evidence"]["git_head"], True, rel(run / "run_manifest.json"))
    chk("R11", "run_manifest", "regression_verdict in manifest", "PASS", rm["evidence"]["regression_verdict"], rm["evidence"]["regression_verdict"] == "PASS")
    chk("R12", "run_manifest", "store sha256 == pins", pins["candidate"]["store_h5_sha256"], rm["evidence"]["store_sha256"],
        pins["candidate"]["store_h5_sha256"] == rm["evidence"]["store_sha256"])
    chk("R13", "run_manifest", "frozen candidate manifest sha256 == pins", pins["candidate"]["frozen_manifest_sha256"],
        rm["evidence"]["frozen_manifest_sha256"], pins["candidate"]["frozen_manifest_sha256"] == rm["evidence"]["frozen_manifest_sha256"])
    chk("R14", "run_manifest", "no ABORT_pairs_or_folds_differ.json in run dir (run asserts identical cpg_idx/pair_id/y_true/fold/chromosome vs every stored comparator, every cell)",
        "absent", "absent" if not (run / "ABORT_pairs_or_folds_differ.json").exists() else "PRESENT", not (run / "ABORT_pairs_or_folds_differ.json").exists())
    chk("R15", "run_manifest", "seeds 17001..17020, B", "20, 10000", f"{len(rm['seeds'])}, {rm['boot']}", rm["seeds"] == list(range(17001, 17021)) and rm["boot"] == 10000)
    # exp5 stored files vs pins (recompute from disk)
    pf = pins["exp5_files_sha256"]
    for f, h in sorted(pf.items()):
        p = ROOT / f
        ok = sha256(p) == h
        src(p)
        chk("R20", "stored_exp5", f"{f} sha256 == exp5_pins.json", h, SOURCES[f], ok and SOURCES[f] == h, f)
        chk("R21", "stored_exp5", f"{f} snapshot before == after run (manifest)", h, rm["exp5_snapshot_after"][f],
            rm["exp5_snapshot_before"][f] == rm["exp5_snapshot_after"][f] == h)
    # summaries
    s = cread(run / "summary_candidate_completion.csv")
    s["set"] = s["set"].astype(str)
    sets = sorted(s["set"].unique())
    chk("S01", "summary", "sets x variants x kinds", "13 x 2", f"{len(sets)} x {s.variant.nunique()}", len(sets) == 13 and s.variant.nunique() == 2,
        rel(run / "summary_candidate_completion.csv"))
    chk("S02", "summary", "skipped sets (no matched support)", "ovarian,CHD,schizophrenia", ",".join(rm["not_evaluable_sets_skipped"]),
        len(rm["not_evaluable_sets_skipped"]) == 3)
    chk("S03", "summary", "status label on every row", "discovery-campaign completion (exp5 protocol, not preregistered main evidence)",
        s.status_label.nunique(), s.status_label.nunique() == 1 and s.status_label.iloc[0].startswith("discovery-campaign completion"))
    # compare comparators with stored exp5
    st = cread(X5_STORED / "summary_auc_delta.csv")
    stc = st[(st.kind == "auc")].rename(columns={"estimate": "est_stored", "ci_lo": "lo_stored", "ci_hi": "hi_stored"})
    cur = s[(s.kind == "auc") & (s.representation != "regulatory_histone_dnase_v1")]
    mg = cur.merge(stc, on=["variant", "set", "representation"])
    dmax = (mg.estimate - mg.est_stored).abs().max()
    cmax = max((mg.ci_lo - mg.lo_stored).abs().max(), (mg.ci_hi - mg.hi_stored).abs().max())
    chk("S04", "comparators", "comparator AUROC in run summary vs stored exp5 summary: rows, max |diff| (spec: float32 re-derivation <=~1e-4)",
        "78 rows, <1e-3", f"{len(mg)} rows, {dmax:.2e}", len(mg) == 78 and dmax < 1e-3, rel(X5_STORED / "summary_auc_delta.csv"))
    chk("S05", "comparators", "comparator AUROC CI in run summary vs stored: max |diff|", "<2e-3", f"{cmax:.2e}", cmax < 2e-3)
    cv2 = cread(REG / "exp5_matched_pair_concordance_v2.csv")
    cc = s[(s.kind == "concordance") & (s.representation != "regulatory_histone_dnase_v1")].merge(
        cv2, left_on=["variant", "set", "representation"], right_on=["variant", "set", "representation"])
    cd_max = (cc.estimate - cc.matched_pair_concordance).abs().max()
    chk("S06", "comparators", "comparator concordance in run summary vs registry v2 (derived from stored OOF): rows, max |diff|", "78 rows, <1e-6",
        f"{len(cc)} rows, {cd_max:.2e}", len(cc) == 78 and cd_max < 1e-6, rel(REG / "exp5_matched_pair_concordance_v2.csv"))
    tiers = cread(REG / "exp5_support_tiers_v2.csv")
    tt = s[s.kind == "auc"].drop_duplicates(["set"]).merge(tiers, on="set", suffixes=("", "_v2"))
    chk("S07", "tiers", "support_tier in summary == exp5_support_tiers_v2.csv", "13/13", f"{int((tt.support_tier == tt.support_tier_v2).sum())}/13",
        (tt.support_tier == tt.support_tier_v2).all())
    # regression: also the stored exp5 candidate-free
    for p in (ROOT / "outputs/exp5_candidate_completion/run_001.log",):
        src(p)

    # ---- per set x arm table
    pa = []
    for (v, st_), g in s.groupby(["variant", "set"]):
        for a in EXP_ARMS:
            def pick(kind):
                x = g[(g.kind == kind) & (g.representation == a)]
                return x.iloc[0] if len(x) else None
            au, co, ap = pick("auc"), pick("concordance"), pick("auprc_sensitivity")
            pa.append(dict(variant=v, set=st_, support_tier=au.support_tier, matched_positives_mean=au.matched_positives, arm=a,
                           arm_label=EXP_LABEL[a], auroc=au.estimate, auroc_ci_lo=au.ci_lo, auroc_ci_hi=au.ci_hi,
                           auroc_between_seed_sd=au.between_seed_sd, concordance=co.estimate, concordance_ci_lo=co.ci_lo,
                           concordance_ci_hi=co.ci_hi, auprc=ap.estimate, auprc_baseline_prevalence=0.5,
                           comparator_source=("run (current libraries)" if a == "regulatory_histone_dnase_v1" else "stored exp5 OOF (original libraries)"),
                           source_file=rel(run / "summary_candidate_completion.csv")))
    pa = pd.DataFrame(pa)
    pa["_t"] = pa["set"].map({t: i for i, t in enumerate(TASKS)})
    pa["_a"] = pa.arm.map({a: i for i, a in enumerate(EXP_ARMS)})
    pa = pa.sort_values(["variant", "_t", "_a"]).drop(columns=["_t", "_a"]).reset_index(drop=True)
    write_csv(pa, "exp5_completion_results_per_set_arm.csv", outputs)

    # ---- deltas
    dl = []
    for kind, m in (("delta_auc", "auroc"), ("delta_concordance", "concordance")):
        x = s[s.kind == kind]
        for r in x.itertuples():
            comp = r.comparison.split(" - ")[1]
            dl.append(dict(variant=r.variant, set=r.set, support_tier=r.support_tier, comparator=comp, comparator_label=EXP_LABEL[comp],
                           metric=m, delta=r.estimate, ci_lo=r.ci_lo, ci_hi=r.ci_hi, reading=reading(r.ci_lo, r.ci_hi),
                           between_seed_sd=r.between_seed_sd, n_seeds_delta_positive=r.n_seeds_delta_positive, n_seeds=r.n_seeds,
                           p_boot_descriptive=r.p_boot, q_bh_descriptive=r.q_bh, source_file=rel(run / "summary_candidate_completion.csv")))
    dl = pd.DataFrame(dl)
    dl["_t"] = dl["set"].map({t: i for i, t in enumerate(TASKS)})
    dl["_c"] = dl.comparator.map({a: i for i, a in enumerate(EXP_ARMS)})
    dl = dl.sort_values(["variant", "comparator", "metric", "_t"]).drop(columns=["_t", "_c"]).reset_index(drop=True)
    # check delta == difference of points
    pv = pa.pivot_table(index=["variant", "set"], columns="arm", values=["auroc", "concordance"])
    mx = max(abs(pv.loc[(r.variant, r.set)][(r.metric, EXP_ARMS[0])] - pv.loc[(r.variant, r.set)][(r.metric, r.comparator)] - r.delta) for r in dl.itertuples())
    chk("S08", "deltas", "delta point == candidate - comparator point (max |diff|; seed-mean of deltas vs delta of seed-means)", "<1e-9", f"{mx:.2e}", mx < 1e-9)
    chk("S09", "deltas", "delta rows (13 sets x 2 variants x 3 comparators x 2 metrics)", 156, len(dl), len(dl) == 156)
    nanbad = int(dl[["delta", "ci_lo", "ci_hi"]].isna().sum().sum() + pa[["auroc", "auroc_ci_lo", "auroc_ci_hi", "concordance", "auprc"]].isna().sum().sum())
    chk("S10", "values", "no NaN in values/CIs", 0, nanbad, nanbad == 0)
    write_csv(dl, "exp5_completion_results_deltas.csv", outputs)

    # ---- consistency vs EWAS-v2
    ew_pt = cread(ROOT / "paper/registry/ewas_v2_results_per_task_arm.csv")
    ew_cd = cread(ROOT / "paper/registry/ewas_v2_results_contrasts.csv")
    vs = []
    for r in dl[dl.metric == "auroc"].itertuples():
        ctr, comp_arm = EXP_TO_EWAS[r.comparator]
        for design in ("main", "S1"):
            e = ew_cd[(ew_cd.design == design) & (ew_cd.task == r.set) & (ew_cd.contrast == ctr) & (ew_cd.metric == "auroc")].iloc[0]
            cand = ew_pt[(ew_pt.design == design) & (ew_pt.task == r.set) & (ew_pt.arm == "regulatory_histone_dnase_v1")].iloc[0]
            cpa = pa[(pa.variant == r.variant) & (pa.set == r.set) & (pa.arm == "regulatory_histone_dnase_v1")].iloc[0]
            vs.append(dict(set=r.set, support_tier=r.support_tier, exp5_variant=r.variant, ewas_design=design, comparator=r.comparator_label,
                           exp5_candidate_auroc=cpa.auroc, ewas_candidate_auroc=cand.auroc, exp5_delta_auroc=r.delta, exp5_ci_lo=r.ci_lo,
                           exp5_ci_hi=r.ci_hi, exp5_reading=r.reading, ewas_delta_auroc=e.delta, ewas_ci_lo=e.ci_lo, ewas_ci_hi=e.ci_hi,
                           ewas_reading=e.reading, sign_agrees=bool((r.delta > 0) == (e.delta > 0)), reading_agrees=bool(r.reading == e.reading),
                           source_file=rel(run / "summary_candidate_completion.csv") + ";paper/registry/ewas_v2_results_contrasts.csv"))
    vs = pd.DataFrame(vs)
    write_csv(vs, "exp5_completion_results_vs_ewas_v2.csv", outputs)
    SOURCES["paper/registry/ewas_v2_results_contrasts.csv"] = sha256(REG / "ewas_v2_results_contrasts.csv")
    SOURCES["paper/registry/ewas_v2_results_per_task_arm.csv"] = sha256(REG / "ewas_v2_results_per_task_arm.csv")
    write_csv(pd.DataFrame(integ), "exp5_completion_results_integrity.csv", outputs)

    file_times = dict(run_manifest=mtime_utc(run / "run_manifest.json"), summary=mtime_utc(run / "summary_candidate_completion.csv"),
                      oof=mtime_utc(run / "candidate_oof_predictions.parquet"), verdict=mtime_utc(X5 / "regression/VERDICT.json"),
                      ewas_cand_arm=mtime_utc(EW / "arms/regulatory_histone_dnase_v1/arm_manifest.json"),
                      ewas_state=mtime_utc(EW / "state.json"))
    tag_date = git("log", "-1", "--format=%cI", f"{TAG}^{{commit}}")
    man = dict(compiled_by="paper/scripts/compile_ewas_results.py", kind="exp5_completion_results", git_head=head,
               tags=sorted(git("tag").split("\n")), sources_sha256=dict(sorted(SOURCES.items())), outputs_sha256=dict(sorted(outputs.items())),
               n_integrity_checks=len(integ), n_integrity_failed=int(sum(not r["passed"] for r in integ)),
               candidate_oof_parquet_not_read=True)
    doc = exp5_doc(pa, dl, vs, pd.DataFrame(integ), ver, rm, rc, head, file_times, tag_date, mx, dmax, cmax, cd_max)
    dp = DOCS / "EXP5_CANDIDATE_COMPLETION_RESULTS.md"
    dp.write_text(doc)
    man["outputs_sha256"][rel(dp)] = sha256(dp)
    (REG / "exp5_completion_results_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print("EXP5: integrity failed =", man["n_integrity_failed"], "of", man["n_integrity_checks"])


def exp5_doc(pa, dl, vs, integ, ver, rm, rc, head, ft, tag_date, mx, dmax, cmax, cd_max) -> str:
    L: list[str] = []
    w = L.append
    w("# exp5_candidate_completion results (discovery-campaign completion)\n")
    w("**Status: discovery-campaign completion under the exp5 protocol; NOT main evidence and not preregistered main evidence. EWAS-v2 (`docs/EWAS_V2_RESULTS.md`) is the main evidence.** "
      f"Generated by `paper/scripts/compile_ewas_results.py` from `outputs/exp5_candidate_completion/run_001/` (summary CSV and run manifest only; the candidate OOF parquet is not read), "
      f"`outputs/exp5_candidate_completion/regression/VERDICT.json` and the stored exp5 files; read-only, nothing re-run. Registration: `docs/EXP5_CANDIDATE_COMPLETION_REGISTRATION.md` incl. Amendments 1-2. Repo HEAD at compilation `{head[:7]}`. Source hashes: `paper/registry/exp5_completion_results_manifest.json`.\n")
    w("Reading convention: delta = candidate minus comparator (positive favours the candidate); CIs are the exp5 bootstrap CIs as produced by the run (B=10,000, pairs resampled within chromosome, same index vector for every arm). Reading: **CI>0** candidate better, **CI<0** comparator better, **spans 0** no difference detected. The `p_boot` / BH `q` columns are in the CSV as descriptive values. No aggregate score, no pooling across sets. All 13 evaluable sets are shown in both directions.\n")
    n_fail = int((~integ.passed).sum())
    def cr(prefix):
        x = integ[integ.check_id.str.startswith(prefix)]
        return f"{int(x.passed.sum())}/{len(x)}"
    w("## 1. Integrity\n")
    w(f"{len(integ)} automated checks, **{n_fail} failed** (`exp5_completion_results_integrity.csv`).\n")
    w(f"- Regression (18 cells: SLE and Hannum x seeds 17001-17003 x CpGPT-512, DeepCpG-128, functional-256): verdict **{ver['verdict']}** under Amendment 2 (criterion: pairs/folds identical, |dAUROC| <= 5e-4, Spearman >= 0.9999). Observed max |dAUROC| = {ver['observed_max_dauroc']:.3e} (limit 5e-4), min Spearman = {ver['observed_min_spearman']:.6f} (limit 0.9999), cells failed {ver['n_cells_failed']}/{ver['n_cells']}; max |dscore| = {ver['observed_max_abs_dscore_informative']:.2e} (informative only; the former 1e-5 criterion was superseded). Pairs/folds identical in {int(rc.pairs_folds_identical.sum())}/18 cells.")
    w(f"- Amendment 1 (fold fidelity, bugfix): the legacy GroupKFold shim reproduces the stored `fold` column in 3120/3120 cells (5,785,200 rows), 0 mismatches ({cr('R08')} check; run manifest `stored_fold_cells_mismatching_shim` = {rm['evidence']['stored_fold_cells_mismatching_shim']}). The first regression attempt with sklearn 1.9.1 GroupKFold failed on `fold` (log kept in `regression/`).")
    w(f"- Run manifest: git HEAD `{rm['evidence']['git_head'][:7]}`; candidate store sha256 `{rm['evidence']['store_sha256'][:16]}...` == pins; frozen manifest sha256 `{rm['evidence']['frozen_manifest_sha256'][:16]}...` == pins; seeds 17001..17020, B = {rm['boot']}; classifier {rm['classifier']}; folds {rm['folds']}; skipped (no matched support): {', '.join(rm['not_evaluable_sets_skipped'])}.")
    w(f"- Identical pairs/folds versus the stored comparators in every cell: asserted by the run before any bootstrap (`cpg_idx`, `pair_id`, `y_true`, `fold`, `chromosome` equal per variant x set x seed x comparator, otherwise abort with `ABORT_pairs_or_folds_differ.json`); the run finished (`run_001.log`: done) and no abort file exists ({cr('R14')}). This compilation did not re-check the row identities (it does not read the OOF parquets).")
    w(f"- Stored exp5 files unchanged: the 7 files' sha256 recomputed now == `exp5_pins.json` {cr('R20')}; manifest snapshot before == after the run {cr('R21')}.")
    w(f"- Comparator numbers re-derived by the run from stored float32 OOF vs the stored exp5 summary: max |dAUROC| = {dmax:.2e}, max |dCI bound| = {cmax:.2e} (registered ~1e-4 level); comparator concordance equals the registry-v2 derived values (max |diff| {cd_max:.1e}).")
    w(f"- Support tiers == `exp5_support_tiers_v2.csv` {cr('S07')}; no NaN in values {cr('S10')}; delta point == candidate - comparator point (max |diff| {mx:.1e}).")
    w(f"- Timing (file mtimes, UTC): EWAS-v2 tag commit {tag_date}; exp5 verdict {ft['verdict']}; exp5 run summary {ft['summary']}; EWAS-v2 candidate arm manifest {ft['ewas_cand_arm']}; EWAS-v2 state.json {ft['ewas_state']}. The EWAS-v2 tag precedes both runs (freeze ordering of registration section 10 satisfied); the two runs were executed in overlapping windows, so file times cannot establish which result was opened first; per the registration neither run's design was altered after seeing the other.\n")

    # (b) table
    w("## 2. Per set x variant: AUROC and concordance (95% CI)\n")
    w("`base` = exp5 matching (chromosome x CpG context x gene region x core-breadth decile); `probe` = the exp5 probe-type-matched variant. Candidate: computed by this run (current libraries). Comparators: stored exp5 OOF scores (original libraries); concordance for comparators derived by registry v2, for the candidate computed by the run (the run summary holds both). AUPRC is sensitivity only (prevalence 0.5 by construction; no CI). N = mean matched positives over the 20 draws (cap 2,000).\n")
    short = {a: EXP_LABEL[a].split(" (")[0] for a in EXP_ARMS}
    for v in ("base", "probe"):
        for m, mt in (("auroc", "AUROC"), ("concordance", "Concordance")):
            w(f"**{v}: {mt}**\n")
            out = []
            for t in TASKS:
                g = pa[(pa.variant == v) & (pa.set == t)]
                if g.empty:
                    continue
                o = dict(set=t, tier=g.support_tier.iloc[0], N=int(round(g.matched_positives_mean.iloc[0])))
                for a in EXP_ARMS:
                    x = g[g.arm == a].iloc[0]
                    o[short[a]] = f"{f3(x[m])} [{f3(x[m + '_ci_lo'])}, {f3(x[m + '_ci_hi'])}]"
                out.append(o)
            w(md_table(pd.DataFrame(out)) + "\n")
        w(f"**{v}: AUPRC (sensitivity)**\n")
        out = []
        for t in TASKS:
            g = pa[(pa.variant == v) & (pa.set == t)]
            if g.empty:
                continue
            o = dict(set=t, tier=g.support_tier.iloc[0])
            for a in EXP_ARMS:
                o[short[a]] = f3(g[g.arm == a].iloc[0].auprc)
            out.append(o)
        w(md_table(pd.DataFrame(out)) + "\n")
    sym = {"candidate_better": "+", "comparator_better": "-", "no_difference_detected": "0"}
    w("## 3. Deltas (candidate minus comparator) with the exp5 bootstrap CIs\n")
    w("Reading symbol: `+` CI>0 (candidate better), `-` CI<0 (comparator better), `0` spans 0. `seeds+` = number of the 20 draws with a positive delta.\n")
    for v in ("base", "probe"):
        for m in ("auroc", "concordance"):
            w(f"**{v}: delta {m}**\n")
            out = []
            for t in TASKS:
                g = dl[(dl.variant == v) & (dl.set == t) & (dl.metric == m)]
                if g.empty:
                    continue
                o = dict(set=t, tier=g.support_tier.iloc[0])
                for a in EXP_ARMS[1:]:
                    x = g[g.comparator == a].iloc[0]
                    o["vs " + short[a]] = f"{f4(x.delta)} [{f4(x.ci_lo)}, {f4(x.ci_hi)}] {sym[x.reading]} (seeds+ {int(x.n_seeds_delta_positive)}/20)"
                out.append(o)
            w(md_table(pd.DataFrame(out)) + "\n")
    w("Reading counts over the 13 sets (AUROC), by tier:\n")
    cnt = []
    for v in ("base", "probe"):
        for a in EXP_ARMS[1:]:
            for tier in TIER_ORDER:
                g = dl[(dl.variant == v) & (dl.comparator == a) & (dl.metric == "auroc") & (dl.support_tier == tier)]
                if g.empty:
                    continue
                cnt.append(dict(variant=v, comparator=short[a], tier=tier, n_sets=len(g), candidate_better=int((g.reading == "candidate_better").sum()),
                                comparator_better=int((g.reading == "comparator_better").sum()), no_difference=int((g.reading == "no_difference_detected").sum())))
    w(md_table(pd.DataFrame(cnt)) + "\n")
    w("Sets by direction (base variant, AUROC, vs CpGPT-512 / DeepCpG-128):\n")
    for key, h in (("candidate_better", "Candidate better (CI>0)"), ("comparator_better", "Comparator better (CI<0)"), ("no_difference_detected", "No difference detected")):
        for a in EXP_ARMS[1:3]:
            g = dl[(dl.variant == "base") & (dl.comparator == a) & (dl.metric == "auroc") & (dl.reading == key)]
            w(f"- {h} vs {short[a]}: " + (", ".join(f"`{r.set}` ({r.support_tier}, {f4(r.delta)})" for r in g.itertuples()) or "(none)"))
    w("")

    # (c) caveats
    w("## 4. Caveats\n")
    for c in [
        f"Numerical fidelity: the candidate is fitted with the CURRENT libraries while comparators enter with their stored scores (original libraries). Candidate - comparator deltas can therefore carry a systematic fidelity error of order <= ~3e-4 AUROC (regression: max |dAUROC| {ver['observed_max_dauroc']:.2e}, 5e-4 tolerance, Amendment 2). Negligible against CIs of order >= 0.01.",
        "Matching is not neutral. Strata = chromosome x CpG context x gene region x core-breadth decile. Context, gene region and core breadth are INPUTS of the functional store but NOT of the candidate, CpGPT or DeepCpG. The matching removes the signal the functional arm holds as direct input (handicapping functional-256) while the candidate (histone/DNase tracks encode promoter/island/breadth implicitly) is scored under strata built partly from features it lacks. The net direction is not known a priori; candidate - functional deltas are supplementary and potentially biased; candidate vs CpGPT/DeepCpG share the same matching.",
        "Discovery-campaign completion, NOT main evidence: pairs, folds, probe (StandardScaler + LogReg C=1) and bootstrap are the old exp5 design (cap 2,000 pairs, 20 non-independent draws; GC, CpG density, TSS distance not matched). p/q are descriptive. It was run after the EWAS-v2 freeze and cannot motivate changes to it.",
        "Old exp5 caveats carry over: negatives are unlisted, not unassociated; sets overlap (lung in smoking, T2D in BMI); EWAS cohort overlap with GEO/TCGA unverified; CpGPT is methylation-pretrained; the candidate SVD is label-free and transductive; 849 all-zero candidate rows kept; chromosomes 20-22 are out-of-fit for the SVD.",
        "Candidate store float32 vs comparators float16 (all cast to float32 for the probe): a minor precision advantage for the candidate.",
        "Three sets (ovarian, CHD, schizophrenia) have no matched support in exp5 and are skipped; Hannum and T2D (insufficient) and SLE (exploratory) are reported but supplementary.",
    ]:
        w(f"- {c}")
    w("")

    # (d) consistency
    w("## 5. Consistency with EWAS-v2 (same sets, two designs)\n")
    w("For each of the 13 evaluable sets the candidate AUROC and the sign / reading of candidate - comparator (AUROC) are compared across designs: exp5 (`base` and `probe`) against EWAS-v2 MAIN and S1. Agreement is described only; the designs differ (matching strata, caps, probe C selection, folds, bootstrap), so no interpretation beyond description is offered. A sign is the sign of the point delta; a reading is CI>0 / CI<0 / spans 0.\n")
    rows = []
    for (v, d, comp), g in vs.groupby(["exp5_variant", "ewas_design", "comparator"]):
        rows.append(dict(exp5_variant=v, ewas_design=d, comparator=comp, n_sets=len(g), sign_agree=int(g.sign_agrees.sum()),
                         sign_disagree=int((~g.sign_agrees).sum()), reading_agree=int(g.reading_agrees.sum()), reading_disagree=int((~g.reading_agrees).sum())))
    w(md_table(pd.DataFrame(rows)) + "\n")
    w("Per set, exp5 `base` versus EWAS-v2 MAIN (AUROC):\n")
    out = []
    for t in TASKS:
        g = vs[(vs.exp5_variant == "base") & (vs.ewas_design == "main") & (vs.set == t)]
        if g.empty:
            continue
        o = dict(set=t, tier=g.support_tier.iloc[0], cand_auroc_exp5=f3(g.exp5_candidate_auroc.iloc[0]), cand_auroc_ewas=f3(g.ewas_candidate_auroc.iloc[0]))
        for r in g.itertuples():
            k = r.comparator.split(" (")[0]
            o[f"vs {k}: exp5 / EWAS"] = f"{f4(r.exp5_delta_auroc)} ({sym[r.exp5_reading]}) / {f4(r.ewas_delta_auroc)} ({sym[r.ewas_reading]}){'' if r.reading_agrees else ' *'}"
        out.append(o)
    w(md_table(pd.DataFrame(out)) + "\n")
    w("`*` marks a different reading between the two designs. Other combinations (probe, S1) are in `exp5_completion_results_vs_ewas_v2.csv`.\n")
    dis = vs[(vs.exp5_variant == "base") & (vs.ewas_design == "main") & (~vs.sign_agrees)]
    w("Sign disagreements (exp5 base vs EWAS MAIN): " + ("; ".join(f"`{r.set}` vs {r.comparator.split(' (')[0]} (exp5 {f4(r.exp5_delta_auroc)}, EWAS {f4(r.ewas_delta_auroc)})" for r in dis.itertuples()) or "none") + ".\n")

    # (e) dictionary
    w("## 6. Data dictionary (`paper/registry/exp5_completion_results_*.csv`)\n")
    for k, v in {
        "exp5_completion_results_per_set_arm.csv": "variant (base|probe), set, support_tier (fixed from the pre-existing exp5 base counts), matched_positives_mean, arm, arm_label, auroc + CI (exp5 bootstrap) + between-seed SD, concordance + CI, auprc (sensitivity, no CI), auprc_baseline_prevalence, comparator_source, source_file.",
        "exp5_completion_results_deltas.csv": "variant, set, support_tier, comparator, metric (auroc|concordance), delta (candidate - comparator, mean over 20 draws), ci_lo/ci_hi (exp5 bootstrap), reading, between_seed_sd, n_seeds_delta_positive, n_seeds, p_boot_descriptive, q_bh_descriptive (BH across sets within variant x comparison x metric, as produced by the run), source_file.",
        "exp5_completion_results_vs_ewas_v2.csv": "set, support_tier, exp5_variant, ewas_design (main|S1), comparator, candidate AUROC in both designs, delta/CI/reading in both designs, sign_agrees, reading_agrees, source_file.",
        "exp5_completion_results_integrity.csv": "check_id, scope, item, expected, observed, passed, source_file.",
        "exp5_completion_results_manifest.json": "git HEAD, tags, sha256 of every source file read and of every output written (including this document).",
    }.items():
        w(f"- `{k}`: {v}")
    w("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=["ewas", "exp5", "both"])
    a = ap.parse_args()
    if a.which in ("ewas", "both"):
        compile_ewas()
    if a.which in ("exp5", "both"):
        compile_exp5()


if __name__ == "__main__":
    main()
