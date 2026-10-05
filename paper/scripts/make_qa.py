"""QA: validates manifests/outputs, aggregates expected-vs-computed checks, doc consistency; writes QA_REPORT.md and manifests/INDEX.json."""
import json
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa: F401,F403,E402

REQ = ["plotting_script", "git_head_commit", "source_result_files", "source_data_tables", "filters_and_metrics_used", "caption_statistic_definitions", "frozen_protocol_tags", "outputs", "qa_checks"]
mans = {p.stem: json.load(open(p)) for p in sorted(MAN.glob("*.json")) if p.stem != "INDEX"}
problems = []; idx = {}
for n, m in mans.items():
    for k in REQ:
        if k not in m: problems.append(f"{n}: missing manifest field {k}")
    if m["status"] != "PLACEHOLDER" and not m["source_result_files"] and not n.startswith("fig5"):
        problems.append(f"{n}: no source result files")
    outs = {}
    for o in m["outputs"]:
        p = ROOT / o
        if not p.exists(): problems.append(f"{n}: output missing {o}")
        else: outs[o] = sha256(p)
    if m["kind"] == "figure":
        for ext in ("pdf", "svg", "png"):
            if not any(o.endswith(f".{ext}") for o in m["outputs"]): problems.append(f"{n}: no .{ext}")
    for t in m["source_data_tables"]:
        p = ROOT / t["path"]
        if not p.exists(): problems.append(f"{n}: source table missing {t['path']}")
        elif sha256(p) != t["sha256"]: problems.append(f"{n}: source table sha mismatch {t['path']}")
    for s in m["source_result_files"]:
        if not (ROOT / s["path"]).exists(): problems.append(f"{n}: source result missing {s['path']}")
        if "loyfer_failure_audit" in s["path"]: problems.append(f"{n}: FORBIDDEN audit path in sources")
    idx[n] = {"kind": m["kind"], "title": m["title"], "status": m["status"], "manifest": f"paper/manifests/{n}.json", "outputs": outs,
              "source_tables": {t["path"]: t["sha256"] for t in m["source_data_tables"]}}
(MAN / "INDEX.json").write_text(json.dumps({"git_head_commit": git_head(), "n_items": len(idx), "items": idx}, indent=1))

# ---- compile + ruff
comp = []
for f in sorted((PAPER / "scripts").glob("*.py")):
    try: py_compile.compile(str(f), doraise=True); comp.append((f.name, "ok"))
    except Exception as e: comp.append((f.name, f"FAIL {e}"))
ruff = shutil.which("ruff") or str(Path(sys.executable).parent / "ruff")
try:
    r = subprocess.run([ruff, "check", "--no-cache", "--isolated", "--select", "E9,F63,F7,F82,F401,F811,F841", str(PAPER / "scripts")], capture_output=True, text=True, timeout=120)
    ruff_out = (r.stdout + r.stderr).strip()[-1500:] or "ok"
except Exception as e:
    ruff_out = f"ruff unavailable: {e}"

# ---- doc consistency (computed value formatted at doc precision must appear in the doc)
O = ROOT / "outputs"; D = ROOT / "docs"
def rd(p): return (D / p).read_text()
doc_checks = []
def dc(doc, s, label): doc_checks.append((label, doc, s, s in rd(doc)))
cm = pd.read_csv(O / "regulatory_confirm_v1/analysis/across_seeds_summary.csv")
for ref, alt, lab in [("regulatory_histone", "regulatory_histone_dnase", "H->H+D"), ("regulatory_histone_dnase", "regulatory_clean", "H+D->Clean")]:
    v = 100 * cm[(cm.reference == ref) & (cm.alternative == alt) & (cm.metric == "mse")].mean_relative.iloc[0]
    dc("REGULATORY_CONFIRMATION_RESULTS.md", f"{v:.2f}%", f"confirm mean rel MSE {lab}")
ec = pd.read_csv(O / "external_reconstruction_v1/analysis/A_v1.2/test/primary_contrast_across_seeds.csv"); ev = pd.read_csv(O / "external_reconstruction_v1/analysis/A_v1.2/test/secondary_contrast_across_seeds.csv")
dc("EXTERNAL_RECONSTRUCTION_FINAL_REPORT.md", f"+{100*ec[(ec.mask_fraction==0.5)&(ec.metric=='mse')].mean_relative.iloc[0]:.2f}%", "external test primary rel MSE@0.5")
dc("EXTERNAL_RECONSTRUCTION_FINAL_REPORT.md", f"+{100*ev[(ev.mask_fraction==0.5)&(ev.metric=='mse')].mean_relative.iloc[0]:.2f}%", "external test secondary rel MSE@0.5")
fa = pd.read_csv(O / "regulatory_family_screen_v1/analysis/per_arm_metrics.csv").set_index("arm").mse
for a in fa.index: dc("REGULATORY_FAMILY_SCREEN_RESULTS.md", f"{fa[a]:.6f}", f"family screen MSE {a}")
sp = pd.read_csv(BV := ROOT / "outputs/biological_validation_v2/bioval-v2-protocol-freeze-v1/report/summary_primary.csv")
for _, r in sp.iterrows():
    if r.endpoint in ("microc_H1_intra10kb",): dc("BIOLOGICAL_VALIDATION_V2_FINAL_REPORT.md", f"{r.value:.4f}", f"bioval {r.endpoint} {r.arm}")
    if r.endpoint == "loyfer_profile": dc("BIOLOGICAL_VALIDATION_V2_FINAL_REPORT.md", f"{r.value:.4f}", f"bioval {r.endpoint} {r.arm}")
tp = pd.read_csv(O / "regulatory_confirmation_v1/analysis/per_run_metrics.csv"); t0 = tp[(tp.arm == "regulatory_histone_dnase") & (tp.mask_fraction == 0.5) & (tp.seed == 17)].mse.iloc[0]
dc("REGULATORY_CONFIRMATION_PHASE_A_RESULTS.md", f"{t0:.6f}", "Phase A candidate seed 17 MSE@0.5")

# ---- pre-amendment A vs A_v1.2
a_old = pd.read_csv(O / "external_reconstruction_v1/analysis/A/validation/per_run_metrics.csv"); a_new = pd.read_csv(O / "external_reconstruction_v1/analysis/A_v1.2/validation/per_run_metrics.csv")
mm = a_old.merge(a_new, on=["arm", "seed", "mask_fraction"], suffixes=("_old", "_new")); maxdiff = float((mm.mse_old - mm.mse_new).abs().max()) if len(mm) else float("nan")

allchecks = [(n, c) for n, m in mans.items() for c in m["qa_checks"]]
nfail = sum(not c["pass"] for _, c in allchecks)
L = ["# QA report (manuscript package part 1)", "", f"Generated at git HEAD `{git_head()}` (nothing committed by this package). All numbers read from frozen machine-readable result files; nothing re-run.", "",
     f"- manifests: {len(mans)}; source-data tables: {len(list(SRC.glob('*.csv')))}; figures (pdf): {len(list(FIG.glob('*.pdf')))}; tables (tex): {len(list(TAB.glob('*.tex')))}",
     f"- structural problems: {len(problems)}", *[f"  - {p}" for p in problems], f"- expected-vs-computed checks: {len(allchecks)} ({nfail} fail)", "",
     "## Expected (brief) vs computed", "", "| item | check | expected | computed | tol | pass |", "|---|---|---:|---:|---:|---|"]
for n, c in allchecks: L.append(f"| {n} | {c['label']} | {c['expected']:.6g} | {c['computed']:.6g} | {c['tol']:g} | {'PASS' if c['pass'] else 'FAIL'} |")
L += ["", "## Doc vs machine-readable consistency (computed value formatted at doc precision must appear in the doc)", "", "| label | doc | string searched | found |", "|---|---|---|---|"]
for lab, doc, s, ok in doc_checks: L.append(f"| {lab} | {doc} | `{s}` | {'yes' if ok else 'NO'} |")
L += ["", "## Inconsistencies and caveats found", "",
      "1. Rounding only: brief quotes DeepCpG H1 Micro-C AUROC 0.503 and CpGPT Loyfer 0.107; files give 0.5025 and 0.1065 (docs print 0.5025/0.1065). Differences < 0.001; no substantive disagreement.",
      "2. CpGPT-large is NOT the same object across the campaign and the later phases: the ENCODE campaign arm `fm/cpgpt_locus_large` is the compact SVD-256 variant (configs/representations/confirmation_matrix.yaml: variance retained 0.838), whereas Phase A / external / biological validation use the native 512-D `cpgpt_large_locus` (Phase A results doc lists dim 512). Fig 1B is labelled accordingly; absolute campaign MSE (CpGPT 0.0178) differs from Phase A (0.01667) by design (different evaluation split/protocol).",
      f"3. Deprecated pre-amendment analysis dir `external_reconstruction_v1/analysis/A/validation` exists next to `A_v1.2`; package uses only A_v1.2 (v1.2, AMENDMENT 2). Max |MSE difference| old vs new on validation per-run MSE over matched rows = {maxdiff:.3g}.",
      "4. The across-seed 'mean relative' is the unweighted mean of per-seed paired relatives (12.77% / 32.74%), which differs slightly from the ratio of seed-mean MSEs; both rounded values match the brief (12.8 / 32.7).",
      "5. The confirmation decision.json is `provisional` (at least one best epoch in the last 10 epochs: not converged) and its verdict is 'Histone+DNase preferred by parsimony pending author judgement'; Fig 1D therefore makes no equivalence claim.",
      "6. The family-screen (1 seed, batch 32, 30 ep.) and confirmation (3 seeds, batch 8, 80 ep.) absolute MSEs are on different protocols and are drawn on separate axes.",
      "7. Fig 1B attribution arms come from the campaign's evaluation split (docs call it 'frozen test patients'); they are labelled DISCOVERY EVIDENCE, not confirmation.",
      "8. FANTOM5 membership N / CpG overlap and platform/tissue descriptors for Loyfer/Micro-C/FANTOM5/RT in Table 1 have no machine-readable source in the result files (flagged `source: doc` in table1 manifest/notes); TCGA test N (918) and 450K platform are doc-sourced.",
      "9. Fig 2/3 CI for the across-seed mean is not provided by the registered analyses (only per-seed CIs and min/max); the package shows per-seed CIs plus the unweighted mean and does not invent a pooled CI.", "",
      "## Missing results / placeholders", ""]
for n, m in mans.items():
    for x in m["missing_results"]: L.append(f"- {n}: {x}")
    if m["status"] == "PLACEHOLDER": L.append(f"- {n}: PLACEHOLDER figure produced")
L += ["- Fig 5 (final) and S-final: scaffold/placeholder only (Loyfer audit pending; audit directory never read).", "- 5 sensitivity items listed as NOT_EXECUTED_REQUIRES_NEW_PREREGISTRATION_OR_ARTIFACTS in S13.", "",
      "## Compilation / lint", ""] + [f"- py_compile {f}: {s}" for f, s in comp] + ["", "ruff (--isolated, selected rules E9,F63,F7,F82,F401,F811,F841; star-imports from common are intentional):", "```", ruff_out, "```", "",
      "## sha256 of every source-data table", "", "| file | sha256 |", "|---|---|"]
for p in sorted(SRC.glob("*.csv")): L.append(f"| {rel(p)} | {sha256(p)} |")
(PAPER / "QA_REPORT.md").write_text("\n".join(L) + "\n")
print(f"QA: problems={len(problems)} check_fail={nfail} doc_missing={sum(not x[3] for x in doc_checks)}")
sys.exit(1 if (problems or nfail) else 0)
