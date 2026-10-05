# ruff: noqa: C408
"""EWAS-v2 subcommands: gate, dry-run, run --arm, contrasts, report-inputs. See docs/EWAS_V2_REGISTRATION.md."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import BlockBootstrap
from cpg_repr_benchmark.bioval_v2_launch.embedding_eval import EmbeddingStore, load_arm_specs

from . import design as D
from . import gate as G
from .metrics import bootstrap_arms, family_summary, point_metrics
from .probe import fit_oof, pair_rows

DESIGNS = ("main", "S1")
CONTRASTS = (("primary_vs_cpgpt", "regulatory_histone_dnase_v1", "cpgpt_large_locus"),
             ("secondary_vs_deepcpg", "regulatory_histone_dnase_v1", "deepcpg_dna_locus"),
             ("supplementary_vs_functional_legacy", "regulatory_histone_dnase_v1", "functional_annotations_pca"))
MAX_THREADS = 4


def default_out_root(root) -> Path:
    return Path(root) / G.OUT_BASE / G.TAG


def _limit_threads(threads: int):
    if threads > MAX_THREADS:
        raise SystemExit(f"--threads {threads} exceeds the per-process cap of {MAX_THREADS}")
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=threads)


def _rel(design: str, *parts) -> Path:
    return Path(*parts) if design == "main" else Path("sensitivity_S1", *parts)


def _design_tasks(root, design: str) -> list[str]:
    s = pd.read_csv(Path(root) / D.DESIGN_DIR / "design_summary.csv")
    col = "n_matched_pairs" if design == "main" else "s1_n_matched_pairs"
    return [t for t, n in zip(s.task, s[col], strict=True) if n > 0]


# --------------------------------------------------------------------------------------------------------------- gate
def cmd_gate(root, out_root, *, read_stores=True, command="run", arm=None) -> list[G.Check]:
    checks = G.run_gate(root, out_root, command=command, arm=arm, read_stores=read_stores,
                        input_paths=[Path(root) / D.DESIGN_DIR, Path(root) / D.LOCI])
    print(G.lg.format_checks(checks))
    return checks


# ----------------------------------------------------------------------------------------------------------- dry-run
def cost_model(root, sec_per_unit: float, iters: float = 150.0) -> pd.DataFrame:
    """Estimated probe cost per arm. unit = (n_train rows x dim x lbfgs iteration); fits per task = 5 outer x (5 C x 4 inner + 1) = 105."""
    spec = yaml.safe_load((Path(root) / G.SPEC).read_text())
    s = pd.read_csv(Path(root) / D.DESIGN_DIR / "design_summary.csv")
    rows = []
    for a in G.ALL_ARMS:
        dim = spec["arms"][a]["dim"]
        tot_fits, tot_sec = 0, 0.0
        for design, col in (("main", "n_matched_pairs"), ("S1", "s1_n_matched_pairs")):
            for n_pairs in s[col]:
                if n_pairs == 0:
                    continue
                n = 2 * n_pairs
                # outer train ~ 0.8 n; inner fits ~ (0.8 x 0.75) n = 0.6 n (80 per task); final refits 0.8 n (5 per task)
                units = (80 * 0.6 + 5 * 0.8) * n * dim * iters
                tot_fits += 105
                tot_sec += units * sec_per_unit
        rows.append(dict(arm=a, dim=dim, fits=tot_fits, est_minutes=tot_sec / 60, role="main" if a in G.MAIN_ARMS else "supplementary"))
    return pd.DataFrame(rows)


def cmd_dry_run(root, out_root, sec_per_unit: float = 5e-9) -> dict:
    checks = cmd_gate(root, out_root, read_stores=False)
    man = json.loads((Path(root) / D.MANIFEST).read_text())
    print("\nread-only counts per task (from the design manifest; no embedding read):")
    print(pd.DataFrame(man["tasks"]).T[["family", "n_universe", "n_dropped_unknown_probe_type", "n_unmatched", "n_matched_pairs",
                                        "tier", "s1_n_matched_pairs", "s1_tier", "non_independent_of"]].to_string())
    cm = cost_model(root, sec_per_unit)
    print(f"\ncost estimate (analytic; sec_per_unit={sec_per_unit:g} per row*dim*iteration, 150 iterations assumed):")
    print(cm.to_string(index=False))
    print("\nproposed launch commands (after the orchestrator commits the registration and creates the tag), one heavy process at a time:")
    cmds = launch_commands()
    print("\n".join(cmds))
    return dict(checks=[c.__dict__ for c in checks], cost=cm.to_dict("records"), commands=cmds)


def launch_commands(threads: int = 4) -> list[str]:
    pre = "cd $REPO && PYTHONPATH=$PWD/src nice -n 10 ~/miniconda3/envs/cpg-repr-benchmark/bin/python scripts/ewas_v2/run_ewas_v2.py"
    out = [f"{pre} gate"]
    out += [f"{pre} run --arm {a} --threads {threads}" for a in G.ALL_ARMS]
    out += [f"{pre} contrasts --threads {threads}", f"{pre} report-inputs"]
    return out


# --------------------------------------------------------------------------------------------------------------- run
def run_arm(root, arm: str, out_root, *, threads: int = MAX_THREADS, tasks=None, designs=DESIGNS, gate_checks=None,
            command: str = "") -> dict:
    root = Path(root)
    checks = gate_checks if gate_checks is not None else G.run_gate(root, out_root, command="run", arm=arm,
                                                                    input_paths=[root / D.DESIGN_DIR], read_stores=True)
    G.lg.enforce(checks)
    w = G.EwasWriter(root, out_root)
    spec = load_arm_specs(root, (arm,))[arm]
    store = EmbeddingStore(spec, root, verify_sha=True)
    zero_ids = store.zero_row_ids()
    plan = json.loads((root / D.DESIGN_DIR / "nested_cv_plan.json").read_text())
    done = {}
    t0 = time.time()
    with _limit_threads(threads):
        for design in designs:
            for task in (tasks or _design_tasks(root, design)):
                pairs = D.load_pairs(root / D.DESIGN_DIR, task, design)
                cpg, y, fold, chrom = pair_rows(pairs)
                digest = D.pairs_digest([(int(p), [int(q)]) for p, q in zip(pairs.pos_row, pairs.neg_row, strict=True)])
                ep_path = _rel(design, "arms", arm, task, "endpoint.json")
                if w.path(ep_path).is_file() and json.loads(w.path(ep_path).read_text()).get("pairs_digest") == digest:
                    done[f"{design}/{task}"] = "skipped_existing"
                    continue
                ts = time.time()
                X = store.load(cpg)
                fit = fit_oof(X, y, fold, plan["grid_C"], plan["max_iter"])
                P = len(pairs)
                oof = fit["oof"]
                pm = point_metrics(oof[:P], oof[P:])
                is_zero = np.isin(cpg, zero_ids)
                frame = pd.DataFrame(dict(row=np.arange(2 * P), pair_id=np.r_[pairs.pair_id, pairs.pair_id], cpg_idx=cpg, y=y,
                                          fold=fold, chrom=chrom, oof_score=oof))
                w.path(_rel(design, "arms", arm, task, "oof.parquet"))
                frame.to_parquet(w.path(_rel(design, "arms", arm, task, "oof.parquet")), index=False)
                ep = dict(arm=arm, task=task, design=design, n_pairs=P, pairs_digest=digest, point=pm,
                          C_per_outer_fold=[float(c) for c in fit["C"]], folds=[int(f) for f in fit["folds"]],
                          n_iter_max=int(fit["n_iter_max"]), n_nonconverged_fits=int(fit["n_nonconverged_fits"]),
                          max_iter=plan["max_iter"], zero_rows_positives=int(is_zero[:P].sum()),
                          zero_rows_backgrounds=int(is_zero[P:].sum()), seconds=round(time.time() - ts, 1))
                w.write_json(ep_path, ep)
                done[f"{design}/{task}"] = pm["auroc"]
    man = G.run_manifest(root, arm, store.identity(len(zero_ids)), checks=checks, threads=threads, command=command,
                         extra=dict(n_endpoints=len(done), wall_seconds=round(time.time() - t0, 1)))
    w.write_json(Path("arms", arm, "arm_manifest.json"), man)
    st = G.read_state(out_root)
    w.set_state(state="ARMS_RUNNING", arms_done=sorted(set(st["arms_done"]) | {arm}))
    return done


# ---------------------------------------------------------------------------------------------------------- contrasts
def _load_task(out_root, arms, design, task):
    scores, ref = {}, None
    for a in arms:
        f = Path(out_root) / _rel(design, "arms", a, task, "oof.parquet")
        df = pd.read_parquet(f)
        key = df[["row", "pair_id", "cpg_idx", "y", "fold", "chrom"]]
        if ref is None:
            ref = key
        elif not ref.equals(key):
            raise AssertionError(f"{task}/{design}: rows of arm {a} are not identical to the other arms (shared-background violation)")
        P = len(df) // 2
        scores[a] = (df.oof_score.to_numpy()[:P], df.oof_score.to_numpy()[P:])
    return scores, ref.chrom.to_numpy()[: len(ref) // 2]


def run_contrasts(root, out_root, *, threads: int = MAX_THREADS, allow_nonconverged: bool = False, gate_checks=None) -> dict:
    root = Path(root)
    checks = gate_checks if gate_checks is not None else G.run_gate(root, out_root, command="contrasts",
                                                                    input_paths=[root / D.DESIGN_DIR], read_stores=False)
    G.lg.enforce([c for c in checks if c.name != "store_sha256_and_coverage"])
    w = G.EwasWriter(root, out_root)
    st = G.read_state(out_root)
    arms = [a for a in G.ALL_ARMS if a in st["arms_done"]]
    contrasts = [c for c in CONTRASTS if c[1] in arms and c[2] in arms]
    blocks = np.array(sorted(D.CHROMS))
    bb = BlockBootstrap(blocks, D.N_BOOT, D.SEED)
    summ = pd.read_csv(root / D.DESIGN_DIR / "design_summary.csv").set_index("task")
    rows, meta = [], []
    with _limit_threads(threads):
        for design in DESIGNS:
            for task in _design_tasks(root, design):
                eps = {a: json.loads((Path(out_root) / _rel(design, "arms", a, task, "endpoint.json")).read_text()) for a in arms}
                nc = sum(e["n_nonconverged_fits"] for e in eps.values())
                if nc and not allow_nonconverged:
                    raise SystemExit(f"{task}/{design}: {nc} non-converged probe fits (use --allow-nonconverged to report with a flag)")
                scores, chrom = _load_task(out_root, arms, design, task)
                res = bootstrap_arms(scores, chrom, bb, contrasts)
                s = summ.loc[task]
                tier = s["tier"] if design == "main" else s["s1_tier"]
                ep = dict(task=task, design=design, family=D.TASK_FAMILY[task], tier=tier, tier_main=s["tier"], tier_s1=s["s1_tier"],
                          non_independent_of=s["non_independent_of"] if isinstance(s["non_independent_of"], str) else "",
                          n_pairs=len(chrom), n_nonconverged_fits=int(nc), arms={})
                for a in arms:
                    ep["arms"][a] = dict(**{m: res["point"][a][m] for m in ("auroc", "concordance", "auprc")},
                                         auroc_ci=res["ci"][a]["auroc"], concordance_ci=res["ci"][a]["concordance"],
                                         zero_rows_positives=eps[a]["zero_rows_positives"], zero_rows_backgrounds=eps[a]["zero_rows_backgrounds"],
                                         C_per_outer_fold=eps[a]["C_per_outer_fold"])
                w.write_json(_rel(design, "endpoints", f"{task}.json") if design != "main" else Path("endpoints", f"{task}.json"), ep)
                meta.append(dict(task=task, design=design, family=ep["family"], tier=tier, non_independent_of=ep["non_independent_of"]))
                for (name, m), r in res["contrast"].items():
                    rows.append(dict(design=design, task=task, contrast=name, metric=m, **r))
    cdf = pd.DataFrame(rows)
    w.write_df("contrasts.csv", cdf)
    w.write_df("sensitivity_S1/contrasts.csv", cdf[cdf.design == "S1"])
    fam = []
    meta_df = pd.DataFrame(meta)
    for design in DESIGNS:
        sub = cdf[cdf.design == design]
        m = meta_df[meta_df.design == design]
        if len(sub):
            fam.append(family_summary(sub, m))
    fdf = pd.concat(fam, ignore_index=True)
    w.write_df("family_summary.csv", fdf)
    w.write_json("bootstrap_manifest.json", dict(draw_table_fingerprint=bb.fingerprint(), n_boot=bb.n_boot, seed=bb.seed,
                                                 blocks=blocks.tolist(), arms=arms, contrasts=[list(c) for c in contrasts]))
    w.set_state(state="EXECUTED", contrasts_done=True)
    return dict(n_contrast_rows=len(cdf), family_rows=len(fdf), arms=arms)


# --------------------------------------------------------------------------------------------------------- report-inputs
def report_inputs(root, out_root, gate_checks=None) -> dict:
    root = Path(root)
    checks = gate_checks if gate_checks is not None else G.run_gate(root, out_root, command="report-inputs",
                                                                    input_paths=[root / D.DESIGN_DIR], read_stores=False)
    G.lg.enforce([c for c in checks if c.name != "store_sha256_and_coverage"])
    w = G.EwasWriter(root, out_root)
    summ = pd.read_csv(root / D.DESIGN_DIR / "design_summary.csv")
    exp5 = pd.read_csv(root / D.EXP5_TIERS).set_index("set")["support_tier"]
    tt = summ[["task", "family", "n_universe", "n_matched_pairs", "tier", "s1_n_matched_pairs", "s1_tier", "non_independent_of"]].copy()
    tt["exp5_tier"] = tt.task.map(exp5)
    tt["tier_changed_vs_exp5"] = tt.tier != tt.exp5_tier
    w.write_df("report_inputs/tier_table.csv", tt)
    wide, zero = [], []
    for design in DESIGNS:
        d = Path(out_root) / _rel(design, "endpoints")
        for f in sorted(d.glob("*.json")):
            e = json.loads(f.read_text())
            for a, v in e["arms"].items():
                wide.append(dict(design=design, task=e["task"], family=e["family"], tier=e["tier"], arm=a, n_pairs=e["n_pairs"],
                                 auroc=v["auroc"], auroc_lo=v["auroc_ci"][0], auroc_hi=v["auroc_ci"][1], concordance=v["concordance"],
                                 concordance_lo=v["concordance_ci"][0], concordance_hi=v["concordance_ci"][1], auprc=v["auprc"]))
                zero.append(dict(design=design, task=e["task"], arm=a, zero_rows_positives=v["zero_rows_positives"],
                                 zero_rows_backgrounds=v["zero_rows_backgrounds"], n_pairs=e["n_pairs"]))
    wd = pd.DataFrame(wide)
    w.write_df("report_inputs/per_task_arm_metrics.csv", wd)
    w.write_df("report_inputs/zero_rows.csv", pd.DataFrame(zero))
    if len(wd):
        m = wd[wd.design == "main"].merge(wd[wd.design == "S1"], on=["task", "arm"], suffixes=("_main", "_s1"))
        m["auroc_s1_minus_main"] = m.auroc_s1 - m.auroc_main
        w.write_df("report_inputs/s1_vs_main_auroc.csv", m[["task", "arm", "auroc_main", "auroc_s1", "auroc_s1_minus_main", "tier_main", "tier_s1"]])
    c = pd.read_csv(Path(out_root) / "contrasts.csv")
    pc = c[c.contrast == "primary_vs_cpgpt"]
    sg = pc.pivot_table(index=["task", "metric"], columns="design", values="delta").reset_index()
    if {"main", "S1"} <= set(sg.columns):
        sg["sign_agree"] = np.sign(sg["main"]) == np.sign(sg["S1"])
        w.write_df("report_inputs/primary_contrast_sign_main_vs_s1.csv", sg)
    w.set_state(state="REPORTED", reported=True)
    return dict(n_wide=len(wd))
