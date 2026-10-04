"""bioval v2 follow-up driver. Registration: docs/BIOLOGICAL_VALIDATION_V2_SENSITIVITY_AND_EXPLORATORY_REGISTRATION.md.

  sensitivity           PART A: frozen protocol sensitivities that are runnable from frozen files (descriptive, no Holm).
  exploratory-loyfer    PART B: EXPLORATORY POST-HOC Loyfer analyses (B1 strata, B2 geometries, B3 profile ridge).
Common: --arm <id> (repeatable) | --all ; --dry-run (gate report + read-only counts, computes NO embedding metric) ;
        --contrasts (after all arms finished; reads only follow-up outputs + frozen primary results) ; --threads <=8.
A real run (or --contrasts) refuses to start unless the follow-up gate passes. Output roots (registered): sensitivity/ and
exploratory_posthoc_loyfer/ under outputs/biological_validation_v2/bioval-v2-protocol-freeze-v1/; the frozen primary result
directories are never written (FollowupWriter). No freeze/prepare command is run or imported; no TCGA / protocol path is read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _early_threads(argv):
    n = 2
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            n = int(argv[i + 1])
        elif a.startswith("--threads="):
            n = int(a.split("=", 1)[1])
    if not 1 <= n <= 8:
        raise SystemExit("--threads must be in [1, 8] (shared machine)")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(n)
    return n


THREADS = _early_threads(sys.argv)
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from cpg_repr_benchmark.bioval_v2_followup import compare as cmp
from cpg_repr_benchmark.bioval_v2_followup import drivers as dr
from cpg_repr_benchmark.bioval_v2_followup import gate as fg
from cpg_repr_benchmark.bioval_v2_followup import interpret as ip
from cpg_repr_benchmark.bioval_v2_followup import registry as rg
from cpg_repr_benchmark.bioval_v2_followup import strata as st
from cpg_repr_benchmark.bioval_v2_followup import targets as tg
from cpg_repr_benchmark.bioval_v2_launch import launch_endpoints as le
from cpg_repr_benchmark.bioval_v2_launch import launch_gate as lg
from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import B_DEFAULT, SEED_DEFAULT, BlockBootstrap
from cpg_repr_benchmark.bioval_v2_launch.embedding_eval import (
    ARMS,
    REFERENCE_ARM,
    EmbeddingStore,
    exclusion_mask,
    load_arm_specs,
    union_zero_loci,
)

COST = """Cost estimate per arm (CPU, 2 threads; ESTIMATES, nothing timed on real embeddings):
 PART A sensitivity: loyfer_mask_lenient / drop_single ~3-6 min each; compartment 50 kb x2 ~10-15 min each; fantom5_membership_N{1,3,10,20}
   ~8-25 min each (one logistic probe each); fantom5 activity (sample + 3 categories) ~2-4 min each.   => ~1.5-2.5 h per arm.
 PART B  B1+B2 (4 geometries x 28 statistics x 1000 replicates; PC1/mean pass over 408k x d) ~25-45 min; B3 ridge (79 targets, Gram-based,
   381k x 128-512; SuffStats bootstrap negligible) ~10-25 min.   => ~0.7-1.2 h per arm.
 RAM per arm process ~4-7 GB. 4 arms in parallel x 2 threads = 8 threads: Part A ~2.5 h wall, Part B ~1.2 h wall; contrasts < 5 min."""


def log(*a):
    print(*a, flush=True)


def resolve_arms(args):
    if args.all:
        return list(ARMS)
    if not args.arm:
        raise SystemExit("give --arm <id> (repeatable) or --all")
    for a in args.arm:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a}; known: {ARMS}")
    return args.arm


def default_out(sub):
    return ROOT / (rg.SENS_OUT if sub == "sensitivity" else rg.EXPL_OUT)


def input_paths(specs):
    return [ROOT / lg.BV / v for v in dr.F.values()] + [ROOT / s.store_h5 for s in specs.values()]


def load_universe():
    u = pd.read_parquet(ROOT / lg.BV / le.FROZEN_FILES["rt"], columns=["cpg_idx", "chrom"]).sort_values("cpg_idx")
    return u.reset_index(drop=True)


def all_zero_loci(specs, verify_sha, log=log):
    zero, stores = {}, {}
    for a, s in specs.items():
        stores[a] = EmbeddingStore(s, ROOT, verify_sha=verify_sha)
        zero[a] = stores[a].zero_row_ids()
        log(f"  zero rows {a}: {len(zero[a])}")
    return zero, stores


# ------------------------------------------------------------------------------------------------- dry-run counts
def dry_counts(zero_union):
    """Read-only counts from frozen files (+ the zero-locus set). No embedding statistic."""
    out = {}
    base = ROOT / lg.BV
    pairs = pd.read_parquet(base / dr.F["loyfer_pairs"])
    prof = tg.LoyferProfiles(base / dr.F["loyfer_beta"])
    out["target_reproduction"] = tg.assert_reproduces_frozen(pairs, prof)
    zkeep = exclusion_mask([pairs.cpg_i.to_numpy(), pairs.cpg_j.to_numpy()], zero_union)
    out["loyfer_pairs"] = {"n": len(pairs), "zero_locus_excluded": int((~zkeep).sum()), "kept": int(zkeep.sum())}
    out["loyfer_variants"] = {}
    for v in ("mask_lenient", "drop_single_sample_groups"):
        r, _ns = tg.variant_target(pairs, prof, v)
        ok = np.isfinite(r)
        out["loyfer_variants"][v] = {"target_defined": int(ok.sum()), "undefined": int((~ok).sum()),
                                     "kept_after_zero_exclusion": int((ok & zkeep).sum()),
                                     "per_stratum_kept": {s: int((ok & zkeep & (pairs.stratum == s).to_numpy()).sum()) for s in rg.ALL_STRATA}}
    out["single_sample_groups"] = [prof.groups[i] for i in prof.single_sample_groups()]
    vmin, mmean = st.pair_strata_values(pairs, prof)
    out["registered_edges_check"] = st.check_registered_edges(vmin, mmean)
    cl = st.cells(pairs.stratum.to_numpy(), vmin[:], mmean[:])
    out["B1_pairs_per_cell_after_zero_exclusion"] = {k: int((m & zkeep).sum()) for k, m in cl.items()}
    out["B1_pooled_strata_per_V_cell"] = {k: {s: int((m & zkeep & (pairs.stratum == s).to_numpy()).sum()) for s in rg.POOLED_STRATA}
                                          for k, m in cl.items() if k.startswith(("V_", "M_"))}
    P = prof.matrix("primary")
    full = ~np.isnan(P).any(1)
    c2f = __import__("cpg_repr_benchmark.biological_validation_v2.splits", fromlist=["x"]).chromosome_blocked_folds(
        [f"chr{i}" for i in range(1, 23)], 5, 17)
    fold = pd.Series(prof.chrom[full]).map(c2f).to_numpy()
    _, cvar, _ = st.per_cpg_stats(P[full])
    out["B3"] = {"n_cpgs_complete_case_39_groups": int(full.sum()), "n_cpgs_total": len(full),
                 "per_fold": {int(f): int((fold == f).sum()) for f in range(5)},
                 "n_variable_cpgs": int((cvar >= rg.VARIABLE_CPG_VAR).sum()), "chrom_to_fold": c2f,
                 "n_ridge_targets": 79, "alpha_grid": list(__import__("cpg_repr_benchmark.bioval_v2_launch.chrom_probes", fromlist=["x"]).RIDGE_ALPHAS)}
    m = pd.read_parquet(base / dr.F["f5_membership"])
    enh = pd.read_parquet(base / dr.F["f5_enhancers"], columns=["enhancer_id", "n_samples_expressed"]).set_index("enhancer_id")
    pos = m[m.label == 1]
    pn = pos.enhancer_id.map(enh.n_samples_expressed)
    out["fantom5_membership_pairs_per_N"] = {int(n): int((pn >= n).sum()) for n in (1, 3, 10, 20)}
    for key, name in (("comp_H1_50", "H1_50kb"), ("comp_GM_50", "GM12878_50kb")):
        d = pd.read_parquet(base / dr.F[key], columns=["E1", "chrom"])
        out[f"compartment_{name}"] = {"n_with_E1": int(d.E1.notna().sum()), "n_total": len(d)}
    a = pd.read_parquet(base / dr.F["f5_activity"])
    out["fantom5_activity_pairs"] = {"n": len(a), "zero_locus_excluded": int((~exclusion_mask([a.i.to_numpy(), a.j.to_numpy()], zero_union)).sum())}
    ci = pd.read_parquet(base / dr.F["f5_act_coll_idx"])
    out["fantom5_activity_groups_per_category"] = ci.category.value_counts().to_dict()
    return out


def cmd_dry_run(args, sub, arms):
    log(f"=== bioval v2 FOLLOW-UP ({sub}): DRY RUN (no embedding statistic is computed) ===")
    specs = load_arm_specs(ROOT, ARMS)
    out_root = Path(args.out_root or default_out(sub))
    if sub == "sensitivity":
        items = args.item or list(rg.ITEMS)
        log("\nRunnable Part A items per arm: " + ", ".join(items))
        log("Not runnable (blockers in the registration doc):")
        for k, (s, why) in rg.NOT_RUNNABLE.items():
            log(f"  {k}: {s} - {why}")
    else:
        log("\nPart B per arm: B1 strata + B2 geometries (exploratory_similarity), B3 chromosome-blocked profile ridge. Label: EXPLORATORY POST-HOC.")
    log(f"\nOutput root: {out_root}")
    log("\nGate (report only in dry-run):")
    checks = fg.run_gate(ROOT, arms, out_root, sub, input_paths=input_paths(specs), skip_heavy=args.skip_heavy)
    log(lg.format_checks(checks))
    log("\nStores / all-zero rows:")
    zero, _stores = all_zero_loci(specs, verify_sha=not args.skip_heavy)
    union = union_zero_loci(zero)
    log(f"  union of all-zero loci: {len(union)}")
    log("\nRead-only counts from frozen files:")
    cnt = dry_counts(union)
    for k, v in cnt.items():
        log(f"  {k}: {json.dumps(v, default=str)}")
    log("\n" + COST)
    log("\nProposed launch (after the orchestrator commits registration + code and the gate passes):")
    for a in ARMS:
        log(f"  nice -n 10 ~/miniconda3/envs/cpg-repr-benchmark/bin/python scripts/bioval_v2/run_followup.py {sub} --arm {a} --threads 2 &")
    log(f"  wait; nice -n 10 python scripts/bioval_v2/run_followup.py {sub} --contrasts --all")
    if args.report:
        w = fg.FollowupWriter(ROOT, Path(args.report).parent)
        w.write_json(Path(args.report).name, {"gate": [c.__dict__ for c in checks], "counts": cnt, "zero_union": len(union)})
        log(f"report written to {args.report}")
    failed = [c.name for c in checks if not c.ok]
    log("\nDRY RUN COMPLETE: no output written under the output root.")
    log("Gate status: " + ("ALL PASS" if not failed else "WOULD REFUSE A REAL RUN (failing: " + ", ".join(failed) + ")"))


# ------------------------------------------------------------------------------------------------- writers
def write_endpoint(w, arm, res, extra=None):
    body = {"endpoint_id": res.endpoint_id, "class": res.cls, "axis": res.axis, "primary_stat": res.primary_stat,
            "meta": res.meta, "stats": {k: s.as_dict() for k, s in res.stats.items()}, **(extra or {})}
    w.write_json(f"{arm}/endpoints/{res.endpoint_id}.json", body)
    reps = {k: s.rep for k, s in res.stats.items() if s.rep is not None}
    if reps:
        w.save_npz(f"{arm}/endpoints/{res.endpoint_id}.replicates.npz", **reps)


def make_ctx(arm):
    specs = load_arm_specs(ROOT, ARMS)
    zero, stores = all_zero_loci(specs, verify_sha=True, log=lambda *a: None)
    union = union_zero_loci(zero)
    store = stores[arm]
    uni = load_universe()
    boot = BlockBootstrap(np.unique(uni.chrom.to_numpy().astype(str)), n_boot=B_DEFAULT, seed=SEED_DEFAULT)
    return le.EvalContext(ROOT, store, union, boot, uni, log=log), zero, union, boot


def manifest_for(arm, sub, ctx, zero, union, boot, checks, man_sha, out_root, tasks):
    m = lg.build_run_manifest(
        ROOT, arm, ctx.store.identity(len(zero[arm])), seed=SEED_DEFAULT, n_boot=B_DEFAULT, boot_fingerprint=boot.fingerprint(),
        threads=THREADS, exclusion_counts={"all_zero_rows_per_store": {a: len(z) for a, z in zero.items()},
                                           "union_zero_loci": len(union), "per_endpoint": {}},
        checks=checks, endpoints=tasks, command=" ".join(sys.argv), manifest_checksums_sha256=man_sha, out_root=out_root)
    m["followup_registration_doc"] = {"path": rg.FOLLOWUP_REGISTRATION_DOC,
                                      "sha256": lg.hashlib.sha256((ROOT / rg.FOLLOWUP_REGISTRATION_DOC).read_bytes()).hexdigest(),
                                      "last_commit": lg._git(ROOT, "log", "-1", "--format=%H", "--", rg.FOLLOWUP_REGISTRATION_DOC)[1]}
    m["launch_registration_doc_note"] = "registration_doc above = launch registration (unchanged); followup_registration_doc = this phase"
    m["subcommand"] = sub
    m["exploratory_posthoc"] = sub == "exploratory-loyfer"
    m["label"] = "EXPLORATORY POST-HOC; does not replace the frozen primary" if sub == "exploratory-loyfer" else \
        "frozen-protocol sensitivity; descriptive, no Holm"
    m["frozen_results_fingerprint_before"] = fg.frozen_results_fingerprint(ROOT)
    return m


def run_arm(args, sub, arm, checks, man_sha):
    out_root = Path(args.out_root or default_out(sub))
    w = fg.FollowupWriter(ROOT, out_root)
    log(f"[{arm}] loading stores + zero-row scan")
    ctx, zero, union, boot = make_ctx(arm)
    if sub == "sensitivity":
        tasks = args.item or list(rg.ITEMS)
        for t in tasks:
            if t not in dr.SENS_DRIVERS:
                raise SystemExit(f"unknown/non-runnable item {t}")
    else:
        tasks = args.part or ["similarity", "ridge"]
    man = manifest_for(arm, sub, ctx, zero, union, boot, checks, man_sha, out_root, tasks)
    w.write_json(f"{arm}/run_manifest.json", man)
    for t in tasks:
        log(f"[{arm}] {t} ...")
        if sub == "sensitivity":
            res = dr.SENS_DRIVERS[t](ctx)
            write_endpoint(w, arm, res)
            if "exclusion" in res.meta:
                man["exclusion_counts"]["per_endpoint"][t] = res.meta["exclusion"]
            s = res.stats[res.primary_stat]
            log(f"    {res.primary_stat}: {s.value:.5f} [{s.ci_lo:.5f}, {s.ci_hi:.5f}] n={s.n}")
        elif t == "similarity":
            res = dr.exploratory_similarity(ctx, arm)
            write_endpoint(w, arm, res, {"EXPLORATORY_POST_HOC": True})
            man["exclusion_counts"]["per_endpoint"]["exploratory_similarity"] = res.meta["exclusion"]
        elif t == "ridge":
            body, reps = dr.exploratory_profile_ridge(ctx, arm)
            w.write_json(f"{arm}/endpoints/B3_profile_ridge.json", body)
            w.save_npz(f"{arm}/endpoints/B3_profile_ridge.replicates.npz", **reps)
            man["exclusion_counts"]["per_endpoint"]["B3_profile_ridge"] = {
                "n_cpgs_complete_case": body["n_cpgs_complete_case"], "zero_rows_kept": True}
        else:
            raise SystemExit(f"unknown part {t}")
        man.setdefault("tasks_done", []).append(t)
        w.write_json(f"{arm}/run_manifest.json", man)
    man["freeze_modules_imported"] = lg.freeze_modules_imported()
    man["no_freeze_command_used"] = not man["freeze_modules_imported"]
    man["frozen_results_fingerprint_after"] = fg.frozen_results_fingerprint(ROOT)
    man["frozen_results_unchanged"] = man["frozen_results_fingerprint_after"] == fg.FROZEN_RESULTS_SHA256
    man["finished_utc"] = lg.datetime.now(lg.timezone.utc).isoformat()
    w.write_json(f"{arm}/run_manifest.json", man)
    log(f"[{arm}] done -> {out_root / arm}")


# ------------------------------------------------------------------------------------------------- contrasts
def _frozen_primary(arm, endpoint, stat):
    return dr.frozen_value(ROOT, arm, endpoint, stat)


def _frozen_primary_delta(comp, endpoint, stat):
    f = ROOT / dr.RESULTS_BASE / "contrasts" / f"{REFERENCE_ARM}__vs__{comp}.json"
    for r in json.loads(f.read_text())["rows"]:
        if r["endpoint"] == endpoint and r["stat"] == stat:
            return r["delta"]
    raise KeyError((comp, endpoint, stat))


def contrasts_sensitivity(args, w, out_root):
    comps = [a for a in ARMS if a != REFERENCE_ARM]
    summary = []
    per_comp_rows = {c: [] for c in comps}
    for item, (status, (pe, ps), note) in rg.ITEMS.items():
        files = {a: out_root / a / "endpoints" / f"{item}.json" for a in ARMS}
        if not all(f.exists() for f in files.values()):
            summary.append({"item": item, "status": "NOT RUN (missing arm outputs)", "protocol_status": status})
            continue
        js = {a: json.loads(f.read_text()) for a, f in files.items()}
        npz = {a: np.load(out_root / a / "endpoints" / f"{item}.replicates.npz") for a in ARMS}
        pstat = js[REFERENCE_ARM]["primary_stat"]
        sens_vals = {a: js[a]["stats"][pstat]["value"] for a in ARMS}
        prim_vals = {a: _frozen_primary(a, pe, ps) for a in ARMS}
        rows = {}
        for comp in comps:
            for name, sa in js[REFERENCE_ARM]["stats"].items():
                if name not in npz[REFERENCE_ARM].files or name not in js[comp]["stats"]:
                    continue
                c = cmp.paired_contrast(sa["value"], npz[REFERENCE_ARM][name], js[comp]["stats"][name]["value"], npz[comp][name])
                row = {"item": item, "reference": REFERENCE_ARM, "comparator": comp, "stat": name, "ref_value": sa["value"],
                       "comp_value": js[comp]["stats"][name]["value"], "label": "descriptive; raw p; no Holm; delta=regulatory-comparator", **c}
                per_comp_rows[comp].append(row)
                if name == pstat:
                    rows[comp] = row
        pdel = {c: _frozen_primary_delta(c, pe, ps) for c in comps}
        v = cmp.item_verdict(pdel, rows, prim_vals, sens_vals)
        summary.append({"item": item, "status": "done", "protocol_status": status, "primary_reference": f"{pe}:{ps}",
                        "primary_values": prim_vals, "sensitivity_values": sens_vals,
                        "primary_deltas": pdel, "sensitivity_contrasts": {c: {k: rows[c][k] for k in ("delta", "ci_lo", "ci_hi", "p")} for c in comps},
                        **v, "n_pairs_or_rows": {a: js[a]["stats"][pstat]["n"] for a in ARMS}})
    for comp, rows in per_comp_rows.items():
        w.write_json(f"contrasts/{REFERENCE_ARM}__vs__{comp}.json", {"rows": rows, "note": "descriptive; no Holm"})
        w.write_text(f"contrasts/{REFERENCE_ARM}__vs__{comp}.csv", pd.DataFrame(rows).to_csv(index=False))
    for k, (s, why) in rg.NOT_RUNNABLE.items():
        summary.append({"item": k, "status": f"not runnable: {why}", "protocol_status": s})
    w.write_json("SUMMARY.json", summary)
    w.write_text("SUMMARY.csv", pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in summary]).to_csv(index=False))
    for r in summary:
        log(f"  {r['item']}: {r.get('verdict', r['status'])}")


def contrasts_exploratory(args, w, out_root):
    comps = [a for a in ARMS if a != REFERENCE_ARM]
    # --- B1/B2 similarity
    fs = {a: out_root / a / "endpoints" / "exploratory_similarity.json" for a in ARMS}
    if all(f.exists() for f in fs.values()):
        js = {a: json.loads(f.read_text()) for a, f in fs.items()}
        nz = {a: np.load(out_root / a / "endpoints" / "exploratory_similarity.replicates.npz") for a in ARMS}
        for comp in comps:
            rows = []
            for name, sa in js[REFERENCE_ARM]["stats"].items():
                if name in js[comp]["stats"] and name in nz[comp].files:
                    c = cmp.paired_contrast(sa["value"], nz[REFERENCE_ARM][name], js[comp]["stats"][name]["value"], nz[comp][name])
                    rows.append({"reference": REFERENCE_ARM, "comparator": comp, "stat": name, "ref_value": sa["value"],
                                 "comp_value": js[comp]["stats"][name]["value"], "label": "EXPLORATORY POST-HOC; raw p; no Holm", **c})
            w.write_json(f"contrasts/similarity__{REFERENCE_ARM}__vs__{comp}.json", {"EXPLORATORY_POST_HOC": True, "rows": rows})
            w.write_text(f"contrasts/similarity__{REFERENCE_ARM}__vs__{comp}.csv", pd.DataFrame(rows).to_csv(index=False))
        prim = {a: _frozen_primary(a, "loyfer_profile", rg.LOYFER_PRIMARY_STAT) for a in ARMS}
        rank = []
        for name in js[REFERENCE_ARM]["stats"]:
            vals = {a: js[a]["stats"][name]["value"] for a in ARMS if name in js[a]["stats"]}
            if len(vals) == 4:
                o = cmp.ordering(vals)
                rank.append({"stat": name, **{f"value__{a}": v for a, v in vals.items()}, "ordering": ">".join(o),
                             "same_ordering_as_frozen_primary": o == cmp.ordering(prim),
                             "candidate_rank": o.index(REFERENCE_ARM) + 1})
        w.write_text("similarity_ranking_by_cell.csv", pd.DataFrame(rank).to_csv(index=False))
        w.write_json("similarity_ranking_by_cell.json", {"frozen_primary_ordering": cmp.ordering(prim), "rows": rank,
                                                         "EXPLORATORY_POST_HOC": True})
        within_rows = []
        for a in ARMS:
            for name in js[a]["stats"]:
                if name.startswith(("spearman_pooled_eqw__", "spearman__")):
                    pre, g, cell = name.split("__")
                    if g == "cosine" or f"{pre}__cosine__{cell}" not in js[a]["stats"]:
                        continue
                    ref_name = f"{pre}__cosine__{cell}"
                    c = cmp.paired_contrast(js[a]["stats"][name]["value"], nz[a][name], js[a]["stats"][ref_name]["value"], nz[a][ref_name])
                    within_rows.append({"arm": a, "geometry": g, "stat_family": pre, "cell": cell,
                                        "label": "EXPLORATORY POST-HOC; within-arm delta = geometry - cosine; raw p; no Holm", **c})
        w.write_text("geometry_within_arm_deltas.csv", pd.DataFrame(within_rows).to_csv(index=False))
        log(f"  B1/B2 contrasts + ranking written ({len(rank)} statistics)")
    # --- B3
    fr = {a: out_root / a / "endpoints" / "B3_profile_ridge.json" for a in ARMS}
    if all(f.exists() for f in fr.values()):
        js = {a: json.loads(f.read_text()) for a, f in fr.items()}
        nz = {a: np.load(out_root / a / "endpoints" / "B3_profile_ridge.replicates.npz") for a in ARMS}
        for comp in comps:
            rows = []
            for key in nz[REFERENCE_ARM].files:
                t, s, metric = key.split("__")
                ra, rb = nz[REFERENCE_ARM][key], nz[comp][key]
                ca = js[REFERENCE_ARM]["targets"][t]["subsets"][s]["model"][metric]
                cb = js[comp]["targets"][t]["subsets"][s]["model"][metric]
                if np.ndim(ra) == 1:
                    items = [(None, ca, cb, ra, rb)]
                else:
                    items = [(g, ca[g], cb[g], ra[:, g], rb[:, g]) for g in range(ra.shape[1])]
                for g, va, vb, xa, xb in items:
                    c = cmp.paired_contrast(va, xa, vb, xb)
                    rows.append({"reference": REFERENCE_ARM, "comparator": comp, "target": t, "subset": s, "metric": metric,
                                 "group": None if g is None else js[REFERENCE_ARM]["groups"][g], "ref_value": va, "comp_value": vb,
                                 "label": "EXPLORATORY POST-HOC; raw p; no Holm", **c})
            w.write_json(f"contrasts/ridge__{REFERENCE_ARM}__vs__{comp}.json", {"EXPLORATORY_POST_HOC": True, "rows": rows})
            w.write_text(f"contrasts/ridge__{REFERENCE_ARM}__vs__{comp}.csv", pd.DataFrame(rows).to_csv(index=False))
        flat = []
        for a in ARMS:
            for t, tb in js[a]["targets"].items():
                for s, sb in tb["subsets"].items():
                    for kind, cell in sb.items():
                        flat.append({"arm": a, "target": t, "subset": s, "kind": kind, **{k: v for k, v in cell.items()
                                                                                          if k in ("r2_global", "r2_macro", "pearson_pooled", "pearson_mean_over_cpgs")}})
        w.write_text("ridge_summary_by_arm.csv", pd.DataFrame(flat).to_csv(index=False))
        # functional vs sequence arms, ridge metric only (registered extra contrasts for the C rule)
        extra = {}
        for seqa in ip.SEQ:
            for key in nz[REFERENCE_ARM].files:
                t, s_, metric = key.split("__")
                if metric != "r2_global":
                    continue
                c = cmp.paired_contrast(js[ip.FUNC]["targets"][t]["subsets"][s_]["model"][metric], nz[ip.FUNC][key],
                                        js[seqa]["targets"][t]["subsets"][s_]["model"][metric], nz[seqa][key])
                extra[(seqa, t, s_)] = c
        w.write_text("ridge_functional_vs_sequence.csv", pd.DataFrame(
            [{"comparator": k[0], "target": k[1], "subset": k[2], "metric": "r2_global", **v} for k, v in extra.items()]).to_csv(index=False))
        log("  B3 contrasts + summary written")
        if (out_root / REFERENCE_ARM / "endpoints" / "exploratory_similarity.json").exists():
            interpretation(w, js, nz, extra, out_root)


def _delta_row(c):
    return {k: c[k] for k in ("delta", "ci_lo", "ci_hi")}


def interpretation(w, js_ridge, nz_ridge, extra, out_root):
    """Mechanical A/B/C reading (interpret.py). Primary reading: target=centered, subset=variable_cpgs, metric=r2_global;
    consistency readings: centered/all_cpgs and raw/all_cpgs."""
    sim = {a: json.loads((out_root / a / "endpoints" / "exploratory_similarity.json").read_text()) for a in ARMS}
    simnz = {a: np.load(out_root / a / "endpoints" / "exploratory_similarity.replicates.npz") for a in ARMS}
    out = {"EXPLORATORY_POST_HOC": True, "thresholds": {"R2_NEAR_BASELINE": ip.R2_NEAR_BASELINE, "GEO_IMPROVEMENT": ip.GEO_IMPROVEMENT},
           "readings": {}}
    gname = {g: f"spearman_pooled_eqw__{g}__all" for g in ("cosine", "centered_cosine", "pc1_removed_cosine", "neg_euclidean")}
    within = {}
    for g in ("centered_cosine", "pc1_removed_cosine", "neg_euclidean"):
        within[g] = _delta_row(cmp.paired_contrast(sim[REFERENCE_ARM]["stats"][gname[g]]["value"], simnz[REFERENCE_ARM][gname[g]],
                                                   sim[REFERENCE_ARM]["stats"][gname["cosine"]]["value"], simnz[REFERENCE_ARM][gname["cosine"]]))
    gap = {g: {s: _delta_row(cmp.paired_contrast(sim[REFERENCE_ARM]["stats"][gname[g]]["value"], simnz[REFERENCE_ARM][gname[g]],
                                                 sim[s]["stats"][gname[g]]["value"], simnz[s][gname[g]])) for s in ip.SEQ} for g in within}
    go = ip.geometry_outcome(within, gap)
    for t, s_ in (("centered", "variable_cpgs"), ("centered", "all_cpgs"), ("raw", "all_cpgs")):
        key = f"{t}__{s_}__r2_global"
        vals = {a: js_ridge[a]["targets"][t]["subsets"][s_]["model"]["r2_global"] for a in ARMS}
        cvs = {sq: _delta_row(cmp.paired_contrast(vals[REFERENCE_ARM], nz_ridge[REFERENCE_ARM][key], vals[sq], nz_ridge[sq][key]))
               for sq in ip.SEQ}
        fvs = {sq: _delta_row(extra[(sq, t, s_)]) for sq in ip.SEQ}
        ro = ip.ridge_outcome(vals, cvs)
        fo = ip.functional_outcome(fvs)
        out["readings"][f"{t}/{s_}"] = {"M_ridge_values": vals, "candidate_vs_seq": cvs, "functional_vs_seq": fvs,
                                         **ip.interpret(ro, go, fo)}
    out["geometry_within_candidate"] = within
    out["geometry_gap_vs_seq"] = gap
    out["primary_reading"] = "centered/variable_cpgs"
    w.write_json("interpretation_ABC.json", out)
    for k, v in out["readings"].items():
        log(f"  [{k}] RO={v['ridge_outcome']} GO={v['geometry_outcome']} FO={v['functional_outcome']} -> A: {v['A']} | B: {v['B']} | C: {v['C']}")


def cmd_contrasts(args, sub, checks):
    out_root = Path(args.out_root or default_out(sub))
    w = fg.FollowupWriter(ROOT, out_root)
    (contrasts_sensitivity if sub == "sensitivity" else contrasts_exploratory)(args, w, out_root)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subcommand", choices=["sensitivity", "exploratory-loyfer"])
    ap.add_argument("--arm", action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--item", action="append", help="sensitivity: restrict to item ids")
    ap.add_argument("--part", action="append", choices=["similarity", "ridge"], help="exploratory-loyfer: restrict parts")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--contrasts", action="store_true")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--skip-heavy", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    sub = args.subcommand
    if args.skip_heavy and not args.dry_run:
        raise SystemExit("--skip-heavy is only allowed with --dry-run")
    arms = resolve_arms(args) if (args.arm or args.all or not args.contrasts) else list(ARMS)
    if args.dry_run:
        return cmd_dry_run(args, sub, arms)
    specs = load_arm_specs(ROOT, ARMS)
    out_root = Path(args.out_root or default_out(sub))
    checks = fg.run_gate(ROOT, arms, out_root, sub, input_paths=input_paths(specs))
    log("Pre-run gate:\n" + lg.format_checks(checks))
    lg.enforce(checks)
    man_sha = lg.hashlib.sha256((ROOT / lg.BV / "MANIFEST_checksums.json").read_bytes()).hexdigest()
    if args.contrasts:
        return cmd_contrasts(args, sub, checks)
    for arm in arms:
        run_arm(args, sub, arm, checks, man_sha)


if __name__ == "__main__":
    main()
