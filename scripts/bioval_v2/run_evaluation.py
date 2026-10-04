"""bioval v2 evaluation driver (launch). Registration: docs/BIOLOGICAL_VALIDATION_V2_LAUNCH_REGISTRATION.md.

  --dry-run     list arms x endpoints, run the gate (report only), check store sha256 + locus coverage + all-zero rows
                (read-only), print per-endpoint exclusion COUNTS and the cost estimate. Computes NO endpoint metric.
  (real run)    one process per arm:  --arm <id> [--endpoint <id> ...]   (or --all = every arm sequentially)
  --contrasts   after all arms finished: paired chromosome-block bootstrap contrasts + Holm (reads only per-arm outputs).

A real run (or --contrasts) refuses to start unless the hard pre-run gate passes (see launch_gate.py). It never writes
outside --out-root, never under data/derived/bioval_v2, configs, src; never reads TCGA or masking-protocol paths; never
runs any freeze/prepare command.
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

from cpg_repr_benchmark.bioval_v2_launch import launch_endpoints as le
from cpg_repr_benchmark.bioval_v2_launch import launch_gate as lg
from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import (
    B_DEFAULT,
    SEED_DEFAULT,
    BlockBootstrap,
    contrast,
    holm_adjust,
)
from cpg_repr_benchmark.bioval_v2_launch.embedding_eval import (
    ARMS,
    REFERENCE_ARM,
    EmbeddingStore,
    exclusion_mask,
    load_arm_specs,
    union_zero_loci,
)

DEFAULT_OUT = ROOT / "outputs" / "biological_validation_v2" / lg.PROTOCOL_TAG

COST_TABLE = """Cost estimate per arm (CPU only, 2 threads, estimated -- nothing has been timed on real embeddings):
  loyfer_profile            ~5-10 min   (386k pairs, 11 statistics x 1000 weighted-kernel replicates)
  microc_H1_intra10kb       ~35-60 min  (2.7M+2.7M pairs: cosine ~2 min, sort 5.4M, 1000 x O(N) weighted AUROC ~120 ms + 4 distance bins)
  microc_HFFc6_intra10kb    ~45-75 min  (3.7M+3.7M pairs, AUROC + delta-cosine)
  microc_H1_inter1Mb        ~2 min
  fantom5_membership        ~10-25 min  (3 logistic probes: 5 folds x (6 C x 4 inner + 1) lbfgs fits on 23k-54k rows x 128-512 dims)
  fantom5_activity_similarity ~3 min
  rt_consensus              ~25-45 min  (16 ridge targets, Gram-based, 408k x 128-512; 32 weighted-Spearman CIs x 1000 reps)
  compartment_E1_probe H1/GM12878 ~10 min each;  pmd_probe ~10 min
  loyfer_marker_knn         ~45-90 min  (exact 408k x 408k cosine top-10 (~85 TFLOP at d=512), then 999 within-chromosome permutations (seconds, vectorised); the most expensive secondary)
  total per arm ~3-5 h wall at 2 threads; 4 arms in parallel x 2 threads = 8 threads, ~4-6 h; contrasts <2 min.
Peak RAM per arm process ~6-8 GB (RT/PMD probes: float32 matrix 0.8 GB + float64 train copies ~1.4 GB + Grams; Micro-C ~3 GB)."""


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


def resolve_endpoints(args):
    eps = args.endpoint or list(le.ENDPOINTS)
    for e in eps:
        if e not in le.ENDPOINTS:
            raise SystemExit(f"unknown endpoint {e}; known: {list(le.ENDPOINTS)}")
    return eps


def input_paths_for_gate(specs):
    return [ROOT / lg.BV / v for v in le.FROZEN_FILES.values()] + [ROOT / s.store_h5 for s in specs.values()]


def load_universe():
    u = pd.read_parquet(ROOT / lg.BV / le.FROZEN_FILES["rt"], columns=["cpg_idx", "chrom"]).sort_values("cpg_idx")
    return u.reset_index(drop=True)


def all_zero_loci(specs, verify_sha, log=log):
    zero, stores = {}, {}
    for a, s in specs.items():
        st = EmbeddingStore(s, ROOT, verify_sha=verify_sha)
        zero[a] = st.zero_row_ids()
        stores[a] = st
        log(f"  zero rows {a}: {len(zero[a])}")
    return zero, stores


def dry_exclusion_counts(zero_union):
    """Counts only (frozen lists read-only; no embedding value used beyond the zero-locus set)."""
    rd = lambda k, cols: pd.read_parquet(ROOT / lg.BV / le.FROZEN_FILES[k], columns=cols)
    out = {}
    d = rd("loyfer_pairs", ["cpg_i", "cpg_j", "stratum"])
    k = exclusion_mask([d.cpg_i.to_numpy(), d.cpg_j.to_numpy()], zero_union)
    out["loyfer_profile"] = {"n": len(d), "excluded": int((~k).sum()), "expected_by_user": 1593}
    for key, eid, exp in (("microc_H1_intra", "microc_H1_intra10kb", 15474), ("microc_HFFc6_intra", "microc_HFFc6_intra10kb", 22205),
                          ("microc_H1_inter", "microc_H1_inter1Mb", None)):
        d = rd(key, ["cpg_i", "cpg_j", "fold", "label"])
        n_all = len(d)
        d = d[d.fold >= 0]
        k = exclusion_mask([d.cpg_i.to_numpy(), d.cpg_j.to_numpy()], zero_union)
        out[eid] = {"n_rows_file": n_all, "n_rows_fold_ge0": len(d), "excluded": int((~k).sum()),
                    "expected_by_user": exp}
    d = rd("f5_activity", ["i", "j"])
    k = exclusion_mask([d.i.to_numpy(), d.j.to_numpy()], zero_union)
    out["fantom5_activity_similarity"] = {"n": len(d), "excluded": int((~k).sum()), "expected_by_user": 176}
    return out


def cmd_dry_run(args, arms, endpoints):
    log("=== bioval v2 evaluation: DRY RUN (no endpoint metric is computed) ===")
    specs = load_arm_specs(ROOT, ARMS)
    log("\nArms x endpoints (stats per endpoint are registered in the registration doc):")
    for a in arms:
        log(f"  {a}: " + ", ".join(f"{e}[{le.ENDPOINTS[e][0]}]" for e in endpoints))
    log("\nHolm family (per contrast; reference arm = " + REFERENCE_ARM + "): " + ", ".join(f"{k}:{v}" for k, v in le.PRIMARY_FAMILY.items()))
    log("Contrasts: " + ", ".join(f"{REFERENCE_ARM} vs {c}" for c in ARMS if c != REFERENCE_ARM))
    out_root = Path(args.out_root)
    log(f"\nOutput root: {out_root} (arm dirs: {', '.join(arms)}; contrasts under contrasts/)")
    log("\nGate (report only in dry-run; a real run needs every check PASS):")
    checks = lg.run_gate(ROOT, arms, out_root, input_paths=input_paths_for_gate(specs), skip_heavy=args.skip_heavy)
    log(lg.format_checks(checks))
    report = {"arms": arms, "endpoints": endpoints, "gate": [c.__dict__ for c in checks]}
    log("\nStores (sha256 verified above unless --skip-heavy), coverage of frozen CpG sets, all-zero rows:")
    pa = lg._load_preexec(ROOT)
    sets = pa.frozen_cpg_sets()
    zero, stores = all_zero_loci(specs, verify_sha=not args.skip_heavy)
    union = union_zero_loci(zero)
    report["zero_rows"] = {a: len(z) for a, z in zero.items()}
    report["zero_union"] = len(union)
    log(f"  union of all-zero loci over the 4 stores: {len(union)}")
    report["coverage"] = {}
    for a in arms:
        st = stores[a]
        cov = {k: pa.coverage(st.ids, v) for k, v in sets.items()}
        miss = {k: c["n_missing"] for k, c in cov.items() if c["n_missing"]}
        report["coverage"][a] = {k: [c["n_needed"], c["n_present"]] for k, c in cov.items()}
        log(f"  {a}: dim={st.dim} dtype={st.raw_dtype} rows={len(st.ids)} cpg_idx_sorted={st.cpg_idx_sorted} "
            f"coverage_missing={miss or 'none'}")
    excl = dry_exclusion_counts(union)
    report["exclusion_counts"] = excl
    log("\nCosine-endpoint exclusion counts (pairs/rows touching a locus all-zero in ANY store):")
    for k, v in excl.items():
        log(f"  {k}: {v}")
    log("\n" + COST_TABLE)
    log("\nProposed launch (after the registration is committed and the gate passes):")
    for a in ARMS:
        log(f"  nice -n 10 ~/miniconda3/envs/cpg-repr-benchmark/bin/python scripts/bioval_v2/run_evaluation.py --arm {a} --threads 2 &")
    log("  wait; nice -n 10 python scripts/bioval_v2/run_evaluation.py --contrasts --all")
    if args.report:
        w = lg.GuardedWriter(ROOT, Path(args.report).parent)
        w.write_json(Path(args.report).name, report)
        log(f"\nreport written to {args.report}")
    log("\nDRY RUN COMPLETE: nothing was computed on embeddings beyond the all-zero-row scan; no output written under the output root.")
    failed = [c.name for c in checks if not c.ok]
    log("Gate status: " + ("ALL PASS" if not failed else "WOULD REFUSE A REAL RUN (failing: " + ", ".join(failed) + ")"))


def write_endpoint(w: lg.GuardedWriter, arm, res: le.EndpointResult):
    body = {"endpoint_id": res.endpoint_id, "class": res.cls, "axis": res.axis, "primary_stat": res.primary_stat,
            "meta": res.meta, "stats": {k: s.as_dict() for k, s in res.stats.items()}}
    w.write_json(f"{arm}/endpoints/{res.endpoint_id}.json", body)
    reps = {k: s.rep for k, s in res.stats.items() if s.rep is not None}
    if reps:
        w.save_npz(f"{arm}/endpoints/{res.endpoint_id}.replicates.npz", **reps)


def assemble_results(w: lg.GuardedWriter, arm):
    d = Path(w.path(arm, "endpoints", "x")).parent
    rows, allj = [], {}
    for f in sorted(d.glob("*.json")):
        b = json.loads(f.read_text())
        allj[b["endpoint_id"]] = b
        for s in b["stats"].values():
            rows.append({"arm": arm, "endpoint": b["endpoint_id"], "endpoint_class": b["class"], "stat": s["name"],
                         "stat_class": s["class"], "value": s["value"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "n": s["n"]})
    w.write_json(f"{arm}/results.json", allj)
    w.write_text(f"{arm}/results.csv", pd.DataFrame(rows).to_csv(index=False))


def cmd_run_arm(args, arm, endpoints, checks, man_sha):
    specs = load_arm_specs(ROOT, ARMS)
    out_root = Path(args.out_root)
    w = lg.GuardedWriter(ROOT, out_root)
    log(f"[{arm}] loading stores + zero-row scan (D2: union over the 4 stores)")
    zero, stores = all_zero_loci(specs, verify_sha=True, log=lambda *a: None)
    union = union_zero_loci(zero)
    store = stores[arm]
    for a in list(stores):
        if a != arm:
            del stores[a]
    uni = load_universe()
    boot = BlockBootstrap(np.unique(uni.chrom.to_numpy().astype(str)), n_boot=args.n_boot, seed=SEED_DEFAULT)
    ctx = le.EvalContext(ROOT, store, union, boot, uni, log=log)
    manifest = lg.build_run_manifest(
        ROOT, arm, store.identity(len(zero[arm])), seed=SEED_DEFAULT, n_boot=args.n_boot,
        boot_fingerprint=boot.fingerprint(), threads=THREADS,
        exclusion_counts={"all_zero_rows_per_store": {a: len(z) for a, z in zero.items()},
                          "union_zero_loci": len(union), "per_endpoint": {}},
        checks=checks, endpoints=endpoints, command=" ".join(sys.argv), manifest_checksums_sha256=man_sha, out_root=out_root)
    w.write_json(f"{arm}/run_manifest.json", manifest)
    for e in endpoints:
        log(f"[{arm}] endpoint {e} ...")
        res = le.ENDPOINTS[e][1](ctx)
        write_endpoint(w, arm, res)
        if "exclusion" in res.meta:
            manifest["exclusion_counts"]["per_endpoint"][e] = res.meta["exclusion"]
        manifest.setdefault("endpoints_done", []).append(e)
        w.write_json(f"{arm}/run_manifest.json", manifest)
        for s in res.stats.values():
            if s.cls in ("primary", "secondary"):
                log(f"    {s.name}: {s.value:.5f} [{s.ci_lo:.5f}, {s.ci_hi:.5f}] n={s.n}")
    manifest["freeze_modules_imported"] = lg.freeze_modules_imported()
    manifest["no_freeze_command_used"] = not manifest["freeze_modules_imported"]
    manifest["finished_utc"] = lg.datetime.now(lg.timezone.utc).isoformat()
    w.write_json(f"{arm}/run_manifest.json", manifest)
    assemble_results(w, arm)
    log(f"[{arm}] done -> {out_root / arm}")


def cmd_contrasts(args, arms):
    out_root = Path(args.out_root)
    w = lg.GuardedWriter(ROOT, out_root)
    comps = [a for a in arms if a != REFERENCE_ARM] or [a for a in ARMS if a != REFERENCE_ARM]
    ref_dir = out_root / REFERENCE_ARM / "endpoints"
    fps = {}
    for comp in comps:
        rows, per_ep = [], {}
        for eid in le.ENDPOINTS:
            fa, fb = ref_dir / f"{eid}.json", out_root / comp / "endpoints" / f"{eid}.json"
            if not (fa.exists() and fb.exists()):
                continue
            ja, jb = json.loads(fa.read_text()), json.loads(fb.read_text())
            ra = np.load(ref_dir / f"{eid}.replicates.npz")
            rb = np.load(out_root / comp / "endpoints" / f"{eid}.replicates.npz")
            for name, sa in ja["stats"].items():
                if name not in jb["stats"] or name not in ra.files or name not in rb.files:
                    continue
                sb = jb["stats"][name]
                c = contrast(sa["value"], ra[name], sb["value"], rb[name])
                is_primary = le.PRIMARY_FAMILY.get(eid) == name
                row = {"reference": REFERENCE_ARM, "comparator": comp, "endpoint": eid, "stat": name,
                       "stat_class": sa["class"], "ref_value": sa["value"], "comp_value": sb["value"],
                       "sign_convention": "delta = regulatory - comparator; higher-is-better metric, so delta>0 = regulatory better",
                       "in_holm_family": is_primary, **c}
                rows.append(row)
                if is_primary:
                    per_ep[eid] = row
        fam = {e: per_ep[e]["p"] for e in per_ep}
        missing = sorted(set(le.PRIMARY_FAMILY) - set(per_ep))
        adj = holm_adjust(fam) if fam else {}
        for e, r in per_ep.items():
            r["holm_p_adjusted"] = adj[e]
            r["holm_alpha"] = 0.05
            r["supports_regulatory"] = bool(adj[e] < 0.05 and r["delta"] > 0)
        for r in rows:
            r.setdefault("holm_p_adjusted", None)
        out = {"reference": REFERENCE_ARM, "comparator": comp, "bootstrap": {"B": args.n_boot, "seed": SEED_DEFAULT},
               "holm_family_missing_endpoints": missing,
               "note": "Holm over the 4 primary endpoints only; every other row is descriptive (raw p, no claim)",
               "rows": rows}
        w.write_json(f"contrasts/{REFERENCE_ARM}__vs__{comp}.json", out)
        w.write_text(f"contrasts/{REFERENCE_ARM}__vs__{comp}.csv", pd.DataFrame(rows).to_csv(index=False))
        log(f"contrast {REFERENCE_ARM} vs {comp}: {len(rows)} rows; Holm family missing: {missing or 'none'}")
        for e, r in per_ep.items():
            log(f"   {e}: delta={r['delta']:.5f} CI=[{r['ci_lo']:.5f},{r['ci_hi']:.5f}] p={r['p']:.4f} holm={r['holm_p_adjusted']:.4f}")
    return fps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--endpoint", action="append")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--contrasts", action="store_true")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=B_DEFAULT)
    ap.add_argument("--skip-heavy", action="store_true", help="dry-run only: skip sha256 passes (a real run never allows it)")
    ap.add_argument("--report", default=None, help="dry-run: write a JSON report here (guarded: never under a protected path)")
    args = ap.parse_args()
    if args.skip_heavy and not args.dry_run:
        raise SystemExit("--skip-heavy is only allowed with --dry-run")
    if args.n_boot != B_DEFAULT and not args.dry_run:
        raise SystemExit(f"registered B is {B_DEFAULT}; a different --n-boot is only allowed in --dry-run")
    arms = resolve_arms(args) if (args.arm or args.all or not args.contrasts) else list(ARMS)
    endpoints = resolve_endpoints(args)
    if args.dry_run:
        return cmd_dry_run(args, arms, endpoints)
    specs = load_arm_specs(ROOT, ARMS)
    checks = lg.run_gate(ROOT, arms, Path(args.out_root), input_paths=input_paths_for_gate(specs))
    log("Pre-run gate:\n" + lg.format_checks(checks))
    lg.enforce(checks)
    man_sha = lg.hashlib.sha256((ROOT / lg.BV / "MANIFEST_checksums.json").read_bytes()).hexdigest()
    if args.contrasts:
        return cmd_contrasts(args, arms)
    for arm in arms:
        cmd_run_arm(args, arm, endpoints, checks, man_sha)


if __name__ == "__main__":
    main()
