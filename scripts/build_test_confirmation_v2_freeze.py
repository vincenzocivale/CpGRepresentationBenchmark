#!/usr/bin/env python3
"""Build the DRAFT freeze manifest + 21 test-evaluation configs for tcga_posthoc_freeze_v2. Reads only run artifacts (file bytes,
history.json, validation metrics); never opens methylation values, patient rows or test outputs. Writes only into
configs/experiments/test_confirmation_v2/. The manifest is always written as freeze_state: draft, test_set_authorized: false."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from cpg_repr_benchmark.experiments import test_confirmation_v2 as t2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / t2.EVAL_CONFIG_DIR_REL


def eval_config(resolved: dict, rid: str) -> dict:
    cfg = copy.deepcopy(resolved)
    ev = cfg["evaluation"]
    ev.pop("require_patient_view", None)
    ev.update({"patient_view": "test", "output_layout": "split_dirs", "allow_overwrite_split_dir": False, "num_workers": 0,
               "preserve_run_files": True, "test_authorization": t2.POLICY, "test_freeze_manifest": t2.MANIFEST_REL,
               "test_freeze_run_id": rid})
    return cfg


def diff_paths(a, b, prefix=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{prefix}{k}")
            else:
                out += diff_paths(a[k], b[k], f"{prefix}{k}.")
    elif a != b:
        out.append(prefix.rstrip("."))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="outputs/reconstruction_phase1_analysis/summary.json")
    args = ap.parse_args()
    used = json.loads((ROOT / args.summary).read_text())["runs_used"]
    pick = {(r["arm"], r["seed"]): r["dir"] for r in used if r["arm"] in t2.PANEL_ARMS}
    # Sei is not in the phase-1 summary: discover completed runs on hal (done marker + 120-epoch history + validation metrics)
    sei_root = ROOT / "outputs/sei_phase1/benchmark/masking/tcga_array_genomewide/sei_latent_native/native_frozen"
    for seed in t2.SEEDS:
        for d in sorted(sei_root.glob(f"seed_{seed}/*")):
            ok = ((ROOT / f"outputs/logs/remote_queue_done/sei_latent_s{seed}").exists() and (d / "checkpoints/best.pt").is_file()
                  and (d / "history.json").is_file() and len(json.loads((d / "history.json").read_text())) == 120
                  and (d / "evaluation/validation/seen/mask_0.50/metrics.json").is_file())
            if ok:
                pick[("sei_latent_native", seed)] = str(d.relative_to(ROOT))
    runs, issues, store_cache = {}, [], {}
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in t2.PANEL_ARMS:
        for seed in t2.SEEDS:
            rid = t2.run_id(arm, seed)
            if (arm, seed) not in pick:
                runs[rid] = {"arm": arm, "seed": seed, "status": "pending"}
                issues.append(f"{rid}: PENDING (not on hal / not complete)"); continue
            rd = ROOT / pick[(arm, seed)]
            best = rd / "checkpoints" / "best.pt"
            if not best.is_file():
                issues.append(f"{rid}: best.pt MISSING"); runs[rid] = {"arm": arm, "seed": seed, "status": "missing_best"}; continue
            resolved = yaml.safe_load((rd / "resolved_config.yaml").read_text())
            assert resolved["representation"].get("mode", "precomputed") == "precomputed", rid
            hist = json.loads((rd / "history.json").read_text())
            best_epoch = min(range(len(hist)), key=lambda i: hist[i]["validation_mse"])  # first strict minimum
            import torch
            ck = torch.load(best, map_location="cpu", weights_only=False)
            if ck["epoch"] != best_epoch or len(hist) != 120:
                issues.append(f"{rid}: checkpoint epoch {ck['epoch']} vs history argmin {best_epoch}, epochs={len(hist)}")
            if abs(ck["metrics"]["validation_mse"] - hist[best_epoch]["validation_mse"]) > 1e-12:
                issues.append(f"{rid}: best.pt metrics differ from history")
            vm = json.loads((rd / "evaluation/validation/seen/mask_0.50/metrics.json").read_text())
            store = resolved["representation"]["store_h5"]
            sp = ROOT / store
            if store not in store_cache:
                store_cache[store] = t2.sha256_file(sp)
            prov = resolved["representation"].get("provenance", {}).get("sha256")
            if prov and prov != store_cache[store]:
                issues.append(f"{rid}: store sha256 differs from config provenance ({prov[:10]} vs {store_cache[store][:10]})")
            ecfg = eval_config(resolved, rid)
            unexpected = [p for p in diff_paths(resolved, ecfg) if not (p.startswith("evaluation.") and p.split(".")[1] in t2.EVAL_KEYS_ALLOWED_TO_DIFFER)]
            if unexpected:
                issues.append(f"{rid}: unexpected config differences {unexpected}")
            epath = OUT / f"{rid}.yaml"
            epath.write_text(yaml.safe_dump(ecfg, sort_keys=False))
            rel = lambda n, rd=rd: str((rd / n).relative_to(ROOT))
            runs[rid] = {
                "arm": arm, "seed": seed, "status": "complete", "run_dir": str(rd.relative_to(ROOT)),
                "best_pt": rel("checkpoints/best.pt"), "best_sha256": t2.sha256_file(best), "best_epoch": int(ck["epoch"]),
                "resolved_config": rel("resolved_config.yaml"), "resolved_config_sha256": t2.sha256_file(rd / "resolved_config.yaml"),
                "store_h5": store, "store_sha256": store_cache[store],
                "patient_split": rel("patient_split.npz"), "patient_split_sha256": t2.sha256_file(rd / "patient_split.npz"),
                "locus_split": rel("locus_split.npz"), "locus_split_sha256": t2.sha256_file(rd / "locus_split.npz"),
                "prior_logit": rel("prior_logit.npy"), "prior_logit_sha256": t2.sha256_file(rd / "prior_logit.npy"),
                "validation_mse_mask050": float(vm["mse"]),
                "eval_config": str(epath.relative_to(ROOT)), "eval_config_sha256": t2.sha256_file(epath),
                "eval_config_canonical_sha256": t2.canonical_cfg_sha256(ecfg),
            }
    glob = {"patient_protocol": "outputs/encode_atlas_v1/patients.npz", "loci_protocol": "outputs/encode_atlas_v1/loci_seen.npz"}
    manifest = {
        "name": "tcga_posthoc_freeze_v2", "freeze_state": "draft", "test_set_authorized": False,
        "policy": t2.POLICY, "panel_arms": t2.PANEL_ARMS, "seeds": t2.SEEDS, "variant": "latent / locus (no functional, no SVD)",
        "checkpoint": "best.pt (min validation MSE at mask 0.50, 120 epochs, selected before any test read)",
        "primary_endpoint": {"metric": "mse", "mask_fraction": 0.5, "patient_view": "test"},
        "protocol_files": {k: {"path": v, "sha256": t2.sha256_file(ROOT / v)} for k, v in glob.items()},
        "runs": runs, "build_issues": issues,
    }
    (OUT / "freeze_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"runs={len(runs)} complete={sum(r.get('status')=='complete' for r in runs.values())} issues={len(issues)}")
    for i in issues:
        print("ISSUE", i)


if __name__ == "__main__":
    main()
