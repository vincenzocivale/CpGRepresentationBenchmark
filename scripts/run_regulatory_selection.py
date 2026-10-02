#!/usr/bin/env python3
"""Validation-only selection of the regulatory CpG representation (docs/REGULATORY_SELECTION_PROTOCOL.md).

Reuses the ENCODE-campaign masking protocol verbatim (template config, frozen patient protocol,
frozen seen-locus protocol, mask seed 17001, 15/30/50/70/90% masks, 2048-locus panels, saved predictions).
Every generated config carries ``evaluation.require_patient_view: validation``; the masking runner refuses
to start otherwise, and this script re-checks before launching.

  generate   write configs/experiments/regulatory_selection/<arm>__seed<seed>.yaml
  validate   no-training dry run: configs, stores, cpg_idx coverage, split sizes
  run        train+evaluate sequentially (one GPU job at a time); resumable
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cpg_repr_benchmark.experiments.guards import enforce_patient_view

CAMPAIGN = Path("outputs/encode_atlas_v1")
TEMPLATE = Path("configs/experiments/masking/functional_pca_genomewide.yaml")
REPS = Path("data/cache/representations")
OUT = Path("outputs/regulatory_selection_v1")
CONFIG_DIR = Path("configs/experiments/regulatory_selection")
SEEDS = (17, 42, 97)
MASK_SEED = 17001
PATIENT_SPLIT_SEED = 20260925
ARMS = {
    # name: (store, role)
    "regulatory_clean_global_svd": (REPS / "regulatory_clean__global_svd256__discovery_chr1_19.h5", "candidate"),
    "regulatory_clean_block_svd": (REPS / "regulatory_clean__block_svd256__discovery_chr1_19.h5", "candidate"),
    "regulatory_clean_block_svd_egm": (REPS / "regulatory_clean__block_svd256_egm__discovery_chr1_19.h5", "candidate"),
    "regulatory_histone_global_svd": (REPS / "regulatory_histone__global_svd256__discovery_chr1_19.h5", "control"),
    # Campaign stores re-used byte-for-byte (read only): `full` ENCODE functional SVD-256 and CpGPT-large SVD-256.
    "functional_annotations_pca": (CAMPAIGN / "embeddings/full_a18b869b.h5", "control"),
    "cpgpt_locus_large": (CAMPAIGN / "embeddings/fm_cpgpt_locus_large_d3474a48.h5", "control"),
}


def build(arm: str, seed: int) -> dict:
    store, role = ARMS[arm]
    cfg = yaml.safe_load((ROOT / TEMPLATE).read_text())
    key = f"{arm}__seed{seed}"
    cfg["experiment"]["name"] = f"regulatory_selection_{key}"
    cfg["experiment"]["output_root"] = str(OUT / "benchmark")
    cfg["experiment"]["locus_split"] = {"heldout_fraction": 0.0, "seed": MASK_SEED,
                                        "protocol_path": str(CAMPAIGN / "loci_seen.npz")}
    cfg["dataset"].update(patient_protocol=str(CAMPAIGN / "patients.npz"), patient_split_seed=PATIENT_SPLIT_SEED)
    cfg["representation"].update(
        name=arm, store_h5=str(store), source=f"regulatory selection arm ({role})",
        provenance={"patient_specific": False, "supervision": "none", "role": role})
    cfg["training"].update(seed=seed, num_workers=0)
    cfg["evaluation"].update(save_predictions=True, mask_seed=MASK_SEED, num_workers=0,
                             patient_view="validation", panel_repeats=1, require_patient_view="validation")
    return cfg


def config_path(arm, seed):
    return ROOT / CONFIG_DIR / f"{arm}__seed{seed}.yaml"


def generate(args):
    (ROOT / CONFIG_DIR).mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        for seed in SEEDS:
            config_path(arm, seed).write_text(yaml.safe_dump(build(arm, seed), sort_keys=False))
    print(f"wrote {len(ARMS) * len(SEEDS)} configs to {CONFIG_DIR}")


def check_config(cfg, template, arm, seed):
    enforce_patient_view(cfg)
    if cfg["evaluation"].get("require_patient_view") != "validation":
        raise PermissionError("config lacks the validation-only guard")
    # Everything except the arm identity/seed must equal the campaign template.
    for section in ("model", "training", "evaluation"):
        a, b = dict(cfg[section]), dict(template[section])
        for k in ("seed", "num_workers", "save_predictions", "mask_seed", "patient_view", "panel_repeats",
                  "require_patient_view"):
            a.pop(k, None); b.pop(k, None)
        if a != b:
            raise ValueError(f"{arm}/{seed}: {section} differs from the campaign template")
    if cfg["training"]["seed"] != seed:
        raise ValueError("seed mismatch")


def validate(args):
    template = yaml.safe_load((ROOT / TEMPLATE).read_text())
    with np.load(ROOT / CAMPAIGN / "patients.npz") as h:
        sizes = {k: len(h[k]) for k in ("train", "validation", "test")}
    seen = np.load(ROOT / CAMPAIGN / "loci_seen.npz")
    train_idx = np.sort(seen["train_cpg_idx"])
    report = {"patients": sizes, "n_loci": len(train_idx), "heldout_loci": len(seen["heldout_cpg_idx"]),
              "arms": {}}
    reg = None
    ref_idx = None
    for arm, (store, _) in ARMS.items():
        for seed in SEEDS:
            path = config_path(arm, seed)
            if not path.exists():
                raise FileNotFoundError(f"{path}; run `generate` first")
            cfg = yaml.safe_load(path.read_text())
            if cfg != build(arm, seed):
                raise ValueError(f"{path} is stale relative to the protocol; re-run generate")
            check_config(cfg, template, arm, seed)
        with h5py.File(ROOT / store, "r") as h:
            idx = h["cpg_idx"][:]
            emb = h["embedding"]
            info = {"shape": list(emb.shape), "dtype": str(emb.dtype),
                    "covers_protocol": bool(np.isin(train_idx, idx).all()),
                    "cpg_idx_identical_to_first": None}
            if ref_idx is None:
                ref_idx = idx
            info["cpg_idx_identical_to_first"] = bool(np.array_equal(idx, ref_idx))
            if emb.shape[1] != 256 or not info["covers_protocol"]:
                raise ValueError(f"{arm}: store violates the 256-d/full-coverage contract: {info}")
            if args.zero_stats:
                import pandas as pd
                if reg is None:
                    reg = pd.read_parquet(ROOT / "data/cpg/registries/array_cpg_map.parquet")
                chrom = reg.set_index("cpg_idx").chr.reindex(idx).to_numpy()
                nz = np.zeros(len(idx), bool)
                for i in range(0, len(idx), 50000):
                    nz[i:i + 50000] = np.any(emb[i:i + 50000] != 0, axis=1)
                late = np.isin(chrom, ["chr20", "chr21", "chr22"])
                info.update(zero_rows=int((~nz).sum()), zero_rows_chr20_22=int((~nz & late).sum()),
                            n_chr20_22=int(late.sum()))
        report["arms"][arm] = info
    report["n_configs"] = len(ARMS) * len(SEEDS)
    print(json.dumps(report, indent=2))


def run(args):
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "PYTHONPATH": str(ROOT / "src")}
    (ROOT / OUT / "logs").mkdir(parents=True, exist_ok=True)
    arms = args.only or list(ARMS)
    for seed in args.seeds:
        for arm in arms:
            path = config_path(arm, seed)
            cfg = yaml.safe_load(path.read_text())
            enforce_patient_view(cfg)
            marker = ROOT / OUT / "logs" / f"{arm}__seed{seed}.done"
            if marker.exists():
                continue
            if args.dry_run:
                print("would run", path)
                continue
            log = ROOT / OUT / "logs" / f"{arm}__seed{seed}.log"
            with log.open("a") as fh:
                rc = subprocess.run([sys.executable, str(ROOT / "scripts/run_masking_benchmark.py"),
                                     "--config", str(path), "--mode", "all"],
                                    cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False).returncode
            if rc:
                raise RuntimeError(f"run failed ({arm}, seed {seed}); see {log}")
            marker.write_text("done\n")
            print(f"done {arm} seed={seed}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate").set_defaults(fn=generate)
    v = sub.add_parser("validate"); v.add_argument("--zero-stats", action="store_true"); v.set_defaults(fn=validate)
    r = sub.add_parser("run")
    r.add_argument("--only", nargs="+", choices=list(ARMS))
    r.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), choices=SEEDS)
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=run)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
