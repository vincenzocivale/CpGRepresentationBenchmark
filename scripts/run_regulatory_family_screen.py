#!/usr/bin/env python3
"""Validation-only family screen (docs/REGULATORY_FAMILY_SCREEN.md): 4 global-SVD-256 arms, decoder seed 17.

Same protocol as scripts/run_regulatory_selection.py (template functional_pca_genomewide.yaml, frozen patient/locus
protocols, mask seed 17001, validation-only guard, saved predictions). TRAINING (including training.mask_fractions)
and the model are identical to the campaign; only evaluation.mask_fractions is restricted to [0.5].

  generate   write configs/experiments/regulatory_family_screen/<arm>__seed17.yaml
  validate   no-training dry run: configs, stores, cpg_idx equality with protocol, split sizes
  run        train+evaluate sequentially (one GPU job at a time); resumable via .done markers
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
OUT = Path("outputs/regulatory_family_screen_v1")
CONFIG_DIR = Path("configs/experiments/regulatory_family_screen")
SEED = 17
MASK_SEED = 17001
PATIENT_SPLIT_SEED = 20260925
EVAL_FRACTIONS = [0.5]
FIT_LOCI_SHA256_PREFIX = "995edc58"
ARMS = {  # arm: (store, expected number of tracks)
    "regulatory_histone": (REPS / "regulatory_histone__global_svd256__discovery_chr1_19.h5", 1959),
    "regulatory_histone_dnase": (REPS / "regulatory_histone_dnase__global_svd256__discovery_chr1_19.h5", 2492),
    "regulatory_histone_tf": (REPS / "regulatory_histone_tf__global_svd256__discovery_chr1_19.h5", 3618),
    "regulatory_clean": (REPS / "regulatory_clean__global_svd256__discovery_chr1_19.h5", 4151),
}


def build(arm: str) -> dict:
    store, n_tracks = ARMS[arm]
    cfg = yaml.safe_load((ROOT / TEMPLATE).read_text())
    cfg["experiment"]["name"] = f"regulatory_family_screen_{arm}__seed{SEED}"
    cfg["experiment"]["output_root"] = str(OUT / "benchmark")
    cfg["experiment"]["locus_split"] = {"heldout_fraction": 0.0, "seed": MASK_SEED,
                                        "protocol_path": str(CAMPAIGN / "loci_seen.npz")}
    cfg["dataset"].update(patient_protocol=str(CAMPAIGN / "patients.npz"), patient_split_seed=PATIENT_SPLIT_SEED)
    cfg["representation"].update(
        name=arm, store_h5=str(store), source=f"regulatory family screen arm ({n_tracks} tracks, global SVD-256)",
        provenance={"patient_specific": False, "supervision": "none", "role": "family_screen",
                    "n_tracks": n_tracks})
    cfg["training"].update(seed=SEED, num_workers=0)  # training.mask_fractions deliberately untouched
    cfg["evaluation"].update(mask_fractions=EVAL_FRACTIONS, save_predictions=True, mask_seed=MASK_SEED,
                             num_workers=0, patient_view="validation", panel_repeats=1,
                             require_patient_view="validation")
    return cfg


def config_path(arm):
    return ROOT / CONFIG_DIR / f"{arm}__seed{SEED}.yaml"


def generate(args):
    (ROOT / CONFIG_DIR).mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        config_path(arm).write_text(yaml.safe_dump(build(arm), sort_keys=False))
    print(f"wrote {len(ARMS)} configs to {CONFIG_DIR}")


def check_config(cfg, template, arm):
    enforce_patient_view(cfg)
    if cfg["evaluation"].get("require_patient_view") != "validation":
        raise PermissionError("config lacks the validation-only guard")
    if cfg["training"]["mask_fractions"] != template["training"]["mask_fractions"]:
        raise ValueError(f"{arm}: TRAINING mask_fractions differ from the campaign template")
    if [float(x) for x in cfg["evaluation"]["mask_fractions"]] != EVAL_FRACTIONS:
        raise ValueError(f"{arm}: evaluation.mask_fractions must be {EVAL_FRACTIONS}")
    if cfg["evaluation"].get("selection_mask_fraction") != template["evaluation"]["selection_mask_fraction"]:
        raise ValueError(f"{arm}: checkpoint-selection fraction differs from the campaign")
    for section in ("model", "training", "evaluation"):
        a, b = dict(cfg[section]), dict(template[section])
        for k in ("seed", "num_workers", "save_predictions", "mask_seed", "patient_view", "panel_repeats",
                  "require_patient_view", "mask_fractions"):
            a.pop(k, None); b.pop(k, None)
        if a != b:
            raise ValueError(f"{arm}: {section} differs from the campaign template")
    if cfg["training"]["seed"] != SEED or cfg["training"]["num_workers"] != 0:
        raise ValueError("seed/num_workers mismatch")


def validate(args):
    template = yaml.safe_load((ROOT / TEMPLATE).read_text())
    with np.load(ROOT / CAMPAIGN / "patients.npz") as h:
        sizes = {k: len(h[k]) for k in ("train", "validation", "test")}  # sizes only; no test data is read
    seen = np.load(ROOT / CAMPAIGN / "loci_seen.npz")
    train_idx = np.sort(seen["train_cpg_idx"])
    report = {"patients": sizes, "n_loci": len(train_idx), "heldout_loci": len(seen["heldout_cpg_idx"]), "arms": {}}
    ref_idx = None
    for arm, (store, _) in ARMS.items():
        path = config_path(arm)
        if not path.exists():
            raise FileNotFoundError(f"{path}; run `generate` first")
        cfg = yaml.safe_load(path.read_text())
        if cfg != build(arm):
            raise ValueError(f"{path} is stale; re-run generate")
        check_config(cfg, template, arm)
        with h5py.File(ROOT / store, "r") as h:
            idx = h["cpg_idx"][:]
            emb = h["embedding"]
            if ref_idx is None:
                ref_idx = idx
            info = {"shape": list(emb.shape), "dtype": str(emb.dtype),
                    "covers_protocol": bool(np.isin(train_idx, idx).all()),
                    "cpg_idx_identical_to_first": bool(np.array_equal(idx, ref_idx)),
                    "fit_loci_sha256": str(h.attrs.get("fit_loci_sha256")),
                    "feature_set_manifest_hash": str(h.attrs.get("feature_set_manifest_hash"))}
        if emb.shape[1] != 256 or not info["covers_protocol"] or not info["fit_loci_sha256"].startswith(
                FIT_LOCI_SHA256_PREFIX) or not info["cpg_idx_identical_to_first"]:
            raise ValueError(f"{arm}: store violates the contract: {info}")
        report["arms"][arm] = info
    print(json.dumps(report, indent=2))


def run(args):
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "PYTHONPATH": str(ROOT / "src")}
    (ROOT / OUT / "logs").mkdir(parents=True, exist_ok=True)
    for arm in args.only or list(ARMS):
        path = config_path(arm)
        enforce_patient_view(yaml.safe_load(path.read_text()))
        marker = ROOT / OUT / "logs" / f"{arm}__seed{SEED}.done"
        if marker.exists():
            continue
        if args.dry_run:
            print("would run", path)
            continue
        log = ROOT / OUT / "logs" / f"{arm}__seed{SEED}.log"
        with log.open("a") as fh:
            rc = subprocess.run([sys.executable, str(ROOT / "scripts/run_masking_benchmark.py"),
                                 "--config", str(path), "--mode", "all"],
                                cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False).returncode
        if rc:
            raise RuntimeError(f"run failed ({arm}); see {log}")
        marker.write_text("done\n")
        print(f"done {arm}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate").set_defaults(fn=generate)
    sub.add_parser("validate").set_defaults(fn=validate)
    r = sub.add_parser("run")
    r.add_argument("--only", nargs="+", choices=list(ARMS))
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=run)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
