#!/usr/bin/env python3
"""Validation-only confirmation (docs/REGULATORY_CONFIRMATION_PROTOCOL.md): 3 arms x 3 decoder seeds, 9 runs.

Arms: regulatory_histone, regulatory_histone_dnase, regulatory_clean (global SVD-256, fit on discovery_chr1_19).
Training regime = ENCODE campaign template (functional_pca_genomewide.yaml): batch 8, lr 1e-4, wd 1e-4, fp16, same
model, same training mask fractions, selection fraction 0.5. Allowed deviations (shared by all arms and seeds):
  * training.epochs = MAX_EPOCHS plus training.early_stopping (patience / min_delta_rel on 50% validation MSE);
    the optimizer has NO LR schedule, so the epoch cap does not alter any schedule;
  * training.num_workers = 8 (throughput only: masks are keyed on seed/epoch/row/fraction, the batch order is
    drawn in the main process; see the protocol document);
  * evaluation.mask_fractions = [0.5], save_predictions, validation-only guard.

  generate   write configs/experiments/regulatory_confirm/<arm>__seed<seed>.yaml
  validate   no-training check: configs vs campaign template, stores, cpg_idx equality, split sizes
  run        train+evaluate sequentially, one GPU job at a time, order seed-major (17: 3 arms, then 42, then 97);
             resumable via .done markers      [--only ARM...] [--seeds S...] [--dry-run]
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
OUT = Path("outputs/regulatory_confirm_v1")
CONFIG_DIR = Path("configs/experiments/regulatory_confirm")
SEEDS = [17, 42, 97]
MASK_SEED = 17001
PATIENT_SPLIT_SEED = 20260925
EVAL_FRACTIONS = [0.5]
BATCH_SIZE = 8            # campaign value (the family screen used 32)
NUM_WORKERS = 8           # throughput only (campaign resolved configs used 0); see protocol
MAX_EPOCHS = 80
EARLY_STOPPING = {"patience": 10, "min_delta_rel": 1.0e-4}
FIT_LOCI_SHA256_PREFIX = "995edc58"
ARMS = {  # arm: (store, expected number of tracks)
    "regulatory_histone": (REPS / "regulatory_histone__global_svd256__discovery_chr1_19.h5", 1959),
    "regulatory_histone_dnase": (REPS / "regulatory_histone_dnase__global_svd256__discovery_chr1_19.h5", 2492),
    "regulatory_clean": (REPS / "regulatory_clean__global_svd256__discovery_chr1_19.h5", 4151),
}
# keys that may differ from the campaign template, per section
ALLOWED = {
    "model": set(),
    "training": {"seed", "epochs", "num_workers", "early_stopping"},
    "evaluation": {"mask_fractions", "save_predictions", "mask_seed", "patient_view", "panel_repeats",
                   "require_patient_view", "num_workers"},
}


def build(arm: str, seed: int) -> dict:
    store, n_tracks = ARMS[arm]
    cfg = yaml.safe_load((ROOT / TEMPLATE).read_text())
    cfg["experiment"]["name"] = f"regulatory_confirm_{arm}__seed{seed}"
    cfg["experiment"]["output_root"] = str(OUT / "benchmark")
    cfg["experiment"]["locus_split"] = {"heldout_fraction": 0.0, "seed": MASK_SEED,
                                        "protocol_path": str(CAMPAIGN / "loci_seen.npz")}
    cfg["dataset"].update(patient_protocol=str(CAMPAIGN / "patients.npz"), patient_split_seed=PATIENT_SPLIT_SEED)
    cfg["representation"].update(
        name=arm, store_h5=str(store), source=f"regulatory confirmation arm ({n_tracks} tracks, global SVD-256)",
        provenance={"patient_specific": False, "supervision": "none", "role": "family_confirmation",
                    "n_tracks": n_tracks})
    cfg["training"].update(seed=seed, epochs=MAX_EPOCHS, num_workers=NUM_WORKERS,
                           early_stopping=dict(EARLY_STOPPING))  # batch_size, lr, wd, masks: template
    cfg["evaluation"].update(mask_fractions=EVAL_FRACTIONS, save_predictions=True, mask_seed=MASK_SEED,
                             num_workers=0, patient_view="validation", panel_repeats=1,
                             require_patient_view="validation")
    return cfg


def config_path(arm: str, seed: int) -> Path:
    return ROOT / CONFIG_DIR / f"{arm}__seed{seed}.yaml"


def jobs(seeds=None, only=None):
    """Seed-major order: all arms of seed 17, then 42, then 97."""
    return [(arm, seed) for seed in (seeds or SEEDS) for arm in (only or list(ARMS))]


def generate(args):
    (ROOT / CONFIG_DIR).mkdir(parents=True, exist_ok=True)
    for arm, seed in jobs():
        config_path(arm, seed).write_text(yaml.safe_dump(build(arm, seed), sort_keys=False))
    print(f"wrote {len(jobs())} configs to {CONFIG_DIR}")


def check_config(cfg: dict, template: dict, arm: str, seed: int) -> None:
    enforce_patient_view(cfg)
    if cfg["evaluation"].get("require_patient_view") != "validation" or cfg["evaluation"].get("patient_view") != "validation":
        raise PermissionError(f"{arm}/{seed}: config lacks the validation-only guard")
    if [float(x) for x in cfg["evaluation"]["mask_fractions"]] != EVAL_FRACTIONS:
        raise ValueError(f"{arm}/{seed}: evaluation.mask_fractions must be {EVAL_FRACTIONS}")
    if cfg["training"]["mask_fractions"] != template["training"]["mask_fractions"]:
        raise ValueError(f"{arm}/{seed}: TRAINING mask_fractions differ from the campaign template")
    if cfg["evaluation"].get("selection_mask_fraction") != template["evaluation"]["selection_mask_fraction"]:
        raise ValueError(f"{arm}/{seed}: checkpoint-selection fraction differs from the campaign")
    if cfg["training"]["batch_size"] != BATCH_SIZE or cfg["evaluation"]["batch_size"] != BATCH_SIZE:
        raise ValueError(f"{arm}/{seed}: batch size must be {BATCH_SIZE}")
    for section, allowed in ALLOWED.items():
        a = {k: v for k, v in cfg[section].items() if k not in allowed}
        b = {k: v for k, v in template[section].items() if k not in allowed}
        if a != b:
            raise ValueError(f"{arm}/{seed}: {section} differs from the campaign template outside {sorted(allowed)}")
    for section in cfg:  # no unexpected top-level sections
        if section not in template:
            raise ValueError(f"{arm}/{seed}: unexpected section {section}")
    t = cfg["training"]
    if t["seed"] != seed or t["num_workers"] != NUM_WORKERS or t["epochs"] != MAX_EPOCHS \
            or t["early_stopping"] != EARLY_STOPPING:
        raise ValueError(f"{arm}/{seed}: seed/num_workers/epochs/early_stopping mismatch")
    if cfg["evaluation"]["mask_seed"] != MASK_SEED or cfg["evaluation"]["panel_size"] != template["evaluation"]["panel_size"]:
        raise ValueError(f"{arm}/{seed}: mask seed / panel size differ")
    if not cfg["evaluation"].get("save_predictions"):
        raise ValueError(f"{arm}/{seed}: save_predictions must be true")


def validate(args):
    template = yaml.safe_load((ROOT / TEMPLATE).read_text())
    with np.load(ROOT / CAMPAIGN / "patients.npz") as h:
        sizes = {k: len(h[k]) for k in ("train", "validation", "test")}  # sizes only; no test data is read
    seen = np.load(ROOT / CAMPAIGN / "loci_seen.npz")
    train_idx = np.sort(seen["train_cpg_idx"])
    report = {"patients": sizes, "n_loci": len(train_idx), "heldout_loci": len(seen["heldout_cpg_idx"]),
              "n_configs": 0, "arms": {}}
    if len(seen["heldout_cpg_idx"]) != 0:
        raise ValueError("loci_seen.npz must have no held-out loci")
    ref_idx = None
    for arm, seed in jobs():
        path = config_path(arm, seed)
        if not path.exists():
            raise FileNotFoundError(f"{path}; run `generate` first")
        cfg = yaml.safe_load(path.read_text())
        if cfg != build(arm, seed):
            raise ValueError(f"{path} is stale; re-run generate")
        check_config(cfg, template, arm, seed)
        report["n_configs"] += 1
        if arm in report["arms"]:
            continue
        with h5py.File(ROOT / ARMS[arm][0], "r") as h:
            idx = h["cpg_idx"][:]
            emb = h["embedding"]
            ref_idx = idx if ref_idx is None else ref_idx
            info = {"shape": list(emb.shape), "dtype": str(emb.dtype),
                    "covers_protocol": bool(np.isin(train_idx, idx).all()),
                    "cpg_idx_identical_to_first": bool(np.array_equal(idx, ref_idx)),
                    "fit_loci_sha256": str(h.attrs.get("fit_loci_sha256"))}
        if emb.shape[1] != 256 or not info["covers_protocol"] or not info["cpg_idx_identical_to_first"] \
                or not info["fit_loci_sha256"].startswith(FIT_LOCI_SHA256_PREFIX):
            raise ValueError(f"{arm}: store violates the contract: {info}")
        report["arms"][arm] = info
    print(json.dumps(report, indent=2))


def run(args):
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "PYTHONPATH": str(ROOT / "src")}
    template = yaml.safe_load((ROOT / TEMPLATE).read_text())
    (ROOT / OUT / "logs").mkdir(parents=True, exist_ok=True) if not args.dry_run else None
    for arm, seed in jobs(args.seeds, args.only):
        path = config_path(arm, seed)
        cfg = yaml.safe_load(path.read_text())
        check_config(cfg, template, arm, seed)
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
        print(f"done {arm} seed {seed}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate").set_defaults(fn=generate)
    sub.add_parser("validate").set_defaults(fn=validate)
    r = sub.add_parser("run")
    r.add_argument("--only", nargs="+", choices=list(ARMS))
    r.add_argument("--seeds", nargs="+", type=int, choices=SEEDS)
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=run)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
