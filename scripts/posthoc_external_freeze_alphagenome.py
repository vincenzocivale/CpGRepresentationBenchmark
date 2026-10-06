#!/usr/bin/env python3
"""Pre-test freeze record for the POST-HOC AlphaGenome arm on GSE40279 (run BEFORE any test access).

Records per seed: run dir, selected checkpoint (sha256 + epoch), configs (+sha256), prior, store, patient/locus protocol hashes
checked against the frozen external manifest, and asserts that no evaluation/test directory exists. The record is committed;
the one-shot test script refuses unless the committed record matches the files on disk.
  python scripts/posthoc_external_freeze_alphagenome.py
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from cpg_repr_benchmark.external.io import sha256_file
from cpg_repr_benchmark.external.freeze import MANIFEST_REL

B = ROOT / "outputs/external_reconstruction_posthoc_alphagenome/benchmark/masking/external_gse40279_v1/alphagenome_locus/native_frozen"
OUT = ROOT / "configs/posthoc/external_alphagenome_freeze.json"
STORE = "data/cache/representations/alphagenome_locus.h5"
STORE_SHA = "3906d888964fe71e303e2437ae64445b3f91803dcc04d15484fc5d5565850e00"
man = json.loads((ROOT / MANIFEST_REL).read_text())
files = {k: (ROOT / man["files"][k]) for k in ("patient_protocol_npz", "locus_protocol_npz", "dataset_h5")}
shas = {k: sha256_file(p) for k, p in files.items()}
assert all(shas[k] == man["files_sha256"][k] for k in shas), "frozen external data differ from the freeze manifest"
assert sha256_file(ROOT / STORE) == STORE_SHA
rec = {"schema": "posthoc_external_alphagenome_freeze/1", "posthoc_comparator": True, "arm": "alphagenome_locus",
       "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
       "statement": ("AlphaGenome was added AFTER the original-arm GSE40279 test results were known. The AlphaGenome test set has never "
                     "been accessed at the time of this record. Exactly one test evaluation per frozen seed/checkpoint on the original "
                     "66 test subjects and the same masking fractions; no retraining, reselection or post-test tuning."),
       "test_accessed": False, "frozen_manifest": MANIFEST_REL, "frozen_data_sha256": shas, "mask_fractions": man["mask_fractions"],
       "store_h5": STORE, "store_sha256": STORE_SHA, "seeds": {}}
for seed in (17, 42, 97):
    runs = sorted((B / f"seed_{seed}").iterdir())
    assert len(runs) == 1, f"seed {seed}: expected exactly one run dir, found {len(runs)}"
    d = runs[0]
    assert (d / "evaluation/validation/summary.json").is_file() and not (d / "evaluation/test").exists(), f"seed {seed}: bad state"
    cfgp = ROOT / f"configs/experiments/posthoc_alphagenome/external/alphagenome_locus__seed{seed}.yaml"
    ck = torch.load(d / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    rec["seeds"][str(seed)] = {"run_dir": str(d.relative_to(ROOT)), "config": str(cfgp.relative_to(ROOT)), "config_sha256": sha256_file(cfgp),
        "resolved_config_sha256": sha256_file(d / "resolved_config.yaml"), "best_pt_sha256": sha256_file(d / "checkpoints/best.pt"),
        "best_epoch_0based": int(ck["epoch"]), "prior_logit_sha256": sha256_file(d / "prior_logit.npy"),
        "patient_split_sha256": sha256_file(d / "patient_split.npz"), "locus_split_sha256": sha256_file(d / "locus_split.npz"),
        "experiment_code_git_commit": json.loads((d / "experiment.json").read_text())["code_git_commit"]}
OUT.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
print(json.dumps({s: (v["best_epoch_0based"], v["best_pt_sha256"][:12]) for s, v in rec["seeds"].items()}))
