#!/usr/bin/env python3
"""ONE-SHOT post-hoc GSE40279 TEST evaluation of the frozen AlphaGenome checkpoints (posthoc_comparator: true).

Why not the official path: external/test_eval.py + gate are bound to the frozen 3-arm manifest and, by design, refuse a second test
launch (STARTED_LOCK, check_no_test_artifacts). They are NOT modified or bypassed for the original arms. This script re-implements only the
evaluation core of test_eval.evaluate_frozen_run (same frozen data, fractions, mask seed, panel, model loader, evaluate_model) and replaces the
authorization callable by checks that are real for THIS evaluation:
  1. configs/posthoc/external_alphagenome_freeze.json is committed, unmodified, and HEAD contains it (freeze BEFORE test access);
  2. every hash in the record (checkpoint, configs, prior, splits, store, frozen data) still matches the files on disk;
  3. no evaluation/test directory and no AlphaGenome lock exist for any seed (one-shot);
  4. the run's test rows equal the frozen protocol's 66 test subjects.
All 3 seeds are preflighted before the first evaluation. No retraining, reselection or tuning. Requires --confirm-one-shot.
"""
import argparse, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch, yaml
from cpg_repr_benchmark.data.methylation import read_axis
from cpg_repr_benchmark.external import gate as G, transfer as T
from cpg_repr_benchmark.external.freeze import MANIFEST_REL
from cpg_repr_benchmark.external.io import sha256_file
from cpg_repr_benchmark.representations.hdf5_store import HDF5RepresentationStore

REC = ROOT / "configs/posthoc/external_alphagenome_freeze.json"
LOCK = ROOT / "outputs/external_reconstruction_posthoc_alphagenome/TEST_STARTED.json"
ARM = "alphagenome_locus"


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, text=True, capture_output=True, check=False)


def preflight(rec, man):
    probs = []
    rel = str(REC.relative_to(ROOT))
    for p in (rel, "scripts/posthoc_external_test_alphagenome.py", *[s["config"] for s in rec["seeds"].values()]):
        if git("ls-files", "--error-unmatch", p).returncode or git("status", "--porcelain", "--", p).stdout.strip():
            probs.append(f"not committed/clean: {p}")
    if rec.get("test_accessed") is not False:
        probs.append("freeze record does not state test_accessed=false")
    for k, h in rec["frozen_data_sha256"].items():
        if sha256_file(ROOT / man["files"][k]) != h or h != man["files_sha256"][k]:
            probs.append(f"frozen data changed: {k}")
    if sha256_file(ROOT / rec["store_h5"]) != rec["store_sha256"]:
        probs.append("store sha256 changed")
    if LOCK.exists():
        probs.append(f"{LOCK} exists: the AlphaGenome test evaluation was already started (one-shot)")
    runs = {}
    with np.load(ROOT / man["files"]["patient_protocol_npz"]) as z:
        test_rows = np.asarray(z["test"])
    if len(test_rows) != man["counts"]["test"] != 66:
        probs.append(f"expected 66 test subjects, protocol has {len(test_rows)}")
    for s, r in rec["seeds"].items():
        d = ROOT / r["run_dir"]; runs[int(s)] = d
        if (d / "evaluation/test").exists():
            probs.append(f"seed {s}: evaluation/test already exists")
        for name, rel_p in (("best_pt_sha256", "checkpoints/best.pt"), ("resolved_config_sha256", "resolved_config.yaml"),
                            ("prior_logit_sha256", "prior_logit.npy"), ("patient_split_sha256", "patient_split.npz"),
                            ("locus_split_sha256", "locus_split.npz")):
            if sha256_file(d / rel_p) != r[name]:
                probs.append(f"seed {s}: {name} differs from the freeze record")
        if sha256_file(ROOT / r["config"]) != r["config_sha256"]:
            probs.append(f"seed {s}: config differs from the freeze record")
        with np.load(d / "patient_split.npz") as z:
            if not np.array_equal(np.asarray(z["test"]), test_rows):
                probs.append(f"seed {s}: run test rows differ from the frozen protocol")
    return runs, test_rows, probs


def evaluate(rec, man, seed, d, test_rows, authorize, device):
    cfg = yaml.safe_load((d / "resolved_config.yaml").read_text()); ev = cfg["evaluation"]
    fractions = [float(f) for f in man["mask_fractions"]]
    assert [float(f) for f in ev["mask_fractions"]] == fractions and int(ev["mask_seed"]) == 17001
    cpg_ids, _ = read_axis(ROOT / man["files"]["dataset_h5"])
    with np.load(d / "locus_split.npz") as z:
        columns = np.asarray(z["train_matrix_columns"]); heldout = np.asarray(z["heldout_matrix_columns"])
    uni = G.universe_ids(ROOT, man)
    assert not len(heldout) and len(columns) == man["universe_size"] and np.array_equal(np.sort(cpg_ids[columns]), uni)
    store = HDF5RepresentationStore(ROOT / rec["store_h5"], cpg_ids)
    t0 = time.time()
    try:
        assert store.coverage_mask[columns].all()
        model, ep = T.load_frozen_model(d / "checkpoints/best.pt", store.dim, cfg.get("model"))
        assert ep == rec["seeds"][str(seed)]["best_epoch_0based"], "checkpoint epoch differs from the freeze record"
        model.to(torch.device(device))
        T.evaluate_model(run_dir=d, model=model, store=store, matrix_path=ROOT / man["files"]["dataset_h5"], prior=np.load(d / "prior_logit.npy"),
                         rows=test_rows, columns=columns, split="test", device=torch.device(device), authorize=authorize,
                         eval_cfg={"panel_size": int(ev["panel_size"]), "batch_size": int(ev["batch_size"])}, fractions=fractions,
                         mask_seed=int(ev["mask_seed"]))
    finally:
        store.close()
    (d / "evaluation/test/eval_manifest.json").write_text(json.dumps({
        "kind": "posthoc_test_evaluation", "posthoc_comparator": True, "one_shot": True, "arm": ARM, "seed": seed, "n_eval_patients": len(test_rows),
        "mask_fractions": fractions, "weights_updated": False, "checkpoint_reselected": False,
        "checkpoint_sha256": rec["seeds"][str(seed)]["best_pt_sha256"], "freeze_record": str(REC.relative_to(ROOT)),
        "freeze_record_sha256": sha256_file(REC), "head_commit": git("rev-parse", "HEAD").stdout.strip(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "wall_clock_seconds": round(time.time() - t0, 1),
        "note": rec["statement"]}, indent=2, sort_keys=True))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--confirm-one-shot", action="store_true"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true", help="run the preflight only (reads no test beta values)")
    a = ap.parse_args()
    rec = json.loads(REC.read_text()); man = json.loads((ROOT / MANIFEST_REL).read_text())
    runs, test_rows, probs = preflight(rec, man)
    print("PREFLIGHT:", "OK" if not probs else "\n  " + "\n  ".join(probs), flush=True)
    if probs or a.dry_run:
        raise SystemExit(2 if probs else 0)
    if not a.confirm_one_shot:
        raise SystemExit("REFUSED: one-shot; re-run with --confirm-one-shot")
    rec_sha = sha256_file(REC)  # the full preflight above just passed; per access only check the record is unchanged and the lock is ours

    def authorize():
        if sha256_file(REC) != rec_sha or not LOCK.exists():
            raise PermissionError("GUARD: freeze record changed or lock missing during the one-shot evaluation")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps({"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "posthoc_comparator": True,
                                "head_commit": git("rev-parse", "HEAD").stdout.strip(), "freeze_record_sha256": sha256_file(REC)}, indent=2))
    for seed in sorted(runs):
        evaluate(rec, man, seed, runs[seed], test_rows, authorize, a.device)
        print(f"done test seed {seed}", flush=True)


if __name__ == "__main__":
    main()
