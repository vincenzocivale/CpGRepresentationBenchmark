"""Experiment A runner logic for the external GSE40279 confirmation (protocol v1.1). Torch-free at import time.

Per-run configs are built from the SAME template and builder as the TCGA confirmation runs
(`experiments.confirmation_matrix.build_config`, read-only use) and then overridden ONLY at the declared fields
(`ALLOWED_DIFF_VS_TCGA`). Nothing here trains or evaluates by itself: `run_all` shells out to `scripts/run_masking_benchmark.py`
and only after the gate is green.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import gate as G
from .budget import epoch_budget
from .freeze import ARMS, MANIFEST_REL, MASK_SEED, SEEDS
from .io import sha256_file

CONFIG_DIR = Path("configs/experiments/external_confirmation")
OUTPUT_ROOT = Path(G.OUTPUT_ROOT_REL)
BENCH_ROOT = OUTPUT_ROOT / "benchmark"
LOGS = OUTPUT_ROOT / "logs"
STATUS_FILE = "confirmation_status.json"
AGG_STATUS_FILE = "phase_A_status.json"
PROTOCOL_TAG = G.TAG_V1_1
PROTOCOL_VERSION = "v1.1"
DATASET_YAML = "configs/datasets/external_gse40279_v1.yaml"
STATUS_VOCABULARY = ("trained_validation_frozen",)   # never 'confirmed' before the authorized test evaluation
MATRIX_REL = "configs/experiments/regulatory_confirmation_matrix/matrix.yaml"

# Fields (dotted paths into the config) in which an external run config may differ from the TCGA confirmation config of the
# same arm and seed. Anything else differing is a protocol deviation and fails `diff_vs_tcga`.
ALLOWED_DIFF_VS_TCGA = frozenset({
    "experiment.name", "experiment.output_root", "experiment.locus_split.protocol_path",
    "dataset.name", "dataset.methylation_h5", "dataset.patient_protocol",
    "training.save_epoch_checkpoints", "training.save_last_checkpoint", "training.record_update_counts",
})


def jobs(seeds=None, only=None) -> list[tuple[str, int]]:
    """Seed-major; arms in the frozen launch order (ARMS)."""
    arms = [a for a in ARMS if only is None or a in only]
    return [(a, s) for s in (seeds or SEEDS) for a in arms]


def config_name(arm: str, seed: int) -> str:
    return f"{arm}__seed{seed}.yaml"


def _matrix_spec(repo: Path) -> dict:
    from cpg_repr_benchmark.experiments import confirmation_matrix as cm
    return cm.load_matrix(Path(repo) / MATRIX_REL)


def tcga_config(arm: str, seed: int, repo: Path) -> dict:
    """The TCGA confirmation config of (arm, seed), freshly built by the TCGA builder (not read from a possibly stale file)."""
    from cpg_repr_benchmark.experiments import confirmation_matrix as cm
    spec = _matrix_spec(repo)
    return cm.build_config(spec, cm.arms_by_id(spec)[arm], seed, Path(repo))


def build_config(arm: str, seed: int, repo: Path, *, keep_last_checkpoint: bool = False, split: str = "validation") -> dict:
    """External config = TCGA config of the same arm/seed with the declared overrides only."""
    if split != "validation":
        raise PermissionError("per-run external configs exist for the validation split only; test configs are produced by "
                              "`make_test_config` after authorization")
    repo = Path(repo)
    ds = yaml.safe_load((repo / DATASET_YAML).read_text())
    cfg = tcga_config(arm, seed, repo)
    cfg["experiment"]["name"] = f"external_confirmation_{arm}__seed{seed}"
    cfg["experiment"]["output_root"] = str(BENCH_ROOT)
    cfg["experiment"]["locus_split"]["protocol_path"] = ds["locus_protocol"]
    block = ds["runner_dataset_block"]
    cfg["dataset"].update(name=block["name"], methylation_h5=block["methylation_h5"], patient_protocol=block["patient_protocol"])
    # D6/D7 (opt-in engine keys): no per-epoch checkpoints; last.pt off unless explicitly requested; count optimizer updates
    cfg["training"].update(save_epoch_checkpoints=False, save_last_checkpoint=bool(keep_last_checkpoint), record_update_counts=True)
    return cfg


def make_test_config(arm: str, seed: int, repo: Path, run_dir: Path) -> dict:
    """Config for the TEST evaluation of an existing run (patient_view test, external authorization hook). Gate-protected by use."""
    cfg = yaml.safe_load((Path(run_dir) / "resolved_config.yaml").read_text())
    cfg["evaluation"].pop("require_patient_view", None)
    cfg["evaluation"].update(patient_view="test", test_authorization="external_gse40279_v1", allow_overwrite_split_dir=False)
    return cfg


def _flat(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, dict):
        out: dict[str, Any] = {}
        for k, v in d.items():
            out.update(_flat(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: d}


def diff_vs_tcga(arm: str, seed: int, repo: Path, cfg: dict | None = None) -> dict[str, tuple[Any, Any]]:
    """{dotted_key: (tcga_value, external_value)} for every field that differs (added/removed keys use None)."""
    a, b = _flat(tcga_config(arm, seed, repo)), _flat(cfg if cfg is not None else build_config(arm, seed, repo))
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def check_diff_vs_tcga(arm: str, seed: int, repo: Path, cfg: dict | None = None) -> list[str]:
    extra = sorted(set(diff_vs_tcga(arm, seed, repo, cfg)) - ALLOWED_DIFF_VS_TCGA)
    return extra


def generate(repo: Path, *, keep_last_checkpoint: bool = False) -> list[Path]:
    repo = Path(repo)
    out = repo / CONFIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for arm, seed in jobs():
        cfg = build_config(arm, seed, repo, keep_last_checkpoint=keep_last_checkpoint)
        bad = check_diff_vs_tcga(arm, seed, repo, cfg)
        if bad:
            raise ValueError(f"{arm}/seed{seed}: undeclared differences vs the TCGA config: {bad}")
        p = out / config_name(arm, seed)
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        paths.append(p)
    return paths


# ----------------------------------------------------------------------------- markers / status
def marker_path(repo: Path, arm: str, seed: int) -> Path:
    return Path(repo) / LOGS / f"{arm}__seed{seed}.done"


def read_marker(repo: Path, arm: str, seed: int) -> dict | None:
    p = marker_path(repo, arm, seed)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def run_dir_from_log(log: Path) -> Path | None:
    lines = [x for x in Path(log).read_text().splitlines() if x.startswith("RUN_DIR=")] if Path(log).is_file() else []
    return Path(lines[-1].split("=", 1)[1]) if lines else None


def build_status(arm: str, seed: int, run_dir: Path, *, wall_clock_seconds: float, repo: Path) -> dict:
    """Status of ONE finished validation run from its files. Raises ValueError on any protocol failure (run not marked done).

    Reads history.json, resolved_config.yaml, experiment.json, best.pt (state dict epoch only) and the evaluation metrics; never
    any beta value and never a test directory.
    """
    import torch

    from cpg_repr_benchmark.config import config_fingerprint
    from cpg_repr_benchmark.experiments.eval_layout import eval_base
    repo, run_dir = Path(repo), Path(run_dir)
    manifest = json.loads((repo / MANIFEST_REL).read_text())
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    hist = json.loads((run_dir / "history.json").read_text())
    exp = json.loads((run_dir / "experiment.json").read_text())
    n_train = int(exp["n_train_patients_rows"])
    eb = epoch_budget(n_train, int(cfg["training"]["batch_size"]), int(cfg["training"]["epochs"]))
    mse = [float(r["validation_mse"]) for r in hist]
    best_epoch = int(np.argmin(mse))   # first minimum == strict-improvement rule (ties keep the earliest epoch)
    ck = run_dir / "checkpoints"
    best = ck / "best.pt"
    updates = [int(r.get("updates_done", -1)) for r in hist]
    val_dir = eval_base(run_dir, "split_dirs", "validation")
    state_epoch = int(torch.load(best, map_location="cpu", weights_only=False)["epoch"]) if best.is_file() else -1
    check = {
        "epochs_run": len(hist), "epochs_ok": len(hist) == eb["epochs"] == 120,
        "n_train_524": n_train == 524, "updates_per_epoch_ok": all(int(r.get("updates_in_epoch", -1)) == 66 for r in hist),
        "updates_total": updates[-1] if updates else None, "updates_ok": bool(updates) and updates[-1] == eb["total_updates"] == 7920,
        "epoch_index_complete": [int(r["epoch"]) for r in hist] == list(range(len(hist))),
        "early_stopping_false": cfg["training"].get("early_stopping") is False,
        "no_early_stopping_json": not (run_dir / "early_stopping.json").exists(),
        "no_epoch_checkpoints": not any(ck.glob("epoch_*.pt")),
        "best_pt_present": best.is_file(), "best_pt_epoch_matches_history": state_epoch == best_epoch,
        "validation_only": cfg["evaluation"].get("patient_view") == "validation"
        and cfg["evaluation"].get("require_patient_view") == "validation",
        "split_dirs_layout": cfg["evaluation"].get("output_layout") == "split_dirs",
        "validation_outputs_present": all((val_dir / "seen" / f"mask_{f:.2f}" / "metrics.json").is_file()
                                          for f in manifest["mask_fractions"]),
        "no_test_dir": not (run_dir / "evaluation" / "test").exists(),
    }
    check["ok"] = all(v for k, v in check.items() if k not in ("epochs_run", "updates_total"))
    if not check["ok"]:
        raise ValueError(f"{arm}/seed{seed}: protocol check failed: {check}")
    prior = run_dir / "prior_logit.npy"
    return {
        "status": "trained_validation_frozen", "arm": arm, "seed": seed, "split": "validation",
        "protocol_tag": PROTOCOL_TAG, "protocol_version": PROTOCOL_VERSION,
        "protocol_doc_sha256": manifest.get("protocol_doc_sha256"),
        "best_epoch_0based": best_epoch, "best_update": updates[best_epoch], "best_val_mse_at_0.50": mse[best_epoch],
        "epochs_run": len(hist), "updates_done": updates[-1], "updates_per_epoch": 66,
        "wall_clock_seconds": round(float(wall_clock_seconds), 1),
        "best_pt_path": str(best), "best_pt_sha256": sha256_file(best),
        "last_pt_present": (ck / "last.pt").is_file(),
        "prior_logit_sha256": sha256_file(prior) if prior.is_file() else None,
        "patient_protocol_sha256": manifest["files_sha256"]["patient_protocol_npz"],
        "locus_protocol_sha256": manifest["files_sha256"]["locus_protocol_npz"],
        "matrix_h5_sha256": manifest["files_sha256"]["dataset_h5"],
        "store_sha256": manifest["arms"][arm]["store_sha256"],
        "universe_size": manifest["universe_size"], "n_train_loci": int(exp["n_train_loci"]),
        "code_git_commit": exp.get("code_git_commit"),
        "config_hash": config_fingerprint(cfg), "run_dir": str(run_dir), "protocol_check": check, "test_read": False,
    }


def write_status(run_dir: Path, status: dict) -> Path:
    p = Path(run_dir) / STATUS_FILE
    p.write_text(json.dumps(status, indent=2, sort_keys=True))
    return p


def aggregate_status(repo: Path) -> dict:
    repo = Path(repo)
    entries = []
    for arm, seed in jobs():
        mk = read_marker(repo, arm, seed)
        if mk and (Path(mk["run_dir"]) / STATUS_FILE).is_file():
            d = json.loads((Path(mk["run_dir"]) / STATUS_FILE).read_text())
            entries.append({k: d[k] for k in ("arm", "seed", "status", "run_dir", "best_epoch_0based", "best_val_mse_at_0.50",
                                              "epochs_run", "updates_done", "wall_clock_seconds", "best_pt_sha256")}
                           | {"status_file": str(Path(mk["run_dir"]) / STATUS_FILE)})
        else:
            entries.append({"arm": arm, "seed": seed, "status": "not_run" if mk is None else "incomplete"})
    n = sum(e["status"] == "trained_validation_frozen" for e in entries)
    return {"phase": "A", "split": "validation", "protocol_tag": PROTOCOL_TAG, "status_vocabulary": list(STATUS_VOCABULARY),
            "n_expected": len(entries), "n_trained_validation_frozen": n, "test_read": False, "runs": entries}


def write_aggregate(repo: Path) -> Path:
    p = Path(repo) / OUTPUT_ROOT / AGG_STATUS_FILE
    G.assert_writable(repo, p)  # never inside a frozen path
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(aggregate_status(repo), indent=2))
    tmp.replace(p)
    return p


# ----------------------------------------------------------------------------- run
def plan(repo: Path, *, seeds=None, only=None) -> list[tuple[str, int, Path, bool]]:
    """[(arm, seed, config path, already_done)] in execution order (seed-major)."""
    return [(a, s, Path(repo) / CONFIG_DIR / config_name(a, s), read_marker(repo, a, s) is not None)
            for a, s in jobs(seeds=seeds, only=only)]


def run_all(repo: Path, *, seeds=None, only=None, dry_run: bool = False, split: str = "validation",
            gate_checks: list | None = None, popen=subprocess.run, out=print) -> int:
    """Gate -> (dry-run listing | sequential execution). Returns a process exit code (2 = refused)."""
    repo = Path(repo)
    if split != "validation":
        out("REFUSED: the external runner executes the VALIDATION phase only; the test evaluation needs the final freeze, an explicit "
            "user authorization and `scripts/run_external_confirmation.py test-eval` (not implemented as a launch path here)")
        return 2
    checks = gate_checks if gate_checks is not None else G.run_gate(repo, split="validation", scope="A")
    bad = G.failures(checks)
    for c in checks:
        if not c.ok:
            out("  " + c.line())
    out(f"GATE (validation, phase A): {'OPEN' if not bad else 'REFUSED'} ({len(checks) - len(bad)}/{len(checks)} checks green)")
    todo = plan(repo, seeds=seeds, only=only)
    out(f"{len(todo)} job(s), seed-major: " + ", ".join(f"{a}#{s}" for a, s, _, _ in todo))
    if dry_run:
        for a, s, p, done in todo:
            out(f"{'skip (done)' if done else 'would run'} {p.relative_to(repo)}")
        return 0 if not bad else 3   # dry-run lists the plan and never launches; exit 3 = listing printed but the gate is closed
    if bad:
        return 2
    os.nice(10)
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "PYTHONPATH": str(repo / "src")}
    (repo / LOGS).mkdir(parents=True, exist_ok=True)
    for arm, seed, path, done in todo:
        if done:
            continue
        log = repo / LOGS / f"{arm}__seed{seed}.log"
        t0 = time.time()
        with log.open("a") as fh:
            rc = popen([sys.executable, str(repo / "scripts/run_masking_benchmark.py"), "--config", str(path), "--mode", "all"],
                       cwd=repo, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False).returncode
        if rc:
            raise RuntimeError(f"run failed ({arm}, seed {seed}); see {log}")
        rd = run_dir_from_log(log)
        if rd is None:
            raise RuntimeError(f"no RUN_DIR in {log}")
        status = build_status(arm, seed, rd, wall_clock_seconds=time.time() - t0, repo=repo)
        write_status(rd, status)
        marker_path(repo, arm, seed).write_text(json.dumps({"run_dir": str(rd), "status_sha256": sha256_file(rd / STATUS_FILE)}))
        write_aggregate(repo)
        out(f"done {arm} seed {seed}: best_epoch={status['best_epoch_0based']} val_mse@0.50={status['best_val_mse_at_0.50']:.6g} "
            f"updates={status['updates_done']} wall={status['wall_clock_seconds']}s")
    return 0


def mask_seed_ok(cfg: dict) -> bool:
    return cfg["evaluation"]["mask_seed"] == MASK_SEED


def evaluate_test_all(repo: Path, *, popen=subprocess.run, out=print) -> int:
    """TEST evaluation of the 12 finished validation runs (best.pt, evaluate mode). Refused unless the manifest is final AND
    test_set_authorized AND every gate check is green. Never part of the validation phase; never run by tooling on its own."""
    repo = Path(repo)
    checks = G.run_gate(repo, split="test", scope="A")
    bad = G.failures(checks)
    for c in bad:
        out("  " + c.line())
    if bad:
        out(f"GATE (test): REFUSED ({len(bad)} failing check(s)); the external TEST split stays locked")
        return 2
    for arm, seed in jobs():
        mk = read_marker(repo, arm, seed)
        if mk is None:
            raise RuntimeError(f"{arm}/seed{seed}: validation run not finished; test evaluation needs all 12 validation runs")
        rd = Path(mk["run_dir"])
        cfg_path = rd / "evaluation" / "external_test_config.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(make_test_config(arm, seed, repo, rd), sort_keys=False))
        rc = popen([sys.executable, str(repo / "scripts/run_masking_benchmark.py"), "--config", str(cfg_path), "--mode", "evaluate",
                    "--run-dir", str(rd)], cwd=repo, env={**os.environ, "PYTHONPATH": str(repo / "src")}, check=False).returncode
        if rc:
            raise RuntimeError(f"test evaluation failed ({arm}, seed {seed})")
    return 0
