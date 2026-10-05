"""ONE-SHOT evaluation of the frozen Experiment-A checkpoints on an external split (protocol v1.2, AMENDMENT 2). Torch only inside functions.

Production use is the external TEST split: `evaluate_test_all` (CLI `scripts/run_external_confirmation.py test-eval`). It

* refuses unless the manifest is `final` + `test_set_authorized` AND the gate is all green (git: protocol v1.2 tag, authorization tag
  `external-recon-test-authorization-v1`, implementation committed clean, manifest/protocol unchanged since the authorization tag) AND
  `--confirm-one-shot` was given;
* verifies the sha256 of every checkpoint (`best.pt` vs `confirmation_status.json`, `phase_A_status.json` and the manifest of the run
  hashes), the arm store, the frozen protocol files and the prior BEFORE the first evaluation of any run;
* never retrains and never re-selects a checkpoint (the `best.pt` chosen on validation is loaded unchanged, weights frozen);
* evaluates at the five frozen mask fractions with the SAME dataset construction as validation (`transfer.evaluate_model`: same panel
  2048, mask seed 17001, one fraction per dataset, context = target = the run's own post-filter locus columns, the run's own
  `prior_logit.npy`, the same adapter/decoder);
* writes ONLY under `<run>/evaluation/test/` (results + `eval_manifest.json`), refuses when `evaluation/test` already exists and
  holds a one-shot lock (`logs/test_eval_started.json`) so that the test cannot be silently repeated.

`evaluate_frozen_run(..., eval_split="validation", rehearsal=True, out_run_dir=<scratch outside outputs/external_reconstruction_v1>)` is the
REHEARSAL entry point: the same function and the same code path, pointed at the VALIDATION split and a scratch directory. It is never
used by the CLI test path; `eval_split="test"` always goes through the authorization callable.
"""
from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from . import gate as G
from . import transfer as T
from .freeze import ARMS, LEGACY_ARMS, MAIN_ARMS, MANIFEST_REL, SEEDS
from .io import sha256_file

STARTED_LOCK = Path(G.OUTPUT_ROOT_REL) / "logs" / "test_eval_started.json"
TEST_STATUS = Path(G.OUTPUT_ROOT_REL) / "test_eval_status.json"
EVAL_MANIFEST = "eval_manifest.json"
SCHEMA = "external_gse40279_v1_eval_manifest/1"


def role_of(arm: str) -> str:
    return "main" if arm in MAIN_ARMS else "legacy_sensitivity_control"


def test_jobs(include_legacy: bool = False, seeds=None) -> list[tuple[str, int]]:
    """Seed-major in the frozen arm order; the 9 main runs, plus the 3 legacy functional runs only with `include_legacy`."""
    allowed = set(MAIN_ARMS) | (set(LEGACY_ARMS) if include_legacy else set())
    return [(a, s) for s in (seeds or SEEDS) for a in ARMS if a in allowed]


test_jobs.__test__ = False   # not a pytest test, despite the name


def _git_commit(repo: Path, ref: str) -> str | None:
    r = subprocess.run(["git", "rev-parse", "-q", "--verify", f"{ref}^{{commit}}"], cwd=repo, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else None


def _inside_output_root(repo: Path, path: Path) -> bool:
    root = (Path(repo) / G.OUTPUT_ROOT_REL).resolve()
    p = Path(path).resolve()
    return p == root or root in p.parents


def check_run_inputs(repo: Path, manifest: dict, arm: str, seed: int, run_dir: Path, *, verify_matrix: bool = True) -> dict:
    """Hash/identity verification of everything one evaluation depends on. Raises ValueError on any mismatch. Reads no beta values."""
    repo, run_dir = Path(repo), Path(run_dir)
    st = json.loads((run_dir / "confirmation_status.json").read_text())
    if st.get("status") != "trained_validation_frozen" or st.get("arm") != arm or int(st.get("seed")) != seed:
        raise ValueError(f"{arm}/seed{seed}: confirmation_status.json is not a trained_validation_frozen status of this run")
    best = run_dir / "checkpoints" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    got = sha256_file(best)
    if got != st["best_pt_sha256"]:
        raise ValueError(f"{arm}/seed{seed}: best.pt sha256 {got[:16]} != confirmation_status.json {st['best_pt_sha256'][:16]}")
    agg = repo / G.OUTPUT_ROOT_REL / "phase_A_status.json"
    if agg.is_file():
        row = [r for r in json.loads(agg.read_text())["runs"] if r.get("arm") == arm and int(r.get("seed", -1)) == seed]
        if not row or row[0].get("best_pt_sha256") != got:
            raise ValueError(f"{arm}/seed{seed}: best.pt sha256 differs from phase_A_status.json")
    files = manifest["files"]
    hashes = {"patient_protocol_sha256": files_sha(repo, manifest, "patient_protocol_npz"),
              "locus_protocol_sha256": files_sha(repo, manifest, "locus_protocol_npz")}
    if verify_matrix:
        hashes["matrix_h5_sha256"] = files_sha(repo, manifest, "dataset_h5")
    for k, v in hashes.items():
        if v != st.get(k):
            raise ValueError(f"{arm}/seed{seed}: {k} {v[:16]} != the hash recorded at training time {str(st.get(k))[:16]}")
    store_rel = manifest["arms"][arm]["store_h5"]
    store_sha = sha256_file(repo / store_rel)
    if store_sha != manifest["arms"][arm]["store_sha256"] or store_sha != st.get("store_sha256"):
        raise ValueError(f"{arm}/seed{seed}: store sha256 mismatch")
    prior = run_dir / "prior_logit.npy"
    if sha256_file(prior) != st.get("prior_logit_sha256"):
        raise ValueError(f"{arm}/seed{seed}: prior_logit.npy sha256 differs from the status file")
    return {"status": st, "best_pt": str(best), "best_pt_sha256": got, "store_sha256": store_sha,
            "prior_logit_sha256": st["prior_logit_sha256"], **hashes, "files": files}


def files_sha(repo: Path, manifest: dict, key: str) -> str:
    got = sha256_file(Path(repo) / manifest["files"][key])
    if got != manifest["files_sha256"][key]:
        raise ValueError(f"{key} sha256 {got[:16]} != manifest {manifest['files_sha256'][key][:16]}")
    return got


def evaluate_frozen_run(repo: Path, arm: str, seed: int, run_dir: Path, *, eval_split: str = "test", out_run_dir: Path | None = None,
                        authorize: Callable[[], None] | None = None, device: str = "cpu", allow_existing: bool = False,
                        rehearsal: bool = False, verify_matrix: bool = True, manifest_path: Path | None = None,
                        expected_eval: dict | None = None) -> dict:
    """Evaluate the frozen `best.pt` of ONE finished Experiment-A run on `eval_split`, five fractions, writing
    `<out_run_dir>/evaluation/<eval_split>/{seen/mask_*/{metrics.json,predictions.npz}, summary.json, eval_manifest.json}`."""
    import torch

    from cpg_repr_benchmark.data.methylation import read_axis
    from cpg_repr_benchmark.representations.hdf5_store import HDF5RepresentationStore
    repo, run_dir = Path(repo), Path(run_dir)
    out_run = Path(out_run_dir) if out_run_dir else run_dir
    manifest = json.loads((Path(manifest_path) if manifest_path else repo / MANIFEST_REL).read_text())
    if eval_split not in ("validation", "test"):
        raise ValueError(f"unknown split {eval_split!r}")
    if eval_split == "test":
        if rehearsal:
            raise PermissionError("GUARD: a rehearsal never touches the test split")
        if authorize is None:
            raise PermissionError("GUARD: the test split needs the authorization callable (final + authorized + green gate + tag)")
        authorize()                                                   # BEFORE any data access
        if out_run.resolve() != run_dir.resolve():
            raise PermissionError("GUARD: test results are written only inside the run's own evaluation/test/")
    else:
        # validation re-evaluation = rehearsal only: scratch dir OUTSIDE the external output root, never the run's own tree
        if not rehearsal:
            raise PermissionError("GUARD: re-evaluating validation through this path is a rehearsal (rehearsal=True)")
        if _inside_output_root(repo, out_run) or out_run.resolve() == run_dir.resolve():
            raise PermissionError(f"GUARD: rehearsal output {out_run} must be outside {G.OUTPUT_ROOT_REL}")
    split_dir = out_run / "evaluation" / eval_split
    if split_dir.exists() and not allow_existing:
        raise FileExistsError(f"{split_dir} already exists: the {eval_split} evaluation is one-shot and is never repeated or overwritten")
    info = check_run_inputs(repo, manifest, arm, seed, run_dir, verify_matrix=verify_matrix)
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    ev = cfg["evaluation"]
    fractions = [float(f) for f in manifest["mask_fractions"]]
    same = expected_eval or {"panel_size": 2048, "batch_size": 8, "panel_repeats": 1, "mask_seed": 17001, "save_predictions": True}
    for k, v in same.items():
        if ev.get(k, T.EVAL.get(k)) != v:
            raise ValueError(f"{arm}/seed{seed}: resolved evaluation.{k} = {ev.get(k)!r} differs from the frozen {v!r}")
    if [float(f) for f in ev["mask_fractions"]] != fractions:
        raise ValueError("resolved mask_fractions differ from the manifest")
    if float(cfg["training"].get("beta_epsilon", 1e-4)) != T.BETA_EPSILON:
        raise ValueError("beta_epsilon differs from the frozen value")
    matrix = repo / manifest["files"]["dataset_h5"]
    cpg_ids, _ = read_axis(matrix)
    with np.load(repo / manifest["files"]["patient_protocol_npz"]) as z:
        rows = np.asarray(z[eval_split])
    expected_n = manifest["counts"][eval_split]
    if len(rows) != expected_n:
        raise ValueError(f"{eval_split} rows {len(rows)} != {expected_n}")
    with np.load(run_dir / "patient_split.npz") as z:
        if not np.array_equal(np.asarray(z[eval_split]), rows):
            raise ValueError(f"{arm}/seed{seed}: the run's patient_split {eval_split} rows differ from the frozen protocol")
    with np.load(run_dir / "locus_split.npz") as z:
        columns = np.asarray(z["train_matrix_columns"])
        heldout = np.asarray(z["heldout_matrix_columns"])
    uni = G.universe_ids(repo, manifest)
    if len(heldout) or len(columns) != manifest["universe_size"] or not np.array_equal(np.sort(cpg_ids[columns]), uni):
        raise ValueError(f"{arm}/seed{seed}: the run's locus columns differ from the frozen locus universe")
    prior = np.load(run_dir / "prior_logit.npy")
    t0 = time.time()
    store = HDF5RepresentationStore(repo / manifest["arms"][arm]["store_h5"], cpg_ids)
    try:
        if not store.coverage_mask[columns].all():
            raise RuntimeError(f"{arm}: store does not cover the frozen universe")
        model, ckpt_epoch = T.load_frozen_model(Path(info["best_pt"]), store.dim, cfg.get("model"))
        if ckpt_epoch != int(info["status"]["best_epoch_0based"]):
            raise ValueError(f"{arm}/seed{seed}: checkpoint epoch {ckpt_epoch} != best epoch of the status file")
        dev = torch.device(device)
        model.to(dev)
        out_run.mkdir(parents=True, exist_ok=True)
        results = T.evaluate_model(run_dir=out_run, model=model, store=store, matrix_path=matrix, prior=prior, rows=rows, columns=columns,
                                   split=eval_split, device=dev, authorize=authorize if eval_split == "test" else None,
                                   eval_cfg={"panel_size": int(ev["panel_size"]), "batch_size": int(ev["batch_size"])},
                                   fractions=fractions, mask_seed=int(ev["mask_seed"]))
    finally:
        store.close()
    rec = {
        "schema": SCHEMA, "kind": "rehearsal_validation" if rehearsal else "test_evaluation", "one_shot": eval_split == "test",
        "arm": arm, "role": role_of(arm), "seed": seed, "split": eval_split, "n_eval_patients": len(rows),
        "mask_fractions": fractions, "mask_seed": int(ev["mask_seed"]), "weights_updated": False, "checkpoint_reselected": False,
        "checkpoint": info["best_pt"], "checkpoint_sha256": info["best_pt_sha256"], "checkpoint_epoch_0based": ckpt_epoch,
        "store_sha256": info["store_sha256"], "prior_logit_sha256": info["prior_logit_sha256"],
        "patient_protocol_sha256": info["patient_protocol_sha256"], "locus_protocol_sha256": info["locus_protocol_sha256"],
        "matrix_h5_sha256": info.get("matrix_h5_sha256"), "universe_size": len(columns),
        "protocol_version": manifest.get("protocol_version"), "protocol_doc_sha256": manifest.get("protocol_doc_sha256"),
        "protocol_tag": G.TAG_V1_2, "protocol_tag_commit": _git_commit(repo, G.TAG_V1_2),
        "authorization_tag": G.TAG_AUTH, "authorization_tag_commit": _git_commit(repo, G.TAG_AUTH), "head_commit": _git_commit(repo, "HEAD"),
        "freeze_state": manifest.get("freeze_state"), "test_set_authorized": manifest.get("test_set_authorized"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "wall_clock_seconds": round(time.time() - t0, 1),
        "device": device, "test_rows_read": eval_split == "test", "source_run_dir": str(run_dir),
    }
    (split_dir / EVAL_MANIFEST).write_text(json.dumps(rec, indent=2, sort_keys=True, default=str))
    return {"manifest": rec, "results": results}


def _completed(run_dir: Path) -> bool:
    p = Path(run_dir) / "evaluation" / "test"
    return (p / EVAL_MANIFEST).is_file() and all((p / "seen" / f"mask_{f:.2f}" / "metrics.json").is_file()
                                                 for f in (0.15, 0.30, 0.50, 0.70, 0.90))


def evaluate_test_all(repo: Path, *, include_legacy: bool = False, confirm_one_shot: bool = False, resume_completed: bool = False,
                      device: str = "cuda", gate_checks: list | None = None, authorize: Callable[[], None] | None = None,
                      evaluate_fn: Callable = evaluate_frozen_run, seeds=None, out=print) -> int:
    """One-shot TEST evaluation of the 9 main runs (+3 legacy functional runs with `include_legacy`). Returns 2 when refused."""
    repo = Path(repo)
    checks = gate_checks if gate_checks is not None else G.run_gate(repo, split="test", scope="A")
    bad = G.failures(checks)
    for c in bad:
        out("  " + c.line())
    if bad:
        out(f"GATE (test): REFUSED ({len(bad)} failing check(s)); the external TEST split stays locked")
        return 2
    if not confirm_one_shot:
        out("REFUSED: the test evaluation is ONE-SHOT; re-run with --confirm-one-shot to proceed")
        return 2
    manifest = json.loads((repo / MANIFEST_REL).read_text())
    base_authorize = authorize or (lambda: G.authorize_external_test(repo))
    verified: list[bool] = []

    def authorize() -> None:   # the full gate hashes the stores/matrix: run it once, then reuse the green verdict within this launch
        if not verified:
            base_authorize()
            verified.append(True)
    authorize()
    jobs = test_jobs(include_legacy, seeds=seeds)
    runs = {}
    from . import runner as R
    for arm, seed in jobs:
        mk = R.read_marker(repo, arm, seed)
        if mk is None:
            raise RuntimeError(f"{arm}/seed{seed}: validation run not finished; the test evaluation needs all its runs")
        runs[(arm, seed)] = Path(mk["run_dir"])
    # preflight EVERYTHING before the first evaluation (a failure must not leave a half-done one-shot)
    pending = []
    for i, ((arm, seed), rd) in enumerate(runs.items()):
        if (rd / "evaluation" / "test").exists():
            if resume_completed and _completed(rd):
                continue
            raise FileExistsError(f"{rd / 'evaluation' / 'test'} already exists: the test evaluation is one-shot (not repeated)")
        check_run_inputs(repo, manifest, arm, seed, rd, verify_matrix=(i == 0))
        pending.append((arm, seed))
    lock = repo / STARTED_LOCK
    if lock.exists() and not resume_completed:
        raise FileExistsError(f"{lock} exists: a test evaluation was already started; it is never silently repeated")
    lock.parent.mkdir(parents=True, exist_ok=True)
    G.assert_writable(repo, lock)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if lock.exists():   # --resume-completed: keep the original start record, append the resume
        rec = json.loads(lock.read_text())
        rec.setdefault("resumed_utc", []).append(now)
    else:
        rec = {"started_utc": now, "include_legacy": include_legacy, "jobs": [f"{a}__seed{s}" for a, s in jobs],
               "head_commit": _git_commit(repo, "HEAD"), "authorization_tag_commit": _git_commit(repo, G.TAG_AUTH)}
    lock.write_text(json.dumps(rec, indent=2))
    done = []
    for arm, seed in pending:
        r = evaluate_fn(repo, arm, seed, runs[(arm, seed)], eval_split="test", authorize=authorize, device=device, verify_matrix=False)
        done.append({"arm": arm, "seed": seed, "role": role_of(arm), "checkpoint_sha256": r["manifest"]["checkpoint_sha256"]})
        out(f"done test {arm} seed {seed} ({role_of(arm)}) wall={r['manifest']['wall_clock_seconds']}s")
    status = {"phase": "test_evaluation", "one_shot": True, "include_legacy": include_legacy, "n_expected": len(jobs),
              "n_done_now": len(done), "runs": [{"arm": a, "seed": s, "role": role_of(a), "run_dir": str(runs[(a, s)]),
                                                 "complete": _completed(runs[(a, s)])} for a, s in jobs]}
    G.assert_writable(repo, repo / TEST_STATUS)
    (repo / TEST_STATUS).write_text(json.dumps(status, indent=2))
    return 0
