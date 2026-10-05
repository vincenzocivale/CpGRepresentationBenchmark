"""Experiment B (secondary): TCGA -> GSE40279 transfer of the frozen phase-A checkpoints (protocol v1.2, AMENDMENT 2).

SCOPE (AMENDMENT 2): the DEFAULT arms are the three MAIN arms (regulatory_histone_dnase, cpgpt_large_locus, deepcpg_dna_locus) x 3 seeds
= 9 checkpoints, VALIDATION split only. `functional_annotations_pca` is a `legacy_sensitivity_control` and is excluded unless
`include_legacy=True` (CLI `--include-legacy`; its outputs carry `arm_role: legacy_sensitivity_control`). B on test is not planned.

NO external training of any kind: the TCGA `best.pt` (sha256-verified against the freeze manifest) is loaded unchanged into
`MaskedMethylomeReconstructor(raw_locus_dim=store.dim, 256, 256, 256, 512)`, the arm's own store is mapped onto the external matrix
columns by global cpg_idx, and the model is evaluated on the external validation patients at the five frozen mask fractions
(mask seed 17001, panel 2048, same pool/order as Experiment A).

* ``B_strict``        prior = ORIGINAL TCGA prior (`prior_logit.npy` of the TCGA phase-A run, re-indexed to the external matrix columns
                      by global cpg_idx). No external value enters the prior or anywhere else.
* ``B_recalibrated``  decoder/adapter unchanged (no weight update); prior recomputed ONLY from the 524 external TRAIN subjects with the
                      repo's leakage-safe `compute_leakage_safe_priors` (validation/test rows never read for the prior). One prior shared
                      by all arms. Uses external methylation in the prior, so it is not "no external data".

Locus-agnostic by construction: the model has no locus-indexed parameter (the adapter is `LayerNorm+Linear` on the raw embedding
dimension); stores are mapped to matrix columns through cpg_idx; the prior is a per-locus array that is simply re-indexed. The TCGA
universe (408,399) is a superset of the external one (408,017): the 382 TCGA-only loci are unused.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from . import gate as G
from .freeze import ARMS, LEGACY_ARMS, MAIN_ARMS, MANIFEST_REL
from .io import sha256_file

MODES = ("B_strict", "B_recalibrated")
TRANSFER_ROOT = Path(G.OUTPUT_ROOT_REL) / "transfer"
MANIFEST_NAME = "transfer_manifest.json"
MODEL_DEFAULTS = {"locus_latent_dim": 256, "token_dim": 256, "patient_dim": 256, "hidden_dim": 512}
EVAL = {"panel_size": 2048, "batch_size": 8, "num_workers": 0, "panel_repeats": 1, "mask_seed": 17001, "save_predictions": True,
        "output_layout": "split_dirs", "allow_overwrite_split_dir": False, "selection_mask_fraction": 0.5}
BETA_EPSILON = 1e-4


def mode_dir(repo: Path, mode: str) -> Path:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    return Path(repo) / TRANSFER_ROOT / mode


def run_dir_for(repo: Path, mode: str, arm: str, seed: int) -> Path:
    return mode_dir(repo, mode) / arm / f"seed_{seed}"


def done_marker(repo: Path, mode: str, arm: str, seed: int) -> Path:
    return run_dir_for(repo, mode, arm, seed) / "transfer.done"


def arm_role(arm: str) -> str:
    return "main" if arm in MAIN_ARMS else "legacy_sensitivity_control"


def jobs(manifest: dict, seeds=None, only=None, arm_order=None, include_legacy: bool = False) -> list[dict]:
    """The frozen checkpoints, seed-major in the Experiment-A arm order. Default = the 9 MAIN-arm checkpoints; the legacy functional
    checkpoints are returned only with `include_legacy` (an explicit request for a legacy arm without it is refused)."""
    entries = {(c["arm"], int(c["seed"])): c for c in manifest["phase_a_checkpoints_experiment_B"]}
    allowed = set(MAIN_ARMS) | (set(LEGACY_ARMS) if include_legacy else set())
    if only is not None and set(only) - allowed:
        raise ValueError(f"arms {sorted(set(only) - allowed)} are not in the Experiment B scope (main arms only; legacy needs include_legacy)")
    order = arm_order or ARMS
    return [entries[(a, s)] for s in (seeds or manifest["seeds"]) for a in order
            if a in allowed and (only is None or a in only) and (a, s) in entries]


# ----------------------------------------------------------------------------- checkpoint + prior artifacts
def verify_checkpoint(repo: Path, entry: dict) -> Path:
    p = Path(repo) / entry["best_pt"]
    if not p.is_file():
        raise FileNotFoundError(p)
    got = sha256_file(p)
    if got != entry["best_pt_sha256"]:
        raise ValueError(f"{entry['arm']}/seed{entry['seed']}: best.pt sha256 {got[:16]} != manifest {entry['best_pt_sha256'][:16]}")
    return p


def tcga_run_dir(repo: Path, entry: dict) -> Path:
    return (Path(repo) / entry["best_pt"]).parent.parent


def read_cpg_idx(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as h:
        return np.asarray(h["cpg_idx"][:], dtype=np.int64)


def reindex_prior(prior_src: np.ndarray, src_ids: np.ndarray, dst_ids: np.ndarray) -> np.ndarray:
    """prior_src[col in src matrix] -> prior for dst matrix columns, matched on global cpg_idx. Every dst id must exist in src."""
    prior_src = np.asarray(prior_src, dtype=np.float32)
    if len(prior_src) != len(src_ids):
        raise ValueError(f"prior length {len(prior_src)} != source matrix axis {len(src_ids)}")
    order = np.argsort(src_ids)
    sorted_ids = src_ids[order]
    pos = np.searchsorted(sorted_ids, dst_ids)
    ok = pos < len(sorted_ids)
    ok[ok] &= sorted_ids[pos[ok]] == dst_ids[ok]
    if not ok.all():
        raise ValueError(f"{int((~ok).sum())} external loci are absent from the TCGA prior axis")
    out = prior_src[order[pos]]
    if not np.isfinite(out).all():
        raise ValueError("re-indexed prior contains non-finite values")
    return out


def load_tcga_prior(repo: Path, entry: dict, ext_cpg_ids: np.ndarray, *, tcga_patients_train: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """B-strict prior: the ORIGINAL TCGA prior of the checkpoint's own run directory, re-indexed to the external columns.

    The prior is stored with every TCGA run (`prior_logit.npy`, full TCGA matrix axis). Provenance asserted from files in the run dir
    (no TCGA beta/test data read): `patient_split.npz` train rows == the frozen TCGA train rows (when given) and `experiment.json`
    `n_train_patients_rows`; `prior_logit.npy` sha256 is recorded.
    """
    import yaml
    rd = tcga_run_dir(repo, entry)
    cfg = yaml.safe_load((rd / "resolved_config.yaml").read_text())
    exp = json.loads((rd / "experiment.json").read_text())
    prior_path = rd / "prior_logit.npy"
    prior = np.load(prior_path)
    tcga_ids = read_cpg_idx(Path(repo) / cfg["dataset"]["methylation_h5"])   # /cpg_idx only
    info: dict[str, Any] = {"tcga_run_dir": str(rd), "tcga_prior_path": str(prior_path), "tcga_prior_sha256": sha256_file(prior_path),
                            "tcga_prior_len": len(prior), "tcga_n_train_patients_rows": int(exp["n_train_patients_rows"]),
                            "tcga_global_train_beta_prior": float(exp["global_train_beta_prior"])}
    with np.load(rd / "patient_split.npz") as z:
        train = np.asarray(z["train"])   # row indices only
    info["tcga_train_rows_in_run"] = len(train)
    if len(train) != int(exp["n_train_patients_rows"]):
        raise ValueError("TCGA run: patient_split.npz train rows != experiment.json n_train_patients_rows")
    if tcga_patients_train is not None and not np.array_equal(np.sort(train), np.sort(tcga_patients_train)):
        raise ValueError("TCGA run: patient_split.npz train rows differ from the frozen TCGA train rows")
    out = reindex_prior(prior, tcga_ids, ext_cpg_ids)
    info["n_external_loci_covered"] = len(out)
    info["external_data_used_in_prior"] = False
    return out, info


def recompute_tcga_prior_check(repo: Path, entry: dict, *, prior_fn: Callable | None = None, atol: float = 1e-6) -> dict:
    """OPTIONAL (heavy, reads TCGA TRAIN betas only, never TCGA test rows): recompute the TCGA prior from the run's own train rows
    with the same leakage-safe function and assert equality with the stored `prior_logit.npy`."""
    import yaml

    from cpg_repr_benchmark.training.priors import compute_leakage_safe_priors
    fn = prior_fn or compute_leakage_safe_priors
    rd = tcga_run_dir(repo, entry)
    cfg = yaml.safe_load((rd / "resolved_config.yaml").read_text())
    with np.load(rd / "patient_split.npz") as z:
        train = np.asarray(z["train"])
    with np.load(rd / "locus_split.npz") as z:
        cols = np.asarray(z["train_matrix_columns"])
    stored = np.load(rd / "prior_logit.npy")
    got, _, _ = fn(Path(repo) / cfg["dataset"]["methylation_h5"], train, cols, len(stored),
                   epsilon=float(cfg["training"].get("beta_epsilon", BETA_EPSILON)))
    err = float(np.max(np.abs(got - stored)))
    if err > atol:
        raise ValueError(f"recomputed TCGA prior differs from the stored one (max abs diff {err})")
    return {"max_abs_diff": err, "n_train_rows": len(train), "tcga_test_rows_read": 0}


def recalibrated_prior(repo: Path, manifest: dict, ext_cpg_ids: np.ndarray, *, prior_fn: Callable | None = None) -> tuple[np.ndarray, dict]:
    """B-recalibrated prior: ONLY the external TRAIN rows (524) x all universe columns. Asserts the exact inputs before reading."""
    from cpg_repr_benchmark.training.priors import compute_leakage_safe_priors
    fn = prior_fn or compute_leakage_safe_priors
    repo = Path(repo)
    with np.load(repo / manifest["files"]["patient_protocol_npz"]) as z:
        parts = {k: np.asarray(z[k]) for k in ("train", "validation", "test")}
    train = parts["train"]
    if len(train) != manifest["counts"]["train"] or len(train) != 524:
        raise ValueError(f"recalibrated prior needs the {manifest['counts']['train']} frozen train rows, got {len(train)}")
    for other in ("validation", "test"):
        if np.intersect1d(train, parts[other]).size:
            raise ValueError(f"train rows overlap the {other} rows")
    with np.load(repo / manifest["files"]["locus_protocol_npz"]) as z:
        uni = np.sort(np.concatenate([z["train_cpg_idx"], z["heldout_cpg_idx"]])).astype(np.int64)
    order = np.argsort(ext_cpg_ids)
    cols = np.sort(order[np.searchsorted(ext_cpg_ids[order], uni)])
    if len(cols) != manifest["universe_size"] or not np.array_equal(np.sort(ext_cpg_ids[cols]), uni):
        raise ValueError("prior columns differ from the frozen locus universe")
    prior, usable, global_beta = fn(repo / manifest["files"]["dataset_h5"], train, cols, len(ext_cpg_ids), epsilon=BETA_EPSILON)
    if not np.asarray(usable).all():
        raise ValueError("some universe loci have no finite external train value (A drops none; refusing to narrow the universe)")
    info = {"prior_rows": "external train only", "n_prior_rows": len(train), "n_prior_columns": len(cols),
            "global_train_beta": float(global_beta), "external_data_used_in_prior": True,
            "validation_test_rows_read_for_prior": 0, "prior_sha256": __import__("hashlib").sha256(
                np.ascontiguousarray(prior, dtype=np.float32).tobytes()).hexdigest()}
    return np.asarray(prior, dtype=np.float32), info


# ----------------------------------------------------------------------------- evaluation
def evaluate_model(*, run_dir: Path, model, store, matrix_path: Path, prior: np.ndarray, rows: np.ndarray, columns: np.ndarray,
                   split: str, device, authorize: Callable[[], None] | None = None, eval_cfg: dict | None = None,
                   fractions=(0.15, 0.30, 0.50, 0.70, 0.90), mask_seed: int = 17001) -> dict:
    """Evaluate `model` at every mask fraction on `rows` and write evaluation/<split>/... with the split_dirs layout (+ guards).

    Same dataset construction as `run_masking_benchmark._evaluate_views` (context = target = `columns`, panel 2048, one fraction
    per dataset, panel_repeats 1)."""
    import torch  # noqa: F401
    from torch.utils.data import DataLoader

    from cpg_repr_benchmark.data.masking import MaskFractionBatchSampler, MaskingDataset
    from cpg_repr_benchmark.evaluation.metrics import reconstruction_metrics
    from cpg_repr_benchmark.experiments.eval_layout import evaluate_split, write_split_summary
    from cpg_repr_benchmark.training.engine import evaluate_loader

    ev = {**EVAL, **(eval_cfg or {}), "mask_fractions": [float(f) for f in fractions], "mask_seed": int(mask_seed), "patient_view": split}
    cfg = {"evaluation": ev, "training": {"seed": int(mask_seed), "beta_epsilon": BETA_EPSILON}}

    def evaluate_fn(view: str, fraction: float):
        ds = MaskingDataset(methylation_h5=matrix_path, representation_store=store, prior_logit_full=prior, sample_indices=rows,
                            context_columns=columns, target_columns=columns, panel_size=int(ev["panel_size"]),
                            mask_fractions=[fraction], seed=int(mask_seed), beta_epsilon=BETA_EPSILON)
        loader = DataLoader(ds, batch_sampler=MaskFractionBatchSampler(ds, int(ev["batch_size"]), shuffle=False), num_workers=0)
        ds.set_epoch(0)
        _, chunk = evaluate_loader(model, loader, device)
        chunk["panel_repeat"] = np.zeros(len(chunk["sample_index"]), dtype=np.int64)
        metrics = reconstruction_metrics(chunk["prediction"], chunk["target"], chunk["prior_prediction"],
                                         target_matrix_column=chunk["target_matrix_column"])
        return metrics, chunk

    results = evaluate_split(run_dir, cfg, {"seen": "per-mode prior (see transfer_manifest.json)"}, ev["mask_fractions"], evaluate_fn,
                             authorize=authorize)
    write_split_summary(run_dir, cfg, split, {"split": split, "evaluation": results, "patient_view": split})
    return results


def build_model(raw_dim: int, model_cfg: dict | None = None):
    from cpg_repr_benchmark.models.model import MaskedMethylomeReconstructor
    m = {**MODEL_DEFAULTS, **(model_cfg or {})}
    return MaskedMethylomeReconstructor(raw_locus_dim=raw_dim, **m)


def load_frozen_model(ckpt_path: Path, raw_dim: int, model_cfg: dict | None = None):
    """Frozen TCGA decoder + adapter: load the state dict only; no optimizer, `eval()`, `requires_grad_(False)`."""
    import torch
    model = build_model(raw_dim, model_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    return model, int(state.get("epoch", -1))


def run_transfer(repo: Path, mode: str, *, split: str = "validation", seeds=None, only=None, dry_run: bool = False, device: str = "auto",
                 gate_checks: list | None = None, out=print, include_legacy: bool = False) -> int:
    """Gate (scope B) -> evaluate the 9 main-arm checkpoints (12 with include_legacy). Returns 2 when refused. Never trains. Test split only if authorized."""
    import torch

    from cpg_repr_benchmark.data.methylation import read_axis
    from cpg_repr_benchmark.data.splits import load_or_create_locus_protocol
    from cpg_repr_benchmark.representations.hdf5_store import HDF5RepresentationStore
    repo = Path(repo)
    if mode not in MODES:
        raise ValueError(mode)
    manifest = json.loads((repo / MANIFEST_REL).read_text())
    checks = gate_checks if gate_checks is not None else G.run_gate(repo, split=split, scope="B")
    bad = G.failures(checks)
    for c in bad:
        out("  " + c.line())
    out(f"GATE (transfer {mode}, {split}): {'OPEN' if not bad else 'REFUSED'}")
    todo = jobs(manifest, seeds=seeds, only=only, include_legacy=include_legacy)
    out(f"{len(todo)} checkpoint(s): " + ", ".join(f"{c['arm']}#{c['seed']}" for c in todo))
    if dry_run:
        return 0 if not bad else 3
    if bad:
        return 2
    authorize = G.authorize_external_test if split == "test" else None
    if split == "test":
        G.authorize_external_test(repo)   # raises PermissionError unless final + authorized + green
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    matrix = repo / manifest["files"]["dataset_h5"]
    cpg_ids, _ = read_axis(matrix)
    with np.load(repo / manifest["files"]["patient_protocol_npz"]) as z:
        rows = np.asarray(z[split])           # validation (or test only after authorization)
    pool = load_or_create_locus_protocol(repo / manifest["files"]["locus_protocol_npz"], np.arange(len(cpg_ids)), cpg_ids,
                                         seed=int(manifest["locus_seed"]), heldout_fraction=0.0).train_columns
    if len(pool) != manifest["universe_size"]:
        raise RuntimeError("locus pool differs from the frozen universe")
    tcga_train = np.load(repo / "outputs/encode_atlas_v1/patients.npz")["train"]
    recal = None
    if mode == "B_recalibrated":
        recal, recal_info = recalibrated_prior(repo, manifest, cpg_ids)
        pdir = mode_dir(repo, mode)
        G.assert_writable(repo, pdir)
        pdir.mkdir(parents=True, exist_ok=True)
        np.save(pdir / "prior_external_train.npy", recal)
        (pdir / "prior_external_train.json").write_text(json.dumps(recal_info, indent=2, sort_keys=True))
    for entry in todo:
        arm, seed = entry["arm"], int(entry["seed"])
        rd = run_dir_for(repo, mode, arm, seed)
        if done_marker(repo, mode, arm, seed).exists():
            continue
        G.assert_writable(repo, rd)
        t0 = time.time()
        ckpt = verify_checkpoint(repo, entry)
        store = HDF5RepresentationStore(repo / manifest["arms"][arm]["store_h5"], cpg_ids)
        if not store.coverage_mask[pool].all():
            raise RuntimeError(f"{arm}: store does not cover the frozen universe")
        model, ckpt_epoch = load_frozen_model(ckpt, store.dim)
        model.to(dev)
        if mode == "B_strict":
            prior, pinfo = load_tcga_prior(repo, entry, cpg_ids, tcga_patients_train=tcga_train)
        else:
            prior, pinfo = recal, {**recal_info}
        rd.mkdir(parents=True, exist_ok=True)
        evaluate_model(run_dir=rd, model=model, store=store, matrix_path=matrix, prior=prior, rows=rows, columns=pool, split=split,
                       device=dev, authorize=authorize)
        man = {"experiment": "B", "mode": mode, "arm": arm, "arm_role": arm_role(arm), "seed": seed, "split": split, "weights_updated": False,
               "checkpoint": str(ckpt), "checkpoint_epoch": ckpt_epoch, "checkpoint_sha256": entry["best_pt_sha256"],
               "store_sha256": manifest["arms"][arm]["store_sha256"], "patient_protocol_sha256": manifest["files_sha256"]["patient_protocol_npz"],
               "locus_protocol_sha256": manifest["files_sha256"]["locus_protocol_npz"], "matrix_h5_sha256": manifest["files_sha256"]["dataset_h5"],
               "universe_size": len(pool), "n_eval_patients": len(rows), "prior": pinfo,
               "protocol_tag": G.TAG_V1_2, "test_read": split == "test", "wall_clock_seconds": round(time.time() - t0, 1)}
        (rd / MANIFEST_NAME).write_text(json.dumps(man, indent=2, sort_keys=True, default=str))
        done_marker(repo, mode, arm, seed).write_text("done\n")
        store.close()
        out(f"done {mode} {arm} seed {seed} ({man['wall_clock_seconds']}s)")
    return 0
