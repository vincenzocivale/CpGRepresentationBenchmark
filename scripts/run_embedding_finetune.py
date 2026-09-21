#!/usr/bin/env python3
"""Fine-tune a sparse-reconstruction encoder end-to-end on one downstream phenotype task.

This is the `mode: fine_tuned` counterpart to `extract_embeddings.py` + `run_embedding_probe.py`
(which only run the encoder frozen). It unlocks `encode_patient` plus a thin 1-layer head and
trains both on the downstream task's train split; the reconstruction decoder is never touched.
Writes the same uniform `summary.json` schema as `run_embedding_probe.py`, with `mode:
fine_tuned`, under `outputs/embedding_probe/<dataset>/<representation>/<track>/seed_<seed>/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from cpg_repr_benchmark.data.legacy import legacy_ids_to_coordinate_ids
from cpg_repr_benchmark.data.methylation import read_axis
from cpg_repr_benchmark.data.splits import patient_disjoint_split
from cpg_repr_benchmark.experiments.result_schema import embedding_summary_payload
from cpg_repr_benchmark.models.model import build_reconstructor
from cpg_repr_benchmark.probing.finetune import fine_tune_encoder
from cpg_repr_benchmark.representations.hdf5_store import HDF5RepresentationStore, inspect_representation_h5
from cpg_repr_benchmark.representations.resolver import resolve_representation
from cpg_repr_benchmark.training.priors import beta_to_logit, compute_leakage_safe_priors


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_embedding(cfg, representation, matrix_ids, cpg_registry: Path | None):
    """Same legacy-namespace-translation contract as `extract_embeddings.py`."""
    if cpg_registry is not None:
        store_ids, store_dim, _id_key, embedding_key = inspect_representation_h5(
            representation.store_path, id_key=representation.id_key, embedding_key=representation.embedding_key
        )
        store_ids = legacy_ids_to_coordinate_ids(store_ids, cpg_registry)
        order = np.argsort(store_ids)
        sorted_ids = store_ids[order]
        pos = np.searchsorted(sorted_ids, matrix_ids)
        present = pos < len(sorted_ids)
        present[present] &= sorted_ids[pos[present]] == matrix_ids[present]
        covered_columns = np.flatnonzero(present)
        if len(covered_columns) < 2:
            raise RuntimeError("representation has insufficient coverage of the downstream methylation matrix")
        store_rows = order[pos[covered_columns]]
        with h5py.File(representation.store_path, "r") as handle:
            read_order = np.argsort(store_rows)
            sorted_values = np.asarray(handle[embedding_key][store_rows[read_order]], dtype=np.float32)
        inverse = np.empty_like(read_order)
        inverse[read_order] = np.arange(len(read_order))
        return sorted_values[inverse], covered_columns, store_dim

    store = HDF5RepresentationStore(
        representation.store_path, matrix_ids, id_key=representation.id_key, embedding_key=representation.embedding_key
    )
    covered_columns = np.flatnonzero(store.coverage_mask)
    if len(covered_columns) < 2:
        raise RuntimeError("representation has insufficient coverage of the downstream methylation matrix")
    return store.get_by_matrix_columns(covered_columns), covered_columns, store.dim


def _row_loader_factory(
    *,
    beta: np.ndarray,
    embedding: np.ndarray,
    prior_logit: np.ndarray,
    target: np.ndarray,
    rows: np.ndarray,
    split_label: str | None,
    beta_epsilon: float,
    batch_size: int,
    max_observed: int,
    seed: int,
):
    """Builds a `loader_factory` closure matching `probing.finetune.fine_tune_encoder`'s contract:
    calling it returns a fresh iterable of dict-batches. A fresh RNG per call keeps the observed
    CpG subsample reproducible-but-varying across epochs (light augmentation), while `rows` fixes
    which patients are ever visible to this particular loader (train-only vs. train+val+test)."""

    def factory():
        rng = np.random.default_rng(seed)
        for start in range(0, len(rows), batch_size):
            chunk_rows = rows[start : start + batch_size]
            per_patient = []
            max_n = 0
            for row in chunk_rows:
                finite = np.flatnonzero(np.isfinite(beta[row]))
                if len(finite) > max_observed:
                    finite = rng.choice(finite, size=max_observed, replace=False)
                per_patient.append(finite)
                max_n = max(max_n, len(finite))
            if max_n == 0:
                continue
            locus_dim = embedding.shape[1]
            observed_locus = np.zeros((len(chunk_rows), max_n, locus_dim), dtype=np.float32)
            observed_residual = np.zeros((len(chunk_rows), max_n), dtype=np.float32)
            observed_valid = np.zeros((len(chunk_rows), max_n), dtype=bool)
            for i, (row, columns) in enumerate(zip(chunk_rows, per_patient)):
                n = len(columns)
                observed_locus[i, :n] = embedding[columns]
                logit = beta_to_logit(beta[row, columns], beta_epsilon)
                observed_residual[i, :n] = logit - prior_logit[columns]
                observed_valid[i, :n] = True
            batch = {
                "observed_locus": torch.from_numpy(observed_locus),
                "observed_residual": torch.from_numpy(observed_residual),
                "observed_valid": torch.from_numpy(observed_valid),
                "target": torch.from_numpy(target[chunk_rows].astype(np.float32)),
            }
            if split_label is not None:
                batch["split"] = [split_label] * len(chunk_rows)
            yield batch

    return factory


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune a sparse-reconstruction encoder on a downstream phenotype")
    p.add_argument("--reconstruction-run", type=Path, required=True)
    p.add_argument("--methylation-h5", type=Path, required=True)
    p.add_argument("--phenotypes-parquet", type=Path, required=True)
    p.add_argument("--phenotype-id-column", default="sample_name")
    p.add_argument("--target-column", required=True)
    p.add_argument("--task-type", choices=("regression", "classification"), required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--representation", required=True)
    p.add_argument("--track", default="native_frozen")
    p.add_argument("--cpg-registry", type=Path, default=None)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--patient-split", default="0.8,0.1,0.1")
    p.add_argument("--max-observed", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--head-dropout", type=float, default=0.2)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--early-stopping-patience", type=int, default=5)
    p.add_argument(
        "--no-freeze-low-level",
        dest="freeze_low_level",
        action="store_false",
        help="unlock locus_adapter/tokenizer too, instead of only fine-tuning the patient-aggregation layers",
    )
    p.add_argument("--pooling", choices=("mean", "flatten"), default="mean")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-root", type=Path, default=None)
    args = p.parse_args()

    root = _repo_root()
    output_root = args.output_root or (root / "outputs")
    run_dir = args.reconstruction_run.resolve()
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())

    matrix_ids, sample_names = read_axis(args.methylation_h5)
    representation = resolve_representation(
        cfg["representation"], registry_path=Path(cfg["dataset"].get("cpg_registry", ".")), run_dir=run_dir, repo_root=root
    )
    embedding, covered_columns, locus_dim = _resolve_embedding(cfg, representation, matrix_ids, args.cpg_registry)

    phenotypes = pd.read_parquet(args.phenotypes_parquet).set_index(args.phenotype_id_column)
    missing = [pid for pid in sample_names if pid not in phenotypes.index]
    if missing:
        raise ValueError(f"phenotype table missing {len(missing)} patients")
    raw_target = phenotypes.loc[sample_names, args.target_column].to_numpy(dtype="float64")
    finite = np.isfinite(raw_target)
    sample_names = [pid for pid, keep in zip(sample_names, finite) if keep]
    if not finite.all():
        print(f"dropping {int((~finite).sum())} patients with missing {args.target_column!r}")

    with h5py.File(args.methylation_h5, "r") as handle:
        beta_full = np.asarray(handle["beta"][:, covered_columns], dtype=np.float32)
    beta = beta_full[finite]
    target = raw_target[finite]

    splits = patient_disjoint_split(sample_names, args.seed, tuple(float(x) for x in args.patient_split.split(",")))
    train_rows = splits["train"]

    target_mean, target_std = 0.0, 1.0
    if args.task_type == "regression":
        # MSE loss on raw units (e.g. age ~20-90) converges far too slowly at these learning
        # rates; z-score using train-only statistics before fitting. R^2/Pearson are invariant
        # to this affine transform, so only MAE below is reported in standardized units.
        target_mean = float(target[train_rows].mean())
        target_std = float(target[train_rows].std())
        if target_std < 1e-8:
            raise ValueError("train target has ~zero variance; cannot standardize")
        target = (target - target_mean) / target_std
    beta_epsilon = float(cfg["training"].get("beta_epsilon", 1e-4))
    prior_logit, usable, _ = compute_leakage_safe_priors(
        args.methylation_h5, splits["train"], covered_columns, len(matrix_ids), epsilon=beta_epsilon
    )
    if not usable.all():
        raise RuntimeError("some covered loci have no finite beta among downstream train patients")

    train_loader = _row_loader_factory(
        beta=beta, embedding=embedding, prior_logit=prior_logit, target=target, rows=train_rows, split_label=None,
        beta_epsilon=beta_epsilon, batch_size=args.batch_size, max_observed=args.max_observed, seed=args.seed,
    )
    validation_loader = _row_loader_factory(
        beta=beta, embedding=embedding, prior_logit=prior_logit, target=target, rows=splits["validation"],
        split_label=None, beta_epsilon=beta_epsilon, batch_size=args.batch_size, max_observed=args.max_observed,
        seed=args.seed,
    )
    eval_rows_by_split = {"train": splits["train"], "validation": splits["validation"], "test": splits["test"]}

    def eval_loader():
        for split_name, rows in eval_rows_by_split.items():
            factory = _row_loader_factory(
                beta=beta, embedding=embedding, prior_logit=prior_logit, target=target, rows=rows,
                split_label=split_name, beta_epsilon=beta_epsilon, batch_size=args.batch_size,
                max_observed=args.max_observed, seed=args.seed,
            )
            yield from factory()

    model_cfg = cfg["model"]
    model = build_reconstructor(locus_dim, model_cfg)
    state = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state))

    device = torch.device(args.device)
    result = fine_tune_encoder(
        model,
        loader_factory=train_loader,
        eval_loader_factory=lambda: list(eval_loader()),
        validation_loader_factory=validation_loader,
        task_type=args.task_type,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        head_dropout=args.head_dropout,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
        early_stopping_patience=args.early_stopping_patience,
        freeze_low_level=args.freeze_low_level,
        pooling=args.pooling,
    )

    out_dir = output_root / "embedding_probe" / args.dataset / args.representation / args.track / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.state_dict, out_dir / "finetuned_encoder_head.pt")

    payload = embedding_summary_payload(
        task="embedding_linear_probe",
        dataset=args.dataset,
        representation=args.representation,
        track=args.track,
        mode="fine_tuned",
        embedding_source_type="patient_embedding",
        embedding_dim=locus_dim,
        metrics=result.metrics,
        n_patients={"train": len(train_rows), "validation": len(splits["validation"]), "test": len(splits["test"])},
        checkpoint=str(run_dir),
        extra={
            "epochs": args.epochs,
            "best_epoch": result.best_epoch,
            "validation_history": result.history,
            "lr": args.lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "head_dropout": args.head_dropout,
            "freeze_low_level": args.freeze_low_level,
            "early_stopping_patience": args.early_stopping_patience,
            "target_standardization": (
                {"mean": target_mean, "std": target_std, "note": "MAE below is in z-scored units; R2/Pearson are affine-invariant"}
                if args.task_type == "regression"
                else None
            ),
        },
    )
    (out_dir / "summary_finetuned.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
