#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cpg_repr_benchmark.config import config_fingerprint, load_config
from cpg_repr_benchmark.data.chromosomes import normalize_chromosome
from cpg_repr_benchmark.data.masking import MaskFractionBatchSampler, MaskingDataset
from cpg_repr_benchmark.data.methylation import read_axis
from cpg_repr_benchmark.data.splits import load_or_create_locus_protocol, patient_disjoint_split
from cpg_repr_benchmark.experiments.run_store import (
    create_run_dir,
    git_revision,
    write_experiment_manifest,
    write_summary,
)
from cpg_repr_benchmark.models.model import MaskedMethylomeReconstructor
from cpg_repr_benchmark.representations.hdf5_store import HDF5RepresentationStore
from cpg_repr_benchmark.representations.online import build_online_store
from cpg_repr_benchmark.representations.resolver import resolve_representation
from cpg_repr_benchmark.training.engine import evaluate_loader, train_model
from cpg_repr_benchmark.training.priors import compute_leakage_safe_priors


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _scoped_columns(cpg_ids: np.ndarray, registry_path: Path | None, chromosomes: list[str] | None) -> np.ndarray:
    columns = np.arange(len(cpg_ids), dtype=np.int64)
    if not chromosomes:
        return columns
    if registry_path is None:
        raise ValueError("dataset.chromosomes requires dataset.cpg_registry")
    table = pd.read_parquet(registry_path, columns=["cpg_idx", "chr"])
    if table["cpg_idx"].duplicated().any():
        raise ValueError("cpg registry has duplicate cpg_idx")
    chrom = table.set_index("cpg_idx")["chr"].reindex(cpg_ids)
    if chrom.isna().any():
        raise ValueError(f"registry is missing {int(chrom.isna().sum())} methylation CpGs")
    requested = {normalize_chromosome(x) for x in chromosomes}
    normalized = chrom.map(normalize_chromosome)
    return columns[normalized.isin(requested).to_numpy()]


def _validate_representation_scope(dataset_cfg: dict, representation_cfg: dict) -> None:
    supported = representation_cfg.get("supported_chromosomes")
    if not supported:
        return
    dataset_chromosomes = dataset_cfg.get("chromosomes")
    if not dataset_chromosomes:
        raise ValueError(
            f"representation {representation_cfg.get('name')!r} is limited to {supported}; "
            "dataset.chromosomes must explicitly restrict the benchmark scope"
        )
    supported_set = {normalize_chromosome(x) for x in supported}
    requested_set = {normalize_chromosome(x) for x in dataset_chromosomes}
    if not requested_set.issubset(supported_set):
        raise ValueError(
            f"dataset requests chromosomes {sorted(requested_set)} but representation "
            f"{representation_cfg.get('name')!r} supports only {sorted(supported_set)}"
        )


def _build_loader(dataset: MaskingDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    kwargs = {
        "batch_sampler": MaskFractionBatchSampler(dataset, batch_size, shuffle=shuffle),
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def _device(training_cfg: dict) -> torch.device:
    requested = str(training_cfg.get("device", "auto"))
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested}, but CUDA is unavailable")
    if requested.startswith("cuda"):
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_locus_split(run_dir: Path, split, cpg_ids: np.ndarray) -> None:
    np.savez_compressed(
        run_dir / "locus_split.npz",
        train_matrix_columns=split.train_columns,
        heldout_matrix_columns=split.heldout_columns,
        train_cpg_idx=cpg_ids[split.train_columns],
        heldout_cpg_idx=cpg_ids[split.heldout_columns],
    )


def _load_locus_split(run_dir: Path):
    data = np.load(run_dir / "locus_split.npz")
    return data["train_matrix_columns"], data["heldout_matrix_columns"]


def _evaluate_views(
    *,
    cfg: dict,
    run_dir: Path,
    model: MaskedMethylomeReconstructor,
    store,
    matrix_path: Path,
    prior: np.ndarray,
    test_rows: np.ndarray,
    train_columns: np.ndarray,
    heldout_columns: np.ndarray,
    device: torch.device,
) -> dict:
    eval_cfg = cfg["evaluation"]
    batch_size = int(eval_cfg.get("batch_size", cfg["training"]["batch_size"]))
    num_workers = int(eval_cfg.get("num_workers", 0))
    panel_size = int(eval_cfg.get("panel_size", cfg["training"]["panel_size"]))
    fractions = [float(x) for x in eval_cfg["mask_fractions"]]
    seed = int(cfg["training"]["seed"])
    save_predictions = bool(eval_cfg.get("save_predictions", False))
    results: dict[str, dict] = {}
    views = {
        "seen": (train_columns, train_columns, "empirical train-patient prior on train loci"),
    }
    if len(heldout_columns):
        views["unseen_locus"] = (train_columns, heldout_columns, "global train-patients x train-loci prior")
    for view_name, (context_pool, target_pool, prior_policy) in views.items():
        view_results: dict[str, dict] = {}
        for fraction in fractions:
            dataset = MaskingDataset(
                methylation_h5=matrix_path,
                representation_store=store,
                prior_logit_full=prior,
                sample_indices=test_rows,
                context_columns=context_pool,
                target_columns=target_pool,
                panel_size=panel_size,
                mask_fractions=[fraction],
                seed=seed,
                beta_epsilon=float(cfg["training"].get("beta_epsilon", 1e-4)),
            )
            loader = _build_loader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            metrics, outputs = evaluate_loader(model, loader, device)
            key = f"mask_{fraction:.2f}"
            out_dir = run_dir / "evaluation" / view_name / key
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {**metrics, "mask_fraction": fraction, "view": view_name, "prior_policy": prior_policy}
            (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True))
            if save_predictions:
                np.savez_compressed(out_dir / "predictions.npz", **outputs)
            view_results[key] = payload
        results[view_name] = view_results
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate the representation-controlled CpG masking benchmark")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--run-dir", type=Path, help="existing run directory for --mode evaluate")
    args = parser.parse_args()

    repo_root = _repo_root()
    cfg = load_config(args.config)
    seed = int(cfg["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.mode == "evaluate":
        if args.run_dir is None:
            raise ValueError("--run-dir is required for --mode evaluate")
        run_dir = args.run_dir.resolve()
    else:
        run_dir = create_run_dir(cfg, repo_root)

    dataset_cfg = cfg["dataset"]
    matrix_path = Path(dataset_cfg["methylation_h5"]).expanduser()
    if not matrix_path.is_absolute():
        matrix_path = (repo_root / matrix_path).resolve()
    registry_path = dataset_cfg.get("cpg_registry")
    registry_path = Path(registry_path).expanduser() if registry_path else None
    if registry_path is not None and not registry_path.is_absolute():
        registry_path = (repo_root / registry_path).resolve()

    cpg_ids, sample_names = read_axis(matrix_path)
    representation_cfg = cfg["representation"]
    _validate_representation_scope(dataset_cfg, representation_cfg)
    representation_mode = str(representation_cfg.get("mode", "precomputed"))
    if representation_mode == "online":
        store = build_online_store(
            representation_cfg, cpg_ids, repo_root=repo_root, registry_path=registry_path
        )
        class _Representation:
            pass
        representation = _Representation()
        representation.name = str(representation_cfg["name"])
        representation.mode = "online"
        representation.dim = store.dim
        manifest = {
            "schema_version": 1,
            "name": representation.name,
            "mode": "online",
            "source": representation_cfg.get("source", "unspecified"),
            "embedding_dim": store.dim,
            "provider": representation_cfg.get("provider", {}),
            "provenance": representation_cfg.get("provenance", {}),
        }
        (run_dir / "representation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
        if int(cfg["training"].get("num_workers", 0)) != 0 or int(cfg["evaluation"].get("num_workers", 0)) != 0:
            raise ValueError("representation.mode=online requires training/evaluation num_workers=0")
    else:
        representation = resolve_representation(
            representation_cfg,
            registry_path=registry_path or Path("."),
            run_dir=run_dir,
            repo_root=repo_root,
        )
        store = HDF5RepresentationStore(
            representation.store_path,
            cpg_ids,
            id_key=representation.id_key,
            embedding_key=representation.embedding_key,
        )

    # The benchmark universe is dataset-defined, not representation-defined. A representation
    # that lacks protocol loci fails explicitly instead of silently receiving an easier/different split.
    scope_columns = _scoped_columns(cpg_ids, registry_path, dataset_cfg.get("chromosomes"))
    candidate_columns = scope_columns
    if len(candidate_columns) < 2:
        raise RuntimeError("dataset scope has insufficient CpGs")

    patient_splits = patient_disjoint_split(sample_names, seed, tuple(cfg["dataset"].get("patient_split", [0.8, 0.1, 0.1])))
    device = _device(cfg["training"])

    if args.mode in {"train", "all"}:
        locus_cfg = cfg["experiment"]["locus_split"]
        protocol_path = Path(locus_cfg.get(
            "protocol_path",
            f"data/protocols/{dataset_cfg.get('name', 'dataset')}_masking_seed{int(locus_cfg.get('seed', seed))}.npz",
        ))
        if not protocol_path.is_absolute():
            protocol_path = (repo_root / protocol_path).resolve()
        locus_split = load_or_create_locus_protocol(
            protocol_path,
            candidate_columns,
            cpg_ids,
            seed=int(locus_cfg.get("seed", seed)),
            heldout_fraction=float(locus_cfg["heldout_fraction"]),
        )
        required_columns = np.concatenate([locus_split.train_columns, locus_split.heldout_columns])
        missing_coverage = required_columns[~store.coverage_mask[required_columns]]
        if len(missing_coverage):
            examples = cpg_ids[missing_coverage[:10]].tolist()
            raise RuntimeError(
                f"representation {representation.name!r} is missing {len(missing_coverage)} CpGs from the shared protocol; "
                f"examples={examples}. Build a common-universe protocol explicitly rather than silently dropping loci."
            )
        prior, train_locus_usable, global_train_beta = compute_leakage_safe_priors(
            matrix_path,
            patient_splits["train"],
            locus_split.train_columns,
            len(cpg_ids),
            epsilon=float(cfg["training"].get("beta_epsilon", 1e-4)),
        )
        train_columns = locus_split.train_columns[train_locus_usable]
        heldout_columns = locus_split.heldout_columns
        if len(train_columns) < 2:
            raise RuntimeError("locus split became empty after train-prior availability filtering")
        if len(heldout_columns) < 1 and float(locus_cfg["heldout_fraction"]) > 0.0:
            raise RuntimeError("locus split became empty after train-prior availability filtering")
        # Persist the exact post-filter split used by training/evaluation.
        class _Split: pass
        persisted = _Split()
        persisted.train_columns, persisted.heldout_columns = train_columns, heldout_columns
        _save_locus_split(run_dir, persisted, cpg_ids)
        np.save(run_dir / "prior_logit.npy", prior)
        np.savez_compressed(
            run_dir / "patient_split.npz",
            train=patient_splits["train"], validation=patient_splits["validation"], test=patient_splits["test"]
        )

        training_cfg = cfg["training"]
        validation_fraction = float(cfg["evaluation"].get("selection_mask_fraction", 0.5))
        train_ds = MaskingDataset(
            methylation_h5=matrix_path,
            representation_store=store,
            prior_logit_full=prior,
            sample_indices=patient_splits["train"],
            context_columns=train_columns,
            target_columns=train_columns,
            panel_size=int(training_cfg["panel_size"]),
            mask_fractions=training_cfg["mask_fractions"],
            seed=seed,
            beta_epsilon=float(training_cfg.get("beta_epsilon", 1e-4)),
        )
        val_ds = MaskingDataset(
            methylation_h5=matrix_path,
            representation_store=store,
            prior_logit_full=prior,
            sample_indices=patient_splits["validation"],
            context_columns=train_columns,
            target_columns=train_columns,
            panel_size=int(training_cfg["panel_size"]),
            mask_fractions=[validation_fraction],
            seed=seed,
            beta_epsilon=float(training_cfg.get("beta_epsilon", 1e-4)),
        )
        train_loader = _build_loader(train_ds, int(training_cfg["batch_size"]), int(training_cfg.get("num_workers", 0)), True)
        val_loader = _build_loader(val_ds, int(training_cfg["batch_size"]), min(2, int(training_cfg.get("num_workers", 0))), False)
        model_cfg = cfg["model"]
        model = MaskedMethylomeReconstructor(
            raw_locus_dim=store.dim,
            locus_latent_dim=int(model_cfg.get("locus_latent_dim", 256)),
            token_dim=int(model_cfg.get("token_dim", 256)),
            patient_dim=int(model_cfg.get("patient_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        )
        write_experiment_manifest(
            run_dir,
            {
                "experiment": cfg["experiment"]["name"],
                "config_fingerprint": config_fingerprint(cfg),
                "code_git_commit": git_revision(repo_root),
                "dataset": dataset_cfg.get("name", "dataset"),
                "representation": representation.name,
                "representation_dim": store.dim,
                "dataset_chromosomes": dataset_cfg.get("chromosomes"),
                "seed": seed,
                "device": str(device),
                "n_train_patients_rows": int(len(patient_splits["train"])),
                "n_validation_rows": int(len(patient_splits["validation"])),
                "n_test_rows": int(len(patient_splits["test"])),
                "n_train_loci": int(len(train_columns)),
                "n_heldout_loci": int(len(heldout_columns)),
                "global_train_beta_prior": float(global_train_beta),
                "unseen_locus_prior_policy": "global train-patients x train-loci mean",
                "locus_protocol": str(protocol_path),
                "mask_fractions_train": [float(x) for x in training_cfg["mask_fractions"]],
                "mask_fractions_evaluation": [float(x) for x in cfg["evaluation"]["mask_fractions"]],
            },
        )
        train_model(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=int(training_cfg["epochs"]),
            learning_rate=float(training_cfg["learning_rate"]),
            weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
            mixed_precision=bool(training_cfg.get("mixed_precision", True)),
            output_dir=run_dir,
        )
    else:
        prior = np.load(run_dir / "prior_logit.npy")
        train_columns, heldout_columns = _load_locus_split(run_dir)
        required_columns = np.concatenate([train_columns, heldout_columns])
        missing_coverage = required_columns[~store.coverage_mask[required_columns]]
        if len(missing_coverage):
            raise RuntimeError(f"representation is missing {len(missing_coverage)} loci required by this run")
        ps = np.load(run_dir / "patient_split.npz")
        patient_splits = {k: ps[k] for k in ("train", "validation", "test")}
        model_cfg = cfg["model"]
        model = MaskedMethylomeReconstructor(
            raw_locus_dim=store.dim,
            locus_latent_dim=int(model_cfg.get("locus_latent_dim", 256)),
            token_dim=int(model_cfg.get("token_dim", 256)),
            patient_dim=int(model_cfg.get("patient_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        )

    if args.mode in {"evaluate", "all"}:
        checkpoint = run_dir / "checkpoints" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"best checkpoint unavailable: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        model.to(device)
        results = _evaluate_views(
            cfg=cfg,
            run_dir=run_dir,
            model=model,
            store=store,
            matrix_path=matrix_path,
            prior=prior,
            test_rows=patient_splits["test"],
            train_columns=train_columns,
            heldout_columns=heldout_columns,
            device=device,
        )
        summary = {
            "experiment": cfg["experiment"]["name"],
            "dataset": dataset_cfg.get("name", "dataset"),
            "representation": representation.name,
            "representation_dim": store.dim,
            "checkpoint": str(checkpoint),
            "seed": seed,
            "n_train_loci": int(len(train_columns)),
            "n_heldout_loci": int(len(heldout_columns)),
            "evaluation": results,
        }
        write_summary(run_dir, summary)
        print(json.dumps(summary, indent=2, allow_nan=True))

    store.close()
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
