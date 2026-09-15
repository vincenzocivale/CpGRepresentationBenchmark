#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cpg_repr_benchmark.config import config_fingerprint, load_config
from cpg_repr_benchmark.data.chromosomes import normalize_chromosome
from cpg_repr_benchmark.data.classification import MethylomeTaskDataset, load_aligned_targets
from cpg_repr_benchmark.data.masking import MaskFractionBatchSampler
from cpg_repr_benchmark.data.methylation import read_axis
from cpg_repr_benchmark.data.splits import patient_disjoint_split
from cpg_repr_benchmark.data.universe import load_locus_protocol
from cpg_repr_benchmark.experiments.run_store import (
    create_run_dir,
    git_revision,
    write_experiment_manifest,
    write_summary,
)
from cpg_repr_benchmark.models.classification import MethylomeTaskModel
from cpg_repr_benchmark.representations.hdf5_store import HDF5RepresentationStore
from cpg_repr_benchmark.representations.resolver import resolve_representation
from cpg_repr_benchmark.training.priors import compute_leakage_safe_priors
from cpg_repr_benchmark.training.supervised import evaluate_supervised, train_supervised


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _scoped_columns(cpg_ids: np.ndarray, registry: Path | None, chromosomes: list[str] | None) -> np.ndarray:
    columns = np.arange(len(cpg_ids), dtype=np.int64)
    if not chromosomes:
        return columns
    if registry is None:
        raise ValueError("dataset.chromosomes requires dataset.cpg_registry")
    table = pd.read_parquet(registry, columns=["cpg_idx", "chr"])
    chrom = table.set_index("cpg_idx")["chr"].reindex(cpg_ids)
    if chrom.isna().any():
        raise ValueError("CpG registry does not cover the methylation matrix")
    requested = {normalize_chromosome(x) for x in chromosomes}
    return columns[chrom.map(normalize_chromosome).isin(requested).to_numpy()]


def _loader(dataset: MethylomeTaskDataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    kwargs = {
        "batch_sampler": MaskFractionBatchSampler(dataset, batch_size, shuffle=shuffle),
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if workers:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Representation-controlled methylome phenotype benchmark")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    root = _root()
    cfg = load_config(args.config)
    cfg["experiment"].setdefault("task", "classification")
    seed = int(cfg["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.mode == "evaluate":
        if args.run_dir is None:
            raise ValueError("--run-dir is required for --mode evaluate")
        run_dir = args.run_dir.resolve()
    else:
        run_dir = create_run_dir(cfg, root)

    matrix_path = _path(root, cfg["dataset"]["methylation_h5"])
    registry_value = cfg["dataset"].get("cpg_registry")
    registry_path = _path(root, registry_value) if registry_value else None
    cpg_ids, sample_names = read_axis(matrix_path)
    candidate_columns = _scoped_columns(cpg_ids, registry_path, cfg["dataset"].get("chromosomes"))
    common_protocol = cfg["dataset"].get("locus_protocol")
    common_protocol_path = _path(root, common_protocol) if common_protocol else None
    if common_protocol_path is not None:
        _, protocol_columns = load_locus_protocol(common_protocol_path, cpg_ids)
        if not np.isin(protocol_columns, candidate_columns).all():
            raise ValueError("dataset.locus_protocol contains CpGs outside the configured chromosome scope")
        candidate_columns = protocol_columns

    representation = resolve_representation(
        cfg["representation"], registry_path=registry_path, run_dir=run_dir, repo_root=root
    )
    dataset_namespace = cfg["dataset"].get("cpg_namespace")
    if dataset_namespace and representation.cpg_namespace != dataset_namespace:
        raise ValueError(
            f"dataset uses cpg_namespace={dataset_namespace!r} but representation uses "
            f"{representation.cpg_namespace!r}; re-key the representation before comparing it"
        )
    store = HDF5RepresentationStore(
        representation.store_path,
        cpg_ids,
        id_key=representation.id_key,
        embedding_key=representation.embedding_key,
    )
    if not store.coverage_mask[candidate_columns].all():
        raise ValueError("representation does not cover the configured classification CpG universe")

    phenotype_cfg = cfg["phenotype"]
    phenotype_path = _path(root, phenotype_cfg["parquet"])
    task_type = str(phenotype_cfg["task_type"])
    eligible_rows, eligible_targets, classes = load_aligned_targets(
        phenotype_path,
        sample_names,
        sample_column=phenotype_cfg.get("sample_column", "sample_name"),
        target_column=phenotype_cfg["target_column"],
        task_type=task_type,
        classes=phenotype_cfg.get("classes"),
    )
    eligible_names = [sample_names[i] for i in eligible_rows]
    split_local = patient_disjoint_split(
        eligible_names,
        seed=seed,
        fractions=tuple(cfg["dataset"].get("patient_split", [0.8, 0.1, 0.1])),
    )
    splits = {name: eligible_rows[idx] for name, idx in split_local.items()}
    target_by_matrix_row = {int(row): target for row, target in zip(eligible_rows, eligible_targets)}
    split_targets = {
        name: np.asarray([target_by_matrix_row[int(row)] for row in rows], dtype=eligible_targets.dtype)
        for name, rows in splits.items()
    }

    prior, usable, global_beta = compute_leakage_safe_priors(
        matrix_path,
        splits["train"],
        candidate_columns,
        len(cpg_ids),
        epsilon=float(cfg["training"].get("beta_epsilon", 1e-4)),
    )
    candidate_columns = candidate_columns[usable]
    if len(candidate_columns) < int(cfg["training"]["panel_size"]):
        raise ValueError("too few CpGs with a finite training prior for configured panel_size")

    requested_device = str(cfg["training"].get("device", "auto"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested_device}, but CUDA is unavailable")
    resolved_device = (
        requested_device
        if requested_device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = torch.device(resolved_device)
    output_dim = 1 if task_type in {"regression", "binary"} else len(classes or [])
    model_cfg = cfg["model"]
    model = MethylomeTaskModel(
        raw_locus_dim=store.dim,
        task_type=task_type,
        output_dim=output_dim,
        locus_latent_dim=int(model_cfg.get("locus_latent_dim", 256)),
        token_dim=int(model_cfg.get("token_dim", 256)),
        patient_dim=int(model_cfg.get("patient_dim", 256)),
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )

    if args.mode in {"train", "all"}:
        train_ds = MethylomeTaskDataset(
            methylation_h5=matrix_path,
            representation_store=store,
            prior_logit_full=prior,
            sample_indices=splits["train"],
            targets=split_targets["train"],
            candidate_columns=candidate_columns,
            panel_size=int(cfg["training"]["panel_size"]),
            mask_fractions=cfg["training"].get("mask_fractions", [0.0]),
            seed=seed,
            beta_epsilon=float(cfg["training"].get("beta_epsilon", 1e-4)),
        )
        validation_ds = MethylomeTaskDataset(
            methylation_h5=matrix_path,
            representation_store=store,
            prior_logit_full=prior,
            sample_indices=splits["validation"],
            targets=split_targets["validation"],
            candidate_columns=candidate_columns,
            panel_size=int(cfg["training"]["panel_size"]),
            mask_fractions=[float(cfg["evaluation"].get("selection_mask_fraction", 0.0))],
            seed=seed,
            beta_epsilon=float(cfg["training"].get("beta_epsilon", 1e-4)),
        )
        train_supervised(
            model,
            _loader(
                train_ds,
                int(cfg["training"]["batch_size"]),
                int(cfg["training"].get("num_workers", 0)),
                True,
            ),
            _loader(
                validation_ds,
                int(cfg["training"]["batch_size"]),
                min(2, int(cfg["training"].get("num_workers", 0))),
                False,
            ),
            task_type=task_type,
            device=device,
            epochs=int(cfg["training"]["epochs"]),
            learning_rate=float(cfg["training"]["learning_rate"]),
            weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
            mixed_precision=bool(cfg["training"].get("mixed_precision", True)),
            output_dir=run_dir,
        )
        np.savez_compressed(run_dir / "patient_split.npz", **splits)
        write_experiment_manifest(
            run_dir,
            {
                "experiment": cfg["experiment"]["name"],
                "task": cfg["experiment"]["task"],
                "config_fingerprint": config_fingerprint(cfg),
                "code_git_commit": git_revision(root),
                "dataset": cfg["dataset"].get("name", "dataset"),
                "representation": representation.name,
                "representation_family": representation.family,
                "representation_track": representation.track,
                "phenotype_target": phenotype_cfg["target_column"],
                "task_type": task_type,
                "classes": classes,
                "n_train": int(len(splits["train"])),
                "n_validation": int(len(splits["validation"])),
                "n_test": int(len(splits["test"])),
                "n_input_loci": int(len(candidate_columns)),
                "locus_protocol": str(common_protocol_path) if common_protocol_path else None,
                "cpg_namespace": dataset_namespace,
                "global_train_beta_prior": float(global_beta),
            },
        )

    if args.mode in {"evaluate", "all"}:
        checkpoint = run_dir / "checkpoints" / "best.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        model.to(device)
        if args.mode == "evaluate":
            split_file = np.load(run_dir / "patient_split.npz")
            splits = {name: split_file[name] for name in ("train", "validation", "test")}
            split_targets["test"] = np.asarray(
                [target_by_matrix_row[int(row)] for row in splits["test"]], dtype=eligible_targets.dtype
            )
        sweep = {}
        for fraction in [float(x) for x in cfg["evaluation"]["mask_fractions"]]:
            dataset = MethylomeTaskDataset(
                methylation_h5=matrix_path,
                representation_store=store,
                prior_logit_full=prior,
                sample_indices=splits["test"],
                targets=split_targets["test"],
                candidate_columns=candidate_columns,
                panel_size=int(cfg["evaluation"].get("panel_size", cfg["training"]["panel_size"])),
                mask_fractions=[fraction],
                seed=seed,
                beta_epsilon=float(cfg["training"].get("beta_epsilon", 1e-4)),
            )
            eval_loader = _loader(
                dataset,
                int(cfg["evaluation"].get("batch_size", cfg["training"]["batch_size"])),
                int(cfg["evaluation"].get("num_workers", 0)),
                False,
            )
            metrics, outputs = evaluate_supervised(
                model,
                eval_loader,
                task_type=task_type,
                device=device,
            )
            key = f"mask_{fraction:.2f}"
            out = run_dir / "evaluation" / key
            out.mkdir(parents=True, exist_ok=True)
            payload = {**metrics, "mask_fraction": fraction}
            (out / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True))
            if bool(cfg["evaluation"].get("save_predictions", False)):
                np.savez_compressed(out / "predictions.npz", **outputs)
            sweep[key] = payload
        summary = {
            "experiment": cfg["experiment"]["name"],
            "task": cfg["experiment"]["task"],
            "dataset": cfg["dataset"].get("name", "dataset"),
            "representation": representation.name,
            "representation_family": representation.family,
            "representation_track": representation.track,
            "target": phenotype_cfg["target_column"],
            "task_type": task_type,
            "evaluation": sweep,
        }
        write_summary(run_dir, summary)
        print(json.dumps(summary, indent=2, allow_nan=True))
    store.close()
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
