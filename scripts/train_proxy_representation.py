#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from cpg_repr_benchmark.data.chromosomes import normalize_chromosome
from cpg_repr_benchmark.data.methylation import read_axis
from cpg_repr_benchmark.data.splits import load_or_create_locus_protocol, patient_disjoint_split
from cpg_repr_benchmark.experiments.run_store import (
    create_run_dir,
    git_revision,
    write_experiment_manifest,
    write_summary,
)
from cpg_repr_benchmark.proxy.model import (
    DenseProxyEncoder,
    FunctionalAnnotationEncoder,
    MeanMethylationProxy,
)
from cpg_repr_benchmark.proxy.stores import DenseProxyInputStore, FunctionalProxyInputStore
from cpg_repr_benchmark.proxy.training import (
    export_proxy_embeddings,
    split_proxy_train_validation,
    train_mean_methylation_proxy,
)
from cpg_repr_benchmark.training.priors import compute_leakage_safe_priors


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a train-locus-only mean-methylation proxy encoder")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    root = _root()
    cfg = yaml.safe_load(args.config.read_text())
    for key in ("experiment", "dataset", "representation", "proxy", "training"):
        if key not in cfg:
            raise ValueError(f"proxy config missing top-level key {key!r}")
    cfg.setdefault("model", {})
    cfg.setdefault("evaluation", {})
    cfg["experiment"].setdefault("task", "proxy_mean_methylation")
    seed = int(cfg["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    run_dir = create_run_dir(cfg, root)

    matrix_path = _path(root, cfg["dataset"]["methylation_h5"])
    registry = cfg["dataset"].get("cpg_registry")
    registry_path = _path(root, registry) if registry else None
    cpg_ids, sample_names = read_axis(matrix_path)
    scope_columns = _scoped_columns(cpg_ids, registry_path, cfg["dataset"].get("chromosomes"))
    patient_split = patient_disjoint_split(
        sample_names,
        seed=seed,
        fractions=tuple(cfg["dataset"].get("patient_split", [0.8, 0.1, 0.1])),
    )

    protocol_cfg = cfg["experiment"]["locus_split"]
    protocol_path = _path(root, protocol_cfg["protocol_path"])
    locus_split = load_or_create_locus_protocol(
        protocol_path,
        scope_columns,
        cpg_ids,
        seed=int(protocol_cfg["seed"]),
        heldout_fraction=float(protocol_cfg["heldout_fraction"]),
    )

    # Mean-beta targets are fitted ONLY on downstream train loci. Held-out loci are not
    # read here and therefore cannot influence proxy training or checkpoint selection.
    prior_logit, usable, global_beta = compute_leakage_safe_priors(
        matrix_path,
        patient_split["train"],
        locus_split.train_columns,
        len(cpg_ids),
        epsilon=float(cfg["training"].get("beta_epsilon", 1e-4)),
    )
    usable_train = locus_split.train_columns[usable]
    target_by_column = np.full(len(cpg_ids), np.nan, dtype=np.float32)
    target_by_column[usable_train] = 1.0 / (1.0 + np.exp(-prior_logit[usable_train]))
    proxy_train, proxy_validation = split_proxy_train_validation(
        usable_train,
        cpg_ids,
        seed=seed,
        validation_fraction=float(cfg["proxy"].get("validation_fraction", 0.1)),
    )
    fitted_columns = np.concatenate([proxy_train, proxy_validation])
    if np.intersect1d(fitted_columns, locus_split.heldout_columns).size:
        raise AssertionError("proxy fit leaked downstream held-out CpGs")

    source_path = _path(root, cfg["representation"]["source_store_h5"])
    input_kind = str(cfg["representation"]["input_kind"])
    if input_kind == "dense_hdf5":
        store = DenseProxyInputStore(
            source_path,
            cpg_ids,
            id_key=cfg["representation"].get("id_key", "auto"),
            embedding_key=cfg["representation"].get("embedding_key", "auto"),
        )
        encoder = DenseProxyEncoder(
            raw_dim=store.raw_dim,
            latent_dim=int(cfg["proxy"].get("embedding_dim", 256)),
            n_blocks=int(cfg["proxy"].get("n_blocks", 2)),
            dropout=float(cfg["proxy"].get("dropout", 0.1)),
        )
    elif input_kind == "functional_hdf5":
        store = FunctionalProxyInputStore(source_path, cpg_ids)
        encoder = FunctionalAnnotationEncoder(
            n_tracks=int(cfg["representation"].get("n_tracks", 4165)),
            dense_dim=store.dense_dim,
            latent_dim=int(cfg["proxy"].get("embedding_dim", 256)),
            n_blocks=int(cfg["proxy"].get("n_blocks", 8)),
            dropout=float(cfg["proxy"].get("dropout", 0.1)),
        )
    else:
        raise ValueError(f"unsupported representation.input_kind={input_kind!r}")

    required = np.concatenate([locus_split.train_columns, locus_split.heldout_columns])
    if not store.coverage_mask[required].all():
        raise ValueError("proxy source does not cover the persistent benchmark CpG protocol")

    requested_device = str(cfg["training"].get("device", "auto"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested_device}, but CUDA is unavailable")
    resolved_device = (
        requested_device
        if requested_device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = torch.device(resolved_device)
    model = MeanMethylationProxy(encoder, encoder.output_dim)
    train_mean_methylation_proxy(
        model,
        store,
        target_by_column,
        proxy_train,
        proxy_validation,
        device=device,
        epochs=int(cfg["training"]["epochs"]),
        batch_size=int(cfg["training"]["batch_size"]),
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
        mixed_precision=bool(cfg["training"].get("mixed_precision", True)),
        seed=seed,
        output_dir=run_dir,
    )

    output_h5 = _path(root, cfg["proxy"]["output_store_h5"])
    export_columns = np.sort(
        np.concatenate([locus_split.train_columns, locus_split.heldout_columns])
    )
    checkpoint = run_dir / "checkpoints" / "best.pt"
    export_proxy_embeddings(
        model,
        store,
        cpg_ids,
        export_columns,
        device=device,
        batch_size=int(cfg["training"].get("export_batch_size", cfg["training"]["batch_size"])),
        output_h5=output_h5,
        attrs={
            "checkpoint": str(checkpoint),
            "representation": cfg["representation"]["name"],
            "track": "proxy_aligned",
            "proxy_objective": "train_patient_mean_beta",
            "protocol": str(protocol_path),
            "git_commit": git_revision(root) or "unknown",
            "cpg_namespace": cfg["dataset"].get("cpg_namespace", "legacy_unspecified"),
            "reference_build": cfg["dataset"].get("reference_build", "unspecified"),
            "coordinate_convention": cfg["dataset"].get("coordinate_convention", "unspecified"),
        },
    )
    manifest = {
        "task": "proxy_mean_methylation",
        "representation": cfg["representation"]["name"],
        "input_kind": input_kind,
        "proxy_train_loci": int(len(proxy_train)),
        "proxy_validation_loci": int(len(proxy_validation)),
        "downstream_heldout_loci_touched": False,
        "output_store_h5": str(output_h5),
        "global_train_beta": float(global_beta),
    }
    write_experiment_manifest(run_dir, manifest)
    write_summary(run_dir, manifest)
    print(json.dumps(manifest, indent=2))
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
