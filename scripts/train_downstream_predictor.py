#!/usr/bin/env python3
"""Step C: train the FROZEN oracle downstream age predictor.

MLP regressor over raw beta values (not embeddings) restricted to the canonical universe U,
trained once on real complete GSE40279 methylomes for TRAIN patients, model-selected on VAL
patients, and frozen. This exact checkpoint is reused unmodified for every representation/
observed-count arm in scripts/run_reconstruction_downstream_benchmark.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from cpg_repr_benchmark.data.splits import patient_disjoint_split
from cpg_repr_benchmark.training.downstream_predictor import fill_locus_mean, train_downstream_predictor

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_universe(seed: int = 17) -> np.ndarray:
    npz = REPO_ROOT / "data/protocols/reduced_observation/gse40279_age_universe_seed17.npz"
    with np.load(npz) as data:
        return np.asarray(data["universe_cpg_idx"], dtype=np.int64)


def load_beta_u(methylation_h5: Path, universe_cpg_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(methylation_h5, "r") as f:
        meth_ids = np.asarray(f["cpg_idx"][:], dtype=np.int64)
        order = np.argsort(meth_ids)
        sorted_ids = meth_ids[order]
        loc = np.searchsorted(sorted_ids, universe_cpg_idx)
        if not np.array_equal(sorted_ids[loc], universe_cpg_idx):
            raise ValueError("universe CpGs missing from methylation matrix")
        matrix_columns = np.sort(order[loc])
        beta = np.asarray(f["beta"][:, matrix_columns], dtype=np.float32)
        sample_names = np.asarray(f["sample_name"][:]).astype(str)
    return beta, sample_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/reconstruction_downstream/gse40279_age/seed_17/downstream_predictor")
    args = parser.parse_args()

    seed = args.seed
    dataset_cfg_h5 = REPO_ROOT / "data/processed/GSE40279/methylation.h5"
    phenotypes = pd.read_parquet(REPO_ROOT / "data/processed/GSE40279/phenotypes.parquet")

    universe = load_universe(seed)
    beta_u, sample_names = load_beta_u(dataset_cfg_h5, universe)

    phenotypes = phenotypes.set_index("sample_name").loc[sample_names].reset_index()
    age = phenotypes["age"].to_numpy(dtype=np.float64)

    splits = patient_disjoint_split(list(sample_names), seed, (0.80, 0.10, 0.10))
    train_rows, val_rows, test_rows = splits["train"], splits["validation"], splits["test"]

    beta_filled, locus_mean = fill_locus_mean(beta_u, train_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = train_downstream_predictor(
        beta_u=beta_filled,
        age=age,
        train_rows=train_rows,
        val_rows=val_rows,
        device=device,
        epochs=args.epochs,
        seed=seed,
        output_dir=args.output_dir,
    )
    np.savez_compressed(
        args.output_dir / "artifacts.npz",
        universe_cpg_idx=universe,
        locus_mean=locus_mean,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
    )
    print(f"best_val_mae={result['best_val_mae']:.4f}")
    print(f"RUN_DIR={args.output_dir}")


if __name__ == "__main__":
    main()
