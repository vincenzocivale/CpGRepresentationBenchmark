#!/usr/bin/env python3
"""Steps D/E/F of the reduced-observation reconstruction -> frozen downstream predictor slice.

For each representation arm and each observed-CpG count in {256, 1024, 4096}, reconstructs
the CpGs missing from that nested observed subset (using the arm's frozen
MaskedMethylomeReconstructor, or the train-mean prior, or the real values for full_oracle),
then feeds the resulting full-U-length beta vector through the FROZEN Step-C downstream
predictor to get predicted age. Writes one CSV under
outputs/reconstruction_downstream/gse40279_age/seed_17/.

Reuses (does not reimplement): MaskedMethylomeReconstructor, MaskingDataset/
MaskFractionBatchSampler, patient_disjoint_split, compute_leakage_safe_priors,
train_model/evaluate_loader, HDF5RepresentationStore, reconstruction_metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cpg_repr_benchmark.data.masking import MaskFractionBatchSampler, MaskingDataset
from cpg_repr_benchmark.data.splits import patient_disjoint_split
from cpg_repr_benchmark.evaluation.metrics import reconstruction_metrics
from cpg_repr_benchmark.models.downstream_predictor import DownstreamBetaMLP
from cpg_repr_benchmark.models.model import MaskedMethylomeReconstructor
from cpg_repr_benchmark.representations.control_providers import ConstantStore, RandomStableStore
from cpg_repr_benchmark.representations.hdf5_store import HDF5RepresentationStore
from cpg_repr_benchmark.training.downstream_predictor import downstream_regression_metrics, fill_locus_mean
from cpg_repr_benchmark.training.engine import train_model
from cpg_repr_benchmark.training.priors import beta_to_logit, compute_leakage_safe_priors

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 17
OBSERVED_COUNTS = [4096, 1024, 256]
LEARNED_ARMS = ["functional", "ntv3_pre", "random_stable", "constant"]
ALL_ARMS = LEARNED_ARMS + ["mean_prior", "full_oracle"]
PANEL_SIZE = 4608

DATASET_H5 = REPO_ROOT / "data/processed/GSE40279/methylation.h5"
FUNCTIONAL_STORE = REPO_ROOT / "data/cache/representations/materialized/GSE40279/shared_chr1/functional_legacy.h5"
NTV3_STORE = REPO_ROOT / "data/cache/representations/materialized/GSE40279/shared_chr1/ntv3_pre.h5"
UNIVERSE_NPZ = REPO_ROOT / "data/protocols/reduced_observation/gse40279_age_universe_seed17.npz"


def _mask_fraction_for_observed(observed: int, panel: int = PANEL_SIZE) -> float:
    target = panel - observed
    return target / panel


def load_beta_u(methylation_h5: Path, universe_cpg_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        full_meth_ids = meth_ids
    return beta, matrix_columns, sample_names, full_meth_ids


def build_store(name: str, full_matrix_cpg_ids: np.ndarray, u_matrix_columns: np.ndarray) -> object:
    if name == "functional":
        return HDF5RepresentationStore(FUNCTIONAL_STORE, full_matrix_cpg_ids, id_key="cpg_idx", embedding_key="embedding")
    if name == "ntv3_pre":
        return HDF5RepresentationStore(NTV3_STORE, full_matrix_cpg_ids, id_key="cpg_idx", embedding_key="embedding")
    if name == "random_stable":
        return RandomStableStore(u_matrix_columns, full_matrix_cpg_ids[u_matrix_columns], dim=32, seed=SEED)
    if name == "constant":
        return ConstantStore(u_matrix_columns, dim=32)
    raise ValueError(name)


def train_reconstruction_model(
    arm: str,
    store,
    matrix_path: Path,
    prior_full: np.ndarray,
    u_matrix_columns: np.ndarray,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    device: torch.device,
    output_dir: Path,
    epochs: int,
) -> MaskedMethylomeReconstructor:
    mask_fractions = sorted({_mask_fraction_for_observed(c) for c in OBSERVED_COUNTS})
    train_ds = MaskingDataset(
        methylation_h5=matrix_path,
        representation_store=store,
        prior_logit_full=prior_full,
        sample_indices=train_rows,
        context_columns=u_matrix_columns,
        target_columns=u_matrix_columns,
        panel_size=PANEL_SIZE,
        mask_fractions=mask_fractions,
        seed=SEED,
    )
    val_ds = MaskingDataset(
        methylation_h5=matrix_path,
        representation_store=store,
        prior_logit_full=prior_full,
        sample_indices=val_rows,
        context_columns=u_matrix_columns,
        target_columns=u_matrix_columns,
        panel_size=PANEL_SIZE,
        mask_fractions=[mask_fractions[len(mask_fractions) // 2]],
        seed=SEED,
    )
    train_loader = DataLoader(train_ds, batch_sampler=MaskFractionBatchSampler(train_ds, 8, shuffle=True), num_workers=0)
    val_loader = DataLoader(val_ds, batch_sampler=MaskFractionBatchSampler(val_ds, 8, shuffle=False), num_workers=0)

    model = MaskedMethylomeReconstructor(raw_locus_dim=store.dim)
    train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=epochs,
        learning_rate=1e-4,
        weight_decay=1e-4,
        mixed_precision=True,
        output_dir=output_dir,
    )
    checkpoint = torch.load(output_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def reconstruct_missing(
    model: MaskedMethylomeReconstructor,
    store,
    beta_filled: np.ndarray,
    prior_u: np.ndarray,
    observed_pos: np.ndarray,
    missing_pos: np.ndarray,
    observed_matrix_columns: np.ndarray,
    missing_matrix_columns: np.ndarray,
    rows: np.ndarray,
    device: torch.device,
    beta_epsilon: float = 1e-4,
    target_chunk: int = 8192,
    patient_batch: int = 32,
) -> np.ndarray:
    """Returns reconstructed beta at missing_pos for every row in ``rows`` (n_rows x n_missing)."""
    observed_locus = torch.from_numpy(store.get_by_matrix_columns(observed_matrix_columns)).to(device)
    prior_obs = torch.from_numpy(prior_u[observed_pos]).to(device)
    out = np.empty((len(rows), len(missing_pos)), dtype=np.float32)
    for b0 in range(0, len(rows), patient_batch):
        batch_rows = rows[b0 : b0 + patient_batch]
        logit_obs = torch.from_numpy(beta_to_logit(beta_filled[batch_rows][:, observed_pos], beta_epsilon)).to(device)
        residual_obs = logit_obs - prior_obs.unsqueeze(0)
        obs_locus_b = observed_locus.unsqueeze(0).expand(len(batch_rows), -1, -1)
        valid = torch.ones(len(batch_rows), len(observed_pos), dtype=torch.bool, device=device)
        for t0 in range(0, len(missing_pos), target_chunk):
            t1 = min(t0 + target_chunk, len(missing_pos))
            target_locus = torch.from_numpy(store.get_by_matrix_columns(missing_matrix_columns[t0:t1])).to(device)
            target_locus_b = target_locus.unsqueeze(0).expand(len(batch_rows), -1, -1)
            target_prior = torch.from_numpy(prior_u[missing_pos[t0:t1]]).to(device).unsqueeze(0).expand(len(batch_rows), -1)
            pred, _ = model(obs_locus_b, residual_obs, valid, target_locus_b, target_prior)
            out[b0 : b0 + len(batch_rows), t0:t1] = pred.cpu().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dataset", default="gse40279_age")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with np.load(UNIVERSE_NPZ) as data:
        universe_cpg_idx = np.asarray(data["universe_cpg_idx"], dtype=np.int64)
        subset_ids = {c: np.asarray(data[f"observed_{c}_cpg_idx"], dtype=np.int64) for c in OBSERVED_COUNTS}

    beta_u, u_matrix_columns, sample_names, full_meth_ids = load_beta_u(DATASET_H5, universe_cpg_idx)
    u_cpg_ids = full_meth_ids[u_matrix_columns]  # aligned to beta_u / u_matrix_columns column order
    n_nan = int(np.sum(~np.isfinite(beta_u)))

    phenotypes = pd.read_parquet(REPO_ROOT / "data/processed/GSE40279/phenotypes.parquet")
    phenotypes = phenotypes.set_index("sample_name").loc[sample_names].reset_index()
    age = phenotypes["age"].to_numpy(dtype=np.float64)

    splits = patient_disjoint_split(list(sample_names), SEED, (0.80, 0.10, 0.10))
    train_rows, val_rows, test_rows = splits["train"], splits["validation"], splits["test"]

    beta_filled, _ = fill_locus_mean(beta_u, train_rows)

    prior_full, _, _ = compute_leakage_safe_priors(
        DATASET_H5, train_rows, u_matrix_columns, len(full_meth_ids), epsilon=1e-4
    )
    prior_u = prior_full[u_matrix_columns]  # aligned to beta_u columns

    # id -> position (within beta_u / u_matrix_columns ordering)
    id_order = np.argsort(u_cpg_ids)
    sorted_ids = u_cpg_ids[id_order]

    def id_to_pos(ids: np.ndarray) -> np.ndarray:
        loc = np.searchsorted(sorted_ids, ids)
        assert np.array_equal(sorted_ids[loc], ids)
        return np.sort(id_order[loc])

    # Frozen Step-C downstream predictor
    dp_dir = REPO_ROOT / "outputs/reconstruction_downstream/gse40279_age/seed_17/downstream_predictor"
    dp_ckpt = torch.load(dp_dir / "downstream_predictor.pt", map_location="cpu", weights_only=False)
    dp_model = DownstreamBetaMLP(n_cpg=dp_ckpt["n_cpg"], hidden_dim=dp_ckpt["hidden_dim"], dropout=dp_ckpt["dropout"])
    dp_model.load_state_dict(dp_ckpt["model"])
    dp_model.to(device).eval()
    age_mean, age_std = dp_ckpt["age_mean"], dp_ckpt["age_std"]

    @torch.no_grad()
    def predict_age(beta_matrix: np.ndarray) -> np.ndarray:
        preds = []
        for s in range(0, len(beta_matrix), 64):
            xb = torch.from_numpy(beta_matrix[s : s + 64]).float().to(device)
            preds.append(dp_model(xb).cpu().numpy())
        return (np.concatenate(preds) * age_std) + age_mean

    run_root = REPO_ROOT / "outputs/reconstruction_downstream" / args.dataset / f"seed_{args.seed}"
    run_root.mkdir(parents=True, exist_ok=True)

    full_matrix_cpg_ids = full_meth_ids
    rows_out = []

    # --- full_oracle: no reconstruction, run once ---
    oracle_beta = beta_filled[test_rows]
    oracle_pred_age = predict_age(oracle_beta)
    oracle_downstream = downstream_regression_metrics(oracle_pred_age, age[test_rows])
    for c in OBSERVED_COUNTS:
        rows_out.append({
            "task": "age", "representation": "full_oracle", "observed_cpgs": c,
            "observed_fraction": c / len(universe_cpg_idx), "seed": args.seed,
            "reconstruction_mse": float("nan"), "reconstruction_mae": float("nan"),
            "reconstruction_mas": float("nan"), "skill_vs_prior": float("nan"),
            "downstream_mae": oracle_downstream["mae"], "downstream_pearson": oracle_downstream["pearson"],
            "downstream_r2": oracle_downstream["r2"],
        })

    for c in OBSERVED_COUNTS:
        observed_pos = id_to_pos(np.sort(subset_ids[c]))
        missing_pos = np.setdiff1d(np.arange(len(u_cpg_ids)), observed_pos)
        observed_matrix_columns = u_matrix_columns[observed_pos]
        missing_matrix_columns = u_matrix_columns[missing_pos]

        # --- mean_prior ---
        mean_prior_missing = np.broadcast_to(
            1.0 / (1.0 + np.exp(-prior_u[missing_pos])), (len(test_rows), len(missing_pos))
        ).astype(np.float32)
        real_missing = beta_filled[test_rows][:, missing_pos]
        m = reconstruction_metrics(mean_prior_missing, real_missing, mean_prior_missing)
        full_beta = beta_filled[test_rows].copy()
        full_beta[:, missing_pos] = mean_prior_missing
        ds = downstream_regression_metrics(predict_age(full_beta), age[test_rows])
        rows_out.append({
            "task": "age", "representation": "mean_prior", "observed_cpgs": c,
            "observed_fraction": c / len(universe_cpg_idx), "seed": args.seed,
            "reconstruction_mse": m["mse"], "reconstruction_mae": m["mae"],
            "reconstruction_mas": m["patient_pearson"], "skill_vs_prior": m["skill_vs_prior"],
            "downstream_mae": ds["mae"], "downstream_pearson": ds["pearson"], "downstream_r2": ds["r2"],
        })

        for arm in LEARNED_ARMS:
            arm_dir = run_root / "reconstruction_models" / arm
            store = build_store(arm, full_matrix_cpg_ids, u_matrix_columns)
            ckpt_path = arm_dir / "checkpoints" / "best.pt"
            if not ckpt_path.exists():
                model = train_reconstruction_model(
                    arm, store, DATASET_H5, prior_full, u_matrix_columns,
                    train_rows, val_rows, device, arm_dir, args.epochs,
                )
            else:
                model = MaskedMethylomeReconstructor(raw_locus_dim=store.dim)
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model.load_state_dict(state["model"])
                model.to(device).eval()

            recon_missing = reconstruct_missing(
                model, store, beta_filled, prior_u, observed_pos, missing_pos,
                observed_matrix_columns, missing_matrix_columns, test_rows, device,
            )
            prior_missing = np.broadcast_to(
                1.0 / (1.0 + np.exp(-prior_u[missing_pos])), recon_missing.shape
            ).astype(np.float32)
            m = reconstruction_metrics(recon_missing, real_missing, prior_missing)
            full_beta = beta_filled[test_rows].copy()
            full_beta[:, missing_pos] = recon_missing
            ds = downstream_regression_metrics(predict_age(full_beta), age[test_rows])
            rows_out.append({
                "task": "age", "representation": arm, "observed_cpgs": c,
                "observed_fraction": c / len(universe_cpg_idx), "seed": args.seed,
                "reconstruction_mse": m["mse"], "reconstruction_mae": m["mae"],
                "reconstruction_mas": m["patient_pearson"], "skill_vs_prior": m["skill_vs_prior"],
                "downstream_mae": ds["mae"], "downstream_pearson": ds["pearson"], "downstream_r2": ds["r2"],
            })
            store.close()

    df = pd.DataFrame(rows_out)
    csv_path = run_root / "reconstruction_downstream_results.csv"
    df.to_csv(csv_path, index=False)
    (run_root / "run_manifest.json").write_text(json.dumps({
        "seed": args.seed, "n_nan_in_universe_beta": n_nan, "universe_size": len(universe_cpg_idx),
        "observed_counts": OBSERVED_COUNTS, "panel_size": PANEL_SIZE, "epochs": args.epochs,
        "downstream_predictor_val_mae": dp_ckpt.get("best_val_mae"),
    }, indent=2))
    print(df.to_string(index=False))
    print(f"CSV={csv_path}")


if __name__ == "__main__":
    main()
