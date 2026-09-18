from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cpg_repr_benchmark.models.downstream_predictor import DownstreamBetaMLP


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def downstream_regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(pred - true))),
        "pearson": _pearson(pred, true),
        "r2": _r2(pred, true),
        "n": len(true),
    }


def fill_locus_mean(beta: np.ndarray, train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaNs with the train-patient locus mean; return (filled_beta, locus_means)."""
    locus_mean = np.nanmean(beta[train_rows], axis=0)
    locus_mean = np.nan_to_num(locus_mean, nan=0.5)
    filled = beta.copy()
    nan_mask = ~np.isfinite(filled)
    rows, cols = np.nonzero(nan_mask)
    if len(rows):
        filled[rows, cols] = locus_mean[cols]
    return filled.astype(np.float32), locus_mean.astype(np.float32)


def train_downstream_predictor(
    *,
    beta_u: np.ndarray,
    age: np.ndarray,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    device: torch.device,
    epochs: int = 60,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    hidden_dim: int = 512,
    dropout: float = 0.1,
    seed: int = 17,
    output_dir: Path | None = None,
) -> dict:
    """Train the frozen oracle downstream predictor on TRAIN patients, select via VAL.

    ``beta_u`` is the real complete methylome restricted to universe U (n_samples x |U|),
    with NaNs already filled by train-locus means (see ``fill_locus_mean``). Age is
    standardized using train-only statistics; predictions are un-standardized at eval time.
    """
    torch.manual_seed(seed)
    n_cpg = beta_u.shape[1]
    age_mean = float(age[train_rows].mean())
    age_std = float(age[train_rows].std()) or 1.0

    model = DownstreamBetaMLP(n_cpg=n_cpg, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    x_train = torch.from_numpy(beta_u[train_rows]).float()
    y_train = torch.from_numpy((age[train_rows] - age_mean) / age_std).float()
    x_val = torch.from_numpy(beta_u[val_rows]).float().to(device)
    y_val = age[val_rows].astype(np.float64)

    rng = np.random.default_rng(seed)
    n = len(train_rows)
    best_val_mae = float("inf")
    best_state = None
    history = []
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n)
        running = 0.0
        steps = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            steps += 1
        model.eval()
        with torch.no_grad():
            val_pred = (model(x_val).cpu().numpy() * age_std) + age_mean
        val_metrics = downstream_regression_metrics(val_pred, y_val)
        history.append({"epoch": epoch, "train_mse": running / max(steps, 1), **{f"val_{k}": v for k, v in val_metrics.items()}})
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    result = {
        "model": model,
        "age_mean": age_mean,
        "age_std": age_std,
        "best_val_mae": best_val_mae,
        "history": history,
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": best_state, "age_mean": age_mean, "age_std": age_std, "n_cpg": n_cpg,
             "hidden_dim": hidden_dim, "dropout": dropout},
            output_dir / "downstream_predictor.pt",
        )
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    return result


@torch.no_grad()
def predict_age(model: DownstreamBetaMLP, beta_u: np.ndarray, age_mean: float, age_std: float, device: torch.device, batch_size: int = 64) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(beta_u), batch_size):
        xb = torch.from_numpy(beta_u[start : start + batch_size]).float().to(device)
        preds.append(model(xb).cpu().numpy())
    return (np.concatenate(preds) * age_std) + age_mean
