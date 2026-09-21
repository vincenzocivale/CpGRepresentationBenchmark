from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from cpg_repr_benchmark.evaluation.metrics import reconstruction_metrics


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate_loader(model: torch.nn.Module, loader, device: torch.device) -> tuple[dict, dict[str, np.ndarray]]:
    model.eval()
    predictions, targets, priors, columns, samples = [], [], [], [], []
    for batch in loader:
        batch = _move(batch, device)
        pred, _ = model(
            batch["observed_locus"],
            batch["observed_residual"],
            batch["observed_valid"],
            batch["target_locus"],
            batch["target_prior_logit"],
        )
        predictions.append(pred.cpu().numpy())
        targets.append(batch["target_beta"].cpu().numpy())
        priors.append(torch.sigmoid(batch["target_prior_logit"]).cpu().numpy())
        columns.append(batch["target_matrix_column"].cpu().numpy())
        samples.append(batch["sample_index"].cpu().numpy())
    pred = np.concatenate(predictions, axis=0)
    true = np.concatenate(targets, axis=0)
    prior = np.concatenate(priors, axis=0)
    outputs = {
        "prediction": pred,
        "target": true,
        "prior_prediction": prior,
        "target_matrix_column": np.concatenate(columns, axis=0),
        "sample_index": np.concatenate(samples, axis=0),
    }
    return reconstruction_metrics(
        pred, true, prior, target_matrix_column=outputs["target_matrix_column"]
    ), outputs


def train_model(
    model: torch.nn.Module,
    train_loader,
    validation_loader,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    mixed_precision: bool,
    output_dir: Path,
) -> list[dict]:
    output_dir = Path(output_dir)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and device.type == "cuda")
    model.to(device)
    history: list[dict] = []
    best = float("inf")
    for epoch in range(epochs):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        n_steps = 0
        for batch in train_loader:
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if mixed_precision and device.type == "cuda"
                else nullcontext()
            )
            with context:
                prediction, _ = model(
                    batch["observed_locus"],
                    batch["observed_residual"],
                    batch["observed_valid"],
                    batch["target_locus"],
                    batch["target_prior_logit"],
                )
                loss = torch.mean((prediction - batch["target_beta"]) ** 2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            n_steps += 1
        validation, _ = evaluate_loader(model, validation_loader, device)
        record = {"epoch": epoch, "train_mse": running / max(n_steps, 1), **{f"validation_{k}": v for k, v in validation.items()}}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": record,
        }
        torch.save(state, checkpoints / "last.pt")
        torch.save(state, checkpoints / f"epoch_{epoch:04d}.pt")
        if float(validation["mse"]) < best:
            best = float(validation["mse"])
            torch.save(state, checkpoints / "best.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    return history
