from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cpg_repr_benchmark.evaluation.supervised_metrics import supervised_metrics


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _loss(task_type: str, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if task_type == "regression":
        return nn.functional.mse_loss(output, target.float())
    if task_type == "binary":
        return nn.functional.binary_cross_entropy_with_logits(output, target.float())
    if task_type == "multiclass":
        return nn.functional.cross_entropy(output, target.long())
    raise ValueError(task_type)


def _probability(task_type: str, output: torch.Tensor) -> torch.Tensor:
    if task_type == "binary":
        return torch.sigmoid(output)
    if task_type == "multiclass":
        return torch.softmax(output, dim=-1)
    return output


@torch.no_grad()
def evaluate_supervised(
    model,
    loader,
    *,
    task_type: str,
    device: torch.device,
) -> tuple[dict, dict[str, np.ndarray]]:
    model.eval()
    predictions, targets, samples = [], [], []
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = _move(batch, device)
        output, _ = model(
            batch["observed_locus"],
            batch["observed_residual"],
            batch["observed_valid"],
        )
        total_loss += float(_loss(task_type, output, batch["target"]).detach())
        n_batches += 1
        predictions.append(_probability(task_type, output).cpu().numpy())
        targets.append(batch["target"].cpu().numpy())
        samples.append(batch["sample_index"].cpu().numpy())
    pred = np.concatenate(predictions, axis=0)
    true = np.concatenate(targets, axis=0)
    metrics = {"loss": total_loss / max(n_batches, 1), **supervised_metrics(task_type, pred, true)}
    return metrics, {
        "prediction": pred,
        "target": true,
        "sample_index": np.concatenate(samples, axis=0),
    }


def train_supervised(
    model,
    train_loader,
    validation_loader,
    *,
    task_type: str,
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
                output, _ = model(
            batch["observed_locus"],
            batch["observed_residual"],
            batch["observed_valid"],
        )
                loss = _loss(task_type, output, batch["target"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            n_steps += 1
        validation, _ = evaluate_supervised(
            model,
            validation_loader,
            task_type=task_type,
            device=device,
        )
        record = {
            "epoch": epoch,
            "train_loss": running / max(n_steps, 1),
            **{f"validation_{k}": v for k, v in validation.items()},
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": record,
        }
        torch.save(state, checkpoints / "last.pt")
        if float(validation["loss"]) < best:
            best = float(validation["loss"])
            torch.save(state, checkpoints / "best.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    return history
