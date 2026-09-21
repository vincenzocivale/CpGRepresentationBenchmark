from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from cpg_repr_benchmark.probing.metrics import classification_metrics, regression_metrics

_PRIMARY_METRIC = {"classification": "auc", "regression": "r2"}


class _ProbeHead(nn.Module):
    def __init__(self, patient_dim: int, *, task_type: str, pooling: str, num_latents: int = 1, dropout: float = 0.2):
        super().__init__()
        self.pooling = pooling
        in_dim = patient_dim * num_latents if pooling == "flatten" else patient_dim
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout), nn.Linear(in_dim, 1))
        self.task_type = task_type

    def pool(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim == 2:
            return raw
        return raw.mean(dim=1) if self.pooling == "mean" else raw.reshape(raw.shape[0], -1)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(raw)
        logit = self.net(pooled).squeeze(-1)
        return torch.sigmoid(logit) if self.task_type == "classification" else logit


def _freeze_low_level_layers(model: nn.Module) -> int:
    """Freeze the shared locus-feature layers (`locus_adapter`, `tokenizer`, present in both the
    DeepSets and Perceiver reconstructors) and leave the patient-aggregation layers
    (`patient_encoder` / latents + attention blocks) trainable. Fine-tuning downstream cohorts
    are tiny (hundreds of patients) relative to the encoder's parameter count, so unlocking the
    entire stack overfits fast; freezing the low-level feature extractor is the standard transfer-
    learning fix. Returns the number of frozen parameters.
    """
    frozen = 0
    for name in ("locus_adapter", "tokenizer"):
        module = getattr(model, name, None)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = False
            frozen += parameter.numel()
    return frozen


def _lr_factor(step: int, *, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


@dataclass
class FinetuneResult:
    metrics: dict[str, dict[str, float | int]]
    state_dict: dict
    best_epoch: int
    history: list[dict[str, float]]


def _run_eval(model, head, loader_factory, device, task_type) -> dict[str, dict[str, float | int]]:
    model.eval()
    head.eval()
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    split_pred: dict[str, list[np.ndarray]] = {}
    split_true: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for batch in loader_factory():
            raw = model.encode_patient(
                batch["observed_locus"].to(device),
                batch["observed_residual"].to(device),
                batch["observed_valid"].to(device),
            )
            pred = head(raw).cpu().numpy()
            true = batch["target"].numpy()
            split = batch.get("split", ["all"] * len(true))
            for s, p, t in zip(split, pred, true):
                split_pred.setdefault(s, []).append(p)
                split_true.setdefault(s, []).append(t)
            all_pred.append(pred)
            all_true.append(true)

    metric_fn = classification_metrics if task_type == "classification" else regression_metrics
    metrics = {s: metric_fn(np.asarray(split_pred[s]), np.asarray(split_true[s])) for s in split_pred}
    if not metrics:
        metrics = {"all": metric_fn(np.concatenate(all_pred), np.concatenate(all_true))}
    return metrics


def fine_tune_encoder(
    model: nn.Module,
    *,
    loader_factory,
    eval_loader_factory=None,
    validation_loader_factory=None,
    task_type: str,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-4,
    head_lr: float = 1e-3,
    weight_decay: float = 0.01,
    pooling: str = "mean",
    freeze_low_level: bool = True,
    head_dropout: float = 0.2,
    warmup_epochs: int = 2,
    min_lr_ratio: float = 0.1,
    early_stopping_patience: int | None = 5,
) -> FinetuneResult:
    """End-to-end fine-tuning of `model.encode_patient` plus a thin linear head.

    `loader_factory()` must return an iterable yielding dict-batches identical in shape to
    `embedding.extract.extract_patient_embeddings`'s expected input, plus a float `target`
    tensor per batch, and must only cover TRAIN patients — held-out/validation/test patients
    must never appear here, or their labels leak into the gradient. Unlike
    `training.sparse_reconstruction.train_sparse_model`, this loop trains only the encoder + a
    1-layer head on the downstream label, never the reconstruction decoder.

    Regularization (added after the first pass overfit badly on ~500-patient cohorts: train
    AUC ~0.95 vs. test AUC ~0.61): `weight_decay` on AdamW, `head_dropout` in the probe head,
    `freeze_low_level` to keep the shared locus-feature layers frozen (see
    `_freeze_low_level_layers`), a warmup+cosine LR schedule, and early stopping on
    `validation_loader_factory` (falls back to no early stopping / last-epoch checkpoint if
    omitted, matching the previous behavior).

    `eval_loader_factory`, if given, is used only for the final metrics pass and may cover every
    patient (train/validation/test), each batch carrying a `split` label per sample so metrics
    can be reported per split. Defaults to `loader_factory` (train-only) when omitted.
    """
    eval_loader_factory = eval_loader_factory or loader_factory
    model.to(device)
    if freeze_low_level:
        n_frozen = _freeze_low_level_layers(model)
    else:
        n_frozen = 0

    sample_batch = next(iter(loader_factory()))
    with torch.no_grad():
        raw = model.encode_patient(
            sample_batch["observed_locus"].to(device),
            sample_batch["observed_residual"].to(device),
            sample_batch["observed_valid"].to(device),
        )
    patient_dim = raw.shape[-1]
    num_latents = raw.shape[1] if raw.ndim == 3 else 1
    head = _ProbeHead(
        patient_dim, task_type=task_type, pooling=pooling, num_latents=num_latents, dropout=head_dropout
    ).to(device)

    loss_fn = nn.BCELoss() if task_type == "classification" else nn.MSELoss()
    trainable_encoder_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable_encoder_params, "lr": lr, "weight_decay": weight_decay},
            {"params": head.parameters(), "lr": head_lr, "weight_decay": weight_decay},
        ]
    )

    train_batches = list(loader_factory())
    steps_per_epoch = max(1, len(train_batches))
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_factor(step, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=min_lr_ratio)
    )

    primary_metric = _PRIMARY_METRIC[task_type]
    best_score = -float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        model.train()
        head.train()
        for batch in train_batches:
            optimizer.zero_grad()
            raw = model.encode_patient(
                batch["observed_locus"].to(device),
                batch["observed_residual"].to(device),
                batch["observed_valid"].to(device),
            )
            pred = head(raw)
            target = batch["target"].to(device).float()
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

        if validation_loader_factory is not None:
            val_metrics = _run_eval(model, head, validation_loader_factory, device, task_type)
            val_score = next(iter(val_metrics.values())).get(primary_metric, float("nan"))
            val_score = -float("inf") if np.isnan(val_score) else val_score
            history.append({"epoch": epoch, "val_score": float(val_score)})
            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch
                best_state = {"encoder": copy.deepcopy(model.state_dict()), "head": copy.deepcopy(head.state_dict())}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state["encoder"])
        head.load_state_dict(best_state["head"])
    else:
        best_epoch = epochs - 1

    metrics = _run_eval(model, head, eval_loader_factory, device, task_type)

    return FinetuneResult(
        metrics=metrics,
        state_dict={"encoder": model.state_dict(), "head": head.state_dict(), "frozen_parameters": n_frozen},
        best_epoch=best_epoch,
        history=history,
    )
