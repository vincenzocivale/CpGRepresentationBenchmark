from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch


def split_proxy_train_validation(
    columns: np.ndarray,
    cpg_ids: np.ndarray,
    *,
    seed: int,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    columns = np.asarray(columns, dtype=np.int64)
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("proxy validation_fraction must be in (0, 0.5)")
    canonical = columns[np.argsort(cpg_ids[columns], kind="stable")]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(canonical))
    n_val = max(1, int(round(len(canonical) * validation_fraction)))
    return np.sort(canonical[perm[n_val:]]), np.sort(canonical[perm[:n_val]])


def train_mean_methylation_proxy(
    model,
    store,
    target_by_column: np.ndarray,
    train_columns: np.ndarray,
    validation_columns: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    mixed_precision: bool,
    seed: int,
    output_dir: Path,
) -> list[dict]:
    output_dir = Path(output_dir)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and device.type == "cuda")
    history: list[dict] = []
    best = float("inf")
    rng = np.random.default_rng(seed)

    @torch.no_grad()
    def evaluate(columns: np.ndarray) -> float:
        model.eval()
        losses = []
        for start in range(0, len(columns), batch_size):
            batch_columns = columns[start : start + batch_size]
            inputs = store.batch(batch_columns, device)
            target = torch.from_numpy(target_by_column[batch_columns].astype(np.float32)).to(device)
            prediction, _ = model(inputs)
            losses.append(float(torch.mean((prediction - target) ** 2).cpu()))
        return float(np.mean(losses))

    for epoch in range(epochs):
        order = train_columns.copy()
        rng.shuffle(order)
        model.train()
        losses = []
        for start in range(0, len(order), batch_size):
            columns = order[start : start + batch_size]
            inputs = store.batch(columns, device)
            target = torch.from_numpy(target_by_column[columns].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if mixed_precision and device.type == "cuda"
                else nullcontext()
            )
            with context:
                prediction, _ = model(inputs)
                loss = torch.mean((prediction - target) ** 2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        validation_mse = evaluate(validation_columns)
        record = {
            "epoch": epoch,
            "train_mse": float(np.mean(losses)),
            "validation_mse": validation_mse,
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
        if validation_mse < best:
            best = validation_mse
            torch.save(state, checkpoints / "best.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    return history


@torch.no_grad()
def export_proxy_embeddings(
    model,
    store,
    matrix_cpg_ids: np.ndarray,
    columns: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    output_h5: Path,
    attrs: dict[str, str | int | float],
) -> None:
    state = torch.load(Path(attrs["checkpoint"]), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    chunks = []
    for start in range(0, len(columns), batch_size):
        batch_columns = columns[start : start + batch_size]
        chunks.append(model.encode(store.batch(batch_columns, device)).float().cpu().numpy())
    embedding = np.concatenate(chunks, axis=0).astype(np.float16)
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_h5.with_suffix(output_h5.suffix + ".tmp")
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset(
            "cpg_idx",
            data=np.asarray(matrix_cpg_ids[columns], dtype=np.int64),
        )
        handle.create_dataset("embedding", data=embedding, compression="gzip", compression_opts=1)
        for key, value in attrs.items():
            handle.attrs[key] = str(value)
    temporary.replace(output_h5)
