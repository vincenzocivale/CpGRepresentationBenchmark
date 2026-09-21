from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def _pool_patient_embedding(raw: torch.Tensor, *, pooling: str = "mean") -> torch.Tensor:
    """Collapse an encoder's patient representation to a single vector per patient.

    `MaskedMethylomeReconstructor.encode_patient` returns `[batch, patient_dim]` already;
    `PerceiverMaskedMethylomeReconstructor.encode_patient` returns a latent bag
    `[batch, num_latents, model_dim]` that must be pooled to compare on equal footing in a
    linear probe.
    """
    if raw.ndim == 2:
        return raw
    if raw.ndim != 3:
        raise ValueError(f"unexpected patient embedding rank: {raw.ndim}")
    if pooling == "mean":
        return raw.mean(dim=1)
    if pooling == "flatten":
        return raw.reshape(raw.shape[0], -1)
    raise ValueError(f"unsupported pooling strategy: {pooling!r}")


@torch.no_grad()
def extract_patient_embeddings(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    pooling: str = "mean",
) -> tuple[list[str], torch.Tensor]:
    """Run a fitted (or frozen pretrained) encoder over a downstream-task loader.

    Each batch must yield a dict with `observed_locus`, `observed_residual`, `observed_valid`,
    and `patient_id` (a list/array of stable string identifiers used to align embeddings with
    downstream-task labels later). Returns `(patient_ids, embeddings)` with
    `embeddings.shape == (n_patients, dim)`, dim depending on the encoder and `pooling`.
    """
    model.eval()
    model.to(device)
    patient_ids: list[str] = []
    chunks: list[torch.Tensor] = []
    for batch in loader:
        observed_locus = batch["observed_locus"].to(device)
        observed_residual = batch["observed_residual"].to(device)
        observed_valid = batch["observed_valid"].to(device)
        raw = model.encode_patient(observed_locus, observed_residual, observed_valid)
        pooled = _pool_patient_embedding(raw, pooling=pooling)
        chunks.append(pooled.detach().to("cpu", dtype=torch.float32))
        batch_ids = batch["patient_id"]
        patient_ids.extend(str(pid) for pid in batch_ids)
    embeddings = torch.cat(chunks, dim=0) if chunks else torch.empty(0, 0)
    return patient_ids, embeddings
