from pathlib import Path

import torch

from cpg_repr_benchmark.embedding.extract import extract_patient_embeddings
from cpg_repr_benchmark.embedding.store import load_patient_embeddings, save_patient_embeddings
from cpg_repr_benchmark.models.model import MaskedMethylomeReconstructor
from cpg_repr_benchmark.models.perceiver import PerceiverMaskedMethylomeReconstructor


def _toy_batches(n_patients: int, n_observed: int, locus_dim: int):
    torch.manual_seed(0)
    for start in range(0, n_patients, 4):
        batch = min(4, n_patients - start)
        yield {
            "observed_locus": torch.randn(batch, n_observed, locus_dim),
            "observed_residual": torch.randn(batch, n_observed),
            "observed_valid": torch.ones(batch, n_observed, dtype=torch.bool),
            "patient_id": [f"p{start + i}" for i in range(batch)],
        }


def test_extract_patient_embeddings_deepsets():
    model = MaskedMethylomeReconstructor(raw_locus_dim=8, locus_latent_dim=16, token_dim=16, patient_dim=12, hidden_dim=24)
    patient_ids, embedding = extract_patient_embeddings(
        model, list(_toy_batches(10, 5, 8)), device=torch.device("cpu")
    )
    assert len(patient_ids) == 10
    assert embedding.shape == (10, 12)


def test_extract_patient_embeddings_perceiver_pools_latents():
    model = PerceiverMaskedMethylomeReconstructor(raw_locus_dim=8, model_dim=16, num_latents=4, num_latent_blocks=1, num_heads=4)
    patient_ids, embedding = extract_patient_embeddings(
        model, list(_toy_batches(6, 5, 8)), device=torch.device("cpu"), pooling="mean"
    )
    assert len(patient_ids) == 6
    assert embedding.shape == (6, 16)


def test_embedding_store_roundtrip(tmp_path: Path):
    save_patient_embeddings(
        tmp_path / "emb.npz",
        patient_ids=["a", "b", "c"],
        embedding=torch.randn(3, 5).numpy(),
        meta={"source_type": "patient_embedding"},
    )
    patient_ids, embedding = load_patient_embeddings(tmp_path / "emb.npz")
    assert patient_ids == ["a", "b", "c"]
    assert embedding.shape == (3, 5)
