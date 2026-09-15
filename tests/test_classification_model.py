import torch

from cpg_repr_benchmark.models.classification import MethylomeTaskModel


def test_multiclass_model_shapes():
    model = MethylomeTaskModel(
        raw_locus_dim=32,
        task_type="multiclass",
        output_dim=4,
        locus_latent_dim=16,
        token_dim=16,
        patient_dim=12,
        hidden_dim=24,
        dropout=0.0,
    )
    logits, patient = model(
        torch.randn(3, 20, 32),
        torch.randn(3, 20),
        torch.ones(3, 20, dtype=torch.bool),
    )
    assert logits.shape == (3, 4)
    assert patient.shape == (3, 12)
