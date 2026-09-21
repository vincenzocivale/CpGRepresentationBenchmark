import numpy as np
import torch

from cpg_repr_benchmark.models.model import MaskedMethylomeReconstructor
from cpg_repr_benchmark.probing.finetune import fine_tune_encoder


def _make_loader(rng, n_patients, n_observed, locus_dim, *, split_label=None):
    locus = rng.normal(size=(n_patients, n_observed, locus_dim)).astype(np.float32)
    weight = rng.normal(size=locus_dim).astype(np.float32)
    signal = (locus.mean(axis=1) @ weight)
    target = (signal > np.median(signal)).astype(np.float32)

    def factory():
        batch = {
            "observed_locus": torch.from_numpy(locus),
            "observed_residual": torch.from_numpy(rng.normal(size=(n_patients, n_observed)).astype(np.float32)),
            "observed_valid": torch.ones(n_patients, n_observed, dtype=torch.bool),
            "target": torch.from_numpy(target),
        }
        if split_label is not None:
            batch["split"] = [split_label] * n_patients
        return [batch]

    return factory


def test_fine_tune_encoder_separates_train_and_eval_loaders():
    rng = np.random.default_rng(17)
    model = MaskedMethylomeReconstructor(raw_locus_dim=6, locus_latent_dim=12, token_dim=12, patient_dim=8, hidden_dim=16)
    train_loader = _make_loader(rng, 20, 5, 6, split_label="train")
    test_loader = _make_loader(rng, 8, 5, 6, split_label="test")

    def eval_loader():
        return train_loader() + test_loader()

    result = fine_tune_encoder(
        model,
        loader_factory=train_loader,
        eval_loader_factory=eval_loader,
        task_type="classification",
        device=torch.device("cpu"),
        epochs=2,
    )
    assert set(result.metrics.keys()) == {"train", "test"}
    assert result.metrics["train"]["n"] == 20
    assert result.metrics["test"]["n"] == 8
    assert "encoder" in result.state_dict and "head" in result.state_dict


def test_fine_tune_encoder_defaults_eval_loader_to_train_loader():
    rng = np.random.default_rng(3)
    model = MaskedMethylomeReconstructor(raw_locus_dim=4, locus_latent_dim=8, token_dim=8, patient_dim=6, hidden_dim=12)
    loader = _make_loader(rng, 10, 4, 4)
    result = fine_tune_encoder(
        model, loader_factory=loader, task_type="regression", device=torch.device("cpu"), epochs=1
    )
    assert "all" in result.metrics
    assert result.metrics["all"]["n"] == 10


def test_fine_tune_encoder_freezes_low_level_layers_by_default():
    rng = np.random.default_rng(5)
    model = MaskedMethylomeReconstructor(raw_locus_dim=6, locus_latent_dim=12, token_dim=12, patient_dim=8, hidden_dim=16)
    loader = _make_loader(rng, 12, 5, 6)
    fine_tune_encoder(model, loader_factory=loader, task_type="regression", device=torch.device("cpu"), epochs=1)
    assert all(not p.requires_grad for p in model.locus_adapter.parameters())
    assert all(not p.requires_grad for p in model.tokenizer.parameters())
    assert any(p.requires_grad for p in model.patient_encoder.parameters())


def test_fine_tune_encoder_early_stopping_selects_best_validation_epoch():
    rng = np.random.default_rng(11)
    model = MaskedMethylomeReconstructor(raw_locus_dim=6, locus_latent_dim=12, token_dim=12, patient_dim=8, hidden_dim=16)
    train_loader = _make_loader(rng, 20, 5, 6)
    val_loader = _make_loader(rng, 8, 5, 6)
    result = fine_tune_encoder(
        model,
        loader_factory=train_loader,
        validation_loader_factory=val_loader,
        task_type="classification",
        device=torch.device("cpu"),
        epochs=6,
        early_stopping_patience=2,
    )
    assert result.best_epoch >= 0
    assert len(result.history) <= 6
    assert result.history  # at least one epoch logged
