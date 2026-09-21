from __future__ import annotations

import torch

from cpg_repr_benchmark.models.model import build_reconstructor
from cpg_repr_benchmark.models.perceiver import PerceiverMaskedMethylomeReconstructor


def _model() -> PerceiverMaskedMethylomeReconstructor:
    torch.manual_seed(17)
    model = PerceiverMaskedMethylomeReconstructor(
        raw_locus_dim=12,
        model_dim=32,
        num_latents=8,
        num_latent_blocks=2,
        num_heads=4,
        ff_mult=2,
        dropout=0.0,
    )
    model.eval()
    return model


def _batch():
    torch.manual_seed(23)
    observed_locus = torch.randn(2, 11, 12)
    observed_residual = torch.randn(2, 11)
    observed_valid = torch.ones(2, 11, dtype=torch.bool)
    target_locus = torch.randn(2, 13, 12)
    target_prior = torch.randn(2, 13)
    return observed_locus, observed_residual, observed_valid, target_locus, target_prior


def test_perceiver_starts_at_prior_baseline():
    model = _model()
    observed_locus, observed_residual, observed_valid, target_locus, target_prior = _batch()
    prediction, latents = model(
        observed_locus,
        observed_residual,
        observed_valid,
        target_locus,
        target_prior,
    )
    assert latents.shape == (2, 8, 32)
    assert torch.allclose(prediction, torch.sigmoid(target_prior), atol=1e-7, rtol=0.0)


def test_perceiver_is_invariant_to_observed_set_order():
    model = _model()
    observed_locus, observed_residual, observed_valid, target_locus, target_prior = _batch()
    with torch.no_grad():
        baseline = model.encode_patient(observed_locus, observed_residual, observed_valid)
        permutation = torch.tensor([7, 1, 10, 3, 0, 9, 5, 2, 8, 6, 4])
        permuted = model.encode_patient(
            observed_locus[:, permutation],
            observed_residual[:, permutation],
            observed_valid[:, permutation],
        )
    assert torch.allclose(baseline, permuted, atol=1e-6, rtol=1e-5)


def test_perceiver_target_decoding_is_chunk_invariant():
    model = _model()
    observed_locus, observed_residual, observed_valid, target_locus, target_prior = _batch()
    with torch.no_grad():
        torch.nn.init.normal_(model.delta_head[-1].weight, std=0.02)
        torch.nn.init.normal_(model.delta_head[-1].bias, std=0.02)
        latents = model.encode_patient(observed_locus, observed_residual, observed_valid)
        all_targets = model.decode_targets(latents, target_locus, target_prior)
        chunks = []
        for start in range(0, target_locus.shape[1], 4):
            chunks.append(
                model.decode_targets(
                    latents,
                    target_locus[:, start : start + 4],
                    target_prior[:, start : start + 4],
                )
            )
        chunked = torch.cat(chunks, dim=1)
    assert torch.allclose(all_targets, chunked, atol=1e-6, rtol=1e-5)


def test_factory_builds_perceiver_from_config():
    model = build_reconstructor(
        12,
        {
            "architecture": "perceiver_io",
            "model_dim": 32,
            "num_latents": 8,
            "num_latent_blocks": 2,
            "num_heads": 4,
            "ff_mult": 2,
            "dropout": 0.0,
        },
    )
    assert isinstance(model, PerceiverMaskedMethylomeReconstructor)
    assert model.num_latents == 8
    assert model.model_dim == 32
