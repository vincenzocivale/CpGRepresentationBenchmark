import numpy as np
import torch

from cpg_repr_benchmark.models.model import MaskedMethylomeReconstructor


def test_untrained_model_exactly_reproduces_prior():
    torch.manual_seed(0)
    model = MaskedMethylomeReconstructor(raw_locus_dim=5, locus_latent_dim=8, token_dim=8, patient_dim=8, hidden_dim=16)
    observed = torch.randn(2, 4, 5)
    residual = torch.randn(2, 4)
    valid = torch.ones(2, 4, dtype=torch.bool)
    target = torch.randn(2, 3, 5)
    prior_logit = torch.tensor([[0.0, 1.0, -1.0], [0.5, -0.5, 2.0]])
    pred, _ = model(observed, residual, valid, target, prior_logit)
    np.testing.assert_allclose(pred.detach().numpy(), torch.sigmoid(prior_logit).numpy(), rtol=0, atol=0)
