import torch

from cpg_repr_benchmark.proxy.model import (
    DenseProxyEncoder,
    FunctionalAnnotationEncoder,
    MeanMethylationProxy,
)


def test_dense_proxy_shape():
    encoder = DenseProxyEncoder(raw_dim=64, latent_dim=16, n_blocks=1, dropout=0.0)
    model = MeanMethylationProxy(encoder, 16)
    prediction, embedding = model({"raw_embedding": torch.randn(7, 64)})
    assert prediction.shape == (7,)
    assert embedding.shape == (7, 16)
    assert torch.all((prediction >= 0) & (prediction <= 1))


def test_functional_encoder_embeddingbag_contract():
    encoder = FunctionalAnnotationEncoder(n_tracks=8, dense_dim=3, latent_dim=6, n_blocks=1, dropout=0.0)
    indices = torch.tensor([0, 2, 1, 3, 4], dtype=torch.long)
    offsets = torch.tensor([0, 2, 5], dtype=torch.long)
    dense = torch.randn(2, 3)
    output = encoder(indices, offsets, dense)
    assert output.shape == (2, 6)
