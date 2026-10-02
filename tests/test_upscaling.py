import numpy as np
import torch

from cpg_repr_benchmark.models.model import build_reconstructor
from cpg_repr_benchmark.upscaling.baselines import RidgePCAImputer, prior_mean_predict
from cpg_repr_benchmark.upscaling.data import ArrayPairSet, fit_priors, locus_holdout_masks
from cpg_repr_benchmark.upscaling.engine import ReconstructionTask, train
from cpg_repr_benchmark.upscaling.metrics import per_sample_table, summarize


def _toy(n=24, n_obs=60, n_tgt=40, seed=0, rank=3):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, rank))
    lo = rng.normal(size=(rank, n_obs + n_tgt))
    logits = z @ lo + rng.normal(scale=0.1, size=(n, n_obs + n_tgt))
    beta = (1 / (1 + np.exp(-logits))).astype(np.float32)
    return ArrayPairSet("toy", beta[:, :n_obs].copy(), beta[:, n_obs:].copy(), [f"s{i}" for i in range(n)], ["g"] * n)


def test_locus_holdout_excludes_heldout_from_priors():
    data = _toy()
    train_mask, held = locus_holdout_masks(data.target.shape[1], 0.25, seed=1)
    assert held.sum() == 10 and not (train_mask & held).any()
    p1 = fit_priors(data, train_mask, 1e-4)
    changed = data.target.copy()
    changed[:, held] = 0.99  # held-out locus values must not influence anything the model sees
    p2 = fit_priors(ArrayPairSet("t", data.obs, changed, data.sample_ids, data.groups), train_mask, 1e-4)
    assert np.allclose(p1.target_logit, p2.target_logit)
    assert np.all(p1.target_logit[held] == p1.global_logit)
    assert np.isnan(p1.target_std[held]).all()


def test_metrics_match_naive():
    rng = np.random.default_rng(3)
    true = rng.uniform(size=(8, 30)).astype(np.float32)
    pred = np.clip(true + rng.normal(scale=0.1, size=true.shape), 0, 1).astype(np.float32)
    true[0, 0] = np.nan
    prior = np.full(30, 0.5, dtype=np.float32)
    s = summarize(pred, true, prior, locus_std=np.nanstd(true, axis=0))
    m = np.isfinite(true)
    assert np.isclose(s["mse"], ((pred - true)[m] ** 2).mean())
    tab = per_sample_table(pred, true, prior)
    mm = m[1]
    assert np.isclose(tab["pearson"][1], np.corrcoef(pred[1][mm], true[1][mm])[0, 1], atol=1e-5)
    assert np.isclose(s["locus_pearson_all"], np.nanmean([np.corrcoef(pred[:, j], true[:, j])[0, 1] for j in range(1, 30)]), atol=1e-2)


def test_batch_never_samples_heldout_targets_and_masks_missing():
    data = _toy()
    data.obs[:, :5] = np.nan  # missing observed probes must be flagged invalid, never used as tokens
    train_mask, held = locus_holdout_masks(data.target.shape[1], 0.25, seed=2)
    priors = fit_priors(data, train_mask, 1e-4)
    task = ReconstructionTask(np.random.randn(60, 8), np.random.randn(40, 8), priors, train_mask, torch.device("cpu"))
    obs, tgt = task.put(data)
    gen = torch.Generator().manual_seed(0)
    b = task.batch(obs, tgt, torch.arange(4), k=50, m=20, gen=gen)
    assert b["observed_locus"].shape == (4, 50, 8) and b["target_locus"].shape == (4, 20, 8)
    assert b["observed_valid"].all()  # 55 finite probes >= k
    b_all = task.batch(obs, tgt, torch.arange(4), k=60, m=20, gen=gen)
    assert (~b_all["observed_valid"]).sum() == 4 * 5
    b_drop = task.batch(obs, tgt, torch.arange(4), k=50, m=20, gen=gen, prior_dropout=1.0)
    assert torch.allclose(b_drop["target_prior_logit"], torch.full_like(b_drop["target_prior_logit"], priors.global_logit))
    assert torch.isin(task.pool, torch.as_tensor(np.flatnonzero(held))).sum() == 0


def test_train_step_and_predict_shapes(tmp_path):
    data = _toy(n=32)
    train_set, val_set = data.subset(np.arange(24), "tr"), data.subset(np.arange(24, 32), "va")
    train_mask, _ = locus_holdout_masks(40, 0.0, 0)
    priors = fit_priors(train_set, train_mask, 1e-4)
    task = ReconstructionTask(np.random.randn(60, 8), np.random.randn(40, 8), priors, train_mask, torch.device("cpu"))
    model = build_reconstructor(8, {"architecture": "deepsets", "locus_latent_dim": 16, "token_dim": 16, "patient_dim": 16, "hidden_dim": 32})
    hist = train(model, task, train_set, val_set, out_dir=tmp_path, steps=6, batch_size=4, context_size=(20, 40), target_size=16,
                 lr=1e-3, weight_decay=0.0, warmup=1, eval_every=3, val_context=30, val_targets=20, obs_noise_std=0.02, seed=1)
    assert len(hist) == 2 and (tmp_path / "checkpoints" / "best.pt").exists()
    pred = task.predict(model, val_set.obs, k=30, repeats=2, seed=1, chunk=16)
    assert pred.shape == (8, 40) and np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    # An untrained decoder reproduces the prior exactly (zero-initialised residual head).
    fresh = build_reconstructor(8, {"architecture": "deepsets", "locus_latent_dim": 16, "token_dim": 16, "patient_dim": 16, "hidden_dim": 32})
    base = task.predict(fresh, val_set.obs, k=30)
    assert np.allclose(base, prior_mean_predict(priors, 8), atol=1e-5)


def test_ridge_baseline_recovers_low_rank_structure():
    data = _toy(n=80, n_obs=120, n_tgt=60, rank=3)
    tr, va, te = data.subset(np.arange(50), "tr"), data.subset(np.arange(50, 65), "va"), data.subset(np.arange(65, 80), "te")
    mask, _ = locus_holdout_masks(60, 0.0, 0)
    priors = fit_priors(tr, mask, 1e-4)
    ridge = RidgePCAImputer(priors, mask, 1e-4, torch.device("cpu"))
    ridge.fit(tr, va, ks=(2, 3, 8), alphas=(0.1, 1.0))
    pred = ridge.predict(te.obs)
    prior = prior_mean_predict(priors, len(te))
    assert ((pred - te.target) ** 2).mean() < 0.5 * ((prior - te.target) ** 2).mean()
