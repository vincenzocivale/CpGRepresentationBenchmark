from __future__ import annotations

import numpy as np
import torch

from cpg_repr_benchmark.upscaling.data import ArrayPairSet, Priors


def prior_mean_predict(priors: Priors, n_samples: int) -> np.ndarray:
    """Train-sample mean beta per EPIC-only locus (global mean for loci without training data)."""
    beta = 1.0 / (1.0 + np.exp(-priors.target_logit.astype(np.float64)))
    return np.broadcast_to(beta.astype(np.float32), (n_samples, len(beta))).copy()


def _logit_np(beta: np.ndarray, eps: float) -> np.ndarray:
    beta = np.clip(beta, eps, 1 - eps)
    return np.log(beta) - np.log1p(-beta)


class RidgePCAImputer:
    """Classical linear imputation baseline that uses NO locus representation.

    observed residuals (logit beta - train prior, NaN -> 0) -> top-k sample-space PCs -> ridge regression
    onto target residuals. k and alpha are picked on validation studies. It can only predict the target
    loci it was trained on; held-out target loci fall back to the prior (it has no way to generalize).
    """

    def __init__(self, priors: Priors, target_train_mask: np.ndarray, epsilon: float, device: torch.device):
        self.priors, self.mask, self.eps, self.device = priors, target_train_mask, epsilon, device

    def _resid(self, obs: np.ndarray) -> torch.Tensor:
        r = _logit_np(np.nan_to_num(obs, nan=0.5), self.eps) - self.priors.obs_logit[None]
        r[~np.isfinite(obs)] = 0.0
        return torch.as_tensor(r, dtype=torch.float32, device=self.device)

    def fit(self, train: ArrayPairSet, val: ArrayPairSet, ks=(16, 32, 64, 128, 256), alphas=(1.0, 10.0, 100.0, 1000.0, 10000.0)) -> dict:
        X = self._resid(train.obs)
        self.mu = X.mean(0, keepdim=True)
        Xc = X - self.mu
        evals, U = torch.linalg.eigh(Xc @ Xc.T)
        order = torch.argsort(evals, descending=True)[: max(ks)]
        sing = evals[order].clamp_min(1e-8).sqrt()
        self.V = (Xc.T @ U[:, order]) / sing  # [|O|, kmax]
        Z = Xc @ self.V
        Yn = train.target
        Y = _logit_np(np.nan_to_num(Yn, nan=0.5), self.eps) - self.priors.target_logit[None]
        Y[~np.isfinite(Yn)] = 0.0
        Y[:, ~self.mask] = 0.0
        Y = torch.as_tensor(Y, dtype=torch.float32, device=self.device)
        Zv = (self._resid(val.obs) - self.mu) @ self.V
        prior_tgt = torch.as_tensor(self.priors.target_logit, device=self.device)
        vt = torch.as_tensor(val.target, device=self.device)
        vm = torch.isfinite(vt) & torch.as_tensor(self.mask, device=self.device)[None]
        best = (float("inf"), None, None, None)
        trace = []
        for k in ks:
            Zk = Z[:, :k]
            for a in alphas:
                W = torch.linalg.solve(Zk.T @ Zk + a * torch.eye(k, device=self.device), Zk.T @ Y)
                pred = torch.sigmoid(prior_tgt[None] + Zv[:, :k] @ W)
                mse = float((((pred - torch.nan_to_num(vt)) ** 2) * vm).sum() / vm.sum())
                trace.append({"k": k, "alpha": a, "val_mse": mse})
                if mse < best[0]:
                    best = (mse, k, a, W)
        self.val_mse, self.k, self.alpha, self.W = best
        return {"selected_k": self.k, "selected_alpha": self.alpha, "val_mse": self.val_mse, "grid": trace}

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        Z = (self._resid(obs) - self.mu) @ self.V[:, : self.k]
        prior_tgt = torch.as_tensor(self.priors.target_logit, device=self.device)
        return torch.sigmoid(prior_tgt[None] + Z @ self.W).cpu().numpy()
