from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cpg_repr_benchmark.upscaling.data import ArrayPairSet, Priors


def _logit(beta: torch.Tensor, eps: float) -> torch.Tensor:
    beta = beta.clamp(eps, 1.0 - eps)
    return torch.log(beta) - torch.log1p(-beta)


class ReconstructionTask:
    """GPU-resident sampler: observed context loci + masked EPIC-only targets, per sample.

    Everything patient-agnostic (locus embeddings, priors) is a fixed table; the model only ever
    sees (locus embedding, beta residual w.r.t. the train prior) tokens, so it stays
    representation-agnostic.
    """

    def __init__(
        self,
        emb_obs: np.ndarray,
        emb_target: np.ndarray,
        priors: Priors,
        target_train_mask: np.ndarray,
        device: torch.device,
        epsilon: float = 1e-4,
    ):
        self.device, self.eps = device, epsilon
        self.emb_obs = torch.as_tensor(emb_obs, dtype=torch.float16, device=device)
        self.emb_tgt = torch.as_tensor(emb_target, dtype=torch.float16, device=device)
        self.prior_obs = torch.as_tensor(priors.obs_logit, device=device)
        self.prior_tgt = torch.as_tensor(priors.target_logit, device=device)
        self.global_logit = float(priors.global_logit)
        self.pool = torch.as_tensor(np.flatnonzero(target_train_mask), device=device)  # trainable target loci

    def put(self, split: ArrayPairSet) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(split.obs, dtype=torch.float16, device=self.device),
            torch.as_tensor(split.target, dtype=torch.float16, device=self.device),
        )

    def batch(
        self, obs, tgt, rows, k: int, m: int, gen: torch.Generator, noise_std: float = 0.0, prior_dropout: float = 0.0
    ) -> dict[str, torch.Tensor]:
        o = obs[rows].float()
        score = torch.rand(o.shape, generator=gen, device=self.device)
        score[~torch.isfinite(o)] = -1.0
        idx = score.topk(k, dim=1).indices
        val = o.gather(1, idx)
        valid = torch.isfinite(val)
        val = torch.nan_to_num(val, nan=0.5)
        if noise_std > 0:
            val = (val + noise_std * torch.randn(val.shape, generator=gen, device=self.device)).clamp(0, 1)
        resid = torch.where(valid, _logit(val, self.eps) - self.prior_obs[idx], torch.zeros_like(val))

        t = tgt[rows][:, self.pool].float()
        tscore = torch.rand(t.shape, generator=gen, device=self.device)
        tscore[~torch.isfinite(t)] = -1.0
        tsel = tscore.topk(m, dim=1).indices
        tval = t.gather(1, tsel)
        tidx = self.pool[tsel]
        target_prior = self.prior_tgt[tidx]
        if prior_dropout > 0:
            # Prior dropout: some targets are queried with the global fallback prior, exactly as never-seen
            # loci are at test time, so the decoder must read the locus level off the representation.
            drop = torch.rand(target_prior.shape, generator=gen, device=self.device) < prior_dropout
            target_prior = torch.where(drop, torch.full_like(target_prior, self.global_logit), target_prior)
        return {
            "observed_locus": self.emb_obs[idx].float(),
            "observed_residual": resid,
            "observed_valid": valid,
            "target_locus": self.emb_tgt[tidx].float(),
            "target_prior_logit": target_prior,
            "target_beta": torch.nan_to_num(tval, nan=0.0),
            "target_valid": torch.isfinite(tval),
        }

    @torch.no_grad()
    def predict(self, model, obs: np.ndarray, k: int | None, repeats: int = 1, seed: int = 17, chunk: int = 65536) -> np.ndarray:
        """Reconstruct every EPIC-only locus for each sample from (a random panel of) its observed loci."""
        model.eval()
        out = np.empty((obs.shape[0], self.emb_tgt.shape[0]), dtype=np.float32)
        gen = torch.Generator().manual_seed(seed)
        for i in range(obs.shape[0]):
            row = torch.as_tensor(obs[i], dtype=torch.float32, device=self.device)
            finite = torch.nonzero(torch.isfinite(row)).squeeze(1)
            acc = torch.zeros(self.emb_tgt.shape[0], device=self.device)
            for _ in range(repeats):
                if k is None or k >= len(finite):
                    sel = finite
                else:
                    sel = finite[torch.randperm(len(finite), generator=gen)[:k].to(self.device)]
                resid = _logit(row[sel], self.eps) - self.prior_obs[sel]
                patient = model.encode_patient(
                    self.emb_obs[sel].float()[None], resid[None], torch.ones(1, len(sel), dtype=torch.bool, device=self.device)
                )
                for a in range(0, self.emb_tgt.shape[0], chunk):
                    b = min(a + chunk, self.emb_tgt.shape[0])
                    acc[a:b] += model.decode_targets(patient, self.emb_tgt[a:b].float()[None], self.prior_tgt[a:b][None])[0]
            out[i] = (acc / repeats).cpu().numpy()
        return out


def train(
    model: torch.nn.Module,
    task: ReconstructionTask,
    train_set: ArrayPairSet,
    val_set: ArrayPairSet,
    *,
    out_dir: Path,
    steps: int,
    batch_size: int,
    context_size: tuple[int, int],
    target_size: int,
    lr: float,
    weight_decay: float,
    warmup: int,
    eval_every: int,
    val_context: int,
    val_targets: int,
    obs_noise_std: float,
    seed: int,
    prior_dropout: float = 0.0,
) -> list[dict]:
    device = task.device
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = lambda s: min(1.0, (s + 1) / max(warmup, 1)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0))))  # noqa: E731
    obs, tgt = task.put(train_set)
    vobs, vtgt = task.put(val_set)
    gen = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    history: list[dict] = []
    best, run_loss, t0 = float("inf"), [], time.time()
    for step in range(steps):
        model.train()
        for g in optimizer.param_groups:
            g["lr"] = lr * schedule(step)
        rows = torch.as_tensor(rng.integers(0, len(train_set), size=batch_size), device=device)
        k = int(rng.integers(context_size[0], context_size[1] + 1))
        b = task.batch(obs, tgt, rows, k, target_size, gen, obs_noise_std, prior_dropout)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred, _ = model(b["observed_locus"], b["observed_residual"], b["observed_valid"], b["target_locus"], b["target_prior_logit"])
        w = b["target_valid"].float()
        loss = (((pred.float() - b["target_beta"]) ** 2) * w).sum() / w.sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        run_loss.append(float(loss.detach()))
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            val = validate(model, task, vobs, vtgt, val_context, val_targets, seed=seed)
            rec = {"step": step + 1, "train_mse": float(np.mean(run_loss)), "val_mse": val["mse"], "val_prior_mse": val["prior_mse"],
                   "val_skill_vs_prior": 1 - val["mse"] / val["prior_mse"], "lr": lr * schedule(step), "elapsed_s": time.time() - t0}
            run_loss = []
            history.append(rec)
            print(json.dumps(rec), flush=True)
            state = {"model": model.state_dict(), "step": step + 1, "metrics": rec}
            torch.save(state, out_dir / "checkpoints" / "last.pt")
            if rec["val_mse"] < best:
                best = rec["val_mse"]
                torch.save(state, out_dir / "checkpoints" / "best.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return history


@torch.no_grad()
def validate(model, task: ReconstructionTask, vobs, vtgt, k: int, m: int, seed: int, batch_size: int = 16) -> dict[str, float]:
    """Fixed-panel MSE on held-out *studies* (simulated protocol); prior MSE shares the same target draw."""
    model.eval()
    gen = torch.Generator(device=task.device).manual_seed(10_000 + seed)
    se = pse = n = 0.0
    for s in range(0, vobs.shape[0], batch_size):
        rows = torch.arange(s, min(s + batch_size, vobs.shape[0]), device=task.device)
        b = task.batch(vobs, vtgt, rows, k, m, gen)
        pred, _ = model(b["observed_locus"], b["observed_residual"], b["observed_valid"], b["target_locus"], b["target_prior_logit"])
        w = b["target_valid"].float()
        se += float((((pred - b["target_beta"]) ** 2) * w).sum())
        pse += float((((torch.sigmoid(b["target_prior_logit"]) - b["target_beta"]) ** 2) * w).sum())
        n += float(w.sum())
    return {"mse": se / n, "prior_mse": pse / n}
