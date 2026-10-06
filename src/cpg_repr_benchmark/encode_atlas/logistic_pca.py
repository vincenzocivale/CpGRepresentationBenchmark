"""Logistic PCA for binary locus x track matrices (label-free, patient-independent).

Model: x_ij ~ Bernoulli(sigmoid(mu_j + (A V^T)_ij)), V (d x k) orthonormal. Fit = monotone MM algorithm (Bernoulli curvature <= 1/4):
  Z = Theta + 4 (X - sigmoid(Theta));  mu <- colmean(Z);  V <- top-k eigenvectors of the centred Gram of Z;  A <- (Z - mu) V.
No hyper-parameters. Embedding of ANY locus (fit or held-out) = `encode`: fixed-V gradient steps a <- a + 4 V^T (x - sigmoid(mu + V a))
from a = 0 (step 1/L is safe because V is orthonormal), so fit and out-of-fit rows go through the identical procedure.
Runs on torch (GPU if available); X is a scipy CSR or dense 0/1 matrix.
"""
from __future__ import annotations

import numpy as np
import torch


def _dense(x, rows, device):
    c = x[rows]
    c = c.toarray() if hasattr(c, "toarray") else np.asarray(c)
    return torch.as_tensor(c, dtype=torch.float32, device=device)


def fit(x, k: int, *, max_iter: int = 1000, tol: float = 1e-6, chunk: int = 32768, device: str = "cuda", c_max: float = 16.0,
        log=print) -> dict:
    """Fit on all rows of `x` (the caller restricts x to the fit loci). Returns mu, V, deviance trace, convergence info.

    MM with an ADAPTIVE majorisation constant c (4 = the guaranteed Bernoulli bound; larger = longer steps, valid when the true curvature
    p(1-p) is below 1/4, as for sparse tracks). If the deviance rises after a step, that step is discarded and c is halved (>= 4), so the
    accepted deviance trace is monotone non-increasing.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    n, d = x.shape
    dev = torch.device(device)
    p = np.clip(np.asarray(x.mean(axis=0)).ravel(), 1e-4, 1 - 1e-4)
    mu = torch.as_tensor(np.log(p / (1 - p)), dtype=torch.float32, device=dev)
    V = torch.zeros(d, k, device=dev)
    A = torch.zeros(n, k, device=dev)
    spans = [slice(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    xg = torch.empty(n, d, dtype=torch.uint8, device=dev)  # binary matrix resident on the device (n*d bytes); no per-iteration host work
    for s in spans:
        xg[s] = _dense(x, s, dev).to(torch.uint8)

    def z_chunk(s, mu_, V_, A_, c_):
        xb = xg[s].float()
        theta = mu_ + A_[s] @ V_.T
        dv = -(xb * torch.nn.functional.logsigmoid(theta) + (1 - xb) * torch.nn.functional.logsigmoid(-theta)).sum().item()
        return theta + c_ * (xb - torch.sigmoid(theta)), dv

    def deviance(mu_, V_, A_):
        return sum(z_chunk(s, mu_, V_, A_, 4.0)[1] for s in spans)

    c, trace, n_reject, converged = 4.0, [], 0, False
    state = (mu, V, A)
    cur = deviance(*state)
    trace.append(cur)
    for it in range(max_iter):
        mu0, V0, A0 = state
        s1 = torch.zeros(d, dtype=torch.float64, device=dev)
        G = torch.zeros(d, d, dtype=torch.float64, device=dev)
        for s in spans:
            z, _ = z_chunk(s, mu0, V0, A0, c)
            s1 += z.sum(0).double()
            G += (z.T @ z).double()
        mu_new = s1 / n
        w, U = torch.linalg.eigh(G - n * torch.outer(mu_new, mu_new))
        V_new = U[:, -k:].flip(1).float().contiguous()  # descending eigenvalue order
        mu_f = mu_new.float()
        A_new = torch.empty_like(A0)
        for s in spans:
            z, _ = z_chunk(s, mu0, V0, A0, c)
            A_new[s] = (z - mu_f) @ V_new
        new = deviance(mu_f, V_new, A_new)
        if new > cur:  # overshoot: discard, tighten the step (never below the guaranteed bound)
            n_reject += 1
            if c <= 4.0:
                log(f"  logistic PCA k={k}: deviance rose at c=4 (numerical floor); stopping"); break
            c = max(4.0, c / 2)
            continue
        rel = (cur - new) / abs(cur)
        state, cur = (mu_f, V_new, A_new), new
        trace.append(cur)
        c = min(c_max, c * 1.25)
        if rel < tol:
            converged = True
            break
        if it % 25 == 0:
            log(f"  logistic PCA k={k} iter {it}: deviance {cur:.6e} rel_step {rel:.2e} c={c:.1f}")
    mu, V, A = state
    return {"mu": mu.cpu().numpy(), "V": V.cpu().numpy(), "deviance": np.asarray(trace), "n_iter": len(trace) - 1,
            "n_rejected_steps": n_reject, "converged": converged, "final_rel_step": float(rel) if trace[1:] else None, "c_final": c,
            "A_head": A[:2000].cpu().numpy()}  # fit-time scores of the first rows, for the encode-consistency diagnostic


def encode(x, mu: np.ndarray, V: np.ndarray, *, steps: int = 100, chunk: int = 32768, device: str = "cuda") -> np.ndarray:
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(device)
    m, Vt = torch.as_tensor(mu, device=dev), torch.as_tensor(V, device=dev)
    out = np.empty((x.shape[0], V.shape[1]), np.float32)
    for i in range(0, x.shape[0], chunk):
        s = slice(i, min(i + chunk, x.shape[0]))
        xb = _dense(x, s, dev)
        a = torch.zeros(xb.shape[0], V.shape[1], device=dev)
        for _ in range(steps):
            a = a + 4 * (xb - torch.sigmoid(m + a @ Vt.T)) @ Vt
        out[s] = a.cpu().numpy()
    return out
