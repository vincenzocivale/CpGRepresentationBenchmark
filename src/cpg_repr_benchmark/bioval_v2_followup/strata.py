"""Part B1: variance / mean-methylation / distance strata of the frozen Loyfer pairs (definitions fixed from frozen data)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import registry as rg
from .targets import LoyferProfiles


def per_cpg_stats(P):
    """Per-CpG mean and variance (ddof=0) over the observed (non-NaN, i.e. mask_primary) groups."""
    ok = ~np.isnan(P)
    n = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(ok, P, 0).sum(1) / n
        var = (np.where(ok, P - mean[:, None], 0.0) ** 2).sum(1) / n
    return mean, var, n


def pair_strata_values(pairs: pd.DataFrame, prof: LoyferProfiles):
    P = prof.matrix("primary")
    mean, var, _ = per_cpg_stats(P)
    ia, ib = prof.rows(pairs.cpg_i.to_numpy()), prof.rows(pairs.cpg_j.to_numpy())
    return np.minimum(var[ia], var[ib]), (mean[ia] + mean[ib]) / 2.0


def check_registered_edges(vmin, mmean, rtol=rg.EDGE_RTOL) -> dict:
    """Run-time assertion that the registered edges are the quantiles of the frozen pairs (no tuning possible)."""
    qv = np.quantile(vmin, [0.25, 0.5, 0.75, 0.9])
    qm = np.quantile(mmean, [0.25, 0.5, 0.75])
    reg_v = np.array([*rg.V_EDGES, rg.V_TOP10])
    reg_m = np.array(rg.M_EDGES)
    if not np.allclose(qv, reg_v, rtol=rtol, atol=0) or not np.allclose(qm, reg_m, rtol=rtol, atol=0):
        raise AssertionError(f"registered strata edges differ from frozen-pair quantiles: V {qv} vs {reg_v}; M {qm} vs {reg_m}")
    return {"V_quantiles": qv.tolist(), "M_quantiles": qm.tolist()}


def _quartile(v, edges):
    return np.digitize(v, edges, right=False)  # 0..3: edge value belongs to the upper bin


def cells(stratum, vmin, mmean) -> dict[str, np.ndarray]:
    """name -> boolean mask over pairs. Overlapping cells ('V_top10pct', 'intra') are allowed; all registered up-front."""
    stratum = np.asarray(stratum)
    out = {"all": np.ones(len(stratum), bool)}
    for s in rg.ALL_STRATA:
        out[f"dist_{s}"] = stratum == s
    out["chrom_intra"] = stratum != "interchromosomal"
    out["chrom_inter"] = stratum == "interchromosomal"
    qv = _quartile(vmin, rg.V_EDGES)
    for k in range(4):
        out[f"V_Q{k + 1}"] = qv == k
    out["V_top10pct"] = vmin >= rg.V_TOP10
    qm = _quartile(mmean, rg.M_EDGES)
    for k in range(4):
        out[f"M_Q{k + 1}"] = qm == k
    return out
