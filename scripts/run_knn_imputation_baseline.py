#!/usr/bin/env python3
"""B1(e): patient-neighbour (KNN) methylation imputation on the exact validation panels of the masking benchmark.

For each VALIDATION patient the observed panel columns are those of `MaskingDataset` (same mask_seed/panel_size/protocols as every
frozen arm); the k nearest TRAIN patients (nan-aware mean squared beta distance on the observed columns) are found and their mean beta at the
masked columns is the prediction (fallback: train-locus prior). No locus representation is used; no test row is read
unless `--split test` is passed (default: validation; the test split is read only by that explicit option).
Metrics use the benchmark's `reconstruction_metrics`, so MSE/MAE/skill are directly comparable with evaluation/validation/summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from cpg_repr_benchmark.data.masking import MaskingDataset
from cpg_repr_benchmark.data.methylation import read_axis
from cpg_repr_benchmark.evaluation.metrics import reconstruction_metrics

ROOT = Path(__file__).resolve().parents[1]


class _NullStore:
    dim = 1

    def get_by_matrix_columns(self, cols):
        return np.zeros((len(cols), 1), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-run", required=True, help="a frozen run dir whose prior_logit.npy/locus_split.npz/patient_split.npz are shared")
    ap.add_argument("--matrix", default=str(ROOT / "data/methylation/tcga_array_official_full.h5"))
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 50])
    ap.add_argument("--primary-k", type=int, default=10, help="fixed a priori; other k reported as sensitivity")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.15, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--out", default=str(ROOT / "outputs/b1_b2_baselines/knn/summary.json"))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--split", choices=["validation", "test"], default="validation",
                    help="patient split to impute (default validation; test reads the test rows with the same masks, mask seed 17001)")
    ap.add_argument("--save-predictions", default=None,
                    help="optional .npz with the per-entry predictions of the primary k (single mask fraction only)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run = Path(a.prior_run)
    prior = np.load(run / "prior_logit.npy")
    ps = np.load(run / "patient_split.npz")
    train_rows, val_rows = ps["train"], ps[a.split]  # val_rows = rows of the requested split; test rows are read only with --split test
    train_cols = np.load(run / "locus_split.npz")["train_matrix_columns"]
    ds = MaskingDataset(methylation_h5=a.matrix, representation_store=_NullStore(), prior_logit_full=prior, sample_indices=val_rows,
                        context_columns=train_cols, target_columns=train_cols, panel_size=2048, mask_fractions=[0.5], seed=17001)
    cpg_ids, _ = read_axis(Path(a.matrix))
    n_cpg = len(cpg_ids)
    print("loading train betas", len(train_rows), "x", n_cpg, flush=True)
    train = np.empty((len(train_rows), n_cpg), dtype=np.float32)
    with h5py.File(a.matrix, "r") as h:
        order = np.argsort(train_rows)
        for s in range(0, len(order), 256):
            idx = order[s : s + 256]
            train[idx] = h["beta"][np.sort(train_rows[idx])]
    train_t = torch.from_numpy(train)
    prior_beta = 1 / (1 + np.exp(-prior))
    out = {"method": "knn_patient_neighbours", "ks": a.ks, "primary_k": a.primary_k, "patient_view": a.split,
           "prior_run": str(run), "results": {}}
    for frac in a.fractions:
        ds.mask_fractions = (frac,)
        preds = {k: [] for k in a.ks}
        targets, priors, cols = [], [], []
        for i in range(len(ds)):
            item = ds[(i, frac)]
            n_obs = int(item["observed_valid"].shape[0])
            tcols = item["target_matrix_column"].numpy()
            # observed columns/values are not exposed as columns; recover via the same deterministic draw
            obs_cols, obs_beta = _observed(ds, i, frac, n_obs)
            ob = torch.from_numpy(obs_beta)
            A = train_t[:, torch.from_numpy(obs_cols)]
            m = torch.isfinite(A)
            d = (torch.where(m, A - ob, torch.zeros(()))**2).sum(1) / m.sum(1).clamp_min(1)
            d[m.sum(1) < 20] = float("inf")
            for k in a.ks:
                nn_idx = torch.topk(d, k, largest=False).indices
                B = train_t[nn_idx][:, torch.from_numpy(tcols)]
                mean = torch.nanmean(B, dim=0).numpy()
                preds[k].append(np.where(np.isfinite(mean), mean, prior_beta[tcols]))
            targets.append(item["target_beta"].numpy())
            priors.append(prior_beta[tcols])
            cols.append(tcols)
            if i % 100 == 0:
                print(frac, i, len(ds), flush=True)
        T, P, C = np.stack(targets), np.stack(priors), np.stack(cols)
        out["results"][f"mask_{frac:.2f}"] = {f"k{k}": reconstruction_metrics(np.stack(preds[k]), T, P, target_matrix_column=C) for k in a.ks}
        if a.save_predictions:
            assert len(a.fractions) == 1 and a.primary_k in a.ks, "--save-predictions needs one fraction and primary-k in --ks"
            Path(a.save_predictions).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(a.save_predictions, prediction=np.stack(preds[a.primary_k]), target=T, prior_prediction=P,
                                target_matrix_column=C, sample_index=np.asarray(val_rows, dtype=np.int64), k=a.primary_k)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, allow_nan=True))
    print(json.dumps({f: {k: v["mse"] for k, v in r.items()} for f, r in out["results"].items()}, indent=1))


def _observed(ds: MaskingDataset, item: int, frac: float, n_obs: int):
    """Re-draw the (deterministic) panel of `ds[item]` to get the observed columns and their beta values."""
    from cpg_repr_benchmark.data.masking import counts_for_mask_fraction

    row = int(ds.sample_indices[item])
    o, t = counts_for_mask_fraction(ds.panel_size, frac)
    assert o == n_obs
    rng = np.random.default_rng(ds.seed + ds._epoch.value * 1_000_003 + row * 97 + round(frac * 1_000_000))
    cols, vals = ds._finite_sample(row, ds.context_columns, rng, o + t)
    return cols[:o], vals[:o]


if __name__ == "__main__":
    main()
