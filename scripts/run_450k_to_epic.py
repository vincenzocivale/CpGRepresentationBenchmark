#!/usr/bin/env python3
"""450K -> EPIC upscaling benchmark arm.

Train on *simulated* 450K profiles (EPIC cohorts restricted to the probes shared with the 450K array),
predict the EPIC-only probes, and test on (a) held-out simulated studies and (b) REAL same-sample
450K/EPIC pairs. Only ``representation`` changes between arms; see docs/UPSCALING_450K_TO_EPIC.md.

    CUDA_VISIBLE_DEVICES=0 python scripts/run_450k_to_epic.py --config configs/experiments/upscale_450k_epic/<arm>.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cpg_repr_benchmark.config import config_fingerprint, load_config
from cpg_repr_benchmark.experiments.run_store import create_run_dir, git_revision, write_experiment_manifest, write_summary
from cpg_repr_benchmark.models.model import build_reconstructor
from cpg_repr_benchmark.representations.control_providers import ConstantStore, RandomStableStore
from cpg_repr_benchmark.representations.hdf5_store import HDF5RepresentationStore
from cpg_repr_benchmark.upscaling.baselines import RidgePCAImputer, prior_mean_predict
from cpg_repr_benchmark.upscaling.data import (
    ArrayPairSet,
    fit_priors,
    load_real_pairs,
    load_simulated_training,
    load_universe,
    locus_holdout_masks,
)
from cpg_repr_benchmark.upscaling.engine import ReconstructionTask, train
from cpg_repr_benchmark.upscaling.metrics import bootstrap_ci, per_sample_table, summarize

REPO = Path(__file__).resolve().parents[1]


def _path(p: str) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else REPO / p


def build_embeddings(rep_cfg: dict, all_ids: np.ndarray) -> np.ndarray:
    kind = str(rep_cfg.get("kind", "hdf5"))
    if kind == "hdf5":
        store = HDF5RepresentationStore(_path(rep_cfg["store_h5"]), all_ids)
        if not store.coverage_mask.all():
            missing = all_ids[~store.coverage_mask][:10].tolist()
            raise RuntimeError(
                f"representation {rep_cfg['name']!r} misses {int((~store.coverage_mask).sum())} loci of the frozen "
                f"450K->EPIC universe (e.g. {missing}); build it on the universe rather than narrowing the task"
            )
        return store.get_by_matrix_columns(np.arange(len(all_ids)))
    dim = int(rep_cfg.get("dim", 256))
    columns = np.arange(len(all_ids))
    if kind == "random_stable":
        return RandomStableStore(columns, all_ids, dim=dim, seed=int(rep_cfg.get("seed", 17))).get_by_matrix_columns(columns)
    if kind == "constant":
        return ConstantStore(columns, dim=dim).get_by_matrix_columns(columns)
    raise ValueError(f"unknown representation.kind {kind!r}")


def eval_sets(real: dict[str, ArrayPairSet], sim_test: ArrayPairSet) -> dict[str, ArrayPairSet]:
    sets = {"sim_test": sim_test}
    for name, s in real.items():
        sets[f"real_{name}"] = s
    g86 = real.get("GSE86833")
    if g86 is not None:  # tissue strata: training cohorts are all whole/sorted blood
        grp = np.asarray(g86.groups)
        sets["real_GSE86833_blood"] = g86.subset(np.flatnonzero(grp == "blood"), "real_GSE86833_blood")
        sets["real_GSE86833_cells"] = g86.subset(np.flatnonzero(grp != "blood"), "real_GSE86833_cells")
    g92 = real.get("GSE92580")
    if g92 is not None:
        ff = np.flatnonzero(np.asarray(g92.groups) == "FF")
        ffpe = np.flatnonzero(np.asarray(g92.groups) != "FF")
        sets["real_GSE92580_FF"] = g92.subset(ff, "real_GSE92580_FF")
        sets["real_GSE92580_FFPE"] = g92.subset(ffpe, "real_GSE92580_FFPE")
    if "GSE86833" in real and g92 is not None:
        a, b = real["GSE86833"], sets["real_GSE92580_FF"]
        sets["real_primary"] = ArrayPairSet(
            "real_primary", np.concatenate([a.obs, b.obs]), np.concatenate([a.target, b.target]),
            a.sample_ids + b.sample_ids, a.groups + b.groups, np.concatenate([a.epic_obs, b.epic_obs]),
        )
    return sets


def evaluate(cfg, run_dir, sets, predict_fn, priors, target_train_mask, target_std) -> dict:
    ev = cfg["evaluation"]
    views = {"seen_loci": np.flatnonzero(target_train_mask)}
    if (~target_train_mask).any():
        views["heldout_loci"] = np.flatnonzero(~target_train_mask)
    prior_beta = (1.0 / (1.0 + np.exp(-priors.target_logit.astype(np.float64)))).astype(np.float32)
    results: dict = {}
    for name, data in sets.items():
        pred = predict_fn(data.obs)
        results[name] = {}
        for view, cols in views.items():
            sub_pred, sub_true = pred[:, cols], data.target[:, cols]
            sub_prior = prior_beta[cols]
            std = target_std[cols] if view == "seen_loci" else np.nanstd(sub_true, axis=0)
            summary = summarize(sub_pred, sub_true, sub_prior, std, float(ev.get("variable_quantile", 0.9)))
            entry = {"summary": summary, "n_loci": int(len(cols))}
            if len(data) >= 3 and int(ev.get("bootstrap", 200)):
                entry["ci95"] = bootstrap_ci(
                    sub_pred, sub_true, sub_prior, std, float(ev.get("variable_quantile", 0.9)), n_boot=int(ev.get("bootstrap", 200))
                )
            out_dir = run_dir / "evaluation" / name / view
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "metrics.json").write_text(json.dumps(entry, indent=2, sort_keys=True, allow_nan=True))
            tab = pd.DataFrame(per_sample_table(sub_pred, sub_true, sub_prior))
            tab.insert(0, "sample_id", data.sample_ids)
            tab.insert(1, "group", data.groups if data.groups else [""] * len(data))
            tab.to_csv(out_dir / "per_sample.csv", index=False)
            results[name][view] = entry
        if name.startswith("real_") and ev.get("save_predictions", True):
            np.savez_compressed(
                run_dir / "evaluation" / name / "predictions.npz",
                prediction=pred.astype(np.float16), target=data.target.astype(np.float16), sample_id=np.asarray(data.sample_ids),
            )
    return results


def platform_concordance(real: dict[str, ArrayPairSet]) -> dict:
    """Same-sample 450K vs EPIC on the shared probes: the technical floor of any 450K->EPIC transfer."""
    out = {}
    for name, s in real.items():
        d = s.obs - s.epic_obs
        r = [np.corrcoef(a[m], b[m])[0, 1] for a, b in zip(s.obs, s.epic_obs) if (m := np.isfinite(a) & np.isfinite(b)).sum() > 3]
        out[name] = {"mean_abs_diff": float(np.nanmean(np.abs(d))), "rmse": float(np.sqrt(np.nanmean(d**2))),
                     "mean_sample_pearson": float(np.mean(r))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tr, ds, rep = cfg["training"], cfg["dataset"], cfg["representation"]
    seed = int(tr["seed"])
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    device = torch.device(str(tr.get("device", "cuda")) if torch.cuda.is_available() else "cpu")
    baseline = str(cfg.get("baseline", "none"))
    run_dir = create_run_dir(cfg, REPO)

    obs_ids, tgt_ids = load_universe(_path(ds["universe"]))
    sim = load_simulated_training(_path(ds["simulated_training"]["h5"]), _path(ds["simulated_training"]["samples"]), obs_ids, tgt_ids)
    real = load_real_pairs(_path(ds["real_pairs"]["dir"]), list(ds["real_pairs"]["cohorts"]), obs_ids, tgt_ids)
    train_mask, heldout_mask = locus_holdout_masks(len(tgt_ids), float(ds["locus_holdout"]["fraction"]), int(ds["locus_holdout"]["seed"]))
    eps = float(tr.get("beta_epsilon", 1e-4))
    priors = fit_priors(sim["train"], train_mask, eps)
    np.savez_compressed(run_dir / "protocol_snapshot.npz", target_train_mask=train_mask, target_heldout_mask=heldout_mask)
    sets = eval_sets(real, sim["test_sim"])
    extra: dict = {"platform_concordance": platform_concordance(real)}

    if baseline == "prior_mean":
        predict_fn = lambda obs: prior_mean_predict(priors, len(obs))  # noqa: E731
        n_params, history = 0, []
    elif baseline == "ridge_pca":
        model = RidgePCAImputer(priors, train_mask, eps, device)
        extra["ridge_fit"] = model.fit(sim["train"], sim["validation"])
        predict_fn = model.predict
        n_params, history = 0, []
    else:
        all_ids = np.concatenate([obs_ids, tgt_ids])
        emb = build_embeddings(rep, all_ids)
        emb_obs, emb_tgt = emb[: len(obs_ids)], emb[len(obs_ids):]
        task = ReconstructionTask(emb_obs, emb_tgt, priors, train_mask, device, eps)
        model = build_reconstructor(emb.shape[1], cfg["model"])
        n_params = sum(p.numel() for p in model.parameters())
        history = train(
            model, task, sim["train"], sim["validation"], out_dir=run_dir,
            steps=int(tr["steps"]), batch_size=int(tr["batch_size"]), context_size=tuple(tr["context_size"]),
            target_size=int(tr["target_size"]), lr=float(tr["learning_rate"]), weight_decay=float(tr.get("weight_decay", 1e-4)),
            warmup=int(tr.get("warmup", 200)), eval_every=int(tr.get("eval_every", 250)),
            val_context=int(tr.get("val_context", 16384)), val_targets=int(tr.get("val_targets", 16384)),
            obs_noise_std=float(tr.get("obs_noise_std", 0.0)), seed=seed,
            prior_dropout=float(tr.get("prior_dropout", 0.0)),
        )
        state = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.to(device)
        ev = cfg["evaluation"]
        predict_fn = lambda obs: task.predict(  # noqa: E731
            model, obs, k=ev.get("context_size", 32768), repeats=int(ev.get("repeats", 3)), seed=seed
        )
        extra["best_step"] = state["step"]
        if ev.get("skip", False):  # development runs: model selection on validation studies only
            print(f"TRAIN_ONLY best_val_mse={state['metrics']['val_mse']:.5f} RUN_DIR={run_dir}")
            return

    write_experiment_manifest(run_dir, {
        "experiment": cfg["experiment"]["name"], "config_fingerprint": config_fingerprint(cfg),
        "code_git_commit": git_revision(REPO), "representation": rep["name"], "baseline": baseline, "seed": seed,
        "n_observed_loci": int(len(obs_ids)), "n_target_loci": int(len(tgt_ids)),
        "n_target_loci_train": int(train_mask.sum()), "n_target_loci_heldout": int(heldout_mask.sum()),
        "n_train_samples": len(sim["train"]), "n_validation_samples": len(sim["validation"]),
        "n_test_sim_samples": len(sim["test_sim"]), "n_parameters": int(n_params),
        "real_pairs": {k: len(v) for k, v in real.items()},
    })
    results = evaluate(cfg, run_dir, sets, predict_fn, priors, train_mask, priors.target_std)
    write_summary(run_dir, {"experiment": cfg["experiment"]["name"], "representation": rep["name"], "baseline": baseline,
                            "seed": seed, "evaluation": results, **extra})
    print(json.dumps({k: {v: e["summary"] for v, e in views.items()} for k, views in results.items()}, indent=1, allow_nan=True))
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
