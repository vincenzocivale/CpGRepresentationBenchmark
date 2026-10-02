from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from cpg_repr_benchmark.training.priors import beta_to_logit


@dataclass
class ArrayPairSet:
    """Samples with the observed-array profile and the target-array profile (NaN = not measured)."""

    name: str
    obs: np.ndarray  # [n, |O|] float32 - what a 450K profile offers (probes shared with EPIC)
    target: np.ndarray  # [n, |T|] float32 - EPIC-only probes to reconstruct
    sample_ids: list[str]
    groups: list[str] = field(default_factory=list)
    epic_obs: np.ndarray | None = None  # real pairs only: the EPIC measurement of the shared probes

    def __len__(self) -> int:
        return len(self.sample_ids)

    def subset(self, rows: np.ndarray, name: str) -> "ArrayPairSet":
        rows = np.asarray(rows)
        return ArrayPairSet(
            name=name,
            obs=self.obs[rows],
            target=self.target[rows],
            sample_ids=[self.sample_ids[i] for i in rows],
            groups=[self.groups[i] for i in rows] if self.groups else [],
            epic_obs=None if self.epic_obs is None else self.epic_obs[rows],
        )


def load_universe(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["observed_cpg_idx"].astype(np.int64), data["target_cpg_idx"].astype(np.int64)


def load_simulated_training(h5_path: Path, samples_parquet: Path, obs_ids: np.ndarray, tgt_ids: np.ndarray) -> dict[str, ArrayPairSet]:
    """Study-disjoint train / validation / test_sim sets from real EPIC cohorts."""
    samples = pd.read_parquet(samples_parquet)
    with h5py.File(h5_path, "r") as h:
        if not np.array_equal(h["cpg_idx_obs"][:], obs_ids) or not np.array_equal(h["cpg_idx_target"][:], tgt_ids):
            raise ValueError(f"{h5_path} loci differ from the frozen universe protocol")
        obs = h["obs_beta"][:]
        tgt = h["target_beta"][:]
    full = ArrayPairSet("all", obs, tgt, samples["sample_name"].tolist(), samples["dataset_id"].tolist())
    return {
        split: full.subset(np.flatnonzero((samples["split"] == split).to_numpy()), split)
        for split in ("train", "validation", "test_sim")
    }


def load_real_pairs(prepared_dir: Path, cohorts: list[str], obs_ids: np.ndarray, tgt_ids: np.ndarray) -> dict[str, ArrayPairSet]:
    """Real same-sample 450K/EPIC pairs. Used for testing only - never for fitting or selection."""
    sets: dict[str, ArrayPairSet] = {}
    for cohort in cohorts:
        pairs = pd.read_parquet(Path(prepared_dir) / f"{cohort}_pairs.parquet")
        with h5py.File(Path(prepared_dir) / f"{cohort}.h5", "r") as h:
            if not np.array_equal(h["cpg_idx_obs"][:], obs_ids) or not np.array_equal(h["cpg_idx_target"][:], tgt_ids):
                raise ValueError(f"{cohort} loci differ from the frozen universe protocol")
            sets[cohort] = ArrayPairSet(
                cohort, h["obs_beta"][:], h["target_beta"][:], pairs["sample_id"].tolist(),
                pairs["group"].tolist(), epic_obs=h["epic_obs_beta"][:],
            )
    return sets


def locus_holdout_masks(n_targets: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random target-locus holdout (train mask, heldout mask); fraction 0 keeps every locus in training."""
    heldout = np.zeros(n_targets, dtype=bool)
    if fraction > 0:
        rng = np.random.default_rng(seed)
        heldout[rng.choice(n_targets, size=int(round(fraction * n_targets)), replace=False)] = True
    return ~heldout, heldout


@dataclass
class Priors:
    obs_logit: np.ndarray  # [|O|] train-sample mean per shared locus (as logit)
    target_logit: np.ndarray  # [|T|] train-sample mean; heldout loci get the global fallback
    target_std: np.ndarray  # [|T|] train-sample beta std (NaN for heldout loci): locus variability
    global_logit: float


def fit_priors(train: ArrayPairSet, target_train_mask: np.ndarray, epsilon: float) -> Priors:
    """Locus means from TRAIN samples only; held-out target loci never contribute (global fallback)."""
    with warnings.catch_warnings():  # loci never measured in the train samples fall back to the global mean
        warnings.simplefilter("ignore", RuntimeWarning)
        obs_mean = np.nanmean(train.obs, axis=0)
        tgt_mean = np.nanmean(train.target, axis=0)
        tgt_std = np.nanstd(train.target, axis=0)
    global_beta = float(np.nanmean(train.obs))
    fallback = float(beta_to_logit(np.asarray([global_beta]), epsilon)[0])
    obs_logit = np.where(np.isfinite(obs_mean), beta_to_logit(np.nan_to_num(obs_mean, nan=global_beta), epsilon), fallback)
    tgt_logit = np.where(
        target_train_mask & np.isfinite(tgt_mean), beta_to_logit(np.nan_to_num(tgt_mean, nan=global_beta), epsilon), fallback
    )
    tgt_std = np.where(target_train_mask, tgt_std, np.nan)
    return Priors(obs_logit.astype(np.float32), tgt_logit.astype(np.float32), tgt_std.astype(np.float32), fallback)
