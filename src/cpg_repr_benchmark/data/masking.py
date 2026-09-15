from __future__ import annotations

from collections.abc import Iterator, Sequence
from multiprocessing import Value
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from cpg_repr_benchmark.representations.base import LocusRepresentationStore
from cpg_repr_benchmark.training.priors import beta_to_logit

from .methylation import read_row_columns


def counts_for_mask_fraction(panel_size: int, mask_fraction: float) -> tuple[int, int]:
    if panel_size < 2:
        raise ValueError("panel_size must be >= 2")
    if not 0.0 < mask_fraction < 1.0:
        raise ValueError("mask_fraction must be between 0 and 1")
    target = int(round(panel_size * mask_fraction))
    target = min(max(target, 1), panel_size - 1)
    return panel_size - target, target


class MaskingDataset(Dataset):
    """Patient-level masked methylome samples with explicit context and target locus pools.

    If context_columns == target_columns, a single finite panel is sampled and split into
    observed/target subsets. For the unseen-locus view the pools are disjoint by construction:
    context comes only from train loci and targets only from held-out loci.
    """

    def __init__(
        self,
        *,
        methylation_h5: Path,
        representation_store: LocusRepresentationStore,
        prior_logit_full: np.ndarray,
        sample_indices: np.ndarray,
        context_columns: np.ndarray,
        target_columns: np.ndarray,
        panel_size: int,
        mask_fractions: Sequence[float],
        seed: int = 17,
        beta_epsilon: float = 1e-4,
    ):
        self.methylation_h5 = Path(methylation_h5)
        self.store = representation_store
        self.prior = np.asarray(prior_logit_full, dtype=np.float32)
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.context_columns = np.asarray(context_columns, dtype=np.int64)
        self.target_columns = np.asarray(target_columns, dtype=np.int64)
        self.panel_size = int(panel_size)
        self.mask_fractions = tuple(float(x) for x in mask_fractions)
        self.seed = int(seed)
        self.beta_epsilon = float(beta_epsilon)
        if not self.mask_fractions:
            raise ValueError("mask_fractions cannot be empty")
        if len(self.context_columns) == 0 or len(self.target_columns) == 0:
            raise ValueError("context and target locus pools must both be non-empty")
        for fraction in self.mask_fractions:
            observed, target = counts_for_mask_fraction(self.panel_size, fraction)
            if observed > len(self.context_columns):
                raise ValueError(f"mask fraction {fraction} requires {observed} context loci, only {len(self.context_columns)} available")
            if target > len(self.target_columns):
                raise ValueError(f"mask fraction {fraction} requires {target} target loci, only {len(self.target_columns)} available")
        self._same_pool = np.array_equal(self.context_columns, self.target_columns)
        if not self._same_pool and np.intersect1d(self.context_columns, self.target_columns).size:
            raise ValueError("separate context/target pools must be disjoint")
        self._epoch = Value("q", 0)
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.sample_indices)

    def set_epoch(self, epoch: int) -> None:
        self._epoch.value = int(epoch)

    def _beta(self):
        if self._handle is None:
            self._handle = h5py.File(self.methylation_h5, "r")
        return self._handle["beta"]

    def _finite_sample(self, row: int, pool: np.ndarray, rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
        candidate_count = min(len(pool), max(count + 64, int(np.ceil(count / 0.75))))
        for _ in range(5):
            candidate_positions = rng.choice(len(pool), size=candidate_count, replace=False)
            candidates = pool[candidate_positions]
            values = read_row_columns(self._beta(), row, candidates)
            finite = np.flatnonzero(np.isfinite(values))
            if len(finite) >= count:
                chosen = rng.choice(finite, size=count, replace=False)
                return candidates[chosen], values[chosen]
            candidate_count = min(len(pool), max(candidate_count + 1, int(candidate_count * 1.5)))
        # Final exact fallback for sparse rows.
        values = read_row_columns(self._beta(), row, pool)
        finite = np.flatnonzero(np.isfinite(values))
        if len(finite) < count:
            raise ValueError(f"sample row {row} has only {len(finite)} finite CpGs in pool; need {count}")
        chosen = rng.choice(finite, size=count, replace=False)
        return pool[chosen], values[chosen]

    def __getitem__(self, item: int | tuple[int, float]) -> dict[str, torch.Tensor]:
        if isinstance(item, tuple):
            item, mask_fraction = item
        else:
            mask_fraction = self.mask_fractions[0]
        row = int(self.sample_indices[int(item)])
        mask_fraction = float(mask_fraction)
        observed_count, target_count = counts_for_mask_fraction(self.panel_size, mask_fraction)
        # Fraction contributes to the RNG key so every sweep point gets a deterministic but distinct mask.
        fraction_key = int(round(mask_fraction * 1_000_000))
        rng = np.random.default_rng(self.seed + self._epoch.value * 1_000_003 + row * 97 + fraction_key)

        if self._same_pool:
            columns, values = self._finite_sample(row, self.context_columns, rng, observed_count + target_count)
            observed_columns, target_columns = columns[:observed_count], columns[observed_count:]
            beta_observed, beta_target = values[:observed_count], values[observed_count:]
        else:
            observed_columns, beta_observed = self._finite_sample(row, self.context_columns, rng, observed_count)
            target_columns, beta_target = self._finite_sample(row, self.target_columns, rng, target_count)

        observed_prior = self.prior[observed_columns]
        target_prior = self.prior[target_columns]
        observed_logit = beta_to_logit(beta_observed, self.beta_epsilon)
        observed_residual = observed_logit - observed_prior

        return {
            "observed_locus": torch.from_numpy(self.store.get_by_matrix_columns(observed_columns)),
            "observed_residual": torch.from_numpy(observed_residual.astype(np.float32)),
            "observed_valid": torch.ones(observed_count, dtype=torch.bool),
            "target_locus": torch.from_numpy(self.store.get_by_matrix_columns(target_columns)),
            "target_prior_logit": torch.from_numpy(target_prior.astype(np.float32)),
            "target_beta": torch.from_numpy(beta_target.astype(np.float32)),
            "target_matrix_column": torch.from_numpy(target_columns.astype(np.int64)),
            "sample_index": torch.tensor(row, dtype=torch.int64),
            "mask_fraction": torch.tensor(mask_fraction, dtype=torch.float32),
        }


class MaskFractionBatchSampler(Sampler[list[tuple[int, float]]]):
    """Use one mask fraction per batch so variable token counts never require padding."""

    def __init__(self, dataset: MaskingDataset, batch_size: int, *, shuffle: bool):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[tuple[int, float]]]:
        rng = np.random.default_rng(self.dataset.seed + self.epoch * 1_000_003)
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            fraction = float(rng.choice(self.dataset.mask_fractions))
            yield [(int(i), fraction) for i in indices[start : start + self.batch_size]]

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
