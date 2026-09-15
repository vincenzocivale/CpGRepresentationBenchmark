from __future__ import annotations

from multiprocessing import Value
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cpg_repr_benchmark.representations.base import LocusRepresentationStore
from cpg_repr_benchmark.training.priors import beta_to_logit

from .methylation import read_row_columns


def load_aligned_targets(
    phenotype_path: Path,
    sample_names: list[str],
    *,
    sample_column: str,
    target_column: str,
    task_type: str,
    classes: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    """Align a phenotype table to methylation-matrix rows.

    Returns eligible matrix rows, encoded targets in the same order, and the class
    vocabulary for classification tasks. Duplicated phenotype sample IDs are rejected.
    """
    table = pd.read_parquet(phenotype_path, columns=[sample_column, target_column])
    if table[sample_column].duplicated().any():
        raise ValueError(f"phenotype table contains duplicated {sample_column!r}")
    target_by_sample = table.set_index(sample_column)[target_column]
    values = target_by_sample.reindex(sample_names)
    eligible = ~values.isna().to_numpy()
    rows = np.flatnonzero(eligible).astype(np.int64)
    selected = values.iloc[rows]

    if task_type == "regression":
        return rows, selected.astype(float).to_numpy(dtype=np.float32), None

    if classes is None:
        class_names = sorted(str(x) for x in selected.unique())
    else:
        class_names = [str(x) for x in classes]
    index = {name: i for i, name in enumerate(class_names)}
    encoded = np.asarray([index.get(str(x), -1) for x in selected], dtype=np.int64)
    if (encoded < 0).any():
        unknown = sorted({str(x) for x, y in zip(selected, encoded) if y < 0})
        raise ValueError(f"phenotype contains labels absent from configured classes: {unknown[:10]}")
    if task_type == "binary" and len(class_names) != 2:
        raise ValueError(f"binary task requires exactly 2 classes, got {class_names}")
    if task_type == "multiclass" and len(class_names) < 2:
        raise ValueError("multiclass task requires at least 2 classes")
    return rows, encoded, class_names


class MethylomeTaskDataset(Dataset):
    """Patient-level dataset with deterministic CpG subsampling and masking.

    `panel_size` is the pre-mask CpG budget. A mask fraction of 0.90 therefore retains
    approximately 10% of the panel. One mask fraction is used per batch through the
    existing MaskFractionBatchSampler, so token counts remain rectangular without padding.
    """

    def __init__(
        self,
        *,
        methylation_h5: Path,
        representation_store: LocusRepresentationStore,
        prior_logit_full: np.ndarray,
        sample_indices: np.ndarray,
        targets: np.ndarray,
        candidate_columns: np.ndarray,
        panel_size: int,
        mask_fractions: Sequence[float],
        seed: int = 17,
        beta_epsilon: float = 1e-4,
    ) -> None:
        self.methylation_h5 = Path(methylation_h5)
        self.store = representation_store
        self.prior = np.asarray(prior_logit_full, dtype=np.float32)
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.targets = np.asarray(targets)
        self.candidate_columns = np.asarray(candidate_columns, dtype=np.int64)
        self.panel_size = int(panel_size)
        self.mask_fractions = tuple(float(x) for x in mask_fractions)
        self.seed = int(seed)
        self.beta_epsilon = float(beta_epsilon)
        if len(self.sample_indices) != len(self.targets):
            raise ValueError("sample_indices and targets must have the same length")
        if self.panel_size < 1 or self.panel_size > len(self.candidate_columns):
            raise ValueError("panel_size must be between 1 and the candidate CpG count")
        if not self.mask_fractions:
            raise ValueError("mask_fractions cannot be empty")
        for fraction in self.mask_fractions:
            if not 0.0 <= fraction < 1.0:
                raise ValueError("classification mask fractions must satisfy 0 <= f < 1")
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

    def _sample_finite(
        self,
        row: int,
        rng: np.random.Generator,
        count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidate_count = min(
            len(self.candidate_columns),
            max(count + 64, int(np.ceil(count / 0.75))),
        )
        for _ in range(5):
            positions = rng.choice(len(self.candidate_columns), size=candidate_count, replace=False)
            columns = self.candidate_columns[positions]
            values = read_row_columns(self._beta(), row, columns)
            finite = np.flatnonzero(np.isfinite(values))
            if len(finite) >= count:
                chosen = rng.choice(finite, size=count, replace=False)
                return columns[chosen], values[chosen]
            candidate_count = min(
                len(self.candidate_columns),
                max(candidate_count + 1, int(candidate_count * 1.5)),
            )
        values = read_row_columns(self._beta(), row, self.candidate_columns)
        finite = np.flatnonzero(np.isfinite(values))
        if len(finite) < count:
            raise ValueError(f"sample row {row} has only {len(finite)} finite CpGs; need {count}")
        chosen = rng.choice(finite, size=count, replace=False)
        return self.candidate_columns[chosen], values[chosen]

    def __getitem__(self, item: int | tuple[int, float]) -> dict[str, torch.Tensor]:
        if isinstance(item, tuple):
            item, mask_fraction = item
        else:
            mask_fraction = self.mask_fractions[0]
        item = int(item)
        row = int(self.sample_indices[item])
        mask_fraction = float(mask_fraction)
        observed_count = max(1, int(round(self.panel_size * (1.0 - mask_fraction))))
        fraction_key = int(round(mask_fraction * 1_000_000))
        rng = np.random.default_rng(
            self.seed + self._epoch.value * 1_000_003 + row * 97 + fraction_key
        )
        columns, beta = self._sample_finite(row, rng, observed_count)
        residual = beta_to_logit(beta, self.beta_epsilon) - self.prior[columns]
        target = self.targets[item]
        target_tensor = (
            torch.tensor(float(target), dtype=torch.float32)
            if np.issubdtype(self.targets.dtype, np.floating)
            else torch.tensor(int(target), dtype=torch.long)
        )
        return {
            "observed_locus": torch.from_numpy(self.store.get_by_matrix_columns(columns)),
            "observed_residual": torch.from_numpy(residual.astype(np.float32)),
            "observed_valid": torch.ones(observed_count, dtype=torch.bool),
            "target": target_tensor,
            "sample_index": torch.tensor(row, dtype=torch.int64),
            "mask_fraction": torch.tensor(mask_fraction, dtype=torch.float32),
        }
