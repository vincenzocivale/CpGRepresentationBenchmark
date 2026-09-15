from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RepresentationInfo:
    name: str
    store_path: Path
    n_loci: int
    dim: int
    mode: str
    source: str
    id_key: str = "cpg_idx"
    embedding_key: str = "embedding"


class LocusRepresentationStore(Protocol):
    @property
    def dim(self) -> int: ...

    def get_by_matrix_columns(self, matrix_columns: np.ndarray) -> np.ndarray: ...

    def close(self) -> None: ...
