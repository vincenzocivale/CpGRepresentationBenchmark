from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

RepresentationFamily = Literal[
    "functional_annotations",
    "genomic_fm",
    "methylation_fm",
    "control",
]
RepresentationTrack = Literal["native_frozen", "legacy"]


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
    family: RepresentationFamily | str = "control"
    track: RepresentationTrack | str = "native_frozen"
    component: str = "locus_only"
    cpg_namespace: str = "legacy_unspecified"
    reference_build: str = "unspecified"
    coordinate_convention: str = "unspecified"

    def __post_init__(self) -> None:
        if self.track not in {"native_frozen", "legacy"}:
            raise ValueError(f"unsupported representation track: {self.track!r}")
        if self.component != "locus_only":
            raise ValueError(
                "representation benchmark accepts only patient-agnostic locus-only embeddings; "
                f"got component={self.component!r}"
            )


class LocusRepresentationStore(Protocol):
    @property
    def dim(self) -> int: ...

    def get_by_matrix_columns(self, matrix_columns: np.ndarray) -> np.ndarray: ...

    def close(self) -> None: ...
