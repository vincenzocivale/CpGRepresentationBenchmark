from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

REFERENCE_BUILD = "GRCh38"
COORDINATE_CONVENTION = "1-based position of CpG cytosine"
CPG_NAMESPACE = "grch38_cpg_cytosine_1based_v1"
CPG_ID_STRIDE = 1_000_000_000

_CHROMS = [*(f"chr{i}" for i in range(1, 23)), "chrX", "chrY", "chrM"]
CHROM_TO_CODE = {chrom: i + 1 for i, chrom in enumerate(_CHROMS)}
CODE_TO_CHROM = {code: chrom for chrom, code in CHROM_TO_CODE.items()}


def normalize_chromosome(chrom: str) -> str:
    value = str(chrom).strip()
    if not value:
        raise ValueError("empty chromosome")
    if value.lower().startswith("chr"):
        value = value[3:]
    upper = value.upper()
    if upper in {"M", "MT"}:
        return "chrM"
    if upper in {"X", "Y"}:
        return f"chr{upper}"
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError(f"unsupported chromosome label: {chrom!r}") from exc
    if not 1 <= number <= 22:
        raise ValueError(f"unsupported chromosome label: {chrom!r}")
    return f"chr{number}"


def encode_cpg_id(chrom: str, pos: int) -> int:
    """Encode a GRCh38 CpG cytosine coordinate as a deterministic int64 ID.

    The coordinate, not a source-specific probe or row number, is the biological
    identity.  `pos` is the 1-based position of the cytosine in the CpG.
    """
    canonical = normalize_chromosome(chrom)
    position = int(pos)
    if not 1 <= position < CPG_ID_STRIDE:
        raise ValueError(f"position out of supported range: {position}")
    return CHROM_TO_CODE[canonical] * CPG_ID_STRIDE + position


def decode_cpg_id(cpg_idx: int) -> tuple[str, int]:
    value = int(cpg_idx)
    code, position = divmod(value, CPG_ID_STRIDE)
    if code not in CODE_TO_CHROM or position < 1:
        raise ValueError(f"not a valid {CPG_NAMESPACE} id: {value}")
    return CODE_TO_CHROM[code], position


def encode_many(chrom: Iterable[str], pos: Iterable[int]) -> np.ndarray:
    return np.fromiter((encode_cpg_id(c, p) for c, p in zip(chrom, pos)), dtype=np.int64)


def chromosome_from_ids(cpg_idx: np.ndarray) -> np.ndarray:
    ids = np.asarray(cpg_idx, dtype=np.int64)
    codes = ids // CPG_ID_STRIDE
    invalid = ~np.isin(codes, np.fromiter(CODE_TO_CHROM, dtype=np.int64))
    if invalid.any():
        raise ValueError(f"array contains {int(invalid.sum())} invalid canonical CpG IDs")
    return np.asarray([CODE_TO_CHROM[int(code)] for code in codes], dtype=object)


@dataclass(frozen=True)
class CpGCoordinate:
    chrom: str
    pos: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "chrom", normalize_chromosome(self.chrom))
        object.__setattr__(self, "pos", int(self.pos))
        encode_cpg_id(self.chrom, self.pos)

    @property
    def cpg_idx(self) -> int:
        return encode_cpg_id(self.chrom, self.pos)

    @property
    def key(self) -> str:
        return f"{self.chrom}:{self.pos}"
