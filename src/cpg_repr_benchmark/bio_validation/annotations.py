from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CpGAnnotations:
    """External, ENCODE-independent biological annotations keyed by coordinate `cpg_idx`.

    `genomic_context[cpg_idx] -> label` (e.g. island/shore/shelf/open_sea, or
    promoter/gene_body/intergenic) is a categorical column, used as a classification target
    for the CpG-locus embedding. `known_sets` maps a literature-curated set name (e.g.
    "horvath_clock", "ewas_catalog_age") to the `cpg_idx` values it contains; each set is
    evaluated as its own binary membership classification target.

    `coefficients` maps a set name (e.g. "horvath_clock") to a `cpg_idx -> weight` Series —
    the actual elastic-net coefficient of that CpG in a published clock, rather than mere
    membership. Evaluated as a regression target: it asks whether the embedding predicts *how
    much* a CpG matters to the clock, a stricter test than the binary membership probe over the
    same set. A `coefficients` entry is independent of whether a matching `known_sets` entry
    exists (both are optional per set name).

    Source files are documented in `docs/EMBEDDING_EVALUATION.md`; none are bundled with the
    repo (see `data/bio_annotations/README.md`), since provenance/licensing must be recorded
    per source before use.
    """

    genomic_context: pd.Series
    known_sets: dict[str, np.ndarray]
    coefficients: dict[str, pd.Series]


def load_cpg_annotations(
    *,
    genomic_context_parquet: Path | None,
    known_sets_dir: Path | None,
    coefficients_dir: Path | None = None,
) -> CpGAnnotations:
    """Load annotation files from disk.

    `genomic_context_parquet` must have columns `cpg_idx` (int64) and `context` (str).
    `known_sets_dir` must contain one `<set_name>.npy` int64 array of `cpg_idx` values per
    known CpG set (e.g. `horvath_clock.npy`). `coefficients_dir` must contain one
    `<set_name>_coefficients.parquet` per set with columns `cpg_idx` (int64) and `coefficient`
    (float) — e.g. `horvath_clock_coefficients.parquet` — for sets that have a published
    per-CpG weight, not just membership. Any argument may be omitted to run a partial
    validation (only the sources that are available).
    """
    genomic_context = pd.Series(dtype=object)
    if genomic_context_parquet is not None:
        table = pd.read_parquet(genomic_context_parquet)
        if not {"cpg_idx", "context"}.issubset(table.columns):
            raise ValueError("genomic context table must have 'cpg_idx' and 'context' columns")
        genomic_context = table.set_index("cpg_idx")["context"]

    known_sets: dict[str, np.ndarray] = {}
    if known_sets_dir is not None:
        known_sets_dir = Path(known_sets_dir)
        for path in sorted(known_sets_dir.glob("*.npy")):
            known_sets[path.stem] = np.load(path).astype(np.int64)

    coefficients: dict[str, pd.Series] = {}
    if coefficients_dir is not None:
        coefficients_dir = Path(coefficients_dir)
        for path in sorted(coefficients_dir.glob("*_coefficients.parquet")):
            table = pd.read_parquet(path)
            if not {"cpg_idx", "coefficient"}.issubset(table.columns):
                raise ValueError(f"{path} must have 'cpg_idx' and 'coefficient' columns")
            set_name = path.stem.removesuffix("_coefficients")
            coefficients[set_name] = table.set_index("cpg_idx")["coefficient"]

    if genomic_context.empty and not known_sets and not coefficients:
        raise ValueError("no biological annotations were loaded: provide at least one source")
    return CpGAnnotations(genomic_context=genomic_context, known_sets=known_sets, coefficients=coefficients)
