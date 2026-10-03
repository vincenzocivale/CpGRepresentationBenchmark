"""Biological validation v2: representation-independent data prep, controls, splits and metric definitions.

Pure numpy/pandas/scipy. Must never import torch, h5py embedding stores, or cpg_repr_benchmark.representations.
"""

from .coordinates import (
    CANONICAL_NAMESPACE,
    harmonize,
    load_benchmark_universe,
    normalize_chrom,
    overlap_audit,
    require_full_universe_coverage,
    verify_cpg_on_reference,
)
from .matching import build_covariates, match_controls, match_pairs
from .metrics import PRIMARY_METRICS
from .pairs import sample_pairs
from .provenance import SourceRecord, sha256_file, write_manifest
from .splits import assert_no_leak, chromosome_blocked_folds, leave_chromosome_out, pair_split_by_chrom

__all__ = [
    "CANONICAL_NAMESPACE", "PRIMARY_METRICS", "SourceRecord", "assert_no_leak", "build_covariates",
    "chromosome_blocked_folds", "harmonize", "leave_chromosome_out", "load_benchmark_universe",
    "match_controls", "match_pairs", "normalize_chrom", "overlap_audit", "pair_split_by_chrom",
    "require_full_universe_coverage", "sample_pairs", "sha256_file", "verify_cpg_on_reference",
    "write_manifest",
]
