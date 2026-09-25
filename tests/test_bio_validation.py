import numpy as np
import pandas as pd

from cpg_repr_benchmark.bio_validation.annotations import CpGAnnotations
from cpg_repr_benchmark.bio_validation.probes import bio_validation_report, locality_probe


def test_bio_validation_report_recovers_separable_context_and_set():
    rng = np.random.default_rng(17)
    n = 400
    cpg_idx = np.arange(n, dtype=np.int64)
    embedding = rng.normal(size=(n, 10))

    context_labels = np.where(embedding[:, 0] > 0, "island", "open_sea")
    genomic_context = pd.Series(context_labels, index=cpg_idx)

    known_member = embedding[:, 1] > np.quantile(embedding[:, 1], 0.85)
    known_sets = {"toy_clock": cpg_idx[known_member]}

    coefficients = {"toy_clock": pd.Series(embedding[:, 2] * 3.0 + 1.0, index=cpg_idx)}

    annotations = CpGAnnotations(
        genomic_context=genomic_context, known_sets=known_sets, coefficients=coefficients
    )
    report = bio_validation_report(embedding=embedding, cpg_idx=cpg_idx, annotations=annotations, seed=17)

    assert report["genomic_context"]["island"]["auc"] > 0.9
    assert report["known_cpg_sets"]["toy_clock"]["auc"] > 0.9
    assert report["clock_coefficients"]["toy_clock"]["pearson"] > 0.9


def test_bio_validation_report_handles_too_few_overlapping_coefficients():
    rng = np.random.default_rng(17)
    n = 50
    cpg_idx = np.arange(n, dtype=np.int64)
    embedding = rng.normal(size=(n, 10))
    coefficients = {"tiny_clock": pd.Series([0.5, -1.0], index=[1000, 1001])}

    annotations = CpGAnnotations(
        genomic_context=pd.Series(dtype=object), known_sets={}, coefficients=coefficients
    )
    report = bio_validation_report(embedding=embedding, cpg_idx=cpg_idx, annotations=annotations, seed=17)

    assert "error" in report["clock_coefficients"]["tiny_clock"]


def _synthetic_registry(cpg_idx: np.ndarray, pos: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"cpg_idx": cpg_idx, "chr": "chr1", "pos": pos})


def test_locality_probe_scores_high_when_embedding_encodes_genomic_position():
    rng = np.random.default_rng(17)
    n = 600
    cpg_idx = np.arange(n, dtype=np.int64)
    pos = np.sort(rng.choice(np.arange(n * 1000), size=n, replace=False)).astype(np.float64)
    # embedding is literally the (normalized) genomic position -> neighbors in embedding space
    # are exactly the genomically closest CpGs.
    embedding = np.stack([pos / pos.max(), np.zeros(n)], axis=1)
    registry = _synthetic_registry(cpg_idx, pos)

    result = locality_probe(embedding, cpg_idx, registry, k=5, distance_kb=5.0, sample_size=300, seed=17)

    assert result["neighbor_hit_rate"] > 0.95
    assert result["neighbor_hit_rate"] > result["null_hit_rate"]
    assert result["enrichment"] > 20.0


def test_locality_probe_near_baseline_for_random_embedding():
    rng = np.random.default_rng(17)
    n = 600
    cpg_idx = np.arange(n, dtype=np.int64)
    pos = np.sort(rng.choice(np.arange(n * 1000), size=n, replace=False)).astype(np.float64)
    embedding = rng.normal(size=(n, 16))
    registry = _synthetic_registry(cpg_idx, pos)

    result = locality_probe(embedding, cpg_idx, registry, k=5, distance_kb=5.0, sample_size=300, seed=17)

    assert result["enrichment"] < 3.0


def test_locality_probe_reports_error_when_too_few_coordinates():
    cpg_idx = np.arange(3, dtype=np.int64)
    embedding = np.zeros((3, 4))
    registry = _synthetic_registry(cpg_idx, np.array([0.0, 1000.0, 2000.0]))

    result = locality_probe(embedding, cpg_idx, registry, k=5)

    assert "error" in result
