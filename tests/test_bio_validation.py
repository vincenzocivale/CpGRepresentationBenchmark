import numpy as np
import pandas as pd

from cpg_repr_benchmark.bio_validation.annotations import CpGAnnotations
from cpg_repr_benchmark.bio_validation.probes import bio_validation_report


def test_bio_validation_report_recovers_separable_context_and_set():
    rng = np.random.default_rng(17)
    n = 400
    cpg_idx = np.arange(n, dtype=np.int64)
    embedding = rng.normal(size=(n, 10))

    context_labels = np.where(embedding[:, 0] > 0, "island", "open_sea")
    genomic_context = pd.Series(context_labels, index=cpg_idx)

    known_member = embedding[:, 1] > np.quantile(embedding[:, 1], 0.85)
    known_sets = {"toy_clock": cpg_idx[known_member]}

    annotations = CpGAnnotations(genomic_context=genomic_context, known_sets=known_sets)
    report = bio_validation_report(embedding=embedding, cpg_idx=cpg_idx, annotations=annotations, seed=17)

    assert report["genomic_context"]["island"]["auc"] > 0.9
    assert report["known_cpg_sets"]["toy_clock"]["auc"] > 0.9
