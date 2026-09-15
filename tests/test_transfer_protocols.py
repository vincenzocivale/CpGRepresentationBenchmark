from pathlib import Path

import numpy as np

from cpg_repr_benchmark.data.transfer_protocols import (
    split_external_locus_sets,
    write_external_locus_protocols,
)


def test_split_external_locus_sets_preserves_dataset_order():
    dataset = np.asarray([50, 10, 40, 20, 30], dtype=np.int64)
    proxy = np.asarray([10, 20, 99], dtype=np.int64)
    result = split_external_locus_sets(dataset, proxy)
    assert result["all"].tolist() == [50, 10, 40, 20, 30]
    assert result["shared"].tolist() == [10, 20]
    assert result["external_locus"].tolist() == [50, 40, 30]


def test_write_external_locus_protocols(tmp_path: Path):
    dataset = np.asarray([11, 12, 13, 14], dtype=np.int64)
    proxy = np.asarray([12, 14, 15], dtype=np.int64)
    manifest = write_external_locus_protocols(
        dataset,
        proxy,
        tmp_path,
        dataset_source="external",
        proxy_source="proxy",
    )
    assert manifest["n_dataset_loci"] == 4
    assert manifest["n_shared_loci"] == 2
    assert manifest["n_external_locus"] == 2
    shared = np.load(tmp_path / "shared.npz")
    external = np.load(tmp_path / "external_locus.npz")
    assert shared["cpg_idx"].tolist() == [12, 14]
    assert shared["matrix_columns"].tolist() == [1, 3]
    assert external["cpg_idx"].tolist() == [11, 13]
    assert external["matrix_columns"].tolist() == [0, 2]
