import numpy as np

from cpg_repr_benchmark.data.universe import resolve_ids_to_columns, write_common_universe_protocol


def test_common_universe_preserves_dataset_order(tmp_path):
    dataset = np.array([30, 10, 20, 40], dtype=np.int64)
    reps = {
        "a": np.array([10, 20, 30], dtype=np.int64),
        "b": np.array([20, 30, 50], dtype=np.int64),
    }
    output = tmp_path / "common.npz"
    summary = write_common_universe_protocol(dataset, reps, output)
    saved = np.load(output)
    assert saved["cpg_idx"].tolist() == [30, 20]
    assert saved["matrix_columns"].tolist() == [0, 2]
    assert summary["common_loci"] == 2
    assert resolve_ids_to_columns(dataset, np.array([40, 10])).tolist() == [3, 1]
