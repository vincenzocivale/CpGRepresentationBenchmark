from pathlib import Path

import h5py
import numpy as np

from cpg_repr_benchmark.representations.hdf5_store import HDF5RepresentationStore, inspect_representation_h5


def test_existing_atlas_aliases_are_auto_detected(tmp_path: Path):
    path = tmp_path / "atlas.h5"
    ids = np.asarray([11, 13, 17], dtype=np.int64)
    emb = np.arange(12, dtype=np.float32).reshape(3, 4)
    with h5py.File(path, "w") as h:
        h.create_dataset("ids", data=ids)
        h.create_dataset("emb", data=emb)

    got_ids, dim, id_key, embedding_key = inspect_representation_h5(path)
    assert np.array_equal(got_ids, ids)
    assert dim == 4
    assert id_key == "ids"
    assert embedding_key == "emb"

    matrix_ids = np.asarray([17, 11, 99], dtype=np.int64)
    store = HDF5RepresentationStore(path, matrix_ids)
    assert store.coverage_mask.tolist() == [True, True, False]
    values = store.get_by_matrix_columns(np.asarray([0, 1], dtype=np.int64))
    assert np.array_equal(values, emb[[2, 0]])
    store.close()
