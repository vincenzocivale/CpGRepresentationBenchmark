import numpy as np

from cpg_repr_benchmark.representations.online import OnlineRepresentationStore


class Encoder:
    dim = 2

    def __init__(self):
        self.calls = []

    def encode(self, ids):
        self.calls.append(ids.copy())
        x = ids.astype(np.float32)
        return np.stack([x, -x], axis=1)


def test_online_store_generates_and_caches():
    enc = Encoder()
    store = OnlineRepresentationStore(enc, np.array([10, 20, 30, 40]), cache_size=4)
    first = store.get_by_matrix_columns(np.array([0, 2]))
    second = store.get_by_matrix_columns(np.array([2, 0]))
    np.testing.assert_array_equal(first, np.array([[10, -10], [30, -30]], dtype=np.float32))
    np.testing.assert_array_equal(second, np.array([[30, -30], [10, -10]], dtype=np.float32))
    assert len(enc.calls) == 1
