import numpy as np

from cpg_repr_benchmark.proxy.training import split_proxy_train_validation


def test_proxy_split_never_introduces_downstream_heldout_loci():
    cpg_ids = np.arange(100, 120, dtype=np.int64)
    downstream_train = np.arange(15, dtype=np.int64)
    downstream_heldout = np.arange(15, 20, dtype=np.int64)

    proxy_train, proxy_validation = split_proxy_train_validation(
        downstream_train,
        cpg_ids,
        seed=17,
        validation_fraction=0.2,
    )

    fitted = np.concatenate([proxy_train, proxy_validation])
    assert set(fitted.tolist()) == set(downstream_train.tolist())
    assert np.intersect1d(fitted, downstream_heldout).size == 0
    assert np.intersect1d(proxy_train, proxy_validation).size == 0
