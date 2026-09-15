from pathlib import Path

import h5py
import numpy as np

from cpg_repr_benchmark.data.coordinates import encode_cpg_id
from cpg_repr_benchmark.representations.materialization import (
    load_protocol_coordinates,
    materialize_coordinate_representation,
)


class _FakeProvider:
    name = "fake"
    dim = 3

    def __init__(self):
        self.calls = 0

    def encode(self, cpg_idx, chrom, pos):
        self.calls += 1
        return np.stack(
            [
                np.asarray(pos, dtype=np.float32),
                np.asarray(cpg_idx % 997, dtype=np.float32),
                np.arange(len(cpg_idx), dtype=np.float32),
            ],
            axis=1,
        )

    def metadata(self):
        return {"checkpoint": "synthetic"}

    def close(self):
        pass


def test_protocol_filter_and_materialization(tmp_path: Path):
    ids = np.asarray(
        [
            encode_cpg_id("chr1", 100),
            encode_cpg_id("chr2", 200),
            encode_cpg_id("chr1", 300),
        ],
        dtype=np.int64,
    )
    protocol = tmp_path / "protocol.npz"
    np.savez_compressed(protocol, cpg_idx=ids)

    subset_ids, chrom, pos = load_protocol_coordinates(protocol, chromosomes=["1"])
    assert subset_ids.tolist() == [ids[0], ids[2]]
    assert chrom.tolist() == ["chr1", "chr1"]
    assert pos.tolist() == [100, 300]

    provider = _FakeProvider()
    output = tmp_path / "repr.h5"
    result = materialize_coordinate_representation(
        provider,
        subset_ids,
        chrom,
        pos,
        output,
        batch_size=1,
    )
    assert result["status"] == "built"
    assert provider.calls == 2

    with h5py.File(output, "r") as handle:
        assert set(handle.keys()) == {"chrom", "cpg_idx", "embedding", "pos"}
        assert handle["embedding"].shape == (2, 3)
        assert handle.attrs["cpg_namespace"] == "grch38_cpg_cytosine_1based_v1"
        assert np.asarray(handle["cpg_idx"][:], dtype=np.int64).tolist() == subset_ids.tolist()

    cached = materialize_coordinate_representation(
        provider,
        subset_ids,
        chrom,
        pos,
        output,
        batch_size=1,
    )
    assert cached["status"] == "cached"
    assert provider.calls == 2
