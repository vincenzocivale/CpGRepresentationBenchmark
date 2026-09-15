from pathlib import Path

import pytest

from cpg_repr_benchmark.representations.base import RepresentationInfo


def test_representation_info_requires_locus_only_component():
    with pytest.raises(ValueError):
        RepresentationInfo(
            name="bad",
            store_path=Path("x.h5"),
            n_loci=10,
            dim=32,
            mode="precomputed",
            source="test",
            family="methylation_fm",
            track="native_frozen",
            component="patient_contextualized",
        )


def test_representation_tracks_are_explicit():
    info = RepresentationInfo(
        name="ntv3",
        store_path=Path("x.h5"),
        n_loci=10,
        dim=256,
        mode="precomputed",
        source="test",
        family="genomic_fm",
        track="proxy_aligned",
    )
    assert info.track == "proxy_aligned"
