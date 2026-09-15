import gzip

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts/data/prepare_gse40279.py"
    spec = spec_from_file_location("prepare_gse40279", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_series_matrix_parser_extracts_characteristics(tmp_path):
    path = tmp_path / "series.txt.gz"
    lines = [
        '!Sample_title\t"age 67y 1001"\t"age 89y 1002"\n',
        '!Sample_geo_accession\t"GSM1"\t"GSM2"\n',
        '!Sample_characteristics_ch1\t"age (y): 67"\t"age (y): 89"\n',
        '!Sample_characteristics_ch1\t"gender: F"\t"gender: M"\n',
        '!Sample_characteristics_ch1\t"tissue: whole blood"\t"tissue: whole blood"\n',
    ]
    with gzip.open(path, "wt") as handle:
        handle.writelines(lines)
    frame = _module().parse_series_matrix(path)
    assert frame["subject_id"].tolist() == ["1001", "1002"]
    assert frame["age"].tolist() == [67, 89]
    assert frame["gender"].tolist() == ["F", "M"]


def test_probe_mapping_collapses_duplicate_coordinates_and_marks_missing():
    module = _module()
    frame = module.build_probe_mapping(
        ["cgA", "cgB", "cgMissing"],
        {"cgA": "1:9", "cgB": "1:9"},
    )
    assert frame.loc[0, "cpg_idx"] == frame.loc[1, "cpg_idx"]
    assert frame.loc[0, "aggregation_group_size"] == 2
    assert frame.loc[1, "mapping_status"] == "mapped_duplicate_coordinate"
    assert frame.loc[2, "mapping_status"] == "missing_illumina_crosswalk"


def test_beta_writer_aggregates_duplicate_probe_rows(tmp_path):
    import h5py
    import numpy as np

    module = _module()
    beta = tmp_path / "beta.txt.gz"
    with gzip.open(beta, "wt") as handle:
        handle.write("ID_REF\tX1001\tX1002\n")
        handle.write("cgA\t0.2\t0.4\n")
        handle.write("cgB\t0.4\t0.8\n")
        handle.write("cgC\t0.1\tNA\n")
    mapping = module.build_probe_mapping(
        ["cgA", "cgB", "cgC"],
        {"cgA": "1:9", "cgB": "1:9", "cgC": "2:19"},
    )
    out = tmp_path / "out.h5"
    qc = module._write_beta_h5(
        beta,
        out,
        mapping,
        ["GSM1", "GSM2"],
        ["X1001", "X1002"],
        chunk_rows=2,
    )
    with h5py.File(out, "r") as h:
        observed = h["beta"][:]
        assert h.attrs["cpg_namespace"] == "grch38_cpg_cytosine_1based_v1"
    assert observed.shape == (2, 2)
    assert np.allclose(observed[:, 0], [0.3, 0.6])
    assert observed[0, 1] == np.float32(0.1)
    assert np.isnan(observed[1, 1])
    assert qc["n_loci"] == 2
