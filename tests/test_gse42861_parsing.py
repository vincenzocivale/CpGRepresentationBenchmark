import gzip

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts/data/prepare_gse42861.py"
    spec = spec_from_file_location("prepare_gse42861", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_series_matrix(path: Path, lines: list[str]) -> None:
    with gzip.open(path, "wt") as handle:
        handle.writelines(lines)


def test_series_matrix_and_probe_scan_single_pass(tmp_path):
    path = tmp_path / "series.txt.gz"
    lines = [
        '!Sample_title\t"Patient genomic DNA from sample 1"\t"Normal genomic DNA from sample 2"\n',
        '!Sample_geo_accession\t"GSM1"\t"GSM2"\n',
        '!Sample_characteristics_ch1\t"disease state: rheumatoid arthritis"\t"disease state: Normal"\n',
        '!Sample_characteristics_ch1\t"age: 67"\t"age: 49"\n',
        '!Sample_characteristics_ch1\t"gender: f"\t"gender: m"\n',
        "!series_matrix_table_begin\n",
        '"ID_REF"\t"GSM1"\t"GSM2"\n',
        '"cgA"\t0.2\t0.4\n',
        '"cgB"\t0.4\t0.8\n',
        "!series_matrix_table_end\n",
    ]
    _write_series_matrix(path, lines)

    module = _module()
    frame, matrix_names, probe_ids = module.parse_series_matrix_and_probes(path)

    assert frame["subject_id"].tolist() == ["1", "2"]
    assert frame["age"].tolist() == [67, 49]
    assert frame["gender"].tolist() == ["f", "m"]
    assert frame["disease_label"].tolist() == [1, 0]
    assert matrix_names == ["GSM1", "GSM2"]
    assert probe_ids == ["cgA", "cgB"]

    header_lines = module._count_header_lines(path)
    assert header_lines == 6  # lines 1-6, i.e. up to and including !series_matrix_table_begin


def test_unknown_disease_state_raises(tmp_path):
    path = tmp_path / "series.txt.gz"
    lines = [
        '!Sample_title\t"Patient genomic DNA from sample 1"\n',
        '!Sample_geo_accession\t"GSM1"\n',
        '!Sample_characteristics_ch1\t"disease state: lupus"\n',
        "!series_matrix_table_begin\n",
        '"ID_REF"\t"GSM1"\n',
        '"cgA"\t0.2\n',
        "!series_matrix_table_end\n",
    ]
    _write_series_matrix(path, lines)

    module = _module()
    try:
        module.parse_series_matrix_and_probes(path)
    except ValueError as exc:
        assert "disease state" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown disease state")


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
    series = tmp_path / "series.txt.gz"
    lines = [
        '!Sample_title\t"Patient genomic DNA from sample 1"\t"Normal genomic DNA from sample 2"\n',
        '!Sample_geo_accession\t"GSM1"\t"GSM2"\n',
        '!Sample_characteristics_ch1\t"disease state: rheumatoid arthritis"\t"disease state: Normal"\n',
        "!series_matrix_table_begin\n",
        '"ID_REF"\t"GSM1"\t"GSM2"\n',
        '"cgA"\t0.2\t0.4\n',
        '"cgB"\t0.4\t0.8\n',
        '"cgC"\t0.1\tNA\n',
        "!series_matrix_table_end\n",
    ]
    _write_series_matrix(series, lines)

    mapping = module.build_probe_mapping(
        ["cgA", "cgB", "cgC"],
        {"cgA": "1:9", "cgB": "1:9", "cgC": "2:19"},
    )
    header_lines = module._count_header_lines(series)
    out = tmp_path / "out.h5"
    qc = module._write_beta_h5(
        series,
        out,
        mapping,
        ["GSM1", "GSM2"],
        ["GSM1", "GSM2"],
        header_lines,
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
