import gzip
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts/data/prepare_gse147221.py"
    spec = spec_from_file_location("prepare_gse147221", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phenotypes_exclude_technical_and_geo_low_quality_samples(tmp_path):
    series = tmp_path / "series.txt.gz"
    lines = [
        '!Sample_title\t"A_R01C01: genomic DNA from whole Blood"\t"B_R01C01: Meth Control"\t"C_R01C01: genomic DNA from whole Blood"\n',
        '!Sample_geo_accession\t"GSM1"\t"GSM2"\t"GSM3"\n',
        '!Sample_description\t""\t""\t"Sample was excluded from the final analysis due to low quality"\n',
        '!Sample_characteristics_ch1\t"status: Case"\t"status: NA"\t"status: Control"\n',
        '!Sample_characteristics_ch1\t"age: 32.5"\t"age: NA"\t"age: 41"\n',
        '!series_matrix_table_begin\n',
    ]
    with gzip.open(series, "wt") as handle:
        handle.writelines(lines)

    frame = _module().parse_phenotypes(series)

    assert frame["array_id"].tolist() == ["A_R01C01", "B_R01C01", "C_R01C01"]
    assert frame["disease_label"].iloc[[0, 2]].tolist() == [1, 0]
    assert frame["disease_label"].isna().iloc[1]
    assert frame["included"].tolist() == [True, False, False]
    assert frame.loc[1, "exclusion_reason"] == "technical_or_unlabelled_control"
    assert frame.loc[2, "exclusion_reason"] == "geo_low_quality"


def test_signal_column_selection_keeps_only_eligible_beta_columns(tmp_path):
    signals = tmp_path / "signals.csv.gz"
    with gzip.open(signals, "wt") as handle:
        handle.write('"","A","A_Detection_Pval","B","B_Detection_Pval"\n')
        handle.write('"cg1",0.1,0,0.2,0\n')

    beta, pval = _module()._signal_columns(signals, {"B"})

    assert beta == ["B"]
    assert pval == ["B_Detection_Pval"]


def test_writer_masks_failed_detection_and_averages_duplicate_coordinates(tmp_path):
    import h5py
    import numpy as np

    signals = tmp_path / "signals.csv.gz"
    with gzip.open(signals, "wt") as handle:
        handle.write('"","A","A_Detection_Pval","B","B_Detection_Pval"\n')
        handle.write('"cg1",0.2,0,0.8,0\n')
        handle.write('"cg2",0.4,0,0.4,0.02\n')
        handle.write('"cg3",0.1,0,0.9,0\n')
    module = _module()
    mapping = module.build_probe_mapping(
        ["cg1", "cg2", "cg3"], {"cg1": "1:9", "cg2": "1:9", "cg3": "2:19"}
    )
    output = tmp_path / "out.h5"
    module.write_h5(signals, output, mapping, ["B", "A"], 0.01, 2)

    with h5py.File(output, "r") as handle:
        assert handle["sample_name"][:].astype(str).tolist() == ["B", "A"]
        observed = handle["beta"][:]
    assert np.allclose(observed[:, 0], [0.8, 0.3])
    assert np.allclose(observed[:, 1], [0.9, 0.1])
