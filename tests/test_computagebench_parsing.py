from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _module():
    path = Path(__file__).parents[1] / "scripts/data/prepare_computagebench.py"
    spec = spec_from_file_location("prepare_computagebench", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parquet_probe_ids_reads_persisted_index_only(tmp_path):
    path = tmp_path / "computage_bench_data_GSE1.parquet"
    pd.DataFrame({"GSM1": [0.1, 0.2], "GSM2": [0.3, 0.4]}, index=["cgA", "cgB"]).to_parquet(path)
    assert _module().parquet_probe_ids(path) == ["cgA", "cgB"]


def test_probe_mapping_keeps_coordinate_identity_and_audits_missing():
    module = _module()
    crosswalk = pd.DataFrame({"probe_id": ["cgA", "cgB"], "chr": ["1", "1"], "pos": [10, 10]})
    mapping = module.build_probe_mapping(["cgA", "cgB", "cgMissing"], module.load_crosswalk_data(crosswalk))
    assert mapping.loc[0, "cpg_idx"] == mapping.loc[1, "cpg_idx"]
    assert mapping.loc[0, "aggregation_group_size"] == 2
    assert mapping.loc[2, "mapping_status"] == "missing_crosswalk"


def test_prepare_writes_canonical_h5_for_a_study_snapshot(tmp_path, monkeypatch):
    module = _module()
    snapshot = tmp_path / "snapshot"
    matrix_dir = snapshot / "data" / "benchmark"
    matrix_dir.mkdir(parents=True)
    pd.DataFrame({"GSM1": [0.1, 0.3], "GSM2": [0.2, 0.4]}, index=["cgA", "cgB"]).to_parquet(
        matrix_dir / "computage_bench_data_GSE1.parquet"
    )
    pd.DataFrame(
        {"DatasetID": ["GSE1", "GSE1"], "Age": [30, 40], "Class": ["HC", "AAC"]},
        index=["GSM1", "GSM2"],
    ).to_csv(snapshot / "computage_bench_meta.tsv", sep="\t")
    crosswalk = tmp_path / "crosswalk.parquet"
    pd.DataFrame({"probe_id": ["cgA", "cgB"], "chr": ["1", "2"], "pos": [10, 20]}).to_parquet(crosswalk)
    output = tmp_path / "processed"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_computagebench.py",
            "--snapshot-dir", str(snapshot),
            "--output-dir", str(output),
            "--probe-crosswalk", str(crosswalk),
        ],
    )
    module.main()
    with h5py.File(output / "methylation.h5", "r") as handle:
        assert handle["beta"].shape == (2, 2)
        assert np.allclose(handle["beta"][:], [[0.1, 0.3], [0.2, 0.4]])
        assert handle["sample_name"].asstr()[:].tolist() == ["GSE1:GSM1", "GSE1:GSM2"]
        assert handle.attrs["cpg_namespace"] == "grch38_cpg_cytosine_1based_v1"
    phenotype = pd.read_parquet(output / "phenotypes.parquet")
    assert phenotype["is_healthy_control"].tolist() == [1, 0]
    assert phenotype["is_aging_accelerating_condition"].tolist() == [0, 1]
