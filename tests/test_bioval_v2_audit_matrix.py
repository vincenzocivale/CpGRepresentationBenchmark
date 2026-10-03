"""The anti-circularity matrix must be well-formed and complete (no biological data needed)."""

from pathlib import Path

import pytest
import yaml

CFG = Path(__file__).resolve().parents[1] / "configs" / "biological_validation_v2" / "anti_circularity.yaml"
AXES = {"external_methylation_programs", "3d_genome", "regulatory_activity", "replication_domains"}


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(CFG.read_text())


def _expected_class(cells):
    vals = set(cells.values())
    return "circular" if "C" in vals else "partially_related" if "P" in vals else "independent"


def test_cells_complete_and_valid(cfg):
    inputs = set(cfg["inputs"])
    assert {"histone_chip", "tf_chip", "ctcf_cohesin_chip", "dnase", "ccre_v4", "gencode_v50", "cpg_islands",
            "sequence_fm"} <= inputs
    for name, t in cfg["targets"].items():
        assert set(t["cells"]) == inputs, name
        assert set(t["cells"].values()) <= set(cfg["codes"]), name
        assert t["status"] in cfg["statuses"], name
        assert t["justification"].strip(), name


def test_every_axis_has_exactly_one_primary_noncircular(cfg):
    for ax in AXES:
        prim = [n for n, t in cfg["targets"].items() if t["axis"] == ax and t["primary_endpoint"]]
        assert len(prim) == 1, ax
        t = cfg["targets"][prim[0]]
        assert t["status"] == "primary" and t["holm_family"] is True
        assert _expected_class(t["cells"]) != "circular"


def test_circular_targets_never_main(cfg):
    for name, t in cfg["targets"].items():
        if _expected_class(t["cells"]) == "circular":
            assert t["status"] == "rejected_circular" and not t["primary_endpoint"], name
        if t["status"] == "rejected_circular":
            assert _expected_class(t["cells"]) == "circular", name
    for required in ("cpg_island_context", "gene_region", "ccre_class", "tss_distance", "chromhmm_same_histone_marks"):
        assert _expected_class(cfg["targets"][required]["cells"]) == "circular"


def test_arms_reference_known_inputs(cfg):
    for arm, a in cfg["arms"].items():
        assert set(a["inputs"]) <= set(cfg["inputs"]), arm


def test_pmd_flagged_methylation_derived(cfg):
    assert cfg["targets"]["pmd_decato2020"]["methylation_derived"] is True
    assert cfg["targets"]["pmd_decato2020"]["status"] == "exploratory"


def test_status_primary_consistency_and_holm_family(cfg):
    t = cfg["targets"]
    for name, x in t.items():
        assert x["primary_endpoint"] == (x["status"] == "primary"), name
        assert x["holm_family"] == x["primary_endpoint"], name
    assert sum(x["holm_family"] for x in t.values()) == 4
    assert t["loyfer_profile_similarity"]["status"] == "primary"
    assert t["loyfer_marker_knn"]["status"] == "secondary"
    assert t["fourdn_contact_inter"]["status"] == "exploratory"
    assert t["fourdn_compartment_ab"]["status"] == "secondary"

