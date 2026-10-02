import pytest

from cpg_repr_benchmark.experiments.guards import enforce_patient_view


def test_guard_is_noop_without_requirement():
    enforce_patient_view({"evaluation": {"patient_view": "test"}})
    enforce_patient_view({"evaluation": {}})


def test_guard_accepts_validation():
    enforce_patient_view({"evaluation": {"require_patient_view": "validation", "patient_view": "validation"}})


@pytest.mark.parametrize("view", ["test", None, "train"])
def test_guard_rejects_non_validation(view):
    evaluation = {"require_patient_view": "validation"}
    if view is not None:
        evaluation["patient_view"] = view
    with pytest.raises(PermissionError):
        enforce_patient_view({"evaluation": evaluation})


def test_guard_rejects_requiring_test():
    with pytest.raises(ValueError):
        enforce_patient_view({"evaluation": {"require_patient_view": "test", "patient_view": "test"}})
