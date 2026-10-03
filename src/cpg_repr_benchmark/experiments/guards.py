"""Opt-in guards for experiments that must never touch held-out test patients."""
from __future__ import annotations

from typing import Any


def enforce_patient_view(cfg: dict[str, Any]) -> None:
    """Fail loudly unless ``evaluation.patient_view`` equals ``evaluation.require_patient_view``.

    Configs without ``require_patient_view`` are unaffected (existing callers keep their behavior).
    The effective view is compared without applying the runner's implicit ``test`` default, so an
    omitted ``patient_view`` fails as well. The required value ``test`` is itself rejected.
    """
    evaluation = cfg.get("evaluation", {})
    if "require_patient_view" not in evaluation:
        return
    required = evaluation["require_patient_view"]
    if required != "validation":
        raise ValueError(f"evaluation.require_patient_view must be 'validation', got {required!r}")
    actual = evaluation.get("patient_view")
    if actual != "validation":
        raise PermissionError(
            f"GUARD: evaluation.patient_view={actual!r} but this protocol permits 'validation' only; "
            "test patients must not be read or evaluated")


def require_test_authorization(matrix_spec: dict[str, Any]) -> None:
    """Raise PermissionError unless a confirmation-matrix spec explicitly authorizes the TEST split.

    Requires ``test_set_authorized is True`` AND ``freeze_state == 'final'``. Pure function: reads no data. It does not
    relax `enforce_patient_view` (which still rejects ``require_patient_view: test``).
    """
    if matrix_spec.get("test_set_authorized") is not True:
        raise PermissionError("GUARD: test_set_authorized is not true; the TEST split must not be read or evaluated")
    if matrix_spec.get("freeze_state") != "final":
        raise PermissionError(
            f"GUARD: freeze_state is {matrix_spec.get('freeze_state')!r}, not 'final'; the TEST split stays locked")
