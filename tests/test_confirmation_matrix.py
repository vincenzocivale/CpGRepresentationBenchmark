from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_matrix_references_frozen_confirmation_budget():
    spec = yaml.safe_load((ROOT / "configs/experiments/regulatory_confirmation_matrix/matrix.yaml").read_text())["matrix"]
    protocol = spec["protocol"]
    assert protocol["max_epochs"] == 120
    assert protocol["early_stopping"] is False
    assert protocol["checkpoint_selection"] == "best_val_mse"
    assert protocol["checkpoint_selection_mask_fraction"] == 0.5
    assert "AMENDMENT 2" in protocol["reference"] and "REGULATORY_CONFIRMATION_PROTOCOL.md" in protocol["reference"]
    assert spec["test_set_authorized"] is False
    assert (ROOT / "docs/REGULATORY_CONFIRMATION_PROTOCOL.md").is_file()
    assert all(arm["status"] in {"not_run", "pending"} for arm in spec["arms"])
