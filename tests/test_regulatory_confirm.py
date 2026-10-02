import importlib.util
from pathlib import Path

import pytest
import yaml

_spec = importlib.util.spec_from_file_location(
    "run_confirm", Path(__file__).resolve().parents[1] / "scripts" / "run_regulatory_confirm.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)
TEMPLATE = yaml.safe_load((mod.ROOT / mod.TEMPLATE).read_text())


def test_job_order_is_seed_major():
    j = mod.jobs()
    assert len(j) == 9
    assert [s for _, s in j] == [17] * 3 + [42] * 3 + [97] * 3
    assert mod.jobs(seeds=[42], only=["regulatory_clean"]) == [("regulatory_clean", 42)]


def test_built_configs_pass_and_match_template():
    for arm, seed in mod.jobs():
        cfg = mod.build(arm, seed)
        mod.check_config(cfg, TEMPLATE, arm, seed)
        assert cfg["training"]["batch_size"] == 8 and cfg["evaluation"]["mask_fractions"] == [0.5]


@pytest.mark.parametrize("section,key,value", [
    ("training", "batch_size", 32), ("training", "learning_rate", 3e-4), ("training", "mask_fractions", [0.5]),
    ("training", "epochs", 30), ("model", "hidden_dim", 256), ("evaluation", "selection_mask_fraction", 0.3),
    ("evaluation", "mask_fractions", [0.3]), ("evaluation", "require_patient_view", "test"),
    ("evaluation", "patient_view", "test"), ("evaluation", "save_predictions", False),
])
def test_validate_rejects_deviations(section, key, value):
    cfg = mod.build("regulatory_clean", 17)
    cfg[section][key] = value
    with pytest.raises((ValueError, PermissionError, KeyError)):
        mod.check_config(cfg, TEMPLATE, "regulatory_clean", 17)


def test_validate_rejects_missing_guard_and_early_stopping_change():
    cfg = mod.build("regulatory_clean", 17)
    del cfg["evaluation"]["require_patient_view"]
    with pytest.raises((ValueError, PermissionError, KeyError)):
        mod.check_config(cfg, TEMPLATE, "regulatory_clean", 17)
    cfg = mod.build("regulatory_clean", 17)
    cfg["training"]["early_stopping"]["patience"] = 5
    with pytest.raises(ValueError):
        mod.check_config(cfg, TEMPLATE, "regulatory_clean", 17)
