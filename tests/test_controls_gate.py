"""Synthetic checks of the R15 test gate (policy tcga_controls_freeze_v1): locked while draft, unlocked when final, locked again on any change."""
import json

import pytest

from cpg_repr_benchmark.experiments import controls_gate as cg
from cpg_repr_benchmark.experiments.test_confirmation_v2 import canonical_cfg_sha256, sha256_file

CFG = {"evaluation": {"patient_view": "test", "output_layout": "split_dirs", "test_freeze_run_id": "r1", "test_authorization": cg.POLICY}}


def _setup(tmp_path, final=True):
    (tmp_path / "best.pt").write_bytes(b"ckpt")
    (tmp_path / "code.py").write_text("x = 1")
    m = {"policy": cg.POLICY, "freeze_state": "final" if final else "draft", "test_set_authorized": final,
         "shared_files": {"code.py": sha256_file(tmp_path / "code.py")},
         "runs": {"r1": {"eval_config_canonical_sha256": canonical_cfg_sha256(CFG), "files": {"best.pt": sha256_file(tmp_path / "best.pt")}}}}
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(m))
    return mp


def test_unlocked_only_when_final_and_unchanged(tmp_path):
    mp = _setup(tmp_path)
    assert cg.verify(CFG, tmp_path, mp) == []
    (tmp_path / "code.py").write_text("x = 2")  # code changed after the freeze
    assert any("sha256 mismatch for code.py" in p for p in cg.verify(CFG, tmp_path, mp))
    mp = _setup(tmp_path, final=False)
    assert any("not final" in p for p in cg.verify(CFG, tmp_path, mp))


def test_config_and_run_id_are_pinned(tmp_path):
    mp = _setup(tmp_path)
    changed = {"evaluation": {**CFG["evaluation"], "mask_seed": 1}}
    assert any("differs from the frozen" in p for p in cg.verify(changed, tmp_path, mp))
    assert any("not in the freeze manifest" in p for p in cg.verify({"evaluation": {**CFG["evaluation"], "test_freeze_run_id": "zz"}}, tmp_path, mp))
    with pytest.raises(PermissionError):
        cg.authorizer_from_cfg(changed)()  # default manifest path is absent/other: locked
