"""Opt-in checkpoint policy of training.engine.train_model (synthetic, CPU, no real data)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cpg_repr_benchmark.experiments import eval_layout as el
from cpg_repr_benchmark.training import engine

ROOT = Path(__file__).resolve().parents[1]


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, observed_locus, observed_residual, observed_valid, target_locus, target_prior_logit):
        return torch.sigmoid(target_prior_logit + self.w), None


class _Loader:
    """n_batches tiny batches per epoch (one optimizer update each)."""
    dataset = object()
    batch_sampler = object()

    def __init__(self, n_batches):
        self.n = n_batches
        b = {"observed_locus": torch.zeros(1, 2, 3), "observed_residual": torch.zeros(1, 2), "observed_valid": torch.ones(1, 2, dtype=torch.bool),
             "target_locus": torch.zeros(1, 4, 3), "target_prior_logit": torch.zeros(1, 4), "target_beta": torch.full((1, 4), 0.7)}
        self.batch = b

    def __iter__(self):
        return iter([self.batch] * self.n)


def _run(monkeypatch, tmp_path, val_mse, epochs, n_batches=2, **kw):
    seq = iter(val_mse)
    calls = []

    def fake_eval(model, loader, device):
        calls.append(1)
        return {"mse": next(seq), "mae": 0.0}, {}

    monkeypatch.setattr(engine, "evaluate_loader", fake_eval)
    hist = engine.train_model(_Model(), _Loader(n_batches), None, device=torch.device("cpu"), epochs=epochs, learning_rate=1e-3,
                              weight_decay=0.0, mixed_precision=False, output_dir=tmp_path, **kw)
    return hist, len(calls)


def _ckpts(tmp_path):
    return sorted(p.name for p in (tmp_path / "checkpoints").iterdir())


def test_default_behaviour_unchanged_tcga(monkeypatch, tmp_path):
    hist, _ = _run(monkeypatch, tmp_path, [1.0, 0.9, 0.8], 3)
    assert _ckpts(tmp_path) == ["best.pt", "epoch_0000.pt", "epoch_0001.pt", "epoch_0002.pt", "last.pt"]
    assert all(set(r) == {"epoch", "train_mse", "validation_mse", "validation_mae"} for r in hist)   # no extra keys by default


def test_save_epoch_checkpoints_false_writes_no_epoch_files(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, [1.0, 0.9, 0.8], 3, save_epoch_checkpoints=False)
    assert _ckpts(tmp_path) == ["best.pt", "last.pt"]          # last.pt default stays on in the engine (config turns it off)
    t2 = tmp_path / "b"
    _run(monkeypatch, t2, [1.0, 0.9, 0.8], 3, save_epoch_checkpoints=False, save_last_checkpoint=False)
    assert _ckpts(t2) == ["best.pt"]
    assert len(json.loads((t2 / "history.json").read_text())) == 3


def test_best_pt_changes_only_on_strict_improvement_ties_keep_earliest(monkeypatch, tmp_path):
    vals = [0.50, 0.40, 0.40, 0.40, 0.45, 0.30, 0.30]   # ties at epochs 2,3 (keep 1); tie at 6 (keep 5)
    _run(monkeypatch, tmp_path, vals, 7, save_epoch_checkpoints=False, save_last_checkpoint=False)
    assert torch.load(tmp_path / "checkpoints" / "best.pt", weights_only=False)["epoch"] == 5
    t2 = tmp_path / "t"
    _run(monkeypatch, t2, [0.5, 0.4, 0.4, 0.4], 4, save_epoch_checkpoints=False, save_last_checkpoint=False)
    assert torch.load(t2 / "checkpoints" / "best.pt", weights_only=False)["epoch"] == 1


def test_120_epochs_no_early_stopping_and_update_counts(monkeypatch, tmp_path):
    vals = [1.0] + [0.5] * 119            # long plateau after epoch 1: must NOT stop
    hist, n_eval = _run(monkeypatch, tmp_path, vals, 120, n_batches=66, early_stopping=False, save_epoch_checkpoints=False,
                        save_last_checkpoint=False, record_update_counts=True)
    assert len(hist) == 120 and n_eval == 120                      # validation after EVERY epoch
    assert hist[-1]["updates_done"] == 7920 == 66 * 120
    assert all(r["updates_in_epoch"] == 66 for r in hist)
    assert [r["epoch"] for r in json.loads((tmp_path / "history.json").read_text())] == list(range(120))
    assert not (tmp_path / "early_stopping.json").exists()
    assert _ckpts(tmp_path) == ["best.pt"]
    assert torch.load(tmp_path / "checkpoints" / "best.pt", weights_only=False)["epoch"] == 1   # first of the tie plateau


def test_validation_uses_selection_fraction_and_runner_passes_keys():
    src = (ROOT / "scripts/run_masking_benchmark.py").read_text()
    assert 'validation_fraction = float(cfg["evaluation"].get("selection_mask_fraction", 0.5))' in src
    assert "mask_fractions=[validation_fraction]" in src
    for key in ("save_epoch_checkpoints", "save_last_checkpoint", "record_update_counts"):
        assert f'training_cfg.get("{key}"' in src


def test_authorizer_default_is_tcga_and_external_is_opt_in():
    assert el.authorizer_from_cfg({"evaluation": {}}) is None
    assert el.authorizer_from_cfg({"evaluation": {"patient_view": "test"}}) is None
    fn = el.authorizer_from_cfg({"evaluation": {"test_authorization": "external_gse40279_v1"}})
    assert callable(fn)
    with pytest.raises(ValueError):
        el.authorizer_from_cfg({"evaluation": {"test_authorization": "nope"}})


def test_runner_smoke_synthetic_cpu_with_external_style_keys(tmp_path):
    """Tiny CPU run of scripts/run_masking_benchmark.py on SYNTHETIC tensors with the external-run opt-in keys."""
    import os
    import subprocess
    import sys

    import h5py
    import numpy as np
    import yaml
    rng = np.random.default_rng(17)
    n_s, n_c = 24, 40
    ids = np.arange(1000, 1000 + n_c, dtype=np.int64)
    with h5py.File(tmp_path / "m.h5", "w") as h:
        h["beta"] = rng.beta(2, 2, (n_s, n_c)).astype(np.float32)
        h["cpg_idx"] = ids
        h["sample_name"] = np.asarray([f"TCGA-{i:02d}-CASE{i:04d}" for i in range(n_s)], dtype="S32")
    with h5py.File(tmp_path / "r.h5", "w") as h:
        h["cpg_idx"], h["embedding"] = ids, rng.normal(size=(n_c, 5)).astype(np.float32)
    cfg = {"experiment": {"name": "smoke", "output_root": str(tmp_path / "out"),
                          "locus_split": {"heldout_fraction": 0.0, "seed": 17, "protocol_path": str(tmp_path / "p.npz")}},
           "dataset": {"name": "synthetic", "methylation_h5": str(tmp_path / "m.h5"), "patient_split": [0.7, 0.15, 0.15]},
           "representation": {"name": "toy", "mode": "precomputed", "source": "synthetic", "store_h5": str(tmp_path / "r.h5")},
           "model": {"locus_latent_dim": 8, "token_dim": 8, "patient_dim": 8, "hidden_dim": 16},
           "training": {"seed": 17, "device": "cpu", "epochs": 3, "batch_size": 4, "learning_rate": 1e-3, "weight_decay": 0.0,
                        "mixed_precision": False, "num_workers": 0, "panel_size": 8, "mask_fractions": [0.5], "beta_epsilon": 1e-4,
                        "early_stopping": False, "save_epoch_checkpoints": False, "save_last_checkpoint": False,
                        "record_update_counts": True},
           "evaluation": {"panel_size": 8, "mask_fractions": [0.5], "selection_mask_fraction": 0.5, "batch_size": 4, "num_workers": 0,
                          "save_predictions": True, "patient_view": "validation", "require_patient_view": "validation",
                          "output_layout": "split_dirs", "allow_overwrite_split_dir": False}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/run_masking_benchmark.py"), "--config", str(tmp_path / "c.yaml"), "--mode", "all"],
                       cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr[-2000:]
    rd = Path([x for x in r.stdout.splitlines() if x.startswith("RUN_DIR=")][-1].split("=", 1)[1])
    assert sorted(p.name for p in (rd / "checkpoints").iterdir()) == ["best.pt"]
    hist = json.loads((rd / "history.json").read_text())
    n_train = json.loads((rd / "experiment.json").read_text())["n_train_patients_rows"]
    assert len(hist) == 3 and hist[-1]["updates_done"] == 3 * -(-n_train // 4)
    assert (rd / "evaluation/validation/seen/mask_0.50/metrics.json").is_file() and not (rd / "evaluation/test").exists()
