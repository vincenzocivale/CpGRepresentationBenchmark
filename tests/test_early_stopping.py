"""Opt-in early stopping in training.engine.train_model and worker-count neutrality of the masking loaders."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from cpg_repr_benchmark.data.masking import MaskFractionBatchSampler, MaskingDataset
from cpg_repr_benchmark.training import engine


class _Loader:
    dataset = object()
    batch_sampler = object()

    def __iter__(self):
        return iter([])  # no optimisation step; validation is scripted below


def _run(monkeypatch, tmp_path, val_mse, epochs, early_stopping):
    seq = iter(val_mse)

    def fake_eval(model, loader, device):
        return {"mse": next(seq), "mae": 0.0}, {}

    monkeypatch.setattr(engine, "evaluate_loader", fake_eval)
    model = torch.nn.Linear(1, 1)
    hist = engine.train_model(model, _Loader(), None, device=torch.device("cpu"), epochs=epochs,
                              learning_rate=1e-3, weight_decay=0.0, mixed_precision=False,
                              output_dir=tmp_path, early_stopping=early_stopping)
    return hist


def test_default_off_runs_all_epochs(monkeypatch, tmp_path):
    hist = _run(monkeypatch, tmp_path, [1.0] * 6, 6, None)
    assert len(hist) == 6
    assert not (tmp_path / "early_stopping.json").exists()


def test_patience_stops_and_best_is_kept(monkeypatch, tmp_path):
    vals = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87, 0.5, 0.4]  # best at epoch 2, then 3 stalled epochs
    hist = _run(monkeypatch, tmp_path, vals, 8, {"patience": 3, "min_delta_rel": 0.0})
    assert len(hist) == 6
    info = json.loads((tmp_path / "early_stopping.json").read_text())
    assert info["stopped_early"] and info["best_epoch"] == 2 and info["epochs_run"] == 6
    assert torch.load(tmp_path / "checkpoints" / "best.pt", weights_only=False)["epoch"] == 2
    assert json.loads((tmp_path / "history.json").read_text())[-1]["epoch"] == 5


def test_min_delta_counts_tiny_gains_as_stall(monkeypatch, tmp_path):
    vals = [1.0, 0.9999, 0.9998, 0.9997, 0.9996]  # each gain 0.01% < min_delta 1%
    hist = _run(monkeypatch, tmp_path, vals, 5, {"patience": 2, "min_delta_rel": 0.01})
    assert len(hist) == 3  # stops at epoch 2
    info = json.loads((tmp_path / "early_stopping.json").read_text())
    assert info["best_epoch"] == 2  # best.pt is still the strict minimum


def test_cap_reached_without_stop(monkeypatch, tmp_path):
    hist = _run(monkeypatch, tmp_path, [1.0, 0.9, 0.8, 0.7], 4, {"patience": 3, "min_delta_rel": 1e-4})
    info = json.loads((tmp_path / "early_stopping.json").read_text())
    assert len(hist) == 4 and not info["stopped_early"] and info["best_epoch"] == 3


class _Store:
    dim = 3

    def get_by_matrix_columns(self, columns):
        c = np.asarray(columns, dtype=np.float32)
        return np.stack([c, c + 1, c + 2], axis=1)


def test_worker_count_does_not_change_batches(tmp_path: Path):
    path = tmp_path / "m.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as h:
        h.create_dataset("beta", data=rng.uniform(0.05, 0.95, (12, 200)).astype(np.float32))

    def batches(num_workers):
        ds = MaskingDataset(methylation_h5=path, representation_store=_Store(),
                            prior_logit_full=np.zeros(200, dtype=np.float32), sample_indices=np.arange(12),
                            context_columns=np.arange(200), target_columns=np.arange(200), panel_size=20,
                            mask_fractions=[0.15, 0.5, 0.9], seed=17)
        kw = {"persistent_workers": True} if num_workers else {}
        loader = DataLoader(ds, batch_sampler=MaskFractionBatchSampler(ds, 4, shuffle=True),
                            num_workers=num_workers, **kw)
        out = []
        for epoch in (0, 3):
            ds.set_epoch(epoch)
            loader.batch_sampler.set_epoch(epoch)
            out += [(b["sample_index"].numpy(), b["target_matrix_column"].numpy(), b["target_beta"].numpy())
                    for b in loader]
        return out

    a, b = batches(0), batches(2)
    assert len(a) == len(b) == 6
    for x, y in zip(a, b):
        for u, v in zip(x, y):
            assert np.array_equal(u, v)
