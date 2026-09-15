from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml


def test_masking_runner_end_to_end(tmp_path: Path):
    rng = np.random.default_rng(17)
    n_samples, n_cpg = 18, 30
    beta = rng.beta(2, 2, size=(n_samples, n_cpg)).astype(np.float32)
    methyl = tmp_path / "methyl.h5"
    representation = tmp_path / "repr.h5"
    cpg_ids = np.arange(1000, 1000 + n_cpg, dtype=np.int64)

    with h5py.File(methyl, "w") as h:
        h.create_dataset("beta", data=beta)
        h.create_dataset("cpg_idx", data=cpg_ids)
        h.create_dataset(
            "sample_name",
            data=np.asarray([f"TCGA-{i:02d}-CASE{i:04d}" for i in range(n_samples)], dtype="S32"),
        )
    with h5py.File(representation, "w") as h:
        h.create_dataset("cpg_idx", data=cpg_ids)
        h.create_dataset("embedding", data=rng.normal(size=(n_cpg, 5)).astype(np.float32))

    cfg = {
        "experiment": {
            "name": "e2e",
            "output_root": str(tmp_path / "outputs"),
            "locus_split": {
                "heldout_fraction": 0.3,
                "seed": 17,
                "protocol_path": str(tmp_path / "protocol.npz"),
            },
        },
        "dataset": {
            "name": "synthetic",
            "methylation_h5": str(methyl),
            "patient_split": [0.7, 0.15, 0.15],
        },
        "representation": {
            "name": "toy",
            "mode": "precomputed",
            "source": "synthetic",
            "store_h5": str(representation),
        },
        "model": {"locus_latent_dim": 8, "token_dim": 8, "patient_dim": 8, "hidden_dim": 16},
        "training": {
            "seed": 17,
            "device": "cpu",
            "epochs": 1,
            "batch_size": 3,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "mixed_precision": False,
            "num_workers": 0,
            "panel_size": 6,
            "mask_fractions": [0.5],
            "beta_epsilon": 1e-4,
        },
        "evaluation": {
            "panel_size": 6,
            "mask_fractions": [0.5],
            "selection_mask_fraction": 0.5,
            "batch_size": 3,
            "num_workers": 0,
            "save_predictions": False,
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src")
    result = subprocess.run(
        [sys.executable, str(repo / "scripts" / "run_masking_benchmark.py"), "--config", str(cfg_path), "--mode", "all"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    run_line = next(line for line in result.stdout.splitlines() if line.startswith("RUN_DIR="))
    run_dir = Path(run_line.split("=", 1)[1])
    summary = json.loads((run_dir / "summary.json").read_text())
    assert (run_dir / "experiment.json").exists()
    assert (run_dir / "representation_manifest.json").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert "mask_0.50" in summary["evaluation"]["seen"]
    assert "mask_0.50" in summary["evaluation"]["unseen_locus"]
    assert summary["n_train_loci"] + summary["n_heldout_loci"] == n_cpg
