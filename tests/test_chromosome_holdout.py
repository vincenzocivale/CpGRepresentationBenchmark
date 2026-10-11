from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from cpg_repr_benchmark.data.splits import (
    assert_no_heldout_chromosomes,
    chromosome_locus_split,
    load_or_create_locus_protocol,
)

CHROMS = np.array(["chr1"] * 10 + ["20"] * 4 + ["chr21"] * 3 + ["chr22"] * 3 + ["chrX"] * 5)
IDS = np.arange(500, 500 + len(CHROMS), dtype=np.int64)
COLS = np.arange(len(CHROMS))


def test_chromosome_split_is_exact():
    s = chromosome_locus_split(COLS, IDS, CHROMS, ["chr20", "chr21", "chr22"])
    assert set(s.heldout_columns) == set(range(10, 20))
    assert set(s.train_columns) == set(range(10)) | set(range(20, 25))
    assert_no_heldout_chromosomes({"train": s.train_columns}, CHROMS, ["chr20", "chr21", "chr22"])
    with pytest.raises(AssertionError):
        assert_no_heldout_chromosomes({"bad": COLS}, CHROMS, ["chr20"])


def test_protocol_create_reload_and_mismatch(tmp_path):
    p = tmp_path / "prot.npz"
    kw = dict(seed=1, heldout_fraction=0.0, heldout_chromosomes=["chr20", "chr21", "chr22"], chromosome_labels=CHROMS)
    a = load_or_create_locus_protocol(p, COLS, IDS, **kw)
    b = load_or_create_locus_protocol(p, COLS, IDS, **kw)
    assert np.array_equal(a.heldout_columns, b.heldout_columns) and len(a.heldout_columns) == 10
    with pytest.raises(ValueError):  # other chromosome list must not reuse the file
        load_or_create_locus_protocol(p, COLS, IDS, **{**kw, "heldout_chromosomes": ["chr21"]})
    with pytest.raises(ValueError):  # a plain protocol must not reuse it either
        load_or_create_locus_protocol(p, COLS, IDS, seed=1, heldout_fraction=0.0)
    with pytest.raises(ValueError):
        load_or_create_locus_protocol(tmp_path / "x.npz", COLS, IDS, **{**kw, "heldout_fraction": 0.2})


def test_default_protocol_unchanged(tmp_path):
    """Regression: without the new option the file format and split are exactly the legacy ones."""
    p = tmp_path / "legacy.npz"
    s = load_or_create_locus_protocol(p, COLS, IDS, seed=3, heldout_fraction=0.0)
    with np.load(p) as z:
        assert sorted(z.files) == ["heldout_cpg_idx", "heldout_fraction", "seed", "train_cpg_idx"]
    assert len(s.heldout_columns) == 0 and np.array_equal(s.train_columns, COLS)
    r = load_or_create_locus_protocol(tmp_path / "r.npz", COLS, IDS, seed=3, heldout_fraction=0.3)
    assert len(r.heldout_columns) == 5 + 1 + 1 or len(r.heldout_columns) == round(len(COLS) * 0.3)


def test_generated_configs_only_differ_in_holdout():
    repo = Path(__file__).resolve().parents[1]
    d = repo / "configs/experiments/unseen_chr20_22"
    files = sorted(d.glob("*.yaml"))
    if not files:
        pytest.skip("configs not generated")
    for f in files:
        c = yaml.safe_load(f.read_text())
        ls = c["experiment"]["locus_split"]
        assert ls["heldout_chromosomes"] == ["chr20", "chr21", "chr22"] and ls["heldout_fraction"] == 0.0
        assert c["experiment"]["output_root"] == "outputs/unseen_chr20_22/benchmark"
        assert "data/protocols" not in ls["protocol_path"] and "encode_atlas_v1" not in ls["protocol_path"]
        assert c["evaluation"]["patient_view"] == "validation" and c["evaluation"]["mask_fractions"] == [0.15, 0.3, 0.5, 0.7, 0.9]
        assert c["dataset"]["patient_protocol"] == "outputs/encode_atlas_v1/patients.npz"


def test_runner_end_to_end_chromosome_holdout(tmp_path: Path):
    rng = np.random.default_rng(17)
    n_samples = 18
    cpg_ids = IDS
    n_cpg = len(cpg_ids)
    beta = rng.beta(2, 2, size=(n_samples, n_cpg)).astype(np.float32)
    methyl, rep, reg = tmp_path / "m.h5", tmp_path / "r.h5", tmp_path / "reg.parquet"
    with h5py.File(methyl, "w") as h:
        h.create_dataset("beta", data=beta)
        h.create_dataset("cpg_idx", data=cpg_ids)
        h.create_dataset("sample_name", data=np.asarray([f"TCGA-{i:02d}-CASE{i:04d}" for i in range(n_samples)], dtype="S32"))
    with h5py.File(rep, "w") as h:
        h.create_dataset("cpg_idx", data=cpg_ids)
        h.create_dataset("embedding", data=rng.normal(size=(n_cpg, 5)).astype(np.float32))
    pd.DataFrame({"cpg_idx": cpg_ids, "chr": CHROMS}).to_parquet(reg)
    cfg = {
        "experiment": {"name": "e2e_chr", "output_root": str(tmp_path / "out"),
                       "locus_split": {"heldout_fraction": 0.0, "seed": 1, "heldout_chromosomes": ["chr20", "chr21", "chr22"],
                                       "protocol_path": str(tmp_path / "prot.npz")}},
        "dataset": {"name": "syn", "methylation_h5": str(methyl), "cpg_registry": str(reg), "patient_split": [0.7, 0.15, 0.15]},
        "representation": {"name": "toy", "mode": "precomputed", "source": "synthetic", "store_h5": str(rep)},
        "model": {"locus_latent_dim": 8, "token_dim": 8, "patient_dim": 8, "hidden_dim": 16},
        "training": {"seed": 17, "device": "cpu", "epochs": 1, "batch_size": 3, "learning_rate": 1e-3, "weight_decay": 0.0,
                     "mixed_precision": False, "num_workers": 0, "panel_size": 6, "mask_fractions": [0.5], "beta_epsilon": 1e-4},
        "evaluation": {"panel_size": 6, "mask_fractions": [0.5], "selection_mask_fraction": 0.5, "batch_size": 3,
                       "num_workers": 0, "save_predictions": True},
    }
    cp = tmp_path / "c.yaml"
    cp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo / "src")}
    res = subprocess.run([sys.executable, str(repo / "scripts/run_masking_benchmark.py"), "--config", str(cp), "--mode", "all"],
                         cwd=repo, env=env, text=True, capture_output=True, check=True)
    run_dir = Path(next(l for l in res.stdout.splitlines() if l.startswith("RUN_DIR=")).split("=", 1)[1])
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["n_heldout_loci"] == 10 and "mask_0.50" in summary["evaluation"]["unseen_locus"]
    ls = np.load(run_dir / "locus_split.npz")
    held = set(map(str, ["chr20", "chr21", "chr22"]))
    lab = dict(zip(cpg_ids.tolist(), ["chr" + c.replace("chr", "") for c in CHROMS]))
    assert not {lab[i] for i in ls["train_cpg_idx"].tolist()} & held
    assert {lab[i] for i in ls["heldout_cpg_idx"].tolist()} == held
    # prior of train loci must not depend on held-out columns: held-out loci carry the single global fallback value
    prior = np.load(run_dir / "prior_logit.npy")
    hc = ls["heldout_matrix_columns"]
    assert np.unique(prior[hc]).size == 1
