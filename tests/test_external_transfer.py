"""Experiment B (TCGA -> GSE40279 transfer) inference code: priors, frozen weights, layout, refusals (synthetic, CPU, tiny)."""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from cpg_repr_benchmark.external import gate as G
from cpg_repr_benchmark.external import transfer as T
from cpg_repr_benchmark.representations.hdf5_store import HDF5RepresentationStore
from cpg_repr_benchmark.training.priors import beta_to_logit, compute_leakage_safe_priors

N_EXT, N_LOCI = 656, 40
ARM = "a1"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture()
def world(tmp_path):
    """External matrix (656 rows: train rows 0.2, validation/test rows 0.9), TCGA run dir, store, checkpoint, frozen manifest."""
    repo = tmp_path / "repo"
    rng = np.random.default_rng(0)
    # external axis: cpg ids in a scrambled order, a subset of the TCGA axis
    ext_ids = np.array(rng.permutation(np.arange(1000, 1000 + N_LOCI)), dtype=np.int64)
    tcga_ids = np.array(rng.permutation(np.arange(990, 1000 + N_LOCI + 10)), dtype=np.int64)    # superset
    (repo / "data").mkdir(parents=True)
    perm = rng.permutation(N_EXT)
    train, val, test = np.sort(perm[:524]), np.sort(perm[524:590]), np.sort(perm[590:])
    beta = np.empty((N_EXT, N_LOCI), np.float32)
    beta[train] = rng.uniform(0.15, 0.25, (524, N_LOCI))
    beta[val] = 0.9
    beta[test] = 0.9
    with h5py.File(repo / "data/m.h5", "w") as h:
        h["cpg_idx"], h["beta"] = ext_ids, beta
        h["sample_name"] = np.array([f"GSM{i}".encode() for i in range(N_EXT)])
    np.savez(repo / "data/pat.npz", train=train, validation=val, test=test, sample_names=np.array([f"GSM{i}" for i in range(N_EXT)]), seed=1)
    np.savez(repo / "data/loci.npz", train_cpg_idx=np.sort(ext_ids), heldout_cpg_idx=np.array([], np.int64), seed=17001, heldout_fraction=0.0)
    # store (own order, covers external ids)
    s_ids = np.array(rng.permutation(np.arange(990, 1060)), dtype=np.int64)
    with h5py.File(repo / "data/store.h5", "w") as h:
        h["cpg_idx"], h["embedding"] = s_ids, rng.normal(size=(len(s_ids), 5)).astype(np.float32)
    # TCGA run
    rd = repo / "tcga/run/seed_17/ts"
    (rd / "checkpoints").mkdir(parents=True)
    tc_train = np.arange(30)
    tprior = rng.normal(size=len(tcga_ids)).astype(np.float32)
    np.save(rd / "prior_logit.npy", tprior)
    np.savez(rd / "patient_split.npz", train=tc_train, validation=np.arange(30, 40), test=np.arange(40, 50))
    (rd / "experiment.json").write_text(json.dumps({"n_train_patients_rows": 30, "global_train_beta_prior": 0.5}))
    (rd / "resolved_config.yaml").write_text(yaml.safe_dump({"dataset": {"methylation_h5": "tcga/tcga.h5"}, "training": {"beta_epsilon": 1e-4}}))
    with h5py.File(repo / "tcga/tcga.h5", "w") as h:
        h["cpg_idx"] = tcga_ids
    model = T.build_model(5, {"locus_latent_dim": 8, "token_dim": 8, "patient_dim": 8, "hidden_dim": 8})
    torch.manual_seed(0)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    torch.save({"model": model.state_dict(), "epoch": 7}, rd / "checkpoints/best.pt")
    entry = {"arm": ARM, "seed": 17, "best_pt": "tcga/run/seed_17/ts/checkpoints/best.pt", "best_pt_sha256": _sha(rd / "checkpoints/best.pt")}
    man = {"files": {"patient_protocol_npz": "data/pat.npz", "locus_protocol_npz": "data/loci.npz", "dataset_h5": "data/m.h5"},
           "counts": {"train": 524, "validation": 66, "test": 66}, "universe_size": N_LOCI, "locus_seed": 17001, "seeds": [17],
           "arms": {ARM: {"store_h5": "data/store.h5", "store_sha256": _sha(repo / "data/store.h5"), "dim": 5}},
           "phase_a_checkpoints_experiment_B": [entry]}
    return {"repo": repo, "ext_ids": ext_ids, "tcga_ids": tcga_ids, "tprior": tprior, "beta": beta, "train": train, "val": val,
            "test": test, "man": man, "entry": entry, "model": model, "tc_train": tc_train}


def test_reindex_prior_matches_on_global_cpg_idx(world):
    out = T.reindex_prior(world["tprior"], world["tcga_ids"], world["ext_ids"])
    pos = {int(i): k for k, i in enumerate(world["tcga_ids"])}
    assert np.array_equal(out, np.array([world["tprior"][pos[int(i)]] for i in world["ext_ids"]], np.float32))
    with pytest.raises(ValueError):
        T.reindex_prior(world["tprior"], world["tcga_ids"][:-20], world["ext_ids"])     # TCGA axis missing external loci


def test_strict_prior_is_the_original_tcga_prior_and_uses_no_external_data(world):
    prior, info = T.load_tcga_prior(world["repo"], world["entry"], world["ext_ids"], tcga_patients_train=world["tc_train"])
    assert np.array_equal(prior, T.reindex_prior(world["tprior"], world["tcga_ids"], world["ext_ids"]))
    assert info["external_data_used_in_prior"] is False and info["tcga_n_train_patients_rows"] == 30
    assert info["tcga_prior_sha256"] == _sha(world["repo"] / "tcga/run/seed_17/ts/prior_logit.npy")
    # no external matrix needed: removing it does not matter
    (world["repo"] / "data/m.h5").unlink()
    T.load_tcga_prior(world["repo"], world["entry"], world["ext_ids"])
    # the TCGA run must have been fitted on the frozen TCGA train rows
    with pytest.raises(ValueError):
        T.load_tcga_prior(world["repo"], world["entry"], world["ext_ids"], tcga_patients_train=np.arange(1, 31))


def test_tcga_prior_recompute_check_uses_train_rows_only(world, tmp_path):
    calls = []

    def spy(h5, rows, cols, n, epsilon=1e-4):
        calls.append(np.asarray(rows))
        return np.load(world["repo"] / "tcga/run/seed_17/ts/prior_logit.npy"), None, None
    rd = world["repo"] / "tcga/run/seed_17/ts"
    np.savez(rd / "locus_split.npz", train_matrix_columns=np.arange(len(world["tcga_ids"])))
    out = T.recompute_tcga_prior_check(world["repo"], world["entry"], prior_fn=spy)
    assert out["tcga_test_rows_read"] == 0 and np.array_equal(calls[0], world["tc_train"])


def test_recalibrated_prior_uses_only_external_train_rows(world):
    seen = []

    def spy(h5, rows, cols, n, epsilon=1e-4):
        seen.append((np.asarray(rows), np.asarray(cols)))
        return compute_leakage_safe_priors(h5, rows, cols, n, epsilon=epsilon)
    prior, info = T.recalibrated_prior(world["repo"], world["man"], world["ext_ids"], prior_fn=spy)
    rows, cols = seen[0]
    assert np.array_equal(rows, world["train"]) and len(rows) == 524
    assert not np.intersect1d(rows, world["val"]).size and not np.intersect1d(rows, world["test"]).size
    assert np.array_equal(np.sort(cols), np.arange(N_LOCI))                      # all universe columns
    expected = beta_to_logit(world["beta"][world["train"]].mean(axis=0), 1e-4)
    assert np.allclose(prior, expected, atol=1e-5)                               # validation/test rows (0.9) did not leak in
    assert info["external_data_used_in_prior"] is True and info["n_prior_rows"] == 524 and info["validation_test_rows_read_for_prior"] == 0
    assert prior.mean() < beta_to_logit(np.array([0.5]), 1e-4)[0]


def test_recalibrated_prior_refuses_wrong_inputs(world):
    man = json.loads(json.dumps(world["man"]))
    man["counts"]["train"] = 523
    with pytest.raises(ValueError):
        T.recalibrated_prior(world["repo"], man, world["ext_ids"])
    man = json.loads(json.dumps(world["man"]))
    man["universe_size"] = N_LOCI + 1
    with pytest.raises(ValueError):
        T.recalibrated_prior(world["repo"], man, world["ext_ids"])
    # train rows overlapping validation -> refused
    pat = dict(np.load(world["repo"] / "data/pat.npz"))
    pat["validation"] = np.concatenate([pat["validation"][:-1], pat["train"][:1]])
    np.savez(world["repo"] / "data/pat.npz", **pat)
    with pytest.raises(ValueError):
        T.recalibrated_prior(world["repo"], world["man"], world["ext_ids"])


def test_strict_and_recalibrated_priors_differ_and_dirs_are_separate(world, tmp_path):
    strict, _ = T.load_tcga_prior(world["repo"], world["entry"], world["ext_ids"])
    recal, _ = T.recalibrated_prior(world["repo"], world["man"], world["ext_ids"])
    assert not np.allclose(strict, recal)
    a, b = T.run_dir_for(world["repo"], "B_strict", ARM, 17), T.run_dir_for(world["repo"], "B_recalibrated", ARM, 17)
    assert a != b and a.parent.parent.name == "B_strict" and "transfer" in a.parts
    with pytest.raises(ValueError):
        T.mode_dir(world["repo"], "B_other")


def test_checkpoint_sha_verified(world):
    assert T.verify_checkpoint(world["repo"], world["entry"]).is_file()
    bad = {**world["entry"], "best_pt_sha256": "0" * 64}
    with pytest.raises(ValueError):
        T.verify_checkpoint(world["repo"], bad)


def _eval(world, split, rd, prior, authorize=None, model=None):
    ids = np.sort(world["ext_ids"])
    store = HDF5RepresentationStore(world["repo"] / "data/store.h5", world["ext_ids"])
    rows = world[{"validation": "val", "test": "test"}[split]]
    cols = np.argsort(world["ext_ids"])[np.searchsorted(ids, ids)]
    return T.evaluate_model(run_dir=rd, model=model or world["model"], store=store, matrix_path=world["repo"] / "data/m.h5", prior=prior,
                            rows=rows, columns=np.arange(N_LOCI), split=split, device=torch.device("cpu"), authorize=authorize,
                            eval_cfg={"panel_size": 20}), cols


def test_evaluation_layout_frozen_weights_and_masks_shared_across_modes(world, tmp_path):
    ckpt = T.verify_checkpoint(world["repo"], world["entry"])
    model, epoch = T.load_frozen_model(ckpt, 5, {"locus_latent_dim": 8, "token_dim": 8, "patient_dim": 8, "hidden_dim": 8})
    assert epoch == 7 and not any(p.requires_grad for p in model.parameters()) and not model.training
    before = {k: v.clone() for k, v in model.state_dict().items()}
    strict, _ = T.load_tcga_prior(world["repo"], world["entry"], world["ext_ids"])
    recal, _ = T.recalibrated_prior(world["repo"], world["man"], world["ext_ids"])
    rs, rr = tmp_path / "B_strict/run", tmp_path / "B_recalibrated/run"
    _eval(world, "validation", rs, strict, model=model)
    _eval(world, "validation", rr, recal, model=model)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])                                   # no weight update
    for rd in (rs, rr):
        for f in (0.15, 0.30, 0.50, 0.70, 0.90):
            m = json.loads((rd / f"evaluation/validation/seen/mask_{f:.2f}/metrics.json").read_text())
            assert m["patient_view"] == "validation" and m["mask_seed"] == 17001
            assert (rd / f"evaluation/validation/seen/mask_{f:.2f}/predictions.npz").is_file()
        assert (rd / "evaluation/validation/summary.json").is_file() and not (rd / "evaluation/test").exists()
    # same frozen masks/panels in both modes (paired), different priors -> different predictions
    a = np.load(rs / "evaluation/validation/seen/mask_0.50/predictions.npz")
    b = np.load(rr / "evaluation/validation/seen/mask_0.50/predictions.npz")
    for k in ("sample_index", "target_matrix_column", "target"):
        assert np.array_equal(a[k], b[k])
    assert not np.allclose(a["prior_prediction"], b["prior_prediction"])
    assert set(np.unique(a["sample_index"])) <= set(world["val"].tolist())      # validation rows only (no test rows)


def test_test_split_refused_without_authorization_and_never_overwrites(world, tmp_path):
    rd = tmp_path / "B_strict/run"
    draft_repo = tmp_path / "draft_repo"                      # synthetic draft/unauthorized external manifest (not the real repo)
    (draft_repo / "configs/external").mkdir(parents=True)
    (draft_repo / G.MANIFEST_REL).write_text(json.dumps({"freeze_state": "draft", "test_set_authorized": False}))
    prior, _ = T.load_tcga_prior(world["repo"], world["entry"], world["ext_ids"])
    _eval(world, "validation", rd, prior)
    with pytest.raises(PermissionError):
        _eval(world, "test", rd, prior)                      # default authorizer = TCGA matrix (draft): locked
    with pytest.raises(PermissionError):
        _eval(world, "test", rd, prior, authorize=functools.partial(G.authorize_external_test, draft_repo))   # draft + unauthorized manifest
    assert not (rd / "evaluation/test").exists()
    with pytest.raises(FileExistsError):
        _eval(world, "validation", rd, prior)                # same split directory is never reused


def test_run_transfer_refuses_on_failed_gate_and_dry_run_lists_12_or_fewer(world):
    bad = [G.Check("x", False)]
    lines = []
    assert T.run_transfer(Path("."), "B_strict", gate_checks=bad, out=lines.append) in (2,)   # refused, nothing evaluated
    assert T.run_transfer(Path("."), "B_strict", dry_run=True, gate_checks=bad, out=lines.append) == 3
    assert T.run_transfer(Path("."), "B_strict", dry_run=True, gate_checks=[], out=lines.append) == 0
    listing = [l for l in lines if l.endswith("checkpoint(s): " + l.split("checkpoint(s): ")[-1]) and "checkpoint(s)" in l]
    assert any("regulatory_histone_dnase#17" in l for l in listing)
    n9 = [l for l in listing if l.startswith("9 checkpoint(s)")]
    assert n9


def test_real_manifest_lists_12_checkpoints_default_scope_is_the_9_main_arm_ones():
    root = Path(__file__).resolve().parents[1]
    man = json.loads((root / "configs/external/gse40279_v1_freeze_manifest.json").read_text())
    assert len(man["phase_a_checkpoints_experiment_B"]) == 12          # all hashes stay recorded (and audited)
    js = T.jobs(man)                                                    # AMENDMENT 2 default: 3 main arms x 3 seeds
    assert len(js) == 9 and {c["arm"] for c in js} == {"regulatory_histone_dnase", "cpgpt_large_locus", "deepcpg_dna_locus"}
    assert [(c["arm"], c["seed"]) for c in js][:3] == [("regulatory_histone_dnase", 17), ("cpgpt_large_locus", 17), ("deepcpg_dna_locus", 17)]
    assert all(c["role"] == "main" for c in js)
    leg = T.jobs(man, include_legacy=True)
    assert len(leg) == 12 and sum(c["role"] == "legacy_sensitivity_control" for c in leg) == 3
    assert [(c["arm"], c["seed"]) for c in leg][:4] == [("regulatory_histone_dnase", 17), ("functional_annotations_pca", 17),
                                                        ("cpgpt_large_locus", 17), ("deepcpg_dna_locus", 17)]
    with pytest.raises(ValueError):
        T.jobs(man, only=["functional_annotations_pca"])                # legacy arm needs the explicit flag
    assert [c["arm"] for c in T.jobs(man, only=["functional_annotations_pca"], include_legacy=True)] == ["functional_annotations_pca"] * 3
    assert T.arm_role("functional_annotations_pca") == "legacy_sensitivity_control" and T.arm_role("cpgpt_large_locus") == "main"


def test_transfer_cli_default_dry_run_lists_9_and_legacy_flag_is_explicit():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_run_tr", Path(__file__).resolve().parents[1] / "scripts/run_external_transfer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    help_text = mod.__doc__
    assert "--include-legacy" in help_text and "3 MAIN arms" in help_text and "legacy_sensitivity_control" in help_text
    lines = []
    assert T.run_transfer(Path(__file__).resolve().parents[1], "B_strict", dry_run=True, gate_checks=[], out=lines.append) == 0
    assert any(l.startswith("9 checkpoint(s)") and "functional_annotations_pca" not in l for l in lines)
    lines = []
    assert T.run_transfer(Path(__file__).resolve().parents[1], "B_strict", dry_run=True, gate_checks=[], include_legacy=True, out=lines.append) == 0
    assert any(l.startswith("12 checkpoint(s)") for l in lines)
