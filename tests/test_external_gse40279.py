"""Synthetic-only tests for the external GSE40279 phase-1 tooling (mapping, split, budget, freeze verifier)."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from cpg_repr_benchmark.encode_atlas.protocol import load_patient_protocol
from cpg_repr_benchmark.external import freeze
from cpg_repr_benchmark.external.budget import (
    SUPERSEDED_BUDGET,
    epoch_budget,
    external_schedule,
    steps_per_epoch,
    total_updates,
)
from cpg_repr_benchmark.external.io import sha256_file
from cpg_repr_benchmark.external.mapping import exact_coordinate_join
from cpg_repr_benchmark.external.split import age_tercile_thresholds, age_terciles, stratified_split


# ------------------------------------------------------------------ mapping
def test_join_exact_and_unmapped():
    u_chr = ["chr1", "chr1", "chr2", "chr3"]
    u_pos = [100, 200, 100, 50]
    c_chr = ["chr2", "chr1", "chr1", "chrM"]
    c_pos = [100, 101, 100, 5]  # chr1:101 is an off-by-one decoy and must NOT match chr1:100/200
    ur, cc, un = exact_coordinate_join(u_chr, u_pos, c_chr, c_pos)
    assert ur.tolist() == [0, 2] and cc.tolist() == [2, 0]
    assert un.tolist() == [1, 3]
    assert len(ur) + len(un) == 4


def test_join_rejects_duplicates():
    with pytest.raises(ValueError):
        exact_coordinate_join(["chr1", "chr1"], [1, 1], ["chr1"], [1])
    with pytest.raises(ValueError):
        exact_coordinate_join(["chr1"], [1], ["chr1", "chr1"], [1, 1])


def test_join_preserves_universe_order():
    ur, cc, _ = exact_coordinate_join(["chr2", "chr1", "chr3"], [5, 5, 5], ["chr1", "chr3", "chr2"], [5, 5, 5])
    assert ur.tolist() == [0, 1, 2] and cc.tolist() == [2, 0, 1]


# ------------------------------------------------------------------ split
def _cohort(n=656, seed=0):
    r = np.random.default_rng(seed)
    ids = [f"{1000 + i}" for i in range(n)]
    sex = r.choice(["F", "M"], n)
    age = r.integers(19, 102, n).astype(float)
    return ids, sex, age


def _strata(sex, age):
    t = age_terciles(age, age_tercile_thresholds(age))
    return [f"{s}_T{k}" for s, k in zip(sex, t)]


def test_split_counts_partition_determinism():
    ids, sex, age = _cohort()
    strata = _strata(sex, age)
    a = stratified_split(ids, strata, 20260925)
    b = stratified_split(ids, strata, 20260925)
    assert (a == b).all()
    assert {k: int((a == k).sum()) for k in ("train", "validation", "test")} == {"train": 524, "validation": 66, "test": 66}
    assert set(a) == {"train", "validation", "test"}
    assert not (a == stratified_split(ids, strata, 1)).all()
    # row-order invariance
    perm = np.random.default_rng(3).permutation(len(ids))
    c = stratified_split([ids[i] for i in perm], [strata[i] for i in perm], 20260925)
    assert dict(zip([ids[i] for i in perm], c)) == dict(zip(ids, a))


def test_split_stratification_balance():
    ids, sex, age = _cohort()
    strata = np.asarray(_strata(sex, age))
    lab = stratified_split(ids, strata.tolist(), 7)
    for s in set(strata):
        n = int((strata == s).sum())
        for sp in ("validation", "test"):
            assert abs((lab[strata == s] == sp).sum() - 0.1 * n) < 1.0


def test_split_small_stratum_and_errors():
    ids = [str(i) for i in range(12)]
    strata = ["A"] * 10 + ["B"] * 2
    lab = stratified_split(ids, strata, 1)
    assert (lab[10:] == "train").sum() >= 1
    with pytest.raises(ValueError):
        stratified_split(["1", "1"], ["a", "a"], 1)  # duplicate subject ids
    with pytest.raises(ValueError):
        stratified_split(["1", "2"], ["a", "a"], 1, fractions=(0.0, 0.5, 0.5))  # no train subject left in stratum


def test_split_protocol_accepted_by_loader(tmp_path):
    ids, sex, age = _cohort()
    lab = stratified_split(ids, _strata(sex, age), 20260925)
    rows = {k: np.flatnonzero(lab == k).astype(np.int64) for k in ("train", "validation", "test")}
    p = tmp_path / "p.npz"
    names = [f"GSM{i}" for i in range(len(ids))]
    np.savez_compressed(p, **rows, sample_names=np.asarray(names, str), seed=np.asarray(20260925))
    got = load_patient_protocol(p, names)
    assert sum(len(v) for v in got.values()) == 656
    inter = set(got["train"]) & (set(got["validation"]) | set(got["test"]))
    assert not inter and not (set(got["validation"]) & set(got["test"]))


def test_tercile_ties_not_split():
    age = np.array([50.0] * 10 + [60.0] * 10 + [70.0] * 10)
    t = age_terciles(age, age_tercile_thresholds(age))
    for a in (50.0, 60.0, 70.0):
        assert len(set(t[age == a])) == 1


# ------------------------------------------------------------------ budget
def test_budget_arithmetic():
    assert steps_per_epoch(7342, 8) == 918 and steps_per_epoch(524, 8) == 66 and steps_per_epoch(8, 8) == 1
    assert total_updates(7342, 8, 120) == 110160
    s = external_schedule(110160, 66)
    assert (s["full_epochs"], s["partial_epoch_updates"], s["n_epochs_started"]) == (1669, 6, 1670)
    assert s["validation_points"][0] == 66 and s["validation_points"][-2] == 1669 * 66 and s["validation_points"][-1] == 110160
    assert s["n_validation_points"] == 1670
    exact = external_schedule(660, 66)  # multiple: no partial epoch, final point not duplicated
    assert exact["partial_epoch_updates"] == 0 and exact["validation_points"][-1] == 660 and exact["n_validation_points"] == 10
    with pytest.raises(ValueError):
        external_schedule(0, 66)


def test_amended_epoch_budget():
    assert steps_per_epoch(524, 8) == 66 and 524 % 8 == 4  # 65 batches of 8 + one of 4
    b = epoch_budget(524, 8, 120)
    assert b == {"epochs": 120, "max_updates": None, "early_stopping": False, "batch_size": 8,
                 "updates_per_epoch": 66, "total_updates": 7920, "validation_points": 120}
    assert b["total_updates"] == 66 * 120 and b["total_updates"] % b["updates_per_epoch"] == 0  # no partial epoch
    assert 7920 / 110160 == pytest.approx(0.0719, abs=1e-4)
    with pytest.raises(ValueError):
        epoch_budget(524, 8, 0)


def test_superseded_budget_flagged_never_executed():
    assert SUPERSEDED_BUDGET["executed"] is False and SUPERSEDED_BUDGET["updates"] == 110160
    assert (SUPERSEDED_BUDGET["epochs_full"], SUPERSEDED_BUDGET["partial_updates"]) == (1669, 6)
    assert epoch_budget(524, 8, 120)["total_updates"] != SUPERSEDED_BUDGET["updates"]


def test_partial_epoch_prefix_determinism():
    """The sampler's batch sequence for an epoch is a function of (seed, epoch) only, so stopping after 6 batches of the
    last epoch yields exactly the first 6 batches of the full epoch."""
    from cpg_repr_benchmark.data.masking import MaskFractionBatchSampler

    class _DS:
        seed = 17001
        mask_fractions = (0.15, 0.3, 0.5, 0.7, 0.9)

        def __len__(self):
            return 524

    s = MaskFractionBatchSampler(_DS(), 8, shuffle=True)
    assert len(s) == 66
    s.set_epoch(1669)
    full = list(s)
    s2 = MaskFractionBatchSampler(_DS(), 8, shuffle=True)
    s2.set_epoch(1669)
    prefix = [b for _, b in zip(range(6), s2)]
    assert prefix == full[:6] and len(full[-1]) == 4


# ------------------------------------------------------------------ freeze verifier
def _synthetic_repo(tmp: Path) -> Path:
    repo = tmp / "repo"
    for rel in list(freeze.FILES.values()):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    n_loc = freeze.EXPECTED_UNIVERSE
    names = [f"GSM{i}" for i in range(656)]
    rows = {"train": np.arange(0, 524), "validation": np.arange(524, 590), "test": np.arange(590, 656)}
    np.savez_compressed(repo / freeze.FILES["patient_protocol_npz"], **rows, sample_names=np.asarray(names, str),
                        seed=np.asarray(freeze.SPLIT_SEED))
    ids = np.arange(1, n_loc + 1, dtype=np.int64)
    np.savez_compressed(repo / freeze.FILES["locus_protocol_npz"], train_cpg_idx=ids,
                        heldout_cpg_idx=np.asarray([], dtype=np.int64), seed=np.asarray(17001), heldout_fraction=np.asarray(0.0))
    with h5py.File(repo / freeze.FILES["dataset_h5"], "w") as h:
        h.create_dataset("beta", shape=(656, n_loc), dtype="float32", chunks=(1, 8192))  # unwritten: no disk
        h.create_dataset("cpg_idx", data=ids)
        h.create_dataset("sample_name", data=np.asarray(names, dtype=object), dtype=h5py.string_dtype())
    for k in ("phenotypes_parquet", "cpg_mapping_external_parquet"):
        pd.DataFrame({"a": [1]}).to_parquet(repo / freeze.FILES[k])
    for k in ("mapping_manifest", "split_manifest"):
        (repo / freeze.FILES[k]).write_text("{}")
    (repo / freeze.FILES["split_csv"]).write_text("row\n0\n")
    (repo / freeze.FILES["dataset_config"]).write_text("name: x\n")
    # TCGA-side facts: patients.npz (train 7342), matrix.yaml, phase_A_status
    (repo / "outputs/encode_atlas_v1").mkdir(parents=True)
    np.savez_compressed(repo / freeze.TCGA_PATIENTS_REL, train=np.arange(7342), validation=np.arange(918), test=np.arange(918))
    arms = []
    for a in freeze.ARMS:
        store = repo / f"stores/{a}.h5"
        store.parent.mkdir(exist_ok=True)
        store.write_bytes(a.encode())
        arms.append({"id": a, "store_h5": f"stores/{a}.h5", "dim": 8, "sha256": sha256_file(store)})
    (repo / freeze.MATRIX_REL).parent.mkdir(parents=True)
    (repo / freeze.MATRIX_REL).write_text(yaml.safe_dump({"matrix": {"arms": arms}}))
    runs = []
    for a in freeze.ARMS:
        for s in freeze.SEEDS:
            ck = repo / f"runs/{a}/{s}/checkpoints/best.pt"
            ck.parent.mkdir(parents=True)
            ck.write_bytes(f"{a}{s}".encode())
            runs.append({"arm": a, "seed": s, "run_dir": f"runs/{a}/{s}", "best_pt_sha256": sha256_file(ck)})
    (repo / freeze.PHASE_A_STATUS_REL).parent.mkdir(parents=True, exist_ok=True)
    (repo / freeze.PHASE_A_STATUS_REL).write_text(json.dumps({"runs": runs}))
    (repo / freeze.MANIFEST_REL).parent.mkdir(parents=True, exist_ok=True)
    (repo / freeze.MANIFEST_REL).write_text(json.dumps(freeze.build_manifest(repo)))
    return repo


def _snapshot(root: Path):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(root.rglob("*")) if p.is_file()}


def test_verifier_green_and_never_writes(tmp_path):
    repo = _synthetic_repo(tmp_path)
    before = _snapshot(repo)
    res = freeze.verify(repo)
    assert all(ok for _, ok, _ in res), [r for r in res if not r[1]]
    assert _snapshot(repo) == before  # no-write guarantee


def _fails(repo) -> set[str]:
    return {n for n, ok, _ in freeze.verify(repo) if not ok}


def test_verifier_detects_tampering(tmp_path):
    repo = _synthetic_repo(tmp_path)
    (repo / freeze.FILES["split_csv"]).write_text("row\n1\n")
    assert "sha256:split_csv" in _fails(repo)


def test_verifier_detects_store_and_checkpoint_change(tmp_path):
    repo = _synthetic_repo(tmp_path)
    (repo / "stores" / f"{freeze.ARMS[0]}.h5").write_bytes(b"other")
    (repo / f"runs/{freeze.ARMS[1]}/42/checkpoints/best.pt").write_bytes(b"other")
    f = _fails(repo)
    assert f"store_sha256:{freeze.ARMS[0]}" in f and f"phaseA_best_pt:{freeze.ARMS[1]}:42" in f


def test_verifier_detects_wrong_counts_and_budget(tmp_path):
    repo = _synthetic_repo(tmp_path)
    names = [f"GSM{i}" for i in range(656)]
    rows = {"train": np.arange(0, 500), "validation": np.arange(500, 578), "test": np.arange(578, 656)}
    np.savez_compressed(repo / freeze.FILES["patient_protocol_npz"], **rows, sample_names=np.asarray(names, str),
                        seed=np.asarray(freeze.SPLIT_SEED))
    assert "split_counts" in _fails(repo)
    repo2 = _synthetic_repo(tmp_path / "b")
    np.savez_compressed(repo2 / freeze.TCGA_PATIENTS_REL, train=np.arange(7000), validation=np.arange(10), test=np.arange(10))
    assert "budget_tcga_recomputed" in _fails(repo2)


def _tamper_budget(tmp_path, **changes):
    repo = _synthetic_repo(tmp_path)
    mp = repo / freeze.MANIFEST_REL
    m = json.loads(mp.read_text())
    m["budget"].update(changes)
    mp.write_text(json.dumps(m))
    return _fails(repo)


def test_manifest_budget_amended_and_green(tmp_path):
    repo = _synthetic_repo(tmp_path)
    m = json.loads((repo / freeze.MANIFEST_REL).read_text())
    assert m["protocol_version"] == "v1.2" and m["freeze_state"] == "draft" and m["test_set_authorized"] is False
    assert m["budget"]["total_updates"] == 7920 and m["budget"]["validation_points"] == 120
    assert m["superseded_budget"]["executed"] is False and m["amendment"]["id"] == "AMENDMENT 1"   # amendment 1 stays visible
    assert m["amendment_2"]["id"] == "AMENDMENT 2" and m["amendment_2"]["decided_after_validation_A_results"] is True
    assert m["amendment_2"]["external_test_data_read_before_amendment"] is False
    assert not _fails(repo)


def test_manifest_amendment_2_structure(tmp_path):
    repo = _synthetic_repo(tmp_path)
    m = json.loads((repo / freeze.MANIFEST_REL).read_text())
    assert m["main_panel"] == ["regulatory_histone_dnase", "cpgpt_large_locus", "deepcpg_dna_locus"]
    assert m["legacy_sensitivity_control"]["arms"] == ["functional_annotations_pca"]
    assert m["legacy_sensitivity_control"]["in_main_inferential_comparison"] is False
    cp = m["comparisons"]
    assert cp["primary"]["comparator"] == "cpgpt_large_locus" and cp["primary"]["n_contrasts"] == 1
    assert cp["secondary"]["comparator"] == "deepcpg_dna_locus" and cp["descriptive"]["inferential"] is False
    assert cp["equivalence_margin"] is None and cp["equivalence_or_non_inferiority_claims"] is False
    assert m["experiment_B_scope"]["n_checkpoints"] == 9 and m["experiment_B_scope"]["arms"] == m["main_panel"]
    assert m["test_one_shot"] is True and m["test_plan"]["authorization_tag"] == "external-recon-test-authorization-v1"
    assert m["test_plan"]["checkpoints"] == {"main": 9, "legacy_sensitivity_control": 3,
                                              "source": "the best.pt of the 12 Experiment A runs, selected on validation"}
    assert m["budget"]["total_updates"] == 7920 and m["seeds"] == [17, 42, 97]       # budget/seeds untouched
    assert sum(c["role"] == "main" for c in m["phase_a_checkpoints_experiment_B"]) == 9


@pytest.mark.parametrize("path,value,check", [
    (("main_panel",), ["regulatory_histone_dnase", "functional_annotations_pca", "cpgpt_large_locus"], "main_panel_three_arms"),
    (("legacy_sensitivity_control", "in_main_inferential_comparison"), True, "legacy_control_functional_outside_main"),
    (("legacy_sensitivity_control", "may_change_main_claim"), True, "legacy_control_functional_outside_main"),
    (("comparisons", "primary", "comparator"), "functional_annotations_pca", "comparisons_structure"),
    (("comparisons", "primary", "n_contrasts"), 2, "comparisons_structure"),
    (("comparisons", "descriptive", "inferential"), True, "comparisons_structure"),
    (("comparisons", "equivalence_margin"), 0.015, "no_equivalence_margin"),
    (("amendment_2", "decided_after_validation_A_results"), False, "amendment_2_disclosure_decided_after_A_validation"),
    (("amendment_2", "external_test_data_read_before_amendment"), True, "amendment_2_disclosure_decided_after_A_validation"),
    (("experiment_B_scope", "n_checkpoints"), 12, "experiment_B_scope_3_main_arms_9_checkpoints"),
    (("experiment_B_scope", "split"), "test", "experiment_B_scope_3_main_arms_9_checkpoints"),
    (("test_one_shot",), False, "test_one_shot"),
    (("test_plan", "authorization_tag"), "other", "test_one_shot"),
    (("protocol_version",), "v1.1", "protocol_version_v1_2"),
    (("freeze_state",), "final", "test_authorization_consistent"),
])
def test_verifier_fails_if_amendment_2_tampered(tmp_path, path, value, check):
    repo = _synthetic_repo(tmp_path)
    mp = repo / freeze.MANIFEST_REL
    m = json.loads(mp.read_text())
    d = m
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    mp.write_text(json.dumps(m))
    assert check in _fails(repo)


def test_verifier_accepts_final_authorized_but_not_mixed_states(tmp_path):
    repo = _synthetic_repo(tmp_path)
    mp = repo / freeze.MANIFEST_REL
    m = json.loads(mp.read_text())
    m["freeze_state"], m["test_set_authorized"] = "final", True
    mp.write_text(json.dumps(m))
    assert "test_authorization_consistent" not in _fails(repo)
    m["freeze_state"], m["test_set_authorized"] = "draft", True
    mp.write_text(json.dumps(m))
    assert "test_authorization_consistent" in _fails(repo)


@pytest.mark.parametrize("changes,check", [
    ({"epochs": 121}, "budget_epochs_120"),
    ({"max_updates": 7920}, "budget_max_updates_null"),
    ({"early_stopping": True}, "budget_early_stopping_false"),
    ({"updates_per_epoch": 65}, "budget_updates_per_epoch_66"),
    ({"total_updates": 7919}, "budget_total_updates_7920"),
    ({"validation_points": 1670}, "budget_validation_points_120"),
    ({"best_rule": "min, ties->last"}, "budget_tie_rule_documented"),
])
def test_verifier_fails_if_budget_tampered(tmp_path, changes, check):
    assert check in _tamper_budget(tmp_path, **changes)


def test_verifier_fails_if_superseded_budget_active(tmp_path):
    f = _tamper_budget(tmp_path, total_updates=110160, epochs=1669, validation_points=1670, updates_per_epoch=66)
    assert "budget_not_superseded_value" in f and "budget_total_updates_7920" in f
    repo = _synthetic_repo(tmp_path / "c")
    mp = repo / freeze.MANIFEST_REL
    m = json.loads(mp.read_text())
    m["superseded_budget"]["executed"] = True
    mp.write_text(json.dumps(m))
    assert "superseded_budget_flagged_never_executed" in _fails(repo)


def test_verifier_detects_authorized_test_and_missing(tmp_path):
    repo = _synthetic_repo(tmp_path)
    m = json.loads((repo / freeze.MANIFEST_REL).read_text())
    m["test_set_authorized"] = True
    (repo / freeze.MANIFEST_REL).write_text(json.dumps(m))
    assert "test_authorization_consistent" in _fails(repo)       # authorized while still draft
    os.remove(repo / freeze.FILES["dataset_h5"])
    assert "sha256:dataset_h5" in _fails(repo)
    shutil.rmtree(repo / "configs")
    assert freeze.verify(repo)[0][1] is False


def test_verifier_cli_exit_code(tmp_path):
    import subprocess
    import sys

    repo = _synthetic_repo(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/verify_external_freeze.py"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    assert subprocess.run([sys.executable, str(script), "--repo", str(repo)], env=env, capture_output=True, check=False).returncode == 0
    (repo / freeze.FILES["split_csv"]).write_text("tampered")
    assert subprocess.run([sys.executable, str(script), "--repo", str(repo)], env=env, capture_output=True, check=False).returncode == 1
