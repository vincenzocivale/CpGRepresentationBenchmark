"""Freeze manifest builder and read-only verifier for the external GSE40279 confirmation (torch-free)."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from .budget import external_schedule, steps_per_epoch, total_updates
from .io import sha256_file

MANIFEST_REL = "configs/external/gse40279_v1_freeze_manifest.json"
ARMS = ("regulatory_histone_dnase", "functional_annotations_pca", "cpgpt_large_locus", "deepcpg_dna_locus")
SEEDS = (17, 42, 97)
MASK_SEED = 17001
SPLIT_SEED = 20260925
BATCH_SIZE = 8
TCGA_EPOCHS = 120
EXPECTED_COUNTS = {"train": 524, "validation": 66, "test": 66}
EXPECTED_UNIVERSE = 408017
TCGA_PATIENTS_REL = "outputs/encode_atlas_v1/patients.npz"
PHASE_A_STATUS_REL = "outputs/regulatory_confirmation_v1/phase_A_status.json"
MATRIX_REL = "configs/experiments/regulatory_confirmation_matrix/matrix.yaml"

FILES = {  # key -> repo-relative path
    "dataset_h5": "data/derived/external_gse40279_v1/methylation.h5",
    "phenotypes_parquet": "data/derived/external_gse40279_v1/phenotypes.parquet",
    "cpg_mapping_external_parquet": "data/derived/external_gse40279_v1/cpg_mapping_external.parquet",
    "mapping_manifest": "configs/external/gse40279_v1_mapping_manifest.json",
    "locus_protocol_npz": "data/derived/external_gse40279_v1/loci_seen_external_gse40279_v1.npz",
    "patient_protocol_npz": "data/derived/external_gse40279_v1/patients_external_gse40279_v1.npz",
    "split_csv": "configs/external/gse40279_v1_split.csv",
    "split_manifest": "configs/external/gse40279_v1_split_manifest.json",
    "dataset_config": "configs/datasets/external_gse40279_v1.yaml",
}


def tcga_update_budget(repo: Path) -> dict:
    with np.load(repo / TCGA_PATIENTS_REL) as z:  # keys train only; sample axis not used
        n_train = len(z["train"])
    spe = steps_per_epoch(n_train, BATCH_SIZE)
    return {"tcga_n_train": n_train, "tcga_steps_per_epoch": spe, "tcga_epochs": TCGA_EPOCHS,
            "tcga_total_updates": total_updates(n_train, BATCH_SIZE, TCGA_EPOCHS)}


def arm_stores(repo: Path) -> dict:
    spec = yaml.safe_load((repo / MATRIX_REL).read_text())["matrix"]["arms"]
    by_id = {a["id"]: a for a in spec}
    return {a: {"store_h5": by_id[a]["store_h5"], "dim": by_id[a]["dim"], "matrix_yaml_sha256": by_id[a]["sha256"]}
            for a in ARMS}


def phase_a_checkpoints(repo: Path) -> list[dict]:
    st = json.loads((repo / PHASE_A_STATUS_REL).read_text())
    out = []
    for r in st["runs"]:
        run_dir = Path(r["run_dir"])
        rel = run_dir.relative_to(repo) if run_dir.is_absolute() else run_dir
        out.append({"arm": r["arm"], "seed": int(r["seed"]), "best_pt": str(rel / "checkpoints" / "best.pt"),
                    "best_pt_sha256": r["best_pt_sha256"]})
    return sorted(out, key=lambda x: (x["arm"], x["seed"]))


def build_manifest(repo: Path) -> dict:
    tc = tcga_update_budget(repo)
    sched = external_schedule(tc["tcga_total_updates"], steps_per_epoch(EXPECTED_COUNTS["train"], BATCH_SIZE))
    sched = {k: v for k, v in sched.items() if k != "validation_points"}
    arms = arm_stores(repo)
    for a in ARMS:
        arms[a]["store_sha256"] = arms[a]["matrix_yaml_sha256"]
    return {
        "schema": "external_gse40279_v1_freeze_manifest/1",
        "protocol_doc": "docs/EXTERNAL_RECONSTRUCTION_PROTOCOL.md",
        "freeze_state": "draft",
        "test_set_authorized": False,
        "files_sha256": {k: sha256_file(repo / v) for k, v in FILES.items()},
        "files": FILES,
        "counts": EXPECTED_COUNTS,
        "universe_size": EXPECTED_UNIVERSE,
        "budget": {**tc, "external_batch_size": BATCH_SIZE, "external_n_train": EXPECTED_COUNTS["train"],
                   "external_update_budget": tc["tcga_total_updates"], **sched,
                   "validation_cadence": "end of every full external epoch and at the final update",
                   "lr_schedule": "constant", "early_stopping": False},
        "seeds": list(SEEDS), "mask_seed": MASK_SEED, "patient_split_seed": SPLIT_SEED, "locus_seed": MASK_SEED,
        "mask_fractions": [0.15, 0.30, 0.50, 0.70, 0.90], "selection_mask_fraction": 0.50,
        "arms": arms,
        "phase_a_checkpoints_experiment_B": phase_a_checkpoints(repo),
    }


def verify(repo: Path, manifest_path: Path | None = None, *, check_stores: bool = True,
           check_checkpoints: bool = True) -> list[tuple[str, bool, str]]:
    """Recompute every fact recorded in the manifest. Strictly read-only. Returns [(check, ok, detail)]."""
    repo = Path(repo)
    manifest_path = Path(manifest_path) if manifest_path else repo / MANIFEST_REL
    res: list[tuple[str, bool, str]] = []

    def chk(name, ok, detail=""):
        res.append((name, bool(ok), str(detail)))

    try:
        m = json.loads(manifest_path.read_text())
    except Exception as e:  # noqa: BLE001
        return [("manifest_readable", False, repr(e))]
    chk("manifest_readable", True)
    chk("test_set_authorized_false", m.get("test_set_authorized") is False)
    chk("freeze_state_valid", m.get("freeze_state") in ("draft", "final"), m.get("freeze_state"))
    for k, rel in m.get("files", {}).items():
        p = repo / rel
        if not p.exists():
            chk(f"sha256:{k}", False, f"missing {rel}")
            continue
        got = sha256_file(p)
        chk(f"sha256:{k}", got == m["files_sha256"].get(k), got[:16])
    if check_stores:
        for a, s in m.get("arms", {}).items():
            p = repo / s["store_h5"]
            got = sha256_file(p) if p.exists() else "missing"
            chk(f"store_sha256:{a}", got == s["store_sha256"] == s["matrix_yaml_sha256"], got[:16])
    if check_checkpoints:
        for c in m.get("phase_a_checkpoints_experiment_B", []):
            p = repo / c["best_pt"]
            got = sha256_file(p) if p.exists() else "missing"
            chk(f"phaseA_best_pt:{c['arm']}:{c['seed']}", got == c["best_pt_sha256"], got[:16])
    # counts / partition / universe (cheap)
    try:
        with np.load(repo / m["files"]["patient_protocol_npz"]) as z:
            parts = {k: z[k] for k in ("train", "validation", "test")}
            names = z["sample_names"]
            seed = int(z["seed"])
        counts = {k: len(v) for k, v in parts.items()}
        chk("split_counts", counts == m["counts"] == EXPECTED_COUNTS, counts)
        joined = np.concatenate(list(parts.values()))
        chk("split_partition", np.array_equal(np.sort(joined), np.arange(len(names))))
        chk("split_seed", seed == m["patient_split_seed"] == SPLIT_SEED, seed)
        with h5py.File(repo / m["files"]["dataset_h5"], "r") as h:
            hn = [x.decode() if isinstance(x, bytes) else str(x) for x in h["sample_name"][:]]
            ids = np.asarray(h["cpg_idx"][:], dtype=np.int64)
            shape = tuple(h["beta"].shape)
        chk("h5_sample_axis_matches_protocol", list(names) == hn and len(set(hn)) == len(hn))
        chk("h5_shape", shape == (len(names), len(ids)), shape)
        with np.load(repo / m["files"]["locus_protocol_npz"]) as z:
            tr, ho = z["train_cpg_idx"], z["heldout_cpg_idx"]
        chk("universe_size", len(tr) == m["universe_size"] == EXPECTED_UNIVERSE and len(ho) == 0, len(tr))
        chk("universe_equals_matrix_axis", np.array_equal(np.sort(tr), np.sort(ids)))
    except Exception as e:  # noqa: BLE001
        chk("split_locus_checks", False, repr(e))
    # budget constant
    try:
        tc = tcga_update_budget(repo)
        b = m["budget"]
        sch = external_schedule(tc["tcga_total_updates"], steps_per_epoch(EXPECTED_COUNTS["train"], BATCH_SIZE))
        chk("budget_tcga_recomputed", tc["tcga_total_updates"] == b["tcga_total_updates"] == 110160, tc["tcga_total_updates"])
        chk("budget_external_equals_tcga", b["external_update_budget"] == tc["tcga_total_updates"])
        chk("budget_schedule", (sch["full_epochs"], sch["partial_epoch_updates"], sch["n_validation_points"]) ==
            (b["full_epochs"], b["partial_epoch_updates"], b["n_validation_points"]) == (1669, 6, 1670),
            (sch["full_epochs"], sch["partial_epoch_updates"], sch["n_validation_points"]))
        chk("seeds_mask", m["seeds"] == list(SEEDS) and m["mask_seed"] == MASK_SEED)
    except Exception as e:  # noqa: BLE001
        chk("budget", False, repr(e))
    return res
