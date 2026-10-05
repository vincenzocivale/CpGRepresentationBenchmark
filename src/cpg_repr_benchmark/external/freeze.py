"""Freeze manifest builder and read-only verifier for the external GSE40279 confirmation (torch-free)."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from .budget import (
    EXTERNAL_EPOCHS,
    SUPERSEDED_BUDGET,
    epoch_budget,
    external_schedule,
    steps_per_epoch,
    total_updates,
)
from .io import sha256_file

PROTOCOL_REL = "docs/EXTERNAL_RECONSTRUCTION_PROTOCOL.md"
PROTOCOL_VERSION = "v1.2"
BEST_RULE = "strict_min_val_mse@0.50, ties->first epoch"
AMENDMENT = {   # AMENDMENT 1 record, kept verbatim (manifest key `amendment`); AMENDMENT 2 is `amendment_2` below
    "id": "AMENDMENT 1", "date": "2026-10-05", "kind": "pre-run (no external run executed before it)",
    "from_version": "v1.0 (commit 2343a79, tag external-recon-protocol-freeze-v1)", "to_version": "v1.1",
    "changed": "training budget: equal-update (110,160 updates) replaced by equal-epoch (120 epochs, 7,920 updates)",
    "rationale": "equal updates gave 1,669 epochs + 6 updates over 524 subjects, far more per-patient exposures than TCGA; "
                 "the amended budget keeps the number of epochs equal to TCGA; absolute update count deliberately not matched",
}
MANIFEST_REL = "configs/external/gse40279_v1_freeze_manifest.json"
ARMS = ("regulatory_histone_dnase", "functional_annotations_pca", "cpgpt_large_locus", "deepcpg_dna_locus")   # Experiment A trained set (frozen order)
# AMENDMENT 2 (v1.2): main comparator panel of the paper; `regulatory_histone_dnase` is the repo id of `regulatory_histone_dnase_v1`.
CANDIDATE = "regulatory_histone_dnase"
MAIN_ARMS = ("regulatory_histone_dnase", "cpgpt_large_locus", "deepcpg_dna_locus")
LEGACY_ARMS = ("functional_annotations_pca",)
PRIMARY_COMPARATOR = "cpgpt_large_locus"
SECONDARY_COMPARATOR = "deepcpg_dna_locus"
TAG_AUTHORIZATION = "external-recon-test-authorization-v1"
AMENDMENT_2_DATE = "2026-10-05"
COMPARISONS = {
    "primary": {"contrast": "cpgpt_large_locus - regulatory_histone_dnase", "candidate": CANDIDATE, "comparator": PRIMARY_COMPARATOR,
                "metric": "mse", "mask_fraction": 0.5, "inferential": True, "n_contrasts": 1,
                "multiplicity_adjustment": "none needed (single contrast)",
                "uncertainty": "paired bootstrap patients x 1 Mb genomic blocks, 2000 replicates, seed 17, per seed"},
    "secondary": {"contrast": "deepcpg_dna_locus - regulatory_histone_dnase", "candidate": CANDIDATE, "comparator": SECONDARY_COMPARATOR,
                  "metric": "mse", "mask_fraction": 0.5, "inferential": True, "label": "secondary",
                  "multiplicity_adjustment": "none (reported with its paired CI, labelled secondary)",
                  "uncertainty": "paired bootstrap patients x 1 Mb genomic blocks, 2000 replicates, seed 17, per seed"},
    "descriptive": {"contrast": "deepcpg_dna_locus - cpgpt_large_locus", "inferential": False,
                    "label": "descriptive secondary, outside every inferential comparison"},
    "secondary_metrics": ["mae", "mas_pcc", "mac_pcc"],
    "equivalence_margin": None,
    "equivalence_or_non_inferiority_claims": False,
}
LEGACY_SENSITIVITY_CONTROL = {
    "arms": list(LEGACY_ARMS), "role": "legacy_sensitivity_control", "in_main_inferential_comparison": False,
    "may_select_representation": False, "may_change_main_claim": False,
    "reason": "belongs to the legacy functional representation with a different feature contract (user rationale)",
    "evaluated_on_test": "yes if technically simple, as a separately labelled block; never in the main inferential family",
}
EXPERIMENT_B_SCOPE = {
    "arms": list(MAIN_ARMS), "seeds": [17, 42, 97], "n_checkpoints": 9, "modes": ["B_strict", "B_recalibrated"],
    "split": "validation", "test": "NOT planned: B on test is not part of this endgame unless separately authorized",
    "legacy_default": "excluded (only via an explicit --include-legacy, labelled; not planned)",
    "results_may_modify": "nothing (not Experiment A, not the protocol, not the arms tested)",
}
TEST_PLAN = {
    "one_shot": True, "split": "test", "n_subjects": 66, "mask_fractions": [0.15, 0.30, 0.50, 0.70, 0.90],
    "checkpoints": {"main": 9, "legacy_sensitivity_control": 3, "source": "the best.pt of the 12 Experiment A runs, selected on validation"},
    "authorization_tag": TAG_AUTHORIZATION,
    "recorded": ["checkpoint sha256", "split and locus-universe hashes", "protocol and authorization tag commits", "timestamp",
                 "authorization state", "arm role (main | legacy_sensitivity_control)"],
    "forbidden_after_test": ["tuning", "checkpoint re-selection", "retraining", "mask or panel changes", "repetition of the test evaluation"],
}
AMENDMENT_2 = {
    "id": "AMENDMENT 2", "date": AMENDMENT_2_DATE, "from_version": "v1.1 (tag external-recon-protocol-freeze-v1.1)", "to_version": "v1.2",
    "kind": "pre-B, pre-test (Experiment A validation results HAD been seen; no external TEST data was read)",
    "decided_after_validation_A_results": True,
    "external_test_data_read_before_amendment": False,
    "changed": "main comparator panel and comparison structure: functional_annotations_pca demoted to legacy_sensitivity_control; "
               "primary comparison = candidate vs cpgpt_large_locus (single contrast); secondary = candidate vs deepcpg_dna_locus; "
               "B scope = 3 main arms; one-shot test plan",
    "rationale": "functional_annotations_pca belongs to the legacy functional representation with a different feature contract "
                 "(stated by the user)",
    "not_pre_registered_before_A": True,
}
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
        out.append({"arm": r["arm"], "seed": int(r["seed"]), "role": "main" if r["arm"] in MAIN_ARMS else "legacy_sensitivity_control",
                    "best_pt": str(rel / "checkpoints" / "best.pt"),
                    "best_pt_sha256": r["best_pt_sha256"]})
    return sorted(out, key=lambda x: (x["arm"], x["seed"]))


def build_manifest(repo: Path) -> dict:
    tc = tcga_update_budget(repo)
    sched = external_schedule(tc["tcga_total_updates"], steps_per_epoch(EXPECTED_COUNTS["train"], BATCH_SIZE))
    sup = {"updates": sched["total_updates"], "epochs_full": sched["full_epochs"],
           "partial_updates": sched["partial_epoch_updates"], "validation_points": sched["n_validation_points"],
           "executed": False, "status": "SUPERSEDED by AMENDMENT 1 (pre-run); never executed"}
    arms = arm_stores(repo)
    for a in ARMS:
        arms[a]["store_sha256"] = arms[a]["matrix_yaml_sha256"]
    return {
        "schema": "external_gse40279_v1_freeze_manifest/1",
        "protocol_doc": PROTOCOL_REL,
        "protocol_doc_sha256": sha256_file(repo / PROTOCOL_REL) if (repo / PROTOCOL_REL).exists() else None,
        "protocol_version": PROTOCOL_VERSION,
        "amendment": AMENDMENT,
        "amendment_2": AMENDMENT_2,
        "main_panel": list(MAIN_ARMS),
        "legacy_sensitivity_control": LEGACY_SENSITIVITY_CONTROL,
        "comparisons": COMPARISONS,
        "experiment_B_scope": EXPERIMENT_B_SCOPE,
        "test_one_shot": True,
        "test_plan": TEST_PLAN,
        "freeze_state": "draft",
        "test_set_authorized": False,
        "files_sha256": {k: sha256_file(repo / v) for k, v in FILES.items()},
        "files": FILES,
        "counts": EXPECTED_COUNTS,
        "universe_size": EXPECTED_UNIVERSE,
        "budget": {**epoch_budget(EXPECTED_COUNTS["train"], BATCH_SIZE, EXTERNAL_EPOCHS),
                   "best_rule": BEST_RULE, "lr_schedule": "constant", "external_n_train": EXPECTED_COUNTS["train"],
                   **tc, "update_ratio_external_over_tcga": round(7920 / tc["tcga_total_updates"], 4),
                   "validation_cadence": "end of each of the 120 epochs (validation MSE @ mask 0.50)"},
        "superseded_budget": sup,
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
    # draft/false until the dated promotion commit; then final/true. Any mixed state is unsafe.
    chk("test_authorization_consistent", (m.get("freeze_state"), m.get("test_set_authorized")) in (("draft", False), ("final", True)),
        f"freeze_state={m.get('freeze_state')!r} test_set_authorized={m.get('test_set_authorized')!r}")
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
    # budget (AMENDMENT 1: epoch-based; the superseded 110,160-update budget must not be active)
    try:
        tc = tcga_update_budget(repo)
        b = m["budget"]
        eb = epoch_budget(EXPECTED_COUNTS["train"], BATCH_SIZE, EXTERNAL_EPOCHS)
        chk("budget_tcga_recomputed", tc["tcga_total_updates"] == b["tcga_total_updates"] == 110160, tc["tcga_total_updates"])
        chk("budget_epochs_120", b.get("epochs") == 120 == eb["epochs"], b.get("epochs"))
        chk("budget_max_updates_null", "max_updates" in b and b["max_updates"] is None, b.get("max_updates", "absent"))
        chk("budget_early_stopping_false", b.get("early_stopping") is False)
        chk("budget_batch_size_8", b.get("batch_size") == 8 == BATCH_SIZE)
        chk("budget_updates_per_epoch_66", b.get("updates_per_epoch") == 66 == eb["updates_per_epoch"], b.get("updates_per_epoch"))
        chk("budget_total_updates_7920", b.get("total_updates") == 7920 == eb["total_updates"] == 66 * 120, b.get("total_updates"))
        chk("budget_validation_points_120", b.get("validation_points") == 120 == eb["validation_points"], b.get("validation_points"))
        chk("budget_tie_rule_documented", b.get("best_rule") == BEST_RULE, b.get("best_rule"))
        chk("budget_not_superseded_value", b.get("total_updates") != SUPERSEDED_BUDGET["updates"]
            and b.get("epochs") != SUPERSEDED_BUDGET["epochs_full"] and b.get("validation_points") != SUPERSEDED_BUDGET["validation_points"])
        sb = m.get("superseded_budget", {})
        chk("superseded_budget_flagged_never_executed", sb.get("updates") == 110160 and sb.get("epochs_full") == 1669
            and sb.get("partial_updates") == 6 and sb.get("executed") is False, sb)
        chk("protocol_version_v1_2", m.get("protocol_version") == PROTOCOL_VERSION == "v1.2", m.get("protocol_version"))
        am = m.get("amendment", {})
        chk("amendment_recorded", am.get("id") == "AMENDMENT 1" and am.get("date") == "2026-10-05"
            and "pre-run" in am.get("kind", ""), am.get("id"))
        a2 = m.get("amendment_2", {})
        chk("amendment_2_recorded", a2.get("id") == "AMENDMENT 2" and a2.get("date") == AMENDMENT_2_DATE and a2.get("to_version") == "v1.2"
            and "pre-test" in a2.get("kind", ""), a2.get("id"))
        chk("amendment_2_disclosure_decided_after_A_validation", a2.get("decided_after_validation_A_results") is True
            and a2.get("external_test_data_read_before_amendment") is False and a2.get("not_pre_registered_before_A") is True
            and bool(a2.get("rationale")))
        chk("main_panel_three_arms", m.get("main_panel") == list(MAIN_ARMS) and CANDIDATE in m["main_panel"], m.get("main_panel"))
        lg = m.get("legacy_sensitivity_control", {})
        chk("legacy_control_functional_outside_main", lg.get("arms") == list(LEGACY_ARMS) and lg.get("role") == "legacy_sensitivity_control"
            and lg.get("in_main_inferential_comparison") is False and lg.get("may_select_representation") is False
            and lg.get("may_change_main_claim") is False and not set(lg.get("arms", [])) & set(m.get("main_panel", [])), lg.get("arms"))
        cp = m.get("comparisons", {})
        chk("comparisons_structure", cp.get("primary", {}).get("comparator") == PRIMARY_COMPARATOR
            and cp["primary"].get("candidate") == CANDIDATE and cp["primary"].get("n_contrasts") == 1 and cp["primary"].get("inferential") is True
            and cp.get("secondary", {}).get("comparator") == SECONDARY_COMPARATOR and cp["secondary"].get("inferential") is True
            and cp.get("descriptive", {}).get("inferential") is False
            and cp["primary"].get("metric") == "mse" and cp["primary"].get("mask_fraction") == 0.5
            and cp.get("secondary_metrics") == ["mae", "mas_pcc", "mac_pcc"], "primary/secondary/descriptive")
        chk("no_equivalence_margin", cp.get("equivalence_margin") is None and cp.get("equivalence_or_non_inferiority_claims") is False)
        chk("functional_not_in_any_inferential_comparison", "functional_annotations_pca" not in
            {cp.get("primary", {}).get("comparator"), cp.get("secondary", {}).get("comparator")} | set(m.get("main_panel", [])))
        eb = m.get("experiment_B_scope", {})
        b_arms = {c["arm"] for c in m.get("phase_a_checkpoints_experiment_B", []) if c.get("role") == "main"}
        chk("experiment_B_scope_3_main_arms_9_checkpoints", eb.get("arms") == list(MAIN_ARMS) and eb.get("n_checkpoints") == 9
            and eb.get("split") == "validation" and "NOT planned" in eb.get("test", "") and b_arms == set(MAIN_ARMS)
            and sum(c.get("role") == "main" for c in m.get("phase_a_checkpoints_experiment_B", [])) == 9, eb.get("arms"))
        tp = m.get("test_plan", {})
        chk("test_one_shot", m.get("test_one_shot") is True and tp.get("one_shot") is True and tp.get("n_subjects") == 66
            and tp.get("checkpoints", {}).get("main") == 9 and tp["checkpoints"].get("legacy_sensitivity_control") == 3
            and tp.get("authorization_tag") == TAG_AUTHORIZATION and tp.get("mask_fractions") == [0.15, 0.30, 0.50, 0.70, 0.90]
            and bool(tp.get("forbidden_after_test")), tp.get("authorization_tag"))
        if m.get("protocol_doc_sha256") is not None:
            pd = repo / m.get("protocol_doc", PROTOCOL_REL)
            got = sha256_file(pd) if pd.exists() else "missing"
            chk("protocol_doc_sha256", got == m["protocol_doc_sha256"], got[:16])
        chk("seeds_mask", m["seeds"] == list(SEEDS) and m["mask_seed"] == MASK_SEED)
    except Exception as e:  # noqa: BLE001
        chk("budget", False, repr(e))
    return res
