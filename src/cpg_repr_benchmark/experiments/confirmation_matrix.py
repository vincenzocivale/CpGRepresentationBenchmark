"""Frozen confirmation matrix: spec loading, pre-test audit, per-run config construction and run gating.

Pure Python (no torch, no GPU). The only data this module opens are HDF5 `/cpg_idx` datasets and the shapes/dtypes of
`/embedding` (metadata), the `train_cpg_idx` of `loci_seen.npz`, and file bytes for sha256. It never reads methylation
values, predictions, or any patient/test data. See docs/REGULATORY_CONFIRMATION_PROTOCOL.md (AMENDMENT 2) and
docs/REGULATORY_CONFIRMATION_FAIRNESS.md.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cpg_repr_benchmark.experiments.guards import require_test_authorization

MATRIX_PATH = Path("configs/experiments/regulatory_confirmation_matrix/matrix.yaml")
CONFIG_DIR = Path("configs/experiments/regulatory_confirmation_matrix/runs")
OUTPUT_ROOT = Path("outputs/regulatory_confirmation_v1")

# ----------------------------------------------------------------------------- frozen specification (user, 2026-10-03)
FROZEN_SEEDS = [17, 42, 97]
FROZEN_FRACTIONS = [0.15, 0.30, 0.50, 0.70, 0.90]
FROZEN_PROTOCOL: dict[str, Any] = {
    "max_epochs": 120,
    "early_stopping": False,
    "checkpoint_selection": "best_val_mse",
    "checkpoint_selection_mask_fraction": 0.50,
    "batch_size": 8,
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-4,
    "mixed_precision": True,
    "lr_schedule": "constant",
    "panel_size": 2048,
    "panel_repeats": 1,
    "mask_seed": 17001,
    "num_workers": 8,
    "heldout_fraction": 0.0,
    "patient_split_seed": 20260925,
    "template": "configs/experiments/masking/functional_pca_genomewide.yaml",
    "patient_split_path": "outputs/encode_atlas_v1/patients.npz",
    "loci_split_path": "outputs/encode_atlas_v1/loci_seen.npz",
}
MAIN_ARM_IDS = [
    "regulatory_histone_dnase", "functional_annotations_pca", "cpgpt_large_locus", "deepcpg_dna_locus",
    "modern_sequence_fm", "modern_sequence_fm_alt",
]
CANDIDATE_ID = "regulatory_histone_dnase"
PROVENANCE_REQUIRED = ["patient_specific", "supervision", "locus_fit_scope", "genome_build", "source"]
REGISTRATION_REQUIRED = [
    "model_name", "checkpoint", "checkpoint_sha256", "version_revision", "input_window_bp", "cpg_position_in_window",
    "strand_handling", "pooling_rule", "extracted_layer", "embedding_dim", "preprocessing_tokenization",
    "genome_build", "missing_failed_loci_policy", "projection_compression", "decided_by", "decided_on",
]
REGISTRATION_ACK = "no_choice_optimized_with_tcga_methylation_reconstruction"
FREEZE_STATES = ("draft", "final")
# audit failure codes that `run --allow-incomplete-validation-only` may tolerate (nothing else, and never for test)
TOLERABLE_CODES = {"PENDING_COMPARATOR", "REGISTRATION_INCOMPLETE"}
# Phase A (validation-only training of the 4 fully registered main arms; the pending modern FM slots are NOT part of it).
# Order matters: seed-major, arms in this order (the 512D CpGPT store is exercised early, in seed 17).
PHASE_A_ARMS = ["regulatory_histone_dnase", "functional_annotations_pca", "cpgpt_large_locus", "deepcpg_dna_locus"]
PHASE_A_SPLIT = "validation"
STATUS_VOCABULARY = ("trained_validation_frozen",)   # NEVER 'confirmed': confirmation needs the final, authorized test evaluation
STATUS_FILE = "confirmation_status.json"
PHASE_A_STATUS_FILE = "phase_A_status.json"
OUTPUT_LAYOUT = "split_dirs"
TRAIN_ALLOWED = {"seed", "epochs", "num_workers", "early_stopping"}
EVAL_ALLOWED = {"mask_fractions", "save_predictions", "mask_seed", "patient_view", "panel_repeats",
                "require_patient_view", "num_workers", "output_layout", "allow_overwrite_split_dir"}
# any of these keys inside an arm's `protocol_overrides` is a protocol deviation (the shared protocol is the only one)
PLACEHOLDER_PREFIXES = ("todo", "tbd", "<", "xxx", "fill me", "fixme")


@dataclass
class Result:
    code: str
    scope: str
    ok: bool
    message: str

    def line(self) -> str:
        return f"[{'PASS' if self.ok else 'FAIL'}] {self.code:<22} {self.scope:<34} {self.message}"


def load_matrix(path: Path | str) -> dict:
    return yaml.safe_load(Path(path).read_text())["matrix"]


def sha256_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def resolve(root: Path, rel: str | None) -> Path | None:
    return None if rel is None else (root / rel)


# ----------------------------------------------------------------------------- registration template check
def registration_problems(reg: dict | None) -> list[str]:
    """Fields of a sequence-FM registration that are missing/placeholder. Empty list == complete."""
    if not isinstance(reg, dict):
        return ["registration is not a mapping"]
    reg = reg.get("registration", reg)
    problems = []
    for key in REGISTRATION_REQUIRED:
        v = reg.get(key)
        if v is None or (isinstance(v, str) and (not v.strip() or v.strip().lower().startswith(PLACEHOLDER_PREFIXES))):
            problems.append(key)
    if reg.get(REGISTRATION_ACK) is not True:
        problems.append(REGISTRATION_ACK)
    return problems


# ----------------------------------------------------------------------------- audit
def _same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-12
        except (TypeError, ValueError):
            return False
    return a == b


def _catalog_has(root: Path, ref: str | None) -> bool:
    if not ref or "#" not in ref:
        return False
    path, key = ref.split("#", 1)
    p = root / path
    if not p.is_file():
        return False
    return key in (yaml.safe_load(p.read_text()) or {}).get("representations", {})


def audit_protocol(spec: dict, root: Path) -> list[Result]:
    out: list[Result] = []
    if list(spec.get("seeds") or []) != FROZEN_SEEDS:
        out.append(Result("SEEDS", "matrix.seeds", False, f"seeds {spec.get('seeds')!r} != frozen {FROZEN_SEEDS}"))
    else:
        out.append(Result("SEEDS", "matrix.seeds", True, f"{FROZEN_SEEDS}"))
    fr = [float(x) for x in spec.get("mask_fractions") or []]
    if fr != FROZEN_FRACTIONS:
        out.append(Result("FRACTIONS", "matrix.mask_fractions", False, f"{spec.get('mask_fractions')!r} != frozen {FROZEN_FRACTIONS}"))
    else:
        out.append(Result("FRACTIONS", "matrix.mask_fractions", True, f"{FROZEN_FRACTIONS}"))
    proto = spec.get("protocol") or {}
    bad = [f"{k}={proto.get(k)!r} (frozen {v!r})" for k, v in FROZEN_PROTOCOL.items()
           if k not in proto or not _same(proto[k], v)]
    out.append(Result("PROTOCOL", "matrix.protocol", not bad, "shared protocol equals frozen values" if not bad else "; ".join(bad)))
    # per-arm deviations
    for arm in spec.get("arms", []):
        ov = arm.get("protocol_overrides") or {}
        out.append(Result("PROTOCOL", f"arm:{arm.get('id')}", not ov,
                          "uses the shared protocol" if not ov else f"arm-specific protocol overrides are forbidden: {sorted(ov)}"))
    # shared splits
    for key, label in (("patient_split", "patients"), ("loci_split", "loci")):
        sp = (spec.get("splits") or {}).get(key) or {}
        frozen_path = FROZEN_PROTOCOL["patient_split_path" if key == "patient_split" else "loci_split_path"]
        ok = sp.get("path") == frozen_path
        msg = f"path {sp.get('path')!r}" if ok else f"path {sp.get('path')!r} != frozen {frozen_path!r}"
        p = resolve(root, sp.get("path"))
        if ok and (p is None or not p.is_file()):
            ok, msg = False, f"{sp.get('path')} missing"
        if ok:
            rec = sp.get("sha256")
            if not rec:
                ok, msg = False, "sha256 not recorded"
            elif sha256_file(p) != rec:
                ok, msg = False, "sha256 does not match the file"
        out.append(Result("PROTOCOL", f"split:{label}", ok, msg if not ok else f"{msg}; sha256 verified"))
    return out


def audit_arm(arm: dict, root: Path, *, train_idx=None, check_hashes: bool = True, check_coverage: bool = True) -> list[Result]:
    aid = arm.get("id", "?")
    scope = f"arm:{aid}"
    out: list[Result] = []
    if arm.get("status") == "pending":
        out.append(Result("PENDING_COMPARATOR", scope, False, f"comparator '{aid}' is still pending (no decided + materialized store)"))
        if arm.get("requires_registration") or arm.get("registration"):
            reg_path = resolve(root, arm.get("registration"))
            if reg_path is None or not reg_path.is_file():
                out.append(Result("REGISTRATION_INCOMPLETE", scope, False,
                                  "no completed registration file (copy sequence_fm_registration.template.yaml and fill ALL fields)"))
            else:
                probs = registration_problems(yaml.safe_load(reg_path.read_text()))
                out.append(Result("REGISTRATION_INCOMPLETE", scope, not probs,
                                  "registration complete" if not probs else f"registration incomplete: {probs}"))
        return out
    # catalog
    ok = _catalog_has(root, arm.get("catalog"))
    out.append(Result("CATALOG_MISSING", scope, ok, f"catalog entry {arm.get('catalog')}" if ok else
                      f"catalog entry {arm.get('catalog')!r} missing"))
    # provenance
    prov = arm.get("provenance") or {}
    missing = [k for k in PROVENANCE_REQUIRED if prov.get(k) in (None, "")]
    if prov.get("patient_specific") is not False:
        missing = sorted(set(missing) | {"patient_specific(must be false)"})
    out.append(Result("PROVENANCE_MISSING", scope, not missing, "provenance fields present" if not missing else f"missing/invalid: {missing}"))
    # registration of a materialized sequence FM
    if arm.get("requires_registration"):
        reg_path = resolve(root, arm.get("registration"))
        probs = ["registration file missing"] if reg_path is None or not reg_path.is_file() \
            else registration_problems(yaml.safe_load(reg_path.read_text()))
        out.append(Result("REGISTRATION_INCOMPLETE", scope, not probs, "registration complete" if not probs else f"registration incomplete: {probs}"))
    # store
    store = resolve(root, arm.get("store_h5"))
    present = store is not None and store.is_file()
    if not arm.get("store_present") or not present:
        out.append(Result("STORE_MISSING", scope, False,
                          f"store {arm.get('store_h5')!r}: store_present={arm.get('store_present')!r}, file {'found' if present else 'missing'}"))
        return out
    out.append(Result("STORE_MISSING", scope, True, f"store present: {arm['store_h5']}"))
    rec = arm.get("sha256")
    if not rec:
        out.append(Result("HASH_MISSING", scope, False, "sha256 of the store is not recorded"))
    elif check_hashes:
        actual = sha256_file(store)
        out.append(Result("HASH_MISMATCH", scope, actual == rec,
                          "sha256 verified against the file" if actual == rec else f"recorded {rec[:12]}... != actual {actual[:12]}..."))
    else:
        out.append(Result("HASH_MISSING", scope, True, "sha256 recorded (re-hash skipped by flag)"))
    if check_coverage:
        out.extend(_coverage(arm, store, train_idx))
    return out


def _coverage(arm: dict, store: Path, train_idx) -> list[Result]:
    import h5py
    import numpy as np
    scope = f"arm:{arm['id']}"
    try:
        with h5py.File(store, "r") as h:
            idx = h["cpg_idx"][:]
            shape, dtype = h["embedding"].shape, str(h["embedding"].dtype)
    except Exception as exc:  # noqa: BLE001 - unreadable store is an audit failure, not a crash
        return [Result("COVERAGE", scope, False, f"cannot read /cpg_idx,/embedding: {exc}")]
    problems = []
    if train_idx is None:
        problems.append("protocol loci unavailable")
    elif not np.isin(train_idx, idx).all():
        problems.append(f"{int((~np.isin(train_idx, idx)).sum())} of {len(train_idx)} protocol loci missing from /cpg_idx")
    if len(np.unique(idx)) != len(idx):
        problems.append("duplicate cpg_idx")
    if shape[0] != len(idx):
        problems.append(f"embedding rows {shape[0]} != len(cpg_idx) {len(idx)}")
    if shape[1] != arm.get("dim"):
        problems.append(f"embedding dim {shape[1]} != recorded {arm.get('dim')}")
    if dtype != arm.get("dtype"):
        problems.append(f"dtype {dtype} != recorded {arm.get('dtype')}")
    return [Result("COVERAGE", scope, not problems,
                   f"covers all {len(train_idx)} protocol loci; shape {tuple(shape)} {dtype}" if not problems else "; ".join(problems))]


def audit_roles(spec: dict) -> list[Result]:
    arms = spec.get("arms", [])
    main = [a["id"] for a in arms if a.get("role") == "main"]
    sens = [a for a in arms if a.get("role") == "sensitivity"]
    other = [a.get("id") for a in arms if a.get("role") not in ("main", "sensitivity")]
    probs = []
    if sorted(main) != sorted(MAIN_ARM_IDS):
        probs.append(f"main arms {sorted(main)} != required {sorted(MAIN_ARM_IDS)}")
    if other:
        probs.append(f"arms without role main|sensitivity: {other}")
    ids = [a.get("id") for a in arms]
    if len(set(ids)) != len(ids):
        probs.append("duplicate arm ids")
    for a in sens:
        if a.get("main_arm") not in MAIN_ARM_IDS:
            probs.append(f"sensitivity arm {a.get('id')} lacks a valid main_arm")
    cand = next((a for a in arms if a.get("id") == CANDIDATE_ID), None)
    if cand is None or cand.get("dim") != 256:
        probs.append("frozen candidate regulatory_histone_dnase (256D) missing")
    return [Result("ROLES", "matrix.arms", not probs, "6 main comparators (+ sensitivity listed separately)" if not probs else "; ".join(probs))]


def audit_state(spec: dict, others: list[Result]) -> list[Result]:
    state = spec.get("freeze_state")
    authorized = spec.get("test_set_authorized")
    fails = [r for r in others if not r.ok]
    out = []
    if state not in FREEZE_STATES:
        out.append(Result("FREEZE_STATE", "matrix.freeze_state", False, f"freeze_state {state!r} not in {FREEZE_STATES}"))
    if not isinstance(authorized, bool):
        out.append(Result("TEST_AUTH", "matrix.test_set_authorized", False, f"must be a boolean, got {authorized!r}"))
    elif authorized and (state != "final" or fails):
        out.append(Result("TEST_AUTH", "matrix.test_set_authorized", False,
                          f"test_set_authorized is true but freeze_state={state!r} and {len(fails)} audit failure(s); "
                          "test requires freeze_state final AND an all-green audit"))
    else:
        out.append(Result("TEST_AUTH", "matrix.test_set_authorized", True,
                          f"test_set_authorized={authorized} (freeze_state={state})"))
    if state == "final" and fails:
        out.append(Result("FREEZE_STATE", "matrix.freeze_state", False,
                          f"freeze_state is 'final' but {len(fails)} audit check(s) fail"))
    elif state in FREEZE_STATES:
        out.append(Result("FREEZE_STATE", "matrix.freeze_state", True, f"freeze_state={state}"))
    return out


def audit(spec: dict, root: Path, *, check_hashes: bool = True, check_coverage: bool = True,
          check_configs: bool = False) -> list[Result]:
    root = Path(root)
    results = audit_protocol(spec, root) + audit_roles(spec)
    train_idx = None
    loci = resolve(root, (spec.get("splits") or {}).get("loci_split", {}).get("path"))
    if check_coverage and loci is not None and loci.is_file():
        import numpy as np
        with np.load(loci) as h:
            train_idx = np.sort(h["train_cpg_idx"])
    for arm in spec.get("arms", []):
        results += audit_arm(arm, root, train_idx=train_idx, check_hashes=check_hashes, check_coverage=check_coverage)
    if check_configs:
        results += audit_configs(spec, root)
    return results + audit_state(spec, results)


def failures(results: list[Result]) -> list[Result]:
    return [r for r in results if not r.ok]


# ----------------------------------------------------------------------------- per-run configs
def arms_by_id(spec: dict) -> dict[str, dict]:
    return {a["id"]: a for a in spec["arms"]}


def job_arms(spec: dict, *, include_sensitivity: bool = False, only: list[str] | None = None,
             phase: str | None = None) -> list[dict]:
    """Runnable arms (never `pending`): main first, then (optionally) sensitivity. `phase="A"` = the 4 PHASE_A_ARMS in order only."""
    if phase is not None:
        if phase != "A":
            raise ValueError(f"unknown phase {phase!r} (only 'A')")
        byid = arms_by_id(spec)
        arms = [byid[i] for i in PHASE_A_ARMS if i in byid and byid[i].get("status") != "pending" and byid[i].get("role") == "main"]
        return [a for a in arms if only is None or a["id"] in only]
    arms = [a for a in spec["arms"] if a.get("status") != "pending" and a.get("role") == "main"]
    if include_sensitivity:
        arms += [a for a in spec["arms"] if a.get("status") != "pending" and a.get("role") == "sensitivity"]
    return [a for a in arms if only is None or a["id"] in only]


def jobs(spec: dict, *, seeds=None, **kw) -> list[tuple[dict, int]]:
    """Seed-major: all arms of seed 17, then 42, then 97."""
    arms = job_arms(spec, **kw)
    return [(a, s) for s in (seeds or spec["seeds"]) for a in arms]


def config_name(arm_id: str, seed: int) -> str:
    return f"{arm_id}__seed{seed}.yaml"


def build_config(spec: dict, arm: dict, seed: int, root: Path, *, split: str = "validation") -> dict:
    if split != "validation":
        raise PermissionError("per-run configs exist for the validation split only; test configs are never generated")
    p = spec["protocol"]
    cfg = yaml.safe_load((Path(root) / p["template"]).read_text())
    cfg["experiment"]["name"] = f"regulatory_confirmation_matrix_{arm['id']}__seed{seed}"
    cfg["experiment"]["output_root"] = str(OUTPUT_ROOT / "benchmark")
    cfg["experiment"]["locus_split"] = {"heldout_fraction": p["heldout_fraction"], "seed": p["mask_seed"],
                                        "protocol_path": p["loci_split_path"]}
    cfg["dataset"].update(patient_protocol=p["patient_split_path"], patient_split_seed=p["patient_split_seed"])
    prov = arm["provenance"]
    cfg["representation"].update(
        name=arm["id"], family=arm.get("family", cfg["representation"].get("family")), store_h5=arm["store_h5"],
        source=f"regulatory confirmation matrix ({arm['role']}): {prov['source']}",
        provenance={"patient_specific": False, "supervision": prov["supervision"], "role": arm["role"],
                    "locus_fit_scope": prov["locus_fit_scope"], "dim": arm["dim"], "sha256": arm["sha256"]})
    cfg["training"].update(seed=seed, epochs=p["max_epochs"], num_workers=p["num_workers"],
                           early_stopping=p["early_stopping"])
    cfg["evaluation"].update(mask_fractions=list(spec["mask_fractions"]), save_predictions=True, mask_seed=p["mask_seed"],
                             selection_mask_fraction=p["checkpoint_selection_mask_fraction"], num_workers=0,
                             patient_view="validation", panel_repeats=p["panel_repeats"], require_patient_view="validation",
                             output_layout=OUTPUT_LAYOUT, allow_overwrite_split_dir=False)
    return cfg


def check_config(cfg: dict, template: dict, spec: dict, arm_id: str, seed: int) -> None:
    from cpg_repr_benchmark.experiments.guards import enforce_patient_view
    p = spec["protocol"]
    w = f"{arm_id}/seed{seed}"
    enforce_patient_view(cfg)
    ev, tr = cfg["evaluation"], cfg["training"]
    if ev.get("require_patient_view") != "validation" or ev.get("patient_view") != "validation":
        raise PermissionError(f"{w}: config lacks the validation-only guard")
    if [float(x) for x in ev["mask_fractions"]] != FROZEN_FRACTIONS:
        raise ValueError(f"{w}: evaluation.mask_fractions must be {FROZEN_FRACTIONS}")
    if tr["mask_fractions"] != template["training"]["mask_fractions"]:
        raise ValueError(f"{w}: TRAINING mask_fractions differ from the campaign template")
    if ev.get("selection_mask_fraction") != template["evaluation"]["selection_mask_fraction"] or \
            ev["selection_mask_fraction"] != p["checkpoint_selection_mask_fraction"]:
        raise ValueError(f"{w}: checkpoint-selection fraction differs from the frozen 0.50")
    if tr["batch_size"] != p["batch_size"] or ev["batch_size"] != p["batch_size"]:
        raise ValueError(f"{w}: batch size must be {p['batch_size']}")
    for section, allowed in (("model", set()), ("training", TRAIN_ALLOWED), ("evaluation", EVAL_ALLOWED)):
        a = {k: v for k, v in cfg[section].items() if k not in allowed}
        b = {k: v for k, v in template[section].items() if k not in allowed}
        if a != b:
            raise ValueError(f"{w}: {section} differs from the campaign template outside {sorted(allowed)}")
    for section in cfg:
        if section not in template:
            raise ValueError(f"{w}: unexpected section {section}")
    if tr["seed"] != seed or tr["num_workers"] != p["num_workers"] or tr["epochs"] != p["max_epochs"] \
            or tr["early_stopping"] is not False:
        raise ValueError(f"{w}: seed/num_workers/epochs/early_stopping mismatch (exactly 120 epochs, early_stopping false)")
    if ev["mask_seed"] != p["mask_seed"] or ev["panel_size"] != p["panel_size"] or ev["panel_repeats"] != p["panel_repeats"]:
        raise ValueError(f"{w}: mask seed / panel size / panel repeats differ")
    if not ev.get("save_predictions"):
        raise ValueError(f"{w}: save_predictions must be true")
    if ev.get("output_layout") != OUTPUT_LAYOUT or ev.get("allow_overwrite_split_dir") is not False:
        raise ValueError(f"{w}: evaluation.output_layout must be {OUTPUT_LAYOUT!r} with allow_overwrite_split_dir false")
    ls = cfg["experiment"]["locus_split"]
    if ls["protocol_path"] != p["loci_split_path"] or ls["heldout_fraction"] != 0.0 \
            or cfg["dataset"]["patient_protocol"] != p["patient_split_path"]:
        raise ValueError(f"{w}: split paths differ from the shared protocol")


def audit_configs(spec: dict, root: Path) -> list[Result]:
    """Every runnable main/sensitivity arm x seed config must exist and equal the freshly built one."""
    root = Path(root)
    template = yaml.safe_load((root / spec["protocol"]["template"]).read_text())
    bad, n = [], 0
    for arm, seed in jobs(spec, include_sensitivity=True):
        n += 1
        path = root / CONFIG_DIR / config_name(arm["id"], seed)
        if not path.is_file():
            bad.append(f"{path.name} missing")
            continue
        cfg = yaml.safe_load(path.read_text())
        try:
            if cfg != build_config(spec, arm, seed, root):
                bad.append(f"{path.name} stale (differs from the generated config)")
            check_config(cfg, template, spec, arm["id"], seed)
        except (ValueError, PermissionError, KeyError) as exc:
            bad.append(f"{path.name}: {exc}")
    return [Result("CONFIG", "runs/", not bad, f"{n} per-run configs match the shared protocol" if not bad else "; ".join(bad))]


# ----------------------------------------------------------------------------- run gate
def phase_a_problems(spec: dict, results: list[Result]) -> list[str]:
    """Why Phase A (4 main arms, validation only) is NOT launchable. Empty list == ready.

    Tolerates ONLY the pending modern-sequence-FM failures (PENDING_COMPARATOR / REGISTRATION_INCOMPLETE scoped to an arm outside
    PHASE_A_ARMS). Every other audit failure, and any failure scoped to a Phase A arm, blocks.
    """
    byid = arms_by_id(spec)
    probs = [f"phase-A arm {i} missing from the matrix" for i in PHASE_A_ARMS if i not in byid]
    probs += [f"phase-A arm {i} is pending" for i in PHASE_A_ARMS if byid.get(i, {}).get("status") == "pending"]
    tolerated_scopes = {f"arm:{a['id']}" for a in spec["arms"] if a["id"] not in PHASE_A_ARMS and a.get("status") == "pending"}
    for r in failures(results):
        if r.code in TOLERABLE_CODES and r.scope in tolerated_scopes:
            continue
        probs.append(f"{r.code}@{r.scope}")
    return probs


def gate(spec: dict, results: list[Result], *, split: str, allow_incomplete_validation_only: bool = False,
         phase: str | None = None) -> tuple[bool, str]:
    """Decide whether `run` may launch. Test is never permitted unless authorized + final + all-green.

    `phase="A"` (validation only) launches the 4 fully registered main arms without waiting for the pending FM slots. The legacy
    `allow_incomplete_validation_only` escape hatch is equivalent (all available arms) and is kept for compatibility.
    """
    fails = failures(results)
    if split == "test":
        if allow_incomplete_validation_only or phase is not None:
            return False, "phase A / --allow-incomplete-validation-only never apply to the test split"
        try:
            require_test_authorization(spec)
        except PermissionError as exc:
            return False, str(exc)
        if fails:
            return False, f"audit has {len(fails)} failure(s): refusing the TEST split"
        return True, "test split authorized, freeze final, audit green"
    if split != "validation":
        return False, f"unknown split {split!r}"
    if not fails:
        return True, "audit green"
    if phase is not None:
        probs = phase_a_problems(spec, results)
        if probs:
            return False, "phase A not ready: " + "; ".join(probs)
        return True, ("PHASE A (validation only): the 4 fully registered main arms pass the audit; the pending modern sequence FM "
                      "slots still block freeze_state final, test_set_authorized true and any test evaluation")
    if allow_incomplete_validation_only:
        intolerable = [r for r in fails if r.code not in TOLERABLE_CODES]
        if intolerable:
            return False, "audit failures beyond pending modern sequence FMs: " + "; ".join(f"{r.code}@{r.scope}" for r in intolerable)
        return True, ("INCOMPLETE matrix (pending comparators only): validation-only runs of already-available arms; "
                      "this is not the final benchmark and never permits test")
    return False, (f"audit has {len(fails)} failure(s) (e.g. pending modern sequence FMs); use `run --phase A` (validation only) "
                   "for the 4 fully registered main arms")


# ----------------------------------------------------------------------------- per-run status (Phase A)
def run_dir_from_log(log: Path) -> Path | None:
    lines = [x for x in Path(log).read_text().splitlines() if x.startswith("RUN_DIR=")] if Path(log).is_file() else []
    return Path(lines[-1].split("=", 1)[1]) if lines else None


def build_status(arm_id: str, seed: int, run_dir: Path, *, wall_clock_seconds: float, spec: dict) -> dict:
    """Status of ONE finished validation run, derived from its files (history.json, best.pt, resolved_config.yaml, evaluation/).

    Raises ValueError when the frozen protocol check fails (the run must then NOT be marked done). Reads no patient/test data.
    """
    import json

    from cpg_repr_benchmark.config import config_fingerprint
    from cpg_repr_benchmark.experiments.eval_layout import eval_base
    run_dir = Path(run_dir)
    p = spec["protocol"]
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    hist = json.loads((run_dir / "history.json").read_text())
    mse = [float(r["validation_mse"]) for r in hist]
    best_epoch = min(range(len(mse)), key=mse.__getitem__)
    best = run_dir / "checkpoints" / "best.pt"
    ev = cfg["evaluation"]
    val_dir = eval_base(run_dir, OUTPUT_LAYOUT, "validation")
    check = {"epochs_run": len(hist), "epochs_expected": p["max_epochs"], "epochs_ok": len(hist) == p["max_epochs"] == cfg["training"]["epochs"],
             "early_stopping_false": cfg["training"].get("early_stopping") is False,
             "validation_only": ev.get("patient_view") == "validation" and ev.get("require_patient_view") == "validation",
             "split_dirs_layout": ev.get("output_layout") == OUTPUT_LAYOUT,
             "validation_outputs_present": all((val_dir / "seen" / f"mask_{f:.2f}" / "metrics.json").is_file()
                                               for f in spec["mask_fractions"]),
             "no_test_dir": not (run_dir / "evaluation" / "test").exists(),
             "best_pt_present": best.is_file()}
    check["ok"] = all(v for k, v in check.items() if k not in ("epochs_run", "epochs_expected"))
    if not check["ok"]:
        raise ValueError(f"{arm_id}/seed{seed}: protocol check failed: {check}")
    return {"status": "trained_validation_frozen", "arm": arm_id, "seed": seed, "split": "validation",
            "best_epoch_0based": best_epoch, "epochs_run": len(hist), "best_val_mse_at_0.50": mse[best_epoch],
            "wall_clock_seconds": round(float(wall_clock_seconds), 1), "best_pt_path": str(best), "best_pt_sha256": sha256_file(best),
            "config_hash": config_fingerprint(cfg), "run_dir": str(run_dir), "protocol_check": check, "test_read": False}


def write_status(run_dir: Path, status: dict) -> Path:
    import json
    path = Path(run_dir) / STATUS_FILE
    path.write_text(json.dumps(status, indent=2, sort_keys=True))
    return path


def phase_a_status(spec: dict, runs_root: Path) -> dict:
    """Aggregate of the Phase A runs (arm x seed, seed-major) from their per-run status files. Results live in outputs, not the spec."""
    import json
    runs_root = Path(runs_root)
    entries = []
    for arm, seed in jobs(spec, phase="A"):
        marker = runs_root / "logs" / f"{arm['id']}__seed{seed}.done"
        rd = run_dir_from_log(runs_root / "logs" / f"{arm['id']}__seed{seed}.log") if marker.exists() else None
        st = rd / STATUS_FILE if rd else None
        if st is not None and st.is_file():
            d = json.loads(st.read_text())
            entries.append({"arm": arm["id"], "seed": seed, "status": d["status"], "run_dir": d["run_dir"],
                            "best_epoch_0based": d["best_epoch_0based"], "best_val_mse_at_0.50": d["best_val_mse_at_0.50"],
                            "wall_clock_seconds": d["wall_clock_seconds"], "best_pt_sha256": d["best_pt_sha256"],
                            "status_file": str(st)})
        else:
            entries.append({"arm": arm["id"], "seed": seed, "status": "not_run" if rd is None else "incomplete"})
    n_done = sum(e["status"] == "trained_validation_frozen" for e in entries)
    return {"phase": "A", "split": "validation", "status_vocabulary": list(STATUS_VOCABULARY), "n_expected": len(entries),
            "n_trained_validation_frozen": n_done, "test_read": False, "runs": entries}


def write_phase_a_status(spec: dict, runs_root: Path) -> Path:
    import json
    path = Path(runs_root) / PHASE_A_STATUS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(phase_a_status(spec, runs_root), indent=2))
    tmp.replace(path)
    return path
