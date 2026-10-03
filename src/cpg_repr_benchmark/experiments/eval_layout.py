"""Evaluation output layouts (torch-free, data-free).

``evaluation.output_layout`` (default ``legacy``) selects where `scripts/run_masking_benchmark.py` writes evaluation results:

* ``legacy``      ``<run>/evaluation/<view>/mask_<f>/{metrics.json,predictions.npz}`` and ``<run>/summary.json``. Unchanged; every
                  historical ENCODE / screen / confirm analysis reads this.
* ``split_dirs``  ``<run>/evaluation/<split>/<view>/mask_<f>/{metrics.json,predictions.npz}`` and
                  ``<run>/evaluation/<split>/summary.json`` where ``<split>`` is ``evaluation.patient_view`` (``validation`` | ``test``).
                  Nothing is written to ``<run>/summary.json`` and nothing outside ``evaluation/<split>/``, so a test evaluation can
                  never touch validation outputs.

Guards (split_dirs): the output directory is resolved by `split_output_dir` and must lie inside ``evaluation/<split>/``; a split
directory that already holds files is never reused unless ``evaluation.allow_overwrite_split_dir`` is true; the ``test`` split
additionally requires `authorize_test_split` (matrix ``test_set_authorized: true`` + ``freeze_state: final`` + all-green audit).
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

LEGACY = "legacy"
SPLIT_DIRS = "split_dirs"
LAYOUTS = (LEGACY, SPLIT_DIRS)
SPLITS = ("validation", "test")


def get_layout(cfg: dict[str, Any]) -> str:
    layout = cfg.get("evaluation", {}).get("output_layout", LEGACY)
    if layout not in LAYOUTS:
        raise ValueError(f"evaluation.output_layout must be one of {LAYOUTS}, got {layout!r}")
    return layout


def get_split(cfg: dict[str, Any]) -> str:
    split = cfg["evaluation"].get("patient_view", "test")
    if split not in SPLITS:
        raise ValueError(f"evaluation.patient_view must be one of {SPLITS}, got {split!r}")
    return split


def eval_base(run_dir: Path, layout: str, split: str) -> Path:
    """Directory that holds the ``<view>/mask_<f>`` folders (and, for split_dirs, ``summary.json``)."""
    run_dir = Path(run_dir)
    if layout == LEGACY:
        return run_dir / "evaluation"
    if layout == SPLIT_DIRS:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        return run_dir / "evaluation" / split
    raise ValueError(f"unknown layout {layout!r}")


def metrics_dir(run_dir: Path, cfg: dict[str, Any], split: str, view: str, key: str) -> Path:
    """Reader helper: where a run configured with `cfg` stored the `<view>/<key>` results of `split`."""
    return eval_base(run_dir, get_layout(cfg), split) / view / key


def summary_path(run_dir: Path, layout: str, split: str) -> Path:
    return Path(run_dir) / "summary.json" if layout == LEGACY else eval_base(run_dir, layout, split) / "summary.json"


def split_output_dir(run_dir: Path, split: str, view: str, key: str) -> Path:
    """Resolve ``evaluation/<split>/<view>/<key>`` and prove it stays inside ``evaluation/<split>/``."""
    base = eval_base(run_dir, SPLIT_DIRS, split).resolve()
    out = (base / view / key).resolve()
    if base != out and base not in out.parents:
        raise PermissionError(f"GUARD: {out} escapes evaluation/{split}/")
    other = [s for s in SPLITS if s != split]
    for o in other:
        ob = eval_base(run_dir, SPLIT_DIRS, o).resolve()
        if ob == out or ob in out.parents:
            raise PermissionError(f"GUARD: refusing to write the {split!r} split into the {o!r} split directory")
    return out


def _nonempty(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def authorize_test_split(matrix_path: Path | None = None, repo_root: Path | None = None) -> None:
    """Raise PermissionError unless the confirmation matrix authorizes the TEST split (authorized + final + green audit)."""
    from cpg_repr_benchmark.experiments import confirmation_matrix as cm
    from cpg_repr_benchmark.experiments.guards import require_test_authorization
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    spec = cm.load_matrix(matrix_path or root / cm.MATRIX_PATH)
    require_test_authorization(spec)
    fails = cm.failures(cm.audit(spec, root))
    if fails:
        raise PermissionError(f"GUARD: confirmation audit has {len(fails)} failure(s); the TEST split stays locked")


def prepare_split_dir(run_dir: Path, cfg: dict[str, Any], split: str, *, authorize: Callable[[], None] | None = None) -> Path | None:
    """Pre-flight for split_dirs (call BEFORE any evaluation). Returns the split dir, or None for the legacy layout."""
    layout = get_layout(cfg)
    if layout == LEGACY:
        return None
    if split == "test":
        (authorize or authorize_test_split)()
    if cfg["evaluation"].get("patient_view") != split:
        raise PermissionError(f"GUARD: split {split!r} != evaluation.patient_view {cfg['evaluation'].get('patient_view')!r}")
    base = eval_base(run_dir, layout, split)
    if _nonempty(base) and not cfg["evaluation"].get("allow_overwrite_split_dir", False):
        raise FileExistsError(f"{base} already holds results; set evaluation.allow_overwrite_split_dir: true to overwrite the "
                              f"same split (the other split is never touched)")
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_view_outputs(run_dir: Path, cfg: dict[str, Any], split: str, view: str, key: str, payload: dict[str, Any],
                       outputs: dict[str, np.ndarray] | None) -> Path:
    layout = get_layout(cfg)
    if layout == LEGACY:
        out_dir = Path(run_dir) / "evaluation" / view / key
    else:
        if payload.get("patient_view") != split:
            raise PermissionError(f"GUARD: payload patient_view {payload.get('patient_view')!r} != split {split!r}")
        out_dir = split_output_dir(run_dir, split, view, key)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True))
    if outputs is not None:
        np.savez_compressed(out_dir / "predictions.npz", **outputs)
    return out_dir


def write_split_summary(run_dir: Path, cfg: dict[str, Any], split: str, summary: dict[str, Any]) -> Path:
    from cpg_repr_benchmark.experiments.run_store import write_summary
    layout = get_layout(cfg)
    if layout == LEGACY:
        write_summary(run_dir, summary)
        return Path(run_dir) / "summary.json"
    path = summary_path(run_dir, layout, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True))
    return path


def evaluate_split(run_dir: Path, cfg: dict[str, Any], views: dict[str, str], fractions: list[float],
                   evaluate_fn: Callable[[str, float], tuple[dict[str, Any], dict[str, np.ndarray]]], *,
                   authorize: Callable[[], None] | None = None) -> dict[str, dict]:
    """Run every (view, fraction) evaluation via `evaluate_fn(view, fraction) -> (metrics, outputs)` and write the results with the
    configured layout. `views` maps view name -> prior_policy text. The single code path used by run_masking_benchmark.py."""
    split = get_split(cfg)
    prepare_split_dir(run_dir, cfg, split, authorize=authorize)
    save = bool(cfg["evaluation"].get("save_predictions", False))
    results: dict[str, dict] = {}
    for view, prior_policy in views.items():
        results[view] = {}
        for fraction in fractions:
            metrics, outputs = evaluate_fn(view, fraction)
            key = f"mask_{fraction:.2f}"
            payload = {**metrics, "mask_fraction": fraction, "view": view, "prior_policy": prior_policy, "patient_view": split,
                       "panel_repeats": int(cfg["evaluation"].get("panel_repeats", 1)),
                       "mask_seed": int(cfg["evaluation"].get("mask_seed", cfg["training"]["seed"]))}
            write_view_outputs(run_dir, cfg, split, view, key, payload, outputs if save else None)
            results[view][key] = payload
    return results
