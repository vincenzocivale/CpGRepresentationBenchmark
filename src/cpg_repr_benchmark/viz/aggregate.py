"""Flatten run summary.json files under outputs/ into tidy DataFrames for plotting."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_masking_df(outputs_dir: Path) -> pd.DataFrame:
    """Same flattening as scripts/summarize_runs.py::flatten_summary, as a DataFrame."""
    paths = sorted({
        *outputs_dir.glob("masking/*/*/*/seed_*/*/summary.json"),
        *outputs_dir.glob("masking/*/*/seed_*/*/summary.json"),
    })
    rows = []
    for path in paths:
        summary = json.loads(path.read_text())
        path_track = path.parent.parent.parent.name
        if path_track == summary.get("representation"):
            path_track = None
        for view, masks in summary.get("evaluation", {}).items():
            for mask_name, metrics in masks.items():
                rows.append(
                    {
                        "run_dir": str(path.parent),
                        "dataset": summary.get("dataset"),
                        "representation": summary.get("representation"),
                        "representation_dim": summary.get("representation_dim"),
                        "representation_track": summary.get("representation_track") or path_track,
                        "seed": summary.get("seed"),
                        "view": view,
                        "mask": mask_name,
                        "mask_fraction": metrics.get("mask_fraction"),
                        "mse": metrics.get("mse"),
                        "mae": metrics.get("mae"),
                        "skill_vs_prior": metrics.get("skill_vs_prior"),
                        "patient_pearson": metrics.get("patient_pearson"),
                        "n_pairs": metrics.get("n_pairs"),
                    }
                )
    return pd.DataFrame(rows)


def load_bio_validation_df(outputs_dir: Path) -> pd.DataFrame:
    """Long-format table: one row per (representation, group, name, metric)."""
    paths = sorted(outputs_dir.glob("bio_validation/*/*/seed_*/summary.json"))
    rows = []
    for path in paths:
        summary = json.loads(path.read_text())
        representation = summary.get("representation")
        track = summary.get("track")
        seed = summary.get("seed") or path.parent.name.replace("seed_", "")
        metrics = summary.get("metrics", {})
        for group in ("genomic_context", "known_cpg_sets", "clock_coefficients"):
            for name, values in metrics.get(group, {}).items():
                if not isinstance(values, dict):
                    continue
                for metric, value in values.items():
                    rows.append(
                        {
                            "representation": representation,
                            "track": track,
                            "seed": seed,
                            "group": group,
                            "name": name,
                            "metric": metric,
                            "value": value,
                        }
                    )
    return pd.DataFrame(rows)
