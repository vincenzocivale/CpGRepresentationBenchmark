"""Shared matplotlib style for benchmark result figures.

Minimal, paper-style presentation: small sans-serif type, thin spines, light
y-only gridlines, no legend frame. Colors and display names are assigned from
fixed, hand-maintained tables (see REPRESENTATION_ORDER / DISPLAY_NAMES below)
so a given representation always gets the same color and label across every
figure, regardless of which ones happen to have data in a given run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# Validated categorical palette (light mode), fixed slot order.
# Source: dataviz skill references/palette.md - CVD-safe adjacent-pair
# ordering. Never cycle or reorder this list; a 9th series should fold into
# "Other" rather than reuse a slot or introduce an unvalidated hue.
CATEGORICAL_PALETTE = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Sequential single-hue ramp (blue, light->dark), for magnitude encodings
# (heatmaps) rather than categorical identity. Source: dataviz skill
# references/palette.md, steps 100-700.
SEQUENTIAL_BLUE = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]

MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Canonical representation order (mirrors configs/representations/future_models.yaml).
# Slot i in CATEGORICAL_PALETTE is reserved for REPRESENTATION_ORDER[i], so
# colors stay stable across figures/runs even as representations come and go.
REPRESENTATION_ORDER = [
    "functional_annotations_pca",
    "ntv3_pre",
    "cpgpt_locus",
    "cpgpt_locus_large",
    "methylgpt_locus",
    "deepcpg_dna_locus",
    "deepcpg_dna_locus_hepg2",
]

# Raw representation id -> paper-facing display name.
DISPLAY_NAMES = {
    "functional_annotations_pca": "Functional PCA",
    "functional_annotations_chr1": "Functional PCA",
    "functional_annotations_online_chr1": "Functional (online)",
    "ntv3_pre": "NT v3 (pretrained)",
    "ntv3_pre_chr1": "NT v3 (pretrained)",
    "ntv3_pre_chr1to5": "NT v3 (pretrained)",
    "cpgpt_locus": "CpGPT (small)",
    "cpgpt_locus_large": "CpGPT (large)",
    "methylgpt_locus": "MethylGPT",
    "deepcpg_dna_locus": "DeepCpG DNA (HCC)",
    "deepcpg_dna_locus_hepg2": "DeepCpG DNA (HepG2)",
}


def display_name(representation: str) -> str:
    return DISPLAY_NAMES.get(representation, representation)


# Metric-axis labels (genomic context classes, clock names) -> paper-facing text.
LABEL_OVERRIDES = {
    "open_sea": "open sea",
    "horvath_clock": "Horvath",
    "hannum_clock": "Hannum",
    "phenoage_clock": "PhenoAge",
    "phastcons100way": "phastCons100way",
}


# known_cpg_sets are a flat bag of 16 sets spanning three different kinds of
# signal; grouped so each can get its own compact figure instead of one
# crowded panel.
KNOWN_SET_CATEGORIES = {
    "cancer": [
        "ewas_atlas_colorectal_cancer",
        "ewas_atlas_ovarian_cancer",
        "ewas_atlas_prostate_cancer",
        "ewas_catalog_breast_cancer",
        "ewas_catalog_lung_cancer",
    ],
    "disease_trait": [
        "ewas_catalog_coronary_heart_disease",
        "ewas_catalog_rheumatoid_arthritis",
        "ewas_catalog_schizophrenia",
        "ewas_catalog_sle",
        "ewas_catalog_type2_diabetes",
        "ewas_catalog_smoking",
        "ewas_catalog_bmi",
        "ewas_catalog_age",
    ],
    "clock": ["hannum_clock", "horvath_clock", "phenoage_clock"],
}

KNOWN_SET_CATEGORY_LABELS = {
    "cancer": "Cancer EWAS sets",
    "disease_trait": "Disease / trait EWAS sets",
    "clock": "Epigenetic clock CpG sets",
}


def pretty_label(name: str) -> str:
    if name in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[name]
    for prefix in ("ewas_atlas_", "ewas_catalog_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if name.endswith("_clock"):
        return f"{pretty_label(name[: -len('_clock')])} clock"
    return name.replace("_", " ").replace("type2", "type 2")


def _canonical_key(representation: str) -> str:
    """Map a dataset-suffixed id (e.g. ntv3_pre_chr1) back to its REPRESENTATION_ORDER slot."""
    if representation in REPRESENTATION_ORDER:
        return representation
    for candidate in REPRESENTATION_ORDER:
        if representation.startswith(candidate):
            return candidate
    return representation


def representation_colors(representations: list[str]) -> dict[str, str]:
    """Assign each representation its fixed categorical slot color.

    Unknown representations (not in REPRESENTATION_ORDER) fall back to muted
    gray rather than borrowing an unvalidated hue.
    """
    colors = {}
    for rep in representations:
        key = _canonical_key(rep)
        if key in REPRESENTATION_ORDER:
            idx = REPRESENTATION_ORDER.index(key)
            colors[rep] = CATEGORICAL_PALETTE[idx % len(CATEGORICAL_PALETTE)]
        else:
            colors[rep] = MUTED_INK
    return colors


def apply_paper_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "text.color": "#0b0b0b",
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": "#0b0b0b",
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.color": MUTED_INK,
            "ytick.color": MUTED_INK,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRIDLINE,
            "grid.alpha": 1.0,
            "grid.linewidth": 0.6,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "figure.facecolor": "#fcfcfb",
            "savefig.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
        }
    )


def sequential_colormap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("paper_sequential_blue", SEQUENTIAL_BLUE)


def savefig(fig: plt.Figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
