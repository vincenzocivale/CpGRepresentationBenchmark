#!/usr/bin/env python3
"""UMAP comparison of raw locus embeddings across representations, on the shared set of
CpGs they all cover.

Reuses the same legacy-axis -> GRCh38-coordinate alignment `scripts/run_bio_validation.py`
uses to score every representation against the same annotations, so the CpGs compared here
are genuinely the same loci across stores (not just the same row count).

Two coloring modes, each answering a different question about what the raw embedding
already encodes without any task-specific training:
  --color-by context     genomic context (island/shore/shelf/open_sea)
  --color-by known-set    membership in a bio_validation known CpG set (e.g. a cancer
                          EWAS set) vs. everything else

ntv3_pre only covers chr1, so any panel set that includes it - and the resulting figure -
is chr1-only; dropping ntv3_pre in favor of e.g. deepcpg_dna_locus lifts that restriction.

Each panel's embedding is standardized then PCA-reduced to <=50 components before UMAP -
without this, high-variance raw dimensions (especially PCA-derived ones, like our own
functional_annotations_pca) dominate the distance metric and wash out cluster structure a
downstream classifier can otherwise find easily (see the bio_probes_auc scores).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
import yaml
from sklearn.decomposition import PCA

from cpg_repr_benchmark.data.legacy import legacy_ids_to_coordinate_ids, legacy_ids_to_coordinate_ids_lenient
from cpg_repr_benchmark.representations.hdf5_store import validate_canonical_h5
from cpg_repr_benchmark.viz.style import apply_paper_style, display_name, savefig

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPRESENTATIONS = ["functional_annotations_pca", "deepcpg_dna_locus", "cpgpt_locus_large"]

CONTEXT_COLORS = {
    "island": "#2a78d6",
    "open_sea": "#eb6834",
    "shelf": "#1baf7a",
    "shore": "#eda100",
}
CONTEXT_ORDER = ["island", "shore", "shelf", "open_sea"]

IN_SET_COLOR = "#e34948"
OUT_SET_COLOR = "#d8d7d0"


def _load_catalog(catalog_path: Path) -> dict:
    entries = yaml.safe_load(catalog_path.read_text())["representations"]
    return {entry["name"]: entry for entry in entries.values()}


def _load_aligned(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    store_path = ROOT / entry["store_h5"]
    registry_path = ROOT / entry["bio_validation_cpg_registry"]
    cpg_idx, _dim = validate_canonical_h5(store_path)
    with h5py.File(store_path, "r") as handle:
        embedding = np.asarray(handle["embedding"][:], dtype=np.float32)
    if entry.get("bio_validation_cpg_registry_lenient"):
        coord_ids, keep = legacy_ids_to_coordinate_ids_lenient(cpg_idx, registry_path)
        embedding = embedding[keep]
    else:
        coord_ids = legacy_ids_to_coordinate_ids(cpg_idx, registry_path)
    return coord_ids, embedding


def _colors_context(common_ids: np.ndarray, genomic_context_parquet: Path) -> tuple[np.ndarray, list]:
    context = pd.read_parquet(genomic_context_parquet).set_index("cpg_idx")["context"]
    colors = context.reindex(common_ids).fillna("open_sea").map(CONTEXT_COLORS).to_numpy()
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=CONTEXT_COLORS[c], label=c.replace("_", " "))
        for c in CONTEXT_ORDER
    ]
    return colors, handles


def _colors_known_set(common_ids: np.ndarray, known_set_path: Path) -> tuple[np.ndarray, list]:
    in_set = set(np.load(known_set_path).tolist())
    is_member = np.isin(common_ids, list(in_set))
    colors = np.where(is_member, IN_SET_COLOR, OUT_SET_COLOR)
    label = known_set_path.stem.replace("_", " ")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=IN_SET_COLOR, label=label),
        plt.Line2D([0], [0], marker="o", linestyle="", color=OUT_SET_COLOR, label="other"),
    ]
    return colors, handles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, default=ROOT / "configs/representations/future_models.yaml")
    parser.add_argument(
        "--representations",
        nargs="+",
        default=DEFAULT_REPRESENTATIONS,
        help="representation catalog names to compare, one UMAP panel each",
    )
    parser.add_argument("--color-by", choices=["context", "known-set"], default="context")
    parser.add_argument(
        "--genomic-context-parquet", type=Path, default=ROOT / "data/bio_annotations/genomic_context.parquet"
    )
    parser.add_argument(
        "--known-set-path",
        type=Path,
        default=ROOT / "data/bio_annotations/known_sets/ewas_atlas_colorectal_cancer.npy",
        help="only used when --color-by known-set",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/figures")
    parser.add_argument("--out-name", type=str, default=None)
    parser.add_argument("--max-points", type=int, default=15000, help="subsample cap per panel")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    catalog = _load_catalog(args.catalog)
    aligned = {name: _load_aligned(catalog[name]) for name in args.representations}

    common_ids = None
    for coord_ids, _emb in aligned.values():
        ids = set(coord_ids.tolist())
        common_ids = ids if common_ids is None else (common_ids & ids)
    common_ids = np.array(sorted(common_ids), dtype=np.int64)
    print(f"shared CpGs across {args.representations}: {len(common_ids)}")

    rng = np.random.default_rng(args.seed)
    force_include = np.array([], dtype=np.int64)
    if args.color_by == "known-set":
        in_set = set(np.load(args.known_set_path).tolist())
        force_include = np.array(sorted(in_set & set(common_ids.tolist())), dtype=np.int64)
        # cap the highlighted class so it can't outnumber (or swamp the visual proportion
        # of) the background sample for a set that isn't actually rare
        max_highlight = max(int(0.15 * args.max_points), 1)
        if len(force_include) > max_highlight:
            force_include = rng.choice(force_include, size=max_highlight, replace=False)
            force_include.sort()
    if len(common_ids) > args.max_points:
        remaining = np.setdiff1d(common_ids, force_include, assume_unique=False)
        n_fill = max(args.max_points - len(force_include), 0)
        sampled = rng.choice(remaining, size=min(n_fill, len(remaining)), replace=False)
        common_ids = np.union1d(force_include, sampled)

    if args.color_by == "context":
        colors, handles = _colors_context(common_ids, args.genomic_context_parquet)
    else:
        colors, handles = _colors_known_set(common_ids, args.known_set_path)

    apply_paper_style()
    fig, axes = plt.subplots(1, len(args.representations), figsize=(3.4 * len(args.representations), 3.2))
    if len(args.representations) == 1:
        axes = [axes]

    for ax, name in zip(axes, args.representations):
        coord_ids, embedding = aligned[name]
        order = np.argsort(coord_ids)
        positions = order[np.searchsorted(coord_ids, common_ids, sorter=order)]
        sub_embedding = embedding[positions]
        # functional_annotations_pca is already a denoised PCA space (its dims are ordered
        # by explained variance at the source); reducing it further just discards the finer
        # structure that separates its own multi-cluster shape. Only FM embeddings - which
        # are raw model activations, not PCA-ordered - benefit from PCA-denoising before UMAP.
        if catalog[name].get("family") != "functional_annotations":
            n_components = min(50, sub_embedding.shape[0], sub_embedding.shape[1])
            sub_embedding = PCA(n_components=n_components, random_state=args.seed).fit_transform(sub_embedding)

        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=args.seed)
        coords_2d = reducer.fit_transform(sub_embedding)

        if args.color_by == "known-set":
            # in-set points drawn last and larger, so a sparse highlighted class isn't buried
            plot_order = np.argsort(colors == OUT_SET_COLOR)
            sizes = np.where(colors[plot_order] == OUT_SET_COLOR, 3, 9)
        else:
            plot_order = np.arange(len(colors))
            sizes = 3
        ax.scatter(
            coords_2d[plot_order, 0],
            coords_2d[plot_order, 1],
            c=colors[plot_order],
            s=sizes,
            alpha=0.6,
            linewidths=0,
        )
        ax.set_title(display_name(name), loc="left", fontweight="normal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)
        ax.grid(False)

    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or f"umap_comparison_{args.color_by.replace('-', '_')}"
    savefig(fig, args.out_dir / out_name)
    plt.close(fig)
    print(f"Wrote {args.out_dir / out_name}.png")


if __name__ == "__main__":
    main()
