#!/usr/bin/env python3
"""Comparable unsupervised CpG maps, biological overlays and neighborhood diagnostics.

Overlay labels are not supplied to sampling, PCA or UMAP. Genomic context is
already included in the functional source representation. All panels use the same sample.
Writes PNG/PDF, coordinates, neighborhood metrics and a reproducibility manifest.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
import yaml
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from cpg_repr_benchmark.data.legacy import (
    legacy_ids_to_coordinate_ids,
    legacy_ids_to_coordinate_ids_lenient,
)
from cpg_repr_benchmark.representations.hdf5_store import inspect_representation_h5
from cpg_repr_benchmark.viz.style import (
    apply_paper_style,
    display_name,
    pretty_label,
    representation_colors,
    savefig,
)

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_COLORS = {
    "island": "#2a78d6",
    "shore": "#eda100",
    "shelf": "#1baf7a",
    "open_sea": "#eb6834",
    "unknown": "#898781",
}


def aligned_index(entry):
    path = ROOT / entry["store_h5"]
    ids, dim, _, key = inspect_representation_h5(path)
    registry = ROOT / entry["bio_validation_cpg_registry"]
    rows = np.arange(len(ids))
    if entry.get("bio_validation_cpg_registry_lenient"):
        ids, keep = legacy_ids_to_coordinate_ids_lenient(ids, registry)
        rows = rows[keep]
    else:
        ids = legacy_ids_to_coordinate_ids(ids, registry)
    order = np.argsort(ids)
    return ids[order], rows[order], path, key, dim


def read_sample(index, common):
    ids, rows, path, key, _ = index
    rows = rows[np.searchsorted(ids, common)]
    order = np.argsort(rows)
    with h5py.File(path, "r") as handle:
        # Read only the selected loci, in HDF5's required ascending row order.
        x = np.asarray(handle[key][rows[order]], dtype=np.float32)[np.argsort(order)]
    if not np.isfinite(x).all():
        raise ValueError(f"Nonfinite embeddings in {path}")
    return x


def neighbors(x, k, metric):
    # X=None explicitly excludes each query itself, including with tied distances.
    return NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=1).fit(x).kneighbors(return_distance=False)


def neighborhood_metrics(nn, context, memberships):
    results = []
    for label in CONTEXT_COLORS:
        if label == "unknown":
            continue
        selected = context == label
        n = int(selected.sum())
        observed = float(selected[nn[selected]].mean()) if n else None
        expected = (n - 1) / (len(context) - 1) if n else None
        results.append(
            {
                "annotation": label,
                "n_positive": n,
                "neighbor_fraction": observed,
                "random_fraction": expected,
                "enrichment": observed / expected if expected and expected > 0 else None,
            }
        )
    for label, selected in memberships.items():
        n = int(selected.sum())
        observed = float(selected[nn[selected]].mean()) if n else None
        expected = (n - 1) / (len(context) - 1) if n else None
        results.append(
            {
                "annotation": label,
                "n_positive": n,
                "neighbor_fraction": observed,
                "random_fraction": expected,
                "enrichment": observed / expected if expected and expected > 0 else None,
            }
        )
    return results


def plot_maps(maps, names, context, memberships, seed, args):
    nrows = 1 + len(memberships)
    fig, axes = plt.subplots(nrows, len(names), figsize=(3.7 * len(names), 3.25 * nrows), squeeze=False)
    order = np.random.default_rng(args.sample_seed).permutation(len(context))
    colors = np.array([CONTEXT_COLORS[c] for c in context])
    for col, name in enumerate(names):
        xy = maps[name]
        ax = axes[0, col]
        ax.scatter(*xy[order].T, c=colors[order], s=3, alpha=0.65, linewidths=0, rasterized=True)
        ax.set_title(display_name(name), fontsize=12, pad=12)
        for row, (label, member) in enumerate(memberships.items(), start=1):
            ax = axes[row, col]
            ax.scatter(*xy[~member].T, c="#d8d7d0", s=3, alpha=0.35, linewidths=0, rasterized=True)
            ax.scatter(*xy[member].T, c="#4a3aa7", s=12, alpha=0.85, linewidths=0, rasterized=True, zorder=3)
            ax.text(
                0.02,
                0.02,
                f"{member.sum():,} annotated CpGs",
                transform=ax.transAxes,
                fontsize=8,
                color="#555555",
            )
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_box_aspect(1)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
    axes[0, 0].set_ylabel("Genomic context\n(included in functional input)", fontsize=10, labelpad=15)
    for row, label in enumerate(memberships, start=1):
        axes[row, 0].set_ylabel(pretty_label(label).capitalize(), fontsize=11, labelpad=15)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, label=label.replace("_", " "))
        for label, color in CONTEXT_COLORS.items()
        if np.any(context == label)
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.97), ncol=len(handles), frameon=False
    )
    fig.suptitle("Biological organization of CpG representations", fontsize=15, y=1.005)
    preprocessing = f"PCA {args.pca_components}" if args.pca_components else "native dimensions"
    fig.text(
        0.5,
        0.005,
        f"Same {len(context):,} uniformly sampled CpGs · {preprocessing} · {args.metric} · "
        f"neighbors={args.n_neighbors} · min_dist={args.min_dist} · seed={seed}\n"
        "UMAP fitted without labels. Context is part of the functional source features; "
        "EWAS overlays share the map above.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(top=0.91, bottom=0.065, wspace=0.12, hspace=0.10)
    savefig(fig, args.out_dir / f"biology_seed_{seed}")
    plt.close(fig)


def plot_diagnostics(table, args):
    if not args.known_sets:
        return
    fig, axes = plt.subplots(1, len(args.known_sets), figsize=(5 * len(args.known_sets), 3.8), squeeze=False)
    colors = representation_colors(args.representations)
    spaces = ["native"] + (["pca"] if args.pca_components else []) + ["umap"]
    for ax, label in zip(axes[0], args.known_sets):
        for i, name in enumerate(args.representations):
            rows = table[(table.representation == name) & (table.annotation == label)]
            for j, space in enumerate(spaces):
                values = rows.loc[rows.space == space, "enrichment"].dropna().to_numpy()
                if not len(values):
                    continue
                x = j + (i - (len(args.representations) - 1) / 2) * 0.15
                mean = values.mean()
                ax.errorbar(
                    x,
                    mean,
                    yerr=[[mean - values.min()], [values.max() - mean]],
                    fmt="o",
                    color=colors[name],
                    capsize=4,
                    markersize=6,
                    label=display_name(name) if j == 0 else None,
                )
        ax.axhline(1, color="#898781", linestyle="--", linewidth=1)
        ax.set_xticks(
            range(len(spaces)),
            [
                "Native embedding"
                if s == "native"
                else f"PCA {args.pca_components}"
                if s == "pca"
                else "UMAP (2D)"
                for s in spaces
            ],
        )
        ax.set_title(pretty_label(label).capitalize(), fontsize=11)
        ax.set_ylabel(f"Same-set enrichment among {args.k} neighbors (fold)")
        ax.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(args.representations), frameon=False)
    fig.text(
        0.5,
        0.015,
        "Dashed line: random mixing. UMAP points: mean and full seed range. "
        "Descriptive; no adjustment for genomic context or proximity.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    savefig(fig, args.out_dir / "ewas_neighborhood_enrichment")
    plt.close(fig)


def render_saved(out_dir):
    manifest = json.loads((out_dir / "manifest.json").read_text())
    args = argparse.Namespace(**manifest["arguments"])
    args.out_dir = out_dir
    sample = pd.read_csv(out_dir / "sample.csv")
    memberships = {name: sample[name].to_numpy(dtype=bool) for name in args.known_sets}
    apply_paper_style()
    for seed in args.seeds:
        maps = {}
        for name in args.representations:
            frame = pd.read_csv(out_dir / f"{name}_seed_{seed}.csv")
            if not np.array_equal(frame.cpg_idx, sample.cpg_idx):
                raise ValueError("Saved coordinate and sample IDs differ")
            maps[name] = frame[["umap_1", "umap_2"]].to_numpy()
        plot_maps(maps, args.representations, sample.context.to_numpy(), memberships, seed, args)
    plot_diagnostics(pd.read_csv(out_dir / "neighborhood_metrics.csv"), args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, default=ROOT / "configs/representations/future_models.yaml")
    p.add_argument(
        "--representations",
        nargs="+",
        default=["functional_annotations_pca", "deepcpg_dna_locus", "cpgpt_locus_large"],
    )
    p.add_argument(
        "--known-sets", nargs="*", default=["ewas_atlas_colorectal_cancer", "ewas_atlas_prostate_cancer"]
    )
    p.add_argument("--annotations-dir", type=Path, default=ROOT / "data/bio_annotations")
    p.add_argument("--max-points", type=int, default=12000)
    p.add_argument("--sample-seed", type=int, default=17)
    p.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 97])
    p.add_argument("--pca-components", type=int, default=50, help="same for every arm; 0 disables PCA")
    p.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--k", type=int, default=30, help="neighbors for label enrichment")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/figures/umap_biology")
    p.add_argument("--render-only", action="store_true", help="redraw saved results without refitting")
    args = p.parse_args()
    if args.render_only:
        render_saved(args.out_dir)
        return
    if args.max_points < 4 or args.k < 1 or args.n_neighbors < 2 or args.pca_components < 0:
        p.error("Invalid sample size, neighbor count or PCA dimension")
    catalog = {v["name"]: v for v in yaml.safe_load(args.catalog.read_text())["representations"].values()}
    indices = {name: aligned_index(catalog[name]) for name in args.representations}
    common = next(iter(indices.values()))[0]
    for index in indices.values():
        common = np.intersect1d(common, index[0], assume_unique=True)
    n_common = len(common)
    if n_common > args.max_points:
        common = np.sort(
            np.random.default_rng(args.sample_seed).choice(common, args.max_points, replace=False)
        )
    if len(common) <= max(args.k, args.n_neighbors, args.pca_components):
        p.error("Too few shared CpGs for requested PCA/neighbor counts")
    if args.pca_components and any(i[4] < args.pca_components for i in indices.values()):
        p.error("PCA dimension must not exceed any representation's native dimension")
    context_table = pd.read_parquet(args.annotations_dir / "genomic_context.parquet")
    context_series = context_table.set_index("cpg_idx")["context"]
    if not context_series.index.is_unique:
        raise ValueError("Duplicate context annotation coordinates")
    context = context_series.reindex(common).fillna("unknown").to_numpy(dtype=str)
    if not set(context) <= set(CONTEXT_COLORS):
        raise ValueError("Unexpected genomic context label")
    memberships = {
        name: np.isin(common, np.load(args.annotations_dir / "known_sets" / f"{name}.npy"))
        for name in args.known_sets
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample = pd.DataFrame({"cpg_idx": common, "context": context, **memberships})
    sample.to_csv(args.out_dir / "sample.csv", index=False)
    manifest = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "shared_universe_size": n_common,
        "sample_size": len(common),
        "sampling": "uniform without replacement, independent of annotations",
        "context_caveat": "Genomic context is included in functional source features; "
        "its visualization is descriptive, not independent validation.",
        "preprocessing": "centered PCA without whitening; no per-feature standardization",
        "versions": {
            package: version(package) for package in ["umap-learn", "numpy", "scikit-learn", "matplotlib"]
        },
        "representations": {},
        "interpretation": "Enrichment is descriptive, not a held-out prediction score or significance test. "
        "EWAS enrichment may reflect genomic context and correlated nearby loci.",
    }
    all_maps = {seed: {} for seed in args.seeds}
    metrics = []
    apply_paper_style()
    for name, index in indices.items():
        print(f"{name}: loading {len(common):,}/{n_common:,} shared CpGs", flush=True)
        x = read_sample(index, common)
        rep_manifest = {
            "catalog_entry": catalog[name],
            "native_dim": x.shape[1],
            "aligned_coverage": len(index[0]),
            "store_size": index[2].stat().st_size,
            "store_mtime_ns": index[2].stat().st_mtime_ns,
        }
        spaces = {"native": x}
        if args.pca_components:
            pca = PCA(
                n_components=args.pca_components, svd_solver="randomized", random_state=args.sample_seed
            )
            projected = pca.fit_transform(x)
            rep_manifest["pca_variance_retained"] = float(pca.explained_variance_ratio_.sum())
            spaces["pca"] = projected
        else:
            projected = x
        manifest["representations"][name] = rep_manifest
        for space, features in spaces.items():
            for row in neighborhood_metrics(neighbors(features, args.k, args.metric), context, memberships):
                metrics.append({"representation": name, "space": space, "seed": None, **row})
        for seed in args.seeds:
            print(f"{name}: UMAP seed {seed}", flush=True)
            xy = umap.UMAP(
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist,
                metric=args.metric,
                random_state=seed,
                n_jobs=1,
            ).fit_transform(projected)
            all_maps[seed][name] = xy
            pd.DataFrame({"cpg_idx": common, "umap_1": xy[:, 0], "umap_2": xy[:, 1]}).to_csv(
                args.out_dir / f"{name}_seed_{seed}.csv", index=False
            )
            for row in neighborhood_metrics(neighbors(xy, args.k, "euclidean"), context, memberships):
                metrics.append({"representation": name, "space": "umap", "seed": seed, **row})
        pd.DataFrame(metrics).to_csv(args.out_dir / "neighborhood_metrics.csv", index=False)
        (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for seed, maps in all_maps.items():
        plot_maps(maps, args.representations, context, memberships, seed, args)
    plot_diagnostics(pd.DataFrame(metrics), args)
    print(f"Saved maps and diagnostics to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
