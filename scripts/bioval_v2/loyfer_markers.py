# ruff: noqa: F841  (one-shot data-prep script; style-only, behaviour unchanged)
"""Markers from the ORIGINAL published source (Loyfer et al. 2023): nloyfer/UXM_deconv supplemental files (hg38).
 - Atlas.U25.l4.hg38.full.tsv : published atlas, top-25 hypomethylated ('U') blocks per cell type (wgbstools find_markers, --min_cpg 4 ('l4')).
 - Markers.U250.hg38.tsv      : top-250 candidate blocks per cell type incl. find_markers stats (tg_mean, bg_mean, delta_means, delta_quants, ...).
Blocks are startCpG..endCpG-1 (end exclusive) in the global wgbstools hg38 CpG index; expanded to CpGs via our fasta-derived index
(verified: block start == pos(startCpG), end == pos(endCpG-1)+1 for 100% of blocks). Marker labels are NOT derived from our data/embeddings.
Cell-type names are the atlas column names (published)."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from loyfer_common import *


def main():
    cid, pos, chroms = load_genome_index(); ch = np.array(chroms)
    u = load_universe()
    stats = pd.read_csv(META / "Markers.U250.hg38.tsv", sep="\t"); stats.columns = [c.lstrip("#") for c in stats.columns]
    stats["name"] = stats.region
    atlas = pd.read_csv(META / "Atlas.U25.l4.hg38.full.tsv", sep="\t").iloc[:, :8]
    atlas = atlas[atlas.target != "target"]
    blocks = []
    a = atlas.copy(); a["source"] = "loyfer2023_atlas_U25_l4"
    s = stats[["name", "tg_mean", "bg_mean", "delta_means", "delta_quants", "delta_maxmin"]]
    a = a.merge(s, on="name", how="left")
    b = stats[["chr", "start", "end", "startCpG", "endCpG", "target", "name", "direction", "tg_mean", "bg_mean", "delta_means", "delta_quants", "delta_maxmin"]].copy()
    b["source"] = "loyfer2023_find_markers_U250"
    b["in_published_atlas_U25"] = b.name.isin(a.name)
    a["in_published_atlas_U25"] = True
    allb = pd.concat([a[b.columns], b[~b.name.isin(a.name)]], ignore_index=True)
    allb[["startCpG", "endCpG"]] = allb[["startCpG", "endCpG"]].astype(np.int64)
    # expand
    rows = []
    n = (allb.endCpG - allb.startCpG).to_numpy()
    idx = np.concatenate([np.arange(s, e) for s, e in zip(allb.startCpG, allb.endCpG)])  # 1-based genome idx
    rep = np.repeat(np.arange(len(allb)), n)
    m = pd.DataFrame({"block_row": rep, "genome_cpg_index": idx, "chrom": ch[cid[idx - 1]], "pos": pos[idx - 1].astype(np.int64)})
    m = m.join(allb[["name", "target", "direction", "tg_mean", "bg_mean", "delta_means", "delta_quants", "delta_maxmin", "source", "in_published_atlas_U25"]], on="block_row")
    m["direction"] = m.direction.map({"U": "hypo", "M": "hyper"})
    m = m.rename(columns={"target": "cell_type", "name": "block", "delta_means": "effect_size_delta_means"})
    m = m.drop(columns="block_row").merge(u, on=["chrom", "pos"], how="left")      # cpg_idx = benchmark id, NaN if outside universe
    m["in_universe"] = m.cpg_idx.notna()
    m["source_note"] = "published blocks (UXM_deconv supplemental, Loyfer 2023); CpG expansion ours"
    m.to_parquet(OUT / "loyfer_markers_all_cpgs.parquet", index=False)
    mu = m[m.in_universe].copy(); mu["cpg_idx"] = mu.cpg_idx.astype(np.int64)
    mu.to_parquet(OUT / "loyfer_markers_universe.parquet", index=False)
    t = (m.groupby(["source", "cell_type"]).agg(blocks=("block", "nunique"), cpgs_total=("pos", "size"), cpgs_in_universe=("in_universe", "sum")).reset_index())
    tu = mu.groupby("cell_type").block.nunique().rename("blocks_with_>=1_universe_cpg")
    t.to_csv(OUT / "loyfer_markers_per_celltype.tsv", sep="\t", index=False)
    print(t.to_string()); print(len(m), len(mu), mu.cpg_idx.nunique())
    print("atlas25 only:", m[m.in_published_atlas_U25].in_universe.sum(), m[m.in_published_atlas_U25].shape[0])
if __name__ == "__main__":
    main()
