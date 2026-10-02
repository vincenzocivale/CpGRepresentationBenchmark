#!/usr/bin/env python3
"""Prepare the real paired 450K<->EPIC test cohorts and the 450K->EPIC locus universe.

Paired cohorts (GEO series matrices, same biological sample profiled on both arrays):
  * GSE86833  - LNCaP/PrEC/fibroblasts/Guthrie-card blood, 15 pairs (Moran/Logue/Clark, EPIC evaluation)
  * GSE92580  - pediatric brain tumours (fresh-frozen and FFPE), pairs matched on identical titles

Universe (probe-level, EPICv1 = GPL21145, 450K = GPL13534):
  * observed set  O = probes present on BOTH arrays (what a 450K profile offers to an EPIC-trained model)
  * target set    T = EPIC probes absent from the 450K array ("EPIC-only" probes to reconstruct)

Only autosomal loci (chr1-22) that are reference-genome CpGs are kept. Probes are mapped to canonical GRCh38 CpG loci with the same crosswalk used for ComputAgeBench, so
the loci line up with the EPIC training matrix. Loci hit by several probes are averaged (nanmean).

Outputs under data/prepared/paired_450k_epic/:
  loci_v1.parquet        probe_id, cpg_idx, chr, pos, role in {observed, target}
  loci_v1.npz            observed_cpg_idx, target_cpg_idx (sorted, unique loci, the frozen protocol)
  <GSE>.h5               obs_beta[n,|O|] (450K array), target_beta[n,|T|] (EPIC array),
                         epic_obs_beta[n,|O|] (EPIC array on O), cpg_idx_obs, cpg_idx_target, pair ids
  <GSE>_pairs.parquet    one row per pair (sample_id, title, gsm_450k, gsm_epic, group)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/raw/paired_450k_epic"
OUT = REPO / "data/prepared/paired_450k_epic"
FUNCTIONAL_STORE = REPO / "data/external/functional/locus_features_v1"
MAPPING = REPO / "data/processed/ComputAgeBench/benchmark/cpg_mapping.parquet"
COHORTS = ("GSE86833", "GSE92580")
PLATFORMS = {"450k": "GPL13534", "epic": "GPL21145"}


def read_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (beta[probe x GSM], meta[GSM -> title])."""
    titles, gsms = [], []
    import gzip

    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("!Sample_title"):
                titles = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_geo_accession"):
                gsms = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!series_matrix_table_begin"):
                break
    beta = pd.read_csv(path, sep="\t", comment="!", index_col=0, compression="gzip").astype("float32")
    beta.index = beta.index.astype(str)
    meta = pd.DataFrame({"gsm": gsms, "title": [" ".join(t.split()) for t in titles]})
    return beta, meta


def in_reference_cpg_store(ids: np.ndarray) -> np.ndarray:
    """Mask of loci that are reference-genome CpGs known to the functional-annotation store.

    Probes whose mapped coordinate is not a reference CpG cytosine (e.g. SNP-altered sites) have no
    locus annotation. They are excluded ONCE here, for every representation arm, never per arm.
    """
    ok = np.zeros(len(ids), dtype=bool)
    chrom, pos = ids // 1_000_000_000, ids % 1_000_000_000
    for code in np.unique(chrom):
        keys = np.load(FUNCTIONAL_STORE / "features" / f"chr{code}" / "locus_key.npy", mmap_mode="r")
        sel = np.flatnonzero(chrom == code)
        query = ((int(code) << 32) | pos[sel]).astype(np.uint64)
        ix = np.searchsorted(keys, query)
        hit = ix < len(keys)
        hit[hit] &= keys[ix[hit]] == query[hit]
        ok[sel] = hit
    return ok


def collapse_to_loci(beta: pd.DataFrame, probe_to_locus: pd.Series) -> pd.DataFrame:
    """Average probes that share a canonical locus; result is [sample x locus] sorted by cpg_idx."""
    keep = beta.index.intersection(probe_to_locus.index)
    sub = beta.loc[keep]
    sub.index = probe_to_locus.loc[keep].to_numpy()
    return sub.groupby(level=0).mean().sort_index().T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_parquet(MAPPING)
    mapping = mapping[mapping["cpg_idx"].notna()]
    probe_to_locus = mapping.drop_duplicates("probe_id").set_index("probe_id")["cpg_idx"].astype("int64")
    locus_coord = mapping.drop_duplicates("cpg_idx").set_index("cpg_idx")[["chr", "pos"]]

    raw = {}
    for cohort in COHORTS:
        for key, gpl in PLATFORMS.items():
            raw[(cohort, key)] = read_series_matrix(RAW / cohort / f"{cohort}-{gpl}_series_matrix.txt.gz")

    # Platform probe manifests come from the matrices themselves (row ids), cg probes only.
    probes_450k = {p for p in raw[("GSE86833", "450k")][0].index if p.startswith("cg")}
    probes_epic = {p for p in raw[("GSE86833", "epic")][0].index if p.startswith("cg")}
    observed_probes = sorted(probes_450k & probes_epic)
    target_probes = sorted(probes_epic - probes_450k)
    print(f"450K probes={len(probes_450k)} EPIC probes={len(probes_epic)} shared={len(observed_probes)} "
          f"epic_only={len(target_probes)} 450k_only={len(probes_450k - probes_epic)}")

    def loci_of(probes: list[str]) -> np.ndarray:
        found = [p for p in probes if p in probe_to_locus.index]
        return np.unique(probe_to_locus.loc[found].to_numpy())

    # Autosomes only: the benchmark's functional-annotation store (and every existing arm) covers chr1-22.
    def autosomal(ids: np.ndarray) -> np.ndarray:
        ids = ids[(ids // 1_000_000_000 >= 1) & (ids // 1_000_000_000 <= 22)]
        keep = in_reference_cpg_store(ids)
        print(f"  dropped {int((~keep).sum())} loci that are not reference CpGs")
        return ids[keep]

    obs_loci, tgt_loci = autosomal(loci_of(observed_probes)), autosomal(loci_of(target_probes))
    overlap = np.intersect1d(obs_loci, tgt_loci)
    # A locus reachable by a shared probe is observed; drop it from the target set (probe collisions).
    tgt_loci = np.setdiff1d(tgt_loci, overlap)
    print(f"observed loci={len(obs_loci)} target loci={len(tgt_loci)} (dropped {len(overlap)} colliding loci)")

    table = pd.concat(
        [
            pd.DataFrame({"probe_id": observed_probes, "role": "observed"}),
            pd.DataFrame({"probe_id": target_probes, "role": "target"}),
        ]
    )
    table["cpg_idx"] = table["probe_id"].map(probe_to_locus)
    table = table.dropna(subset=["cpg_idx"]).astype({"cpg_idx": "int64"})
    table = table[table.cpg_idx.isin(obs_loci) | table.cpg_idx.isin(tgt_loci)]
    table = table[(table.role == "observed") | ~table.cpg_idx.isin(overlap)]
    table = table.join(locus_coord, on="cpg_idx")
    table.to_parquet(args.out / "loci_v1.parquet", index=False)
    np.savez_compressed(args.out / "loci_v1.npz", observed_cpg_idx=obs_loci, target_cpg_idx=tgt_loci)

    qc = {"n_450k_probes": len(probes_450k), "n_epic_probes": len(probes_epic),
          "n_observed_loci": int(len(obs_loci)), "n_target_loci": int(len(tgt_loci)), "cohorts": {}}
    for cohort in COHORTS:
        b450, m450 = raw[(cohort, "450k")]
        bepic, mepic = raw[(cohort, "epic")]
        pairs = m450.merge(mepic, on="title", suffixes=("_450k", "_epic"))
        pairs = pairs.rename(columns={"gsm_450k": "gsm_450k", "gsm_epic": "gsm_epic"})
        pairs["sample_id"] = [f"{cohort}:{t}" for t in pairs["title"]]
        pairs["group"] = [
            ("FFPE_REPLI-g" if "REPLI" in t else "FFPE_Infinium" if "FFPE" in t else "FF") if cohort == "GSE92580"
            else ("blood" if "Guthrie" in t else "cell_or_fibroblast")
            for t in pairs["title"]
        ]
        pairs["cohort"] = cohort
        obs450 = collapse_to_loci(b450[pairs["gsm_450k"]], probe_to_locus).reindex(columns=obs_loci)
        # rows of collapse_to_loci are GSMs; reorder to pair order
        obs450 = obs450.loc[pairs["gsm_450k"]]
        epic_all = collapse_to_loci(bepic[pairs["gsm_epic"]], probe_to_locus).loc[pairs["gsm_epic"]]
        epic_obs = epic_all.reindex(columns=obs_loci)
        epic_tgt = epic_all.reindex(columns=tgt_loci)
        pairs.to_parquet(args.out / f"{cohort}_pairs.parquet", index=False)
        with h5py.File(args.out / f"{cohort}.h5", "w") as h:
            h.create_dataset("obs_beta", data=obs450.to_numpy(np.float32), compression="gzip", compression_opts=1)
            h.create_dataset("target_beta", data=epic_tgt.to_numpy(np.float32), compression="gzip", compression_opts=1)
            h.create_dataset("epic_obs_beta", data=epic_obs.to_numpy(np.float32), compression="gzip", compression_opts=1)
            h.create_dataset("cpg_idx_obs", data=obs_loci)
            h.create_dataset("cpg_idx_target", data=tgt_loci)
            h.create_dataset("sample_id", data=np.asarray(pairs["sample_id"], dtype="S64"))
            h.attrs["cohort"] = cohort
            h.attrs["obs_platform"] = "Illumina HumanMethylation450 (GPL13534)"
            h.attrs["target_platform"] = "Illumina MethylationEPIC v1 (GPL21145)"
            h.attrs["cpg_namespace"] = "grch38_cpg_cytosine_1based_v1"
        qc["cohorts"][cohort] = {
            "n_pairs": int(len(pairs)),
            "groups": pairs["group"].value_counts().to_dict(),
            "obs_nan_fraction": float(np.isnan(obs450.to_numpy()).mean()),
            "target_nan_fraction": float(np.isnan(epic_tgt.to_numpy()).mean()),
        }
        print(cohort, qc["cohorts"][cohort])
    (args.out / "qc.json").write_text(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
