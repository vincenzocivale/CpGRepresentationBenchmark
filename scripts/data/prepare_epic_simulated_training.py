#!/usr/bin/env python3
"""Build the simulated-450K training matrix from real EPIC cohorts (ComputAgeBench).

A 450K profile is *simulated* from an EPIC profile by keeping only the probes present on both arrays
(observed set O); the EPIC-only probes (target set T) are the reconstruction targets. Only studies
whose EPIC-only probes were actually measured are kept (EPICv2 studies and studies whose released
matrix was pre-restricted to 450K probes are dropped).

The loci universe comes from data/prepared/paired_450k_epic/loci_v1.npz (built by
prepare_paired_450k_epic.py) so training and the real paired test share exactly the same O / T.

Study-disjoint train / validation / test-sim split (seed-controlled): the model is selected on
validation studies, and 'test-sim' studies quantify performance on unseen cohorts under the *simulated*
protocol. The real paired cohorts are never used for training or selection.

Outputs:
  data/prepared/epic_sim_training/epic_blood.h5      obs_beta[n,|O|], target_beta[n,|T|] (NaN = missing)
  data/prepared/epic_sim_training/samples.parquet    sample metadata + split
  data/protocols/paired_450k_epic/study_split_seed<seed>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CAB = REPO / "data/processed/ComputAgeBench/benchmark"
LOCI = REPO / "data/prepared/paired_450k_epic/loci_v1.npz"
OUT = REPO / "data/prepared/epic_sim_training"
PROTOCOL_DIR = REPO / "data/protocols/paired_450k_epic"
EPIC_PLATFORMS = ("GPL21145", "GPL23976")  # EPICv1 arrays; GPL29753 is EPICv2 (probe ids do not map)
MIN_TARGET_COVERAGE = 0.5


def assign_splits(study_sizes: pd.Series, seed: int, val_frac: float, test_frac: float) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(study_sizes.index.to_numpy()))
    total = float(study_sizes.sum())
    split, val_n, test_n = {}, 0.0, 0.0
    for study in order:
        n = float(study_sizes[study])
        if test_n < test_frac * total:
            split[study], test_n = "test_sim", test_n + n
        elif val_n < val_frac * total:
            split[study], val_n = "validation", val_n + n
        else:
            split[study] = "train"
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Seed 18 was the first seed (from 17) giving a usable size balance (>=62% train, >=15% test-sim,
    # >=10% validation of the samples) with the 411-sample GSE217633 study landing in train. Chosen on
    # split sizes only; no model was evaluated to pick it.
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--skip-copy", action="store_true", help="reuse an existing epic_blood.h5 (re-split only)")
    parser.add_argument("--val-frac", type=float, default=0.12)
    parser.add_argument("--test-frac", type=float, default=0.15)
    args = parser.parse_args()

    loci = np.load(LOCI)
    obs_loci, tgt_loci = loci["observed_cpg_idx"], loci["target_cpg_idx"]
    pheno = pd.read_parquet(CAB / "phenotypes.parquet")
    pheno["row"] = np.arange(len(pheno))

    with h5py.File(CAB / "methylation.h5", "r") as src:
        cpg_idx = src["cpg_idx"][:]
        names = src["sample_name"][:].astype(str)
        assert (names == pheno["sample_name"].to_numpy()).all(), "phenotype rows are not aligned with /beta"
        order = np.argsort(cpg_idx)
        sorted_ids = cpg_idx[order]

        def columns(ids: np.ndarray) -> np.ndarray:
            pos = np.searchsorted(sorted_ids, ids)
            if (pos >= len(sorted_ids)).any() or (sorted_ids[pos] != ids).any():
                raise RuntimeError("loci protocol contains CpGs absent from the training matrix")
            return order[pos]

        obs_cols, tgt_cols = columns(obs_loci), columns(tgt_loci)
        # Screen studies by the fraction of measured EPIC-only probes in their first sample.
        keep_studies = []
        for (platform, study), group in pheno[pheno["PlatformID"].isin(EPIC_PLATFORMS)].groupby(["PlatformID", "dataset_id"]):
            row = src["beta"][int(group["row"].iloc[0])]
            coverage = float(np.isfinite(row[tgt_cols]).mean())
            print(f"{platform} {study} n={len(group)} target_coverage={coverage:.3f}")
            if coverage >= MIN_TARGET_COVERAGE:
                keep_studies.append(study)
        samples = pheno[pheno["dataset_id"].isin(keep_studies)].sort_values("row").reset_index(drop=True)
        print(f"kept {len(keep_studies)} studies, {len(samples)} samples")

        split = assign_splits(samples.groupby("dataset_id").size(), args.seed, args.val_frac, args.test_frac)
        samples["split"] = samples["dataset_id"].map(split)

        OUT.mkdir(parents=True, exist_ok=True)
        if not args.skip_copy:
            with h5py.File(OUT / "epic_blood.h5", "w") as dst:
                d_obs = dst.create_dataset("obs_beta", (len(samples), len(obs_cols)), dtype="float32", chunks=(16, 8192))
                d_tgt = dst.create_dataset("target_beta", (len(samples), len(tgt_cols)), dtype="float32", chunks=(16, 8192))
                for i, row in enumerate(samples["row"].to_numpy()):
                    full = src["beta"][int(row)]
                    d_obs[i] = full[obs_cols]
                    d_tgt[i] = full[tgt_cols]
                    if i % 100 == 0:
                        print(f"  copied {i}/{len(samples)}", flush=True)
                dst.create_dataset("cpg_idx_obs", data=obs_loci)
                dst.create_dataset("cpg_idx_target", data=tgt_loci)
                dst.create_dataset("sample_name", data=samples["sample_name"].to_numpy().astype("S64"))
                dst.attrs["source"] = "ComputAgeBench benchmark split (EPICv1 studies)"
                dst.attrs["cpg_namespace"] = "grch38_cpg_cytosine_1based_v1"
                dst.attrs["simulation"] = "observed = EPIC probes shared with 450K; target = EPIC-only probes"
        else:
            with h5py.File(OUT / "epic_blood.h5", "r") as dst:
                assert (dst["sample_name"][:].astype(str) == samples["sample_name"].to_numpy()).all()

    keep_cols = ["sample_name", "dataset_id", "PlatformID", "Tissue", "CellType", "age", "condition", "split"]
    samples[keep_cols].to_parquet(OUT / "samples.parquet", index=False)
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    counts = samples.groupby("split").size().to_dict()
    (PROTOCOL_DIR / f"study_split_seed{args.seed}.json").write_text(
        json.dumps({"seed": args.seed, "study_to_split": split, "n_samples_per_split": counts}, indent=2, sort_keys=True)
    )
    print(counts)


if __name__ == "__main__":
    main()
