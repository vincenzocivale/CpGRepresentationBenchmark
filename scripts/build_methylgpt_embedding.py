#!/usr/bin/env python3
"""Build the canonical MethylGPT locus representation store for this benchmark.

MethylGPT's CpG token table is a static, per-probe embedding learned during
pretraining: no sample methylation values are consumed and no forward pass
occurs (see scripts/methylgpt/README.md). It is therefore patient-agnostic and
label-free, matching this benchmark's native_frozen / locus_only contract
(see docs/ADDING_REPRESENTATIONS.md).

Source contract: this repo's CpG registry parquet, e.g.
data/cpg/registries/array_cpg_map.parquet, columns [cpg_idx, chr, pos] with
1-based cytosine positions (see docs/COORDINATE_NATIVE_DATA.md).

MethylGPT's checkpoint is keyed by Illumina probe ID, not genomic coordinate,
so this script needs a probe_id -> (chr, pos) crosswalk to join the registry
to MethylGPT's vocabulary. data/processed/ComputAgeBench/illumina_probe_grch38.parquet
(columns [probe_id, chr, pos]) already provides this for the array registry.
A registry CpG whose (chr, pos) has no probe_id in the crosswalk, or whose
probe_id is outside MethylGPT's trained vocabulary, is dropped -- this script
reports coverage and refuses to silently narrow the universe: pass
--allow-partial-coverage to proceed anyway, otherwise incomplete coverage is
a hard error (the resolver/registry step enforces "never build a per-
representation intersection" -- see CLAUDE.md).

Output contract (canonical representation store):
  /cpg_idx    int64 [N]
  /embedding  float16/float32 [N, D]

This script must run in a plain Python environment with only numpy/torch/h5py
/pandas installed (see scripts/methylgpt/README.md) -- MethylGPT's checkpoint
is read as a raw state dict, so no MethylGPT package import is required and
this never needs the cpg-repr-benchmark conda env's dependencies either way.

Example:
  python scripts/build_methylgpt_embedding.py \\
      --registry data/cpg/registries/array_cpg_map.parquet \\
      --probe-crosswalk data/processed/ComputAgeBench/illumina_probe_grch38.parquet \\
      --checkpoint-path dependencies/methylgpt/tiny-best_model_epoch10.pt \\
      --probe-ids-csv dependencies/methylgpt/probe_ids_type3.csv \\
      --output data/cache/representations/methylgpt_locus.h5
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "methylgpt"))
from extract_cpg_embeddings import extract_cpg_embeddings  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", type=Path, required=True, help="CpG registry parquet with [cpg_idx, chr, pos]")
    p.add_argument("--probe-crosswalk", type=Path, required=True,
                    help="Parquet with [probe_id, chr, pos] mapping Illumina probe IDs to GRCh38 coordinates, "
                         "e.g. data/processed/ComputAgeBench/illumina_probe_grch38.parquet")
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--probe-ids-csv", type=Path, required=True,
                    help="Ordered illumina_probe_id vocabulary CSV matching --checkpoint-path")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--autodownload", action="store_true",
                    help="Fetch the published tiny checkpoint/probe CSV to the given paths if missing")
    p.add_argument("--allow-partial-coverage", action="store_true",
                    help="Proceed even if some registry loci have no probe_id in the crosswalk or vocabulary "
                         "(default: hard error, per CLAUDE.md's 'never build a per-representation intersection')")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N registry loci (smoke test)")
    args = p.parse_args()

    registry = pd.read_parquet(args.registry, columns=["cpg_idx", "chr", "pos"])
    if args.limit is not None:
        registry = registry.iloc[: args.limit]
    if registry["cpg_idx"].duplicated().any():
        raise ValueError(f"{args.registry} contains duplicate cpg_idx values")

    crosswalk = pd.read_parquet(args.probe_crosswalk, columns=["probe_id", "chr", "pos"])
    n_coord_dupes = int(crosswalk[["chr", "pos"]].duplicated().sum())
    if n_coord_dupes:
        # Multiple Illumina probe IDs can share one CpG coordinate (e.g. EPIC replicate
        # probes / overlapping array designs); MethylGPT's vocabulary is still one row per
        # probe_id, so pick the lexicographically first probe_id per coordinate deterministically.
        print(f"NOTE: {n_coord_dupes} duplicate (chr, pos) rows in {args.probe_crosswalk}; "
              "keeping the lexicographically first probe_id per coordinate")
        crosswalk = crosswalk.sort_values("probe_id").drop_duplicates(["chr", "pos"], keep="first")

    joined = registry.merge(crosswalk, on=["chr", "pos"], how="left")
    missing_probe = joined["probe_id"].isna()

    vocab_ids = set(
        pd.read_csv(args.probe_ids_csv, usecols=["illumina_probe_id"])["illumina_probe_id"]
    )
    out_of_vocab = joined["probe_id"].notna() & ~joined["probe_id"].isin(vocab_ids)

    dropped = missing_probe | out_of_vocab
    n_missing = int(missing_probe.sum())
    n_out_of_vocab = int(out_of_vocab.sum())
    if dropped.any():
        message = (
            f"{n_missing}/{len(joined)} registry loci have no probe_id in {args.probe_crosswalk}; "
            f"{n_out_of_vocab}/{len(joined)} have a probe_id outside {args.probe_ids_csv}'s "
            "trained vocabulary (MethylGPT covers only its pretraining array's probe set, a small "
            "fraction of this registry's genome-wide/multi-array CpG universe)"
        )
        if not args.allow_partial_coverage:
            raise ValueError(message + "; pass --allow-partial-coverage to proceed with reduced coverage")
        print(f"WARNING: {message}; dropping these loci (--allow-partial-coverage set)")
    joined = joined.loc[~dropped].reset_index(drop=True)

    cpg_idx = joined["cpg_idx"].to_numpy(dtype=np.int64)
    probe_ids = joined["probe_id"].tolist()

    embeddings = extract_cpg_embeddings(
        probe_ids,
        checkpoint_path=args.checkpoint_path,
        probe_ids_csv=args.probe_ids_csv,
        autodownload=args.autodownload,
    ).astype(np.float16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as out:
        out.create_dataset("cpg_idx", data=cpg_idx, dtype="int64")
        out.create_dataset("embedding", data=embeddings, dtype="float16")
        out.attrs["representation"] = "methylgpt_locus"
        out.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        out.attrs["source"] = str(args.registry)
        out.attrs["probe_crosswalk"] = str(args.probe_crosswalk)
        out.attrs["checkpoint_path"] = str(args.checkpoint_path)
        out.attrs["probe_ids_csv"] = str(args.probe_ids_csv)
        out.attrs["patient_specific"] = False
        out.attrs["supervision"] = "none"
        out.attrs["locus_fit_scope"] = "external_pretrained"
        out.attrs["coordinate_convention"] = "1-based position of CpG cytosine"
        out.attrs["reference_build"] = "GRCh38"

    sidecar = {
        "n_loci": int(len(cpg_idx)),
        "n_loci_dropped_no_probe_id": n_missing,
        "n_loci_dropped_out_of_vocab": n_out_of_vocab,
        "embedding_dim": int(embeddings.shape[1]),
        "source": str(args.registry),
        "probe_crosswalk": str(args.probe_crosswalk),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(sidecar, indent=2))
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
