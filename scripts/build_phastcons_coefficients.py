#!/usr/bin/env python3
"""Build a per-CpG phastCons conservation score for bio_validation's regression probe.

Source: UCSC hg38.phastCons100way.bw (100-way vertebrate multiple alignment conservation
score, 0-1 per base, higher = more conserved). This is a purely comparative-genomics/sequence
signal -- not derived from ENCODE, not seen as input by any representation in this benchmark
(functional_annotations_pca's feature contract was checked directly; see
docs/EMBEDDING_EVALUATION.md) -- so it is a fair axis for every arm, including
functional_annotations_pca.

Output: a `<set_name>_coefficients.parquet` (cpg_idx, coefficient) consumed by
cpg_repr_benchmark.bio_validation.annotations.load_cpg_annotations's `--coefficients-dir`,
evaluated as a regression target via RidgeCV (same mechanism as horvath_clock_coefficients).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpg_repr_benchmark.data.coordinates import encode_many  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bigwig", type=Path, required=True, help="local hg38.phastCons100way.bw")
    p.add_argument("--registry", type=Path, required=True,
                    help="CpG registry parquet with [cpg_idx, chr, pos] (1-based cytosine)")
    p.add_argument("--output", type=Path, required=True,
                    help="e.g. data/bio_annotations/phastcons100way_coefficients.parquet")
    args = p.parse_args()

    registry = pd.read_parquet(args.registry, columns=["chr", "pos"])
    canonical_idx = encode_many(registry["chr"], registry["pos"])

    bw = pyBigWig.open(str(args.bigwig))
    chrom_lengths = bw.chroms()

    scores = np.full(len(registry), np.nan, dtype=np.float64)
    for chrom, group in registry.groupby("chr", sort=False):
        if chrom not in chrom_lengths:
            continue
        positions = group["pos"].to_numpy(dtype=np.int64)
        # 1-based cytosine position -> 0-based bigWig query [pos-1, pos).
        zero_based = positions - 1
        valid = (zero_based >= 0) & (zero_based < chrom_lengths[chrom])
        rows = group.index.to_numpy()[valid]
        values = [
            bw.values(chrom, int(pos), int(pos) + 1)[0]
            for pos in zero_based[valid]
        ]
        scores[rows] = values
    bw.close()

    valid_mask = ~np.isnan(scores)
    out = pd.DataFrame({"cpg_idx": canonical_idx[valid_mask], "coefficient": scores[valid_mask]})
    out = out.drop_duplicates(subset="cpg_idx")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"phastcons100way: {len(out)}/{len(registry)} loci scored -> {args.output}")
    print(out["coefficient"].describe())


if __name__ == "__main__":
    main()
