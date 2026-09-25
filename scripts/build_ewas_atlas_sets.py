#!/usr/bin/env python3
"""Build known-CpG-set files from the EWAS Atlas bulk download for bio_validation.

Source: https://ngdc.cncb.ac.cn/ewas/downloads/batch?file=EWAS_Atlas_associations.tsv
(single bulk associations table, ~106MB, ~805k rows; columns confirmed from the actual
downloaded header: `Association_ID`, `probe_ID`, `trait`, `case_description`, `case_beta`,
`control_description`, `control_beta`, `correlation`, `p_value`, `rank_in_study`,
`effect_size`, `study_ID`, `PMID`).

This is a separate curated knowledgebase from the EWAS Catalog
(scripts/build_ewas_catalog_sets.py) with much better coverage for some cancer traits that
were too small in the EWAS Catalog after its uniform p<1e-7 filter: colorectal (23 CpGs),
prostate (8), ovarian (14), pancreatic (0) cancer all fell below the bio_validation 10-CpG
minimum there. EWAS Atlas curates only associations each source study itself reported as
significant, per its own study-specific threshold -- there is no single uniform p-value
cutoff to apply on top, and empirically every row's `p_value` in this file is already well
below 0.001 (checked across the four cancer traits below: max p_value == 9.85e-4). So this
script does NOT apply a p-value filter; it takes every row EWAS Atlas already curated for the
matched trait. Pancreatic cancer remains too few (6 unique probes, 2 studies) even here and is
deliberately NOT built -- kept in TRAIT_GROUPS commented out for provenance, matching the
build_ewas_catalog_sets.py convention of leaving excluded groups documented in the source.

For each named trait group, rows are matched by exact `trait` label, then probe IDs are
deduplicated directly (no per-study or p-value filtering, per above), mapped to GRCh38 via
data/processed/ComputAgeBench/illumina_probe_grch38.parquet (same source used by
build_ewas_catalog_sets.py, horvath_clock, etc.), then to this repo's coordinate-native
cpg_idx via cpg_repr_benchmark.data.coordinates.encode_many.

None of this trait/association data is derived from ENCODE or from any representation's
input features, so it is a fair bio-validation axis for every representation arm, including
functional_annotations_pca.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpg_repr_benchmark.data.coordinates import encode_many  # noqa: E402

# set_name -> exact `trait` label to match (associations table). Deliberately excludes
# variant traits (e.g. "colorectal cancer classification", "colorectal cancer prognosis",
# "familial prostate cancer risk") to keep each set a clean disease-status signal, matching
# the ewas_catalog_breast_cancer / ewas_catalog_lung_cancer convention.
TRAIT_GROUPS: dict[str, list[str]] = {
    "ewas_atlas_colorectal_cancer": ["colorectal cancer"],
    "ewas_atlas_prostate_cancer": ["prostate cancer"],
    "ewas_atlas_ovarian_cancer": ["ovarian cancer"],
}
# Deliberately excluded: pancreatic cancer, checked in EWAS Atlas too -- only 6 unique probes
# across 2 studies for exact trait "pancreatic cancer", still below the 10-CpG-per-class
# minimum bio_validation_report requires. Not built from either EWAS Catalog (0 CpGs at
# p<1e-7) or EWAS Atlas (6 CpGs, no p-value filter even applied).


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--associations", type=Path, required=True,
                    help="EWAS_Atlas_associations.tsv (bulk download, kept outside the repo)")
    p.add_argument("--probe-grch38", type=Path, required=True,
                    help="data/processed/ComputAgeBench/illumina_probe_grch38.parquet")
    p.add_argument("--output-dir", type=Path, required=True, help="e.g. data/bio_annotations/known_sets")
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="build only these set names (default: all of TRAIT_GROUPS), e.g. "
        "--only ewas_atlas_colorectal_cancer",
    )
    args = p.parse_args()

    probe_coords = pd.read_parquet(args.probe_grch38).set_index("probe_id")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    trait_groups = TRAIT_GROUPS
    if args.only is not None:
        unknown = set(args.only) - set(TRAIT_GROUPS)
        if unknown:
            raise ValueError(f"--only names not in TRAIT_GROUPS: {sorted(unknown)}")
        trait_groups = {name: TRAIT_GROUPS[name] for name in args.only}

    all_traits = {t for traits in trait_groups.values() for t in traits}
    probe_ids_by_trait: dict[str, set[str]] = {t: set() for t in all_traits}

    # Single pass over the bulk file (single-pass, no parallel workers, per repo I/O
    # constraints); the file has non-UTF8 bytes in some free-text description columns, so
    # decode with latin-1 (we only ever read probe_ID/trait, both plain ASCII).
    for chunk in pd.read_csv(
        args.associations, sep="\t", dtype=str, chunksize=500_000,
        usecols=["probe_ID", "trait"], encoding="latin-1",
    ):
        for trait in all_traits:
            matched = chunk.loc[chunk["trait"] == trait, "probe_ID"]
            if not matched.empty:
                probe_ids_by_trait[trait].update(matched)

    for set_name, traits in trait_groups.items():
        probe_ids: set[str] = set()
        for trait in traits:
            probe_ids.update(probe_ids_by_trait[trait])
        if not probe_ids:
            raise ValueError(f"{set_name}: no rows matched traits={traits}")

        coords = probe_coords.reindex(list(probe_ids)).dropna()
        dropped = len(probe_ids) - len(coords)
        canonical = np.unique(encode_many(coords["chr"], coords["pos"].astype(int)))

        out_path = args.output_dir / f"{set_name}.npy"
        np.save(out_path, canonical)
        print(
            f"{set_name}: traits={traits}, {len(probe_ids)} unique curated probes, "
            f"{dropped} dropped (not in probe_grch38 lookup), "
            f"{len(canonical)} mapped to GRCh38 -> {out_path}"
        )


if __name__ == "__main__":
    main()
