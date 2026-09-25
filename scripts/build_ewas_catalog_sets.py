#!/usr/bin/env python3
"""Build known-CpG-set files from the EWAS Catalog bulk download for bio_validation.

Source: https://www.ewascatalog.org/static/docs/ewascatalog-results.txt.gz +
ewascatalog-studies.txt.gz (bulk association + study metadata tables).

For each named trait group, studies are matched by exact `trait` label, then associations
from those studies are filtered at the standard EWAS genome-wide-significance threshold
(p < 1e-7) and deduplicated by cpg_idx. This mirrors the existing sets documented in
data/bio_annotations/README.md (ewas_catalog_age/_rheumatoid_arthritis/_schizophrenia).

Probe IDs are mapped to GRCh38 coordinates via
data/processed/ComputAgeBench/illumina_probe_grch38.parquet (same source already used for
horvath_clock and the existing EWAS sets), then to this repo's coordinate-native cpg_idx via
cpg_repr_benchmark.data.coordinates.encode_many.

None of this trait/association data is derived from ENCODE or from any representation's
input features (see docs/EMBEDDING_EVALUATION.md discussion), so it is a fair bio-validation
axis for every representation arm, including functional_annotations_pca.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpg_repr_benchmark.data.coordinates import encode_many  # noqa: E402

# set_name -> exact `trait` labels to match (studies table), deliberately excluding
# maternal/prenatal/pack-years/treatment variants to keep each set a clean adult-exposure
# or disease-status signal comparable to the existing ewas_catalog_age/_ra/_schizophrenia sets.
# Cancer is split per tumor type (rather than one aggregated ewas_catalog_cancer set) so
# bio-validation can show which tumor types a representation's embedding separates well,
# at the cost of smaller/noisier per-set sample counts. "Cancer treatment: ..." traits
# (chemo/RT exposure signatures) and "... progression" traits are deliberately excluded —
# they describe treatment or trajectory, not disease status.
TRAIT_GROUPS: dict[str, list[str]] = {
    "ewas_catalog_smoking": ["smoking", "Smoking", "Tobacco smoking"],
    "ewas_catalog_bmi": ["BMI", "body mass index", "Body mass index"],
    "ewas_catalog_breast_cancer": [
        "Breast cancer",
        "breast cancer",
        "Incident Breast Cancer",
        "Prevalent Breast Cancer (Self-report)",
    ],
    "ewas_catalog_lung_cancer": [
        "Lung cancer",
        "lung cancer",
        "Incident Lung Cancer",
        "Prevalent Lung Cancer (Self-report)",
    ],
}
# Deliberately excluded (built once, then dropped): colorectal/prostate/ovarian/pancreatic
# cancer trait groups yielded 23/8/14/0 CpGs respectively after p<1e-7 filtering + dedup — too
# few for a balanced CV probe (bio_validation_report requires >=10 members and >=10 non-members).
# Not a pipeline bug; see the bio_validation run that surfaced this.

P_THRESHOLD = 1e-7


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True, help="ewascatalog-results.txt.gz")
    p.add_argument("--studies", type=Path, required=True, help="ewascatalog-studies.txt.gz")
    p.add_argument("--probe-grch38", type=Path, required=True,
                    help="data/processed/ComputAgeBench/illumina_probe_grch38.parquet")
    p.add_argument("--output-dir", type=Path, required=True, help="e.g. data/bio_annotations/known_sets")
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="build only these set names (default: all of TRAIT_GROUPS), e.g. "
        "--only ewas_catalog_breast_cancer ewas_catalog_lung_cancer",
    )
    args = p.parse_args()

    studies = pd.read_csv(args.studies, sep="\t", dtype=str)
    probe_coords = pd.read_parquet(args.probe_grch38).set_index("probe_id")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    trait_groups = TRAIT_GROUPS
    if args.only is not None:
        unknown = set(args.only) - set(TRAIT_GROUPS)
        if unknown:
            raise ValueError(f"--only names not in TRAIT_GROUPS: {sorted(unknown)}")
        trait_groups = {name: TRAIT_GROUPS[name] for name in args.only}

    for set_name, traits in trait_groups.items():
        matched_study_ids = set(studies.loc[studies["trait"].isin(traits), "study_id"])
        if not matched_study_ids:
            raise ValueError(f"{set_name}: no studies matched traits={traits}")

        cpg_ids: set[str] = set()
        for chunk in pd.read_csv(
            args.results, sep="\t", dtype=str, chunksize=500_000,
            usecols=["cpg", "p", "study_id"],
        ):
            chunk = chunk[chunk["study_id"].isin(matched_study_ids)]
            if chunk.empty:
                continue
            p_values = pd.to_numeric(chunk["p"], errors="coerce")
            chunk = chunk[p_values < P_THRESHOLD]
            cpg_ids.update(chunk["cpg"])

        coords = probe_coords.reindex(list(cpg_ids)).dropna()
        canonical = np.unique(encode_many(coords["chr"], coords["pos"].astype(int)))

        out_path = args.output_dir / f"{set_name}.npy"
        np.save(out_path, canonical)
        print(
            f"{set_name}: {len(matched_study_ids)} studies, "
            f"{len(cpg_ids)} unique probes p<{P_THRESHOLD}, "
            f"{len(canonical)} mapped to GRCh38 -> {out_path}"
        )


if __name__ == "__main__":
    main()
