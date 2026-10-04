#!/usr/bin/env python3
"""Build the canonical external GSE40279 dataset for the external reconstruction confirmation (phase 1, data only).

Reads the already prepared GSE40279 files (read-only) and writes NEW artifacts only:
  <out>/methylation.h5            /beta [656, n_universe_covered] float32 (untouched values), /cpg_idx (benchmark global
                                  cpg_idx, universe order), /sample_name, plus chrom/pos/source_cpg_idx for traceability
  <out>/phenotypes.parquet        per-row subject/sex/age/site/plate metadata
  <out>/cpg_mapping_external.parquet  one row per benchmark-universe locus (mapped / unmapped + reason)
  <out>/loci_seen_external_gse40279_v1.npz  locus protocol (loci_seen.npz format; all universe loci -> train, no hold-out)
  <manifest>                      mapping manifest JSON (tracked)
No model, no GPU, no representation is touched.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from cpg_repr_benchmark.external.io import sha256_file, write_json_new
from cpg_repr_benchmark.external.mapping import exact_coordinate_join

REPO = Path(__file__).resolve().parents[2]
LOCUS_SEED = 17001


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=Path, default=REPO / "data/processed/GSE40279")
    ap.add_argument("--registry", type=Path, default=REPO / "data/cpg/registries/array_cpg_map.parquet")
    ap.add_argument("--tcga-matrix", type=Path, default=REPO / "data/methylation/tcga_array_official_full.h5",
                    help="only /cpg_idx is read (axis check); no beta is read")
    ap.add_argument("--loci-seen", type=Path, default=REPO / "outputs/encode_atlas_v1/loci_seen.npz",
                    help="frozen TCGA locus protocol; only train_cpg_idx is read (universe check)")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data/derived/external_gse40279_v1")
    ap.add_argument("--manifest", type=Path, default=REPO / "configs/external/gse40279_v1_mapping_manifest.json")
    ap.add_argument("--row-block", type=int, default=32)
    args = ap.parse_args()

    out = args.out_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; refusing to overwrite")
    if args.manifest.exists():
        raise SystemExit(f"{args.manifest} exists; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    src_h5 = args.src_dir / "methylation.h5"
    assertions: dict[str, object] = {}

    # ---- benchmark universe ------------------------------------------------------------------
    reg = pd.read_parquet(args.registry).sort_values("raw_cpg_row").reset_index(drop=True)
    assert reg["cpg_idx"].is_unique and (reg["raw_cpg_row"].to_numpy() == np.arange(len(reg))).all()
    with h5py.File(args.tcga_matrix, "r") as g:
        tcga_axis = np.asarray(g["cpg_idx"][:], dtype=np.int64)
    loci_seen_ids = np.load(args.loci_seen)["train_cpg_idx"]
    assertions["registry_order_equals_tcga_matrix_axis"] = bool(np.array_equal(tcga_axis, reg["cpg_idx"].to_numpy()))
    assertions["registry_equals_loci_seen_universe"] = bool(
        np.array_equal(np.sort(tcga_axis), np.sort(loci_seen_ids)))
    assert all(assertions.values()), assertions
    n_universe = len(reg)

    # ---- cohort axes -------------------------------------------------------------------------
    with h5py.File(src_h5, "r") as f:
        chrom = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["chrom"][:]])
        pos = np.asarray(f["pos"][:], dtype=np.int64)
        src_cpg_idx = np.asarray(f["cpg_idx"][:], dtype=np.int64)
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in f["sample_name"][:]]
        n_samples, n_src_loci = f["beta"].shape
    assert len(set(names)) == len(names) == 656, "sample names must be 656 unique"
    urows, ccols, unmapped = exact_coordinate_join(reg["chr"], reg["pos"], chrom, pos)
    assertions["join_exact_no_offset"] = True
    assertions["n_mapped_plus_unmapped_equals_universe"] = bool(len(urows) + len(unmapped) == n_universe)
    assertions["cohort_coordinates_unique"] = True
    # id consistency: coordinate-native id = chrom_number * 1e9 + pos for mapped loci
    chrnum = np.array([int(c[3:]) for c in chrom[ccols]])
    assertions["source_cpg_idx_equals_chrom_1e9_plus_pos"] = bool(
        np.array_equal(src_cpg_idx[ccols], chrnum * 1_000_000_000 + pos[ccols]))
    assertions["mapped_cpg_idx_unique"] = bool(len(np.unique(reg["cpg_idx"].to_numpy()[urows])) == len(urows))
    assert all(assertions.values()), assertions
    new_ids = reg["cpg_idx"].to_numpy()[urows].astype(np.int64)

    # ---- phenotypes --------------------------------------------------------------------------
    ph = pd.read_parquet(args.src_dir / "phenotypes.parquet").sort_values("matrix_column").reset_index(drop=True)
    assertions["phenotype_rows_follow_matrix_axis"] = bool(
        (ph["matrix_column"].to_numpy() == np.arange(len(ph))).all() and ph["sample_name"].tolist() == names)
    assertions["subject_ids_unique"] = bool(ph["subject_id"].astype(str).is_unique and len(ph) == 656)
    assertions["sex_age_complete"] = bool(ph["gender"].notna().all() and ph["age"].notna().all())
    assertions["sex_values"] = sorted(ph["gender"].unique().tolist()) == ["F", "M"]
    assert all(assertions.values()), assertions
    pheno = pd.DataFrame({
        "row": np.arange(len(ph), dtype=np.int64), "sample_name": ph["sample_name"],
        "subject_id": ph["subject_id"].astype(str), "sex": ph["gender"], "age": ph["age"].astype(float),
        "site": ph["source"], "plate": ph["plate"].astype(str), "beadchip_position": ph["beadchip_position"],
    })
    pheno.to_parquet(out / "phenotypes.parquet", index=False)

    # ---- matrix (streamed by row blocks; values copied bit-for-bit) ------------------------------
    h5_path = out / "methylation.h5"
    n_cov = len(ccols)
    n_nan = 0
    n_out_of_range = 0
    with h5py.File(src_h5, "r") as f, h5py.File(h5_path, "w") as o:
        src = f["beta"]
        dst = o.create_dataset("beta", shape=(n_samples, n_cov), dtype="float32", chunks=(1, 8192))
        for r0 in range(0, n_samples, args.row_block):
            r1 = min(r0 + args.row_block, n_samples)
            block = np.asarray(src[r0:r1, :], dtype=np.float32)[:, ccols]
            n_nan += int(np.isnan(block).sum())
            n_out_of_range += int(((block < 0) | (block > 1)).sum())
            dst[r0:r1, :] = block
        o.create_dataset("cpg_idx", data=new_ids)
        o.create_dataset("sample_name", data=np.asarray(names, dtype=object), dtype=h5py.string_dtype())
        o.create_dataset("chrom", data=np.asarray(chrom[ccols], dtype=object), dtype=h5py.string_dtype())
        o.create_dataset("pos", data=pos[ccols])
        o.create_dataset("source_cpg_idx", data=src_cpg_idx[ccols])
        o.attrs.update({
            "cpg_namespace": "MethylProphet_global_cpg_idx", "source": "GSE40279 (data/processed/GSE40279) re-keyed",
            "reference_build": "GRCh38", "coordinate_convention": "1-based position of CpG cytosine",
            "beta_dtype": "float32", "beta_values": "untouched copy (no renormalisation, no clipping)",
        })
    assertions["no_nan_in_universe_columns"] = n_nan == 0
    assertions["beta_in_0_1"] = n_out_of_range == 0
    # bitwise re-verification of the written matrix against the source
    with h5py.File(src_h5, "r") as f, h5py.File(h5_path, "r") as o:
        ok = True
        for r0 in range(0, n_samples, args.row_block):
            r1 = min(r0 + args.row_block, n_samples)
            a = np.asarray(f["beta"][r0:r1, :], dtype=np.float32)[:, ccols]
            ok &= bool(np.array_equal(a.view(np.uint32), np.asarray(o["beta"][r0:r1, :]).view(np.uint32)))
        assertions["written_beta_bitwise_equal_source"] = ok
    from cpg_repr_benchmark.data.methylation import read_axis  # the runner's own axis reader/validator
    rid, rnames = read_axis(h5_path)
    assertions["runner_read_axis_ok"] = bool(len(rid) == n_cov and rnames == names)
    assert all(assertions.values()), assertions

    # ---- per-locus mapping table ---------------------------------------------------------------
    status = np.full(n_universe, "unmapped_absent_from_cohort_array", dtype=object)
    status[urows] = "mapped"
    src_col = np.full(n_universe, -1, dtype=np.int64)
    src_col[urows] = ccols
    probe = pd.read_parquet(args.src_dir / "cpg_mapping.parquet")
    group = probe.groupby("cpg_idx").size()
    n_probes_at = np.zeros(n_universe, dtype=np.int64)
    n_probes_at[urows] = group.reindex(src_cpg_idx[ccols]).fillna(0).astype(int).to_numpy()
    table = pd.DataFrame({
        "universe_row": np.arange(n_universe), "cpg_idx": reg["cpg_idx"], "chr": reg["chr"], "pos": reg["pos"],
        "status": status, "cohort_column": src_col, "n_source_probes": n_probes_at,
    })
    table.to_parquet(out / "cpg_mapping_external.parquet", index=False)

    # ---- locus protocol (loci_seen.npz format) ---------------------------------------------------
    locus_path = out / "loci_seen_external_gse40279_v1.npz"
    np.savez_compressed(
        locus_path, train_cpg_idx=np.sort(new_ids), heldout_cpg_idx=np.asarray([], dtype=np.int64),
        seed=np.asarray(LOCUS_SEED, dtype=np.int64), heldout_fraction=np.asarray(0.0, dtype=np.float64))

    # ---- manifest ----------------------------------------------------------------------------
    unm = table[table["status"] != "mapped"]
    in_matched = probe["cpg_idx"].isin(set(src_cpg_idx[ccols].tolist()))
    manifest = {
        "schema": "external_gse40279_v1_mapping_manifest/1",
        "id_system": ("benchmark/TCGA global cpg_idx (MethylProphet); the cohort file uses coordinate-native "
                      "chrom*1e9+pos (GRCh38, 1-based cytosine); joined on exact (chr,pos), no offset, no tolerance. "
                      "The new matrix /cpg_idx is the benchmark global cpg_idx so every frozen store matches by id."),
        "column_order": "benchmark universe order (array_cpg_map.parquet raw_cpg_row) restricted to covered loci",
        "universe": {"n_benchmark": int(n_universe), "n_mapped": len(urows), "n_unmapped": len(unmapped),
                     "n_mapped_chr1_19": int((~table.loc[urows, "chr"].isin(["chr20", "chr21", "chr22"])).sum()),
                     "n_mapped_chr20_22": int(table.loc[urows, "chr"].isin(["chr20", "chr21", "chr22"]).sum())},
        "unmapped_loci": {
            "reason": "no cohort probe at this exact (chr,pos): locus absent from the GSE40279 450K array content",
            "records": [{"cpg_idx": int(r.cpg_idx), "chr": r.chr, "pos": int(r.pos), "reason": r.status}
                        for r in unm.itertuples()],
        },
        "cohort": {"n_samples": int(n_samples), "n_unique_subjects": 656, "n_source_loci": int(n_src_loci),
                   "n_source_probes": len(probe),
                   "probe_status_counts": probe["mapping_status"].value_counts().to_dict(),
                   "n_probes_on_mapped_loci": int(in_matched.sum()),
                   "n_mapped_loci_with_multiple_probes": int((n_probes_at[urows] > 1).sum()),
                   "multi_probe_policy": "nanmean across probes at the same coordinate (already applied in the prepared file; beta untouched here)"},
        "beta": {"nan_policy": "NO NaN expected; asserted 0 NaN in the written matrix (runner would otherwise skip NaN in panels and drop loci without finite train values)",
                 "n_nan": int(n_nan), "n_out_of_0_1": int(n_out_of_range),
                 "renormalisation": "none (bitwise copy, verified)"},
        "assertions": assertions,
        "locus_protocol": {"path": "data/derived/external_gse40279_v1/loci_seen_external_gse40279_v1.npz",
                           "n_train_cpg_idx": int(n_cov), "n_heldout_cpg_idx": 0, "seed": LOCUS_SEED,
                           "heldout_fraction": 0.0, "policy": "genome-wide seen-only (CLAUDE.md); all universe loci -> train; "
                           "849 all-zero-embedding regulatory loci KEPT (identical for all arms)"},
        "sha256": {
            "input_beta_h5": sha256_file(src_h5),
            "input_phenotypes_parquet": sha256_file(args.src_dir / "phenotypes.parquet"),
            "input_cpg_mapping_parquet": sha256_file(args.src_dir / "cpg_mapping.parquet"),
            "input_array_cpg_map_parquet": sha256_file(args.registry),
            "input_loci_seen_tcga_npz": sha256_file(args.loci_seen),
            "output_methylation_h5": sha256_file(h5_path),
            "output_phenotypes_parquet": sha256_file(out / "phenotypes.parquet"),
            "output_cpg_mapping_external_parquet": sha256_file(out / "cpg_mapping_external.parquet"),
            "output_locus_protocol_npz": sha256_file(locus_path),
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    write_json_new(args.manifest, manifest)
    print(json.dumps({k: manifest[k] for k in ("universe", "beta")}, indent=1))
    print("sha256", json.dumps(manifest["sha256"], indent=1))


if __name__ == "__main__":
    main()
