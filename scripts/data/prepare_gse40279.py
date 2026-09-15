#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from cpg_repr_benchmark.data.coordinates import (
    CPG_NAMESPACE,
    COORDINATE_CONVENTION,
    REFERENCE_BUILD,
    encode_cpg_id,
    normalize_chromosome,
)


SERIES_FILE = "GSE40279_series_matrix.txt.gz"
BETA_FILE = "GSE40279_average_beta.txt.gz"
SAMPLE_KEY_FILE = "GSE40279_sample_key.txt.gz"


def _load_pickled_key(db_path: Path, key: str) -> dict:
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute("SELECT value FROM unnamed WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise KeyError(f"{key!r} not found in {db_path}")
        return pickle.loads(row[0])
    finally:
        con.close()


def _subject_from_title(title: str) -> str:
    match = re.search(r"(?:^|\s)(\d+)$", title.strip())
    if not match:
        raise ValueError(f"cannot recover subject id from GEO title: {title!r}")
    return match.group(1)


def _normalize_characteristic_key(key: str) -> str:
    value = key.strip().lower()
    aliases = {
        "age (y)": "age",
        "gender": "gender",
        "source": "source",
        "plate": "plate",
        "ethnicity": "ethnicity",
        "tissue": "tissue",
    }
    return aliases.get(value, re.sub(r"[^a-z0-9]+", "_", value).strip("_"))


def parse_series_matrix(path: Path) -> pd.DataFrame:
    selected: dict[str, list[str]] = {}
    characteristics: list[list[str]] = []
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            key = row[0]
            if key in {"!Sample_title", "!Sample_geo_accession", "!Sample_source_name_ch1"}:
                selected[key] = row[1:]
            elif key == "!Sample_characteristics_ch1":
                characteristics.append(row[1:])

    required = {"!Sample_title", "!Sample_geo_accession"}
    missing = required - set(selected)
    if missing:
        raise ValueError(f"series matrix missing fields: {sorted(missing)}")
    n = len(selected["!Sample_geo_accession"])
    if len(selected["!Sample_title"]) != n:
        raise ValueError("GEO title/accession length mismatch")
    if any(len(row) != n for row in characteristics):
        raise ValueError("GEO characteristics length mismatch")

    rows = []
    for i in range(n):
        title = selected["!Sample_title"][i]
        record = {
            "sample_name": selected["!Sample_geo_accession"][i],
            "subject_id": _subject_from_title(title),
            "geo_title": title,
        }
        if "!Sample_source_name_ch1" in selected:
            record["geo_source_name"] = selected["!Sample_source_name_ch1"][i]
        for char_row in characteristics:
            value = char_row[i]
            if ":" not in value:
                raise ValueError(f"malformed GEO characteristic for {record['sample_name']}: {value!r}")
            key, raw = value.split(":", 1)
            normalized = _normalize_characteristic_key(key)
            if normalized in record:
                raise ValueError(f"duplicate GEO characteristic {normalized!r} for {record['sample_name']}")
            record[normalized] = raw.strip()
        rows.append(record)

    frame = pd.DataFrame(rows)
    if frame["sample_name"].duplicated().any() or frame["subject_id"].duplicated().any():
        raise ValueError("duplicate GEO sample or subject id")
    if "age" not in frame:
        raise ValueError("age characteristic missing from GSE40279")
    frame["age"] = pd.to_numeric(frame["age"], errors="raise").astype("int64")
    if "plate" in frame:
        numeric_plate = pd.to_numeric(frame["plate"], errors="coerce")
        present = frame["plate"].notna() & frame["plate"].astype(str).str.strip().ne("")
        if numeric_plate[present].notna().all():
            frame["plate"] = numeric_plate.astype("Int64")
    return frame


def parse_sample_key(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["sample_key_row", "subject_id", "beadchip_position"],
        dtype={"sample_key_row": "int64", "subject_id": "string", "beadchip_position": "string"},
        compression="gzip",
    )
    if frame["subject_id"].duplicated().any():
        raise ValueError("duplicate subject_id in GSE40279 sample key")
    if frame["beadchip_position"].duplicated().any():
        raise ValueError("duplicate BeadChip position in GSE40279 sample key")
    return frame


def beta_header(path: Path) -> tuple[list[str], list[str]]:
    with gzip.open(path, "rt", newline="") as handle:
        header = next(csv.reader(handle, delimiter="\t"))
    if not header or header[0] != "ID_REF":
        raise ValueError("unexpected GSE40279 beta header")
    matrix_names = header[1:]
    subject_ids = [name[1:] if name.startswith("X") else name for name in matrix_names]
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("duplicate sample columns in beta matrix")
    return matrix_names, subject_ids


def scan_probe_ids(path: Path) -> list[str]:
    probes: list[str] = []
    with gzip.open(path, "rb") as handle:
        handle.readline()
        for line in handle:
            if not line.strip():
                continue
            probes.append(line.split(b"\t", 1)[0].decode())
    if len(probes) != len(set(probes)):
        raise ValueError("GSE40279 contains duplicate probe IDs")
    return probes


def build_probe_mapping(probe_ids: list[str], probe_to_location: dict[str, str]) -> pd.DataFrame:
    records = []
    for source_row, probe in enumerate(probe_ids):
        location = probe_to_location.get(probe)
        if location is None:
            records.append(
                {
                    "source_row": source_row,
                    "probe_id": probe,
                    "illumina_location": None,
                    "chr": None,
                    "pos": pd.NA,
                    "cpg_idx": pd.NA,
                    "mapping_status": "missing_illumina_crosswalk",
                }
            )
            continue
        chrom_raw, pos0_raw = location.split(":")
        chrom = normalize_chromosome(chrom_raw)
        pos = int(pos0_raw) + 1
        records.append(
            {
                "source_row": source_row,
                "probe_id": probe,
                "illumina_location": location,
                "chr": chrom,
                "pos": pos,
                "cpg_idx": encode_cpg_id(chrom, pos),
                "mapping_status": "mapped",
            }
        )

    frame = pd.DataFrame(records)
    mapped = frame["cpg_idx"].notna()
    frame["cpg_idx"] = frame["cpg_idx"].astype("Int64")
    frame["pos"] = frame["pos"].astype("Int64")
    sizes = frame.loc[mapped].groupby("cpg_idx")["probe_id"].transform("size").astype("int64")
    frame.loc[mapped, "aggregation_group_size"] = sizes.to_numpy()
    frame["aggregation_group_size"] = frame["aggregation_group_size"].fillna(0).astype("int64")
    duplicate = mapped & (frame["aggregation_group_size"] > 1)
    frame.loc[mapped & ~duplicate, "mapping_status"] = "mapped_unique"
    frame.loc[duplicate, "mapping_status"] = "mapped_duplicate_coordinate"

    output_cols: dict[int, int] = {}
    next_col = 0
    output_col_values = []
    for value in frame["cpg_idx"]:
        if pd.isna(value):
            output_col_values.append(pd.NA)
            continue
        cpg = int(value)
        if cpg not in output_cols:
            output_cols[cpg] = next_col
            next_col += 1
        output_col_values.append(output_cols[cpg])
    frame["output_column"] = pd.array(output_col_values, dtype="Int64")
    return frame


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _write_beta_h5(
    beta_path: Path,
    output_path: Path,
    mapping: pd.DataFrame,
    sample_names: list[str],
    matrix_names: list[str],
    *,
    chunk_rows: int,
) -> dict:
    mapped = mapping["cpg_idx"].notna()
    firsts = mapping.loc[mapped].drop_duplicates("cpg_idx", keep="first").sort_values("output_column")
    cpg_ids = firsts["cpg_idx"].astype("int64").to_numpy()
    chrom = firsts["chr"].astype(str).to_numpy()
    pos = firsts["pos"].astype("int64").to_numpy()
    group_size = mapping["aggregation_group_size"].to_numpy(dtype=np.int64)
    out_col = mapping["output_column"].fillna(-1).to_numpy(dtype=np.int64)
    probe_expected = mapping["probe_id"].astype(str).to_numpy()

    duplicate_columns = set(
        mapping.loc[mapping["aggregation_group_size"] > 1, "output_column"].dropna().astype("int64").tolist()
    )
    dup_sum = {col: np.zeros(len(sample_names), dtype=np.float64) for col in duplicate_columns}
    dup_count = {col: np.zeros(len(sample_names), dtype=np.int16) for col in duplicate_columns}

    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    finite_total = 0
    out_of_range = 0
    beta_min = np.inf
    beta_max = -np.inf
    cursor = 0
    string_dtype = h5py.string_dtype("utf-8")

    with h5py.File(tmp, "w") as h:
        beta = h.create_dataset(
            "beta",
            shape=(len(sample_names), len(cpg_ids)),
            dtype="f4",
            chunks=(1, min(4096, max(1, len(cpg_ids)))),
            compression="lzf",
            shuffle=True,
            fillvalue=np.nan,
        )
        h.create_dataset("cpg_idx", data=cpg_ids, dtype="i8")
        h.create_dataset("sample_name", data=np.asarray(sample_names, dtype=object), dtype=string_dtype)
        h.create_dataset("chrom", data=np.asarray(chrom, dtype=object), dtype=string_dtype)
        h.create_dataset("pos", data=pos, dtype="i8")
        h.attrs["reference_build"] = REFERENCE_BUILD
        h.attrs["coordinate_convention"] = COORDINATE_CONVENTION
        h.attrs["cpg_namespace"] = CPG_NAMESPACE
        h.attrs["source_accession"] = "GSE40279"

        reader = pd.read_csv(
            beta_path,
            sep="\t",
            compression="gzip",
            chunksize=chunk_rows,
            na_values=["NA", "NaN", "nan", ""],
            low_memory=False,
        )
        for chunk in reader:
            if chunk.columns[0] != "ID_REF":
                raise ValueError("first beta column must be ID_REF")
            observed_matrix_names = list(chunk.columns[1:])
            if observed_matrix_names != matrix_names:
                raise ValueError("beta matrix sample columns changed while streaming")
            n = len(chunk)
            source_slice = slice(cursor, cursor + n)
            probes = chunk.iloc[:, 0].astype(str).to_numpy()
            expected = probe_expected[source_slice]
            if not np.array_equal(probes, expected):
                raise ValueError(f"probe row order mismatch at source row {cursor}")
            values = chunk.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)
            finite = np.isfinite(values)
            if finite.any():
                finite_values = values[finite]
                beta_min = min(beta_min, float(finite_values.min()))
                beta_max = max(beta_max, float(finite_values.max()))
                out_of_range += int(((finite_values < 0.0) | (finite_values > 1.0)).sum())

            cols = out_col[source_slice]
            sizes = group_size[source_slice]
            single = (cols >= 0) & (sizes == 1)
            if single.any():
                single_cols = cols[single]
                if len(single_cols) > 1 and np.any(single_cols[1:] <= single_cols[:-1]):
                    raise RuntimeError("single-probe output columns are not strictly increasing")
                single_values = values[single]
                beta[:, single_cols] = single_values.T
                finite_total += int(np.isfinite(single_values).sum())

            duplicate_rows = np.flatnonzero((cols >= 0) & (sizes > 1))
            for local_row in duplicate_rows:
                col = int(cols[local_row])
                row_values = values[local_row].astype(np.float64)
                good = np.isfinite(row_values)
                dup_sum[col][good] += row_values[good]
                dup_count[col][good] += 1
            cursor += n

        if cursor != len(mapping):
            raise ValueError(f"read {cursor} beta rows but mapping has {len(mapping)}")
        for col in sorted(duplicate_columns):
            count = dup_count[col]
            result = np.full(len(sample_names), np.nan, dtype=np.float32)
            good = count > 0
            result[good] = (dup_sum[col][good] / count[good]).astype(np.float32)
            beta[:, col] = result
            finite_total += int(good.sum())

    if out_of_range:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"found {out_of_range} finite beta values outside [0,1]")
    os.replace(tmp, output_path)
    total = len(sample_names) * len(cpg_ids)
    return {
        "n_loci": int(len(cpg_ids)),
        "n_samples": int(len(sample_names)),
        "finite_beta": int(finite_total),
        "missing_beta_fraction": float(1.0 - finite_total / total) if total else 0.0,
        "beta_min": float(beta_min) if np.isfinite(beta_min) else None,
        "beta_max": float(beta_max) if np.isfinite(beta_max) else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare coordinate-native GSE40279 for CpGRepresentationBenchmark")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/GSE40279"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed/GSE40279"))
    p.add_argument("--illumina-db", type=Path, default=os.environ.get("CPGPT_ILLUMINA_DB"))
    p.add_argument("--chunk-rows", type=int, default=2048)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.illumina_db is None:
        raise ValueError("provide --illumina-db or set CPGPT_ILLUMINA_DB")

    raw = args.raw_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    beta_path = raw / BETA_FILE
    series_path = raw / SERIES_FILE
    key_path = raw / SAMPLE_KEY_FILE
    for path in (beta_path, series_path, key_path, args.illumina_db):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    targets = [
        output / "methylation.h5",
        output / "phenotypes.parquet",
        output / "sample_mapping.parquet",
        output / "cpg_mapping.parquet",
        output / "qc.json",
        output / "manifest.json",
    ]
    if any(path.exists() for path in targets) and not args.force:
        raise FileExistsError(f"processed GSE40279 already exists under {output}; pass --force to replace")
    if args.force:
        for path in targets:
            path.unlink(missing_ok=True)

    phenotype = parse_series_matrix(series_path)
    sample_key = parse_sample_key(key_path)
    matrix_names, beta_subjects = beta_header(beta_path)
    if set(beta_subjects) != set(sample_key["subject_id"].astype(str)):
        raise ValueError("beta subjects and sample_key subjects differ")

    sample_mapping = pd.DataFrame({"matrix_column": np.arange(len(beta_subjects)), "matrix_name": matrix_names, "subject_id": beta_subjects})
    sample_mapping = sample_mapping.merge(sample_key, on="subject_id", how="left", validate="one_to_one")
    sample_mapping = sample_mapping.merge(
        phenotype[["sample_name", "subject_id", "geo_title"]], on="subject_id", how="left", validate="one_to_one"
    )
    if sample_mapping[["sample_name", "beadchip_position"]].isna().any().any():
        raise ValueError("incomplete sample mapping")
    sample_names = sample_mapping["sample_name"].astype(str).tolist()
    phenotype = sample_mapping[["matrix_column", "matrix_name", "sample_name", "subject_id", "beadchip_position", "geo_title"]].merge(
        phenotype.drop(columns=["subject_id", "geo_title"]), on="sample_name", how="left", validate="one_to_one"
    )
    if phenotype["age"].isna().any():
        raise ValueError("age missing after sample alignment")

    probe_ids = scan_probe_ids(beta_path)
    probe_to_location = _load_pickled_key(Path(args.illumina_db), "homo_sapiens")
    mapping = build_probe_mapping(probe_ids, probe_to_location)
    n_missing_crosswalk = int((mapping["mapping_status"] == "missing_illumina_crosswalk").sum())
    mapped = mapping["cpg_idx"].notna()
    n_unique_loci = int(mapping.loc[mapped, "cpg_idx"].nunique())
    n_duplicate_extra = int(mapped.sum() - n_unique_loci)

    _atomic_parquet(mapping, output / "cpg_mapping.parquet")
    _atomic_parquet(sample_mapping, output / "sample_mapping.parquet")
    _atomic_parquet(phenotype, output / "phenotypes.parquet")
    matrix_qc = _write_beta_h5(
        beta_path,
        output / "methylation.h5",
        mapping,
        sample_names,
        matrix_names,
        chunk_rows=args.chunk_rows,
    )

    qc = {
        "schema_version": 1,
        "accession": "GSE40279",
        "source_probes": int(len(mapping)),
        "crosswalk_mapped_probes": int(mapped.sum()),
        "crosswalk_missing_probes": n_missing_crosswalk,
        "crosswalk_coverage_fraction": float(mapped.mean()),
        "unique_coordinate_loci": n_unique_loci,
        "extra_probes_collapsed_by_coordinate": n_duplicate_extra,
        **matrix_qc,
        "age_min": int(phenotype["age"].min()),
        "age_max": int(phenotype["age"].max()),
        "age_mean": float(phenotype["age"].mean()),
    }
    (output / "qc.json").write_text(json.dumps(qc, indent=2, sort_keys=True))
    manifest = {
        "schema_version": 1,
        "dataset": "GSE40279",
        "organism": "Homo sapiens",
        "tissue": "whole blood",
        "platform": "GPL13534",
        "platform_name": "Illumina HumanMethylation450 BeadChip",
        "reference_build": REFERENCE_BUILD,
        "coordinate_convention": COORDINATE_CONVENTION,
        "cpg_namespace": CPG_NAMESPACE,
        "beta_source": str(beta_path),
        "phenotype_source": str(series_path),
        "sample_key_source": str(key_path),
        "probe_crosswalk": str(Path(args.illumina_db).resolve()),
        "probe_crosswalk_species": "homo_sapiens",
        "probe_crosswalk_position_conversion": "Illumina/CpGPT 0-based -> benchmark 1-based cytosine (+1)",
        "multi_probe_policy": "nanmean beta across probes mapping to the same GRCh38 CpG coordinate",
        "artifacts": {path.name: str(path) for path in targets},
        "qc": qc,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(qc, indent=2, sort_keys=True))
    print(f"\nWROTE={output}")


if __name__ == "__main__":
    main()
