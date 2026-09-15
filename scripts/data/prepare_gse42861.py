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

# Unlike GSE40279, GSE42861 ships phenotype metadata and the full beta matrix in a
# single combined series-matrix file: header lines, then the "!series_matrix_table_begin"
# marker, then a tab-separated ID_REF x GSM beta table, then "!series_matrix_table_end".
SERIES_FILE = "GSE42861_series_matrix.txt.gz"

DISEASE_STATES = {"rheumatoid arthritis": 1, "normal": 0}


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
    match = re.search(r"sample\s+(\d+)$", title.strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot recover subject id from GEO title: {title!r}")
    return match.group(1)


def _normalize_characteristic_key(key: str) -> str:
    value = key.strip().lower()
    aliases = {
        "cell type": "cell_type",
        "disease state": "disease_state",
        "subject": "subject_group",
        "age": "age",
        "gender": "gender",
        "smoking status": "smoking_status",
    }
    return aliases.get(value, re.sub(r"[^a-z0-9]+", "_", value).strip("_"))


def parse_series_matrix_and_probes(path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Single pass over the combined series-matrix file.

    Returns (phenotype_frame, matrix_names, probe_ids). matrix_names are the beta-table
    sample columns in on-disk order (== !Sample_geo_accession order for GSE42861).
    """
    selected: dict[str, list[str]] = {}
    characteristics: list[list[str]] = []
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            key = row[0]
            if key == "!series_matrix_table_begin":
                break
            if key in {"!Sample_title", "!Sample_geo_accession", "!Sample_source_name_ch1"}:
                selected[key] = row[1:]
            elif key == "!Sample_characteristics_ch1":
                characteristics.append(row[1:])
        else:
            raise ValueError("!series_matrix_table_begin not found in series matrix")

        table_header = next(reader)
        if not table_header or table_header[0] != "ID_REF":
            raise ValueError("unexpected GSE42861 beta table header")
        matrix_names = table_header[1:]

        probe_ids: list[str] = []
        for row in reader:
            if not row:
                continue
            if row[0] == "!series_matrix_table_end":
                break
            probe_ids.append(row[0])
        else:
            raise ValueError("!series_matrix_table_end not found in series matrix")

    required = {"!Sample_title", "!Sample_geo_accession"}
    missing = required - set(selected)
    if missing:
        raise ValueError(f"series matrix missing fields: {sorted(missing)}")
    n = len(selected["!Sample_geo_accession"])
    if len(selected["!Sample_title"]) != n:
        raise ValueError("GEO title/accession length mismatch")
    if any(len(row) != n for row in characteristics):
        raise ValueError("GEO characteristics length mismatch")
    if len(matrix_names) != n:
        raise ValueError("beta table sample columns do not match phenotype sample count")

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
    if frame["sample_name"].tolist() != matrix_names:
        raise ValueError("phenotype sample order does not match beta table column order")
    if "disease_state" not in frame:
        raise ValueError("disease state characteristic missing from GSE42861")

    normalized_state = frame["disease_state"].str.strip().str.lower()
    unknown = sorted(set(normalized_state) - set(DISEASE_STATES))
    if unknown:
        raise ValueError(f"unexpected GSE42861 disease state values: {unknown}")
    frame["disease_label"] = normalized_state.map(DISEASE_STATES).astype("int64")

    if "age" in frame:
        frame["age"] = pd.to_numeric(frame["age"], errors="raise").astype("int64")

    return frame, matrix_names, probe_ids


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
    series_path: Path,
    output_path: Path,
    mapping: pd.DataFrame,
    sample_names: list[str],
    matrix_names: list[str],
    header_lines: int,
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
        h.attrs["source_accession"] = "GSE42861"

        reader = pd.read_csv(
            series_path,
            sep="\t",
            compression="gzip",
            skiprows=header_lines,
            nrows=len(mapping),
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


def _count_header_lines(path: Path) -> int:
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        count = 0
        for row in reader:
            count += 1
            if row and row[0] == "!series_matrix_table_begin":
                return count
    raise ValueError("!series_matrix_table_begin not found in series matrix")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare coordinate-native GSE42861 for CpGRepresentationBenchmark")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/GSE42861"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed/GSE42861"))
    p.add_argument("--illumina-db", type=Path, default=os.environ.get("CPGPT_ILLUMINA_DB"))
    p.add_argument("--chunk-rows", type=int, default=2048)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.illumina_db is None:
        raise ValueError("provide --illumina-db or set CPGPT_ILLUMINA_DB")

    raw = args.raw_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    series_path = raw / SERIES_FILE
    for path in (series_path, args.illumina_db):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    targets = [
        output / "methylation.h5",
        output / "phenotypes.parquet",
        output / "cpg_mapping.parquet",
        output / "qc.json",
        output / "manifest.json",
    ]
    if any(path.exists() for path in targets) and not args.force:
        raise FileExistsError(f"processed GSE42861 already exists under {output}; pass --force to replace")
    if args.force:
        for path in targets:
            path.unlink(missing_ok=True)

    phenotype, matrix_names, probe_ids = parse_series_matrix_and_probes(series_path)
    if len(probe_ids) != len(set(probe_ids)):
        raise ValueError("GSE42861 contains duplicate probe IDs")
    header_lines = _count_header_lines(series_path)
    sample_names = phenotype["sample_name"].astype(str).tolist()

    probe_to_location = _load_pickled_key(Path(args.illumina_db), "homo_sapiens")
    mapping = build_probe_mapping(probe_ids, probe_to_location)
    n_missing_crosswalk = int((mapping["mapping_status"] == "missing_illumina_crosswalk").sum())
    mapped = mapping["cpg_idx"].notna()
    n_unique_loci = int(mapping.loc[mapped, "cpg_idx"].nunique())
    n_duplicate_extra = int(mapped.sum() - n_unique_loci)

    _atomic_parquet(mapping, output / "cpg_mapping.parquet")
    _atomic_parquet(phenotype, output / "phenotypes.parquet")
    matrix_qc = _write_beta_h5(
        series_path,
        output / "methylation.h5",
        mapping,
        sample_names,
        matrix_names,
        header_lines,
        chunk_rows=args.chunk_rows,
    )

    disease_counts = phenotype["disease_label"].value_counts().to_dict()
    qc = {
        "schema_version": 1,
        "accession": "GSE42861",
        "source_probes": int(len(mapping)),
        "crosswalk_mapped_probes": int(mapped.sum()),
        "crosswalk_missing_probes": n_missing_crosswalk,
        "crosswalk_coverage_fraction": float(mapped.mean()),
        "unique_coordinate_loci": n_unique_loci,
        "extra_probes_collapsed_by_coordinate": n_duplicate_extra,
        **matrix_qc,
        "n_rheumatoid_arthritis": int(disease_counts.get(1, 0)),
        "n_control": int(disease_counts.get(0, 0)),
    }
    (output / "qc.json").write_text(json.dumps(qc, indent=2, sort_keys=True))
    manifest = {
        "schema_version": 1,
        "dataset": "GSE42861",
        "organism": "Homo sapiens",
        "tissue": "peripheral blood leukocytes",
        "platform": "GPL13534",
        "platform_name": "Illumina HumanMethylation450 BeadChip",
        "reference_build": REFERENCE_BUILD,
        "coordinate_convention": COORDINATE_CONVENTION,
        "cpg_namespace": CPG_NAMESPACE,
        "beta_source": str(series_path),
        "phenotype_source": str(series_path),
        "probe_crosswalk": str(Path(args.illumina_db).resolve()),
        "probe_crosswalk_species": "homo_sapiens",
        "probe_crosswalk_position_conversion": "Illumina/CpGPT 0-based -> benchmark 1-based cytosine (+1)",
        "multi_probe_policy": "nanmean beta across probes mapping to the same GRCh38 CpG coordinate",
        "disease_label_encoding": {"control": 0, "rheumatoid_arthritis": 1},
        "artifacts": {path.name: str(path) for path in targets},
        "qc": qc,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(qc, indent=2, sort_keys=True))
    print(f"\nWROTE={output}")


if __name__ == "__main__":
    main()
