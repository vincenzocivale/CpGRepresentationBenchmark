#!/usr/bin/env python3
"""Prepare the GSE147221 schizophrenia/control cohort for classification.

GEO publishes phenotype data in the series matrix but the beta values in the
``Dublin_blood_processed_signals`` supplement.  That CSV alternates a beta and
its detection P value for every array barcode, so it must not be consumed as a
plain beta matrix.
"""
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
    CPG_NAMESPACE, COORDINATE_CONVENTION, REFERENCE_BUILD, encode_cpg_id,
    normalize_chromosome,
)

SERIES_FILE = "GSE147221_series_matrix.txt.gz"
SIGNALS_FILE = "GSE147221_Dublin_blood_processed_signals.csv.gz"


def _unquote(value: str) -> str:
    return value.strip().strip('"')


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _array_id(title: str) -> str:
    value = title.split(":", 1)[0].strip()
    if not value:
        raise ValueError(f"cannot recover array barcode from GEO title {title!r}")
    return value


def parse_phenotypes(path: Path) -> pd.DataFrame:
    """Return eligible biological samples, excluding technical/QC exclusions."""
    selected: dict[str, list[str]] = {}
    characteristics: list[list[str]] = []
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            if row[0] == "!series_matrix_table_begin":
                break
            if row[0] in {"!Sample_title", "!Sample_geo_accession", "!Sample_source_name_ch1", "!Sample_description"}:
                selected[row[0]] = [_unquote(x) for x in row[1:]]
            elif row[0] == "!Sample_characteristics_ch1":
                characteristics.append([_unquote(x) for x in row[1:]])
        else:
            raise ValueError("!series_matrix_table_begin not found")

    required = {"!Sample_title", "!Sample_geo_accession"}
    if required - set(selected):
        raise ValueError(f"series matrix missing {sorted(required - set(selected))}")
    n = len(selected["!Sample_title"])
    if any(len(x) != n for x in selected.values()) or any(len(x) != n for x in characteristics):
        raise ValueError("inconsistent GEO sample metadata lengths")
    records = []
    for i in range(n):
        item = {
            "sample_name": selected["!Sample_geo_accession"][i],
            "array_id": _array_id(selected["!Sample_title"][i]),
            "geo_title": selected["!Sample_title"][i],
            "geo_source_name": selected.get("!Sample_source_name_ch1", [""] * n)[i],
            "geo_description": selected.get("!Sample_description", [""] * n)[i],
        }
        for values in characteristics:
            if ":" not in values[i]:
                raise ValueError(f"malformed characteristic for {item['sample_name']}: {values[i]!r}")
            name, value = values[i].split(":", 1)
            name = _key(name)
            if name in item:
                raise ValueError(f"duplicate characteristic {name!r}")
            item[name] = value.strip()
        records.append(item)
    frame = pd.DataFrame(records)
    if frame["sample_name"].duplicated().any() or frame["array_id"].duplicated().any():
        raise ValueError("GEO sample accessions and array barcodes must be unique")
    if "status" not in frame:
        raise ValueError("GSE147221 status characteristic is missing")
    status = frame["status"].str.strip().str.lower()
    frame["disease_label"] = status.map({"control": 0, "case": 1}).astype("Int64")
    frame["included"] = frame["disease_label"].notna() & ~frame["geo_description"].str.contains(
        "excluded from the final analysis", case=False, na=False
    )
    frame["exclusion_reason"] = pd.NA
    frame.loc[frame["disease_label"].isna(), "exclusion_reason"] = "technical_or_unlabelled_control"
    frame.loc[
        frame["disease_label"].notna() & ~frame["included"], "exclusion_reason"
    ] = "geo_low_quality"
    if "age" in frame:
        frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    return frame


def _load_crosswalk(db_path: Path) -> dict[str, str]:
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute("SELECT value FROM unnamed WHERE key = ?", ("homo_sapiens",)).fetchone()
        if row is None:
            raise KeyError("'homo_sapiens' not found in Illumina database")
        return pickle.loads(row[0])
    finally:
        con.close()


def build_probe_mapping(probe_ids: list[str], probe_to_location: dict[str, str]) -> pd.DataFrame:
    records = []
    for source_row, probe in enumerate(probe_ids):
        location = probe_to_location.get(probe)
        if location is None:
            records.append((source_row, probe, None, None, pd.NA, pd.NA, "missing_illumina_crosswalk"))
            continue
        chrom_raw, pos0_raw = location.split(":")
        chrom, pos = normalize_chromosome(chrom_raw), int(pos0_raw) + 1
        records.append((source_row, probe, location, chrom, pos, encode_cpg_id(chrom, pos), "mapped"))
    frame = pd.DataFrame(records, columns=["source_row", "probe_id", "illumina_location", "chr", "pos", "cpg_idx", "mapping_status"])
    frame["pos"] = frame["pos"].astype("Int64")
    frame["cpg_idx"] = frame["cpg_idx"].astype("Int64")
    mapped = frame["cpg_idx"].notna()
    size = frame.loc[mapped].groupby("cpg_idx")["probe_id"].transform("size")
    frame["aggregation_group_size"] = 0
    frame.loc[mapped, "aggregation_group_size"] = size.to_numpy()
    frame.loc[mapped & (frame["aggregation_group_size"] == 1), "mapping_status"] = "mapped_unique"
    frame.loc[mapped & (frame["aggregation_group_size"] > 1), "mapping_status"] = "mapped_duplicate_coordinate"
    output = {}
    frame["output_column"] = pd.array(
        [pd.NA if pd.isna(v) else output.setdefault(int(v), len(output)) for v in frame["cpg_idx"]], dtype="Int64"
    )
    return frame


def _signal_columns(path: Path, included_arrays: set[str]) -> tuple[list[str], list[str]]:
    columns = pd.read_csv(path, compression="gzip", nrows=0).columns.tolist()
    beta = [x for x in columns[1:] if not x.endswith("_Detection_Pval")]
    pvals = [f"{x}_Detection_Pval" for x in beta]
    missing = [x for x in beta if f"{x}_Detection_Pval" not in columns]
    if missing:
        raise ValueError(f"processed signals lack detection P values for {missing[:5]}")
    absent = sorted(included_arrays - set(beta))
    if absent:
        raise ValueError(f"eligible GEO arrays absent from processed signals: {absent[:5]}")
    selected = [x for x in beta if x in included_arrays]
    return selected, [f"{x}_Detection_Pval" for x in selected]


def write_h5(signals: Path, output: Path, mapping: pd.DataFrame, arrays: list[str], threshold: float, chunk_rows: int) -> dict:
    first = mapping.loc[mapping["cpg_idx"].notna()].drop_duplicates("cpg_idx", keep="first").sort_values("output_column")
    cpg_ids = first["cpg_idx"].astype("int64").to_numpy()
    columns = mapping["output_column"].fillna(-1).to_numpy(dtype=np.int64)
    sizes = mapping["aggregation_group_size"].to_numpy(dtype=np.int64)
    use_beta, use_pval = _signal_columns(signals, set(arrays))
    if set(use_beta) != set(arrays):
        raise AssertionError("signal column selection lost an eligible array")
    order = [use_beta.index(x) for x in arrays]
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    string = h5py.string_dtype("utf-8")
    finite_total = 0
    duplicate_columns = sorted(set(columns[(columns >= 0) & (sizes > 1)]))
    duplicate_sum = {col: np.zeros(len(arrays), dtype=np.float64) for col in duplicate_columns}
    duplicate_count = {col: np.zeros(len(arrays), dtype=np.int16) for col in duplicate_columns}
    with h5py.File(tmp, "w") as h:
        beta_h5 = h.create_dataset("beta", shape=(len(arrays), len(cpg_ids)), dtype="f4", chunks=(1, min(4096, len(cpg_ids))), compression="lzf", shuffle=True, fillvalue=np.nan)
        h.create_dataset("cpg_idx", data=cpg_ids, dtype="i8")
        h.create_dataset("sample_name", data=np.asarray(arrays, dtype=object), dtype=string)
        h.create_dataset("chrom", data=np.asarray(first["chr"].astype(str), dtype=object), dtype=string)
        h.create_dataset("pos", data=first["pos"].astype("int64"), dtype="i8")
        h.attrs.update(reference_build=REFERENCE_BUILD, coordinate_convention=COORDINATE_CONVENTION, cpg_namespace=CPG_NAMESPACE, source_accession="GSE147221", detection_pvalue_threshold=threshold)
        cursor = 0
        probe_column = pd.read_csv(signals, compression="gzip", nrows=0).columns[0]
        for chunk in pd.read_csv(
            signals, compression="gzip", usecols=[probe_column, *use_beta, *use_pval], chunksize=chunk_rows
        ):
            n = len(chunk)
            expected = mapping["probe_id"].iloc[cursor:cursor + n].tolist()
            probes = chunk.iloc[:, 0].astype(str).tolist()
            if probes != expected:
                raise ValueError(f"processed-signals probe order mismatch at source row {cursor}")
            values = chunk[use_beta].to_numpy(dtype=np.float32, copy=False)[:, order]
            pvals = chunk[use_pval].to_numpy(dtype=np.float32, copy=False)[:, order]
            values[(pvals > threshold) | ~np.isfinite(pvals)] = np.nan
            source_columns = columns[cursor:cursor + n]
            source_sizes = sizes[cursor:cursor + n]
            single = (source_columns >= 0) & (source_sizes == 1)
            if single.any():
                single_columns = source_columns[single]
                if len(single_columns) > 1 and np.any(single_columns[1:] <= single_columns[:-1]):
                    raise RuntimeError("unique mapped output columns must be strictly increasing")
                beta_h5[:, single_columns] = values[single].T
            for local in np.flatnonzero((source_columns >= 0) & (source_sizes > 1)):
                col = int(source_columns[local])
                # Coordinate collisions are rare on 450K; retain an explicit nanmean policy.
                if col in duplicate_sum:
                    good = np.isfinite(values[local])
                    duplicate_sum[col][good] += values[local, good]
                    duplicate_count[col][good] += 1
            finite_total += int(np.isfinite(values[source_columns >= 0]).sum())
            cursor += n
        for col in duplicate_columns:
            result = np.full(len(arrays), np.nan, dtype=np.float32)
            good = duplicate_count[col] > 0
            result[good] = (duplicate_sum[col][good] / duplicate_count[col][good]).astype(np.float32)
            beta_h5[:, col] = result
    if cursor != len(mapping):
        tmp.unlink(missing_ok=True)
        raise ValueError(f"processed signals have {cursor} probes; expected {len(mapping)}")
    os.replace(tmp, output)
    return {"n_samples": len(arrays), "n_loci": len(cpg_ids), "finite_source_values": finite_total}


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare GSE147221 schizophrenia classification data")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/GSE147221"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed/GSE147221"))
    p.add_argument("--illumina-db", type=Path, default=os.environ.get("CPGPT_ILLUMINA_DB"))
    p.add_argument("--detection-pvalue-threshold", type=float, default=0.01)
    p.add_argument("--chunk-rows", type=int, default=512)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.illumina_db is None:
        raise ValueError("provide --illumina-db or set CPGPT_ILLUMINA_DB")
    raw, output = args.raw_dir.resolve(), args.output_dir.resolve()
    series, signals, db = raw / SERIES_FILE, raw / SIGNALS_FILE, Path(args.illumina_db)
    for path in (series, signals, db):
        if not path.is_file(): raise FileNotFoundError(path)
    if not 0 <= args.detection_pvalue_threshold <= 1: raise ValueError("detection P-value threshold must be in [0, 1]")
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / x for x in ("methylation.h5", "phenotypes.parquet", "cpg_mapping.parquet", "qc.json", "manifest.json")]
    if any(x.exists() for x in targets) and not args.force: raise FileExistsError(f"processed data exist under {output}; use --force")
    if args.force:
        for x in targets: x.unlink(missing_ok=True)
    phenotype = parse_phenotypes(series)
    eligible = phenotype.loc[phenotype["included"]].copy()
    if eligible["disease_label"].isna().any() or eligible.empty: raise ValueError("no eligible labelled samples")
    probe_ids = pd.read_csv(signals, compression="gzip", usecols=[0]).iloc[:, 0].astype(str).tolist()
    if len(probe_ids) != len(set(probe_ids)): raise ValueError("processed signals contain duplicate probe IDs")
    mapping = build_probe_mapping(probe_ids, _load_crosswalk(db))
    arrays = eligible["array_id"].tolist()
    qc = write_h5(signals, output / "methylation.h5", mapping, arrays, args.detection_pvalue_threshold, args.chunk_rows)
    phenotype.to_parquet(output / "phenotypes.parquet", index=False)
    mapping.to_parquet(output / "cpg_mapping.parquet", index=False)
    labels = eligible["disease_label"].astype(int).value_counts().to_dict()
    qc.update(accession="GSE147221", source_samples=int(len(phenotype)), included_samples=int(len(eligible)), n_schizophrenia=int(labels.get(1, 0)), n_control=int(labels.get(0, 0)), n_excluded=int((~phenotype["included"]).sum()), crosswalk_coverage_fraction=float(mapping["cpg_idx"].notna().mean()))
    (output / "qc.json").write_text(json.dumps(qc, indent=2, sort_keys=True))
    manifest = {"schema_version": 1, "dataset": "GSE147221", "organism": "Homo sapiens", "tissue": "whole blood/buffy coat", "platform": "GPL13534", "platform_name": "Illumina HumanMethylation450 BeadChip", "reference_build": REFERENCE_BUILD, "coordinate_convention": COORDINATE_CONVENTION, "cpg_namespace": CPG_NAMESPACE, "beta_source": str(signals), "phenotype_source": str(series), "detection_pvalue_threshold": args.detection_pvalue_threshold, "technical_sample_policy": "exclude status other than Case/Control and GEO low-quality exclusions", "disease_label_encoding": {"control": 0, "schizophrenia": 1}, "qc": qc}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(qc, indent=2, sort_keys=True)); print(f"WROTE={output}")


if __name__ == "__main__": main()
