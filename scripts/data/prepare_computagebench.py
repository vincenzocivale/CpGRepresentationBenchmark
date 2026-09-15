#!/usr/bin/env python3
"""Convert a local ComputAgeBench snapshot to the benchmark HDF5 contract.

ComputAgeBench stores one Parquet file per GEO study, with Illumina probe IDs in
rows and GEO sample IDs in columns.  This script deliberately does not download
the snapshot or version its (large) contents.  Download it with the official
``huggingface_hub.snapshot_download`` command, then provide a GRCh38
probe-to-coordinate crosswalk.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cpg_repr_benchmark.data.coordinates import (
    COORDINATE_CONVENTION,
    CPG_NAMESPACE,
    REFERENCE_BUILD,
    encode_many,
    normalize_chromosome,
)


def _study_id(path: Path, split: str) -> str:
    prefix = f"computage_{'bench' if split == 'benchmark' else 'train'}_data_"
    if not path.name.startswith(prefix) or path.suffix != ".parquet":
        raise ValueError(f"not a ComputAgeBench {split} matrix: {path}")
    return path.stem.removeprefix(prefix)


def discover_studies(snapshot: Path, split: str, selected: set[str] | None = None) -> list[tuple[str, Path]]:
    directory = snapshot / "data" / split
    if not directory.is_dir():
        raise FileNotFoundError(f"missing ComputAgeBench directory: {directory}")
    studies = sorted((_study_id(path, split), path) for path in directory.glob("*.parquet"))
    if selected is not None:
        found = {name for name, _ in studies}
        missing = selected - found
        if missing:
            raise ValueError(f"requested studies absent from {split}: {sorted(missing)}")
        studies = [(name, path) for name, path in studies if name in selected]
    if not studies:
        raise ValueError("no ComputAgeBench study matrices selected")
    return studies


def parquet_probe_ids(path: Path) -> list[str]:
    """Read only the persisted Pandas index (the probe IDs), not the matrix."""
    schema = pq.ParquetFile(path).schema_arrow
    metadata = schema.pandas_metadata or {}
    index_columns = metadata.get("index_columns", [])
    physical = [value for value in index_columns if isinstance(value, str) and value in schema.names]
    if len(physical) != 1:
        raise ValueError(f"cannot identify exactly one persisted probe index in {path}")
    table = pq.read_table(path, columns=physical)
    return table.column(physical[0]).to_pylist()


def load_crosswalk_data(table: pd.DataFrame) -> pd.DataFrame:
    required = {"probe_id", "chr", "pos"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"probe crosswalk is missing columns: {sorted(missing)}")
    table = table.loc[:, ["probe_id", "chr", "pos"]].dropna().copy()
    table["probe_id"] = table["probe_id"].astype(str)
    if table["probe_id"].duplicated().any():
        raise ValueError("probe crosswalk contains duplicate probe_id values")
    table["chr"] = table["chr"].map(normalize_chromosome)
    table["pos"] = pd.to_numeric(table["pos"], errors="raise").astype("int64")
    table["cpg_idx"] = encode_many(table["chr"], table["pos"])
    return table


def load_crosswalk(path: Path) -> pd.DataFrame:
    table = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path, sep="\t")
    return load_crosswalk_data(table)


def build_probe_mapping(probe_ids: list[str], crosswalk: pd.DataFrame) -> pd.DataFrame:
    probes = pd.DataFrame({"source_row": np.arange(len(probe_ids), dtype=np.int64), "probe_id": probe_ids})
    if probes["probe_id"].duplicated().any():
        raise ValueError("study matrix contains duplicate probe IDs")
    mapping = probes.merge(crosswalk, on="probe_id", how="left", validate="one_to_one")
    mapped = mapping["cpg_idx"].notna()
    mapping["mapping_status"] = np.where(mapped, "mapped", "missing_crosswalk")
    mapping["cpg_idx"] = mapping["cpg_idx"].astype("Int64")
    mapping["pos"] = mapping["pos"].astype("Int64")
    size = mapping.loc[mapped].groupby("cpg_idx")["probe_id"].transform("size")
    mapping["aggregation_group_size"] = size.reindex(mapping.index).fillna(0).astype("int64")
    mapping.loc[mapped & (mapping["aggregation_group_size"] == 1), "mapping_status"] = "mapped_unique"
    mapping.loc[mapped & (mapping["aggregation_group_size"] > 1), "mapping_status"] = "mapped_duplicate_coordinate"
    return mapping


def _metadata(snapshot: Path, split: str, studies: set[str]) -> pd.DataFrame:
    filename = "computage_bench_meta.tsv" if split == "benchmark" else "computage_train_meta.tsv"
    path = snapshot / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    meta = pd.read_csv(path, sep="\t", index_col=0)
    meta.index = meta.index.astype(str)
    if "DatasetID" not in meta or "Age" not in meta:
        raise ValueError("ComputAgeBench metadata must contain DatasetID and Age")
    meta = meta.loc[meta["DatasetID"].astype(str).isin(studies)].copy()
    meta["dataset_id"] = meta["DatasetID"].astype(str)
    # Six GEO IDs in the released benchmark occur in two distinct studies.  A
    # dataset-qualified ID keeps the HDF5 axis unique without discarding either
    # matrix column; retain the original ID for lookup and provenance.
    meta["source_sample_name"] = meta.index
    meta["sample_name"] = meta["dataset_id"] + ":" + meta["source_sample_name"]
    if meta["sample_name"].duplicated().any():
        raise ValueError("ComputAgeBench has duplicate dataset-qualified sample IDs")
    meta["age"] = pd.to_numeric(meta["Age"], errors="coerce").astype("float32")
    if meta["age"].isna().any():
        raise ValueError("selected metadata contains missing/non-numeric Age values")
    meta["condition"] = meta.get("Condition", pd.Series("unknown", index=meta.index)).astype(str)
    meta["condition_class"] = meta.get("Class", pd.Series("unknown", index=meta.index)).astype(str)
    meta["is_healthy_control"] = (meta["condition_class"] == "HC").astype("int64")
    meta["is_aging_accelerating_condition"] = (meta["condition_class"] != "HC").astype("int64")
    return meta.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare coordinate-native ComputAgeBench data")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("data/raw/ComputAgeBench"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/ComputAgeBench/benchmark"))
    parser.add_argument("--probe-crosswalk", type=Path, required=True, help="GRCh38 table with probe_id,chr,pos")
    parser.add_argument("--split", choices=("benchmark", "train"), default="benchmark")
    parser.add_argument("--study", action="append", help="repeatable GEO study ID; default: all")
    parser.add_argument("--compression", default="gzip")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    snapshot, output = args.snapshot_dir.resolve(), args.output_dir.resolve()
    studies = discover_studies(snapshot, args.split, set(args.study) if args.study else None)
    meta = _metadata(snapshot, args.split, {name for name, _ in studies})
    crosswalk = load_crosswalk(args.probe_crosswalk.resolve())
    targets = [output / name for name in ("methylation.h5", "phenotypes.parquet", "cpg_mapping.parquet", "qc.json", "manifest.json")]
    if any(path.exists() for path in targets) and not args.force:
        raise FileExistsError(f"processed ComputAgeBench already exists under {output}; pass --force to replace")
    output.mkdir(parents=True, exist_ok=True)
    if args.force:
        for path in targets:
            path.unlink(missing_ok=True)

    # The same array probes occur in many studies.  Retain one auditable mapping
    # per source probe rather than materializing a 65-study Cartesian duplicate.
    observed_probes: set[str] = set()
    for _, matrix_path in studies:
        observed_probes.update(map(str, parquet_probe_ids(matrix_path)))
    mapping = build_probe_mapping(sorted(observed_probes), crosswalk)
    mapped = mapping.dropna(subset=["cpg_idx"]).copy()
    loci = mapped.loc[:, ["cpg_idx", "chr", "pos"]].drop_duplicates("cpg_idx").sort_values("cpg_idx")
    loci["cpg_idx"] = loci["cpg_idx"].astype("int64")
    loci["output_column"] = np.arange(len(loci), dtype=np.int64)
    cpg_to_column = dict(zip(loci["cpg_idx"], loci["output_column"], strict=True))
    mapping["output_column"] = mapping["cpg_idx"].map(cpg_to_column).astype("Int64")
    meta = meta.sort_values("dataset_id", kind="stable").reset_index(drop=True)
    counts = meta.groupby("dataset_id", sort=False).size()
    if set(counts.index) != {study for study, _ in studies}:
        raise ValueError("selected study matrices and metadata disagree")

    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(output / "methylation.h5", "w") as handle:
        beta = handle.create_dataset(
            "beta", shape=(len(meta), len(loci)), dtype="float32", fillvalue=np.nan,
            chunks=(max(1, min(32, len(meta))), max(1, min(4096, len(loci)))), compression=args.compression,
        )
        handle.create_dataset("cpg_idx", data=loci["cpg_idx"].to_numpy(dtype=np.int64))
        handle.create_dataset("sample_name", data=meta["sample_name"].to_numpy(dtype=object), dtype=string_dtype)
        handle.create_dataset("chrom", data=loci["chr"].to_numpy(dtype=object), dtype=string_dtype)
        handle.create_dataset("pos", data=loci["pos"].to_numpy(dtype=np.int64))
        handle.attrs["reference_build"] = REFERENCE_BUILD
        handle.attrs["coordinate_convention"] = COORDINATE_CONVENTION
        handle.attrs["cpg_namespace"] = CPG_NAMESPACE
        handle.attrs["source_dataset"] = "ComputAgeBench"
        cursor = 0
        for study, matrix_path in studies:
            frame = pd.read_parquet(matrix_path)
            frame.index = frame.index.astype(str)
            source_samples = meta.loc[meta["dataset_id"] == study, "source_sample_name"].tolist()
            missing_samples = set(source_samples) - set(frame.columns.astype(str))
            if missing_samples:
                raise ValueError(f"{study} matrix is missing metadata samples: {sorted(missing_samples)[:5]}")
            source = mapping.loc[mapping["cpg_idx"].notna()].copy()
            source = source.set_index("probe_id").reindex(frame.index).dropna(subset=["cpg_idx"])
            source["output_column"] = source["output_column"].astype("int64")
            values = frame.loc[source.index, source_samples].to_numpy(dtype=np.float32, copy=True).T
            columns = source["output_column"].to_numpy(dtype=np.int64)
            unique = ~source["cpg_idx"].duplicated(keep=False).to_numpy()
            if unique.any():
                order = np.argsort(columns[unique], kind="stable")
                beta[cursor : cursor + len(source_samples), columns[unique][order]] = values[:, unique][:, order]
            for _, group in source.loc[~unique].groupby("cpg_idx", sort=False):
                positions = source.index.get_indexer(group.index)
                beta[cursor : cursor + len(source_samples), int(group["output_column"].iloc[0])] = np.nanmean(values[:, positions], axis=1)
            cursor += len(source_samples)

    mapping.to_parquet(output / "cpg_mapping.parquet", index=False)
    meta.to_parquet(output / "phenotypes.parquet", index=False)
    qc = {
        "schema_version": 1, "dataset": "ComputAgeBench", "split": args.split,
        "n_studies": len(studies), "n_samples": len(meta), "n_loci": len(loci),
        "mapped_probe_fraction": float(mapping["cpg_idx"].notna().mean()),
        "age_min": float(meta["age"].min()), "age_max": float(meta["age"].max()),
        "condition_class_counts": meta["condition_class"].value_counts().to_dict(),
    }
    (output / "qc.json").write_text(json.dumps(qc, indent=2, sort_keys=True))
    manifest = {
        "schema_version": 1, "dataset": "ComputAgeBench", "source_repository": "computage/computage_bench",
        "license": "CC-BY-SA-4.0", "split": args.split, "snapshot_dir": str(snapshot),
        "probe_crosswalk": str(args.probe_crosswalk.resolve()), "reference_build": REFERENCE_BUILD,
        "coordinate_convention": COORDINATE_CONVENTION, "cpg_namespace": CPG_NAMESPACE,
        "studies": [study for study, _ in studies], "artifacts": {path.name: str(path) for path in targets}, "qc": qc,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(qc, indent=2, sort_keys=True))
    print(f"WROTE={output}")


if __name__ == "__main__":
    main()
