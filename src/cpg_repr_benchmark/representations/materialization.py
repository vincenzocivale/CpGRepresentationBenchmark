from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

import h5py
import numpy as np

from cpg_repr_benchmark.data.coordinates import (
    COORDINATE_CONVENTION,
    CPG_NAMESPACE,
    REFERENCE_BUILD,
    decode_cpg_id,
    normalize_chromosome,
)


class CoordinateProvider(Protocol):
    name: str
    dim: int

    def encode(
        self,
        cpg_idx: np.ndarray,
        chrom: np.ndarray,
        pos: np.ndarray,
    ) -> np.ndarray: ...

    def metadata(self) -> dict[str, str | int | float | bool]: ...

    def close(self) -> None: ...


def load_protocol_coordinates(
    protocol_path: Path,
    *,
    chromosomes: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    protocol_path = Path(protocol_path)
    data = np.load(protocol_path)
    if "cpg_idx" not in data:
        raise ValueError(f"{protocol_path} must contain cpg_idx")
    ids = np.asarray(data["cpg_idx"], dtype=np.int64)
    if len(ids) != len(np.unique(ids)):
        raise ValueError(f"{protocol_path} contains duplicate cpg_idx")

    decoded = [decode_cpg_id(int(value)) for value in ids]
    chrom = np.asarray([item[0] for item in decoded], dtype=object)
    pos = np.asarray([item[1] for item in decoded], dtype=np.int64)

    if chromosomes:
        requested = {normalize_chromosome(value) for value in chromosomes}
        keep = np.asarray([value in requested for value in chrom], dtype=bool)
        ids, chrom, pos = ids[keep], chrom[keep], pos[keep]
    if not len(ids):
        raise ValueError("locus protocol is empty after chromosome filtering")
    return ids, chrom, pos


def _fingerprint(
    cpg_idx: np.ndarray,
    provider_name: str,
    provider_metadata: dict,
    storage_dtype: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(cpg_idx, dtype=np.int64).tobytes())
    payload = {
        "provider": provider_name,
        "provider_metadata": provider_metadata,
        "storage_dtype": storage_dtype,
        "cpg_namespace": CPG_NAMESPACE,
        "reference_build": REFERENCE_BUILD,
        "coordinate_convention": COORDINATE_CONVENTION,
    }
    digest.update(json.dumps(payload, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _validate_existing(path: Path, cpg_idx: np.ndarray, fingerprint: str) -> dict | None:
    if not path.exists():
        return None
    with h5py.File(path, "r") as handle:
        required = {"cpg_idx", "chrom", "pos", "embedding"}
        if not required.issubset(handle):
            raise ValueError(f"existing representation {path} is not coordinate-native; keys={list(handle.keys())}")
        existing = np.asarray(handle["cpg_idx"][:], dtype=np.int64)
        if not np.array_equal(existing, cpg_idx):
            raise ValueError(
                f"existing representation {path} has a different locus protocol; use a different output path or --force"
            )
        existing_fp = str(handle.attrs.get("materialization_fingerprint", ""))
        if existing_fp and existing_fp != fingerprint:
            raise ValueError(
                f"existing representation {path} was built with a different provider configuration; use --force"
            )
        return {
            "status": "cached",
            "output": str(path),
            "n_loci": int(len(existing)),
            "embedding_dim": int(handle["embedding"].shape[1]),
        }


def materialize_coordinate_representation(
    provider: CoordinateProvider,
    cpg_idx: np.ndarray,
    chrom: np.ndarray,
    pos: np.ndarray,
    output_h5: Path,
    *,
    batch_size: int,
    storage_dtype: str = "float16",
    force: bool = False,
    resume: bool = True,
    extra_metadata: dict | None = None,
) -> dict:
    ids = np.asarray(cpg_idx, dtype=np.int64)
    chrom = np.asarray(chrom, dtype=object)
    pos = np.asarray(pos, dtype=np.int64)
    if not (len(ids) == len(chrom) == len(pos)):
        raise ValueError("cpg_idx/chrom/pos length mismatch")
    if len(ids) != len(np.unique(ids)):
        raise ValueError("materialization locus IDs must be unique")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1")
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError("storage_dtype must be float16 or float32")

    for value, c, p in zip(ids, chrom, pos):
        dc, dp = decode_cpg_id(int(value))
        if dc != normalize_chromosome(str(c)) or dp != int(p):
            raise ValueError(f"coordinate/ID disagreement for cpg_idx={int(value)}")

    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    provider_metadata = dict(provider.metadata())
    fp = _fingerprint(ids, provider.name, provider_metadata, storage_dtype)

    if output_h5.exists():
        if force:
            output_h5.unlink()
        else:
            cached = _validate_existing(output_h5, ids, fp)
            if cached is not None:
                return cached

    partial = output_h5.with_suffix(output_h5.suffix + ".partial")
    if partial.exists() and not resume:
        partial.unlink()
    if force and partial.exists():
        partial.unlink()

    dtype = np.float16 if storage_dtype == "float16" else np.float32
    if not partial.exists():
        with h5py.File(partial, "w") as handle:
            handle.create_dataset("cpg_idx", data=ids, dtype="i8")
            handle.create_dataset("chrom", data=np.asarray(chrom, dtype="S5"))
            handle.create_dataset("pos", data=pos, dtype="i8")
            handle.create_dataset(
                "embedding",
                shape=(len(ids), int(provider.dim)),
                dtype=dtype,
                chunks=(max(1, min(512, len(ids))), int(provider.dim)),
            )
            handle.create_dataset("completed", data=np.zeros(len(ids), dtype=np.bool_))
            handle.attrs["materialization_fingerprint"] = fp
            handle.attrs["provider"] = str(provider.name)
            handle.attrs["embedding_dim"] = int(provider.dim)
            handle.attrs["storage_dtype"] = storage_dtype
            handle.attrs["cpg_namespace"] = CPG_NAMESPACE
            handle.attrs["reference_build"] = REFERENCE_BUILD
            handle.attrs["coordinate_convention"] = COORDINATE_CONVENTION
            for key, value in provider_metadata.items():
                handle.attrs[f"provider_{key}"] = str(value)
            for key, value in (extra_metadata or {}).items():
                handle.attrs[str(key)] = str(value)
    else:
        with h5py.File(partial, "r") as handle:
            existing = np.asarray(handle["cpg_idx"][:], dtype=np.int64)
            if not np.array_equal(existing, ids):
                raise ValueError(f"partial materialization {partial} has a different locus protocol")
            if str(handle.attrs.get("materialization_fingerprint", "")) != fp:
                raise ValueError(f"partial materialization {partial} has a different provider configuration")
            if handle["embedding"].shape != (len(ids), int(provider.dim)):
                raise ValueError("partial materialization embedding shape mismatch")

    with h5py.File(partial, "r+") as handle:
        completed = np.asarray(handle["completed"][:], dtype=bool)
        n_total = len(ids)
        for start in range(0, n_total, int(batch_size)):
            stop = min(n_total, start + int(batch_size))
            local_pending = np.flatnonzero(~completed[start:stop])
            if not len(local_pending):
                continue
            rows = start + local_pending
            values = np.asarray(provider.encode(ids[rows], chrom[rows], pos[rows]), dtype=np.float32)
            expected = (len(rows), int(provider.dim))
            if values.shape != expected:
                raise ValueError(f"provider returned {values.shape}; expected {expected}")
            if not np.isfinite(values).all():
                raise ValueError("provider produced non-finite embeddings")
            handle["embedding"][rows] = values.astype(dtype, copy=False)
            handle["completed"][rows] = True
            completed[rows] = True
            handle.flush()
            done = int(completed.sum())
            print(f"[{provider.name}] {done}/{n_total} loci ({done / n_total:.2%})", flush=True)

        if not completed.all():
            raise RuntimeError(f"materialization stopped with {int((~completed).sum())} incomplete loci")
        del handle["completed"]
        handle.attrs["status"] = "complete"

    os.replace(partial, output_h5)
    sidecar = {
        "schema_version": 1,
        "status": "built",
        "provider": provider.name,
        "provider_metadata": provider_metadata,
        "output": str(output_h5),
        "n_loci": int(len(ids)),
        "embedding_dim": int(provider.dim),
        "storage_dtype": storage_dtype,
        "cpg_namespace": CPG_NAMESPACE,
        "reference_build": REFERENCE_BUILD,
        "coordinate_convention": COORDINATE_CONVENTION,
        "materialization_fingerprint": fp,
        "metadata": extra_metadata or {},
    }
    output_h5.with_suffix(output_h5.suffix + ".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    return sidecar
