from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_patient_embeddings(
    path: Path,
    *,
    patient_ids: list[str],
    embedding: np.ndarray,
    meta: dict[str, Any] | None = None,
) -> None:
    """Persist a patient embedding matrix under the fixed contract used by every probing script.

    Contract: `patient_ids` (str array, length N), `embedding` (float32, shape [N, D]). A
    sibling `<path>.meta.json` records provenance (source type, checkpoint/store, dim,
    architecture) so a linear probe run can be traced back without re-reading the model config.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.ndim != 2 or embedding.shape[0] != len(patient_ids):
        raise ValueError("embedding must be a 2D array with one row per patient_id")
    np.savez_compressed(
        path,
        patient_ids=np.asarray(patient_ids, dtype=object),
        embedding=embedding,
    )
    meta_path = path.with_suffix(path.suffix + ".meta.json") if path.suffix else Path(str(path) + ".meta.json")
    payload = {"n_patients": int(embedding.shape[0]), "dim": int(embedding.shape[1]), **(meta or {})}
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_patient_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        patient_ids = [str(pid) for pid in data["patient_ids"]]
        embedding = np.asarray(data["embedding"], dtype=np.float32)
    return patient_ids, embedding
