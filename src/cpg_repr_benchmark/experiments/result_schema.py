from __future__ import annotations

from typing import Any, Literal

EmbeddingSourceType = Literal["patient_embedding", "cpg_locus_embedding"]
ProbeMode = Literal["frozen_pretrained", "fine_tuned"]


def embedding_summary_payload(
    *,
    task: str,
    dataset: str,
    representation: str,
    track: str,
    mode: ProbeMode,
    embedding_source_type: EmbeddingSourceType,
    embedding_dim: int,
    metrics: dict[str, Any],
    n_patients: dict[str, int] | None = None,
    checkpoint: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the uniform `summary.json` payload shared by every embedding-centric experiment
    (`embedding_linear_probe`, `embedding_finetune`, `bio_validation`). Keeping a single
    schema function means every new script writes results that read the same way regardless
    of which representation or task produced them.
    """
    payload: dict[str, Any] = {
        "schema_version": 1,
        "task": task,
        "dataset": dataset,
        "representation": representation,
        "track": track,
        "mode": mode,
        "embedding_source": {
            "type": embedding_source_type,
            "dim": int(embedding_dim),
            "checkpoint": checkpoint,
        },
        "metrics": metrics,
    }
    if n_patients is not None:
        payload["n_patients"] = n_patients
    if extra:
        payload.update(extra)
    return payload
