#!/usr/bin/env python3
"""Fit and evaluate a frozen linear probe on patient embeddings for a downstream task.

Consumes the `.npz` written by `extract_embeddings.py` (our model) or any embedding store
following the same `patient_ids` + `embedding` contract (e.g. a colleague's methylation FM
embedding, once they export it in that format — see docs/EMBEDDING_EVALUATION.md). Writes a
uniform `summary.json` via `experiments.result_schema` under the standard content-addressed
`outputs/embedding_probe/<dataset>/<representation>/<track>/seed_<seed>/<run>/` layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cpg_repr_benchmark.data.splits import patient_disjoint_split
from cpg_repr_benchmark.embedding.store import load_patient_embeddings
from cpg_repr_benchmark.experiments.result_schema import embedding_summary_payload
from cpg_repr_benchmark.probing.linear_probe import fit_linear_probe


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Linear probe a patient embedding store against a downstream phenotype")
    p.add_argument("--embedding", type=Path, required=True, help=".npz written by extract_embeddings.py")
    p.add_argument("--dataset", required=True, help="dataset name used in the output path, e.g. gse40279_age")
    p.add_argument("--representation", required=True)
    p.add_argument("--track", default="native_frozen")
    p.add_argument("--phenotypes-parquet", type=Path, required=True)
    p.add_argument(
        "--phenotype-id-column",
        default="sample_name",
        help="phenotype-table column to join embedded patient_ids against; some datasets' "
        "methylation-matrix axis is the array/beadchip id rather than the GEO sample_name "
        "(e.g. GSE147221), so this may need to be set to that column instead",
    )
    p.add_argument("--target-column", required=True)
    p.add_argument("--task-type", choices=("regression", "classification"), required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--patient-split", default="0.8,0.1,0.1")
    p.add_argument("--output-root", type=Path, default=None)
    args = p.parse_args()

    root = _repo_root()
    output_root = args.output_root or (root / "outputs")

    patient_ids, embedding = load_patient_embeddings(args.embedding)
    meta_path = args.embedding.with_suffix(args.embedding.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    phenotypes = pd.read_parquet(args.phenotypes_parquet).set_index(args.phenotype_id_column)
    missing = [pid for pid in patient_ids if pid not in phenotypes.index]
    if missing:
        raise ValueError(f"phenotype table missing {len(missing)} embedded patients")
    raw_target = phenotypes.loc[patient_ids, args.target_column].to_numpy(dtype="float64")
    finite = np.isfinite(raw_target)
    if not finite.all():
        dropped = int((~finite).sum())
        print(f"dropping {dropped} patients with missing {args.target_column!r}")
        patient_ids = [pid for pid, keep in zip(patient_ids, finite) if keep]
        embedding = embedding[finite]
        raw_target = raw_target[finite]
    target = raw_target

    splits = patient_disjoint_split(patient_ids, args.seed, tuple(float(x) for x in args.patient_split.split(",")))
    train_rows = np.concatenate([splits["train"], splits["validation"]])
    result = fit_linear_probe(
        embedding=embedding,
        target=target,
        train_rows=train_rows,
        test_rows=splits["test"],
        task_type=args.task_type,
    )

    run_dir = (
        output_root
        / "embedding_probe"
        / args.dataset
        / args.representation
        / args.track
        / f"seed_{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    result.save(run_dir / "linear_probe.joblib")

    payload = embedding_summary_payload(
        task="embedding_linear_probe",
        dataset=args.dataset,
        representation=args.representation,
        track=args.track,
        mode="frozen_pretrained",
        embedding_source_type=str(meta.get("source_type", "patient_embedding")),
        embedding_dim=int(embedding.shape[1]),
        metrics=result.metrics,
        n_patients={"train": len(train_rows), "test": len(splits["test"])},
        checkpoint=meta.get("reconstruction_run"),
        extra={"alpha": result.alpha, "embedding_meta": meta},
    )
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
