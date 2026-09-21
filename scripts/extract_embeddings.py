#!/usr/bin/env python3
"""Extract frozen patient embeddings from a fitted sparse-reconstruction checkpoint.

This is the entry point for goals #1/#2 in docs/EMBEDDING_EVALUATION.md: instead of
reconstructing a full methylome and evaluating a downstream predictor on the reconstructed
beta values, we run the trained encoder (`encode_patient`) once per downstream-task patient
using every finite CpG that patient has in common with the representation's universe, and
persist the resulting patient embedding for `run_embedding_probe.py` to consume.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from cpg_repr_benchmark.data.legacy import legacy_ids_to_coordinate_ids
from cpg_repr_benchmark.data.methylation import read_axis
from cpg_repr_benchmark.data.splits import patient_disjoint_split
from cpg_repr_benchmark.embedding.extract import extract_patient_embeddings
from cpg_repr_benchmark.embedding.store import save_patient_embeddings
from cpg_repr_benchmark.models.model import build_reconstructor
from cpg_repr_benchmark.representations.hdf5_store import HDF5RepresentationStore, inspect_representation_h5
from cpg_repr_benchmark.representations.resolver import resolve_representation
from cpg_repr_benchmark.training.priors import beta_to_logit, compute_leakage_safe_priors


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


class _PatientRowLoader:
    """Minimal in-memory loader yielding one dict-batch per patient chunk.

    Deliberately not `torch.utils.data.DataLoader`-backed: downstream-task cohorts here are
    small (hundreds of patients) compared to sparse-reconstruction training, so building the
    per-patient observed set eagerly in one pass keeps this script simple and dependency-free.
    """

    def __init__(
        self,
        *,
        beta: np.ndarray,
        embedding: np.ndarray,
        prior_logit: np.ndarray,
        patient_ids: list[str],
        rows: np.ndarray,
        beta_epsilon: float,
        batch_size: int,
        max_observed: int,
        seed: int,
    ):
        self.beta = beta
        self.embedding = embedding
        self.prior_logit = prior_logit
        self.patient_ids = patient_ids
        self.rows = rows
        self.beta_epsilon = beta_epsilon
        self.batch_size = batch_size
        self.max_observed = max_observed
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for start in range(0, len(self.rows), self.batch_size):
            chunk_rows = self.rows[start : start + self.batch_size]
            max_n = 0
            per_patient = []
            for row in chunk_rows:
                finite = np.flatnonzero(np.isfinite(self.beta[row]))
                if len(finite) > self.max_observed:
                    finite = self.rng.choice(finite, size=self.max_observed, replace=False)
                per_patient.append(finite)
                max_n = max(max_n, len(finite))
            if max_n == 0:
                continue
            locus_dim = self.embedding.shape[1]
            observed_locus = np.zeros((len(chunk_rows), max_n, locus_dim), dtype=np.float32)
            observed_residual = np.zeros((len(chunk_rows), max_n), dtype=np.float32)
            observed_valid = np.zeros((len(chunk_rows), max_n), dtype=bool)
            for i, (row, columns) in enumerate(zip(chunk_rows, per_patient)):
                n = len(columns)
                observed_locus[i, :n] = self.embedding[columns]
                logit = beta_to_logit(self.beta[row, columns], self.beta_epsilon)
                observed_residual[i, :n] = logit - self.prior_logit[columns]
                observed_valid[i, :n] = True
            yield {
                "observed_locus": torch.from_numpy(observed_locus),
                "observed_residual": torch.from_numpy(observed_residual),
                "observed_valid": torch.from_numpy(observed_valid),
                "patient_id": [self.patient_ids[r] for r in chunk_rows],
            }


def main() -> None:
    p = argparse.ArgumentParser(description="Extract frozen patient embeddings for a downstream-task cohort")
    p.add_argument("--reconstruction-run", type=Path, required=True, help="sparse_reconstruction run dir with checkpoints/best.pt")
    p.add_argument("--methylation-h5", type=Path, required=True, help="downstream-task methylation matrix")
    p.add_argument("--output", type=Path, required=True, help="output .npz path for the embedding store")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--patient-split", default="0.8,0.1,0.1")
    p.add_argument("--max-observed", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pooling", choices=("mean", "flatten"), default="mean")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--cpg-registry",
        type=Path,
        default=None,
        help="legacy cpg_idx -> chr,pos registry; required when the representation store used by "
        "the reconstruction run holds legacy (non-coordinate-namespace) CpG IDs, which does not "
        "match a downstream dataset's coordinate-native methylation matrix axis",
    )
    args = p.parse_args()

    root = _repo_root()
    run_dir = args.reconstruction_run.resolve()
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())

    matrix_ids, sample_names = read_axis(args.methylation_h5)
    representation = resolve_representation(
        cfg["representation"], registry_path=Path(cfg["dataset"].get("cpg_registry", ".")), run_dir=run_dir, repo_root=root
    )
    if args.cpg_registry is not None:
        store_ids, store_dim, id_key, embedding_key = inspect_representation_h5(
            representation.store_path, id_key=representation.id_key, embedding_key=representation.embedding_key
        )
        store_ids = legacy_ids_to_coordinate_ids(store_ids, args.cpg_registry)
        order = np.argsort(store_ids)
        sorted_ids = store_ids[order]
        pos = np.searchsorted(sorted_ids, matrix_ids)
        present = pos < len(sorted_ids)
        present[present] &= sorted_ids[pos[present]] == matrix_ids[present]
        covered_columns = np.flatnonzero(present)
        if len(covered_columns) < 2:
            raise RuntimeError("representation has insufficient coverage of the downstream methylation matrix")
        store_rows = order[pos[covered_columns]]
        import h5py

        with h5py.File(representation.store_path, "r") as handle:
            read_order = np.argsort(store_rows)
            sorted_values = np.asarray(handle[embedding_key][store_rows[read_order]], dtype=np.float32)
        inverse = np.empty_like(read_order)
        inverse[read_order] = np.arange(len(read_order))
        embedding = sorted_values[inverse]
        locus_dim = store_dim
    else:
        store = HDF5RepresentationStore(
            representation.store_path,
            matrix_ids,
            id_key=representation.id_key,
            embedding_key=representation.embedding_key,
        )
        covered_columns = np.flatnonzero(store.coverage_mask)
        if len(covered_columns) < 2:
            raise RuntimeError("representation has insufficient coverage of the downstream methylation matrix")
        embedding = store.get_by_matrix_columns(covered_columns)
        locus_dim = store.dim

    splits = patient_disjoint_split(sample_names, args.seed, tuple(float(x) for x in args.patient_split.split(",")))
    beta_epsilon = float(cfg["training"].get("beta_epsilon", 1e-4))
    prior_logit, usable, _ = compute_leakage_safe_priors(
        args.methylation_h5, splits["train"], covered_columns, len(matrix_ids), epsilon=beta_epsilon
    )
    if not usable.all():
        raise RuntimeError("some covered loci have no finite beta among downstream train patients")

    import h5py

    with h5py.File(args.methylation_h5, "r") as handle:
        beta = np.asarray(handle["beta"][:, covered_columns], dtype=np.float32)

    model_cfg = cfg["model"]
    model = build_reconstructor(locus_dim, model_cfg)
    state = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state))

    device = torch.device(args.device)
    all_rows = np.arange(len(sample_names))
    loader = _PatientRowLoader(
        beta=beta,
        embedding=embedding,
        prior_logit=prior_logit,
        patient_ids=list(sample_names),
        rows=all_rows,
        beta_epsilon=beta_epsilon,
        batch_size=args.batch_size,
        max_observed=args.max_observed,
        seed=args.seed,
    )
    patient_ids, embeddings = extract_patient_embeddings(model, loader, device=device, pooling=args.pooling)

    save_patient_embeddings(
        args.output,
        patient_ids=patient_ids,
        embedding=embeddings.numpy(),
        meta={
            "source_type": "patient_embedding",
            "reconstruction_run": str(run_dir),
            "representation": str(cfg["representation"]["name"]),
            "track": str(cfg["representation"].get("track", "native_frozen")),
            "architecture": str(model_cfg.get("architecture", "deepsets")),
            "pooling": args.pooling,
            "patient_split": {k: v.tolist() for k, v in splits.items()},
        },
    )
    print(f"saved {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]} to {args.output}")


if __name__ == "__main__":
    main()
