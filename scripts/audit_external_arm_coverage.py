#!/usr/bin/env python3
"""Read-only arm coverage audit of the 4 frozen stores against the external GSE40279 locus universe.

No model, no forward pass, no evaluation: only sha256, /cpg_idx coverage, dim/dtype, all-zero rows and finite check on
the universe rows. Writes outputs/external_reconstruction_v1/audit/arm_coverage.{json,md} (new dir).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml

from cpg_repr_benchmark.external.io import sha256_file, write_json_new

REPO = Path(__file__).resolve().parents[1]
ARMS = ("regulatory_histone_dnase", "functional_annotations_pca", "cpgpt_large_locus", "deepcpg_dna_locus")


def audit_store(store: Path, universe_ids: np.ndarray, expected_sha: str, *, chunk: int = 50_000) -> dict:
    sha = sha256_file(store)
    with h5py.File(store, "r") as h:
        ids = np.asarray(h["cpg_idx"][:], dtype=np.int64)
        emb = h["embedding"]
        dim, dtype = int(emb.shape[1]), str(emb.dtype)
        order = np.argsort(ids)
        pos = np.searchsorted(ids[order], universe_ids)
        pos_c = np.clip(pos, 0, len(ids) - 1)
        covered = ids[order][pos_c] == universe_ids
        in_universe = np.zeros(len(ids), dtype=bool)
        in_universe[order[pos_c[covered]]] = True
        n_zero = n_nonfinite = n_rows = 0
        for r0 in range(0, len(ids), chunk):
            sel = in_universe[r0:r0 + chunk]
            if not sel.any():
                continue
            block = np.asarray(emb[r0:r0 + chunk])[sel].astype(np.float32)
            n_rows += len(block)
            n_zero += int((~block.any(axis=1)).sum())
            n_nonfinite += int((~np.isfinite(block)).sum())
    return {
        "store": str(store), "sha256": sha, "expected_sha256": expected_sha, "sha256_match": sha == expected_sha,
        "n_store_loci": len(ids), "n_universe": len(universe_ids), "n_universe_covered": int(covered.sum()),
        "coverage_fraction": float(covered.mean()), "full_coverage": bool(covered.all()),
        "missing_examples": universe_ids[~covered][:10].tolist(),
        "dim": dim, "dtype": dtype, "n_universe_rows_read": int(n_rows),
        "zero_rows_in_universe": int(n_zero), "nonfinite_values_in_universe_rows": int(n_nonfinite),
        "finite": bool(n_nonfinite == 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-yaml", type=Path, default=REPO / "configs/experiments/regulatory_confirmation_matrix/matrix.yaml")
    ap.add_argument("--locus-protocol", type=Path, default=REPO / "data/derived/external_gse40279_v1/loci_seen_external_gse40279_v1.npz")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs/external_reconstruction_v1/audit")
    args = ap.parse_args()
    spec = yaml.safe_load(args.matrix_yaml.read_text())["matrix"]["arms"]
    by_id = {a["id"]: a for a in spec}
    with np.load(args.locus_protocol) as z:
        universe = np.sort(np.concatenate([z["train_cpg_idx"], z["heldout_cpg_idx"]]).astype(np.int64))
    result = {"universe_size": len(universe), "locus_protocol": str(args.locus_protocol), "arms": {}}
    for arm in ARMS:
        a = by_id[arm]
        r = audit_store(REPO / a["store_h5"], universe, a["sha256"])
        r["expected_dim"] = a["dim"]
        r["dim_match"] = r["dim"] == a["dim"]
        result["arms"][arm] = r
    result["blockers"] = [f"{k}: {why}" for k, r in result["arms"].items()
                          for why, bad in (("sha256 mismatch", not r["sha256_match"]),
                                           ("incomplete universe coverage", not r["full_coverage"]),
                                           ("non-finite rows", not r["finite"]), ("dim mismatch", not r["dim_match"])) if bad]
    result["all_green"] = not result["blockers"]
    write_json_new(args.out_dir / "arm_coverage.json", result)
    lines = ["# External arm coverage audit (read-only; no model evaluated)", "",
             f"Universe: {result['universe_size']} loci. all_green: {result['all_green']}. blockers: {result['blockers'] or 'none'}", "",
             "| arm | sha256 match | coverage | dim | dtype | zero rows | non-finite |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, r in result["arms"].items():
        lines.append(f"| {k} | {r['sha256_match']} ({r['sha256'][:12]}) | {r['n_universe_covered']}/{r['n_universe']} | {r['dim']} | {r['dtype']} | "
                     f"{r['zero_rows_in_universe']} | {r['nonfinite_values_in_universe_rows']} |")
    (args.out_dir / "arm_coverage.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
