#!/usr/bin/env python3
"""Step A/B of the reduced-observation reconstruction->downstream diagnostic.

Builds the canonical CpG universe U = GSE40279 CpGs ∩ functional-representation coverage
∩ NTv3-pre-representation coverage (both are already GSE40279-materialized stores over the
reconstruction model's supported chromosome scope), then a fixed seed=17 random ordering of
U and nested observed-CpG prefixes {256, 1024, 4096}. Writes one npz + one json manifest
under data/protocols/reduced_observation/. No test-patient data is touched anywhere here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 17
OBSERVED_COUNTS = [256, 1024, 4096]

METHYLATION_H5 = REPO_ROOT / "data/processed/GSE40279/methylation.h5"
FUNCTIONAL_STORE = REPO_ROOT / "data/cache/representations/materialized/GSE40279/shared_chr1/functional_legacy.h5"
NTV3_STORE = REPO_ROOT / "data/cache/representations/materialized/GSE40279/shared_chr1/ntv3_pre.h5"
OUT_DIR = REPO_ROOT / "data/protocols/reduced_observation"


def _cpg_ids(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(f["cpg_idx"][:], dtype=np.int64)


def main() -> None:
    meth_ids = _cpg_ids(METHYLATION_H5)
    func_ids = _cpg_ids(FUNCTIONAL_STORE)
    ntv3_ids = _cpg_ids(NTV3_STORE)

    u = np.intersect1d(np.intersect1d(meth_ids, func_ids), ntv3_ids)
    u = np.sort(u).astype(np.int64)
    if len(u) == 0:
        raise RuntimeError("universe U is empty; check representation coverage")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(u))
    ordered_u = u[order]

    counts = [c for c in OBSERVED_COUNTS if c <= len(u)]
    if len(counts) < len(OBSERVED_COUNTS):
        dropped = [c for c in OBSERVED_COUNTS if c > len(u)]
        print(f"WARNING: |U|={len(u)} < requested counts {dropped}; dropping those.")
    if not counts:
        raise RuntimeError(f"|U|={len(u)} is smaller than the smallest requested observed count")

    subsets = {str(c): ordered_u[:c] for c in counts}

    u_bytes = u.tobytes()
    u_hash = hashlib.sha256(u_bytes).hexdigest()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUT_DIR / "gse40279_age_universe_seed17.npz"
    np.savez_compressed(
        npz_path,
        universe_cpg_idx=u,
        ordered_universe_cpg_idx=ordered_u,
        **{f"observed_{c}_cpg_idx": subsets[str(c)] for c in counts},
    )
    manifest = {
        "seed": SEED,
        "universe_size": len(u),
        "universe_sha256": u_hash,
        "sources": {
            "methylation_h5": str(METHYLATION_H5),
            "functional_store": str(FUNCTIONAL_STORE),
            "ntv3_pre_store": str(NTV3_STORE),
        },
        "requested_observed_counts": OBSERVED_COUNTS,
        "used_observed_counts": counts,
        "npz_path": str(npz_path),
    }
    json_path = OUT_DIR / "gse40279_age_universe_seed17.json"
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
