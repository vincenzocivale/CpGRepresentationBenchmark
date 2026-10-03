"""Verify (or, with --write, create) the frozen-candidate manifest of `regulatory_histone_dnase_v1`.

Read-only on every data artifact. Recomputes: track-catalog sha256, feature-set hash (Step-4 resolver on the current
catalog), compressor-manifest hash (sha256 of the canonical JSON of the compressor manifest), compressor/store file
sha256, store shape/dtype/finiteness, fit-loci hash (Step-5 protocol) and a 200-locus re-transform through the saved
compressor compared to the stored embedding. Exit code 1 on any mismatch.

    python scripts/verify_frozen_candidate.py [--skip-store-sha] [--manifest configs/frozen/regulatory_histone_dnase_v1.json]
    python scripts/verify_frozen_candidate.py --write      # (re)create the manifest; only for the initial freeze

Compressor-manifest hash definition: sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'),
allow_nan=False).encode()) where `manifest` is the JSON stored in the compressor .npz (identical to the .compressor.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NAME = "regulatory_histone_dnase"
VERSION = "regulatory_histone_dnase_v1"
STEM = "regulatory_histone_dnase__global_svd256__discovery_chr1_19"
REPS = Path("data/cache/representations")
MANIFEST = ROOT / "configs/frozen/regulatory_histone_dnase_v1.json"
SAMPLE_N, SAMPLE_SEED, SAMPLE_TOL = 200, 17, 1e-5
FILES = {"store_h5": REPS / f"{STEM}.h5", "store_manifest_json": REPS / f"{STEM}.h5.json",
         "compressor_npz": REPS / f"{STEM}.compressor.npz", "compressor_json": REPS / f"{STEM}.compressor.json"}
STORE_FIELDS = ("path", "n_loci", "n_tracks", "cpg_idx_sha256")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def data_present() -> bool:
    return all((ROOT / p).exists() for p in FILES.values())


def collect(skip_store_sha: bool) -> dict:
    """Recompute every frozen fact from the artifacts on disk (no regeneration)."""
    import h5py

    from cpg_repr_benchmark.encode_atlas.compression import RegulatoryCompressor, TrackStore, split_loci
    from cpg_repr_benchmark.encode_atlas.feature_sets import resolve_feature_set

    f = {k: ROOT / v for k, v in FILES.items()}
    comp_json = json.loads(f["compressor_json"].read_text())
    store_json = json.loads(f["store_manifest_json"].read_text())
    with np.load(f["compressor_npz"], allow_pickle=False) as z:
        npz_manifest = json.loads(str(z["manifest_json"]))
    catalog_rel = comp_json["catalog"]["path"]
    fs = resolve_feature_set(NAME, ROOT / catalog_rel, check_expected=True)
    with h5py.File(f["store_h5"], "r") as h:
        cpg = h["cpg_idx"][:]
        emb_ds = h["embedding"]
        shape, dtype, cpg_dtype = tuple(emb_ds.shape), str(emb_ds.dtype), str(cpg.dtype)
        finite = all(bool(np.isfinite(emb_ds[i:i + 50000]).all()) for i in range(0, shape[0], 50000))
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in h.attrs.items()}
    # fit loci (Step-5 protocol) and 200-locus re-transform through the saved compressor
    src = comp_json["source_store"]
    store = TrackStore(ROOT / src["path"])
    registry = ROOT / "data/cpg/registries/array_cpg_map.parquet"
    fit, _heldout = split_loci(attrs["fit_protocol"], store.cpg_idx, registry)
    comp = RegulatoryCompressor.load(f["compressor_npz"], fs)
    rng = np.random.default_rng(SAMPLE_SEED)
    pick = np.sort(rng.choice(len(cpg), SAMPLE_N, replace=False))
    with h5py.File(f["store_h5"], "r") as h:
        stored = h["embedding"][pick]
    again = comp.transform(store, cpg[pick], fs)
    out = {
        "name": NAME, "version": VERSION,
        "catalog_sha256": sha256_file(ROOT / catalog_rel), "catalog_path": catalog_rel,
        "feature_set_hash_recomputed": fs.manifest_hash, "feature_set_hash_recorded": comp_json["feature_set"]["manifest_hash"],
        "feature_columns_sha256": comp_json["feature_set"]["feature_columns_sha256"],
        "n_features": fs.n_selected, "counts_per_assay": dict(fs.counts_per_assay),
        "counts_per_block": dict(fs.counts_per_block),
        "fit_loci_recomputed": {"protocol": fit.name, "n": int(fit.n), "sha256": fit.sha256},
        "fit_loci_recorded": comp_json["fit_loci"], "store_attrs_fit_loci_sha256": attrs.get("fit_loci_sha256"),
        "compressor_manifest_hash_npz": canonical_hash(npz_manifest), "compressor_manifest_hash_json": canonical_hash(comp_json),
        "compressor_json_matches_store_json_core": all(comp_json[k] == store_json[k] for k in
                                                       ("feature_set", "catalog", "fit_loci", "method", "params", "seed")),
        "compressor_sha256": {"npz": sha256_file(f["compressor_npz"]), "json": sha256_file(f["compressor_json"])},
        "store_manifest_json_sha256": sha256_file(f["store_manifest_json"]),
        "store_sha256": None if skip_store_sha else sha256_file(f["store_h5"]),
        "embedding": {"shape": list(shape), "dtype": dtype, "cpg_idx_dtype": cpg_dtype, "all_finite": finite,
                      "cpg_idx_unique": bool(np.unique(cpg).size == cpg.size),
                      "cpg_idx_equals_source_store": bool(np.array_equal(cpg, store.cpg_idx))},
        "sample_retransform": {"n": SAMPLE_N, "rng_seed": SAMPLE_SEED, "max_abs_diff": float(np.abs(again - stored).max())},
        "source_store_identity": {k: src[k] for k in STORE_FIELDS},
        "method": comp_json["method"], "params": comp_json["params"], "seed": comp_json["seed"],
        "libraries_recorded": comp_json["libraries"], "attrs": {k: str(v) for k, v in attrs.items()},
    }
    return out


def build_manifest(c: dict) -> dict:
    return {
        "name": NAME, "version": VERSION, "status": "candidate_frozen",
        "frozen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "patient_independent": True, "beta_values_used": False, "dense_columns": 0,
        "feature_set": {"name": NAME, "manifest_hash": c["feature_set_hash_recomputed"],
                        "feature_columns_sha256": c["feature_columns_sha256"], "n_tracks": c["n_features"],
                        "tracks_per_assay": c["counts_per_assay"], "tracks_per_block": c["counts_per_block"]},
        "catalog": {"path": c["catalog_path"], "sha256": c["catalog_sha256"]},
        "fit_loci": c["fit_loci_recomputed"],
        "method": c["method"], "params": c["params"], "seed": c["seed"],
        "embedding": {"shape": c["embedding"]["shape"], "dtype": c["embedding"]["dtype"],
                      "cpg_idx_dtype": c["embedding"]["cpg_idx_dtype"], "note": "rows of chr20-22 are out-of-fit (transformed only)"},
        "compressor_manifest_hash": c["compressor_manifest_hash_npz"],
        "compressor_manifest_hash_definition": "sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'), "
                                               "allow_nan=False)); manifest = JSON inside the compressor .npz (== .compressor.json)",
        "file_sha256": {"store_h5": c["store_sha256"], "store_manifest_json": c["store_manifest_json_sha256"],
                        "compressor_npz": c["compressor_sha256"]["npz"], "compressor_json": c["compressor_sha256"]["json"]},
        "path_convention": {"store_h5": str(FILES["store_h5"]), "store_manifest_json": str(FILES["store_manifest_json"]),
                            "compressor_npz": str(FILES["compressor_npz"]), "compressor_json": str(FILES["compressor_json"]),
                            "pattern": "data/cache/representations/<feature_set>__<method><dim>__<fit_protocol>.{h5,h5.json,compressor.npz,compressor.json}"},
        "source_store": c["source_store_identity"],
        "libraries_recorded_at_build": c["libraries_recorded"],
        "code": {"commit_at_build_recorded": False,
                 "freeze_head_commit": git("rev-parse", "HEAD"),
                 "compression_code_last_commit": git("log", "-1", "--format=%H", "--", "src/cpg_repr_benchmark/encode_atlas/compression.py",
                                                     "src/cpg_repr_benchmark/encode_atlas/feature_sets.py"),
                 "build_command": "python scripts/build_regulatory_embeddings.py --config configs/regulatory_embeddings.yaml "
                                  "--only regulatory_histone_dnase__global_svd256__discovery_chr1_19"},
        "verification": {"sample_retransform_n": SAMPLE_N, "sample_rng_seed": SAMPLE_SEED,
                         "sample_max_abs_diff_at_freeze": c["sample_retransform"]["max_abs_diff"],
                         "sample_tolerance": SAMPLE_TOL},
    }


def check(c: dict, m: dict, skip_store_sha: bool) -> list[str]:
    bad: list[str] = []

    def eq(label, a, b):
        if a != b:
            bad.append(f"{label}: {a!r} != {b!r}")

    eq("feature-set hash (resolver vs compressor record)", c["feature_set_hash_recomputed"], c["feature_set_hash_recorded"])
    eq("feature-set hash vs manifest", c["feature_set_hash_recomputed"], m["feature_set"]["manifest_hash"])
    eq("feature columns sha256", c["feature_columns_sha256"], m["feature_set"]["feature_columns_sha256"])
    eq("n tracks", c["n_features"], m["feature_set"]["n_tracks"])
    eq("tracks per assay", c["counts_per_assay"], m["feature_set"]["tracks_per_assay"])
    eq("catalog sha256", c["catalog_sha256"], m["catalog"]["sha256"])
    eq("fit loci (recomputed vs recorded)", c["fit_loci_recomputed"], c["fit_loci_recorded"])
    eq("fit loci vs manifest", c["fit_loci_recomputed"], m["fit_loci"])
    eq("store attr fit-loci hash", c["store_attrs_fit_loci_sha256"], m["fit_loci"]["sha256"])
    eq("compressor manifest hash (npz)", c["compressor_manifest_hash_npz"], m["compressor_manifest_hash"])
    eq("compressor manifest hash (json == npz)", c["compressor_manifest_hash_json"], c["compressor_manifest_hash_npz"])
    eq("compressor json vs store json", c["compressor_json_matches_store_json_core"], True)
    eq("compressor npz sha256", c["compressor_sha256"]["npz"], m["file_sha256"]["compressor_npz"])
    eq("compressor json sha256", c["compressor_sha256"]["json"], m["file_sha256"]["compressor_json"])
    eq("store manifest json sha256", c["store_manifest_json_sha256"], m["file_sha256"]["store_manifest_json"])
    if not skip_store_sha:
        eq("store h5 sha256", c["store_sha256"], m["file_sha256"]["store_h5"])
    e = c["embedding"]
    eq("embedding shape", e["shape"], m["embedding"]["shape"])
    eq("embedding dtype", e["dtype"], m["embedding"]["dtype"])
    eq("cpg_idx dtype", e["cpg_idx_dtype"], "int64")
    eq("embedding finite", e["all_finite"], True)
    eq("cpg_idx unique", e["cpg_idx_unique"], True)
    eq("cpg_idx == source store axis", e["cpg_idx_equals_source_store"], True)
    eq("seed", c["seed"], m["seed"])
    eq("method/params", [c["method"], c["params"]], [m["method"], m["params"]])
    eq("source store identity", c["source_store_identity"], m["source_store"])
    if not c["sample_retransform"]["max_abs_diff"] <= m["verification"]["sample_tolerance"]:
        bad.append(f"sample re-transform max abs diff {c['sample_retransform']['max_abs_diff']:.3e} > tolerance")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--skip-store-sha", action="store_true", help="skip sha256 of the 421 MB store (opt-in)")
    ap.add_argument("--write", action="store_true", help="create the manifest from the artifacts on disk")
    args = ap.parse_args()
    if not data_present():
        print("frozen artifacts not found on disk; nothing verified", file=sys.stderr)
        return 2
    c = collect(args.skip_store_sha and not args.write)
    if args.write:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(build_manifest(c), indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.manifest}")
    m = json.loads(args.manifest.read_text())
    bad = check(c, m, args.skip_store_sha)
    print(f"sample re-transform ({SAMPLE_N} loci) max abs diff = {c['sample_retransform']['max_abs_diff']:.3e}")
    print(f"feature-set hash {c['feature_set_hash_recomputed']}\ncatalog sha256   {c['catalog_sha256']}\n"
          f"fit-loci sha256  {c['fit_loci_recomputed']['sha256']} (n={c['fit_loci_recomputed']['n']})\n"
          f"compressor-manifest hash {c['compressor_manifest_hash_npz']}\nstore sha256     {c['store_sha256']}")
    if bad:
        print("MISMATCH:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    print("OK: frozen candidate verified" + (" (store sha256 skipped)" if args.skip_store_sha else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
