"""READ-ONLY pre-execution audit of biological_validation_v2 (phase 1). Separate from the freeze code.

Never writes under data/derived/bioval_v2, configs/ or src/; never computes an endpoint metric. Subcommands:

  verify   sha256 of every file in MANIFEST_checksums.json and of every hash in the protocol doc section 7,
           against the files on disk (+ row counts of the frozen pair/CpG lists).
  stores   per representation store: sha256 vs matrix.yaml, dim/dtype, cpg_idx coverage of every frozen
           endpoint CpG set (counts only, never an intersection), chunked finite/all-zero-row scan.

    nice -n 10 python scripts/bioval_v2/preexec_audit.py verify --out /path/outside/frozen/verify.json
    nice -n 10 python scripts/bioval_v2/preexec_audit.py stores --out /path/outside/frozen/stores.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
BV = Path("data/derived/bioval_v2")
PROTECTED = (ROOT / "data/derived/bioval_v2", ROOT / "configs", ROOT / "src")


def sha256_file(path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def verify_checksum_manifest(root: Path, manifest: dict) -> dict:
    """Compare {rel: {sha256, bytes}} with files under ``root``. Returns ok/missing/mismatch lists."""
    res = {"n": 0, "ok": 0, "missing": [], "size_mismatch": [], "sha_mismatch": []}
    for rel, rec in sorted(manifest["files"].items()):
        res["n"] += 1
        p = root / rel
        if not p.is_file():
            res["missing"].append(rel)
            continue
        if p.stat().st_size != rec["bytes"]:
            res["size_mismatch"].append(rel)
            continue
        if sha256_file(p) != rec["sha256"]:
            res["sha_mismatch"].append(rel)
            continue
        res["ok"] += 1
    res["extra_on_disk"] = sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*")
        if p.is_file() and p.name != "MANIFEST_checksums.json" and not p.name.endswith(".tmp")
        and p.relative_to(root).as_posix() not in manifest["files"])
    return res


_ROW = re.compile(r"^\|\s*`([^`]+)`[^|]*\|\s*`([0-9a-f]{64})`\s*\|\s*$")


def parse_doc_hashes(doc_text: str) -> dict:
    """Section-7 tables of the protocol doc -> {relative path: sha256}. '...'-abbreviated names are expanded."""
    out = {}
    for line in doc_text.splitlines():
        m = _ROW.match(line.strip())
        if m:
            path, sha = m.groups()
            if "..._" in path:
                path = path.replace("..._", "frozen_enhancer_membership_")
            out[path] = sha
    return out


def coverage(store_ids: np.ndarray, needed: np.ndarray) -> dict:
    """Counts only: how many of ``needed`` (unique) cpg ids are present in the store index."""
    needed = np.unique(np.asarray(needed, dtype=np.int64))
    present = np.isin(needed, np.asarray(store_ids, dtype=np.int64))
    return {"n_needed": len(needed), "n_present": int(present.sum()), "n_missing": int((~present).sum()),
            "frac": float(present.mean()) if len(needed) else float("nan"),
            "missing_examples": needed[~present][:5].tolist()}


def scan_finite(ds, chunk_rows=50_000) -> dict:
    """Chunked full pass over a 2-D dataset (raw dtype, then float32): NaN/Inf/all-zero-row counts."""
    n, d = ds.shape
    nan = inf = zero_rows = nonfinite_rows = 0
    amax = 0.0
    for s in range(0, n, chunk_rows):
        a = np.asarray(ds[s:s + chunk_rows])
        f = a.astype(np.float32)
        isn, isi = np.isnan(f), np.isinf(f)
        nan += int(isn.sum())
        inf += int(isi.sum())
        nonfinite_rows += int((isn | isi).any(axis=1).sum())
        zero_rows += int((a == 0).all(axis=1).sum())
        fin = f[np.isfinite(f)]
        if fin.size:
            amax = max(amax, float(np.abs(fin).max()))
    return {"shape": [int(n), int(d)], "dtype": str(ds.dtype), "n_nan": nan, "n_inf": inf,
            "n_rows_nonfinite": nonfinite_rows, "n_all_zero_rows": zero_rows, "max_abs_finite": amax}


def _guard_out(out: Path | None) -> None:
    if out is None:
        return
    o = out.resolve()
    for p in PROTECTED:
        if o == p or p in o.parents:
            raise SystemExit(f"refusing to write inside protected path {p}")


def frozen_cpg_sets() -> dict:
    """endpoint -> array of cpg ids needed (read from frozen lists only)."""
    import pandas as pd
    rd = lambda rel, cols: pd.read_parquet(ROOT / BV / rel, columns=cols)
    s = {}
    d = rd("external_methylation_programs/frozen_pairs_seed17.parquet", ["cpg_i", "cpg_j"])
    s["loyfer_pairs(386268)"] = np.r_[d.cpg_i, d.cpg_j]
    for name, rel in [("microc_H1_intra10kb", "3d_genome/frozen_pairs_H1_intra10kb_seed17.parquet"),
                      ("microc_HFFc6_intra10kb", "3d_genome/frozen_pairs_HFFc6_intra10kb_seed17.parquet"),
                      ("microc_H1_inter1Mb(explor.)", "3d_genome/frozen_pairs_H1_inter1Mb_seed17.parquet")]:
        d = rd(rel, ["cpg_i", "cpg_j"])
        s[name] = np.r_[d.cpg_i, d.cpg_j]
    for name, rel in [("fantom5_membership_primary", "frozen_enhancer_membership_seed17.parquet"),
                      ("fantom5_win500(sens.)", "frozen_enhancer_membership_win500_seed17.parquet"),
                      ("fantom5_tss5k(sens.)", "frozen_enhancer_membership_tss5kpool_seed17.parquet")]:
        s[name] = rd("regulatory_activity/" + rel, ["cpg_idx"]).cpg_idx.to_numpy()
    d = rd("regulatory_activity/frozen_activity_pairs_seed17.parquet", ["i", "j"])
    s["fantom5_activity_pairs(171017)"] = np.r_[d.i, d.j]
    d = rd("replication_domains/cpg_rt_universe.parquet", ["cpg_idx", "rt_consensus_z"])
    s["rt_eligible(408345)"] = d.cpg_idx[d.rt_consensus_z.notna()].to_numpy()
    s["rt_universe_all(408399)"] = d.cpg_idx.to_numpy()
    for name in ("H1", "HFFc6", "GM12878"):
        d = rd(f"3d_genome/frozen_compartments_{name}_100kb.parquet", ["cpg_idx", "E1"])
        s[f"compartments_{name}_with_E1"] = d.cpg_idx[d.E1.notna()].to_numpy()
    d = rd("external_methylation_programs/loyfer_markers_reconciled.parquet", ["cpg_idx"])
    s["loyfer_markers(secondary)"] = d.cpg_idx.to_numpy()
    d = rd("pmd_decato2020/cpg_pmd_summary_universe.parquet", ["cpg_idx"])
    s["pmd_universe(explor.)"] = d.cpg_idx.to_numpy()
    return s


def cmd_verify(a) -> dict:
    man = json.loads((ROOT / BV / "MANIFEST_checksums.json").read_text())
    res = {"checksum_manifest": verify_checksum_manifest(ROOT / BV, man)}
    doc = parse_doc_hashes((ROOT / "docs/BIOLOGICAL_VALIDATION_V2.md").read_text())
    dres = {"n": len(doc), "ok": 0, "mismatch": [], "missing": []}
    for rel, sha in doc.items():
        p = ROOT / BV / rel
        if not p.is_file():
            dres["missing"].append(rel)
        elif sha256_file(p) == sha:
            dres["ok"] += 1
        else:
            dres["mismatch"].append(rel)
    res["doc_section7"] = dres
    return res


def cmd_stores(a) -> dict:
    import h5py
    import yaml
    mx = yaml.safe_load((ROOT / "configs/experiments/regulatory_confirmation_matrix/matrix.yaml").read_text())
    arms = {x["id"]: x for x in mx["arms"] if x.get("store_h5")} if "arms" in mx else {}
    if not arms:  # nested layout fallback
        def walk(o):
            if isinstance(o, dict):
                if "id" in o and o.get("store_h5"):
                    yield o
                for v in o.values():
                    yield from walk(v)
            elif isinstance(o, list):
                for v in o:
                    yield from walk(v)
        arms = {x["id"]: x for x in walk(mx)}
    want = a.arms or ["regulatory_histone_dnase", "functional_annotations_pca", "cpgpt_large_locus", "deepcpg_dna_locus"]
    sets = frozen_cpg_sets()
    out = {}
    for arm in want:
        e = arms[arm]
        p = ROOT / e["store_h5"]
        r = {"path": e["store_h5"], "sha256_expected": e["sha256"], "bytes_expected": e.get("size_bytes")}
        r["bytes_actual"] = p.stat().st_size
        r["sha256_actual"] = sha256_file(p) if not a.skip_sha else None
        r["sha256_ok"] = (r["sha256_actual"] == e["sha256"]) if r["sha256_actual"] else None
        with h5py.File(p, "r") as h:
            from cpg_repr_benchmark.representations.hdf5_store import inspect_representation_h5
            ids, dim, idk, ek = inspect_representation_h5(p)
            r.update(id_key=idk, embedding_key=ek, dim=dim, dim_expected=e.get("dim"), n_rows=len(ids),
                     cpg_idx_sorted=bool(np.all(np.diff(ids) > 0)), cpg_idx_unique=True)
            r["coverage"] = {k: coverage(ids, v) for k, v in sets.items()}
            r["finite_scan"] = scan_finite(h[ek]) if not a.skip_scan else None
        out[arm] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    sp = sub.add_parser("stores")
    sp.add_argument("--arms", nargs="*")
    sp.add_argument("--skip-sha", action="store_true")
    sp.add_argument("--skip-scan", action="store_true")
    for s in sub.choices.values():
        s.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    _guard_out(a.out)
    res = {"verify": cmd_verify, "stores": cmd_stores}[a.cmd](a)
    txt = json.dumps(res, indent=1, default=str)
    if a.out:
        a.out.write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
