#!/usr/bin/env python3
"""READ-ONLY audit of locally available external methylation cohorts (no model, no representation evaluated).

What it does (all metadata/coverage/summary statistics, nothing is trained or scored):
  * builds the benchmark locus universe (array_cpg_map.parquet, 408,399 CpGs, TCGA global cpg_idx) as (chr,pos) keys;
  * reads the regulatory store ONLY to flag all-zero embedding rows (counts of loci; no embedding values are used);
  * for each prepared cohort h5 (coordinate-native cpg_idx = chrom*1e9+pos, GRCh38 1-based): number of samples/loci,
    overlap with the universe (total, chr1-19, chr20-22, zero-embedding loci), beta distribution on a column/row subsample,
    NaN encoding/fraction, outside-[0,1] fraction, bimodality fractions, duplicates;
  * for raw/unprepared cohorts: probe-id lists mapped through the repo's GRCh38 probe crosswalk;
  * TCGA overlap check by identifiers ONLY (sample names read from the patient-protocol npz; no beta of TCGA is read).

Outputs go to outputs/external_reconstruction_audit/ (new directory). Nothing under data/derived/bioval_v2, configs/frozen,
data/cache/representations or outputs/biological_validation_v2 is written.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CHROM_CODE = {**{f"chr{i}": i for i in range(1, 23)}, "chrX": 23, "chrY": 24, "chrM": 25}
TCGA_ID_RE = re.compile(r"^(TCGA|TARGET)-", re.IGNORECASE)


def coord_key(chrom: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """int64 key chrom_code*1e9+pos (identical to the external cpg_idx convention); -1 for unknown chromosomes."""
    codes = np.array([CHROM_CODE.get(str(c), -1) for c in chrom], dtype=np.int64)
    key = codes * 1_000_000_000 + np.asarray(pos, dtype=np.int64)
    key[codes < 0] = -1
    return key


def overlap_counts(cohort_keys: np.ndarray, uni_keys: np.ndarray, uni_chr_code: np.ndarray, uni_zero: np.ndarray) -> dict:
    """Counts of universe loci covered by the cohort (cohort keys may contain duplicates / -1)."""
    present = np.isin(uni_keys, np.unique(cohort_keys[cohort_keys >= 0]))
    return {
        "universe": int(len(uni_keys)),
        "covered": int(present.sum()),
        "covered_fraction": float(present.mean()),
        "covered_chr1_19": int((present & (uni_chr_code <= 19)).sum()),
        "universe_chr1_19": int((uni_chr_code <= 19).sum()),
        "covered_chr20_22": int((present & (uni_chr_code >= 20)).sum()),
        "universe_chr20_22": int((uni_chr_code >= 20).sum()),
        "universe_zero_embedding": int(uni_zero.sum()),
        "covered_zero_embedding": int((present & uni_zero).sum()),
        "cohort_loci_not_in_universe": int(len(np.unique(cohort_keys[cohort_keys >= 0])) - present.sum()),
    }


def beta_summary(block: np.ndarray) -> dict:
    """Summary of a (rows x cols) beta block. NaN = missing."""
    block = np.asarray(block, dtype=np.float64)
    finite = np.isfinite(block)
    v = block[finite]
    out = {"n_values": int(block.size), "nan_fraction": float(1.0 - finite.mean())}
    if v.size == 0:
        return out
    qs = np.quantile(v, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    out.update(
        quantiles=dict(zip(["q00", "q01", "q05", "q25", "q50", "q75", "q95", "q99", "q100"], map(float, qs))),
        mean=float(v.mean()),
        outside_0_1_fraction=float(((v < 0) | (v > 1)).mean()),
        exact_0_or_1_fraction=float(((v == 0) | (v == 1)).mean()),
        frac_lt_0_2=float((v < 0.2).mean()),
        frac_gt_0_8=float((v > 0.8).mean()),
        frac_mid_0_2_0_8=float(((v >= 0.2) & (v <= 0.8)).mean()),
    )
    per_sample_nan = 1.0 - finite.mean(axis=1)
    per_cpg_nan = 1.0 - finite.mean(axis=0)
    out["per_sample_nan"] = {"max": float(per_sample_nan.max()), "p95": float(np.quantile(per_sample_nan, 0.95)),
                             "n_samples_nan_gt_0_05": int((per_sample_nan > 0.05).sum())}
    out["per_cpg_nan"] = {"max": float(per_cpg_nan.max()), "n_cpg_all_nan": int((per_cpg_nan == 1.0).sum()),
                          "n_cpg_nan_gt_0_05": int((per_cpg_nan > 0.05).sum())}
    return out


def tcga_id_check(names: list[str], tcga_names: set[str]) -> dict:
    names = [str(n) for n in names]
    tcga_like = [n for n in names if TCGA_ID_RE.match(n)]
    return {"n_names": len(names), "n_tcga_pattern": len(tcga_like), "n_exact_overlap_with_tcga_names": len(set(names) & tcga_names)}


def universe(repo: Path) -> dict:
    m = pd.read_parquet(repo / "data/cpg/registries/array_cpg_map.parquet")
    keys = coord_key(m["chr"].to_numpy(), m["pos"].to_numpy())
    return {"table": m, "keys": keys, "chr_code": np.array([CHROM_CODE[c] for c in m["chr"]], dtype=np.int64),
            "cpg_idx": m["cpg_idx"].to_numpy(np.int64)}


def zero_embedding_mask(store_h5: Path, uni_cpg_idx: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """True for universe loci whose regulatory embedding row is all zeros (read in chunks; values are not used otherwise)."""
    with h5py.File(store_h5, "r") as f:
        ids = f["cpg_idx"][:].astype(np.int64)
        emb = f["embedding"]
        zero = np.zeros(len(ids), dtype=bool)
        for s in range(0, len(ids), chunk):
            zero[s:s + chunk] = ~np.any(np.asarray(emb[s:s + chunk]) != 0, axis=1)
    zset = set(ids[zero].tolist())
    return np.array([int(i) in zset for i in uni_cpg_idx])


def _blocks(rng, n_cols: int, n_blocks: int, width: int = 4096) -> np.ndarray:
    starts = rng.choice(max(n_cols // width, 1), size=min(n_blocks, max(n_cols // width, 1)), replace=False) * width
    return np.concatenate([np.arange(s, min(s + width, n_cols)) for s in np.sort(starts)])


def audit_h5(path: Path, repo: Path, uni: dict, zero: np.ndarray, *, rows: np.ndarray | None, n_cols: int = 20000,
             seed: int = 17, aligned_blocks: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        beta = f["beta"]
        n_s, n_c = beta.shape
        chrom = f["chrom"][:].astype(str)
        pos = f["pos"][:]
        keys = coord_key(chrom, pos)
        res = {"path": str(path.relative_to(repo)), "n_samples": int(n_s), "n_loci": int(n_c),
               "n_duplicate_coordinates": int(n_c - len(np.unique(keys))),
               "attrs": {k: str(v) for k, v in f.attrs.items()},
               "overlap": overlap_counts(keys, uni["keys"], uni["chr_code"], zero)}
        in_uni = np.flatnonzero(np.isin(keys, uni["keys"]))
        if aligned_blocks:   # cheap random access for coarse chunked matrices
            cols = np.intersect1d(_blocks(rng, n_c, 14), in_uni)
        else:
            cols = np.sort(rng.choice(in_uni, size=min(n_cols, len(in_uni)), replace=False))
        cols = cols[:n_cols] if len(cols) > n_cols else cols
        sel = np.arange(n_s) if rows is None else rows
        block = np.empty((len(sel), len(cols)), dtype=np.float32)
        for k, r in enumerate(sel):
            block[k] = beta[int(r), cols]
        res["beta_subsample"] = {"n_rows": int(len(sel)), "n_cols": int(len(cols)), **beta_summary(block)}
        # exact duplicate samples on the subsample (rounded hash)
        h = [hash(np.round(np.nan_to_num(b, nan=-1.0), 6).tobytes()) for b in block]
        res["n_duplicate_sample_rows_on_subsample"] = int(len(h) - len(set(h)))
    return res


def raw_probe_overlap(probe_file: Path, crosswalk: pd.DataFrame, uni: dict, zero: np.ndarray, skip_header: int = 0) -> dict:
    probes = [p.strip().strip('"') for p in probe_file.read_text().splitlines()[skip_header:]]
    probes = [p for p in probes if p.startswith(("cg", "ch", "rs"))]
    x = crosswalk[crosswalk["probe_id"].isin(set(probes))]
    keys = coord_key(x["chr"].to_numpy(), x["pos"].to_numpy())
    out = overlap_counts(keys, uni["keys"], uni["chr_code"], zero)
    out.update(n_probe_rows=len(probes), n_unique_probes=len(set(probes)), n_probes_mapped_by_crosswalk=int(len(x)))
    return out


def raw_head_summary(gz_path: Path, n_rows: int = 3000, delimiter: str = "\t") -> dict:
    """Beta summary from the first n_rows probe rows of a raw gz beta table (columns named like 'Detection Pval' are skipped).

    Only reads the head of the file (a few thousand probes x all samples). Header ids are returned for replicate/TCGA checks.
    """
    import gzip

    with gzip.open(gz_path, "rt") as f:
        header = f.readline().rstrip("\n").split(delimiter)
        keep = [i for i, h in enumerate(header) if i > 0 and "pval" not in h.lower() and "detection" not in h.lower()]
        rows = []
        for _ in range(n_rows):
            line = f.readline()
            if not line:
                break
            parts = line.rstrip("\n").split(delimiter)
            rows.append([float(parts[i]) if parts[i] not in ("", "NA", "NaN", "nan") else np.nan for i in keep])
    block = np.asarray(rows, dtype=np.float64).T  # samples x probes
    return {"n_sample_columns": len(keep), "n_probe_rows_read": block.shape[1], "sample_ids_head": [header[i] for i in keep[:3]],
            "tcga_id_check": tcga_id_check([header[i] for i in keep], set()), **beta_summary(block)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/external_reconstruction_audit")
    ap.add_argument("--probe-dir", type=Path, default=None, help="dir with <ACC>.txt probe-id lists of raw cohorts (optional)")
    ap.add_argument("--raw-head", action="append", default=[], metavar="NAME=PATH",
                    help="raw gz beta table whose first rows are summarised (repeatable); written to raw_head_summaries.json")
    ap.add_argument("--only-raw-head", action="store_true", help="skip the full audit and only run --raw-head summaries")
    ap.add_argument("--regulatory-store", type=Path,
                    default=REPO / "data/cache/representations/regulatory_histone_dnase__global_svd256__discovery_chr1_19.h5")
    args = ap.parse_args()
    repo = args.repo
    args.out.mkdir(parents=True, exist_ok=True)

    if args.only_raw_head:
        heads = {}
        for item in args.raw_head:
            name, path = item.split("=", 1)
            heads[name] = {"path": path, **raw_head_summary(Path(path))}
        (args.out / "raw_head_summaries.json").write_text(json.dumps(heads, indent=1, default=str))
        print(json.dumps({k: {kk: v[kk] for kk in ("n_sample_columns", "n_probe_rows_read", "nan_fraction", "outside_0_1_fraction", "frac_mid_0_2_0_8")} for k, v in heads.items()}, indent=1))
        return
    uni = universe(repo)
    zero = zero_embedding_mask(args.regulatory_store, uni["cpg_idx"])
    np.savez_compressed(args.out / "universe_zero_loci.npz", zero_cpg_idx=uni["cpg_idx"][zero], zero_chr_code=uni["chr_code"][zero])
    # TCGA metadata ids only (sample_names axis of the frozen patient protocol; NO beta read)
    with np.load(repo / "outputs/encode_atlas_v1/patients.npz", allow_pickle=False) as p:
        tcga_names = {str(x) for x in p["sample_names"]}
    result: dict = {"universe": {"n": int(len(uni["keys"])), "n_chr1_19": int((uni["chr_code"] <= 19).sum()),
                                 "n_chr20_22": int((uni["chr_code"] >= 20).sum()), "n_zero_embedding": int(zero.sum()),
                                 "zero_by_chr_group": {"chr1_19": int((zero & (uni["chr_code"] <= 19)).sum()),
                                                       "chr20_22": int((zero & (uni["chr_code"] >= 20)).sum())},
                                 "id_system": "TCGA global cpg_idx (MethylProphet); external cohorts use chrom*1e9+pos (GRCh38, 1-based); joined on (chr,pos)"},
                    "cohorts": {}}

    proc = repo / "data/processed"
    cfgs = {
        "GSE40279": dict(h5=proc / "GSE40279/methylation.h5", pheno=proc / "GSE40279/phenotypes.parquet", id_col="subject_id", name_col="sample_name", rows=None),
        "GSE42861": dict(h5=proc / "GSE42861/methylation.h5", pheno=proc / "GSE42861/phenotypes.parquet", id_col="subject_id", name_col="sample_name", rows=None),
        "GSE147221": dict(h5=proc / "GSE147221/methylation.h5", pheno=proc / "GSE147221/phenotypes.parquet", id_col="array_id", name_col="sample_name", rows=None),
    }
    for acc, c in cfgs.items():
        r = audit_h5(c["h5"], repo, uni, zero, rows=c["rows"])
        ph = pd.read_parquet(c["pheno"])
        with h5py.File(c["h5"], "r") as f:
            names = [x.decode() if isinstance(x, bytes) else str(x) for x in f["sample_name"][:]]
        r["tcga_id_check"] = tcga_id_check(names, tcga_names)
        r["phenotype_columns"] = list(ph.columns)
        if c["id_col"] in ph:
            r["n_unique_subject_or_array_ids"] = int(ph[c["id_col"]].nunique())
        result["cohorts"][acc] = r
        print(acc, r["n_samples"], r["overlap"]["covered"], flush=True)

    # ComputAgeBench: pooled matrix, chunk-aligned row blocks per study (coarse (32,4096) gzip chunks)
    cab = proc / "ComputAgeBench/benchmark"
    ph = pd.read_parquet(cab / "phenotypes.parquet")
    rng = np.random.default_rng(17)
    with h5py.File(cab / "methylation.h5", "r") as f:
        h5_names = [x.decode() if isinstance(x, bytes) else str(x) for x in f["sample_name"][:]]
        if h5_names != ph["sample_name"].tolist():
            raise RuntimeError("ComputAgeBench phenotypes row order differs from h5 sample axis")
        n_s, n_c = f["beta"].shape
        keys = coord_key(f["chrom"][:].astype(str), f["pos"][:])
        in_uni = np.flatnonzero(np.isin(keys, uni["keys"]))
        cols = np.intersect1d(_blocks(rng, n_c, 14), in_uni)
        per_study = {}
        for sid, g in ph.reset_index(drop=True).groupby("dataset_id"):
            rows = g.index.to_numpy()
            start = (rows.min() // 32) * 32
            rr = np.arange(start, min(start + 32, n_s))
            rr = rr[np.isin(rr, rows)]
            blk = np.asarray(f["beta"][rr[0]:rr[-1] + 1][:, cols], dtype=np.float32)
            blk = blk[np.isin(np.arange(rr[0], rr[-1] + 1), rr)]
            s = beta_summary(blk)
            per_study[sid] = {"n_samples": int(len(g)), "platforms": sorted(set(g["PlatformID"])), "cell_types": sorted(set(g["CellType"])),
                              "tissues": sorted(set(g["Tissue"])), "n_probe_rows_sampled": int(len(rr)),
                              "nan_fraction_on_universe_blocks": s.get("nan_fraction"), "q50": s.get("quantiles", {}).get("q50"),
                              "outside_0_1_fraction": s.get("outside_0_1_fraction"), "frac_mid_0_2_0_8": s.get("frac_mid_0_2_0_8")}
    result["cohorts"]["ComputAgeBench_benchmark"] = {
        "n_samples": int(n_s), "n_loci": int(n_c), "overlap": overlap_counts(keys, uni["keys"], uni["chr_code"], zero),
        "n_studies": int(ph["dataset_id"].nunique()), "platform_counts": ph["PlatformID"].value_counts().to_dict(),
        "cell_type_counts": ph["CellType"].value_counts().to_dict(), "tissue_counts": ph["Tissue"].value_counts().to_dict(),
        "condition_class_counts": ph["condition_class"].value_counts().to_dict(),
        "tcga_id_check": tcga_id_check(ph["sample_name"].tolist(), tcga_names), "per_study": per_study}
    print("ComputAgeBench done", flush=True)

    # derived/prepared small sets (coverage only)
    for tag, p, cols_ in [("epic_sim_training", repo / "data/prepared/epic_sim_training/epic_blood.h5", ("cpg_idx_obs", "cpg_idx_target")),
                          ("paired_450k_epic_GSE86833", repo / "data/prepared/paired_450k_epic/GSE86833.h5", ("cpg_idx_obs", "cpg_idx_target")),
                          ("paired_450k_epic_GSE92580", repo / "data/prepared/paired_450k_epic/GSE92580.h5", ("cpg_idx_obs", "cpg_idx_target"))]:
        with h5py.File(p, "r") as f:
            ids = np.concatenate([f[k][:] for k in cols_]).astype(np.int64)
            n = int(f["target_beta"].shape[0])
        # these store the coordinate-native cpg_idx (chrom*1e9+pos): join to universe on the key directly
        result["cohorts"][tag] = {"n_samples": n, "n_loci": int(len(np.unique(ids))), "overlap": overlap_counts(ids, uni["keys"], uni["chr_code"], zero)}

    # raw / unprepared cohorts: probe lists (optional)
    if args.probe_dir and args.probe_dir.exists():
        cw = pd.read_parquet(repo / "data/processed/ComputAgeBench/illumina_probe_grch38.parquet")
        for f in sorted(args.probe_dir.glob("*.txt")):
            acc = f.stem.replace("_probes", "")
            skip = 1 if acc in {"gse55763"} else 0
            result["cohorts"][f"raw_{acc}"] = {"probe_file": f.name, **raw_probe_overlap(f, cw, uni, zero, skip_header=skip)}
    (args.out / "cohort_audit.json").write_text(json.dumps(result, indent=1, default=str))
    rows = []
    for k, v in result["cohorts"].items():
        o = v.get("overlap", v)
        rows.append({"cohort": k, "n_samples": v.get("n_samples"), "n_loci": v.get("n_loci"), "covered": o.get("covered"),
                     "covered_fraction": o.get("covered_fraction"), "covered_chr1_19": o.get("covered_chr1_19"),
                     "covered_chr20_22": o.get("covered_chr20_22"), "covered_zero_embedding": o.get("covered_zero_embedding")})
    pd.DataFrame(rows).to_csv(args.out / "cohort_overlap_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
