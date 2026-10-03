#!/usr/bin/env python
# ruff: noqa: DTZ011, EXE001, F841  (one-shot data-prep script; style-only, behaviour unchanged)
"""Prepare replication-timing (RT) validation target for biological validation v2. CPU only.

Source (pre-declared primary): UW Repli-seq WaveSignal (Hansen 2010 PNAS / ENCODE UW), hg19, 1 kb steps, 15 cell lines.
Steps: read bigWigs -> verify shared bin grid -> lift grid hg19->GRCh38 once (harmonize, interval rule) ->
bin table (long) -> CpG table over the 408,399 universe (value of containing bin, NaN otherwise, no extrapolation)
-> per-line z-score consensus (mean of z, requires >=3 lines) -> overlap audit, autocorrelation, manifest.
Raw files must already be in data/external/bioval_v2/replication_domains/ (see PREP report for URLs).
"""
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cpg_repr_benchmark.biological_validation_v2 import coordinates as C  # sibling package (read-only use)

RAW = ROOT / "data/external/bioval_v2/replication_domains"
OUT = ROOT / "data/derived/bioval_v2/replication_domains"
BASE = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeUwRepliSeq"
LINES = ["Bg02es", "Bj", "Gm06990", "Gm12801", "Gm12812", "Gm12813", "Gm12878", "Helas3", "Hepg2", "Huvec",
         "Imr90", "K562", "Mcf7", "Nhek", "Sknsh"]
CHROMS = C.CANONICAL_CHROMS


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_line(path):
    bw = pyBigWig.open(str(path))
    rows = []
    for c in [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]:
        if c in bw.chroms():
            iv = bw.intervals(c)
            if iv:
                a = np.array(iv, dtype=np.float64)
                rows.append(pd.DataFrame({"chrom": c, "start0": a[:, 0].astype(np.int64),
                                          "end0": a[:, 1].astype(np.int64), "rt": a[:, 2]}))
    return pd.concat(rows, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = {"retrieval_date": datetime.date.today().isoformat(), "source": "UCSC hg19 encodeDCC wgEncodeUwRepliSeq (WaveSignal Rep1)",
           "base_url": BASE, "build_in": "hg19/GRCh37", "build_out": "GRCh38", "files": {}}
    grid, vals = None, {}
    for ln in LINES:
        fn = f"wgEncodeUwRepliSeq{ln}WaveSignalRep1.bigWig"
        p = RAW / fn
        d = read_line(p)
        if grid is None:
            grid = d[["chrom", "start0", "end0"]].copy()
            grid["bin_id"] = np.arange(len(grid))
        key = grid[["chrom", "start0", "end0"]]
        if len(d) != len(grid) or not (d[["chrom", "start0", "end0"]].values == key.values).all():
            m = grid.merge(d, on=["chrom", "start0", "end0"], how="left")  # fallback: align on shared grid
            vals[ln] = m["rt"].to_numpy()
            man.setdefault("grid_mismatch_lines", []).append(ln)
        else:
            vals[ln] = d["rt"].to_numpy()
        man["files"][fn] = {"url": f"{BASE}/{fn}", "sha256": sha256(p), "n_bins": len(d),
                            "bytes": p.stat().st_size}
    print("grid bins", len(grid), "mismatch", man.get("grid_mismatch_lines"))
    # lift grid once
    lifted, rep = C.harmonize(grid, "chrom", "start0", "hg19", pos_base=0, end_col="end0")
    print(rep)
    lifted = lifted.rename(columns={"pos": "start", "end": "end"})  # 1-based inclusive in GRCh38
    # NOTE harmonize dedups on (chrom,start,end); bin_id survives for kept rows
    man["liftover_report_bins"] = {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in rep.items()}
    bid = lifted["bin_id"].to_numpy()
    lifted["bin_width_38"] = lifted["end"] - lifted["start"] + 1
    # long bin table
    parts = []
    for ln in LINES:
        v = vals[ln][bid]
        ok = ~np.isnan(v)
        t = pd.DataFrame({"chrom": lifted["chrom"].to_numpy()[ok], "start": lifted["start"].to_numpy()[ok],
                          "end": lifted["end"].to_numpy()[ok], "rt": v[ok].astype(np.float32),
                          "cell_line": ln.upper(), "source": "UW_RepliSeq_WaveSignal_Rep1_ENCODE_Mar2012",
                          "build_in": "hg19",
                          "start_hg19_0based": grid["start0"].to_numpy()[bid][ok], "end_hg19": grid["end0"].to_numpy()[bid][ok]})
        parts.append(t)
    bins = pd.concat(parts, ignore_index=True)
    bins["cell_line"] = bins["cell_line"].astype("category")
    bins.to_parquet(OUT / "rt_bins_grch38_long.parquet", index=False, compression="zstd")
    man["bin_table_rows"] = len(bins)
    # z-score stats per line (genome-wide bins, autosomes+X+Y as provided)
    zs = {ln: (float(np.nanmean(vals[ln])), float(np.nanstd(vals[ln]))) for ln in LINES}
    man["zscore_params_per_line_(mean,sd)"] = zs
    # CpG lookup
    U = C.load_benchmark_universe()
    L = lifted.sort_values(["chrom", "start"], kind="mergesort").reset_index(drop=True)
    out = U[["cpg_idx", "chrom", "pos"]].copy()
    binrow = np.full(len(U), -1, dtype=np.int64)
    for c in CHROMS:
        um = np.where(U["chrom"].to_numpy() == c)[0]
        lm = np.where(L["chrom"].to_numpy() == c)[0]
        if not len(um) or not len(lm):
            continue
        st = L["start"].to_numpy()[lm]; en = L["end"].to_numpy()[lm]
        i = np.searchsorted(st, U["pos"].to_numpy()[um], side="right") - 1
        ok = i >= 0
        ii = np.where(ok, i, 0)
        ok &= en[ii] >= U["pos"].to_numpy()[um]
        binrow[um[ok]] = lm[ii[ok]]
    has = binrow >= 0
    obid = np.where(has, L["bin_id"].to_numpy()[np.where(has, binrow, 0)], -1)
    zmat = []
    for ln in LINES:
        v = np.where(has, vals[ln][np.where(has, obid, 0)], np.nan)
        out[f"rt_{ln.upper()}"] = v.astype(np.float32)
        zmat.append((v - zs[ln][0]) / zs[ln][1])
    zmat = np.vstack(zmat)
    nl = (~np.isnan(zmat)).sum(0)
    cons = np.where(nl >= 3, np.nanmean(np.where(np.isnan(zmat), 0, zmat) * 0 + zmat, axis=0), np.nan) if False else \
        np.where(nl >= 3, np.nansum(zmat, 0) / np.maximum(nl, 1), np.nan)
    out["rt_consensus_z"] = cons.astype(np.float32)
    out["n_lines_with_rt"] = nl.astype(np.int8)
    out["bin_hg19_start0"] = np.where(has, grid["start0"].to_numpy()[np.where(has, obid, 0)], -1)
    out.to_parquet(OUT / "cpg_rt_universe.parquet", index=False, compression="zstd")
    assert len(out) == C.UNIVERSE_SIZE
    # overlap audit
    cov = out.assign(any_rt=(nl >= 1)).groupby("chrom").agg(n_universe=("cpg_idx", "size"), n_with_rt=("any_rt", "sum"),
                                                          n_consensus=("rt_consensus_z", lambda s: s.notna().sum()))
    cov["frac"] = cov["n_with_rt"] / cov["n_universe"]
    cov = cov.reindex([c for c in CHROMS if c in cov.index])
    cov.to_csv(OUT / "coverage_per_chrom.csv")
    perline = {f"rt_{ln.upper()}": int(out[f"rt_{ln.upper()}"].notna().sum()) for ln in LINES}
    man["cpg_table"] = {"rows": len(out), "n_any_rt": int((nl >= 1).sum()), "n_consensus": int(np.isfinite(cons).sum()),
                        "per_line_n_with_rt": perline, "coverage_per_chrom": cov.reset_index().to_dict("records")}
    # autocorrelation on 1 kb bin series (hg19 grid, contiguous runs only), lines K562 / GM12878 / IMR90
    ac = {}
    for ln in ["K562", "Gm12878", "Imr90"]:
        v = vals[ln]; chrs = grid["chrom"].to_numpy()
        num = {}; var = 0.0; n = 0
        lags = [1, 10, 50, 100, 250, 500, 1000, 2000]
        s = {l: 0.0 for l in lags}; cnt = {l: 0 for l in lags}
        mu = np.nanmean(v); sd = np.nanstd(v)
        for c in sorted(set(chrs)):
            x = (v[chrs == c] - mu) / sd
            x = x[~np.isnan(x)]
            for l in lags:
                if len(x) > l:
                    s[l] += float((x[:-l] * x[l:]).sum()); cnt[l] += len(x) - l
        ac[ln.upper()] = {f"rho_lag{l}kb": s[l] / cnt[l] for l in lags}
    man["bin_autocorr_1kb_steps"] = ac
    man["tool_versions"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                            "pyBigWig": getattr(pyBigWig, "__version__", "?"),
                            "pyliftover": rep.get("liftover_tool"), "chain": rep.get("chain")}
    man["chain_sha256"] = sha256(ROOT / C.DEFAULT_CHAIN)
    man["outputs"] = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in OUT.glob("*.parquet")}
    (OUT / "MANIFEST_rt.json").write_text(json.dumps(man, indent=1, default=str))
    print(cov); print(perline); print(ac)


if __name__ == "__main__":
    main()
