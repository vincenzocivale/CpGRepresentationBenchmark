"""Shared helpers: hashing, minimal xlsx reader (no openpyxl), weighted primary-style statistics, chromosome-block bootstrap glue."""
from __future__ import annotations

import hashlib
import re
import zipfile
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from cpg_repr_benchmark.bioval_v2_launch import launch_endpoints as le
from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import (
    BlockBootstrap,
    RankPlan,
    block_index,
    contrast,
    percentile_ci,
    weighted_spearman,
)

from . import registry as rg


def sha256_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ---------------------------------------------------------------------------------------------- minimal xlsx reader
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def _col_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_xlsx_sheet(path, sheet_name: str, header_key: str | None = None) -> pd.DataFrame:
    """Read one sheet of an .xlsx as strings. Header = first non-empty row, or the first row containing ``header_key``.
    zipfile + ElementTree only."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rid = None
        for s in wb.find("m:sheets", _NS):
            if s.get("name") == sheet_name:
                rid = s.get(f"{{{_NS['r']}}}id")
        if rid is None:
            raise KeyError(f"sheet {sheet_name!r} not found")
        target = next(r.get("Target") for r in rels if r.get("Id") == rid)
        target = target.lstrip("/")
        sheet = ET.fromstring(z.read(target if target.startswith("xl/") else f"xl/{target}"))
    rows = []
    for row in sheet.iter(f"{{{_NS['m']}}}row"):
        cells = {}
        for c in row.findall("m:c", _NS):
            v = c.find("m:v", _NS)
            t = c.get("t")
            if t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter(f"{{{_NS['m']}}}t"))
            elif v is None:
                continue
            elif t == "s":
                val = shared[int(v.text)]
            else:
                val = v.text
            cells[_col_index(c.get("r"))] = val
        if cells:
            rows.append(cells)
    width = max(max(r) for r in rows) + 1
    mat = [[r.get(i) for i in range(width)] for r in rows]
    h0 = 0
    if header_key is not None:
        h0 = next(i for i, r in enumerate(mat) if header_key in r)
    header = [h if h is not None else f"col{i}" for i, h in enumerate(mat[h0])]
    return pd.DataFrame(mat[h0 + 1:], columns=header)


# ---------------------------------------------------------------------------------------------- statistics glue
def make_boot(chroms=rg.CHROMS, n_boot: int = 1000, seed: int = rg.SEED) -> BlockBootstrap:
    return BlockBootstrap(np.asarray(sorted(chroms)), n_boot=n_boot, seed=seed)


def block_of(boot: BlockBootstrap, chroms) -> np.ndarray:
    return block_index(chroms, boot.names)


def pooled_base_weights(strat, pooled=rg.POOLED_STRATA):
    """Equal stratum weight: item weight 1/n_stratum within each pooled stratum; returns (selector, weights over selector)."""
    strat = np.asarray(strat)
    sel = np.isin(strat, pooled)
    w = np.zeros(int(sel.sum()))
    s = strat[sel]
    for k in pooled:
        m = s == k
        if m.sum() == 0:
            return sel, None
        w[m] = 1.0 / m.sum()
    return sel, w


def _ok(x, y):
    return np.isfinite(x) & np.isfinite(y)


def stat_unweighted(boot, name, x, y, blk, mask=None, ci=True, cls="exploratory_posthoc"):
    x, y, blk = np.asarray(x, float), np.asarray(y, float), np.asarray(blk)
    keep = _ok(x, y) if mask is None else (_ok(x, y) & mask)
    x, y, blk = x[keep], y[keep], blk[keep]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    px, py = RankPlan(x), RankPlan(y)
    return le.make_stat(SimpleNamespace(boot=boot), name, lambda w: weighted_spearman(px, py, w), blk, len(x), cls=cls, ci=ci)


def stat_pooled(boot, name, x, y, strat, blk, mask=None, ci=True, cls="exploratory_posthoc", pooled=rg.POOLED_STRATA):
    """Primary-style statistic: weighted Spearman pooled over >1Mb + interchromosomal, weight 1/n_stratum (D1)."""
    x, y, strat, blk = np.asarray(x, float), np.asarray(y, float), np.asarray(strat), np.asarray(blk)
    keep = _ok(x, y) if mask is None else (_ok(x, y) & mask)
    x, y, strat, blk = x[keep], y[keep], strat[keep], blk[keep]
    sel, w0 = pooled_base_weights(strat, pooled)
    if w0 is None or np.ptp(x[sel]) == 0 or np.ptp(y[sel]) == 0:
        return None
    px, py = RankPlan(x[sel]), RankPlan(y[sel])
    return le.make_stat(SimpleNamespace(boot=boot), name, lambda w: weighted_spearman(px, py, w), blk[sel], int(sel.sum()),
                        base_w=w0, cls=cls, ci=ci)


def stat_dict(s):
    if s is None:
        return None
    return {"value": s.value, "ci_lo": s.ci_lo, "ci_hi": s.ci_hi, "n": s.n}


def paired_delta(a, b):
    """delta = a - b with paired block-bootstrap CI (both Stat with .rep from the same draw table)."""
    if a is None or b is None or a.rep is None or b.rep is None:
        return None
    return contrast(a.value, a.rep, b.value, b.rep)


def spearman_fast(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    k = _ok(x, y)
    if k.sum() < 3 or np.ptp(x[k]) == 0 or np.ptp(y[k]) == 0:
        return float("nan")
    a, b = rankdata(x[k]), rankdata(y[k])
    return float(np.corrcoef(a, b)[0, 1])


def float16_ulp(x):
    """Spacing of float16 at |x| (subnormals: 2^-24)."""
    a = np.abs(np.asarray(x, np.float64))
    e = np.floor(np.log2(np.maximum(a, 2.0 ** -14)))
    return 2.0 ** (e - 10)


def percentile_summary(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    lo, hi = percentile_ci(v)
    return {"n": len(v), "median": float(np.median(v)), "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
            "q025": lo, "q975": hi}
