"""Canonical coordinates and universe handling for biological validation v2.

CANONICAL: GRCh38, 1-based position of the CpG cytosine, chrom 'chr1'..'chr22','chrX','chrY'
(namespace ``grch38_cpg_cytosine_1based_v1``; see data/cpg/master_cpg_registry.parquet, columns chr,pos).

Liftover tool: ``pyliftover`` (pure python, reads UCSC chain, optionally gzipped) with the chain
data/raw/liftover/hg19ToHg38.over.chain.gz. pyliftover works in 0-based coordinates; this module converts.

Benchmark universe (408,399 CpGs): rows of ``data/cpg/registries/array_cpg_map.parquet`` (Illumina array
registry). Its ``cpg_idx`` column is the registry id that also keys every representation store's
``/cpg_idx`` (verified: functional_annotations__pca_native_genomewide.h5 /cpg_idx == this column);
``raw_cpg_row`` is the legacy row axis. The universe is defined ONLY by this file, never by a representation.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_BUILD = "GRCh38"
CANONICAL_NAMESPACE = "grch38_cpg_cytosine_1based_v1"
UNIVERSE_SIZE = 408_399
UNIVERSE_REGISTRY = "data/cpg/registries/array_cpg_map.parquet"
DEFAULT_CHAIN = "data/raw/liftover/hg19ToHg38.over.chain.gz"
DEFAULT_FASTA = "data/external/reference/hg38.fa.gz"
# Plain-text copy of hg38.fa.gz (the .gz is plain gzip, not bgzip, so pyfaidx cannot index it) + hg38.fa.fai.
PLAIN_FASTA = "data/external/reference/hg38.fa"
GENOMIC_CONTEXT_SOURCE = "data/bio_annotations/genomic_context.parquet"
GENOMIC_CONTEXT_CANONICAL = "data/derived/bioval_v2/coordinates/genomic_context_canonical.parquet"
# Legacy "coordinate id" encoding (genomic_context.parquet, master_cpg_registry.parquet, known_sets/*):
#   cpg_idx = chrom_code * 10**9 + pos   (pos = GRCh38 1-based CpG cytosine; chrom_code 1..22, X=23, Y=24, M=25)
COORD_ID_BASE = 10**9
_CHROM_CODE_TO_NAME = {**{i: f"chr{i}" for i in range(1, 23)}, 23: "chrX", 24: "chrY", 25: "chrM"}
_CHROM_NAME_TO_CODE = {v: k for k, v in _CHROM_CODE_TO_NAME.items()}
CANONICAL_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

_BUILD_ALIASES = {"grch38": "GRCh38", "hg38": "GRCh38", "grch37": "GRCh37", "hg19": "GRCh37"}


def normalize_chrom(s):
    """Return canonical chrom name ('1'/'chr1'/'CHR1'/'23'->chrX, 'M'/'MT'->chrM). Unknown -> None."""
    if s is None:
        return None
    t = str(s).strip()
    if t.lower().startswith("chr"):
        t = t[3:]
    t = t.upper()
    if t == "23":
        t = "X"
    elif t == "24":
        t = "Y"
    elif t in ("M", "MT"):
        return "chrM"
    if re.fullmatch(r"([1-9]|1[0-9]|2[0-2])", t):
        return "chr" + t
    if t in ("X", "Y"):
        return "chr" + t
    return None


def _norm_build(build: str) -> str:
    try:
        return _BUILD_ALIASES[str(build).lower()]
    except KeyError:
        raise ValueError(f"unsupported genome build {build!r}") from None


def _make_converter(chain_path):
    from pyliftover import LiftOver

    return LiftOver(str(chain_path))


def _lift_array(lo, chroms, pos1):
    """Lift 1-based positions; returns (chrom, pos1, status) with status in ok/unmapped/ambiguous."""
    n = len(pos1)
    oc = np.empty(n, dtype=object)
    op = np.zeros(n, dtype=np.int64)
    st = np.empty(n, dtype=object)
    for i in range(n):
        r = lo.convert_coordinate(chroms[i], int(pos1[i]) - 1)
        if not r:
            st[i] = "unmapped"
        elif len(r) > 1:
            st[i] = "ambiguous"
        else:
            oc[i], op[i], st[i] = r[0][0], int(r[0][1]) + 1, "ok"
    return oc, op, st


def harmonize(df, chrom_col, pos_col, build, *, pos_base=1, end_col=None, liftover=True, chain_path=None,
              converter=None):
    """Bring a table to canonical coordinates. Returns (DataFrame[chrom,pos(,end)+orig cols], report).

    pos_base=0 means pos (and end, if given, being a half-open BED end) are 0-based; start is shifted +1
    (end of a half-open BED interval is already the 1-based inclusive end and is left unchanged).
    Steps: chrom normalization -> 0->1 base -> liftover (hg19 only) -> drop unmapped/ambiguous -> dedup.
    Interval rule: both ends lifted; dropped if ends land on different chroms, reversed, or the size
    changes by more than 2x. Dedup is deterministic (sorted, keep first). Counts are reported at each step.
    """
    b_in = _norm_build(build)
    rep = {"n_in": len(df), "build_in": b_in, "build_out": CANONICAL_BUILD, "pos_base_in": pos_base,
           "liftover_tool": None, "chain": None}
    d = df.copy().reset_index(drop=True)
    chroms = d[chrom_col].map(normalize_chrom)
    rep["n_bad_chrom"] = int(chroms.isna().sum())
    d["_chrom"] = chroms
    d["_pos"] = d[pos_col].astype(np.int64) + (1 if pos_base == 0 else 0)
    if end_col is not None:
        d["_end"] = d[end_col].astype(np.int64)
    d = d[d["_chrom"].notna()].reset_index(drop=True)
    rep["n_unmapped"] = 0
    rep["n_ambiguous"] = 0
    rep["n_interval_dropped"] = 0
    if b_in == "GRCh37":
        if not liftover:
            raise ValueError("GRCh37 input requires liftover=True")
        lo = converter
        if lo is None:
            chain = Path(chain_path or DEFAULT_CHAIN)
            lo = _make_converter(chain)
            rep["chain"] = str(chain)
        else:
            rep["chain"] = str(chain_path) if chain_path else "injected"
        try:
            import pyliftover

            rep["liftover_tool"] = f"pyliftover {getattr(pyliftover, '__version__', 'unknown')}"
        except ImportError:  # injected converter
            rep["liftover_tool"] = "injected"
        c1, p1, s1 = _lift_array(lo, d["_chrom"].to_numpy(), d["_pos"].to_numpy())
        ok = s1 == "ok"
        if end_col is not None:
            c2, p2, s2 = _lift_array(lo, d["_chrom"].to_numpy(), d["_end"].to_numpy())
            ok2 = s2 == "ok"
            both = ok & ok2
            rep["n_unmapped"] += int(((s1 == "unmapped") | (s2 == "unmapped")).sum())
            rep["n_ambiguous"] += int((((s1 == "ambiguous") | (s2 == "ambiguous")) & ~(
                (s1 == "unmapped") | (s2 == "unmapped"))).sum())
            same = both & (c1 == c2) & (p2 >= p1)
            old = (d["_end"].to_numpy() - d["_pos"].to_numpy() + 1).astype(float)
            new = (p2 - p1 + 1).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(old > 0, new / old, np.inf)
            sized = same & (ratio <= 2.0) & (ratio >= 0.5)
            rep["n_interval_dropped"] = int((both & ~sized).sum())
            keep = sized
            d["_end"] = p2
        else:
            rep["n_unmapped"] += int((s1 == "unmapped").sum())
            rep["n_ambiguous"] += int((s1 == "ambiguous").sum())
            keep = ok
        d["_chrom"] = c1
        d["_pos"] = p1
        d = d[keep].reset_index(drop=True)
        d["_chrom"] = d["_chrom"].map(normalize_chrom)
        d = d[d["_chrom"].notna()].reset_index(drop=True)
    out = d.drop(columns=[chrom_col, pos_col] + ([end_col] if end_col else []), errors="ignore")
    out.insert(0, "chrom", d["_chrom"].to_numpy())
    out.insert(1, "pos", d["_pos"].to_numpy().astype(np.int64))
    if end_col is not None:
        out.insert(2, "end", d["_end"].to_numpy().astype(np.int64))
    out = out.drop(columns=[c for c in ("_chrom", "_pos", "_end") if c in out.columns])
    keys = ["chrom", "pos"] + (["end"] if end_col else [])
    out = out.sort_values(keys, kind="mergesort").reset_index(drop=True)
    n_before = len(out)
    out = out.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)
    rep["n_dupe"] = int(n_before - len(out))
    rep["n_out"] = len(out)
    return out, rep


def verify_cpg_on_reference(df, fasta_path=DEFAULT_FASTA, sample=100_000, seed=0):
    """Fraction of (sampled) rows whose canonical pos is the C of 'CG' (+) or the G of 'CG' (-strand).

    Fraction ~1 for correct coordinates; ~0.1-0.25 suggests off-by-one / wrong build. Reference is read
    via pyfaidx (a .gz needs bgzip+faidx index; falls back to a plain-text copy if given).
    """
    from pyfaidx import Fasta

    d = df
    if len(d) > sample:
        d = d.sample(n=sample, random_state=seed)
    fa = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
    good = tot = 0
    for c, p in zip(d["chrom"].to_numpy(), d["pos"].to_numpy(), strict=True):
        if c not in fa:
            continue
        s = str(fa[c][int(p) - 1:int(p) + 1])  # bases at p and p+1 (1-based)
        prev = str(fa[c][int(p) - 2:int(p)]) if p > 1 else ""
        tot += 1
        good += int(s == "CG" or prev == "CG")
    return good / tot if tot else float("nan")


def encode_coord_id(chrom, pos):
    """(canonical chrom names, 1-based pos) -> legacy coordinate id ``chrom_code*1e9 + pos`` (int64)."""
    code = pd.Series(np.asarray(chrom, dtype=object)).map(_CHROM_NAME_TO_CODE)
    if code.isna().any():
        raise ValueError("encode_coord_id: unknown chromosome name(s)")
    return code.to_numpy(np.int64) * COORD_ID_BASE + np.asarray(pos, dtype=np.int64)


def decode_coord_id(ids):
    """Legacy coordinate id -> (chrom names array, pos int64 array). Unknown chrom codes -> None."""
    ids = np.asarray(ids, dtype=np.int64)
    code = ids // COORD_ID_BASE
    chrom = pd.Series(code).map(_CHROM_CODE_TO_NAME).to_numpy(dtype=object)
    chrom = np.where(pd.isna(chrom), None, chrom)
    return chrom, (ids % COORD_ID_BASE).astype(np.int64)


ISLAND_TRACK = "data/bio_annotations/cpgIslandExt.hg38.txt.gz"  # UCSC hg38 cpgIslandExt (0-based half-open BED)


def load_islands(path):
    """UCSC cpgIslandExt table -> DataFrame[chrom, start, end] restricted to canonical chroms (0-based start)."""
    t = pd.read_csv(path, sep="\t", header=None, usecols=[1, 2, 3], names=["chrom", "start", "end"])
    return t[t["chrom"].isin(CANONICAL_CHROMS)].reset_index(drop=True)


def classify_island_context(chrom, pos, islands):
    """island (pos inside an island) / shore (<=2 kb) / shelf (<=4 kb) / open_sea; pos is 1-based.
    Same rule as data/bio_annotations/README.md; agrees with the legacy table on 99.997% of CpGs
    (the remainder are exact-boundary conventions, 14 of 408,391)."""
    chrom = np.asarray(chrom, dtype=object)
    pos = np.asarray(pos, dtype=np.int64)
    out = np.empty(len(pos), dtype=object)
    for ch in pd.unique(chrom):
        m = chrom == ch
        g = islands[islands["chrom"] == ch].sort_values("start")
        st = g["start"].to_numpy() + 1  # 1-based inclusive start
        en = g["end"].to_numpy()
        p = pos[m]
        d = np.full(len(p), 10**12, dtype=np.int64)
        if len(st):
            i = np.searchsorted(st, p, side="right") - 1
            for k in (i, i + 1):
                kk = np.clip(k, 0, len(st) - 1)
                d = np.minimum(d, np.where(p < st[kk], st[kk] - p, np.where(p > en[kk], p - en[kk], 0)))
        out[m] = np.where(d == 0, "island", np.where(d <= 2000, "shore", np.where(d <= 4000, "shelf", "open_sea")))
    return out


def map_genomic_context(ctx, universe, *, id_col="cpg_idx", islands=None):
    """Map a legacy-encoded ``genomic_context`` table (cpg_idx, context) onto the benchmark universe.

    The legacy id decodes to (chrom, pos) in GRCh38 1-based CpG-cytosine coordinates; rows are joined to the
    universe by exact (chrom, pos) (never by the numeric id). Returns (canonical DataFrame with columns
    cpg_idx [universe id], chrom, pos, context, source_cpg_idx, mapping_method; one row per universe CpG
    that has a context label; audit dict). Universe CpGs without a label are simply absent (reported).
    """
    chrom, pos = decode_coord_id(ctx[id_col].to_numpy())
    src = pd.DataFrame({"source_cpg_idx": ctx[id_col].to_numpy(np.int64), "chrom": chrom, "pos": pos,
                        "context": ctx["context"].to_numpy()})
    n_bad = int(src["chrom"].isna().sum())
    src = src[src["chrom"].notna()]
    n_src_dupe = int(src.duplicated(["chrom", "pos"]).sum())
    src = src.drop_duplicates(["chrom", "pos"], keep="first")
    u = universe[["cpg_idx", "chrom", "pos"]]
    n_u_dupe = int(u.duplicated(["chrom", "pos"]).sum())
    out = u.merge(src, on=["chrom", "pos"], how="left")
    out["mapping_method"] = "decode_chrom_pos_exact"
    miss = out["context"].isna()
    n_miss = int(miss.sum())
    n_filled = 0
    if islands is not None and n_miss:
        out.loc[miss, "context"] = classify_island_context(out.loc[miss, "chrom"], out.loc[miss, "pos"], islands)
        out.loc[miss, "mapping_method"] = "ucsc_cpgIslandExt_rule_recomputed"
        n_filled = n_miss
    out = out[out["context"].notna()]
    out["source_cpg_idx"] = out["source_cpg_idx"].astype("Int64")
    out = out[["cpg_idx", "chrom", "pos", "context", "source_cpg_idx", "mapping_method"]]
    out = out.sort_values("cpg_idx").reset_index(drop=True)
    audit = {
        "n_source": len(ctx), "n_source_bad_chrom": n_bad, "n_source_dup_chrom_pos": n_src_dupe,
        "n_universe": len(u), "n_universe_dup_chrom_pos": n_u_dupe,
        "n_universe_mapped": int(out["cpg_idx"].nunique()), "n_exact_match_missing": n_miss,
        "n_recomputed_from_island_track": n_filled,
        "frac_universe_mapped": float(out["cpg_idx"].nunique() / max(1, len(u))),
        "n_universe_unmapped": int(len(u) - out["cpg_idx"].nunique()),
        "n_output_rows": len(out), "output_cpg_idx_unique": bool(out["cpg_idx"].is_unique),
        "n_source_not_in_universe": int(len(src) - (len(out) - n_filled)),
    }
    return out, audit


def load_genomic_context_canonical(root=None, path=None) -> pd.DataFrame:
    """Canonical genomic context: DataFrame[cpg_idx (universe id), chrom, pos, context, source_cpg_idx,
    mapping_method]. Reads the materialized parquet; if absent, builds it in memory from the legacy table."""
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    p = Path(path) if path else root / GENOMIC_CONTEXT_CANONICAL
    if p.exists():
        return pd.read_parquet(p)
    out, _ = map_genomic_context(pd.read_parquet(root / GENOMIC_CONTEXT_SOURCE), load_benchmark_universe(root))
    return out


def cg_consistency(df, fasta_path, *, sample=None, seed=0):
    """CG rate at offsets {-1,0,+1} for (chrom,pos): fraction where bases (pos+k, pos+k+1) == 'CG'.

    Correct 1-based cytosine coordinates -> offset 0 ~ 1 (plus the minus-strand case 'G of CG' = offset -1
    reported separately); offsets +-1 should be ~ background (<0.1)."""
    from pyfaidx import Fasta

    d = df if sample is None or len(df) <= sample else df.sample(n=sample, random_state=seed)
    fa = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
    res = {k: 0 for k in (-1, 0, 1)}
    tot = 0
    for c, g in d.groupby("chrom"):
        if c not in fa:
            continue
        seq = str(fa[c][:])
        arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        p0 = g["pos"].to_numpy() - 1  # 0-based index of the C
        tot += len(p0)
        for k in res:
            i = p0 + k
            ok = (i >= 0) & (i + 1 < len(arr))
            res[k] += int(((arr[np.where(ok, i, 0)] == ord("C")) & (arr[np.where(ok, i + 1, 0)] == ord("G")) & ok).sum())
    return {f"cg_rate_offset_{k:+d}": (v / tot if tot else float("nan")) for k, v in res.items()} | {"n_checked": tot}


def load_benchmark_universe(root=None) -> pd.DataFrame:
    """The fixed 408,399 benchmark CpGs: DataFrame[cpg_idx, chrom, pos, raw_cpg_row] (canonical coords)."""
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    a = pd.read_parquet(root / UNIVERSE_REGISTRY)
    u = pd.DataFrame({
        "cpg_idx": a["cpg_idx"].astype(np.int64),
        "chrom": a["chr"].map(normalize_chrom),
        "pos": a["pos"].astype(np.int64),
        "raw_cpg_row": a["raw_cpg_row"].astype(np.int64),
    })
    assert len(u) == UNIVERSE_SIZE, f"universe has {len(u)} rows, expected {UNIVERSE_SIZE}"
    assert u["cpg_idx"].is_unique and u["chrom"].notna().all()
    return u.sort_values("cpg_idx").reset_index(drop=True)


def overlap_audit(df, universe=None, *, tol_bp=0) -> dict:
    """Coverage of a canonical table (chrom,pos) against the universe. Never builds an intersection set
    for downstream use; reports counts only."""
    u = load_benchmark_universe() if universe is None else universe
    n_match = 0
    in_u = np.zeros(len(df), dtype=bool)
    u_cov = np.zeros(len(u), dtype=bool)
    per = []
    uc = u["chrom"].to_numpy()
    up = u["pos"].to_numpy()
    dc = df["chrom"].to_numpy()
    dp = df["pos"].to_numpy()
    for c in sorted(set(uc) | set(dc)):
        um = np.where(uc == c)[0]
        dm = np.where(dc == c)[0]
        order = np.argsort(up[um])
        us = up[um][order]
        if len(us) and len(dm):
            i = np.searchsorted(us, dp[dm])
            lo = np.clip(i - 1, 0, len(us) - 1)
            hi = np.clip(i, 0, len(us) - 1)
            dlo = np.abs(us[lo] - dp[dm])
            dhi = np.abs(us[hi] - dp[dm])
            hit = np.minimum(dlo, dhi) <= tol_bp
            in_u[dm] = hit
            # universe coverage
            ds = np.sort(dp[dm])
            j = np.searchsorted(ds, us)
            l2 = np.clip(j - 1, 0, len(ds) - 1)
            h2 = np.clip(j, 0, len(ds) - 1)
            cov = np.minimum(np.abs(ds[l2] - us), np.abs(ds[h2] - us)) <= tol_bp
            u_cov[um[order]] = cov
        per.append({"chrom": c, "n_df": len(dm), "n_universe": len(um),
                    "n_df_in_universe": int(in_u[dm].sum()),
                    "n_universe_covered": int(u_cov[um].sum())})
    n_match = int(in_u.sum())
    return {
        "n_df": len(df), "n_universe": len(u), "tol_bp": tol_bp,
        "n_exact_matches": n_match,
        "frac_universe_covered": float(u_cov.mean()) if len(u) else float("nan"),
        "frac_df_in_universe": float(in_u.mean()) if len(df) else float("nan"),
        "per_chrom": pd.DataFrame(per),
    }


def require_full_universe_coverage(representation_index, universe):
    """Raise unless every universe cpg_idx is covered. Never narrows the universe."""
    ids = universe["cpg_idx"].to_numpy() if hasattr(universe, "columns") else np.asarray(universe)
    missing = np.setdiff1d(ids, np.asarray(representation_index))
    if len(missing):
        raise ValueError(
            f"representation covers {len(ids) - len(missing)}/{len(ids)} universe CpGs "
            f"({len(missing)} missing, e.g. {missing[:5].tolist()}); refusing to narrow the universe")
    return True
