"""GC fraction per 10 kb bin from data/external/reference/hg38.fa.gz (stream, CPU). Output parquet chrom,bin,gc_count,acgt_count,n_count."""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BIN = 10000
lut = np.zeros(256, dtype=np.uint8)  # 1=GC,2=AT,3=N-or-other
lut[:] = 3
for c in b"GCgc": lut[c] = 1
for c in b"ATat": lut[c] = 2
rows = []
def flush(name, buf):
    if name is None or not buf: return
    a = lut[np.frombuffer(b"".join(buf), dtype=np.uint8)]
    n = len(a); nb = -(-n // BIN)
    idx = np.arange(n) // BIN
    gc = np.bincount(idx, weights=(a == 1), minlength=nb); at = np.bincount(idx, weights=(a == 2), minlength=nb)
    nn = np.bincount(idx, weights=(a == 3), minlength=nb)
    rows.append(pd.DataFrame({"chrom": name, "bin": np.arange(nb), "gc": gc.astype(np.int32), "at": at.astype(np.int32), "other": nn.astype(np.int32)}))
name, buf = None, []
with gzip.open(ROOT / "data/external/reference/hg38.fa.gz", "rb") as f:
    for line in f:
        if line.startswith(b">"):
            flush(name, buf); name = line[1:].split()[0].decode(); buf = []
            if not (name.startswith("chr") and "_" not in name): name = "__skip__"
            if name == "__skip__": name = None
            continue
        if name: buf.append(line.strip())
flush(name, buf)
df = pd.concat(rows); df.to_parquet(ROOT / "data/derived/bioval_v2/3d_genome/hg38_gc_10kb.parquet", index=False)
print(df.groupby("chrom").size().head(30))
