"""Build the wgbstools-style hg38 CpG index (global 1-based index -> chrom, 1-based C coordinate)
directly from data/external/reference/hg38.fa.gz. Order: chr1..22, X, Y, M (as observed in the pat files).
Output: <out>/hg38_CpG_index.npz (chrom_id uint8, pos int32; row i <-> CpG index i+1) and chrom offsets."""
import gzip
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FA = ROOT / "data/external/reference/hg38.fa.gz"
OUT = ROOT / "data/derived/bioval_v2/external_methylation_programs/hg38_CpG_index.npz"
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]

def read_main_chroms():
    seqs, name, buf = {}, None, []
    with gzip.open(FA, "rt") as fh:
        for line in fh:
            if line[0] == ">":
                if name in CHROMS:
                    seqs[name] = "".join(buf)
                name = line[1:].strip().split()[0]; buf = []
            elif name in CHROMS:
                buf.append(line.strip())
        if name in CHROMS:
            seqs[name] = "".join(buf)
    return seqs

def main():
    seqs = read_main_chroms()
    cid, pos = [], []
    for ci, c in enumerate(CHROMS):
        a = np.frombuffer(seqs[c].upper().encode(), dtype=np.uint8)
        p = np.flatnonzero((a[:-1] == ord("C")) & (a[1:] == ord("G")))
        cid.append(np.full(len(p), ci, np.uint8)); pos.append((p + 1).astype(np.int32))  # 1-based C
        print(c, len(a), len(p), flush=True)
    cid = np.concatenate(cid); pos = np.concatenate(pos)
    np.savez_compressed(OUT, chrom_id=cid, pos=pos, chroms=np.array(CHROMS))
    print("total CpG", len(pos))
if __name__ == "__main__":
    main()
