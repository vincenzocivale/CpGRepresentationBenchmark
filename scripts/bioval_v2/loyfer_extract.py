"""Stream .hg38.pat.gz -> per-sample (meth, cov) counts at the 408,399 benchmark CpGs. <=16 workers.
Pat line: chr, global 1-based wgbstools CpG index of first CpG, string over {C,T,.}, count."""
import gzip
import sys
import time

import numba as nb
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from loyfer_common import *


@nb.njit(cache=True)
def parse(buf, meth, cov):
    n = buf.shape[0]; i = 0; nlines = 0
    while i < n:
        while i < n and buf[i] != 9: i += 1          # chrom
        i += 1; s = 0
        while buf[i] != 9: s = s * 10 + (buf[i] - 48); i += 1
        i += 1; st = i
        while buf[i] != 9: i += 1
        en = i; i += 1; c = 0
        while i < n and buf[i] != 10: c = c * 10 + (buf[i] - 48); i += 1
        i += 1; nlines += 1
        for j in range(st, en):
            ch = buf[j]; k = s - 1 + (j - st)
            if ch == 67: meth[k] += c; cov[k] += c
            elif ch == 84: cov[k] += c
    return nlines

def work(args):
    f, uidx = args
    t = time.time()
    G = 29401795
    meth = np.zeros(G, np.uint32); cov = np.zeros(G, np.uint32)
    tot = 0
    with gzip.open(f, "rb") as fh:
        carry = b""
        while True:
            chunk = fh.read(1 << 28)
            if not chunk:
                if carry: tot += parse(np.frombuffer(carry, np.uint8), meth, cov)
                break
            chunk = carry + chunk
            k = chunk.rfind(b"\n")
            carry = chunk[k + 1:]
            tot += parse(np.frombuffer(chunk[:k + 1], np.uint8), meth, cov)
    m = meth[uidx]; c = cov[uidx]
    return f.name, m, c, {"pat_lines": int(tot), "genome_cov_sum": int(cov.sum(dtype=np.uint64)),
                              "genome_cpg_cov_ge1": int((cov > 0).sum()), "seconds": time.time() - t}
