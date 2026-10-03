# ruff: noqa: SIM115  (one-shot data-prep script; style-only, behaviour unchanged)
"""Step 1-2 driver: inventory + per-sample counts at the benchmark universe + per-cell-type matrices.
Usage: python prepare_loyfer.py extract|aggregate   (aggregate is in loyfer_aggregate.py, called from here)"""
import json
import sys
from multiprocessing import Pool

import h5py
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import loyfer_extract as ex
from loyfer_common import *


def extract(workers=16):
    u = load_universe(); g = universe_to_global(u)
    assert (g >= 0).all()
    inv = inventory(); inv.to_csv(OUT / "sample_inventory.tsv", sep="\t", index=False)
    jobs = [(PAT_DIR / f, g) for f in inv.file]
    M = np.zeros((len(inv), len(u)), np.uint16); C = np.zeros_like(M); stats = {}; clipped = 0
    pos = {f: i for i, f in enumerate(inv.file)}
    with Pool(workers) as p:
        for k, (name, m, c, st) in enumerate(p.imap_unordered(ex.work, jobs)):
            i = pos[name]; clipped += int((c > 65535).sum())
            M[i] = np.minimum(m, 65535); C[i] = np.minimum(c, 65535); stats[name] = st
            print(k, name, st, flush=True)
    with h5py.File(OUT / "loyfer_sample_counts.h5", "w") as h:
        h["cpg_idx"] = u.cpg_idx.to_numpy(np.int64); h["chrom"] = u.chrom.to_numpy("S5"); h["pos"] = u.pos.to_numpy(np.int64)
        h["meth_count"] = M; h["cov"] = C; h["sample_file"] = inv.file.to_numpy("S100")
        h.attrs["clipped_cells_cov_gt65535"] = clipped
    json.dump(stats, open(OUT / "extract_stats.json", "w"), indent=1)

if __name__ == "__main__":
    {"extract": extract}[sys.argv[1]]()
