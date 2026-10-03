# ruff: noqa: SIM115  (one-shot data-prep script; style-only, behaviour unchanged)
"""Freeze the seed-17 CpG pair list + Loyfer profile similarity BEFORE any embedding is seen.
Intra-chromosomal: 40,000 pairs per distance stratum (<1kb,1-10kb,10-100kb,100kb-1Mb,>1Mb) via loyfer_profile_similarity.sample_pairs(seed=17);
inter-chromosomal: 200,000. Pairs stored as cpg_idx (cpg_a < cpg_b), duplicates (unordered) dropped; only pairs with n_shared >= 20 groups kept.
Output: frozen_pairs_seed17.parquet (+ frozen_pairs_seed17_counts.tsv); sha256 recorded in MANIFEST.json by loyfer_manifest.py."""
import hashlib
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import loyfer_profile_similarity as lps
from loyfer_common import OUT


def main():
    prof = lps.Profiles(); (ia, ib), (ja, jb) = lps.sample_pairs(prof, seed=17, n_intra=200_000, n_inter=200_000)
    a = np.concatenate([ia, ja]); b = np.concatenate([ib, jb])
    ca, cb = prof.cpg[a], prof.cpg[b]
    lo, hi = np.minimum(ca, cb), np.maximum(ca, cb)
    d = lps.profile_similarity(lo, hi, min_shared=20, prof=prof).rename(columns={"cpg_a": "cpg_i", "cpg_b": "cpg_j"})
    d["draw_order"] = np.arange(len(d))
    n_gen = d.groupby("stratum").size().reindex(lps.STRATA)
    d = d[d.cpg_i != d.cpg_j]; dd = d.drop_duplicates(["cpg_i", "cpg_j"]); n_dedup = dd.groupby("stratum").size().reindex(lps.STRATA)
    keep = dd[dd.n_shared >= 20].copy()
    keep["pearson_defined"] = keep.pearson.notna()
    keep = keep.rename(columns={"n_shared": "n_shared_groups", "pearson": "loyfer_pearson", "spearman": "loyfer_spearman"})
    keep = keep[["cpg_i", "cpg_j", "stratum", "distance_bp", "n_shared_groups", "loyfer_pearson", "loyfer_spearman", "pearson_defined", "draw_order"]]
    keep["cpg_i"] = keep.cpg_i.astype(np.int64); keep["cpg_j"] = keep.cpg_j.astype(np.int64)
    keep.to_parquet(OUT / "frozen_pairs_seed17.parquet", index=False)
    cnt = pd.DataFrame({"generated": n_gen, "after_dedup": n_dedup, "survive_ge20_shared": keep.groupby("stratum").size().reindex(lps.STRATA),
                            "survive_and_pearson_defined": keep[keep.pearson_defined].groupby("stratum").size().reindex(lps.STRATA)})
    cnt["frac_survive"] = cnt.survive_ge20_shared / cnt.generated
    cnt.to_csv(OUT / "frozen_pairs_seed17_counts.tsv", sep="\t")
    sha = hashlib.sha256(open(OUT / "frozen_pairs_seed17.parquet", "rb").read()).hexdigest()
    json.dump({"sha256": sha, "n_pairs": len(keep)}, open(OUT / "frozen_pairs_seed17_sha256.json", "w"))
    print(cnt.to_string()); print(sha, len(keep), "i<j:", bool((keep.cpg_i < keep.cpg_j).all()))
if __name__ == "__main__":
    main()
