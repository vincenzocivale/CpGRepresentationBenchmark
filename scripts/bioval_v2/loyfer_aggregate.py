# ruff: noqa: I001, SIM115  (one-shot data-prep script; style-only, behaviour unchanged)
"""Per-sample beta + per-cell-type (group) beta matrices on the full 408,399 universe.
Rules (pre-declared): sample-CpG valid iff cov>=10. Donor = published S1 PatientID (sample_inventory.patient_id; v1 used sample ids).
Donor beta = mean of that donor's valid sample betas; group beta = unweighted mean over donors (donor-level mean, so a deep donor or
a donor with many samples does not dominate). beta=NaN if no valid donor. mask_primary = n_valid_donors >= min(2, n_donors_in_group);
mask_lenient = n_valid_donors >= 1. Only atlas samples with a PUBLISHED group assignment (sample_provenance.csv; 205) enter group means."""
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from loyfer_common import *

MINCOV = 10

def main():
    inv = pd.read_csv(OUT / "sample_inventory.tsv", sep="\t")
    with h5py.File(OUT / "loyfer_sample_counts.h5") as h:
        cpg = h["cpg_idx"][:]; chrom = h["chrom"][:]; pos = h["pos"][:]
        M = h["meth_count"][:].astype(np.float32); C = h["cov"][:].astype(np.float32); files = [x.decode() for x in h["sample_file"][:]]
    assert files == list(inv.file)
    valid = C >= MINCOV
    B = np.where(valid, M / np.maximum(C, 1), np.nan).astype(np.float32)
    atlas = inv[(inv.kind == "atlas") & inv.group.notna()]
    groups = sorted(atlas.group.unique())
    G = len(groups); N = len(cpg)
    beta = np.full((N, G), np.nan, np.float32); nval = np.zeros((N, G), np.uint8); mcov = np.zeros((N, G), np.uint16)
    nsamp = []; nsmp = []; nvsamp = np.zeros((N, G), np.uint8)
    for gi, g in enumerate(groups):
        ix = atlas.index[atlas.group == g].to_numpy()
        pats = atlas.loc[ix, "patient_id"].astype(str).to_numpy(); up = sorted(set(pats)); nsamp.append(len(up)); nsmp.append(len(ix))
        nv = np.zeros(N, np.int32); s = np.zeros(N, np.float32); nvs = np.zeros(N, np.int32)
        for p in up:
            jx = ix[pats == p]; v = valid[jx]; k = v.sum(0)
            dv = np.where(k > 0, np.where(v, B[jx], 0).sum(0) / np.maximum(k, 1), 0)
            nv += (k > 0); s += dv; nvs += k
        beta[:, gi] = np.where(nv > 0, s / np.maximum(nv, 1), np.nan)
        nval[:, gi] = nv; nvsamp[:, gi] = nvs; mcov[:, gi] = np.minimum(C[ix].mean(0), 65535)
    nsamp = np.array(nsamp)
    mask_primary = nval >= np.minimum(2, nsamp)[None, :]
    mask_lenient = nval >= 1
    with h5py.File(OUT / "loyfer_celltype_beta.h5", "w") as h:
        h["cpg_idx"] = cpg; h["chrom"] = chrom; h["pos"] = pos
        h["groups"] = np.array(groups, "S40"); h["n_donors_per_group"] = nsamp; h["n_samples_per_group"] = np.array(nsmp)
        h.create_dataset("n_valid_samples", data=nvsamp, compression="gzip")
        h.create_dataset("beta", data=beta.astype(np.float16), compression="gzip", chunks=(8192, G))
        h.create_dataset("n_valid_donors", data=nval, compression="gzip")
        h.create_dataset("mean_cov", data=mcov, compression="gzip")
        h.create_dataset("mask_primary", data=mask_primary, compression="gzip")
        h.create_dataset("mask_lenient", data=mask_lenient, compression="gzip")
        h.create_dataset("sample_beta", data=B.astype(np.float16), compression="gzip", chunks=(8, N))
        h.create_dataset("sample_cov", data=C.astype(np.uint16), compression="gzip", chunks=(8, N))
        h["sample_file"] = np.array(files, "S100")
        h.attrs["min_cov_per_sample_cpg"] = MINCOV
        h.attrs["rule"] = "donor=published PatientID; donor beta = mean of its valid (cov>=10) sample betas; group beta = mean over donors; mask_primary: n_valid_donors>=min(2,n_donors); mask_lenient: n_valid_donors>=1"
        h.attrs["build"] = "GRCh38; 1-based cytosine of CpG"
    # coverage audit
    rows = []
    for gi, g in enumerate(groups):
        rows.append({"group": g, "n_samples": int(nsmp[gi]), "n_donors": int(nsamp[gi]), "frac_lenient": mask_lenient[:, gi].mean(), "frac_primary": mask_primary[:, gi].mean(),
                         "median_mean_cov": float(np.median(mcov[:, gi]))})
    a = pd.DataFrame(rows)
    allp = mask_primary.all(1); allp_l = mask_lenient.all(1)
    s = valid[atlas.index.to_numpy()]
    summ = {"n_universe": N, "n_groups": G, "n_atlas_samples_used": len(atlas), "n_donors_used": int(sum(nsamp)),
                "frac_cpg_all_groups_primary": float(allp.mean()), "frac_cpg_all_groups_lenient": float(allp_l.mean()),
                "frac_cpg_ge20_groups_primary": float((mask_primary.sum(1) >= 20).mean()),
                "frac_cpg_ge_1_group_lenient": float(mask_lenient.any(1).mean()),
                "median_groups_per_cpg_primary": float(np.median(mask_primary.sum(1))),
                "frac_sample_cpg_valid_mean": float(s.mean()), "cov_median_per_sample": float(np.median(np.median(C[atlas.index.to_numpy()], 1)))}
    a.to_csv(OUT / "celltype_coverage_audit.tsv", sep="\t", index=False)
    import json; json.dump(summ, open(OUT / "coverage_summary.json", "w"), indent=1)
    print(a.to_string()); print(summ)

if __name__ == "__main__":
    main()
