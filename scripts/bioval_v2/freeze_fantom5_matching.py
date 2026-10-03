# ruff: noqa: F841  (one-shot data-prep script; style-only, behaviour unchanged)
"""Freeze the FANTOM5 enhancer-membership matched design and the activity-similarity pair sample.

Produced BEFORE any embedding metric; uses only non-embedding covariates + FANTOM5 annotations.
Pre-declared policy (see docs/bioval_v2_prep/regulatory_activity_PREP.md section 6):
  positives  : CpGs inside a FANTOM5 hg38 enhancer interval (in_enhancer), all 12,551 (window sensitivity: |pos-mid|<=500).
  pool       : bg_enh5k (>=5 kb from any enhancer). bg_enh5k_tss5k is a sensitivity list (see PREP).
  matching   : exact chrom x context; numeric calipers 0.25 SD (SD over the full 408,399-CpG universe) on
               cpg_density, cpg_density_hg38, gc_content, tss_dist; 1:1 without replacement; seed 17;
               distance to nearest enhancer is NOT matched (controls >=5 kb by construction).
  folds      : splits.chromosome_blocked_folds(n_folds=5, seed=17) over universe chromosomes.
  pairs      : informative (enhancer expressed TPM>=1 in >=5 libraries) in-enhancer CpGs; 100k intra (equal over
               distance bins) + 100k inter; same-enhancer pairs and NaN-similarity pairs dropped; seed 17.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/bioval_v2"))
from prepare_fantom5 import enhancer_profile_similarity

from cpg_repr_benchmark.biological_validation_v2.matching import match_controls
from cpg_repr_benchmark.biological_validation_v2.pairs import sample_pairs
from cpg_repr_benchmark.biological_validation_v2.splits import chromosome_blocked_folds

OUT = ROOT / "data/derived/bioval_v2/regulatory_activity"
COV = ROOT / "data/derived/bioval_v2/covariates/universe_covariates.parquet"
SEED = 17
CAL_SD = 0.25
NUM = ["cpg_density", "cpg_density_hg38", "gc_content", "tss_dist"]
EXACT = ["chrom", "context"]
N_PAIRS_INTRA, N_PAIRS_INTER = 100_000, 100_000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def smd_table(pos, ctl):
    rows = []
    for c in NUM + ["probe_unknown", "probe_I"] + [f"ctx_{k}" for k in ["island", "shore", "shelf", "open_sea"]]:
        a, b = pos[c].to_numpy(float), ctl[c].to_numpy(float)
        sd = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"covariate": c, "smd": float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0,
                     "mean_pos": float(a.mean()), "mean_ctl": float(b.mean())})
    return pd.DataFrame(rows)


def run_design(name, d, pos_mask, pool_mask, folds, sd):
    cases, pool = d[pos_mask], d[pool_mask]
    cal = {c: CAL_SD * sd[c] for c in NUM}
    mp, _ = match_controls(cases, pool, exact=EXACT, numeric=cal, seed=SEED)
    kept = d.set_index("cpg_idx")
    P = kept.loc[mp["case"]].reset_index()
    C = kept.loc[mp["control"]].reset_index()
    before = smd_table(cases, pool)
    after = smd_table(P, C)
    bal = before.merge(after, on="covariate", suffixes=("_before", "_after"))[
        ["covariate", "smd_before", "smd_after", "mean_pos_after", "mean_ctl_after"]]
    cols = ["cpg_idx", "chrom", "pos", "context", *NUM, "probe_type"]
    P = P.assign(label=1, matched_to=mp["control"].to_numpy(), enhancer_id=P["enhancer_id"])
    C = C.assign(label=0, matched_to=mp["case"].to_numpy(), enhancer_id=None)
    out = pd.concat([P, C], ignore_index=True)
    out["fold"] = out["chrom"].map(folds).astype(int)
    out = out[["cpg_idx", "chrom", "pos", "label", "matched_to", "enhancer_id", "fold",
               "context", *NUM, "probe_type"]]
    out = out.sort_values(["label", "cpg_idx"], ascending=[False, True]).reset_index(drop=True)
    dropped = cases[~cases["cpg_idx"].isin(mp["case"])]
    info = {"n_positives_total": len(cases), "n_pool": len(pool), "n_matched_pairs": len(mp),
            "n_dropped_positives": len(dropped), "drop_rate": float(len(dropped) / len(cases)),
            "max_abs_smd_after_numeric": float(bal[bal.covariate.isin(NUM)].smd_after.abs().max()),
            "max_abs_smd_after_all": float(bal.smd_after.abs().max()),
            "balance": bal.round(5).to_dict("records")}
    return out, dropped[["cpg_idx", "chrom", "pos"]], info


def build(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    cov = pd.read_parquet(COV)
    m = pd.read_parquet(OUT / "cpg_enhancer_map.parquet")[
        ["cpg_idx", "enhancer_id", "in_enhancer", "in_window500", "bg_enh5k", "bg_enh5k_tss5k"]]
    d = cov.merge(m, on="cpg_idx", how="inner", validate="1:1").sort_values("cpg_idx").reset_index(drop=True)
    assert len(d) == len(cov) == 408399
    d["probe_unknown"] = d["probe_type"].isna().astype(float)  # categorical 'unknown' level, reporting only
    d["probe_I"] = (d["probe_type"] == "I").astype(float)
    for k in ["island", "shore", "shelf", "open_sea"]:
        d[f"ctx_{k}"] = (d["context"] == k).astype(float)
    sd = {c: float(d[c].std()) for c in NUM}
    folds = chromosome_blocked_folds(d["chrom"].unique(), 5, SEED)
    res, summary = {}, {}
    designs = {
        "frozen_enhancer_membership_seed17": (d.in_enhancer, d.bg_enh5k),
        "frozen_enhancer_membership_win500_seed17": (d.in_window500, d.bg_enh5k),
        "frozen_enhancer_membership_tss5kpool_seed17": (d.in_enhancer, d.bg_enh5k_tss5k),
    }
    for name, (pm, bm) in designs.items():
        out, dropped, info = run_design(name, d, pm.to_numpy(), bm.to_numpy(), folds, sd)
        out.to_parquet(outdir / f"{name}.parquet", index=False)
        dropped.sort_values("cpg_idx").to_parquet(outdir / f"{name}_dropped_positives.parquet", index=False)
        summary[name] = info
    # ---- activity-similarity pair sample
    enh = pd.read_parquet(OUT / "enhancers.parquet")[["enhancer_id", "n_samples_expressed"]]
    inf = d[d.in_enhancer].merge(enh, on="enhancer_id")
    inf = inf[inf.n_samples_expressed >= 5][["cpg_idx", "chrom", "pos", "enhancer_id"]].reset_index(drop=True)
    pi = sample_pairs(inf, N_PAIRS_INTRA, kind="intra", seed=SEED)
    pe = sample_pairs(inf, N_PAIRS_INTER, kind="inter", seed=SEED)
    pr = pd.concat([pi, pe], ignore_index=True)
    n_sampled = len(pr)
    emap = inf.set_index("cpg_idx")["enhancer_id"]
    pr["enh_i"], pr["enh_j"] = pr["i"].map(emap), pr["j"].map(emap)
    same = pr["enh_i"] == pr["enh_j"]
    pr = pr[~same].reset_index(drop=True)
    pr["profile_sim"] = enhancer_profile_similarity(pr[["i", "j"]].to_numpy(), activity_dir=OUT)
    n_after_same = len(pr)
    pr = pr[pr["profile_sim"].notna()].reset_index(drop=True)
    folds_inv = folds
    pr["fold_i"] = pr["chrom_i"].map(folds_inv).astype(int)
    pr["fold_j"] = pr["chrom_j"].map(folds_inv).astype(int)
    pr = pr.sort_values(["kind", "i", "j"]).reset_index(drop=True)
    pr.to_parquet(outdir / "frozen_activity_pairs_seed17.parquet", index=False)
    summary["pairs"] = {
        "n_informative_cpgs": len(inf), "n_informative_enhancers": int(inf.enhancer_id.nunique()),
        "n_sampled": int(n_sampled), "n_same_enhancer_dropped": int(same.sum()),
        "n_after_same_enhancer_filter": int(n_after_same), "n_final_pairs": len(pr),
        "n_nan_similarity_dropped": int(n_after_same - len(pr)),
        "by_kind_bin": {f"{k}|{b}": int(v) for (k, b), v in pr.groupby(["kind", "bin"]).size().items()},
        "sim_mean": float(pr.profile_sim.mean()), "sim_sd": float(pr.profile_sim.std()),
        "n_intra_sampled_by_bin": {int(k): int(v) for k, v in pi.groupby("bin").size().items()},
    }
    summary["folds"] = folds
    summary["calipers_raw"] = {c: CAL_SD * sd[c] for c in NUM}
    summary["universe_sd"] = sd
    return summary


def main():
    s1 = build(OUT / "_run1")
    s2 = build(OUT / "_run2")
    names = ["frozen_enhancer_membership_seed17", "frozen_enhancer_membership_win500_seed17",
             "frozen_enhancer_membership_tss5kpool_seed17", "frozen_activity_pairs_seed17"]
    names += [n + "_dropped_positives" for n in names[:3]]
    shas = {}
    for n in names:
        a, b = sha(OUT / "_run1" / f"{n}.parquet"), sha(OUT / "_run2" / f"{n}.parquet")
        # parquet bytes can embed nothing non-deterministic here, but also compare content hash
        da = pd.read_parquet(OUT / "_run1" / f"{n}.parquet")
        db = pd.read_parquet(OUT / "_run2" / f"{n}.parquet")
        assert da.equals(db), n
        shas[n] = {"sha256_run1": a, "sha256_run2": b, "identical": a == b}
        assert a == b, f"non-deterministic: {n}"
    for n in names:
        (OUT / "_run1" / f"{n}.parquet").replace(OUT / f"{n}.parquet")
    import shutil
    shutil.rmtree(OUT / "_run1", ignore_errors=True)
    shutil.rmtree(OUT / "_run2", ignore_errors=True)
    s1["file_sha256"] = {n: shas[n]["sha256_run1"] for n in names}
    s1["determinism"] = shas
    s1["policy"] = {
        "seed": SEED, "caliper_sd_multiplier": CAL_SD, "numeric_matched": NUM, "exact": EXACT,
        "n_per_case": 1, "replace": False, "main_pool": "bg_enh5k",
        "sensitivity_pools": ["bg_enh5k_tss5k (list ..._tss5kpool_seed17)"],
        "sensitivity_positives": "in_window500 (|cytosine-enhancer midpoint|<=500 bp)",
        "probe_type": "not matched (6,429 NaN kept as 'unknown' level; balance reported only)",
        "dist_to_enhancer": "not matched (controls >=5 kb by construction)",
        "folds": "chromosome_blocked_folds(n_folds=5, seed=17)",
    }
    s1["statement"] = "produced before any embedding metric; no embedding/representation result was read or used"
    (OUT / "FROZEN.json").write_text(json.dumps(s1, indent=2, default=str))
    print(json.dumps({k: v for k, v in s1.items() if k in ("file_sha256",)}, indent=1))
    for k in list(s1)[:3]:
        v = s1[k]
        print(k, {x: v[x] for x in v if x != "balance"})
        print(pd.DataFrame(v["balance"]).to_string())
    print(s1["pairs"])


if __name__ == "__main__":
    main()
