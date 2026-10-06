#!/usr/bin/env python3
"""POST-HOC AlphaGenome arm for the EWAS-v3 main panel (outside the approved 3-arm scope).

Reuses the frozen pairs, probe contract and bootstrap draws of run_experiments.py but writes ONLY to
outputs/ewas_v3_posthoc_alphagenome/. Frozen artifacts (outputs/ewas_v3/<tag>, data/registry/ewas_v3_results) are read-only.
Scope: primary threshold, main matching design (the 13 tasks of Fig 3).
"""
from __future__ import annotations
import concurrent.futures, hashlib, json, multiprocessing, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ewas_v3"))
import numpy as np, pandas as pd, yaml
import run_experiments as R
from cpg_repr_benchmark.bioval_v2_launch.embedding_eval import ArmSpec, EmbeddingStore
from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import BlockBootstrap, RankPlan, block_index, weighted_auroc, weighted_mean, percentile_ci, contrast

ARM = "alphagenome_locus"
FROZEN = R.OUT
OUT = ROOT / "outputs/ewas_v3_posthoc_alphagenome"
R.OUT, R.ARMS = OUT, (ARM,)  # endpoint_dir() reads the module global -> new tree; fork keeps it in workers
assert OUT != FROZEN and FROZEN not in OUT.parents
STORE = "data/cache/representations/alphagenome_locus.h5"
SPEC = dict(store=STORE, sha256="3906d888964fe71e303e2437ae64445b3f91803dcc04d15484fc5d5565850e00", dim=1536, dtype="float16")
REF = ("regulatory_histone_dnase_v1", "cpgpt_large_locus", "deepcpg_dna_locus")


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts/ewas_v3/check_freeze.py")], cwd=ROOT, check=True)  # read-only
    m = yaml.safe_load((ROOT / "configs/ewas_v3/main_trait_panel.yaml").read_text())
    universe = pd.read_parquet(ROOT / "outputs/encode_atlas_v1/loci.parquet", columns=["cpg_idx"]).cpg_idx.to_numpy()
    st = EmbeddingStore(ArmSpec(ARM, ARM, STORE, SPEC["sha256"], SPEC["dim"], SPEC["dtype"]), ROOT)  # verifies sha
    st.rows_for(universe)
    print("store verified, 408399 loci covered", flush=True)
    bb = BlockBootstrap(np.array(sorted(f"chr{i}" for i in range(1, 23))), 1000, 17)
    draws = pd.read_csv(ROOT / "data/registry/ewas_v3/chromosome_bootstrap_multiplicities.csv").to_numpy(np.int64)
    assert np.array_equal(draws, bb.draws), "bootstrap draws differ from freeze"
    contract = m["evaluation_contract_from_ewas_v2"]["probe"]
    jobs = [(t["task"], "primary", "main", ARM, t["sets"]["primary"]["matching"]["main"]["path"], SPEC, contract) for t in m["main_tasks"]]
    with concurrent.futures.ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("fork")) as pool:
        for f in concurrent.futures.as_completed([pool.submit(R.run_endpoint, j) for j in jobs]):
            print(json.dumps(f.result()), flush=True)
    rows, cons = [], []
    for t in m["main_tasks"]:
        name = t["task"]
        e = json.loads((R.endpoint_dir(name, "primary", "main", ARM) / "endpoint.json").read_text())
        s = t["sets"]["primary"]
        r = dict(category=t["category"], trait=t["trait"], task=name, source_analysis_id=t["source_analysis_id"], PMID=t["PMID"],
                 n_positive_universe=s["n_positive_universe"], n_matched=e["n_matched"], representation=ARM, threshold="primary",
                 matching_design="main", status=e["status"], n_nonconverged_fits=e.get("n_nonconverged_fits", 0), posthoc=True)
        if e["status"] == "complete":
            df = pd.read_parquet(R.endpoint_dir(name, "primary", "main", ARM) / "oof.parquet")
            n = len(df) // 2; v = df.oof_score.to_numpy(); y = df.y.to_numpy().astype(bool)
            blk = block_index(df.chrom.to_numpy(), bb.names); plan = RankPlan(v)
            rep = bb.replicates(lambda w: weighted_auroc(plan, y, w), blk)
            ci = percentile_ci(rep)
            cc = percentile_ci(bb.replicates(lambda w: weighted_mean((v[:n] > v[n:]).astype(float), w), blk[:n]))
            r.update(AUROC=e["AUROC"], ci_lo=ci[0], ci_hi=ci[1], AUPRC=e["AUPRC"], matched_pair_concordance=e["matched_pair_concordance"],
                     concordance_ci_lo=cc[0], concordance_ci_hi=cc[1])
            for ref in REF:  # paired contrast vs the FROZEN arm outputs (read-only), same pairs/draws
                fp = FROZEN / "primary/main/arms" / ref / name
                fe = json.loads((fp / "endpoint.json").read_text())
                if fe["status"] != "complete" or fe["n_nonconverged_fits"] or e["n_nonconverged_fits"]:
                    cons.append(dict(task=name, contrast=f"AlphaGenome - {ref}", status="not_evaluable")); continue
                fd = pd.read_parquet(fp / "oof.parquet")
                assert fd.drop(columns=["oof_score"]).equals(df.drop(columns=["oof_score"])), "pairs differ"
                fplan = RankPlan(fd.oof_score.to_numpy())
                frep = bb.replicates(lambda w: weighted_auroc(fplan, y, w), blk)
                cons.append(dict(category=t["category"], trait=t["trait"], task=name, contrast=f"AlphaGenome - {ref}", metric="AUROC",
                                 status="complete", **contrast(e["AUROC"], rep, fe["AUROC"], frep)))
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "posthoc_results.csv", index=False)
    pd.DataFrame(cons).to_csv(OUT / "posthoc_paired_contrasts.csv", index=False)
    (OUT / "README.md").write_text("POST-HOC arm, outside the approved EWAS-v3 scope (3 arms). Same frozen pairs/probe/bootstrap draws; "
                                   "frozen outputs untouched. Store sha256 %s.\n" % SPEC["sha256"])
    print("DONE", len(rows), "rows;", sum(r["status"] == "complete" for r in rows), "complete", flush=True)


if __name__ == "__main__":
    main()
