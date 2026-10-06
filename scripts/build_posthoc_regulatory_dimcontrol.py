#!/usr/bin/env python3
"""POST-HOC diagnostic stores: is compression the bottleneck of regulatory_histone_dnase (2,492 binary tracks)?

Builds, from the SAME feature set/columns, fit protocol (chr1-19; chr20-22 transformed only) and store as the frozen SVD-256 arm:
  raw2492   : the binary locus x 2,492-track matrix itself (float16, exact), no compression
  lpca256 / lpca512 : logistic PCA (src/.../encode_atlas/logistic_pca.py), fitted on chr1-19 loci only, applied without refit to all loci
Label-free and patient-independent. The frozen SVD-256 store is not touched. (The SVD-512 dimension control uses
scripts/build_regulatory_embeddings.py with configs/posthoc/regulatory_embeddings_dimcontrol.yaml.)
  python scripts/build_posthoc_regulatory_dimcontrol.py --kind raw2492|lpca256|lpca512 [--limit-iter N]
"""
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import h5py, numpy as np, yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cpg_repr_benchmark.encode_atlas import logistic_pca as lp
from cpg_repr_benchmark.encode_atlas.compression import TrackStore, array_sha256, assert_disjoint, split_loci
from cpg_repr_benchmark.encode_atlas.feature_sets import resolve_feature_set

cfg = yaml.safe_load((ROOT / "configs/regulatory_embeddings.yaml").read_text())
FROZEN_COLS_SHA = "e5922c35e7b86b45683b5bdf9cbe5e748722830895402293211c318408537330"  # configs/frozen/regulatory_histone_dnase_v1.json


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True, choices=["raw2492", "lpca256", "lpca512"])
    ap.add_argument("--max-iter", type=int, default=1000); a = ap.parse_args()
    out = ROOT / cfg["output_dir"] / f"regulatory_histone_dnase__{a.kind}" ; out = out.with_name(out.name + ("" if a.kind == "raw2492" else "__discovery_chr1_19"))
    if out.with_suffix(".h5").exists():
        raise SystemExit(f"{out}.h5 exists")
    store = TrackStore(ROOT / cfg["store"])
    fit, heldout = split_loci(cfg["protocol"], store.cpg_idx, ROOT / cfg["registry"]); assert_disjoint(fit, heldout)
    fs = resolve_feature_set("regulatory_histone_dnase", ROOT / cfg["catalog"], check_expected=True)
    cols = np.asarray(fs.feature_columns, dtype=np.int64)
    assert array_sha256(cols) == FROZEN_COLS_SHA and len(cols) == 2492, "feature columns differ from the frozen arm"
    X = store.matrix[:, cols].tocsr()                      # all loci, store order, binary
    info = {"kind": a.kind, "n_tracks": int(len(cols)), "feature_columns_sha256": array_sha256(cols), "fit_protocol": fit.name,
            "n_fit_loci": fit.n, "fit_loci_sha256": fit.sha256}
    t0 = time.time()
    if a.kind == "raw2492":
        emb = X.toarray().astype(np.float16)
    else:
        k = int(a.kind[4:]); rows = store.rows_for(fit.cpg_idx)
        m = lp.fit(X[rows], k, max_iter=a.max_iter, log=lambda *s: print(*s, flush=True))
        emb = lp.encode(X, m["mu"], m["V"])
        # consistency: encoding the first fit rows must reproduce the fit-time scores
        head = lp.encode(X[rows[:2000]], m["mu"], m["V"]); A = m["A_head"]
        info.update(k=k, n_iter=int(m["n_iter"]), converged=bool(m["converged"]), final_rel_step=m["final_rel_step"], n_rejected_steps=int(m["n_rejected_steps"]), c_final=float(m["c_final"]), deviance_first=float(m["deviance"][0]), deviance_last=float(m["deviance"][-1]),
                    deviance_trace=[float(v) for v in m["deviance"]], encode_vs_fit_scores_corr=float(np.corrcoef(head.ravel(), A.ravel())[0, 1]),
                    encode_steps=100, method="logistic_pca_mm_orthonormal")
        np.savez_compressed(out.with_suffix(".compressor.npz"), mu=m["mu"], V=m["V"])
    assert emb.shape == (len(store.cpg_idx), len(cols) if a.kind == "raw2492" else k) and np.isfinite(emb.astype(np.float32)).all()
    tmp = out.with_suffix(".tmp.h5")
    with h5py.File(tmp, "w") as h:
        h.create_dataset("cpg_idx", data=store.cpg_idx, dtype="int64")
        h.create_dataset("embedding", data=emb, dtype=emb.dtype)
        h.attrs.update(representation=out.name, patient_specific=False, supervision="none", posthoc_comparator=True, reference_build="GRCh38",
                       cpg_namespace="grch38_cpg_cytosine_1based_v1", created_utc=datetime.now(timezone.utc).isoformat(),
                       fit_protocol=fit.name if a.kind != "raw2492" else "none", n_fit_loci=fit.n, fit_loci_sha256=fit.sha256,
                       feature_set=fs.name, feature_set_manifest_hash=fs.manifest_hash,
                       note="POST-HOC diagnostic of compression; rows of chr20-22 are out-of-fit (transformed only)")
    tmp.replace(out.with_suffix(".h5"))
    info.update(seconds=round(time.time() - t0, 1), shape=list(emb.shape), dtype=str(emb.dtype), posthoc_comparator=True)
    out.with_suffix(".h5.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "deviance_trace"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
