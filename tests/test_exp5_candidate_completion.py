"""Synthetic-data tests for the exp5_candidate_completion scaffold (no biological data, no training)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("exp5cc", ROOT / "scripts/followup/exp5_candidate_completion.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def synth_per_seed(n_seeds=3, n_pairs=60, arms=("a", "b"), rng=None):
    rng = rng or np.random.default_rng(0)
    out = {}
    for s in range(n_seeds):
        chrom = np.array([f"chr{1 + i % 7}" for i in range(n_pairs)])
        d = {"chrom": chrom}
        for k, arm in enumerate(arms):
            d[arm] = (rng.normal(0.4 * (k + 1), 1, n_pairs), rng.normal(0, 1, n_pairs))
        out[17001 + s] = d
    return out


def test_concordance_definition():
    assert M.matched_pair_concordance([1, 2, 3], [0, 0, 0]) == 1.0
    assert M.matched_pair_concordance([0, 0], [1, 1]) == 0.0
    assert M.matched_pair_concordance([1, 1], [1, 1]) == 0.5            # ties = 0.5
    assert M.matched_pair_concordance([2, 0, 1], [1, 1, 1]) == pytest.approx((1 + 0 + 0.5) / 3)


def test_support_tiers_thresholds():
    assert [M.support_tier(x) for x in (200, 199.9, 100, 99, 28, 2000)] == [
        "main-quality", "exploratory", "exploratory", "insufficient", "insufficient", "main-quality"]


def test_guard_blocks_audit_dir():
    with pytest.raises(RuntimeError):
        M.guard("outputs/biological_validation_v2/loyfer_failure_audit/x")


def test_stage2_extended_reproduces_exp5_stage2(monkeypatch):
    E = pytest.importorskip("cpg_repr_benchmark.encode_atlas.ewas_robust")
    ps = synth_per_seed()
    monkeypatch.setattr(E, "REPRESENTATIONS", {"a": "x", "b": "y"})
    monkeypatch.setattr(E, "COMPARISONS", [("a", "b")])
    ref = pd.DataFrame(E._stage2(("base", "s", ps, 40, 123)))
    new = pd.DataFrame(M.stage2_extended("base", "s", ps, 40, 123, ["a", "b"], [("a", "b")], with_auprc=False))
    r_auc = ref[ref.kind == "auc"].set_index("representation")
    n_auc = new[new.kind == "auc"].set_index("representation")
    for col in ("estimate", "ci_lo", "ci_hi", "between_seed_sd"):
        assert np.allclose(r_auc[col], n_auc.loc[r_auc.index, col])
    r_d = ref[ref.kind == "delta"].iloc[0]
    n_d = new[new.kind == "delta_auc"].iloc[0]
    for col in ("estimate", "ci_lo", "ci_hi", "p_boot", "n_seeds_delta_positive"):
        assert r_d[col] == pytest.approx(n_d[col])


def test_rng_stream_independent_of_arm_list():
    ps = synth_per_seed(arms=("a", "b", "c"))
    full = pd.DataFrame(M.stage2_extended("base", "s", ps, 30, 5, ["a", "b", "c"], [("a", "b")], with_auprc=False))
    sub = pd.DataFrame(M.stage2_extended("base", "s", ps, 30, 5, ["a", "b"], [("a", "b")], with_auprc=False))
    f = full[(full.kind == "delta_auc")].iloc[0]
    s = sub[(sub.kind == "delta_auc")].iloc[0]
    assert f.ci_lo == s.ci_lo and f.ci_hi == s.ci_hi and f.estimate == s.estimate


def test_per_seed_roundtrip_from_oof_table():
    ps = synth_per_seed(n_seeds=2, n_pairs=20)
    rows = []
    for sd, d in ps.items():
        for arm in ("a", "b"):
            pos, neg = d[arm]
            n = len(pos)
            pid = np.repeat(np.arange(n), 2)
            y = np.tile([True, False], n)
            sc = np.empty(2 * n, np.float32)
            sc[y], sc[~y] = pos, neg
            ch = np.repeat(d["chrom"], 2)
            perm = np.random.default_rng(1).permutation(2 * n)          # row order must not matter
            rows.append(pd.DataFrame(dict(variant="base", set="s", match_seed=sd, pair_id=pid[perm], y_true=y[perm],
                                          representation=arm, oof_score=sc[perm], chromosome=ch[perm])))
    back = M.per_seed_from_oof(pd.concat(rows), ["a", "b"])[("base", "s")]
    for sd, d in ps.items():
        assert np.allclose(back[sd]["a"][0], d["a"][0].astype(np.float32))
        assert np.array_equal(back[sd]["chrom"], d["chrom"])


def test_verify_same_pairs_detects_differences():
    base = pd.DataFrame(dict(cpg_idx=[1, 2, 3, 4], pair_id=[0, 0, 1, 1], y_true=[True, False, True, False],
                             fold=[0, 0, 1, 1], chromosome=["chr1", "chr1", "chr2", "chr2"]))
    assert M.verify_same_pairs(base, base.copy()) == []
    other = base.copy()
    other.loc[0, "fold"] = 4
    assert M.verify_same_pairs(base, other) == ["fold"]


def test_stage2_k_index_matches_exp5_convention():
    ev = pd.DataFrame(dict(variant=["probe", "base", "base", "base"], set=["x", "z", "a", "n"],
                           status=["EVALUABLE", "EVALUABLE", "EVALUABLE", "NOT EVALUABLE"]))
    assert M.stage2_k_index(ev) == {("base", "a"): 0, ("base", "z"): 1, ("probe", "x"): 2}


def test_bh_by_comparison_runs():
    res = pd.DataFrame(dict(variant="base", set=list("abc"), kind="delta_auc", comparison="c - d", p_boot=[0.01, 0.2, 0.5]))
    out = M.bh_by_comparison(res)
    assert out.q_bh.notna().all() and (out.q_bh >= res.p_boot).all()


def test_legacy_groupkfold_shim_greedy_largest_first_lightest_fold():
    # sizes: A=5, B=4, C=3, D=2, E=1, F=1 -> greedy: A->0, B->1, C->2, D->3, E->4, F->4 (lightest, first on ties: fold 4 load 1 vs others>=2)
    g = np.array(list("A" * 5 + "B" * 4 + "C" * 3 + "D" * 2 + "E" + "F"))
    f = M.LegacyGroupKFold(5).fold_of(g)
    assert [f[g == c][0] for c in "ABCDEF"] == [0, 1, 2, 3, 4, 4]
    for k, (tr, te) in enumerate(M.LegacyGroupKFold(5).split(np.zeros(len(g)), None, g)):
        assert np.array_equal(te, np.flatnonzero(f == k)) and len(set(tr) & set(te)) == 0
    # groups never split across folds, deterministic
    assert all(len(set(f[g == c])) == 1 for c in "ABCDEF")
    assert np.array_equal(f, M.LegacyGroupKFold(5).fold_of(g))


def test_check_stored_folds_detects_mismatch():
    g = np.array(list("A" * 5 + "B" * 4 + "C" * 3 + "D" * 2 + "E" + "F"))
    f = M.LegacyGroupKFold(5).fold_of(g)
    df = pd.DataFrame(dict(variant="base", set="s", match_seed=1, representation="r", chromosome=g, fold=f))
    assert M.check_stored_folds(df) == []
    df.loc[0, "fold"] = (f[0] + 1) % 5
    assert len(M.check_stored_folds(df)) == 1


def _cell(n=60, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile([True, False], n // 2)
    sc = np.clip(rng.random(n) * 0.5 + 0.4 * y, 1e-6, 1 - 1e-6)
    a = pd.DataFrame(dict(y_true=y, oof_score=sc))
    b = a.copy()
    b["oof_score"] = np.clip(sc + noise, 1e-6, 1 - 1e-6)
    return a, b


def test_regression_cell_verdict_amendment2():
    a, b = _cell(noise=1e-6)
    ok, m = M.regression_cell_verdict(a, b, [], 5e-4, 0.9999)
    assert ok and m["abs_dauroc"] == 0.0
    ok, _ = M.regression_cell_verdict(a, b, ["fold"], 5e-4, 0.9999)          # pairs/folds not identical -> fail
    assert not ok
    a2, b2 = _cell()
    b2["oof_score"] = b2.oof_score.iloc[::-1].to_numpy()                       # scrambled scores: AUROC and rank change
    ok, m = M.regression_cell_verdict(a2, b2, [], 5e-4, 0.9999)
    assert not ok
    # |dAUROC| just above the threshold fails even with perfect rank agreement of the rest
    a3, b3 = _cell()
    b3 = b3.copy()
    i, j = np.flatnonzero(b3.y_true.to_numpy())[0], np.flatnonzero(~b3.y_true.to_numpy())[0]
    b3.loc[i, "oof_score"], b3.loc[j, "oof_score"] = b3.oof_score[j], b3.oof_score[i]
    ok, m = M.regression_cell_verdict(a3, b3, [], 1e-4, 0.0)
    assert (m["abs_dauroc"] > 1e-4) == (not ok)
