# ruff: noqa: C408
"""Synthetic tests for src/cpg_repr_benchmark/ewas_v2 (no embeddings, no real data, no TCGA)."""
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cpg_repr_benchmark.biological_validation_v2.splits import chromosome_blocked_folds
from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import BlockBootstrap, block_index
from cpg_repr_benchmark.ewas_v2 import design as D
from cpg_repr_benchmark.ewas_v2 import gate as G
from cpg_repr_benchmark.ewas_v2 import metrics as M
from cpg_repr_benchmark.ewas_v2.probe import fit_oof, pair_rows

ROOT = Path(__file__).resolve().parents[1]


def universe(n=6000, seed=0, nchr=5):
    rng = np.random.default_rng(seed)
    chrom = rng.choice([f"chr{i}" for i in range(1, nchr + 1)], n)
    pt = rng.choice([1, 2, 0], n, p=[0.3, 0.65, 0.05])
    return chrom, pt


# ------------------------------------------------------------------------------------------------ matching
def test_tiers():
    assert [D.support_tier(x) for x in (500, 200, 199, 100, 99, 0)] == [
        "main-quality", "main-quality", "exploratory", "exploratory", "insufficient", "insufficient"]


def test_main_background_exactness_exclusion_determinism():
    chrom, pt = universe()
    rng = np.random.default_rng(1)
    pos = rng.choice(len(chrom), 300, replace=False)
    excl = np.zeros(len(chrom), bool)
    excl[rng.choice(len(chrom), 800, replace=False)] = True
    excl[pos] = True                                     # a set's own members are excluded
    a = D.technical_background(pos, chrom, pt, excl, seed=17)
    b = D.technical_background(pos[::-1].copy(), chrom, pt, excl, seed=17)
    assert D.pairs_digest(a["pairs"]) == D.pairs_digest(b["pairs"])
    assert a["n_matched_positives"] + a["n_unmatched"] + a["n_dropped_unknown_probe_type"] == 300
    assert a["n_dropped_unknown_probe_type"] == int((pt[pos] == 0).sum())
    for p, (n,) in a["pairs"]:
        assert chrom[p] == chrom[n] and pt[p] == pt[n] and pt[n] > 0 and not excl[n] and n != p
    negs = [n for _, (n,) in a["pairs"]]
    assert len(set(negs)) == len(negs)
    assert D.pairs_digest(a["pairs"]) != D.pairs_digest(D.technical_background(pos, chrom, pt, excl, seed=18)["pairs"])


def test_background_excludes_other_set_members_and_counts_unmatched():
    chrom = np.array(["chr1"] * 10)
    pt = np.array([1] * 10)
    excl = np.zeros(10, bool)
    excl[[0, 1, 2, 3]] = True                            # other-set members: never backgrounds
    r = D.technical_background(np.array([0, 1, 2]), chrom, pt, excl, seed=1)
    assert r["n_matched_positives"] == 3 and all(n >= 4 for _, (n,) in r["pairs"])
    excl[4:9] = True                                     # only row 9 eligible, 3 positives
    r = D.technical_background(np.array([0, 1, 2]), chrom, pt, excl, seed=1)
    assert r["n_matched_positives"] == 1 and r["n_unmatched"] == 2


def test_matches_draft_scaffold():
    spec = importlib.util.spec_from_file_location("draft", ROOT / "scripts/ewas_v2/ewas_v2_design.py")
    draft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(draft)
    chrom, pt = universe(seed=5)
    pos = np.random.default_rng(2).choice(len(chrom), 150, replace=False)
    excl = np.zeros(len(chrom), bool)
    excl[pos] = True
    assert D.pairs_digest(D.technical_background(pos, chrom, pt, excl, 17)["pairs"]) == \
        draft.pairs_digest(draft.technical_background(pos, chrom, pt, excl, 17)["pairs"])


def test_s1_matching_exact_strata_caliper_determinism():
    rng = np.random.default_rng(4)
    n = 8000
    chrom, pt = universe(n, 3, nchr=3)
    island = rng.choice(["island", "shore", "open_sea"], n)
    Z = rng.normal(size=(n, 3))
    pos = rng.choice(n, 200, replace=False)
    excl = np.zeros(n, bool)
    excl[pos] = True
    a = D.context_matched_background(pos, chrom, pt, island, Z, excl, 17, 0.25)
    b = D.context_matched_background(pos[::-1].copy(), chrom, pt, island, Z, excl, 17, 0.25)
    assert D.pairs_digest(a["pairs"]) == D.pairs_digest(b["pairs"])
    assert a["n_matched_positives"] + a["n_unmatched"] + a["n_dropped_unknown_probe_type"] == 200
    negs = []
    for (p, (q,)), d in zip(a["pairs"], a["dist"], strict=True):
        assert chrom[p] == chrom[q] and pt[p] == pt[q] and island[p] == island[q] and not excl[q]
        assert d <= 0.25 and abs(np.linalg.norm(Z[p] - Z[q]) - d) < 1e-9
        negs.append(q)
    assert len(set(negs)) == len(negs)
    far = Z.copy()
    far[pos] += 100.0                                    # nothing within the caliper -> all dropped, counted
    c = D.context_matched_background(pos, chrom, pt, island, far, excl, 17, 0.25)
    assert c["n_matched_positives"] == 0 and c["n_unmatched"] == 200 - c["n_dropped_unknown_probe_type"]


def test_containment_rule_and_partner():
    sets = {"big": np.arange(100), "small": np.arange(60, 100), "other": np.arange(200, 230), "half": np.r_[np.arange(0, 5), np.arange(300, 305)]}
    rows = D.containment_flags(sets)
    ni = D.non_independent_map(rows)
    assert ni == {"small": "big", "half": "big"} and all(r["containment"] >= 0.5 for r in rows)
    assert "other" not in ni and "big" not in ni


# ------------------------------------------------------------------------------------------------ folds / leakage
def test_fold_map_is_frozen_bioval_map_and_plan_structure():
    c2f = D.chrom_to_fold()
    assert c2f == chromosome_blocked_folds(list(D.CHROMS), 5, seed=17)
    plan = D.nested_cv_plan(c2f)
    assert plan["n_fits_per_task"] == 105 and plan["grid_C"] == [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    for o in plan["outer"]:
        f = o["test_fold"]
        assert f not in o["train_folds"] and len(o["train_folds"]) == 4
        assert not set(o["test_chroms"]) & {c for c, g in c2f.items() if g in o["train_folds"]}
        for i in o["inner"]:
            assert i["inner_val_fold"] in o["train_folds"] and i["inner_val_fold"] not in i["inner_train_folds"]
            assert f not in i["inner_train_folds"] and f != i["inner_val_fold"]      # inner CV never sees the test fold


def test_fit_oof_no_leakage_and_signal():
    rng = np.random.default_rng(0)
    P = 400
    chrom = rng.choice(list(D.CHROMS[:10]), P)
    c2f = D.chrom_to_fold(D.CHROMS[:10])
    fold = np.array([c2f[c] for c in chrom])
    pairs = pd.DataFrame(dict(pos_cpg_idx=np.arange(P), neg_cpg_idx=np.arange(P, 2 * P), fold=fold, chrom=chrom))
    _, y, f, ch = pair_rows(pairs)
    assert (f[:P] == f[P:]).all() and (ch[:P] == ch[P:]).all()                  # a pair lies in one fold / one chromosome
    X = rng.normal(size=(2 * P, 6))
    # feature 0 relates to y with OPPOSITE signs inside fold 3 and elsewhere: a probe that never sees fold 3 must score fold 3 < chance
    X[:, 0] = np.where(f == 3, y * 2.0, -y * 2.0) + rng.normal(size=2 * P) * 0.5
    res = fit_oof(X, y, f, Cs=(0.01, 1.0), max_iter=2000)
    from cpg_repr_benchmark.biological_validation_v2.metrics import auroc_binary
    assert auroc_binary(y[f == 3], res["oof"][f == 3]) < 0.2                    # fold-3 labels never reached the fold-3 fit
    assert auroc_binary(y[f != 3], res["oof"][f != 3]) > 0.5
    assert res["n_nonconverged_fits"] == 0 and len(res["C"]) == len(np.unique(f))
    X2 = rng.normal(size=(2 * P, 6))
    X2[:, 1] += 1.5 * y
    r2 = fit_oof(X2, y, f, Cs=(0.01, 1.0), max_iter=2000)
    assert auroc_binary(y, r2["oof"]) > 0.8


# ------------------------------------------------------------------------------------------------ metrics
def test_concordance_auroc_auprc():
    assert M.matched_pair_concordance([1, 1], [1, 1]) == 0.5
    assert M.matched_pair_concordance([2, 0, 1], [1, 1, 1]) == pytest.approx((1 + 0 + 0.5) / 3)
    pos, neg = np.array([3.0, 2.0, 1.0]), np.array([0.0, 2.5, 1.0])
    m = M.point_metrics(pos, neg)
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert m["auroc"] == pytest.approx(brute)
    assert M.point_metrics(np.ones(5), np.zeros(5)) == {"auroc": 1.0, "concordance": 1.0, "auprc": 1.0}


def test_bootstrap_pairing_across_arms_and_block_weights():
    rng = np.random.default_rng(7)
    names = np.array(sorted(D.CHROMS[:6]))
    P = 300
    chrom = rng.choice(names, P)
    neg = rng.normal(size=P)
    good = neg + rng.normal(1.0, 1, P)
    bb = BlockBootstrap(names, 60, 17)
    # identical arms -> contrast exactly 0 in every replicate
    out = M.bootstrap_arms({"a": (good, neg), "b": (good.copy(), neg.copy())}, chrom, bb, [("c", "a", "b")])
    r = out["contrast"][("c", "auroc")]
    assert r["delta"] == 0 and r["ci_lo"] == 0 and r["ci_hi"] == 0
    # replicate b equals explicit concatenation of resampled blocks (pairs kept intact, same draw for both arms)
    out = M.bootstrap_arms({"a": (good, neg)}, chrom, bb, [])
    idx = {n: np.flatnonzero(chrom == n) for n in names}
    manual = []
    for b in range(bb.n_boot):
        sel = np.concatenate([idx[names[j]] for j in bb.draws[b]])
        manual.append(M.point_metrics(good[sel], neg[sel])["concordance"])
    assert out["ci"]["a"]["concordance"] == pytest.approx(tuple(np.quantile(manual, [0.025, 0.975])))
    # the shared draw table is the design draw table
    assert np.array_equal(BlockBootstrap(np.array(sorted(D.CHROMS)), D.N_BOOT, D.SEED).draws, D.draw_table(sorted(D.CHROMS)))
    assert block_index(chrom, names).max() < len(names)


def test_bootstrap_sign_and_blocks_without_pairs():
    rng = np.random.default_rng(8)
    names = np.array(sorted(D.CHROMS[:8]))
    P = 200
    chrom = rng.choice(names[:5], P)                      # blocks 5-7 have no pairs
    neg = rng.normal(size=P)
    sc = {"good": (neg + rng.normal(1.5, 1, P), neg), "chance": (rng.normal(size=P), neg)}
    out = M.bootstrap_arms(sc, chrom, BlockBootstrap(names, 100, 17), [("c", "good", "chance")])
    r = out["contrast"][("c", "concordance")]
    assert r["delta"] > 0 and r["ci_lo"] < r["ci_hi"]


def test_family_summary_counts_and_independence():
    con = pd.DataFrame([dict(task=t, design="main", contrast="c", metric="auroc", ci_lo=lo, ci_hi=hi) for t, lo, hi in
                        [("s", 0.1, 0.2), ("l", 0.1, 0.2), ("b", -0.2, -0.1), ("x", -0.1, 0.1)]])
    meta = pd.DataFrame(dict(task=["s", "l", "b", "x"], family=["exposure", "disease", "exposure", "disease"],
                             tier=["main-quality"] * 4, non_independent_of=["", "s", "", ""]))
    f = M.family_summary(con, meta)
    ind = f[f.scope == "independent"].set_index("family")
    assert ind.loc["exposure", "n_candidate_better"] == 1 and ind.loc["exposure", "n_candidate_worse"] == 1
    assert ind.loc["disease", "n_tasks"] == 1 and ind.loc["disease", "n_spans_zero"] == 1      # lung-like task counted via its partner only
    allf = f[f.scope == "all"].set_index("family")
    assert allf.loc["disease", "n_tasks"] == 2 and allf.loc["disease", "n_candidate_better"] == 1
    assert M.direction(0.0, 1.0) == "spans_zero" and M.direction(1e-9, 1) == "candidate_better" and M.direction(-1, -1e-9) == "candidate_worse"


# ------------------------------------------------------------------------------------------------ gate / writer
def git(root, *a):
    subprocess.run(["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t", *a], check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    for rel in G.DESIGN_FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_gate_registration_and_tag_failure_modes(repo):
    assert not G.check_registration(repo).ok                                   # missing
    (repo / G.REGISTRATION_DOC).write_text("v1")
    assert not G.check_registration(repo).ok                                   # untracked
    git(repo, "add", "."); git(repo, "commit", "-qm", "reg")
    assert G.check_registration(repo).ok
    (repo / G.REGISTRATION_DOC).write_text("v2")
    assert not G.check_registration(repo).ok                                   # modified
    git(repo, "checkout", "--", G.REGISTRATION_DOC)
    assert not G.check_tag(repo).ok and not G.check_design_files_unchanged(repo).ok   # no tag
    for rel in G.DESIGN_FILES[1:]:
        (repo / rel).write_text("x")
    git(repo, "add", "."); git(repo, "commit", "-qm", "design"); git(repo, "tag", G.TAG)
    assert G.check_tag(repo).ok and G.check_design_files_unchanged(repo).ok
    (repo / G.SPEC).write_text("changed")
    git(repo, "commit", "-qam", "change spec after the tag")
    assert not G.check_design_files_unchanged(repo).ok                         # design change after the freeze is refused
    assert any(G.SPEC in line for line in [str(G._git(repo, "diff", "--name-only", G.TAG, "HEAD")[1])])


def test_gate_design_artifact_hash_mismatch(tmp_path):
    (tmp_path / "configs/ewas_v2").mkdir(parents=True)
    dd = tmp_path / D.DESIGN_DIR
    dd.mkdir(parents=True)
    (dd / "a.csv").write_text("1\n")
    (tmp_path / D.PINS.rsplit("/", 1)[0]).mkdir(parents=True)
    (tmp_path / D.PINS).write_text(json.dumps({"known_sets_sha256": {}}))
    man = dict(artifacts_sha256={"a.csv": D.sha256_file(dd / "a.csv")}, inputs_sha256={}, bootstrap_fingerprint="x")
    (tmp_path / D.MANIFEST).write_text(json.dumps(man))
    assert G.check_design_artifacts(tmp_path).ok
    (dd / "a.csv").write_text("2\n")
    assert not G.check_design_artifacts(tmp_path).ok
    (dd / "a.csv").write_text("1\n")
    (dd / "extra.csv").write_text("zz")
    assert not G.check_design_artifacts(tmp_path).ok


def test_gate_out_root_inputs_state(tmp_path):
    ok = tmp_path / "outputs/ewas_v2/tag"
    assert G.check_out_root(tmp_path, ok).ok
    for bad in (tmp_path / "outputs/other_experiment", tmp_path / "configs/ewas_v2", tmp_path / "outputs/ewas_v2/loyfer_failure_audit",
                tmp_path / "data/derived/ewas_v2_v1"):
        assert not G.check_out_root(tmp_path, bad).ok
    assert not G.check_inputs(["outputs/x/tcga/betas.parquet"]).ok
    assert not G.check_inputs(["outputs/biological_validation_v2/loyfer_failure_audit/x"]).ok
    assert not G.check_inputs(["data/protocols/split.npz"]).ok
    assert G.check_inputs(["data/derived/ewas_v2_v1/pairs_main.csv"]).ok
    ok.mkdir(parents=True)
    assert G.check_state(ok, "run", "a").ok and not G.check_state(ok, "contrasts").ok and not G.check_state(ok, "report-inputs").ok
    (ok / "state.json").write_text(json.dumps(dict(state="ARMS_RUNNING", arms_done=list(G.MAIN_ARMS), contrasts_done=False)))
    assert G.check_state(ok, "contrasts").ok
    (ok / "state.json").write_text(json.dumps(dict(state="EXECUTED", arms_done=list(G.MAIN_ARMS), contrasts_done=True)))
    assert not G.check_state(ok, "run", "a").ok and G.check_state(ok, "report-inputs").ok


def test_writer_guards(tmp_path):
    (tmp_path / "outputs/ewas_v2").mkdir(parents=True)
    w = G.EwasWriter(tmp_path, tmp_path / "outputs/ewas_v2/tag")
    p = w.path("arms", "a", "x.json")
    assert tmp_path.resolve() in p.parents
    w.write_json("state.json", {"a": 1})
    for bad in ("../../other/x.json", "../../../configs/x", "loyfer_failure_audit/x.json"):
        with pytest.raises(PermissionError):
            w.path(bad)
    for bad_root in ("outputs/other", "configs", "data/derived/ewas_v2_v1", "outputs/biological_validation_v2/loyfer_failure_audit"):
        with pytest.raises(PermissionError):
            G.EwasWriter(tmp_path, tmp_path / bad_root)


# ------------------------------------------------------------------------------------------------ real design artifacts (embedding-free)
@pytest.mark.skipif(not (ROOT / D.MANIFEST).exists(), reason="design manifest not built")
def test_manifest_consistent_with_spec_and_rules():
    import yaml
    man = json.loads((ROOT / D.MANIFEST).read_text())
    spec = yaml.safe_load((ROOT / "configs/ewas_v2/ewas_v2_spec_v1.yaml").read_text())
    for tier, key in (("main-quality", "main_quality"), ("exploratory", "exploratory"), ("insufficient", "insufficient")):
        assert man["tiers_main"][tier] == sorted(spec["tasks"]["tiers_main"][key])
    assert spec["probe"]["C_grid"] == list(D.C_GRID) and spec["probe"]["max_iter"] == D.MAX_ITER
    fam = {t: f for f, ts in spec["tasks"]["families"].items() for t in ts}
    assert fam == D.TASK_FAMILY
    assert man["non_independent"] == spec["tasks"]["overlap_rule"]["expected"]
    assert man["chrom_to_fold"] == D.chrom_to_fold()
    assert set(spec["arms"]) == set(G.ALL_ARMS)
    for r in man["tasks"].values():
        assert r["n_matched_pairs"] + r["n_unmatched"] + r["n_dropped_unknown_probe_type"] == r["n_universe"]
        assert r["tier"] == D.support_tier(r["n_matched_pairs"]) and r["s1_tier"] == D.support_tier(r["s1_n_matched_pairs"])
        assert sum(r["main_pairs_by_chrom"].values()) == r["n_matched_pairs"]
    pm = pd.read_csv(ROOT / D.DESIGN_DIR / "pairs_main.csv")
    c2f = man["chrom_to_fold"]
    assert (pm.chrom.map(c2f) == pm.fold).all()
    u = D.load_universe(ROOT)
    assert (u.chrom.to_numpy()[pm.pos_row] == u.chrom.to_numpy()[pm.neg_row]).all()
    assert (u.probe_type.to_numpy()[pm.pos_row] == u.probe_type.to_numpy()[pm.neg_row]).all()
    members = set(pd.read_csv(ROOT / D.DESIGN_DIR / "sets_membership.csv", usecols=["row"]).row)
    assert not set(pm.neg_row) & members                  # no background is a member of ANY panel set
    assert pm.groupby("task").neg_row.apply(lambda x: x.is_unique).all()      # without replacement within a task


# ------------------------------------------------------------------------------------------------ end-to-end on a synthetic root
def _synthetic_root(tmp_path):
    import h5py
    import yaml
    root = tmp_path
    rng = np.random.default_rng(0)
    dd = root / D.DESIGN_DIR
    dd.mkdir(parents=True)
    c2f = D.chrom_to_fold()
    chroms = np.array(sorted(D.CHROMS))
    n_cpg = 4000
    ids = np.arange(1000, 1000 + n_cpg)
    pairs_all, s1_all, summ = [], [], []
    for ti, (task, fam) in enumerate([("ewas_catalog_bmi", "exposure"), ("ewas_catalog_sle", "disease")]):
        P = 300 if ti == 0 else 120
        ch = rng.choice(chroms, P)
        pos = rng.choice(n_cpg // 2, P, replace=False)
        neg = n_cpg // 2 + rng.choice(n_cpg // 2, P, replace=False)
        df = pd.DataFrame(dict(task=task, pair_id=np.arange(P), pos_row=pos, pos_cpg_idx=ids[pos], neg_row=neg, neg_cpg_idx=ids[neg],
                               chrom=ch, probe_type=2, fold=[c2f[c] for c in ch]))
        pairs_all.append(df)
        s1_all.append(df.iloc[: P - 10].assign(island="island", dist="0.1"))
        summ.append(dict(task=task, family=fam, n_universe=P, n_matched_pairs=P, tier=D.support_tier(P), s1_n_matched_pairs=P - 10,
                         s1_tier=D.support_tier(P - 10), non_independent_of="" if ti == 0 else "ewas_catalog_bmi"))
    pd.concat(pairs_all).to_csv(dd / "pairs_main.csv", index=False)
    pd.concat(s1_all).to_csv(dd / "pairs_s1.csv", index=False)
    pd.DataFrame(summ).to_csv(dd / "design_summary.csv", index=False)
    (dd / "nested_cv_plan.json").write_text(json.dumps(D.nested_cv_plan(c2f, max_iter=500)))
    entries = []
    for arm, dim, sig in (("regulatory_histone_dnase_v1", 8, 1.5), ("cpgpt_large_locus", 10, 0.5), ("deepcpg_dna_locus", 6, 0.0)):
        E = rng.normal(size=(n_cpg, dim)).astype(np.float32)
        E[: n_cpg // 2, 0] += sig                                        # positives live in the first half
        E[3] = 0                                                        # one all-zero row
        p = root / f"stores/{arm}.h5"
        p.parent.mkdir(exist_ok=True, parents=True)
        with h5py.File(p, "w") as h:
            h["cpg_idx"] = ids
            h["embedding"] = E
        entries.append(dict(id="regulatory_histone_dnase" if arm.startswith("regulatory") else arm, store_h5=f"stores/{arm}.h5",
                            sha256=D.sha256_file(p), dim=dim, dtype="float32"))
    mp = root / "configs/experiments/regulatory_confirmation_matrix/matrix.yaml"
    mp.parent.mkdir(parents=True)
    mp.write_text(yaml.safe_dump({"arms": entries}))
    (root / "data/registry").mkdir(parents=True)
    pd.DataFrame(dict(set=["ewas_catalog_bmi", "ewas_catalog_sle"], support_tier=["main-quality", "main-quality"])).to_csv(root / D.EXP5_TIERS, index=False)
    return root


def test_end_to_end_synthetic_run_contrasts_report(tmp_path):
    from cpg_repr_benchmark.ewas_v2 import runner as R
    root = _synthetic_root(tmp_path)
    out = root / "outputs/ewas_v2/tag"
    for arm in G.MAIN_ARMS:
        done = R.run_arm(root, arm, out, threads=1, gate_checks=[], command="test")
        assert set(done) == {"main/ewas_catalog_bmi", "main/ewas_catalog_sle", "S1/ewas_catalog_bmi", "S1/ewas_catalog_sle"}
    ep = json.loads((out / "arms/regulatory_histone_dnase_v1/ewas_catalog_bmi/endpoint.json").read_text())
    assert ep["zero_rows_positives"] + ep["zero_rows_backgrounds"] <= 2 and ep["n_nonconverged_fits"] == 0
    assert ep["point"]["auroc"] > 0.6
    again = R.run_arm(root, "regulatory_histone_dnase_v1", out, threads=1, gate_checks=[])      # resume: nothing refit
    assert set(again.values()) == {"skipped_existing"}
    assert G.read_state(out)["arms_done"] == sorted(G.MAIN_ARMS)
    res = R.run_contrasts(root, out, threads=1, gate_checks=[])
    assert res["arms"] == list(G.MAIN_ARMS)
    c = pd.read_csv(out / "contrasts.csv")
    prim = c[(c.contrast == "primary_vs_cpgpt") & (c.design == "main") & (c.metric == "auroc") & (c.task == "ewas_catalog_bmi")].iloc[0]
    assert prim.delta > 0 and prim.ci_lo < prim.ci_hi
    assert set(c.contrast) == {"primary_vs_cpgpt", "secondary_vs_deepcpg"}                      # no functional arm run
    fam = pd.read_csv(out / "family_summary.csv")
    assert {"independent", "all"} == set(fam.scope) and {"main", "S1"} == set(fam.design)
    assert json.loads((out / "bootstrap_manifest.json").read_text())["draw_table_fingerprint"] == \
        BlockBootstrap(np.array(sorted(D.CHROMS)), D.N_BOOT, D.SEED).fingerprint()
    # all arms share identical rows (OOF key columns)
    k = [pd.read_parquet(out / f"arms/{a}/ewas_catalog_sle/oof.parquet")[["pair_id", "cpg_idx", "y", "fold"]] for a in G.MAIN_ARMS]
    assert k[0].equals(k[1]) and k[0].equals(k[2])
    R.report_inputs(root, out, gate_checks=[])
    t = pd.read_csv(out / "report_inputs/tier_table.csv").set_index("task")
    assert t.loc["ewas_catalog_sle", "tier"] == "exploratory" and bool(t.loc["ewas_catalog_sle", "tier_changed_vs_exp5"])
    assert (out / "report_inputs/s1_vs_main_auroc.csv").exists() and G.read_state(out)["state"] == "REPORTED"
    with pytest.raises(PermissionError):
        G.EwasWriter(root, root / "outputs/ewas_v2/tag").path("..", "..", "elsewhere.json")


def test_run_arm_refuses_when_gate_fails_and_coverage_check(tmp_path):
    from cpg_repr_benchmark.ewas_v2 import runner as R
    root = _synthetic_root(tmp_path)
    with pytest.raises(SystemExit):
        R.run_arm(root, "cpgpt_large_locus", root / "outputs/ewas_v2/tag", gate_checks=[G.Check("x", False, "boom")])
    assert not (root / "outputs/ewas_v2/tag/arms").exists()
    with pytest.raises(SystemExit):
        R.run_arm(root, "cpgpt_large_locus", root / "outputs/ewas_v2/tag", threads=8, gate_checks=[])      # thread cap
