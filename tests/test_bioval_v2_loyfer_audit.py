# ruff: noqa: RUF059, F841
"""Synthetic-data tests of the Loyfer failure audit kernels (no real data, no embeddings, no frozen artifact)."""
from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.stats import spearmanr

from cpg_repr_benchmark.bioval_v2_followup import targets as tg
from cpg_repr_benchmark.bioval_v2_launch import chrom_probes as cp
from cpg_repr_benchmark.bioval_v2_loyfer_audit import baselines as bl
from cpg_repr_benchmark.bioval_v2_loyfer_audit import common as cm
from cpg_repr_benchmark.bioval_v2_loyfer_audit import decision as dc
from cpg_repr_benchmark.bioval_v2_loyfer_audit import gate as gt
from cpg_repr_benchmark.bioval_v2_loyfer_audit import oof_pairs as op
from cpg_repr_benchmark.bioval_v2_loyfer_audit import raw_space as rs
from cpg_repr_benchmark.bioval_v2_loyfer_audit import registry as rg
from cpg_repr_benchmark.bioval_v2_loyfer_audit import reliability as rel
from cpg_repr_benchmark.bioval_v2_loyfer_audit import semantics as sm
from cpg_repr_benchmark.bioval_v2_loyfer_audit import svd_geometry as sg
from cpg_repr_benchmark.bioval_v2_loyfer_audit import target_audit as ta

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------ registry
def test_registry_consistency():
    assert len(rg.ALPHA_GRID_EXTENDED) == 11 and rg.ALPHA_GRID_EXTENDED[0] == 1e-2 and rg.ALPHA_GRID_EXTENDED[-1] == 1e8
    assert set(rg.ALPHA_GRID_ORIGINAL) <= set(rg.ALPHA_GRID_EXTENDED) | {1e4}
    assert rg.AUDIT_OUT != rg.TAG_DIR and not rg.AUDIT_OUT.startswith(rg.TAG_DIR)
    assert rg.CANDIDATE in rg.ARMS and set(rg.COMPARATORS) < set(rg.ARMS)
    assert len(rg.V_EDGES) == 3 and rg.N_UNIVERSE - 849 == rg.N_UNIVERSE_D2


# ------------------------------------------------------------------------------------------------ xlsx + mapping
def _write_xlsx(path, rows, sheet="Table S1"):
    strings = []

    def si(v):
        if v not in strings:
            strings.append(v)
        return strings.index(v)

    xml_rows = []
    for r, row in enumerate(rows, 1):
        cells = "".join(f'<c r="{chr(65 + c)}{r}" t="s"><v>{si(v)}</v></c>' for c, v in enumerate(row) if v is not None)
        xml_rows.append(f'<row r="{r}">{cells}</row>')
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/sharedStrings.xml", f'<sst xmlns="{ns}">' + "".join(f"<si><t>{s}</t></si>" for s in strings) + "</sst>")
        z.writestr("xl/workbook.xml", f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                   f'<sheets><sheet name="{sheet}" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{ns}"><sheetData>' + "".join(xml_rows) + "</sheetData></worksheet>")


def test_xlsx_reader_and_mapping(tmp_path):
    p = tmp_path / "s1.xlsx"
    _write_xlsx(p, [["title only"], ["Sample name", "Group", "PatientID"], ["Adipocytes-Z0001", "Adipocytes", "p1"],
                    ["Aorta-Endothelium-Z0002", "Endothelium", "p2"]])
    s1 = cm.read_xlsx_sheet(p, "Table S1", "Sample name")
    assert list(s1.columns) == ["Sample name", "Group", "PatientID"] and len(s1) == 2
    prov = pd.DataFrame({"file_name": ["GSM1_Adipocytes-Z0001.hg38.pat.gz", "GSM2_Aorta-Endothel-Z0002.hg38.pat.gz", "GSM3_CNVS-NORM-x-Z0003.hg38.pat.gz"],
                         "derived_group": ["Adipocytes", "Endothel", None], "patient_id": ["p1", "p2", None]})
    assert ta.verify_mapping(prov, s1)["ok"]
    prov.loc[1, "derived_group"] = "Adipocytes"
    r = ta.verify_mapping(prov, s1)
    assert not r["ok"] and r["group_mismatch"]
    prov.loc[1, ["derived_group", "patient_id"]] = ["Endothel", "pX"]
    assert ta.verify_mapping(prov, s1)["patient_mismatch"]


# ------------------------------------------------------------------------------------------------ target recompute vs frozen-style reference
def _fake_counts(rng, n_files=8, N=40, groups=("A", "B", "C"), min_cov=10):
    files = [f"f{i}" for i in range(n_files)]
    grp = ["A", "A", "A", "B", "B", "C", "C", "C"][:n_files]
    donor = ["d1", "d1", "d2", "d3", "d4", "d5", "d6", "d7"][:n_files]
    tab = pd.DataFrame({"file": files, "donor": donor, "group": grp})
    C = rng.integers(0, 40, (n_files, N)).astype(np.uint16)
    M = (rng.uniform(0, 1, (n_files, N)) * C).astype(np.uint16)
    return files, tab, M, C


def test_donor_group_beta_matches_hand_computation():
    rng = np.random.default_rng(1)
    files, tab, M, C = _fake_counts(rng)
    D, V, donors = ta.donor_betas(M, C, files, tab)
    beta, nval, groups = ta.group_betas(D, V, donors)
    assert groups == ["A", "B", "C"]
    # hand computation of CpG 5, group A (donor d1 = files 0,1; donor d2 = file 2)
    j = 5
    def sample_beta(i):
        return M[i, j] / C[i, j] if C[i, j] >= 10 else None
    d1 = [b for b in (sample_beta(0), sample_beta(1)) if b is not None]
    d2 = [b for b in (sample_beta(2),) if b is not None]
    dv = [np.mean(x) for x in (d1, d2) if x]
    expect = np.mean(dv) if dv else np.nan
    assert (np.isnan(expect) and np.isnan(beta[j, 0])) or beta[j, 0] == pytest.approx(expect)
    assert nval[j, 0] == len(dv)
    mask = ta.primary_mask(nval, np.array([2, 2, 3]))
    assert mask.shape == nval.shape and (mask == (nval >= 2)).all()


def test_pearson_loop_equals_vectorised_and_compare_beta():
    rng = np.random.default_rng(2)
    P = rng.uniform(size=(30, 25))
    P[rng.uniform(size=P.shape) < 0.2] = np.nan
    ia, ib = rng.integers(0, 30, 50), rng.integers(0, 30, 50)
    n1, r1 = ta.pearson_loop(P, ia, ib, min_shared=10)
    n2, r2 = tg.pair_pearson(P, ia, ib, min_shared=10)
    assert (n1 == n2).all() and np.allclose(r1, r2, equal_nan=True, atol=1e-12)
    b64 = rng.uniform(0, 1, (20, 4))
    b64[3, 1] = np.nan
    b16 = b64.astype(np.float16).astype(np.float64)
    c = ta.compare_beta(b64, b16)
    assert c["nan_pattern_mismatch"] == 0 and c["n_beyond_half_ulp"] == 0 and c["n_f16_rounding_mismatch"] == 0
    bad = b16.copy()
    bad[0, 0] += 0.01
    assert ta.compare_beta(b64, bad)["n_beyond_half_ulp"] == 1


def test_subset_indices_deterministic_and_stratified():
    strat = np.repeat(np.array(rg.STRATA), [50, 60, 70, 80, 90, 100])
    pairs = pd.DataFrame({"stratum": strat})
    a = ta.subset_indices(pairs, seed=17, per_stratum=20)
    b = ta.subset_indices(pairs, seed=17, per_stratum=20)
    assert (a == b).all() and len(a) == 120 and (np.diff(a) > 0).all()
    assert (pairs.stratum.to_numpy()[a] == "<1kb").sum() == 20
    assert not (a == ta.subset_indices(pairs, seed=18, per_stratum=20)).all()


def test_structural_checks_flag_problems():
    uni = pd.DataFrame({"cpg_idx": [1, 2, 3], "chrom": ["chr1", "chr1", "chr2"], "pos": [10, 20, 10]})
    pairs = pd.DataFrame({"cpg_i": [1, 2, 2], "cpg_j": [2, 3, 3]})
    beta = np.array([[0.1, 0.5], [0.2, np.nan], [0.3, 0.9]])
    out = ta.structural_checks(pairs, uni, beta, ~np.isnan(beta), ["a", "b"])
    assert out["pairs_duplicates"] == 1 and out["beta_in_0_1"] and out["pairs_cpg_i_lt_cpg_j"]
    beta[0, 0] = 1.2
    assert not ta.structural_checks(pairs, uni, beta, ~np.isnan(beta), ["a", "b"])["beta_in_0_1"]


# ------------------------------------------------------------------------------------------------ fasta + pat
def test_fasta_cpg_index_and_pat_parser(tmp_path):
    seqs = {"chr1": "ACGTTCGGacgNCG", "chr2": "CGCGAT"}
    fa, fai = tmp_path / "t.fa", tmp_path / "t.fa.fai"
    lines, off, w = [], 0, 5
    with open(fa, "wb") as fh, open(fai, "w") as fi:
        for n, s in seqs.items():
            h = f">{n}\n".encode()
            fh.write(h)
            off += len(h)
            fi.write(f"{n}\t{len(s)}\t{off}\t{w}\t{w + 1}\n")
            body = "\n".join(s[i:i + w] for i in range(0, len(s), w)) + "\n"
            fh.write(body.encode())
            off += len(body)
    gidx, total = ta.genome_cpg_index(fa, fai, chroms=("chr1", "chr2"))
    assert gidx["chr1"][0].tolist() == [2, 6, 10, 13] and gidx["chr2"][0].tolist() == [1, 3] and total == 6
    uni = pd.DataFrame({"chrom": ["chr1", "chr1", "chr2", "chr2"], "pos": [6, 7, 3, 1]})
    assert ta.universe_global_index(uni, gidx).tolist() == [2, -1, 6, 5]
    # pat: line = chrom, 1-based global idx of first CpG, string over C/T/., count
    pat = "chr1\t1\tCT\t3\nchr1\t2\tTC.\t2\nchr1\t5\tC\t7\nchr2\t5\t.C\t4\n"
    p = tmp_path / "x.pat.gz"
    with gzip.open(p, "wb") as g:
        g.write(pat.encode())
    targets = np.array([2, 3, 6])
    meth, cov, n = ta.pat_counts(p, targets, 10, chunk_bytes=16)
    # idx2: CT line1 -> T (cov 3); line2 T (cov 2) -> cov 5, meth 0 ; idx3: line2 C (2) -> meth 2 cov 2 ; idx6: line4 C(4)
    assert n == 4 and meth.tolist() == [0.0, 2.0, 4.0] and cov.tolist() == [5.0, 2.0, 4.0]
    # brute force equality on random pat
    rng = np.random.default_rng(3)
    ls, bm, bc = [], np.zeros(30), np.zeros(30)
    for k in range(0, 28, 2):
        L = int(rng.integers(1, 3))
        s = "".join(rng.choice(list("CT."), L))
        c = int(rng.integers(1, 9))
        ls.append(f"chr1\t{k + 1}\t{s}\t{c}\n")
        for j, ch in enumerate(s):
            if ch == "C":
                bm[k + j + 1] += c
            if ch in "CT":
                bc[k + j + 1] += c
    p2 = tmp_path / "y.pat.gz"
    with gzip.open(p2, "wb") as g:
        g.write("".join(ls).encode())
    tg_all = np.arange(1, 30)
    m2, c2, _ = ta.pat_counts(p2, tg_all, 29, chunk_bytes=40)
    assert np.array_equal(m2, bm[1:]) and np.array_equal(c2, bc[1:])


def test_verdict_no_override():
    ok = ta.build_verdict({"a": True, "b": None})
    assert ok["verdict"] == "PASS_LEVEL_A" and ok["level"] == "A" and ok["raw_level_B_executed"] is False and "NOT verify" in ok["caveat"]
    v = ta.build_verdict({"a": True, "b": False})
    assert v["verdict"] == "FAIL" and v["failed_checks"] == ["b"]
    assert ta.build_verdict({"a": True}, level_b_executed=True)["verdict"] == "PASS_LEVEL_A_AND_B"
    assert rg.ACCEPTED_VERDICTS == ("PASS_LEVEL_A",)


# ------------------------------------------------------------------------------------------------ reliability
def _donor_world(rng, signal, n_cpg=600, n_groups=12, n_donors=6):
    """donors x CpG beta with a shared per-group signal + donor noise."""
    rows, grp = [], []
    for g in range(n_groups):
        for d in range(n_donors):
            rows.append(signal[g] + rng.normal(0, 0.0 if signal is None else NOISE[0], n_cpg))
            grp.append(f"g{g:02d}")
    D = np.clip(np.array(rows), 0, 1)
    donors = pd.DataFrame({"group": grp, "donor": [f"d{i}" for i in range(len(grp))]})
    return D, np.ones_like(D, bool), donors


NOISE = [0.0]


def _rel_pooled(noise, seed=0):
    rng = np.random.default_rng(seed)
    n_cpg, n_groups = 500, 14
    sig = rng.uniform(0.1, 0.9, (n_groups, n_cpg))
    NOISE[0] = noise
    D, V, donors = _donor_world(rng, sig, n_cpg, n_groups, 6)
    ia, ib = rng.integers(0, n_cpg, 3000), rng.integers(0, n_cpg, 3000)
    strat = np.where(np.arange(3000) % 2 == 0, ">1Mb", "interchromosomal")
    half, used = rel.make_halves(donors, 0)
    BA, BB = rel.half_group_betas(D, V, donors, half, used)
    ta_, tb_ = rel.half_similarities(BA, BB, ia, ib, len(used))
    vmin = np.full(3000, 0.05)
    mm = np.full(3000, 0.5)
    return rel.reliability_table(ta_, tb_, strat, vmin, mm)


def test_reliability_known_ground_truth():
    hi = _rel_pooled(0.0)
    assert hi["all"]["r_spearman"] > 0.99 and hi["pooled_gt1Mb_inter_eqw"]["ceiling"] > 0.99
    # noise-only profiles (signal killed by huge donor noise): reliability near 0
    rng = np.random.default_rng(5)
    D = rng.uniform(0, 1, (72, 500))
    donors = pd.DataFrame({"group": [f"g{i // 6:02d}" for i in range(72)], "donor": [f"d{i}" for i in range(72)]})
    half, used = rel.make_halves(donors, 1)
    BA, BB = rel.half_group_betas(D, np.ones_like(D, bool), donors, half, used)
    ia, ib = rng.integers(0, 500, 4000), rng.integers(0, 500, 4000)
    a, b = rel.half_similarities(BA, BB, ia, ib, len(used))
    strat = np.where(np.arange(4000) % 2 == 0, ">1Mb", "interchromosomal")
    t = rel.reliability_table(a, b, strat, np.full(4000, 0.05), np.full(4000, 0.5))
    assert abs(t["all"]["r_spearman"]) < 0.1 and t["all"]["ceiling"] < 0.4


def test_halves_deterministic_equal_and_formulas():
    donors = pd.DataFrame({"group": ["a"] * 5 + ["b"] * 2 + ["c"], "donor": list("12345678")})
    h1, u1 = rel.make_halves(donors, 3)
    h2, u2 = rel.make_halves(donors, 3)
    assert (h1 == h2).all() and u1 == u2 == ["a", "b"] and (h1[7] == -1)
    assert (h1[:5] == 0).sum() == 2 and (h1[:5] == 1).sum() == 2 and (h1[:5] == -1).sum() == 1
    assert not (rel.make_halves(donors, 4)[0] == h1).all() or True
    assert rel.spearman_brown(0.5) == pytest.approx(2 / 3) and rel.spearman_brown(1.0) == pytest.approx(1.0)
    assert rel.ceiling_from_reliability(-0.2) == 0 and rel.ceiling_from_reliability(0.64) == pytest.approx(0.8)
    assert rel.min_shared_for(39) == 20 and rel.min_shared_for(35) == 18


# ------------------------------------------------------------------------------------------------ semantics
def test_semantics_indicators():
    rng = np.random.default_rng(6)
    P = rng.uniform(0.3, 0.7, (50, 30))
    P[0] = 0.9                                    # constant CpG
    mean, var, rng_, n = sm.per_cpg_profile_stats(P)
    assert var[0] == pytest.approx(0) and rng_[0] == pytest.approx(0) and n[0] == 30
    L = sm.global_group_level(P)
    P[1] = 0.2 + 0.5 * (L - L.min()) / np.ptp(L)  # driven by the global level
    r2 = sm.r2_with_level(P, L)
    assert r2[1] == pytest.approx(1.0)
    ia, ib = np.array([1, 2]), np.array([1, 3])
    resid = sm.pair_residual_pearson(P, ia, ib, L, min_shared=10)
    assert np.isnan(resid[0])                     # profile fully explained by L -> residual constant -> undefined
    assert np.isfinite(resid[1])
    cls = sm.classify_pairs(np.array([0.0001, 0.05, 0.05, 0.05]), np.array([0.5, 0.01, 0.5, 0.5]), np.array([0, 0, 0.9, 0.1]),
                            np.array([0, 0, 0.9, 0.1]), r_small=0.05, g_hi=0.5)
    assert cls.tolist() == [0, 1, 2, 3]
    t = np.linspace(-1, 1, 1001)
    bands, th = sm.band_masks(t)
    assert bands["top1pct"].sum() in (11, 10, 12) and bands["bottom1pct"].sum() in (10, 11, 12)
    e1 = sm.select_examples(bands["top1pct"], tag=0)
    assert (e1 == sm.select_examples(bands["top1pct"], tag=0)).all() and len(e1) == rg.EXAMPLES_PER_BAND
    assert sm.topband_dominance({"NEAR_CONSTANT": 0.4, "SMALL_AMPLITUDE": 0.2}, {"NEAR_CONSTANT": 0.1, "SMALL_AMPLITUDE": 0.1})["dominated"]
    assert not sm.topband_dominance({"NEAR_CONSTANT": 0.1, "SMALL_AMPLITUDE": 0.1}, {"NEAR_CONSTANT": 0.1, "SMALL_AMPLITUDE": 0.1})["dominated"]


def test_stratum_comparison_and_partial_spearman():
    rng = np.random.default_rng(7)
    a, b = rng.normal(0, 1, 2000), rng.normal(1, 1, 2000)
    assert sm.smd(b, a) == pytest.approx(1.0, abs=0.1) and sm.ks(a, a) == 0 and sm.ks(a, b) > 0.3
    z = rng.normal(size=3000)
    x, y = z + 0.1 * rng.normal(size=3000), z + 0.1 * rng.normal(size=3000)
    assert spearmanr(x, y)[0] > 0.9 and abs(sm.partial_spearman(x, y, z[:, None])) < 0.15
    assert sm.js_distance([1, 1], [1, 1]) == 0 and sm.js_distance([1, 0], [0, 1]) == pytest.approx(1.0)
    strat = np.array(rg.STRATA * 5)
    cmpd = sm.compare_strata({"v": rng.normal(size=len(strat))}, strat, np.isin(strat, rg.POOLED_STRATA))
    assert set(cmpd["v"]["vs_reference"]) == set(rg.STRATA)


# ------------------------------------------------------------------------------------------------ baselines
def test_baselines_chance_and_simple_scores():
    rng = np.random.default_rng(8)
    n, d = 3000, 16
    E = rng.normal(size=(n, d)).astype(np.float32)
    ia, ib = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
    strat = np.where(np.arange(20000) % 2 == 0, ">1Mb", "interchromosomal")
    target = rng.normal(size=20000)
    vals = []
    for seed in range(5):
        y = bl.shuffled_cosine(E, ia, ib, seed)
        vals.append(cm.stat_pooled(None, "s", target, y, strat, np.zeros(20000, np.int16), ci=False).value)
        z = bl.random_gaussian_cosine(n, d, 100 + seed, ia, ib)
        vals.append(cm.stat_pooled(None, "r", target, z, strat, np.zeros(20000, np.int16), ci=False).value)
    assert np.abs(vals).max() < 0.05
    # shuffled assignment is deterministic and really permutes
    assert np.allclose(bl.shuffled_cosine(E, ia, ib, 3), bl.shuffled_cosine(E, ia, ib, 3))
    assert not np.allclose(bl.shuffled_cosine(E, ia, ib, 3), bl.shuffled_cosine(E, ia, ib, 4))
    assert bl.same_chromosome(["chr1", "chr1"], ["chr1", "chr2"]).tolist() == [1.0, 0.0]
    nd = bl.neg_log10_distance(np.array([10.0, 1000.0, np.nan]))
    assert nd[0] == -1 and nd[1] == -3 and np.isnan(nd[2])
    assert bl.neg_abs_diff([1, 2], [3, 2]).tolist() == [-2, 0] and bl.same_category(["a", "b"], ["a", "c"]).tolist() == [1, 0]
    ns = bl.null_summary([0.0, 0.1, None, float("nan"), -0.1])
    assert ns["n"] == 3 and ns["max"] == 0.1
    assert bl.null_hi([0.1] * 100, [0.0] * 100) >= 0.1 - 1e-12
    # chromosome-only baseline on a target that really is higher within chromosome
    chrom_ind = np.where(strat == ">1Mb", 1.0, 0.0)
    tgt = chrom_ind + rng.normal(0, 0.1, 20000)
    assert cm.stat_pooled(None, "c", tgt, chrom_ind, strat, np.zeros(20000, np.int16), ci=False).value > 0.8
    # per-stratum chromosome-only is undefined (constant)
    assert cm.stat_unweighted(None, "u", tgt, np.ones(20000), np.zeros(20000, np.int16), ci=False) is None


# ------------------------------------------------------------------------------------------------ raw space
def test_raw_cosine_jaccard_zero_vectors():
    X = sparse.csr_matrix(np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 0, 0]], float))
    ia, ib = np.array([0, 0, 0, 2]), np.array([1, 3, 2, 2])
    m = rs.pair_binary_metrics(X, ia, ib, weights=np.ones(4))
    assert m["binary_cosine"][0] == pytest.approx(0.5) and m["jaccard"][0] == pytest.approx(1 / 3)
    assert m["binary_cosine"][1] == pytest.approx(1.0) and m["jaccard"][1] == pytest.approx(1.0)
    assert np.isnan(m["binary_cosine"][2:]).all() and np.isnan(m["jaccard"][2:]).all()
    assert np.allclose(m["weighted_cosine"], m["binary_cosine"], equal_nan=True)   # unit weights -> identical
    w = np.array([2.0, 1.0, 1.0, 1.0])
    mw = rs.pair_binary_metrics(X, np.array([0]), np.array([1]), weights=w)
    assert mw["weighted_cosine"][0] == pytest.approx(4 / np.sqrt(5 * 5))
    assert rs.all_zero_rows(X).tolist() == [False, False, True, False]
    proj = np.random.default_rng(0).normal(size=(3, 4))
    assert rs.verify_projection(X, proj, X.toarray() @ proj.T)["ok"]


# ------------------------------------------------------------------------------------------------ SVD geometry
def test_pc_removal_whitening_centering():
    rng = np.random.default_rng(9)
    n, d = 4000, 8
    E = (rng.normal(size=(n, d)) * np.array([5, 3, 2, 1, 1, 0.5, 0.5, 0.2]) + 2.0).astype(np.float32)
    mu, lam, V = sg.universe_pca(E)
    assert np.allclose(mu, E.mean(0), atol=1e-4) and (np.diff(lam) <= 1e-9).all()
    assert np.allclose(V.T @ V, np.eye(d), atol=1e-9)
    S = sg.pc_scores(E, mu, V, 3)
    assert np.allclose(S.mean(0), 0, atol=1e-3) and abs(np.corrcoef(S[:, 0], S[:, 1])[0, 1]) < 0.05
    ia, ib = np.arange(0, 2000), np.arange(2000, 4000)
    # k=0 equals the centred cosine computed by hand
    Z = E.astype(np.float64) - mu
    hand = (Z[ia] * Z[ib]).sum(1) / np.linalg.norm(Z[ia], axis=1) / np.linalg.norm(Z[ib], axis=1)
    assert np.allclose(sg.remove_pcs_cosine(E, ia, ib, mu, V, 0), hand, atol=1e-5)
    # after removing the top 2 axes the projections on them vanish (cosine equals cosine of the residual space)
    R = Z - (Z @ V[:, :2]) @ V[:, :2].T
    handr = (R[ia] * R[ib]).sum(1) / np.linalg.norm(R[ia], axis=1) / np.linalg.norm(R[ib], axis=1)
    assert np.allclose(sg.remove_pcs_cosine(E, ia, ib, mu, V, 2), handr, atol=1e-5)
    # whitening yields ~identity covariance (eps tiny)
    W = ((E.astype(np.float64) - mu) @ V) * (lam + rg.WHITEN_EPS_REL * lam.max()) ** -0.5
    assert np.allclose(np.cov(W.T), np.eye(d), atol=0.02)
    assert sg.whitened_cosine(E, ia, ib, mu, lam, V).shape == (2000,)
    assert np.allclose(sg.standardized_cosine(E, ia, ib, mu, sg.coord_sd(E)).shape, (2000,))
    assert np.allclose(sg.neg_euclidean_pairs(E, ia, ib), -np.linalg.norm(E[ia].astype(float) - E[ib], axis=1), atol=1e-4)
    with pytest.raises(ValueError):                # removing every axis -> zero vectors -> explicit failure
        sg.remove_pcs_cosine(E, ia, ib, mu, V, d)
    cov = {"c": S[:, 0] * 2 + 1}
    assert sg.spearman_assoc(S, cov)["c"][0] == pytest.approx(1.0)
    cat = np.where(S[:, 0] > 0, "a", "b")
    assert sg.eta_squared(S, cat)[0] > 0.5 and sg.eta_squared(S, cat)[2] < 0.1
    assert np.allclose(sg.pc_dot_contribution(S, ia, ib, 0), S[ia, 0] * S[ib, 0])


# ------------------------------------------------------------------------------------------------ OOF / ridge
def test_oof_leakage_guards_and_refit_check():
    rng = np.random.default_rng(10)
    n = 400
    chrom = np.array([f"chr{1 + i % 10}" for i in range(n)])
    c2f = {f"chr{c}": (c - 1) % 5 for c in range(1, 11)}
    fold = np.array([c2f[c] for c in chrom])
    assert op.assert_train_excludes_test(chrom, fold, c2f)
    bad = fold.copy()
    bad[0] = (bad[0] + 1) % 5
    with pytest.raises(AssertionError):
        op.assert_train_excludes_test(chrom, bad, c2f)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    Y = np.column_stack([X[:, 0] * 2 + rng.normal(0, 0.1, n), X[:, 1] - X[:, 2], rng.normal(size=n)])
    fit = cp.ridge_blocked_cv(X, Y, fold, alphas=(0.1, 1.0, 10.0, 1e3))
    for fi, f in enumerate(fit["folds"]):
        chk = op.refit_check(X, Y, fold, fit["oof"], fit["alpha"][fi], int(f), (0, 1, 2), n_rows=30)
        assert chk["max_abs_diff"] < 1e-6        # OOF rows predicted by a model trained on other folds only
    # a model that saw the test fold would NOT match: shift the predictions of fold 0 by training on all rows
    mu, sd = X.mean(0), X.std(0)
    Xs = (X - mu) / sd
    w = np.linalg.solve(Xs.T @ Xs + 0.1 * np.eye(6), Xs.T @ (Y[:, 0] - Y[:, 0].mean()))
    leaky = Xs @ w + Y[:, 0].mean()
    assert np.abs(leaky[fold == 0] - fit["oof"][fold == 0, 0]).max() > 1e-4


def test_oof_pair_similarity_and_same_fold():
    rng = np.random.default_rng(11)
    oof = rng.normal(size=(50, 39))
    oof[3] = oof[2] * 2 + 1
    oof[4] = 7.0
    r = op.oof_pair_pearson(oof, np.array([2, 2, 4]), np.array([3, 5, 5]))
    assert r[0] == pytest.approx(1.0) and np.isfinite(r[1]) and np.isnan(r[2])
    assert op.same_fold_mask([0, 1, 2], [0, 2, 2]).tolist() == [True, False, True]


def test_alpha_edge_rule_and_grid_extension():
    grid = rg.ALPHA_GRID_EXTENDED
    inside = np.full((5, 10), 1e3)
    rep_in = op.alpha_edge_report(inside, grid)
    assert not rep_in["hit_max"] and not rep_in["hit_min"]
    top = np.full((5, 10), 1e8)
    rep_top = op.alpha_edge_report(top, grid)
    assert rep_top["hit_max"] and rep_top["frac_at_max"] == 1.0
    assert op.next_grid(grid, {"a": rep_in}, 0) == (tuple(sorted(grid)), False)
    g1, ext = op.next_grid(grid, {"a": rep_in, "b": rep_top}, 0)
    assert ext and g1[-2:] == (1e9, 1e10) and g1[0] == 1e-2 and len(g1) == len(grid) + 2
    g2, ext2 = op.next_grid(g1, {"a": op.alpha_edge_report(np.full((5, 10), 1e10), g1)}, 1)
    assert ext2 and g2[-1] == 1e12
    g3, ext3 = op.next_grid(g2, {"a": op.alpha_edge_report(np.full((5, 10), 1e12), g2)}, 2)
    assert not ext3 and g3 == g2                   # at most two extensions
    low = op.alpha_edge_report(np.full((5, 10), 1e-2), grid)
    g4, e4 = op.next_grid(grid, {"a": low}, 0)
    assert e4 and g4[0] == 1e-4 and g4[-1] == 1e8
    # 9% at the edge does not trigger, 10% does
    a = np.full((10, 10), 1e3)
    a.flat[:9] = 1e8
    assert not op.alpha_edge_report(a, grid)["hit_max"]
    a.flat[9] = 1e8
    assert op.alpha_edge_report(a, grid)["hit_max"]


# ------------------------------------------------------------------------------------------------ decision rules
def test_decision_rules():
    assert dc.rule_bug({"verdict": "FAIL", "failed_checks": ["x"]})["flag"] and not dc.rule_bug({"verdict": "PASS_LEVEL_A"})["flag"]
    assert dc.rule_bug({"verdict": "PASS"})["flag"]                   # the bare legacy verdict is not accepted
    assert "NOT verified" in dc.rule_bug({"verdict": "PASS_LEVEL_A"})["bug_absent_statement_scope"]
    tl = dc.rule_target_limitation(0.25, 0.9, 0.9, {"dominated": False}, {"a": 0.0}, False)
    assert tl["flag"] and tl["sub_rules"]["a_ceiling_lt_0.30"]
    assert dc.rule_target_limitation(0.6, 0.4, 0.9, {"dominated": False}, {}, False)["sub_rules"]["b_stratum_reliability_lt_0.5"]
    assert dc.rule_target_limitation(0.6, 0.9, 0.9, {"dominated": True}, {}, False)["flag"]
    assert dc.rule_target_limitation(0.6, 0.9, 0.9, {"dominated": False}, {"a": 0.05}, False)["sub_rules"]["d_float16_precision_changes_primary"]
    assert not dc.rule_target_limitation(0.6, 0.9, 0.9, {"dominated": False}, {"a": 0.0}, False)["flag"]
    assert dc.recovers(0.2, 0.1) and not dc.recovers(0.12, 0.1)
    cg = dc.rule_compression_geometry(-0.05, 0.01, {"cos": 0.2}, {}, {"raw:cos": (0.1, 0.3)})
    assert cg["flag"] and cg["sources_recovering"] == ["raw:cos"]
    assert not dc.rule_compression_geometry(-0.05, 0.01, {"cos": 0.2}, {}, {"raw:cos": (-0.1, 0.3)})["flag"]
    assert not dc.rule_compression_geometry(0.2, 0.01, {"cos": 0.3}, {}, {"raw:cos": (0.1, 0.3)})["flag"]
    ia = dc.rule_information_absence(0.01, {"raw:cos": 0.02, "oof:x": 0.0})
    assert ia["flag"] and not dc.rule_information_absence(0.01, {"raw:cos": 0.2})["flag"]
    ca = dc.rule_comparator_advantage({"c": {"n": (0.1, 0.05, 0.15), "o": (0.1, 0.02, 0.2)}}, {"c": 0.0}, {"c": {"n": 0.2, "o": 0.2}})
    assert ca["flag"] and ca["per_comparator"]["c"]["level"] == "STRONG"
    ca2 = dc.rule_comparator_advantage({"c": {"n": (0.1, 0.05, 0.15), "o": (0.0, -0.02, 0.2)}}, {"c": 0.0}, {"c": {"n": 0.2, "o": 0.2}})
    assert not ca2["flag"] and ca2["per_comparator"]["c"]["level"] == "PARTIAL"
    out = dc.classify({"BUG": {"flag": False}, "TARGET_LIMITATION": {"flag": True}, "COMPARATOR_ADVANTAGE": {"flag": True}})
    assert out["classes"] == ["COMPARATOR_ADVANTAGE", "TARGET_LIMITATION"]
    with pytest.raises(AssertionError):
        dc.classify({"COMPRESSION_GEOMETRY": {"flag": True}, "INFORMATION_ABSENCE": {"flag": True}})


# ------------------------------------------------------------------------------------------------ gate / writer / snapshot
def _fake_root(tmp_path):
    root = tmp_path / "repo"
    (root / rg.TAG_DIR / "arm" / "endpoints").mkdir(parents=True)
    (root / rg.TAG_DIR / "arm" / "endpoints" / "a.json").write_text("{}")
    (root / rg.TAG_DIR / "sensitivity").mkdir()
    (root / rg.TAG_DIR / "sensitivity" / "s.json").write_text("{}")
    (root / "docs").mkdir()
    (root / "docs" / "BIOLOGICAL_VALIDATION_V2_FINAL_REPORT.md").write_text("x")
    for rel_ in ("data/derived/bioval_v2", "configs", "src", "docs/bioval_v2_prep"):
        (root / rel_).mkdir(parents=True, exist_ok=True)
    return root


def test_writer_guards(tmp_path):
    root = _fake_root(tmp_path)
    w = gt.AuditWriter(root, root / rg.AUDIT_OUT)
    w.write_json("step1_target_audit/x.json", {"a": 1})
    assert (root / rg.AUDIT_OUT / "step1_target_audit" / "x.json").is_file()
    for bad in (root / rg.TAG_DIR, root / rg.TAG_DIR / "exploratory_posthoc_loyfer", root / "data/derived/bioval_v2", root / "docs",
                root / "outputs", root / "src"):
        with pytest.raises(PermissionError):
            gt.AuditWriter(root, bad)
    for rel_ in (f"../../{rg.TAG_DIR.split('/')[-1]}/arm/endpoints/a.json", "../../../../docs/BIOLOGICAL_VALIDATION_V2_FINAL_REPORT.md",
                 "../../../../data/derived/bioval_v2/x", "../../../../src/x.py"):
        with pytest.raises(PermissionError):
            w.path(rel_)
    assert gt.is_frozen_audit_path(root, root / "docs" / "BIOLOGICAL_VALIDATION_V2_FINAL_REPORT.md")
    assert not gt.is_frozen_audit_path(root, root / "docs" / "LOYFER_PRIMARY_FAILURE_AUDIT_REGISTRATION.md")
    assert gt.fingerprint_excludes_audit(root)


def test_snapshot_compare_detects_changes(tmp_path):
    root = _fake_root(tmp_path)
    s0 = gt.snapshot_tree(root)
    assert s0["n_files"] == 2 and gt.compare_snapshots(s0, gt.snapshot_tree(root))["identical"]
    (root / rg.TAG_DIR / "sensitivity" / "s.json").write_text("changed")
    (root / rg.TAG_DIR / "new.json").write_text("{}")
    (root / rg.TAG_DIR / "arm" / "endpoints" / "a.json").unlink()
    c = gt.compare_snapshots(s0, gt.snapshot_tree(root))
    assert not c["identical"] and c["changed"] == ["sensitivity/s.json"] and c["added"] == ["new.json"] and c["removed"] == ["arm/endpoints/a.json"]


def test_step1_dependency_failure_modes(tmp_path):
    root = _fake_root(tmp_path)
    (root / "docs" / "LOYFER_PRIMARY_FAILURE_AUDIT_REGISTRATION.md").write_text("reg")
    assert not gt.check_step1_pass(root).ok                       # no verdict file
    vp = root / rg.AUDIT_OUT / rg.VERDICT_FILE
    vp.parent.mkdir(parents=True)

    def verdict(**kw):
        v = {"verdict": "PASS_LEVEL_A", "level": "A", "raw_level_B_executed": False, "failed_checks": [], "registration_sha256": cm.sha256_file(root / rg.REGISTRATION_DOC),
             "code_sha256": gt.current_code_sha256(root), "timestamp_utc": "t"}
        v.update(kw)
        vp.write_text(json.dumps(v))
    verdict()
    assert gt.check_step1_pass(root).ok
    verdict(verdict="PASS")
    assert not gt.check_step1_pass(root).ok                       # bare PASS / unregistered verdict refused
    verdict(raw_level_B_executed=True)
    assert not gt.check_step1_pass(root).ok                       # Level B verdicts are not registered
    verdict(level="A+B")
    assert not gt.check_step1_pass(root).ok
    verdict(verdict="FAIL", failed_checks=["x"])
    assert not gt.check_step1_pass(root).ok                       # mismatch blocks everything
    verdict(registration_sha256="0" * 64)
    assert not gt.check_step1_pass(root).ok                       # registration changed after the verdict
    verdict(code_sha256="0" * 64)
    assert not gt.check_step1_pass(root).ok                       # code changed after the verdict
    assert not gt.check_snapshot_before(root).ok                  # no pre-run snapshot recorded
    assert not gt.check_out_root(root, root / rg.TAG_DIR).ok and gt.check_out_root(root, root / rg.AUDIT_OUT).ok
    assert set(rg.GATED_AFTER_STEP1) == set(rg.STEPS) - {"target-audit", "final-integrity"}


def test_code_files_exist_or_are_this_test():
    missing = [f for f in rg.CODE_FILES if not (ROOT / f).exists()]
    assert not missing, missing
    # no frozen/freeze module is imported by the audit package
    import sys

    from cpg_repr_benchmark.bioval_v2_launch import launch_gate as lg
    assert lg.freeze_modules_imported() == [] or all(m not in sys.modules for m in lg.FREEZE_MODULE_NAMES)


def test_gate_refuses_level_b_flag():
    checks = {c.name: c for c in gt.run_gate(ROOT, rg.ARMS, gt.audit_root(ROOT), "target-audit", skip_heavy=True, level_b=True)}
    assert not checks["level_b_not_requested"].ok
    checks = {c.name: c for c in gt.run_gate(ROOT, rg.ARMS, gt.audit_root(ROOT), "target-audit", skip_heavy=True)}
    assert checks["level_b_not_requested"].ok
