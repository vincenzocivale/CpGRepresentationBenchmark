"""Synthetic tests for the EWAS-v2 design primitives (no data, no embeddings)."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ewasv2", ROOT / "scripts/ewas_v2/ewas_v2_design.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def universe(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    chrom = rng.choice([f"chr{i}" for i in range(1, 6)], n)
    pt = rng.choice([1, 2, np.nan], n, p=[0.3, 0.65, 0.05])
    return chrom, pt


def test_tiers():
    assert [D.support_tier(x) for x in (500, 200, 199, 100, 99, 0)] == [
        "main-quality", "main-quality", "exploratory", "exploratory", "insufficient", "insufficient"]


def test_containment_rule():
    sets = {"big": np.arange(100), "small": np.arange(60, 100), "other": np.arange(200, 230)}
    rows = D.containment_flags(sets)
    assert len(rows) == 1 and rows[0]["small"] == "small" and rows[0]["large"] == "big"
    assert rows[0]["containment"] == 1.0 and rows[0]["n_set_difference"] == 0


def test_background_deterministic_and_embedding_free():
    chrom, pt = universe()
    rng = np.random.default_rng(1)
    pos = rng.choice(len(chrom), 300, replace=False)
    excl = np.zeros(len(chrom), bool)
    excl[rng.choice(len(chrom), 400, replace=False)] = True
    excl[pos] = True
    a = D.technical_background(pos, chrom, pt, excl, seed=17)
    b = D.technical_background(pos[::-1].copy(), chrom, pt, excl, seed=17)   # input order must not matter
    assert D.pairs_digest(a["pairs"]) == D.pairs_digest(b["pairs"])
    assert a["n_matched_positives"] + a["n_unmatched"] + a["n_dropped_unknown_probe_type"] == 300
    for p, (n,) in a["pairs"]:
        assert chrom[p] == chrom[n] and pt[p] == pt[n] and not excl[n]
    negs = [n for _, (n,) in a["pairs"]]
    assert len(set(negs)) == len(negs)                                   # without replacement
    assert D.pairs_digest(a["pairs"]) != D.pairs_digest(D.technical_background(pos, chrom, pt, excl, seed=18)["pairs"])


def test_concordance_and_shared_draw_table():
    assert D.matched_pair_concordance([1, 1], [1, 1]) == 0.5
    t = D.draw_table(22)
    assert t.shape == (1000, 22) and np.array_equal(t, D.draw_table(22))


def test_paired_bootstrap_signs():
    rng = np.random.default_rng(3)
    n = 200
    blocks = [f"chr{i}" for i in range(1, 6)]
    pb = np.array(blocks)[rng.integers(0, 5, n)]
    neg = rng.normal(size=n)
    scores = {"good": (neg + rng.normal(1.0, 1, n), neg), "chance": (rng.normal(size=n), neg)}
    out = D.paired_block_bootstrap(blocks, pb, scores, [("good", "chance")], D.draw_table(5, B=200))
    assert out["point"]["good"]["auroc"] > 0.6
    d = out["delta"][("good", "chance", "concordance")]
    assert d["estimate"] > 0 and d["ci"][0] < d["ci"][1]
