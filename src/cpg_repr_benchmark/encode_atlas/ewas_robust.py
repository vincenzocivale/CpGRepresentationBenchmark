"""Robustness analysis of matched-negative EWAS / clock-set membership probes (follow-up, Exp 5/6).

Protocol (identical to ``ewas.py`` except where stated):
  * positives  = members of an external CpG set (``data/bio_annotations/known_sets/*.npy``, EWAS Catalog/Atlas
    and the Horvath / Hannum / PhenoAge clocks; clock sets are separate rows, ``family == 'clock'``);
  * negatives  = one unlisted CpG per positive drawn without replacement inside the same stratum
    (chromosome x CpG context x gene region x core-breadth decile), <= 2,000 pairs;
  * classifier = StandardScaler (fit in the training fold only) + LogisticRegression(C=1), 5 chromosome folds
    (GroupKFold), pooled out-of-fold (OOF) scores -> pooled AUC;
  * NEW: ``n_seeds`` independent matching draws (seeds ``base_seed + i``; ``base_seed`` = mask_seed 17001 reproduces
    the earlier single draw), native and compacted sequence embeddings, and an optional probe-type stratum.

Variants
  ``base``  : original strata;
  ``probe`` : original strata + Infinium design type (I / II) and restriction to loci with an unambiguous type.

Uncertainty ("full procedure" interval)
  Each bootstrap replicate first draws one matching seed uniformly from the ``n_seeds`` draws, then resamples that
  seed's matched pairs with replacement *within chromosome* (a pair lives on one chromosome by construction of the
  strata). The same pair indices are applied to every representation, so paired differences are exact paired
  bootstraps. Point estimates are means over matching seeds of the pooled-OOF AUC. Percentile intervals over the
  10,000 mixture replicates therefore contain both matching-draw and pair-sampling variability. Two-sided
  bootstrap p = 2 * min(P(d<=0), P(d>=0)) with a 1/(B+1) floor; BH is applied across evaluable sets within each
  variant x comparison. Matching draws are NOT treated as independent biological replicates.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cpg_repr_benchmark.data.coordinates import encode_many
from cpg_repr_benchmark.encode_atlas.campaign import arm_id

REPRESENTATIONS = {
    'functional_256': 'campaign:full',
    'cpgpt_large_native512': 'data/cache/representations/cpgpt_locus_large.h5',
    'deepcpg_hcc_native128': 'data/cache/representations/deepcpg_dna_locus.h5',
    'cpgpt_large_svd256': 'campaign:fm/cpgpt_locus_large',
    'deepcpg_hcc_pad256': 'campaign:fm/deepcpg_dna_locus',
    'context_only_256': 'campaign:context_only',
}
COMPARISONS = [('functional_256', 'cpgpt_large_native512'), ('functional_256', 'deepcpg_hcc_native128'),
               ('functional_256', 'cpgpt_large_svd256'), ('functional_256', 'deepcpg_hcc_pad256'),
               ('functional_256', 'context_only_256')]
PRIMARY = COMPARISONS[:2]
_G = {}   # fork-shared read-only globals


def probe_type_codes(registry: pd.DataFrame, manifest: str) -> np.ndarray:
    """0 = unknown/ambiguous, 1 = Infinium I, 2 = Infinium II via hg38 (chr, pos=CpG_beg+1) join to EPICv2."""
    m = pd.read_csv(manifest, sep='\t', usecols=['CpG_chrm', 'CpG_beg', 'type', 'target'], dtype=str)
    m = m[m.CpG_beg.notna() & m.CpG_chrm.notna() & m.type.notna() & (m.target == 'CG')].copy()
    m['pos'] = m.CpG_beg.astype('int64') + 1
    key = m.groupby(['CpG_chrm', 'pos']).type.agg(lambda s: ''.join(sorted(set(s)))).rename('t').reset_index()
    k = registry[['chr', 'pos']].merge(key, left_on=['chr', 'pos'], right_on=['CpG_chrm', 'pos'], how='left')
    return k['t'].map({'I': 1, 'II': 2}).fillna(0).to_numpy(np.int8)


def _strata(registry, dense):
    return (registry.chr.astype(str).to_numpy() + ':' + np.argmax(dense[:, :4], 1).astype(str)
            + ':' + np.argmax(dense[:, 4:8], 1).astype(str)
            + ':' + np.minimum((dense[:, 22] * 10).astype(int), 9).astype(str))


def _index_strata(strata):
    uniq, codes = np.unique(strata, return_inverse=True)
    order = np.argsort(codes, kind='stable')
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    return codes, order, bounds


def match_pairs(labels, index, eligible, seed, maximum=2000):
    """Same draw order as ``ewas.matched_membership`` (strata in sorted order) -> array (n, 2) [pos, neg]."""
    codes, order, bounds = index
    rng = np.random.default_rng(seed)
    pos_all = np.flatnonzero(labels & eligible)
    pos_codes = codes[pos_all]
    pairs = []
    for c in np.unique(pos_codes):
        pos = pos_all[pos_codes == c]
        members = order[bounds[c]:bounds[c + 1]]
        members = np.sort(members)
        neg = members[~labels[members] & eligible[members]]
        size = min(len(pos), len(neg))
        if size:
            pairs.extend(zip(rng.choice(pos, size, replace=False), rng.choice(neg, size, replace=False)))
    if not pairs:
        return np.empty((0, 2), int)
    pairs = np.asarray(pairs)
    return pairs[rng.choice(len(pairs), min(maximum, len(pairs)), replace=False)]


def pooled_oof(x, y, chrom):
    p = np.full(len(y), np.nan)
    fold = np.zeros(len(y), np.int8)
    for f, (tr, te) in enumerate(GroupKFold(5).split(x, y, chrom)):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000))
        model.fit(x[tr], y[tr])
        p[te] = model.predict_proba(x[te])[:, 1]
        fold[te] = f
    return p, fold


def fast_auc(pos, neg):
    n = len(pos)
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:n].sum() - n * (n + 1) / 2) / (n * n)


def _stage1(task):
    variant, name, seed = task
    g = _G
    labels = g['labels'][name]
    elig = g['probe'] > 0 if variant == 'probe' else np.ones(len(labels), bool)
    index = g['index_probe'] if variant == 'probe' else g['index_base']
    P = match_pairs(labels, index, elig, seed)
    n = len(P)
    reg_chr = g['chr']
    rows = P.ravel()
    pid = np.repeat(np.arange(n), 2)
    o = np.argsort(rows, kind='stable')            # sorted rows, as in ewas.py, for identical GroupKFold ties
    rows, pid = rows[o], pid[o]
    y = labels[rows]
    chrom = reg_chr[rows]
    info = dict(variant=variant, set=name, seed=seed, n_pairs=n, n_chrom=int(len(np.unique(chrom))))
    pt = g['probe'][rows]
    known = pt > 0
    info.update(pos_I=int(((pt == 1) & y).sum()), pos_II=int(((pt == 2) & y).sum()),
                neg_I=int(((pt == 1) & ~y).sum()), neg_II=int(((pt == 2) & ~y).sum()),
                pos_unknown=int(((pt == 0) & y).sum()), neg_unknown=int(((pt == 0) & ~y).sum()))
    if len(rows) < 100 or info['n_chrom'] < 5:
        info['status'] = 'insufficient_matched_support'
        return info, None, None
    scores, fold = {}, None
    for rep in REPRESENTATIONS:
        x = g['emb'][rep][rows].astype(np.float32)
        scores[rep], f = pooled_oof(x, y, chrom)
        fold = f
    info['status'] = 'complete'
    frame = pd.DataFrame(dict(variant=variant, set=name, match_seed=seed, cpg_idx=g['cpg_idx'][rows],
                              chromosome=pd.Categorical(chrom), pair_id=pid.astype(np.int32), y_true=y,
                              fold=fold))
    long = []
    for rep, s in scores.items():
        d = frame.copy()
        d['representation'] = rep
        d['oof_score'] = s.astype(np.float32)
        long.append(d)
    # per-pair arrays for the bootstrap (positive / negative score per pair, chromosome per pair)
    pos_i = np.flatnonzero(y)[np.argsort(pid[y])]
    neg_i = np.flatnonzero(~y)[np.argsort(pid[~y])]
    arr = dict(chrom=chrom[pos_i], **{rep: (s[pos_i], s[neg_i]) for rep, s in scores.items()})
    return info, pd.concat(long), arr


def _stage2(args):
    variant, name, per_seed, B, rng_seed = args
    seeds = sorted(per_seed)
    chroms = {s: per_seed[s]['chrom'] for s in seeds}
    groups = {s: [np.flatnonzero(chroms[s] == c) for c in np.unique(chroms[s])] for s in seeds}
    reps = list(REPRESENTATIONS)
    auc_seed = {rep: np.array([fast_auc(*per_seed[s][rep]) for s in seeds]) for rep in reps}
    rng = np.random.default_rng(rng_seed)
    draws = rng.integers(0, len(seeds), B)
    boot = {rep: np.empty(B) for rep in reps}
    for b in range(B):
        s = seeds[draws[b]]
        ix = np.concatenate([rng.choice(gr, len(gr)) for gr in groups[s]])
        for rep in reps:
            pos, neg = per_seed[s][rep]
            boot[rep][b] = fast_auc(pos[ix], neg[ix])
    rows = []
    for rep in reps:
        lo, hi = np.percentile(boot[rep], [2.5, 97.5])
        rows.append(dict(variant=variant, set=name, kind='auc', representation=rep, comparison='',
                         estimate=auc_seed[rep].mean(), between_seed_sd=auc_seed[rep].std(ddof=1),
                         ci_lo=lo, ci_hi=hi, seed_min=auc_seed[rep].min(), seed_max=auc_seed[rep].max()))
    for a, b_ in COMPARISONS:
        d_seed = auc_seed[a] - auc_seed[b_]
        d = boot[a] - boot[b_]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = min(1.0, 2 * (min((d <= 0).mean(), (d >= 0).mean()) + 1 / (len(d) + 1)))
        rows.append(dict(variant=variant, set=name, kind='delta', representation=a, comparison=f'{a} - {b_}',
                         estimate=d_seed.mean(), between_seed_sd=d_seed.std(ddof=1), ci_lo=lo, ci_hi=hi,
                         seed_min=d_seed.min(), seed_max=d_seed.max(), p_boot=p,
                         n_seeds_delta_positive=int((d_seed > 0).sum()), n_seeds=len(seeds)))
    return rows


def _load_embeddings(out: Path, registry):
    emb = {}
    for rep, src in REPRESENTATIONS.items():
        if src.startswith('campaign:'):
            path = out / 'embeddings' / (arm_id(src[9:]) + '.h5')
        else:
            path = Path(src)
        with h5py.File(path) as h:
            assert np.array_equal(h['cpg_idx'][:], registry.cpg_idx.to_numpy()), f'{rep}: order differs'
            emb[rep] = h['embedding'][:]
    return emb


def run(cfg: dict, outdir: Path, *, n_seeds=20, boot=10000, workers=32, manifest=None, only_sets=None,
        variants=('base', 'probe')):
    root = Path(__file__).resolve().parents[3]
    out = Path(cfg.get('output', 'outputs/encode_atlas_v1'))
    outdir.mkdir(parents=True, exist_ok=True)
    registry = pd.read_parquet(out / 'loci.parquet')
    canonical = encode_many(registry.chr, registry.pos)
    with h5py.File(cfg['features']) as h:
        dense = h['dense'][:].astype(np.float32)
    strata = _strata(registry, dense)
    probe = probe_type_codes(registry, manifest)
    pd.DataFrame(dict(cpg_idx=registry.cpg_idx, probe_type=probe)).to_parquet(outdir / 'probe_type_annotation.parquet')
    sets = sorted((root / 'data/bio_annotations/known_sets').glob('*.npy'))
    if only_sets:
        sets = [s for s in sets if s.stem in only_sets]
    labels = {s.stem: np.isin(canonical, np.load(s, allow_pickle=False)) for s in sets}
    _G.update(labels=labels, probe=probe, chr=registry.chr.to_numpy(), cpg_idx=registry.cpg_idx.to_numpy(),
              index_base=_index_strata(strata), index_probe=_index_strata(strata + ':' + probe.astype(str)),
              emb=_load_embeddings(out, registry))
    base = int(cfg['mask_seed'])
    tasks = [(v, n, base + i) for v in variants for n in labels for i in range(n_seeds)]
    ctx = mp.get_context('fork')
    with ctx.Pool(workers) as pool:
        stage1 = pool.map(_stage1, tasks, chunksize=1)
    infos = pd.DataFrame([r[0] for r in stage1])
    infos.to_csv(outdir / 'matching_summary_per_seed.csv', index=False)
    long = pd.concat([r[1] for r in stage1 if r[1] is not None])
    long['variant'] = long['variant'].astype('category')
    long['set'] = long['set'].astype('category')
    long['representation'] = long['representation'].astype('category')
    long.to_parquet(outdir / 'oof_predictions.parquet', index=False)
    per_seed = {}
    for (info, _, arr) in stage1:
        if arr is not None:
            per_seed.setdefault((info['variant'], info['set']), {})[info['seed']] = arr
    jobs = [(v, n, ps, boot, 20260929 + k) for k, ((v, n), ps) in enumerate(sorted(per_seed.items()))]
    with ctx.Pool(min(workers, len(jobs))) as pool:
        stage2 = pool.map(_stage2, jobs, chunksize=1)
    res = pd.DataFrame([r for rows in stage2 for r in rows])
    res['family'] = np.where(res.set.str.endswith('_clock'), 'clock', 'ewas')
    res['q_bh'] = np.nan
    for (v, c), idx in res[res.kind == 'delta'].groupby(['variant', 'comparison']).groups.items():
        res.loc[idx, 'q_bh'] = false_discovery_control(res.loc[idx, 'p_boot'].to_numpy())
    res.to_csv(outdir / 'summary_auc_delta.csv', index=False)
    # matching / probe-type balance (mean over seeds, pooled counts)
    bal = infos[infos.status == 'complete'].groupby(['variant', 'set'])[
        ['n_pairs', 'pos_I', 'pos_II', 'neg_I', 'neg_II', 'pos_unknown', 'neg_unknown']].mean().reset_index()
    bal['typeI_frac_pos'] = bal.pos_I / (bal.pos_I + bal.pos_II)
    bal['typeI_frac_neg'] = bal.neg_I / (bal.neg_I + bal.neg_II)
    bal.to_csv(outdir / 'probe_type_balance.csv', index=False)
    ev = infos.groupby(['variant', 'set']).agg(n_seeds_evaluable=('status', lambda s: int((s == 'complete').sum())),
                                              pairs_mean=('n_pairs', 'mean'), pairs_min=('n_pairs', 'min'),
                                              pairs_max=('n_pairs', 'max')).reset_index()
    ev['status'] = np.where(ev.n_seeds_evaluable == n_seeds, 'EVALUABLE',
                            np.where(ev.n_seeds_evaluable == 0, 'NOT EVALUABLE', 'PARTIAL'))
    ev.to_csv(outdir / 'evaluable_sets.csv', index=False)
    (outdir / 'protocol.json').write_text(json.dumps(dict(
        n_matching_seeds=n_seeds, seeds=[base + i for i in range(n_seeds)], bootstrap_replicates=boot,
        representations=REPRESENTATIONS, comparisons=COMPARISONS, probe_manifest=manifest,
        probe_type_coverage=float((probe > 0).mean()), classifier='StandardScaler(fold-wise)+LogReg(C=1)',
        folds='GroupKFold(5) by chromosome', combination='seed-mixture bootstrap, pairs resampled within chromosome',
    ), indent=1))
    return res
