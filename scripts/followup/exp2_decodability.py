"""Follow-up Exp 2: direct representation-level decodability of training-patient mean beta.

Reuses the campaign probe (nested chromosome CV, StandardScaler(with_mean=False)+Ridge, alpha grid) but keeps
out-of-fold predictions so chromosome-level bootstrap CIs and paired differences can be computed.
"""
import json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '4'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
from pathlib import Path
import h5py, numpy as np, pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from cpg_repr_benchmark.encode_atlas.features import FeatureData, make_transform  # noqa: E402

V1 = ROOT / 'outputs/encode_atlas_v1'
OUT = V1 / 'followup/exp2'
cfg = __import__('yaml').safe_load(open(ROOT / 'configs/encode_atlas.yaml'))
loci = pd.read_parquet(V1 / 'loci.parquet')
t = np.load(V1 / 'discovery_targets.npz')
rows, mean, count = t['rows'], t['mean'].copy(), t['count']
y_all = mean.copy(); y_all[count < max(20, int(len(t['patients']) * .5))] = np.nan
chrom_all = loci.chr.to_numpy()[rows]
valid = np.isfinite(y_all)
y, chrom = y_all[valid], chrom_all[valid]
vrows = rows[valid]
print('n loci', len(y), 'chromosomes', len(np.unique(chrom)), flush=True)


def probe(x, seed_unused=0):
    preds = np.full(len(y), np.nan); folds = []
    for tr, te in GroupKFold(5).split(x, y, chrom):
        inner = GroupKFold(min(3, len(np.unique(chrom[tr]))))
        m = GridSearchCV(make_pipeline(StandardScaler(with_mean=False), Ridge(solver='lsqr')),
                         {'ridge__alpha': [0.1, 10., 1000.]}, cv=inner, scoring='neg_mean_squared_error', n_jobs=1)
        m.fit(x[tr], y[tr], groups=chrom[tr]); preds[te] = m.predict(x[te])
        folds.append({'test_chr': sorted(set(chrom[te])), 'alpha': m.best_params_['ridge__alpha']})
    return preds, folds


def read_rows(path, native=True):
    with h5py.File(path, 'r') as f:
        ids = f['cpg_idx'][:]; e = f['embedding']
        pos = pd.Series(np.arange(len(ids)), index=ids).loc[loci.cpg_idx.to_numpy()[vrows]].to_numpy()
        order = np.argsort(pos)
        x = np.empty((len(pos), e.shape[1]), np.float32); x[order] = e[pos[order].tolist()]
    return x


def functional128():
    cache = OUT / 'functional_128_probe_rows.npy'
    if cache.exists():
        return np.load(cache)
    ex1 = V1 / 'followup/exp1'
    for c in sorted(ex1.glob('**/*128*.h5')) if ex1.exists() else []:
        print('using exp1 store', c); return read_rows(c)
    data = FeatureData.read(Path(cfg['features']), Path(cfg['catalog']))
    spec = json.load(open(V1 / 'embeddings/full_a18b869b.json'))['spec']
    tr, de = data.matrix(spec['tracks'], spec['dense'], breadth_policy=spec.get('breadth_policy', 'recompute'))
    emb, tf, var = make_transform(tr, de, np.arange(len(data.ids)), width=128, seed=17)
    print('functional-128 variance retained', var, flush=True)
    np.save(cache, emb[vrows]); json.dump({'variance_retained': var, 'n_fit_loci': len(data.ids), 'width': 128},
                                          open(OUT / 'functional_128_meta.json', 'w'))
    return emb[vrows]


def metrics(yv, pv):
    r2 = 1 - ((yv - pv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    return dict(r2=r2, pearson=np.corrcoef(yv, pv)[0, 1], spearman=spearmanr(yv, pv)[0], mse=((yv - pv) ** 2).mean())


def main():
    reps = {}
    E = V1 / 'embeddings'
    reps['Functional-256'] = read_rows(E / 'full_a18b869b.h5')
    reps['Functional-128'] = functional128()
    reps['CpGPT-large native-512'] = read_rows(ROOT / 'data/cache/representations/cpgpt_locus_large.h5')
    reps['CpGPT-large compacted-256'] = read_rows(E / 'fm_cpgpt_locus_large_d3474a48.h5')
    reps['DeepCpG native-128'] = read_rows(ROOT / 'data/cache/representations/deepcpg_dna_locus.h5')
    reps['DeepCpG campaign-256 (std+pad)'] = read_rows(E / 'fm_deepcpg_dna_locus_1facb46c.h5')
    reps['Context-only'] = read_rows(E / 'context_only_2805a394.h5')
    active = {k: int((v.std(0) > 1e-8).sum()) for k, v in reps.items()}
    res = Parallel(n_jobs=7)(delayed(probe)(v) for v in reps.values())
    oof = {k: r[0] for k, r in zip(reps, res)}; alphas = {k: [f['alpha'] for f in r[1]] for k, r in zip(reps, res)}

    # family analyses (raw sparse blocks, exactly as campaign screen) recomputed for CIs
    data = FeatureData.read(Path(cfg['features']), Path(cfg['catalog']))
    groups = json.load(open(V1 / 'groups.json'))
    x_tr, dn = data.tracks[vrows], data.dense[vrows]
    fam_specs = {'Histone': groups['assay/Histone ChIP-seq'], 'H3K4me3': groups['target/H3K4me3'],
                 'TF': groups['assay/TF ChIP-seq'], 'DNase': groups['assay/DNase-seq'],
                 'Context (18 dense)': {'tracks': [], 'dense': list(range(18))},
                 'Breadth (dense 18-22)': groups['dense/breadth'],
                 'Full raw (4165 tracks + 23 dense)': {'tracks': list(range(x_tr.shape[1])), 'dense': list(range(23))}}
    def fam(spec, plus):
        b = sparse.hstack([x_tr[:, spec['tracks']], sparse.csr_matrix(dn[:, spec['dense']])], format='csr')
        if plus: b = sparse.hstack([sparse.csr_matrix(dn[:, :18]), b], format='csr')
        return probe(b)[0]
    jobs = [(k, p) for k in fam_specs for p in (False, True)]
    fres = Parallel(n_jobs=8)(delayed(fam)(fam_specs[k], p) for k, p in jobs)
    foof = {f'{k} | {"plus_context" if p else "alone"}': r for (k, p), r in zip(jobs, fres)}
    np.savez_compressed(OUT / 'oof_predictions.npz', y=y, chrom=chrom, cpg_row=vrows,
                        **{'rep::' + k: v for k, v in oof.items()}, **{'fam::' + k: v for k, v in foof.items()})

    chroms = np.unique(chrom); idx = {c: np.flatnonzero(chrom == c) for c in chroms}
    rng = np.random.default_rng(20260930); B = 2000
    boots = [rng.choice(len(chroms), len(chroms)) for _ in range(B)]
    def bmetric(p, b):
        ii = np.concatenate([idx[chroms[j]] for j in b]); yv, pv = y[ii], p[ii]
        return 1 - ((yv - pv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum(), ((yv - pv) ** 2).mean()
    allp = {**{'rep::' + k: v for k, v in oof.items()}, **{'fam::' + k: v for k, v in foof.items()}}
    bm = {k: np.array([bmetric(p, b) for b in boots]) for k, p in allp.items()}
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    rows_out = []
    for k, p in allp.items():
        m = metrics(y, p); r2c, msec = ci(bm[k][:, 0]), ci(bm[k][:, 1])
        rows_out.append(dict(kind=k.split('::')[0], name=k.split('::')[1], n=len(y), **m, r2_ci_lo=r2c[0], r2_ci_hi=r2c[1],
                             mse_ci_lo=msec[0], mse_ci_hi=msec[1],
                             active_dims=active.get(k.split('::')[1]) if k.startswith('rep') else None,
                             alphas=str(alphas.get(k.split('::')[1], ''))))
    pd.DataFrame(rows_out).to_csv(OUT / 'summary.csv', index=False)
    pc = []
    for k, p in allp.items():
        for c in chroms:
            i = idx[c]; yv, pv = y[i], p[i]
            pc.append(dict(name=k, chromosome=c, n=len(i), mse=((yv - pv) ** 2).mean(),
                           r2_within_chr=1 - ((yv - pv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum(),
                           pearson=np.corrcoef(yv, pv)[0, 1]))
    pd.DataFrame(pc).to_csv(OUT / 'per_chromosome.csv', index=False)
    base = 'rep::Functional-256'; diffs = []
    for k in allp:
        if k == base or not k.startswith('rep::'): continue
        dr, dm = bm[base][:, 0] - bm[k][:, 0], bm[base][:, 1] - bm[k][:, 1]
        pt = metrics(y, allp[base]); pk = metrics(y, allp[k])
        # chromosomes where Functional has lower MSE
        wins = sum(((y[idx[c]] - allp[base][idx[c]]) ** 2).mean() < ((y[idx[c]] - allp[k][idx[c]]) ** 2).mean() for c in chroms)
        diffs.append(dict(comparator=k[5:], d_r2=pt['r2'] - pk['r2'], d_r2_ci=str(ci(dr)), d_mse=pt['mse'] - pk['mse'],
                          d_mse_ci=str(ci(dm)), chrom_wins_functional=f'{wins}/{len(chroms)}',
                          p_boot_two_sided=float(min(1, 2 * min((dr <= 0).mean(), (dr >= 0).mean())))))
    pd.DataFrame(diffs).to_csv(OUT / 'paired_vs_functional256.csv', index=False)
    print(pd.DataFrame(rows_out).drop(columns=['alphas']).round(4).to_string()); print(pd.DataFrame(diffs).round(4).to_string())


if __name__ == '__main__':
    main()
