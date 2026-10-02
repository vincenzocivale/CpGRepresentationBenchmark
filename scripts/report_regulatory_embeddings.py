#!/usr/bin/env python3
"""Descriptive report of materialized regulatory embeddings (representation properties only; no methylation).

Reads each embedding h5 + saved compressor, and the track CSR of the feature store (never /dense) to attribute energy to
blocks. Writes JSON, spectra, PNG plots and a markdown table under outputs/regulatory_compression_reports/.

Definitions (fit loci F, embedding E = rows of F):
  explained energy   = sum_i mean_F(z_i^2) / mean_F ||x_w||^2          (uncentered; the SVD is uncentered)
  explained variance = sum_i var_F(z_i)   / sum_j var_F(x_wj)           (centered share captured by the same projection)
  singular value i   = sqrt(|F| * mean_F(z_i^2))
  effective rank     = participation ratio (sum l)^2 / sum l^2 and exp-entropy rank exp(-sum p log p), p = l / sum l,
                       with l the eigenvalues of the 256x256 second-moment matrix of E (uncentered) or covariance (centered)
  block-only energy  = ||X_b M_b^T||_F^2 / ||X M^T||_F^2 on F (X_b: tracks of block b, M_b: projection columns of b);
                       does not sum to 1 because of cross-block terms
  subspace overlap   = cosines of principal angles between the column spaces of two embeddings on F
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from cpg_repr_benchmark.encode_atlas.compression import (
    BLOCKS,
    RegulatoryCompressor,
    TrackStore,
    split_loci,
)

QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def effective_ranks(e: np.ndarray, centered: bool) -> dict:
    x = e.astype(np.float64)
    if centered:
        x = x - x.mean(axis=0)
    lam = np.clip(np.linalg.eigvalsh(x.T @ x / len(x)), 0, None)
    lam = lam[lam > 0]
    p = lam / lam.sum()
    return {'participation_ratio': float(lam.sum() ** 2 / (lam**2).sum()), 'exp_entropy_rank': float(np.exp(-(p * np.log(p)).sum()))}


def row_norm_quantiles(norms: np.ndarray) -> dict:
    return {**{f'q{int(q * 100):02d}': float(np.quantile(norms, q)) for q in QUANTILES}, 'mean': float(norms.mean())}


def block_energy(store: TrackStore, comp: RegulatoryCompressor, rows: np.ndarray, emb_fit: np.ndarray) -> dict:
    total = float((emb_fit.astype(np.float64) ** 2).sum())
    proj_t = {b: np.ascontiguousarray(comp.projection[:, comp.blocks == b].T) for b in BLOCKS if (comp.blocks == b).any()}
    energy = dict.fromkeys(proj_t, 0.0)
    for s in range(0, len(rows), 50_000):
        xc = store.matrix[rows[s:s + 50_000]][:, comp.columns]
        for b, m in proj_t.items():
            energy[b] += float((np.asarray(xc[:, comp.blocks == b] @ m, dtype=np.float64) ** 2).sum())
    return {b: v / total for b, v in energy.items()}


def input_energy_share(store: TrackStore, comp: RegulatoryCompressor, rows: np.ndarray) -> dict:
    p = np.asarray(store.matrix[rows][:, comp.columns].getnnz(axis=0), dtype=np.float64) / len(rows)
    e = comp.weights.astype(np.float64) ** 2 * p
    return {b: float(e[comp.blocks == b].sum() / e.sum()) for b in BLOCKS if (comp.blocks == b).any()}


def principal_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    qa, _ = np.linalg.qr(a.astype(np.float64))
    qb, _ = np.linalg.qr(b.astype(np.float64))
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


def plots(items: dict, out: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#7F7F7F']
    files = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for color, (name, r) in zip(palette * 3, items.items()):
        sv = np.asarray(r['singular_values'])
        ls = '--' if '_egm' in name else '-'
        label = name.replace('__discovery_chr1_19', '')
        axes[0].plot(np.arange(1, len(sv) + 1), sv, color=color, ls=ls, lw=1.5, label=label)
        axes[1].plot(np.arange(1, len(sv) + 1), np.cumsum(r['energy_per_component']) / r['total_energy'], color=color, ls=ls,
                     lw=1.5, label=label)
    axes[0].set(yscale='log', xlabel='component', ylabel='singular value (fit loci)', title='Singular-value spectrum')
    axes[1].set(xlabel='component', ylabel='cumulative explained energy', title='Explained energy (uncentered)')
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[1].legend(fontsize=7, frameon=False, loc='lower right')
    fig.savefig(out / 'spectra.png', dpi=140)
    plt.close(fig)
    files.append('spectra.png')
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    names = list(items)
    width = 0.25
    for k, b in enumerate(BLOCKS):
        ax.bar(np.arange(len(names)) + (k - 1) * width, [items[n]['block_only_energy'].get(b, 0) for n in names], width,
               color=palette[k], label=b)
    ax.set_xticks(np.arange(len(names)), [n.replace('__discovery_chr1_19', '') for n in names], rotation=30, ha='right', fontsize=7)
    ax.set(ylabel='block-only share of embedding energy', title='Block contribution (fit loci)')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(out / 'block_contribution.png', dpi=140)
    plt.close(fig)
    files.append('block_contribution.png')
    return files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=Path('configs/regulatory_embeddings.yaml'))
    p.add_argument('--out', type=Path, default=Path('outputs/regulatory_compression_reports'))
    args = p.parse_args()
    import h5py
    from build_regulatory_embeddings import embedding_name  # same folder
    cfg = yaml.safe_load(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    store = TrackStore(cfg['store'])
    fit, heldout = split_loci(cfg['protocol'], store.cpg_idx, cfg['registry'])
    fit_rows, held_rows = store.rows_for(fit.cpg_idx), store.rows_for(heldout)
    items: dict = {}
    embs: dict = {}
    for entry in cfg['embeddings']:
        name = embedding_name(entry, cfg['protocol'])
        h5 = Path(cfg['output_dir']) / f'{name}.h5'
        if not h5.exists():
            print('missing', h5)
            continue
        with h5py.File(h5, 'r') as h:
            ids, e = h['cpg_idx'][:], h['embedding'][:]
        assert np.array_equal(ids, store.cpg_idx)
        comp = RegulatoryCompressor.load(Path(cfg['output_dir']) / f'{name}.compressor.npz')
        ef, eh = e[fit_rows], e[held_rows]
        a = comp.arrays
        energy = a['stat_embedding_energy']
        var = a['stat_embedding_variance']
        is_block = comp.method == 'block_svd'
        # block_svd rescales blocks, so raw-input denominators are meaningless: use the block-normalized concatenation
        e_total = float(a['stat_concat_total_energy' if is_block else 'stat_total_energy'])
        r = {
            'feature_set': comp.feature_set.name, 'method': comp.method, 'replicate_weighting': comp.replicate_weighting,
            'dim': comp.dim, 'n_features': len(comp.columns), 'dtype': str(e.dtype), 'shape': list(e.shape),
            'size_mb': h5.stat().st_size / 1e6, 'compressor_mb': (h5.parent / f'{name}.compressor.npz').stat().st_size / 1e6,
            'fit_seconds': comp.info['fit_seconds'],
            'total_energy': e_total, 'total_variance': None if is_block else float(a['stat_total_variance']),
            'energy_per_component': energy.tolist(),
            'singular_values': np.sqrt(len(fit_rows) * energy).tolist(),
            'explained_energy_cum': {str(d): float(energy[:d].sum() / e_total) for d in (1, 16, 64, 128, 256) if d <= comp.dim},
            'explained_variance_cum': None if is_block else {
                str(d): float(var[:d].sum() / a['stat_total_variance']) for d in (1, 16, 64, 128, 256) if d <= comp.dim},
            'effective_rank_uncentered': effective_ranks(ef, False), 'effective_rank_centered': effective_ranks(ef, True),
            'row_norm_fit': row_norm_quantiles(np.linalg.norm(ef, axis=1)),
            'row_norm_heldout_chr20_22': row_norm_quantiles(np.linalg.norm(eh, axis=1)),
            'fraction_zero_rows_fit': float((np.linalg.norm(ef, axis=1) == 0).mean()),
            'fraction_zero_rows_heldout': float((np.linalg.norm(eh, axis=1) == 0).mean()),
            'block_only_energy': block_energy(store, comp, fit_rows, ef),
            'input_energy_share': input_energy_share(store, comp, fit_rows),
            'replicate_groups': comp.manifest()['replicate_groups'],
        }
        if comp.method == 'block_svd':
            r['block_pre_post'] = {b: {'pre_frobenius_sq_per_locus': float(a[f'stat_block_{b}_pre_frobenius_sq_per_locus']),
                                       'post_frobenius_sq_per_locus': float(a[f'stat_block_{b}_post_frobenius_sq_per_locus']),
                                       'scale': float(a[f'stat_block_{b}_scale']),
                                       'block_dim': int(a['block_dims'][list(a['block_order']).index(b)])}
                                   for b in a['block_order']}
            r['block_raw_energy_captured'] = {
                b: float(a[f'stat_block_{b}_pre_comp_energy'].sum() / a[f'stat_block_{b}_total_energy']) for b in a['block_order']}
        items[name] = r
        embs[name] = ef
        print('done', name, flush=True)
    # none vs equal_group_mass on regulatory_clean
    comparison = {}
    for method, dim in (('global_svd', 256), ('block_svd', 256)):
        n0 = f'regulatory_clean__{method}{dim}__{cfg["protocol"]}'
        n1 = f'regulatory_clean__{method}{dim}_egm__{cfg["protocol"]}'
        if n0 in embs and n1 in embs:
            cos = principal_cosines(embs[n0], embs[n1])
            lead = principal_cosines(embs[n0][:, :32], embs[n1][:, :32])
            comparison[method] = {
                'mean_cosine_principal_angles_all': float(cos.mean()), 'min_cosine': float(cos.min()),
                'n_cosine_gt_0.9': int((cos > 0.9).sum()), 'mean_cosine_leading32_subspaces': float(lead.mean()),
                'effective_rank_centered_none': items[n0]['effective_rank_centered'],
                'effective_rank_centered_egm': items[n1]['effective_rank_centered'],
                'block_only_energy_none': items[n0]['block_only_energy'], 'block_only_energy_egm': items[n1]['block_only_energy']}
    figs = plots(items, args.out)
    (args.out / 'report.json').write_text(json.dumps({'protocol': cfg['protocol'], 'n_fit': fit.n, 'fit_loci_sha256': fit.sha256,
                                                      'n_heldout': int(heldout.size), 'embeddings': items,
                                                      'none_vs_equal_group_mass': comparison, 'figures': figs}, indent=1,
                                                     allow_nan=False))
    hdr = ('| embedding | dim | E@16 | E@64 | E@256 (block: of normalized concat) | V@256 (global only) | PR (cent.) | expH (cent.) | PR (uncent.) | norm med fit | norm med chr20-22 '
           '| hist/tf/dnase block-only | fit s | MB |\n|' + '---|' * 14)
    rows = []
    for n, r in items.items():
        bo = r['block_only_energy']
        rows.append(f"| {n.replace('__discovery_chr1_19', '')} | {r['dim']} | {r['explained_energy_cum']['16']:.3f} | "
                    f"{r['explained_energy_cum']['64']:.3f} | {r['explained_energy_cum']['256']:.3f} | "
                    f"{(r['explained_variance_cum'] or {'256': float('nan')})['256']:.3f} | {r['effective_rank_centered']['participation_ratio']:.1f} | "
                    f"{r['effective_rank_centered']['exp_entropy_rank']:.1f} | {r['effective_rank_uncentered']['participation_ratio']:.1f} | "
                    f"{r['row_norm_fit']['q50']:.2f} | {r['row_norm_heldout_chr20_22']['q50']:.2f} | "
                    f"{bo.get('histone', 0):.2f}/{bo.get('tf', 0):.2f}/{bo.get('accessibility', 0):.2f} | "
                    f"{r['fit_seconds']:.0f} | {r['size_mb']:.0f} |")
    (args.out / 'report.md').write_text(hdr + '\n' + '\n'.join(rows) + '\n')
    print(hdr)
    print('\n'.join(rows))
    print(json.dumps(comparison, indent=1))


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
