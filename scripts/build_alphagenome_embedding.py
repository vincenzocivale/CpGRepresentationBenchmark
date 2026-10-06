#!/usr/bin/env python3
"""Build the canonical AlphaGenome locus store (native_frozen, locus_only) from the hg38 reference.

AlphaGenome is run on the reference sequence only (no patient data, no methylation objective), so the
vector for a CpG depends solely on its genomic position. To avoid 408k forward passes, each chromosome is
tiled with WINDOW-bp windows; a CpG is read from the tile where it is farthest from an edge (>= MARGIN bp
of context on both sides whenever the chromosome allows). Per-CpG vector = the model's sequence embedding
at the cytosine: 1 bp decoder representation (primary, default) or the 128 bp bin (--resolution 128).
The 128 bp bin is NOT primary: 34% of registry CpGs share a bin with another CpG (audit 2026-10-06).

Output: /cpg_idx int64 [N], /embedding float16 [N, D] + .json sidecar (same contract as cpgpt_locus.h5).

  python scripts/build_alphagenome_embedding.py --registry data/cpg/registries/array_cpg_map.parquet \
      --fasta /data2/utility/references/genomics/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta \
      --output data/cache/representations/alphagenome_locus.h5
  add --dummy --limit 2000 for a model-free smoke test of the tiling/gather logic.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pyfaidx import Fasta


def load_embedder(device: str, resolution: int, version: str = "all_folds", source: str = "huggingface"):
    """fn(seq, offsets [n]) -> float [n, D]: AlphaGenome trunk embedding (human) at the given bases.

    resolution 1 -> embeddings_1bp (D=1536, single-base decoder output); 128 -> embeddings_128bp (D=3072,
    Transformer path, bin offsets//128). Trunk only: no head outputs, reference sequence in, no patient data.
    """
    import jax
    import jax.numpy as jnp
    from alphagenome_research.model import dna_model, one_hot_encoder

    create = dna_model.create_from_huggingface if source == "huggingface" else dna_model.create_from_kaggle
    dev = jax.devices("gpu")[0] if device == "cuda" else jax.devices(device)[0]
    # Human only, no FASTA/GTF/calibration: those are variant-scoring extras fetched from gs:// (not needed here).
    from alphagenome.models import dna_model as api
    m = create(version, organism_settings={api.Organism.HOMO_SAPIENS: dna_model.OrganismSettings()}, device=dev)
    enc = one_hot_encoder.DNAOneHotEncoder()
    org = jnp.zeros((1,), jnp.int32)  # 0 = HOMO_SAPIENS

    @jax.jit  # shape depends only on the tile length -> compiled once; the gather below varies per tile
    def trunk(x):
        return m._trunk_apply_fn(m._params, m._state, x, org).get_sequence_embeddings(resolution)[0]

    def fn(seq: str, offsets: np.ndarray) -> np.ndarray:
        e = trunk(jnp.asarray(enc.encode(seq))[None])
        idx = jnp.asarray(offsets // resolution if resolution > 1 else offsets)
        return np.asarray(e[idx].astype(jnp.float16))  # gather on device, only [n, D] reaches host

    return fn


def dummy_embedder(resolution: int):
    lut = np.zeros((256, 4), np.float32)
    for i, b in enumerate(b"ACGT"):
        lut[b, i] = lut[b + 32, i] = 1
    def fn(seq: str, offsets: np.ndarray) -> np.ndarray:
        x = lut[np.frombuffer(seq.encode(), np.uint8)]
        return x.reshape(-1, resolution, 4).mean(1)[offsets // resolution]
    return fn


def tile_starts(length: int, window: int, margin: int) -> np.ndarray:
    stride = window - 2 * margin
    return np.arange(0, max(length - window, 0) + stride, stride)  # last tile may overhang; clamped below


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--window", type=int, default=2**20)
    p.add_argument("--margin", type=int, default=2**17)
    p.add_argument("--resolution", type=int, default=1, choices=[1, 128])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    reg = pd.read_parquet(a.registry, columns=["cpg_idx", "chr", "pos"])
    if a.limit:
        reg = reg.sample(a.limit, random_state=0)
    reg = reg.sort_values(["chr", "pos"]).reset_index(drop=True)
    fa = Fasta(str(a.fasta), as_raw=True, sequence_always_upper=True)
    embed = dummy_embedder(a.resolution) if a.dummy else load_embedder(a.device, a.resolution)

    out_rows, out_idx, dim = [], [], None
    for chrom, g in reg.groupby("chr", sort=False):
        L = len(fa[chrom])
        pos0 = g["pos"].to_numpy() - 1  # registry is 1-based
        starts = tile_starts(L, a.window, a.margin)
        # tile whose centre is nearest -> max context on both sides
        t = np.clip(np.rint((pos0 - a.window // 2) / (a.window - 2 * a.margin)).astype(int), 0, len(starts) - 1)
        for k in np.unique(t):
            s = min(int(starts[k]), max(L - a.window, 0))
            seq = str(fa[chrom][s:min(s + a.window, L)])
            seq += "N" * (-len(seq) % a.window if len(seq) < a.window else 0)  # pad short chroms/overhang
            m = t == k
            off = pos0[m] - s
            assert off.min() >= 0 and off.max() < a.window, (chrom, s)
            emb = np.asarray(embed(seq, off))
            out_rows.append(emb.astype(np.float16))
            out_idx.append(g["cpg_idx"].to_numpy()[m])
            dim = emb.shape[1]
        print(chrom, len(g), flush=True)

    idx, E = np.concatenate(out_idx).astype(np.int64), np.concatenate(out_rows)
    assert len(np.unique(idx)) == len(reg), "coverage lost: every registry CpG must be embedded"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.output, "w") as f:
        f.create_dataset("cpg_idx", data=idx)
        f.create_dataset("embedding", data=E, compression="gzip")
        f.attrs.update(representation="alphagenome_sequence_embedding", created_utc=datetime.now(timezone.utc).isoformat(),
                       source=str(a.registry), fasta=str(a.fasta), window=a.window, margin=a.margin,
                       resolution=a.resolution, dummy=a.dummy, patient_specific=False, supervision="supervised_multimodal_functional_genomics",
                       locus_fit_scope="external_pretrained", coordinate_convention="1-based position of CpG cytosine",
                       reference_build="GRCh38")
    side = {"n_loci": int(len(idx)), "embedding_dim": int(dim), "window": a.window, "resolution": a.resolution,
            "source": str(a.registry), "dummy": a.dummy}
    a.output.with_suffix(a.output.suffix + ".json").write_text(json.dumps(side, indent=2))
    print(json.dumps(side, indent=2))


if __name__ == "__main__":
    main()
