#!/usr/bin/env python3
"""Build canonical CpG embeddings from the functional-locus branch used in MehylPredictor.

This is a compatibility adapter, not a claim that a supervised MehylPredictor checkpoint is the
final representation to benchmark. It exists so the functional representation already used in
that project can be deployed immediately in the new controlled masking harness.

Output contract:
  /cpg_idx   int64 [N]
  /embedding float16 [N, D]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn


def _load_methylpredictor(root: Path):
    src = root / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"{src} not found; pass --methylpredictor-root")
    sys.path.insert(0, str(src))
    from methylation_predictor.modeling.retrieval import FeedForwardResidual
    from methylation_predictor.storage import open_functional_locus_cache

    return FeedForwardResidual, open_functional_locus_cache


class FunctionalLocusEncoder(nn.Module):
    N_TRACKS = 4165
    DENSE_DIM = 23
    WIDTH = 256

    def __init__(self, ff_cls, n_blocks: int = 8, dropout: float = 0.1):
        super().__init__()
        self.track_embedding = nn.EmbeddingBag(self.N_TRACKS, self.WIDTH, mode="mean", include_last_offset=True)
        self.dense_encoder = nn.Sequential(nn.LayerNorm(self.DENSE_DIM), nn.Linear(self.DENSE_DIM, self.WIDTH), nn.GELU())
        self.locus_norm = nn.LayerNorm(self.WIDTH)
        self.functional_ffn = nn.ModuleList([ff_cls(self.WIDTH, dropout) for _ in range(n_blocks)])

    def forward(self, track_indices: torch.Tensor, offsets: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        x = self.locus_norm(self.track_embedding(track_indices, offsets) + self.dense_encoder(dense))
        for block in self.functional_ffn:
            x = block(x)
        return x


def _load_state(checkpoint: Path, encoder: nn.Module) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_state = state.get("model_state", state.get("model", state))
    prefixes = ("track_embedding.", "dense_encoder.", "locus_norm.", "functional_ffn.")
    selected = {k: v for k, v in model_state.items() if k.startswith(prefixes)}
    if not selected:
        raise ValueError("checkpoint contains no recognized functional-locus encoder parameters")
    missing, unexpected = encoder.load_state_dict(selected, strict=False)
    required_missing = [k for k in missing if k.startswith(prefixes)]
    if required_missing:
        raise ValueError(f"checkpoint is missing functional encoder parameters: {required_missing}")
    if unexpected:
        raise ValueError(f"unexpected functional encoder parameters: {unexpected}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--array-cpg-map", type=Path, required=True)
    p.add_argument("--locus-store", type=Path, required=True)
    p.add_argument("--methylpredictor-root", type=Path, default=Path(__file__).resolve().parents[2] / "MehylPredictor")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chromosomes", nargs="*")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--n-functional-ffn-blocks", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ff_cls, open_cache = _load_methylpredictor(args.methylpredictor_root)
    columns = ["cpg_idx", "chr", "pos"]
    registry = pd.read_parquet(args.array_cpg_map, columns=columns)
    if registry["cpg_idx"].duplicated().any():
        raise ValueError("array CpG registry contains duplicate cpg_idx")
    if args.chromosomes:
        registry = registry[registry["chr"].isin(set(args.chromosomes))]
    registry = registry.sort_values("cpg_idx", kind="stable")
    cpg_ids = registry["cpg_idx"].to_numpy(np.int64)

    cache = open_cache(locus_store=args.locus_store, cpg_registry=args.array_cpg_map)
    encoder = FunctionalLocusEncoder(ff_cls, args.n_functional_ffn_blocks, args.dropout)
    _load_state(args.checkpoint, encoder)
    encoder = encoder.to(args.device).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as out:
        out.attrs["representation"] = "methylpredictor_functional_locus_encoder"
        out.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        out.attrs["checkpoint"] = str(args.checkpoint)
        out.attrs["locus_store"] = str(args.locus_store)
        out.create_dataset("cpg_idx", data=cpg_ids, dtype="int64")
        chunk_rows = max(1, min(1024, len(cpg_ids)))
        emb = out.create_dataset(
            "embedding",
            shape=(len(cpg_ids), encoder.WIDTH),
            dtype="float16",
            chunks=(chunk_rows, encoder.WIDTH),
        )
        with torch.no_grad():
            for start in range(0, len(cpg_ids), args.batch_size):
                batch_ids = cpg_ids[start : start + args.batch_size]
                features = cache.get(batch_ids)
                h = encoder(
                    torch.from_numpy(features["track_indices"]).to(args.device),
                    torch.from_numpy(features["offsets"]).to(args.device),
                    torch.from_numpy(features["dense"].astype(np.float32)).to(args.device),
                )
                emb[start : start + len(batch_ids)] = h.float().cpu().numpy().astype(np.float16)
                if start % (args.batch_size * 20) == 0:
                    print(f"{start}/{len(cpg_ids)} loci", flush=True)

    sidecar = {
        "representation": "methylpredictor_functional_locus_encoder",
        "n_loci": int(len(cpg_ids)),
        "embedding_dim": int(encoder.WIDTH),
        "checkpoint": str(args.checkpoint),
        "locus_store": str(args.locus_store),
        "array_cpg_map": str(args.array_cpg_map),
        "chromosomes": args.chromosomes,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {len(cpg_ids)} x {encoder.WIDTH} embeddings to {args.output}")


if __name__ == "__main__":
    main()
