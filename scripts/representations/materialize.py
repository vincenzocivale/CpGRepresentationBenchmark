#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cpg_repr_benchmark.representations.materialization import (
    load_protocol_coordinates,
    materialize_coordinate_representation,
)


def _common(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--locus-protocol", type=Path, required=True)
    subparser.add_argument("--chromosomes", nargs="*")
    subparser.add_argument("--output", type=Path, required=True)
    subparser.add_argument("--batch-size", type=int, required=True)
    subparser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    subparser.add_argument("--force", action="store_true")
    subparser.add_argument("--no-resume", action="store_true")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Materialize a frozen coordinate-native CpG representation for an explicit locus protocol. "
            "Expensive representation inference happens here once, never inside downstream training."
        )
    )
    sub = p.add_subparsers(dest="provider", required=True)

    ntv3 = sub.add_parser("ntv3-pre", help="direct inference with InstaDeepAI NTv3 pretrained checkpoint")
    _common(ntv3)
    ntv3.add_argument("--fasta", type=Path, required=True)
    ntv3.add_argument("--checkpoint", default="InstaDeepAI/NTv3_650M_pre")
    ntv3.add_argument("--window-length", type=int, default=32768)
    ntv3.add_argument("--device", default="cuda")
    ntv3.add_argument("--embedding-dim", type=int, default=1536)
    ntv3.add_argument("--no-bf16", action="store_true")
    ntv3.add_argument("--compile", action="store_true")
    ntv3.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")

    legacy = sub.add_parser(
        "functional-legacy",
        help="smoke-test-only direct inference with the historical functional checkpoint",
    )
    _common(legacy)
    legacy.add_argument("--legacy-registry", type=Path, required=True)
    legacy.add_argument("--methylpredictor-root", type=Path, required=True)
    legacy.add_argument("--checkpoint", type=Path, required=True)
    legacy.add_argument("--locus-store", type=Path, required=True)
    legacy.add_argument("--device", default="cuda")
    legacy.add_argument("--encoder-batch-size", type=int, default=4096)
    legacy.add_argument("--n-functional-ffn-blocks", type=int, default=8)
    legacy.add_argument("--dropout", type=float, default=0.1)

    return p


def main() -> None:
    args = _parser().parse_args()
    ids, chrom, pos = load_protocol_coordinates(
        args.locus_protocol,
        chromosomes=args.chromosomes,
    )

    if args.provider == "ntv3-pre":
        from cpg_repr_benchmark.representations.providers.ntv3_pre import NTv3PreProvider

        provider = NTv3PreProvider(
            fasta_path=args.fasta,
            checkpoint=args.checkpoint,
            window_length=args.window_length,
            device=args.device,
            bf16=not args.no_bf16,
            compile_model=args.compile,
            compile_mode=args.compile_mode,
            embedding_dim=args.embedding_dim,
        )
    elif args.provider == "functional-legacy":
        from cpg_repr_benchmark.representations.providers.functional_legacy import LegacyFunctionalProvider

        provider = LegacyFunctionalProvider(
            legacy_registry=args.legacy_registry,
            methylpredictor_root=args.methylpredictor_root,
            checkpoint=args.checkpoint,
            locus_store=args.locus_store,
            device=args.device,
            encoder_batch_size=args.encoder_batch_size,
            n_functional_ffn_blocks=args.n_functional_ffn_blocks,
            dropout=args.dropout,
        )
    else:  # pragma: no cover - argparse guarantees this
        raise AssertionError(args.provider)

    try:
        summary = materialize_coordinate_representation(
            provider,
            ids,
            chrom,
            pos,
            args.output,
            batch_size=args.batch_size,
            storage_dtype=args.storage_dtype,
            force=args.force,
            resume=not args.no_resume,
            extra_metadata={
                "locus_protocol": str(args.locus_protocol.resolve()),
                "chromosome_filter": ",".join(args.chromosomes or []),
            },
        )
    finally:
        provider.close()

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"WROTE={args.output.resolve()}")


if __name__ == "__main__":
    main()
