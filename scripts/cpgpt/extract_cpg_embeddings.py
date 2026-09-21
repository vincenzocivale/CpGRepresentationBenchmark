"""Minimal CpGPT CpG embedding extraction.

Cached upstream DNA vectors are read from CpGPT's mmap.  With the explicit
``generate_missing`` option, absent vectors are computed in memory with the
same DNA language model path used by CpGPT.

This module has no dependency on ``cpg_repr_benchmark`` and is meant to run
inside a separate, CpGPT-pinned Python environment (see scripts/cpgpt/README.md).
Its only integration point with the rest of this repo is the canonical HDF5
contract produced by scripts/build_cpgpt_embedding.py, which imports this file.
"""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sqlitedict import SqliteDict

try:
    from cpgpt.model.components.legacy_numerics import enable_legacy_numerics
except ImportError:
    # Only present in CpGPT revisions that shipped a numerics compatibility shim
    # for newer torch/lightning; harmless to skip on revisions that predate it.
    def enable_legacy_numerics() -> None:
        return None


DEFAULT_DNA_LLM = "nucleotide-transformer-v2-500m-multi-species"
DEFAULT_CONTEXT_LENGTH = 2001
Representation = Literal["sequence", "locus"]


def _resolve_model_paths(
    *,
    model_name: str,
    checkpoint_path: str | Path | None,
    config_path: str | Path | None,
    model_resources_dir: str | Path,
    download_if_missing: bool,
) -> tuple[Path, Path]:
    """Resolve model files and optionally download them through CpGPT's public API."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    resources_dir = Path(model_resources_dir).expanduser()
    checkpoint = (
        Path(checkpoint_path).expanduser()
        if checkpoint_path is not None
        else resources_dir / "model" / "weights" / f"{model_name}.ckpt"
    )
    config = (
        Path(config_path).expanduser()
        if config_path is not None
        else resources_dir / "model" / "config" / f"{model_name}.yaml"
    )
    checkpoint_missing = not checkpoint.is_file()
    config_missing = not config.is_file()
    if not checkpoint_missing and not config_missing:
        return checkpoint, config

    missing = [
        str(path)
        for path, absent in ((checkpoint, checkpoint_missing), (config, config_missing))
        if absent
    ]
    if not download_if_missing:
        raise FileNotFoundError(
            "Missing CpGPT model resource(s): "
            + ", ".join(missing)
            + ". Pass download_if_missing=True to download them explicitly."
        )

    try:
        from cpgpt import download_cpgpt
    except ImportError as exc:
        raise ImportError(
            "Downloading model resources requires the pinned CpGPT integration; "
            "install it in the dedicated cpgpt environment (see scripts/cpgpt/README.md)"
        ) from exc

    resources = download_cpgpt(model=model_name, local_dir=resources_dir)
    if checkpoint_missing:
        if resources.checkpoint_path is None:
            raise RuntimeError("CpGPT's downloader returned no checkpoint path")
        checkpoint = Path(resources.checkpoint_path)
    if config_missing:
        if resources.config_path is None:
            raise RuntimeError("CpGPT's downloader returned no configuration path")
        config = Path(resources.config_path)
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError(
            "CpGPT reported a completed model download, but its checkpoint or configuration is missing"
        )
    return checkpoint, config


def _load_network(checkpoint_path: Path, config_path: Path, device: torch.device):
    """Instantiate CpGPT's network and load only its trained network weights."""
    enable_legacy_numerics()
    config = OmegaConf.load(config_path)
    network = instantiate(config.model.net)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=zipfile.is_zipfile(checkpoint_path),
    )
    network_state = {}
    for name, value in checkpoint["state_dict"].items():
        if name.startswith("net._orig_mod."):
            network_state[name.removeprefix("net._orig_mod.")] = value
        elif name.startswith("net."):
            network_state[name.removeprefix("net.")] = value

    network.load_state_dict(network_state, strict=True)
    return network.to(device).eval()


def _encode_locus_positions(
    network: torch.nn.Module,
    sequence_embeddings: torch.Tensor,
    positions: torch.Tensor,
    chromosomes: torch.Tensor,
) -> torch.Tensor:
    """Mirror CpGPT's configured pre-methylation ``locus_embeddings`` path."""
    positional_encoding = getattr(network, "positional_encoding", None)
    if positional_encoding == "positional":
        locus_embeddings = network.absolute_position_encoder(sequence_embeddings, positions)
        return network.position_encoder(locus_embeddings, chromosomes)
    if positional_encoding == "rotary":
        locus_embeddings = network.absolute_position_encoder(sequence_embeddings, positions)
        batch_size, sequence_length, embedding_dim = locus_embeddings.shape
        attention_heads = network.n_attention_heads
        if embedding_dim % attention_heads:
            raise ValueError(
                "network embedding dimension must be divisible by its number of attention heads"
            )
        locus_embeddings = locus_embeddings.view(
            batch_size,
            sequence_length,
            attention_heads,
            -1,
        ).transpose(1, 2)
        locus_embeddings = network.position_encoder(locus_embeddings)
        return locus_embeddings.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, embedding_dim
        )
    raise ValueError(
        "representation='locus' is unsupported for the checkpoint's "
        f"positional_encoding={positional_encoding!r}"
    )


def _parse_location(location: str) -> tuple[str, int]:
    if not isinstance(location, str):
        raise TypeError("locations must contain 'chromosome:position' strings")
    chromosome, separator, raw_position = location.rpartition(":")
    if (
        not separator
        or not chromosome
        or not raw_position.isascii()
        or not raw_position.isdecimal()
    ):
        raise ValueError(
            f"invalid location {location!r}; expected chromosome:non-negative-integer"
        )
    return chromosome, int(raw_position)


def _validate_generation_locations(
    locations: Sequence[str], species_metadata: dict[str, Any], species: str
) -> None:
    """Reject malformed or unknown chromosomes before CpGPT can turn them into N-only DNA."""
    vocab = species_metadata.get("vocab")
    for location in locations:
        chromosome, _ = _parse_location(location)
        if ":" in chromosome:
            raise ValueError(
                f"invalid location {location!r}; generation requires a chromosome name without ':'"
            )
        if isinstance(vocab, dict) and chromosome not in vocab:
            raise ValueError(f"unknown chromosome {chromosome!r} for species {species!r}")


def _compute_missing_dna_embeddings(
    locations: Sequence[str],
    *,
    dependencies_dir: Path,
    species: str,
    dna_llm: str,
    context_length: int,
    genome_file: Path | None,
    generation_batch_size: int,
) -> dict[str, np.ndarray]:
    """Compute upstream vectors without copying or extending CpGPT's large mmap."""
    try:
        from cpgpt.data.components.dna_llm_embedder import (
            DNALLMEmbedder,
            GenomicSequenceDataset,
            collate_sequences,
        )
        from pyfaidx import Fasta
        from torch.utils.data import DataLoader
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError(
            "Generating uncached coordinates requires the pinned CpGPT integration; "
            "install it in the dedicated cpgpt environment (see scripts/cpgpt/README.md)"
        ) from exc

    # DNALLMEmbedder owns CpGPT's reference-genome and DNA-model resolution.
    # We deliberately do not call parse_dna_embeddings(): that method rewrites
    # the multi-gigabyte shared mmap merely to add a handful of rows.
    embedder = DNALLMEmbedder(dependencies_dir=str(dependencies_dir))
    resolved_genome = str(genome_file) if genome_file is not None else embedder._load_genome(species)
    if not resolved_genome:
        raise RuntimeError(f"CpGPT could not load a reference genome for {species!r}")

    genome = None
    model = None
    try:
        genome = Fasta(resolved_genome)
        for location in locations:
            chromosome, raw_position = location.rsplit(":", 1)
            if chromosome not in genome:
                raise ValueError(
                    f"chromosome {chromosome!r} is absent from reference genome {resolved_genome}"
                )
            # CpGPT treats positions as assembly coordinates without applying an
            # offset.  Accept the terminal one-based coordinate, but nothing beyond it.
            if int(raw_position) > len(genome[chromosome]):
                raise ValueError(
                    f"position {raw_position} is outside chromosome {chromosome!r} "
                    f"in reference genome {resolved_genome}"
                )

        model, tokenizer = embedder._load_dna_model(dna_llm)
        try:
            model_device = model.device
        except AttributeError:
            model_device = next(model.parameters()).device

        dataset = GenomicSequenceDataset(list(locations), genome, context_length)
        loader = DataLoader(
            dataset,
            batch_size=generation_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: collate_sequences(batch, tokenizer=tokenizer),
        )

        generated: dict[str, np.ndarray] = {}
        progress = tqdm(
            loader,
            total=len(loader),
            desc=f"CpGPT DNA-LLM generation ({dna_llm}, batch={generation_batch_size})",
            unit="batch",
        )
        with torch.inference_mode():
            for batch in progress:
                input_ids = batch["input_ids"].to(model_device)
                attention_mask = batch["attention_mask"].float().to(model_device)
                if dna_llm == "DNABERT-2-117M":
                    hidden = model(input_ids)[0]
                else:
                    hidden = model(input_ids, output_hidden_states=True).hidden_states[-1]
                pooled = (hidden * attention_mask.unsqueeze(-1)).sum(dim=1)
                pooled = pooled / attention_mask.sum(dim=1, keepdim=True)
                values = pooled.to(dtype=torch.float32).cpu().numpy()
                for location, vector in zip(batch["locations"], values, strict=True):
                    generated[location] = np.array(vector, dtype=np.float32, copy=True)
        return generated
    finally:
        if genome is not None:
            genome.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def extract_cpg_embeddings(
    locations: Sequence[str],
    *,
    model_name: str = "small",  # small or large
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    model_resources_dir: str | Path = "dependencies",
    download_if_missing: bool = False,
    dependencies_dir: str | Path = "dependencies/human",
    species: str = "homo_sapiens",
    dna_llm: str = DEFAULT_DNA_LLM,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    representation: Representation = "sequence",
    device: str = "cpu",
    batch_size: int = 4096,
    generate_missing: bool = False,
    genome_file: str | Path | None = None,
    generation_batch_size: int = 1,
) -> np.ndarray:
    """Return one CpGPT sequence or configured locus embedding per location.

    ``locations`` must use CpGPT's ``"chromosome:position"`` keys, for example
    ``["16:53434199", "3:172198246"]``. Input order and duplicates are kept.
    The returned array has shape ``(len(locations), 128)`` for CpGPT-small.
    Coordinates must match the assembly and naming convention of the selected
    species dependencies. CpGPT keys use the zero-based start of the two-base
    CpG interval. The function performs no coordinate conversion, liftover, or
    check that the referenced bases are biologically a CpG.

    When checkpoint/config paths are omitted they are derived as
    ``<model_resources_dir>/model/{weights,config}/<model_name>.*``. Missing
    model files are an error unless ``download_if_missing=True``; that explicit
    option invokes CpGPT's official downloader and uses the paths it returns.

    ``representation="sequence"`` returns only CpGPT's projection of the local
    DNA vector. ``representation="locus"`` mirrors the checkpoint-configured
    pre-methylation positional path from ``CpGPT.encode_sample``. For the small
    rotary checkpoint this means absolute genomic-position encoding followed by
    RoPE. The complete input list is treated as one ordered locus sequence, so
    unlike "sequence" it is NOT a stable, order-independent per-CpG feature and
    is not suitable for this benchmark's patient-agnostic locus representation.

    By default this function uses only the precomputed DNA vectors in
    ``dependencies_dir`` and raises ``KeyError`` for uncached coordinates.  If
    ``generate_missing=True``, unique misses are computed in memory using
    CpGPT's reference-genome and DNA-language-model path.  That explicit mode
    may download a genome and Hugging Face model; pass ``genome_file`` to use an
    existing FASTA.  Generated upstream vectors are not written into CpGPT's
    shared mmap.
    """
    dependencies_dir = Path(dependencies_dir)
    device = torch.device(device)

    if representation not in ("sequence", "locus"):
        raise ValueError("representation must be 'sequence' or 'locus'")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if generation_batch_size <= 0:
        raise ValueError("generation_batch_size must be positive")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    checkpoint_path, config_path = _resolve_model_paths(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        model_resources_dir=model_resources_dir,
        download_if_missing=download_if_missing,
    )

    metadata_path = dependencies_dir / "ensembl_metadata.db"
    with SqliteDict(metadata_path, flag="r") as metadata:
        species_metadata = metadata[species]
    location_to_row = species_metadata[dna_llm][context_length]

    locations = list(locations)
    parsed_locations = (
        [_parse_location(location) for location in locations]
        if representation == "locus"
        else None
    )
    missing = [location for location in dict.fromkeys(locations) if location not in location_to_row]
    if missing and not generate_missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise KeyError(
            f"CpGPT's precomputed cache does not contain: {preview}{suffix}. "
            "Pass generate_missing=True to compute them explicitly."
        )

    generated: dict[str, np.ndarray] = {}
    if missing:
        _validate_generation_locations(missing, species_metadata, species)
        generated = _compute_missing_dna_embeddings(
            missing,
            dependencies_dir=dependencies_dir,
            species=species,
            dna_llm=dna_llm,
            context_length=context_length,
            genome_file=Path(genome_file) if genome_file is not None else None,
            generation_batch_size=generation_batch_size,
        )
        still_missing = [location for location in missing if location not in generated]
        if still_missing:
            raise RuntimeError(
                "DNA embedding generation returned no vector for: " + ", ".join(still_missing)
            )

    # Generate first so the large upstream model can be released before the
    # CpGPT checkpoint is placed on the requested compute device.
    network = _load_network(checkpoint_path, config_path, device)

    dna_dimension = network.d_dna_embedding
    mmap_path = (
        dependencies_dir
        / "dna_embeddings"
        / species
        / dna_llm
        / f"{context_length}bp_dna_embeddings.mmap"
    )
    flat_cache = np.memmap(mmap_path, mode="r", dtype=np.float32)
    if flat_cache.size % dna_dimension:
        raise ValueError(f"Invalid DNA embedding cache shape: {mmap_path}")
    dna_cache = flat_cache.reshape(-1, dna_dimension)
    if location_to_row and max(location_to_row.values()) >= dna_cache.shape[0]:
        raise ValueError(f"DNA embedding metadata points outside the cache: {mmap_path}")

    for location, vector in generated.items():
        if vector.ndim != 1 or vector.shape[0] != dna_dimension:
            raise ValueError(
                f"generated DNA embedding for {location} has shape {vector.shape}; "
                f"expected ({dna_dimension},)"
            )

    output = np.empty((len(locations), network.d_embedding), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(locations), batch_size):
            batch_locations = locations[start : start + batch_size]
            vectors = [
                generated[location]
                if location in generated
                else dna_cache[location_to_row[location]]
                for location in batch_locations
            ]
            dna_vectors = torch.from_numpy(np.asarray(vectors, dtype=np.float32)).to(device)
            embeddings = network.encode_sequence(dna_vectors.unsqueeze(0)).squeeze(0)
            output[start : start + len(batch_locations)] = embeddings.cpu().numpy()

    if representation == "locus" and locations:
        assert parsed_locations is not None
        chromosome_vocab = species_metadata.get("vocab")
        if not isinstance(chromosome_vocab, dict):
            raise ValueError("CpGPT species metadata has no chromosome vocabulary")
        try:
            chromosome_indices = [chromosome_vocab[chromosome] for chromosome, _ in parsed_locations]
        except KeyError as exc:
            raise ValueError(f"unknown chromosome {exc.args[0]!r} for species {species!r}") from exc
        sequence_embeddings = torch.from_numpy(output).to(device).unsqueeze(0)
        position_tensor = torch.tensor(
            [[position for _, position in parsed_locations]], dtype=torch.long, device=device
        )
        chromosome_tensor = torch.tensor(
            [chromosome_indices], dtype=torch.long, device=device
        )
        locus_embeddings = _encode_locus_positions(
            network,
            sequence_embeddings,
            position_tensor,
            chromosome_tensor,
        )
        output = locus_embeddings.squeeze(0).cpu().numpy()

    return output
