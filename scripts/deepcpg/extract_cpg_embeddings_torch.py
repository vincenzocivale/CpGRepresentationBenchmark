"""GPU DeepCpG DNA-module CpG embedding extraction (PyTorch weight-port).

DeepCpG's own package pins legacy standalone Keras 2.2.4 + TensorFlow 1.15,
which has no GPU support on this machine's Ampere (A100) GPU (TF1.15's GPU
wheels predate Ampere's compute capability, and NVIDIA's Ampere-patched
`nvidia-tensorflow` fork is not reachable from this network's package index).

This module ports a zoo DNA-module checkpoint's *weights* (not the Keras
graph) into an equivalent PyTorch `nn.Sequential`, built dynamically from the
checkpoint's own `model.json` layer list, and runs the forward pass on GPU.
It is numerically equivalent to `extract_cpg_embeddings.py` (verified layer
by layer against a CPU/Keras reference; see scripts/deepcpg/README.md) but
has no dependency on the `deepcpg` package or standalone `keras`, only
`torch`/`h5py`/`pyfaidx` -- so it runs in any modern, GPU-enabled Python
environment (this repo uses the pre-existing `cpgpt` conda env for this
purpose; see scripts/deepcpg/README.md).

Supported layer classes (covers every DeepCpG zoo DNA-module checkpoint,
whose DNA models are plain feedforward CNNs -- CnnL1h*/CnnL2h*/CnnL3h*):
Convolution1D, Activation('relu'), MaxPooling1D, Flatten, Dense, Dropout.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import torch
from pyfaidx import Fasta
from torch import nn

CHAR_TO_INT = {"A": 0, "T": 1, "G": 2, "C": 3, "N": 4}


class _ChannelsLastFlatten(nn.Module):
    """Match Keras's channels-last Flatten order: torch conv output is
    (batch, channels, length); Keras's saved Dense weight matrix expects
    the (batch, length, channels) flatten order it was trained against."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).reshape(x.shape[0], -1)


def _read_weight(weights: h5py.File, layer_name: str, suffix: str) -> np.ndarray:
    base = layer_name.rsplit("/", 1)[-1]
    return np.asarray(weights[f"model_weights/{layer_name}/{base}_{suffix}:0"])


def build_stem_from_zoo_checkpoint(model_dir: str | Path) -> tuple[nn.Sequential, int]:
    """Parse a DeepCpG zoo DNA-module's model.json (in original layer order,
    stopping before its per-cell `cpg/*` output heads) and return an
    equivalent PyTorch nn.Sequential with weights loaded from
    model_weights.h5, plus the expected DNA input window length.
    """
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "model.json").read_text())["config"]
    output_names = {name for name, _, _ in config["output_layers"]}

    modules: list[nn.Module] = []
    in_channels = 4
    dna_wlen: int | None = None

    with h5py.File(model_dir / "model_weights.h5", "r") as weights:
        for layer in config["layers"]:
            name = layer["name"]
            if name in output_names:
                break  # stem ends where the per-cell output heads begin
            class_name = layer["class_name"]
            cfg = layer["config"]

            if class_name == "InputLayer":
                dna_wlen = int(cfg["batch_input_shape"][1])
                continue
            if class_name == "Convolution1D":
                out_channels = int(cfg["nb_filter"])
                kernel = int(cfg["filter_length"])
                conv = nn.Conv1d(in_channels, out_channels, kernel)
                w = _read_weight(weights, name, "W")  # (kernel, 1, in, out)
                b = _read_weight(weights, name, "b")
                conv.weight.data = torch.from_numpy(
                    np.ascontiguousarray(w[:, 0].transpose(2, 1, 0))
                ).float()  # -> (out, in, kernel)
                conv.bias.data = torch.from_numpy(b).float()
                modules.append(conv)
                in_channels = out_channels
            elif class_name == "Activation":
                if cfg["activation"] != "relu":
                    raise ValueError(f"unsupported activation {cfg['activation']!r} in {name}")
                modules.append(nn.ReLU())
            elif class_name == "MaxPooling1D":
                modules.append(nn.MaxPool1d(int(cfg["pool_length"])))
            elif class_name == "Flatten":
                modules.append(_ChannelsLastFlatten())
            elif class_name == "Dense":
                out_features = int(cfg["output_dim"])
                w = _read_weight(weights, name, "W")  # (in, out)
                b = _read_weight(weights, name, "b")
                in_features = w.shape[0]
                dense = nn.Linear(in_features, out_features)
                dense.weight.data = torch.from_numpy(np.ascontiguousarray(w.T)).float()
                dense.bias.data = torch.from_numpy(b).float()
                modules.append(dense)
            elif class_name == "Dropout":
                pass  # eval-mode identity regardless of stored rate
            else:
                raise ValueError(f"unsupported DeepCpG DNA-module layer class {class_name!r} ({name})")

    if dna_wlen is None:
        raise ValueError(f"{model_dir}/model.json has no 'dna' InputLayer")
    return nn.Sequential(*modules).eval(), dna_wlen


def _one_hot_window(sequence: str) -> np.ndarray:
    wlen = len(sequence)
    encoded = np.zeros((4, wlen), dtype=np.float32)  # channels-first for torch
    for position, char in enumerate(sequence.upper()):
        index = CHAR_TO_INT.get(char, 4)
        if index < 4:
            encoded[index, position] = 1.0
    return encoded


def _sequence_window(genome: Fasta, chrom: str, one_based_pos: int, wlen: int) -> str:
    half = wlen // 2
    start = one_based_pos - 1 - half
    end = start + wlen
    chrom_len = len(genome[chrom])
    if start < 0 or end > chrom_len:
        raise ValueError(
            f"{chrom}:{one_based_pos} +/-{half}bp window falls outside chromosome bounds "
            f"(length {chrom_len})"
        )
    return str(genome[chrom][start:end])


def extract_cpg_embeddings(
    loci: Sequence[tuple[str, int]],
    *,
    model_dir: str | Path,
    genome_file: str | Path,
    dna_wlen: int = 1001,
    batch_size: int = 4096,
    device: str = "cuda",
) -> np.ndarray:
    """Return one DeepCpG DNA-module stem embedding per locus, computed on
    `device`. Same contract as extract_cpg_embeddings.extract_cpg_embeddings
    (the Keras/CPU reference implementation)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    stem, model_wlen = build_stem_from_zoo_checkpoint(model_dir)
    if model_wlen != dna_wlen:
        raise ValueError(
            f"model_dir's DNA input window is {model_wlen}bp, but dna_wlen={dna_wlen} was requested"
        )
    stem = stem.to(torch_device)

    genome = Fasta(str(genome_file))
    try:
        windows = [_sequence_window(genome, chrom, pos, dna_wlen) for chrom, pos in loci]
    finally:
        genome.close()

    with torch.inference_mode():
        # Probe output dim from a throwaway forward pass on a single all-N window.
        probe = torch.zeros((1, 4, dna_wlen), device=torch_device)
        stem_dim = stem(probe).shape[-1]

        output = np.empty((len(loci), stem_dim), dtype=np.float32)
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            one_hot = np.stack([_one_hot_window(window) for window in batch], axis=0)
            batch_tensor = torch.from_numpy(one_hot).to(torch_device)
            output[start : start + len(batch)] = stem(batch_tensor).cpu().numpy()

    return output
