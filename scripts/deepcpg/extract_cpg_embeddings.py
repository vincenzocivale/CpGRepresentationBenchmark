"""Minimal DeepCpG DNA-module CpG embedding extraction.

Loads a DeepCpG "DNA model" from the official model zoo (a standalone,
sequence-only submodule of the DNA/CpG/Joint architecture -- see
scripts/deepcpg/README.md for why only this submodule is used), strips its
pretraining output head, and returns the stem's pooled activation for a
window of reference-genome sequence around each requested locus.

This module has no dependency on ``cpg_repr_benchmark`` and is meant to run
inside a separate, DeepCpG-pinned Python environment (legacy standalone
``keras`` + TensorFlow 1.x; see scripts/deepcpg/README.md). Its only
integration point with the rest of this repo is the canonical HDF5 contract
produced by scripts/build_deepcpg_embedding.py, which imports this file.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pyfaidx import Fasta

# Standalone `keras` (TF1-backend), not `tensorflow.keras`: DeepCpG's saved
# model.json configs and custom layers (e.g. ScaledSigmoid) were authored
# against this package.
from deepcpg.data.dna import CHAR_TO_INT
from deepcpg.models.utils import CUSTOM_OBJECTS


def _upgrade_legacy_regularizers(node):
    """Zoo checkpoints were serialized with Keras 1's flat regularizer format
    (``{"name": "WeightRegularizer", "l1": ..., "l2": ...}``), which Keras
    2.2.4's deserializer rejects (it expects ``{"class_name": ..., "config":
    ...}``). Recursively rewrite any such dict in place before deserializing.
    """
    if isinstance(node, dict):
        if node.get("name") == "WeightRegularizer" and "l1" in node and "l2" in node:
            return {"class_name": "L1L2", "config": {"l1": node["l1"], "l2": node["l2"]}}
        return {key: _upgrade_legacy_regularizers(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_upgrade_legacy_regularizers(item) for item in node]
    return node


def _load_legacy_model(model_dir: Path):
    """Load a DeepCpG zoo model directory (model.json + model_weights.h5)
    with Keras 2.2.4, upgrading the JSON config's legacy regularizer format
    first (see `_upgrade_legacy_regularizers`)."""
    from keras.models import model_from_json

    model_dir = Path(model_dir)
    config = json.loads((model_dir / "model.json").read_text())
    config = _upgrade_legacy_regularizers(config)
    model = model_from_json(json.dumps(config), custom_objects=CUSTOM_OBJECTS)
    model.load_weights(str(model_dir / "model_weights.h5"))
    return model


def _remove_output_layers(model):
    """Mirror dcpg_train.py's `remove_outputs`: drop the pretraining output
    head(s) so the returned model exposes the shared DNA-module stem
    representation, not a methylation-state prediction.

    The zoo's saved DNA modules were trained standalone to predict per-cell
    methylation state from sequence alone, so they carry one Dense output
    head per cell (named e.g. ``cpg/Ca01`` .. ``cpg/Ca25``), all of which
    are fed by a single shared stem layer (e.g. ``dna/dropout_1``). This
    finds that shared stem generically, from the model's own output layers'
    inbound connectivity, rather than assuming any fixed naming or a single
    output.
    """
    output_layers = [model.get_layer(name) for name in model.output_names]
    if not output_layers:
        raise ValueError("model has no output layers to strip")

    def _feeder_layers(layer):
        return [inbound for node in layer._inbound_nodes for inbound in node.inbound_layers]

    feeder_sets = [tuple(_feeder_layers(layer)) for layer in output_layers]
    distinct = {tuple(l.name for l in feeders) for feeders in feeder_sets}
    if len(distinct) != 1 or len(feeder_sets[0]) != 1:
        raise ValueError(
            "expected every output head to share exactly one common stem layer; "
            f"found feeder sets {sorted(distinct)}"
        )
    stem_layer = feeder_sets[0][0]

    from keras.models import Model as KerasModel

    return KerasModel(model.inputs, stem_layer.output)


def _one_hot_window(sequence: str) -> np.ndarray:
    """[wlen] sequence string -> [wlen, 4] one-hot array; DeepCpG excludes the
    special 'N' channel from model inputs (get_alphabet(special=False))."""
    wlen = len(sequence)
    encoded = np.zeros((wlen, 4), dtype=np.float32)
    for position, char in enumerate(sequence.upper()):
        index = CHAR_TO_INT.get(char, CHAR_TO_INT["N"])
        if index < 4:
            encoded[position, index] = 1.0
        # 'N' bases are left as an all-zero row, matching DeepCpG's own
        # int_to_onehot() for the non-special four-letter alphabet.
    return encoded


def _sequence_window(genome: Fasta, chrom: str, one_based_pos: int, wlen: int) -> str:
    half = wlen // 2
    # `one_based_pos` is the 1-based coordinate of the CpG cytosine (this
    # repo's coordinate contract). pyfaidx slicing is 0-based half-open, so
    # centering on the cytosine means start = pos - 1 - half.
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
    batch_size: int = 256,
) -> np.ndarray:
    """Return one DeepCpG DNA-module stem embedding per locus.

    Parameters
    ----------
    loci:
        Sequence of ``(chromosome, one_based_cpg_cytosine_position)`` pairs.
        Chromosome names must match the FASTA's naming convention exactly
        (no implicit ``chr`` stripping/adding).
    model_dir:
        Directory containing an unzipped DeepCpG DNA-module model
        (``model.json`` + ``model_weights.h5``, as unpacked from a model zoo
        download -- see scripts/deepcpg/README.md).
    genome_file:
        FASTA (with a ``.fai`` index, or one pyfaidx can build) for the same
        assembly build the model_dir checkpoint was trained on.
    dna_wlen:
        Sequence window length in bp, centered on the CpG cytosine. Must
        match the window length encoded in the model's saved input shape.

    Returns
    -------
    np.ndarray
        ``[len(loci), stem_dim]`` float32 array, in input order.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model = _load_legacy_model(Path(model_dir))
    stem_model = _remove_output_layers(model)

    input_shape = stem_model.inputs[0].shape
    model_wlen = int(input_shape[1])
    if model_wlen != dna_wlen:
        raise ValueError(
            f"model_dir's DNA input window is {model_wlen}bp, but dna_wlen={dna_wlen} was requested"
        )

    genome = Fasta(str(genome_file))
    try:
        windows = [
            _sequence_window(genome, chrom, pos, dna_wlen) for chrom, pos in loci
        ]
    finally:
        genome.close()

    output = np.empty((len(loci), int(stem_model.outputs[0].shape[-1])), dtype=np.float32)
    for start in range(0, len(windows), batch_size):
        batch = windows[start : start + batch_size]
        one_hot = np.stack([_one_hot_window(window) for window in batch], axis=0)
        output[start : start + len(batch)] = stem_model.predict(one_hot, batch_size=len(batch))

    return output
