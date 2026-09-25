"""Look up pretrained MethylGPT token vectors by Illumina probe ID.

This reads an already learned embedding table. It does not run the model on
sample methylation values or train new embeddings.
"""

import csv
import os
import pickle
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import torch


_SPECIAL_TOKENS = ("<pad>", "<cls>", "<eoc>")
_WEIGHT_KEY = "encoder.embedding.weight"
# Direct files linked from the pinned MethylGPT README's model/probe sections.
# Its first model folder is titled "methylGPT-tiny" and contains this checkpoint.
_TINY_CHECKPOINT_URL = (
    "https://drive.google.com/uc?export=download&id=1o6qv2iQuhF67FqbAzOAYZSM_31x8ZDeC"
)
_TYPE3_PROBES_URL = (
    "https://www.dropbox.com/scl/fi/2n6bx7j8v0aon0kwfsghp/probe_ids_type3.csv"
    "?rlkey=ly133xlce1xxjiku6tiski6qq&dl=1"
)


def extract_cpg_embeddings(
    probe_ids: Sequence[str],
    *,
    checkpoint_path: str | Path,
    probe_ids_csv: str | Path,
    autodownload: bool = False,
) -> np.ndarray:
    """Return pretrained CpG token vectors in request order, including repeats.

    ``probe_ids_csv`` must be the ``illumina_probe_id`` CSV used for the
    checkpoint. Its row order defines token IDs after the three special tokens.
    A matching row count alone cannot prove the checkpoint and CSV are paired.
    If ``autodownload`` is true, missing paths are filled with the published
    tiny checkpoint and type3 probe CSV. Existing files are never replaced.
    """
    # Input: probe names such as "cg00000109", without methylation values.
    if isinstance(probe_ids, (str, bytes)):
        raise TypeError("probe_ids must be a sequence of probe ID strings, not a string")
    requested = list(probe_ids)
    for probe_id in requested:
        if not isinstance(probe_id, str) or not probe_id or probe_id != probe_id.strip():
            raise ValueError(f"Invalid probe ID in request: {probe_id!r}")

    # Resources: the CSV defines token order; the checkpoint holds the vectors.
    csv_path = Path(probe_ids_csv)
    model_path = Path(checkpoint_path)
    if autodownload:
        _download_if_missing(csv_path, _TYPE3_PROBES_URL, "type3 probe CSV")
        _download_if_missing(model_path, _TINY_CHECKPOINT_URL, "tiny checkpoint")
    for path, label in ((csv_path, "Probe CSV"), (model_path, "Checkpoint")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")

    probe_to_index = _read_probe_indices(csv_path)
    token_embeddings = _read_embedding_table(model_path, len(probe_to_index))

    missing = list(
        dict.fromkeys(probe_id for probe_id in requested if probe_id not in probe_to_index)
    )
    if missing:
        preview = ", ".join(repr(probe_id) for probe_id in missing[:5])
        suffix = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
        raise KeyError(f"Unknown probe ID(s) in {csv_path}: {preview}{suffix}")

    # Output: one saved checkpoint row per input ID, including duplicates.
    token_indices = [probe_to_index[probe_id] for probe_id in requested]
    return token_embeddings.detach()[token_indices].numpy()


def _download_if_missing(path: Path, url: str, label: str) -> None:
    """Stream one published resource to a temporary file, then save it atomically."""
    if path.is_file():
        return
    if path.exists():
        raise ValueError(f"{label} path exists but is not a file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".download", delete=False
        ) as output:
            temp_path = Path(output.name)
            with urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, output)
        if temp_path.stat().st_size == 0:
            raise ValueError("the server returned an empty file")
        # A sharing page can return HTTP 200 with HTML instead of the file.
        if label == "type3 probe CSV":
            with temp_path.open("r", encoding="utf-8-sig", newline="") as file:
                if "illumina_probe_id" not in next(csv.reader(file), []):
                    raise ValueError("the response is not an illumina_probe_id CSV")
        else:
            downloaded_state = torch.load(temp_path, map_location="cpu", weights_only=True)
            if not isinstance(downloaded_state, dict) or _WEIGHT_KEY not in downloaded_state:
                raise ValueError(f"the response is not a MethylGPT checkpoint with {_WEIGHT_KEY!r}")
        os.replace(temp_path, path)
    except (OSError, URLError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise RuntimeError(f"Could not download {label} to {path}: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _read_probe_indices(csv_path: Path) -> dict[str, int]:
    """Copy MethylVocab's order: three special tokens, then CSV probe IDs."""
    probe_to_index = {}
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or reader.fieldnames.count("illumina_probe_id") != 1:
                raise ValueError(
                    f"Probe CSV must have exactly one illumina_probe_id column: {csv_path}"
                )

            for line_number, row in enumerate(reader, start=2):
                probe_id = row["illumina_probe_id"]
                if probe_id is None or not probe_id.strip() or probe_id != probe_id.strip():
                    raise ValueError(f"Blank or malformed probe ID at CSV line {line_number}")
                if probe_id in _SPECIAL_TOKENS or probe_id in probe_to_index:
                    raise ValueError(
                        f"Duplicate or special-token probe ID at CSV line {line_number}: {probe_id!r}"
                    )
                probe_to_index[probe_id] = len(_SPECIAL_TOKENS) + len(probe_to_index)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Cannot read probe CSV {csv_path}: {exc}") from exc
    return probe_to_index


def _read_embedding_table(model_path: Path, probe_count: int) -> torch.Tensor:
    """Read the learned token table and check it fits the CSV vocabulary."""
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Cannot load checkpoint {model_path}: {exc}") from exc
    if not isinstance(state, dict) or _WEIGHT_KEY not in state:
        raise ValueError(
            f"Checkpoint {model_path} must be a plain state dictionary containing {_WEIGHT_KEY!r}"
        )

    embeddings = state[_WEIGHT_KEY]
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 2
        or not embeddings.is_floating_point()
    ):
        raise ValueError(f"Checkpoint {_WEIGHT_KEY!r} must be a 2D floating-point tensor")
    expected_rows = len(_SPECIAL_TOKENS) + probe_count
    if embeddings.shape[0] != expected_rows or embeddings.shape[1] == 0:
        raise ValueError(
            f"Checkpoint embedding shape {tuple(embeddings.shape)} is incompatible with "
            f"{probe_count} CSV probes; expected ({expected_rows}, positive embedding dimension)"
        )
    if not torch.isfinite(embeddings).all().item():
        raise ValueError(f"Checkpoint {_WEIGHT_KEY!r} contains non-finite values")
    return embeddings
