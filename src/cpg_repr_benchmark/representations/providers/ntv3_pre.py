from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


class NTv3PreProvider:
    """Direct coordinate -> NTv3-pre locus embedding provider.

    Contract used by the benchmark:
      - checkpoint: InstaDeepAI/NTv3_650M_pre by default;
      - GRCh38 centred window (default 32,768 bp);
      - forward orientation;
      - final decoder representation;
      - mean of the two output bins covering the central C/G.

    The checkpoint is loaded once.  The materializer, not downstream training,
    owns inference so the expensive genomic FM computation is cached and reused.
    """

    name = "ntv3_pre"

    def __init__(
        self,
        *,
        fasta_path: Path,
        checkpoint: str = "InstaDeepAI/NTv3_650M_pre",
        window_length: int = 32768,
        device: str = "cuda",
        bf16: bool = True,
        compile_model: bool = False,
        compile_mode: str = "max-autotune-no-cudagraphs",
        embedding_dim: int = 1536,
    ) -> None:
        try:
            from pyfaidx import Fasta
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "NTv3 materialization requires pyfaidx and transformers>=4.55; "
                "install them in the benchmark environment before running this provider"
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested {device}, but CUDA is unavailable")
        if int(window_length) % 128:
            raise ValueError("NTv3 window_length must be a multiple of 128")
        if int(window_length) < 2:
            raise ValueError("window_length must be >= 2")

        self.fasta_path = Path(fasta_path)
        if not self.fasta_path.exists():
            raise FileNotFoundError(self.fasta_path)
        self.checkpoint = str(checkpoint)
        self.window_length = int(window_length)
        self.device = torch.device(device)
        self.bf16 = bool(bf16)
        self.compile_model = bool(compile_model)
        self.compile_mode = str(compile_mode)
        self.dim = int(embedding_dim)

        kwargs = {"trust_remote_code": True}
        if self.bf16:
            kwargs.update(
                {
                    name: "bfloat16"
                    for name in (
                        "stem_compute_dtype",
                        "down_convolution_compute_dtype",
                        "transformer_qkvo_compute_dtype",
                        "transformer_ffn_compute_dtype",
                        "up_convolution_compute_dtype",
                        "modulation_compute_dtype",
                    )
                }
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, **kwargs)
        self.model = AutoModelForMaskedLM.from_pretrained(self.checkpoint, **kwargs).to(self.device).eval()
        if self.compile_model:
            self.model = torch.compile(self.model, mode=self.compile_mode)
        self._token_lut = self._build_fast_char_token_lut()
        self.genome = Fasta(
            str(self.fasta_path),
            as_raw=True,
            sequence_always_upper=True,
            rebuild=False,
        )
        self._genome_keys = set(self.genome.keys())

    def _build_fast_char_token_lut(self) -> np.ndarray:
        lut = np.full(256, -1, dtype=np.int64)
        for base in "ACGTN":
            ids = self.tokenizer(base, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise RuntimeError(f"NTv3 tokenizer is not character-level for {base!r}: {ids}")
            lut[ord(base)] = int(ids[0])
        probe = ("ACGTN" * 103)[:511]
        expected = np.asarray(
            self.tokenizer(probe, add_special_tokens=False)["input_ids"], dtype=np.int64
        )
        observed = lut[np.frombuffer(probe.encode("ascii"), dtype=np.uint8)]
        if not np.array_equal(expected, observed):
            raise RuntimeError("fast NTv3 character tokenizer failed equivalence check")
        return lut

    def _tokenize(self, sequences: list[str]) -> torch.Tensor:
        if not sequences:
            raise ValueError("empty NTv3 sequence batch")
        length = len(sequences[0])
        if any(len(sequence) != length for sequence in sequences):
            raise ValueError("NTv3 sequence batch contains unequal sequence lengths")
        raw = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8).reshape(
            len(sequences), length
        )
        ids = self._token_lut[raw]
        if (ids < 0).any():
            unsupported = sorted(chr(int(x)) for x in np.unique(raw[ids < 0]))
            raise ValueError(f"unsupported FASTA characters: {unsupported}")
        return torch.from_numpy(np.ascontiguousarray(ids)).to(self.device)

    @staticmethod
    def _base_to_output(base_index: int, input_length: int, output_length: int) -> int:
        if not 0 <= base_index < input_length or output_length < 1:
            raise ValueError("invalid base/output coordinate")
        return min(output_length - 1, int(((base_index + 0.5) * output_length) // input_length))

    def _forward(self, input_ids: torch.Tensor):
        model = self.model
        raw_model = getattr(model, "_orig_mod", model)
        if hasattr(raw_model, "encode_species"):
            species_ids = raw_model.encode_species(["human"] * input_ids.shape[0]).to(input_ids.device)
            return model(input_ids=input_ids, species_ids=species_ids)
        core = raw_model.core
        core.config.embeddings_layers_to_save = [len(core.transformer_blocks)]
        core.config.deconv_layers_to_save = [len(core.deconv_tower_blocks)]
        output = core(input_ids=input_ids, output_hidden_states=False)
        final = output[f"embeddings_deconv_{len(core.deconv_tower_blocks)}"].permute(0, 2, 1)
        transformer = output[f"embeddings_{len(core.transformer_blocks)}"]
        return SimpleNamespace(embedding=final, after_transformer_embedding=transformer)

    def _centre_embedding(self, output) -> torch.Tensor:
        tensor = output.embedding
        c = self.window_length // 2 - 1
        g = self.window_length // 2
        ci = self._base_to_output(c, self.window_length, tensor.shape[1])
        gi = self._base_to_output(g, self.window_length, tensor.shape[1])
        return tensor[:, [ci, gi]].float().mean(dim=1)

    def _sequence(self, chrom: str, pos1: int) -> str:
        cname = str(chrom)
        if cname not in self._genome_keys:
            alternate = cname.removeprefix("chr")
            if alternate in self._genome_keys:
                cname = alternate
            else:
                raise KeyError(f"FASTA contains neither {chrom!r} nor {alternate!r}")
        chrom_len = len(self.genome[cname])
        start0 = int(pos1) - self.window_length // 2
        end0 = start0 + self.window_length
        left = max(0, -start0)
        right = max(0, end0 - chrom_len)
        body = str(self.genome[cname][max(0, start0) : min(chrom_len, end0)])
        sequence = "N" * left + body + "N" * right
        if len(sequence) != self.window_length:
            raise RuntimeError(f"failed to build exact NTv3 window for {chrom}:{pos1}")
        centre = sequence[self.window_length // 2 - 1 : self.window_length // 2 + 1]
        if centre != "CG":
            raise ValueError(
                f"GRCh38 CpG validation failed for {chrom}:{pos1}; central dinucleotide={centre!r}"
            )
        return sequence

    @torch.inference_mode()
    def encode(self, cpg_idx: np.ndarray, chrom: np.ndarray, pos: np.ndarray) -> np.ndarray:
        del cpg_idx
        sequences = [self._sequence(str(c), int(p)) for c, p in zip(chrom, pos)]
        input_ids = self._tokenize(sequences)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.bf16 and self.device.type == "cuda",
        ):
            output = self._forward(input_ids)
            centre = self._centre_embedding(output)
        values = centre.detach().cpu().numpy().astype(np.float32)
        if values.ndim != 2 or values.shape[1] != self.dim:
            raise RuntimeError(
                f"unexpected NTv3 embedding shape {values.shape}; configured dimension={self.dim}"
            )
        return values

    def metadata(self) -> dict:
        return {
            "track": "native_frozen",
            "component": "locus_only",
            "checkpoint": self.checkpoint,
            "posttrained": False,
            "fasta": str(self.fasta_path),
            "window_length_bp": self.window_length,
            "pooling": "central_CG_mean_final_decoder",
            "orientation": "forward",
            "bf16": self.bf16,
            "compiled": self.compile_model,
        }

    def close(self) -> None:
        try:
            self.genome.close()
        except Exception:
            pass
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
