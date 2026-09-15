from __future__ import annotations

import importlib
from collections import OrderedDict
from typing import Any, Protocol

import numpy as np


class OnlineEncoder(Protocol):
    """Minimal runtime encoder contract used by online representation mode."""

    @property
    def dim(self) -> int: ...

    def encode(self, cpg_ids: np.ndarray) -> np.ndarray: ...


class OnlineRepresentationStore:
    """Generate CpG representations lazily from canonical CpG IDs.

    The provider is instantiated once in the main process and queried by matrix column.
    Online mode intentionally requires DataLoader ``num_workers=0`` because genomic FMs
    commonly own GPU state and are unsafe/expensive to replicate across worker processes.
    An optional bounded LRU cache avoids re-encoding recently queried loci.
    """

    def __init__(self, encoder: OnlineEncoder, matrix_cpg_ids: np.ndarray, cache_size: int = 0):
        self.encoder = encoder
        self.matrix_cpg_ids = np.asarray(matrix_cpg_ids, dtype=np.int64)
        self._dim = int(encoder.dim)
        self.coverage_mask = np.ones(len(self.matrix_cpg_ids), dtype=bool)
        self.cache_size = int(cache_size)
        if self.cache_size < 0:
            raise ValueError("cache_size must be >= 0")
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    @property
    def dim(self) -> int:
        return self._dim

    def _cache_get(self, cpg_id: int) -> np.ndarray | None:
        value = self._cache.get(int(cpg_id))
        if value is not None:
            self._cache.move_to_end(int(cpg_id))
        return value

    def _cache_put(self, cpg_id: int, value: np.ndarray) -> None:
        if self.cache_size == 0:
            return
        self._cache[int(cpg_id)] = np.asarray(value, dtype=np.float32)
        self._cache.move_to_end(int(cpg_id))
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def get_by_matrix_columns(self, matrix_columns: np.ndarray) -> np.ndarray:
        matrix_columns = np.asarray(matrix_columns, dtype=np.int64)
        cpg_ids = self.matrix_cpg_ids[matrix_columns]
        result = np.empty((len(cpg_ids), self.dim), dtype=np.float32)
        missing_positions: list[int] = []
        missing_ids: list[int] = []
        for i, cpg_id in enumerate(cpg_ids):
            cached = self._cache_get(int(cpg_id))
            if cached is None:
                missing_positions.append(i)
                missing_ids.append(int(cpg_id))
            else:
                result[i] = cached
        if missing_ids:
            encoded = np.asarray(self.encoder.encode(np.asarray(missing_ids, dtype=np.int64)), dtype=np.float32)
            if encoded.shape != (len(missing_ids), self.dim):
                raise ValueError(
                    f"online encoder returned shape {encoded.shape}; expected {(len(missing_ids), self.dim)}"
                )
            for position, cpg_id, value in zip(missing_positions, missing_ids, encoded):
                result[position] = value
                self._cache_put(cpg_id, value)
        return result

    def close(self) -> None:
        close = getattr(self.encoder, "close", None)
        if callable(close):
            close()
        self._cache.clear()


def import_factory(spec: str):
    """Import ``module:function`` and return the callable."""
    if ":" not in spec:
        raise ValueError("online provider.factory must be 'module:function'")
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attr, None)
    if not callable(factory):
        raise ValueError(f"online provider factory is not callable: {spec}")
    return factory


def build_online_store(cfg: dict[str, Any], matrix_cpg_ids: np.ndarray, *, repo_root, registry_path):
    provider_cfg = cfg.get("provider") or {}
    factory_spec = provider_cfg.get("factory")
    if not factory_spec:
        raise ValueError("representation.mode=online requires provider.factory")
    factory = import_factory(str(factory_spec))
    kwargs = dict(provider_cfg.get("kwargs") or {})
    # These common paths are injected only when the factory asks to use them; factories
    # may simply ignore/remove them via **kwargs.
    kwargs.setdefault("repo_root", repo_root)
    kwargs.setdefault("registry_path", registry_path)
    encoder = factory(**kwargs)
    return OnlineRepresentationStore(
        encoder,
        matrix_cpg_ids,
        cache_size=int(provider_cfg.get("cache_size", 0)),
    )
