from __future__ import annotations

from pathlib import Path

import numpy as np

from cpg_repr_benchmark.data.splits import patient_disjoint_split, patient_id


def load_patient_protocol(path: Path, names: list[str]) -> dict:
    with np.load(path, allow_pickle=False) as h:
        if not np.array_equal(h['sample_names'], np.asarray(names, str)):
            raise ValueError('Patient protocol sample axis differs')
        splits = {k: h[k] for k in ('train', 'validation', 'test')}
    joined = np.concatenate(list(splits.values()))
    if (any(len(v) == 0 for v in splits.values())
            or not np.array_equal(np.sort(joined), np.arange(len(names)))):
        raise ValueError('Patient protocol must partition the complete sample axis')
    patients = [{patient_id(names[i]) for i in rows} for rows in splits.values()]
    if any(patients[i] & patients[j] for i in range(3) for j in range(i)):
        raise ValueError('Patient leakage across protocol splits')
    return splits


def freeze_patient_protocol(path: Path, names: list[str], seed=20260925):
    if path.exists():
        return load_patient_protocol(path, names)
    splits = patient_disjoint_split(names, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **splits, sample_names=np.asarray(names, str), seed=seed)
    tmp.replace(path)
    return load_patient_protocol(path, names)
