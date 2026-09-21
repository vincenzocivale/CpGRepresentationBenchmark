# Adding a CpG representation

The reconstruction model must remain representation-agnostic. New approaches are integrated at the representation boundary only.

## Preferred path: precompute once

Export a canonical HDF5 file:

```text
/cpg_idx    int64 [N]
/embedding  float16/float32 [N,D]
```

Then create a config with:

```yaml
representation:
  name: dnambert
  mode: precomputed
  store_h5: data/cache/representations/dnambert.h5
```

This is the preferred paper path because feature extraction and downstream training become independently reproducible.

## Automatic materialization

Use `generate_if_missing` or `regenerate` and supply a command that writes the canonical HDF5. Available placeholders are `{output}`, `{registry}`, `{repo_root}` and `{run_dir}`.

## True online encoding

For models that should encode CpGs at runtime, implement a factory returning an object with:

```python
@property
def dim(self) -> int: ...

def encode(self, cpg_ids: np.ndarray) -> np.ndarray: ...
```

Configure it as `representation.mode: online`. Online mode requires DataLoader worker counts of zero so a GPU FM is not replicated into subprocesses. `functional_provider.py` is the reference implementation.

## Fairness checklist

Before accepting an arm into the main table:

1. Use the exact same persistent locus protocol file.
2. Use the exact same patient split seed and training budget.
3. Do not alter reconstruction architecture or losses for one representation.
4. Do not silently drop loci that an encoder cannot represent.
5. Record checkpoint/model version, sequence window, pooling rule and genome build in `provenance`.
6. For sequence models, define exactly how a CpG vector is obtained (C/G token, mean, CLS, window pooling, layer).
7. Keep representation extraction patient-agnostic for the primary claim.
8. Run both `seen` and `unseen_locus` views at all masking fractions.
9. Add the fixed-dimensional train-locus PCA control before claiming a pure representation advantage when raw dimensions differ.

## Planned adapters

The initial deployment only needs the already-saved functional and NTv3-pre stores. Next adapters should cover MethylGPT, DNAmBERT, MethylProphet/its sequence representation and additional genomic foundation models. Their extraction code should never leak into the downstream model package.

CpGPT is implemented: `scripts/cpgpt/` holds a self-contained extractor with no
dependency on `cpg_repr_benchmark` (it must run in a separate, CpGPT-pinned
environment; see `scripts/cpgpt/README.md`), and `scripts/build_cpgpt_embedding.py`
converts this repo's CpG registry into CpGPT location keys and writes the
canonical store registered as `cpgpt_locus_native` in
`configs/representations/future_models.yaml`.
