# Representation configs

Experiment YAMLs currently embed representation settings directly so every run is self-contained. This directory documents reusable templates for future adapters.

Recommended provenance fields:

```yaml
representation:
  name: example
  mode: precomputed
  store_h5: data/cache/representations/example.h5
  provenance:
    genome_build: hg38
    checkpoint: ...
    layer: ...
    pooling: ...
    sequence_window_bp: ...
    patient_specific: false
    supervision: none | methylation_proxy | masked_methylation | other
    locus_fit_scope: reference_only | train_protocol_only | external_pretrained | unknown
```

`locus_fit_scope` matters for the strongest unseen-locus claim. The reconstruction pipeline always holds loci out downstream; only representations fitted on the train protocol (or genuinely label-free reference features) support a stricter representation-OOD statement.
