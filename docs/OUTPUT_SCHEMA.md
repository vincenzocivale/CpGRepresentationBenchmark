# Output schema

Every experiment is discoverable at:

```text
outputs/
  masking/
    <dataset>/
      <representation>/
        seed_<seed>/
          <UTC timestamp>-<config hash>/
```

A run contains:

```text
experiment.json
resolved_config.yaml
representation_manifest.json
representation_generation.log        # only when cache generation ran
locus_split.npz
patient_split.npz
prior_logit.npy
history.json
checkpoints/
  epoch_XXXX.pt
  last.pt
  best.pt
evaluation/
  seen/
    mask_0.15/metrics.json
    ...
  unseen_locus/            # only present when experiment.locus_split.heldout_fraction > 0
    mask_0.15/metrics.json
    ...
summary.json
```

The default masking protocol is genome-wide and seen-only (`heldout_fraction: 0.0`), so most runs write
only `evaluation/seen/`. `evaluation/unseen_locus/` appears only for runs that opt into a nonzero
`heldout_fraction` for strict locus-OOD comparisons.

In addition, `experiment.locus_split.protocol_path` points to a persistent shared protocol under `data/protocols/` by default. That protocol is reused across representation arms and is separate from the per-run copy in `locus_split.npz`.

`experiment.json` contains the minimal provenance needed to identify a run without opening checkpoints. `summary.json` is the paper-table-friendly aggregate of every evaluation view and masking fraction.

To gather all completed runs into one CSV:

```bash
python scripts/summarize_runs.py --outputs outputs --csv outputs/masking_summary.csv
```
