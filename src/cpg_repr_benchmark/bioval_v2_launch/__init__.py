"""bioval v2 launch (phase 2a): embedding loading, endpoint drivers, probes, bootstrap, pre-run gate.

Lives OUTSIDE biological_validation_v2 on purpose: that package is definitions-only and must not import h5py/representations
(tests/test_bioval_v2_matching_pairs_splits.py::test_no_embedding_imports).
"""
