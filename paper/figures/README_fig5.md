# Figure 5 scaffold (Loyfer mechanism)

`fig5_scaffold.{pdf,svg,png}` is a layout-only placeholder. Every panel is stamped "EXPLORATORY - pending Loyfer audit" and contains no data.
No result file was read to build it (in particular nothing under the Loyfer failure-audit output directory).

Panels and the inputs that will populate them once the audit is final and frozen:
- A frozen native cosine: Loyfer primary per-arm values (frozen summary_primary.csv, endpoint loyfer_profile) vs audit levels.
- B raw Histone+DNase similarity: audit output of similarity computed on the raw 2,492-track feature matrix.
- C SVD/native geometry: audit comparison of SVD-256 geometry vs native geometry per arm.
- D PC-removal/centering diagnostic: audit outputs after removing leading PCs / centering (already-frozen post-hoc geometry deltas exist only as Supplement S14, labelled EXPLORATORY POST-HOC).
- E OOF predicted methylation-profile similarity: audit outputs from out-of-fold predicted profiles.
Replace `make_fig5_scaffold.py` by `make_fig5.py` reading the audit's frozen tables; keep the manifest/source-data conventions of the other figures.
