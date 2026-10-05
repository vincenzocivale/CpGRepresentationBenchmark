# QA report (manuscript package part 1)

Generated at git HEAD `305f538275ff1183dfea66903631d1d817f7fd7d` (nothing committed by this package). All numbers read from frozen machine-readable result files; nothing re-run.

- manifests: 22; source-data tables: 39; figures (pdf): 20; tables (tex): 4
- structural problems: 0
- expected-vs-computed checks: 35 (0 fail)

## Expected (brief) vs computed

| item | check | expected | computed | tol | pass |
|---|---|---:|---:|---:|---|
| fig1 | histone tracks (frozen config vs encode arms.json) | 1959 | 1959 | 0 | PASS |
| fig1 | DNase tracks (frozen config vs encode arms.json) | 533 | 533 | 0 | PASS |
| fig1 | histone tracks = 1,959 | 1959 | 1959 | 0 | PASS |
| fig1 | DNase tracks = 533 | 533 | 533 | 0 | PASS |
| fig1 | discovery fit loci chr1-19 = 388,599 | 388599 | 388599 | 0 | PASS |
| fig1 | embedding dim = 256 | 256 | 256 | 0 | PASS |
| fig1 | Histone->Histone+DNase mean rel dMSE (%) | -1.75 | -1.74867 | 0.01 | PASS |
| fig1 | Histone+DNase->Clean mean rel dMSE (%) | -0.51 | -0.509358 | 0.01 | PASS |
| fig2 | TCGA MSE@0.50 regulatory_histone_dnase | 0.014784 | 0.0147839 | 1e-05 | PASS |
| fig2 | TCGA MSE@0.50 cpgpt_large_locus | 0.016672 | 0.0166722 | 1e-05 | PASS |
| fig2 | TCGA MSE@0.50 deepcpg_dna_locus | 0.019624 | 0.0196242 | 1e-05 | PASS |
| fig2 | TCGA rel dMSE@0.50 mean cpgpt_large_locus (%) | 12.8 | 12.7726 | 0.1 | PASS |
| fig2 | TCGA rel dMSE@0.50 mean deepcpg_dna_locus (%) | 32.7 | 32.7401 | 0.1 | PASS |
| fig3 | GSE40279 individuals total = 656 | 656 | 656 | 0 | PASS |
| fig3 | train=524 | 524 | 524 | 0 | PASS |
| fig3 | val=66 | 66 | 66 | 0 | PASS |
| fig3 | test=66 | 66 | 66 | 0 | PASS |
| fig3 | external test rel dMSE@0.50 CpGPT (%) | 1.82 | 1.82259 | 0.01 | PASS |
| fig3 | external test rel dMSE@0.50 DeepCpG (%) | 6.13 | 6.12837 | 0.01 | PASS |
| fig3 | TCGA CpGPT % | 12.8 | 12.7726 | 0.1 | PASS |
| fig3 | TCGA DeepCpG % | 32.7 | 32.7401 | 0.1 | PASS |
| fig4 | microc_H1_intra10kb regulatory_histone_dnase_v1 | 0.5685 | 0.568457 | 0.0006 | PASS |
| fig4 | microc_H1_intra10kb cpgpt_large_locus | 0.531 | 0.531445 | 0.0006 | PASS |
| fig4 | microc_H1_intra10kb deepcpg_dna_locus | 0.503 | 0.502459 | 0.0006 | PASS |
| fig4 | fantom5_membership regulatory_histone_dnase_v1 | 0.833 | 0.833006 | 0.0006 | PASS |
| fig4 | fantom5_membership cpgpt_large_locus | 0.701 | 0.700821 | 0.0006 | PASS |
| fig4 | fantom5_membership deepcpg_dna_locus | 0.623 | 0.622856 | 0.0006 | PASS |
| fig4 | rt_consensus regulatory_histone_dnase_v1 | 0.703 | 0.703136 | 0.0006 | PASS |
| fig4 | rt_consensus cpgpt_large_locus | 0.486 | 0.486389 | 0.0006 | PASS |
| fig4 | rt_consensus deepcpg_dna_locus | 0.359 | 0.359355 | 0.0006 | PASS |
| fig4 | loyfer_profile regulatory_histone_dnase_v1 | -0.063 | -0.0627925 | 0.0012 | PASS |
| fig4 | loyfer_profile cpgpt_large_locus | 0.107 | 0.106477 | 0.0012 | PASS |
| fig4 | loyfer_profile deepcpg_dna_locus | 0.116 | 0.115456 | 0.0012 | PASS |
| table1_datasets | TCGA N train+val+test | 9178 | 9178 | 0 | PASS |
| table2_main_results | TCGA candidate MSE@0.5 | 0.014784 | 0.0147839 | 1e-05 | PASS |

## Doc vs machine-readable consistency (computed value formatted at doc precision must appear in the doc)

| label | doc | string searched | found |
|---|---|---|---|
| confirm mean rel MSE H->H+D | REGULATORY_CONFIRMATION_RESULTS.md | `-1.75%` | yes |
| confirm mean rel MSE H+D->Clean | REGULATORY_CONFIRMATION_RESULTS.md | `-0.51%` | yes |
| external test primary rel MSE@0.5 | EXTERNAL_RECONSTRUCTION_FINAL_REPORT.md | `+1.82%` | yes |
| external test secondary rel MSE@0.5 | EXTERNAL_RECONSTRUCTION_FINAL_REPORT.md | `+6.13%` | yes |
| family screen MSE regulatory_histone | REGULATORY_FAMILY_SCREEN_RESULTS.md | `0.016925` | yes |
| family screen MSE regulatory_histone_dnase | REGULATORY_FAMILY_SCREEN_RESULTS.md | `0.016680` | yes |
| family screen MSE regulatory_histone_tf | REGULATORY_FAMILY_SCREEN_RESULTS.md | `0.016798` | yes |
| family screen MSE regulatory_clean | REGULATORY_FAMILY_SCREEN_RESULTS.md | `0.016633` | yes |
| bioval loyfer_profile regulatory_histone_dnase_v1 | BIOLOGICAL_VALIDATION_V2_FINAL_REPORT.md | `-0.0628` | yes |
| bioval loyfer_profile functional_annotations_pca | BIOLOGICAL_VALIDATION_V2_FINAL_REPORT.md | `-0.0381` | yes |
| bioval loyfer_profile cpgpt_large_locus | BIOLOGICAL_VALIDATION_V2_FINAL_REPORT.md | `0.1065` | yes |
| bioval loyfer_profile deepcpg_dna_locus | BIOLOGICAL_VALIDATION_V2_FINAL_REPORT.md | `0.1155` | yes |
| bioval microc_H1_intra10kb regulatory_histone_dnase_v1 | BIOLOGICAL_VALIDATION_V2_FINAL_REPORT.md | `0.5685` | yes |
| bioval microc_H1_intra10kb functional_annotations_pca | BIOLOGICAL_VALIDATION_V2_FINAL_REPORT.md | `0.5676` | yes |
| bioval microc_H1_intra10kb cpgpt_large_locus | BIOLOGICAL_VALIDATION_V2_FINAL_REPORT.md | `0.5314` | yes |
| bioval microc_H1_intra10kb deepcpg_dna_locus | BIOLOGICAL_VALIDATION_V2_FINAL_REPORT.md | `0.5025` | yes |
| Phase A candidate seed 17 MSE@0.5 | REGULATORY_CONFIRMATION_PHASE_A_RESULTS.md | `0.014786` | yes |

## Inconsistencies and caveats found

1. Rounding only: brief quotes DeepCpG H1 Micro-C AUROC 0.503 and CpGPT Loyfer 0.107; files give 0.5025 and 0.1065 (docs print 0.5025/0.1065). Differences < 0.001; no substantive disagreement.
2. CpGPT-large is NOT the same object across the campaign and the later phases: the ENCODE campaign arm `fm/cpgpt_locus_large` is the compact SVD-256 variant (configs/representations/confirmation_matrix.yaml: variance retained 0.838), whereas Phase A / external / biological validation use the native 512-D `cpgpt_large_locus` (Phase A results doc lists dim 512). Fig 1B is labelled accordingly; absolute campaign MSE (CpGPT 0.0178) differs from Phase A (0.01667) by design (different evaluation split/protocol).
3. Deprecated pre-amendment analysis dir `external_reconstruction_v1/analysis/A/validation` exists next to `A_v1.2`; package uses only A_v1.2 (v1.2, AMENDMENT 2). Max |MSE difference| old vs new on validation per-run MSE over matched rows = 0.
4. The across-seed 'mean relative' is the unweighted mean of per-seed paired relatives (12.77% / 32.74%), which differs slightly from the ratio of seed-mean MSEs; both rounded values match the brief (12.8 / 32.7).
5. The confirmation decision.json is `provisional` (at least one best epoch in the last 10 epochs: not converged) and its verdict is 'Histone+DNase preferred by parsimony pending author judgement'; Fig 1D therefore makes no equivalence claim.
6. The family-screen (1 seed, batch 32, 30 ep.) and confirmation (3 seeds, batch 8, 80 ep.) absolute MSEs are on different protocols and are drawn on separate axes.
7. Fig 1B attribution arms come from the campaign's evaluation split (docs call it 'frozen test patients'); they are labelled DISCOVERY EVIDENCE, not confirmation.
8. FANTOM5 membership N / CpG overlap and platform/tissue descriptors for Loyfer/Micro-C/FANTOM5/RT in Table 1 have no machine-readable source in the result files (flagged `source: doc` in table1 manifest/notes); TCGA test N (918) and 450K platform are doc-sourced.
9. Fig 2/3 CI for the across-seed mean is not provided by the registered analyses (only per-seed CIs and min/max); the package shows per-seed CIs plus the unweighted mean and does not invent a pooled CI.

## Missing results / placeholders

- S4_svd_compression: compression statistics for the frozen regulatory_histone_dnase feature set (not in report.json)
- S8_chr20_22_robustness: TCGA Phase A per-chromosome (chr20-22) analysis (no machine-readable source)
- S_final_loyfer_audit_placeholder: Loyfer failure audit final outputs
- S_final_loyfer_audit_placeholder: PLACEHOLDER figure produced
- fig5_scaffold: Loyfer failure audit outputs (exploratory post-hoc audit in progress/paused)
- fig5_scaffold: PLACEHOLDER figure produced
- Fig 5 (final) and S-final: scaffold/placeholder only (Loyfer audit pending; audit directory never read).
- 5 sensitivity items listed as NOT_EXECUTED_REQUIRES_NEW_PREREGISTRATION_OR_ARTIFACTS in S13.

## Compilation / lint

- py_compile common.py: ok
- py_compile make_fig1.py: ok
- py_compile make_fig2.py: ok
- py_compile make_fig3.py: ok
- py_compile make_fig4.py: ok
- py_compile make_fig5_scaffold.py: ok
- py_compile make_qa.py: ok
- py_compile make_supp.py: ok
- py_compile make_tables.py: ok

ruff (--isolated, selected rules E9,F63,F7,F82,F401,F811,F841; star-imports from common are intentional):
```
All checks passed!
```

## sha256 of every source-data table

| file | sha256 |
|---|---|
| paper/source_data/S10_external_convergence_history.csv | 0c27d5937968c64e97cb971a472b0f8ebad21e1494c38be688a5c758101a1d16 |
| paper/source_data/S11_transfer_B_contrasts.csv | 6b9252a3609144a0643c32419ceff97404b5ccd62c3cf7c8939695565c340462 |
| paper/source_data/S11_transfer_B_metrics.csv | c81180a692eb00e851e6011437b6269319df8056be4117e7f5bd8887fd87dc1e |
| paper/source_data/S12_bioval_complete_primary.csv | db83664904074f8a07f056ff667ab9f4cd2b50353e9c3b9172015235c050fcb2 |
| paper/source_data/S12_bioval_complete_primary_contrasts.csv | 02c2a604f08ae61b2fb3520476f054874c27622263565f94336344ae24ecd2c2 |
| paper/source_data/S13_bioval_sensitivities_verdicts.csv | 87fd97a08ed6b63835b55f10154ebc6c45c3a3a19824a4e4e74298cfff303220 |
| paper/source_data/S14_loyfer_exploratory_posthoc_geometry.csv | 1d6d05fbc2a659d44bfe9662f0da4f560c49486ee1b553e39d60ba10d52ff451 |
| paper/source_data/S14_loyfer_exploratory_posthoc_ridge.csv | 90cc6e15b5425293b5b04f213d2e31ae14ec4522d4da8b5dd2e2c142b60c09ac |
| paper/source_data/S1_encode_campaign_confirm.csv | 8296dcfd7bc3235d29a30c74d81fb4b5ffa1ba116bc77bf82eab3b127329fbde |
| paper/source_data/S1_encode_campaign_discovery.csv | d67ce7fc5182287b765a996d18d400eae9a87485f4854e22ae8387a1fde0ac59 |
| paper/source_data/S2_matched_histone_contrasts.csv | 449b29c20ab7da705cd8f13b95786d289e5405ddeb0b0cc1b69cbd439fd809a3 |
| paper/source_data/S2_matched_histone_mse.csv | 516fb935c845f8b20a6e75a49c94bb45a4282d3e3b64542ee1cad5372b612499 |
| paper/source_data/S3_h3k4me3_contrasts.csv | af6b70959b02701bae114721e40f7bf176a70b342d3235854b27b91c390f89c7 |
| paper/source_data/S3_h3k4me3_mse.csv | 7e01409d60e5513e0d958652e97020bb6c4d29b8b4378e383b7293a32b7b4b79 |
| paper/source_data/S4_svd_compression_cumulative_energy.csv | 1e0c207a1d05bb931c3947668d54d0d55c5f46a5dc17ee7c856e630b324fff49 |
| paper/source_data/S4_svd_compression_summary.csv | f83157e7000178b1e990c720e7555b8b12fa7f503441b4f2c3c03b4e58013bd0 |
| paper/source_data/S5_training_curves_curves.csv | 867295484b352376473d9279a207c1834729ffc44a0443198b2ae68a849ebf75 |
| paper/source_data/S6_tcga_all_metrics_metrics.csv | fb276f4b14ef3fce25c578733bc59113ddaf3d3af7013eeac9b07b1747d49c4d |
| paper/source_data/S7_tcga_paired_deltas_paired.csv | d6e7f21647104dae3271da01d3eef22d6142a1d6bb1e79b91a2d1b3556c27af3 |
| paper/source_data/S8_chr20_22_robustness_contrasts.csv | 9cb840cb2c76766b1f11fc791a7d71e29f5708ef16f24824a021ff6388d2a3b5 |
| paper/source_data/S8_chr20_22_robustness_mse_by_scope.csv | 519e054ede39f66096888832823f5c3ed8e9ee992939833d16bd3a6b84702115 |
| paper/source_data/S9_external_all_metrics_metrics.csv | fbe283be38ee9d7a90805db2a7ab1bd4505bef947a3d862db19d7cc627a22fd1 |
| paper/source_data/fig1_A_counts.csv | f18ecdfb1c01a77c7f7caaa7b6fd13581ea5f8d2514351748c6a5a3b2f689fc7 |
| paper/source_data/fig1_B.csv | 631b185f3e980db1caee35427e14dfa7f313327154ac9fbef630ab11b0a34c49 |
| paper/source_data/fig1_C_confirmation.csv | 158702f98c5b1370ef5f42b0a7d1c579ee08f29bc807f5be5d715dd7081631b7 |
| paper/source_data/fig1_C_family_screen.csv | 88c21b4c17844587f2020679052376d34f03b5fb5845b3a972302eb06a117dcc |
| paper/source_data/fig1_D.csv | 63605a3a89db11a8eecbfde8281df2a7d43dac1d06cc69f44ba1207670e81877 |
| paper/source_data/fig2_A.csv | 46d5cc549639436af768fbe4f5c515db9aa8671f817643c4245d74cd58299300 |
| paper/source_data/fig2_B.csv | 72c0f3012589c84e5bd28aa6269c57be0f1217af89dadf1c56e2ffe1357e8542 |
| paper/source_data/fig2_C.csv | 48bf2885a11313deeb97c4d99548f54fb4ff071f936d9075172065397a740aa1 |
| paper/source_data/fig2_D.csv | c2af98bcc46251a29106240d2374b49e0bd1701b1f254467e30ad838e8fc507c |
| paper/source_data/fig3_A_split.csv | 651bc4f69167a3ade0cf7837c990901f213f0f514f551096d62731678ecdc381 |
| paper/source_data/fig3_B.csv | bb28a1493c5b27d208c6733745c979f06f21d1e2b3f7e0d4e8a87620b5b4815a |
| paper/source_data/fig3_C.csv | fa84c03a1ba429bd3c2594e84a0d205dc0507415790d3f08106b6c8492e08a0b |
| paper/source_data/fig3_D.csv | 12865a522e676052fb1473081fdf46e5338ef33b9b312114610bb3f135cc44cc |
| paper/source_data/fig4_A.csv | dcb1d01532f530aa6b868e762ec02734c6faf65a0ed241d6708954472234fc16 |
| paper/source_data/fig4_B.csv | e17b1617940f8498a26822398d6c9b2a2c9a05d22f99256784b494193cd36abf |
| paper/source_data/fig4_C.csv | 10175c5b71d460bc2414f25a28cb48525be925354b033fc9d99cf56f9282fb43 |
| paper/source_data/fig4_D.csv | e86ddac3895a21880e21544b5601180b41bc6c3a676b0ecfdb3c4ce730943d2d |
