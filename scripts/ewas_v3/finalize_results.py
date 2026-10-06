#!/usr/bin/env python3
"""Finalize model outputs, correct AP tie handling uniformly, audit shared pairs and export paper artifacts."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,pandas as pd,yaml
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'outputs/ewas_v3/ewas-v3-main-panel-freeze-v1'
reg=ROOT/'data/registry/ewas_v3_results';reg.mkdir(parents=True,exist_ok=True)
m=yaml.safe_load((ROOT/'configs/ewas_v3/main_trait_panel.yaml').read_text())
ex=json.loads((out/'execution_manifest.json').read_text());assert ex['status']=='COMPLETE'
# Correct the inherited row-sorted AP kernel at tied scores; prediction/probe/selection unchanged.
from sklearn.metrics import average_precision_score
import subprocess
corrections=[]
for endpoint in out.rglob('endpoint.json'):
 e=json.loads(endpoint.read_text())
 if e['status']!='complete':continue
 df=pd.read_parquet(endpoint.parent/'oof.parquet')
 value=float(average_precision_score(df.y.to_numpy(),df.oof_score.to_numpy()))
 legacy=e.get('AUPRC_legacy_unaggregated_ties',e['AUPRC'])
 corrections.append({k:e[k] for k in ['task','threshold','design','arm']} | {'legacy_row_sorted_AP':legacy,'standard_AP':value,'delta':value-legacy})
 e['AUPRC_legacy_unaggregated_ties']=legacy;e['AUPRC']=value
 e['AUPRC_definition']='sklearn.metrics.average_precision_score: aggregate exact tied-score groups'
 endpoint.write_text(json.dumps(e,indent=2)+'\n')
corrected=pd.read_csv(out/'all_results.csv')
for c in corrections:
 mask=(corrected.task==c['task'])&(corrected.threshold==c['threshold'])&(corrected.matching_design==c['design'])&(corrected.representation==c['arm'])
 assert mask.sum()==1
 corrected.loc[mask,'AUPRC']=c['standard_AP']
corrected.to_csv(out/'all_results.csv',index=False)
corrected_main=corrected[(corrected.threshold=='primary')&(corrected.matching_design=='main')]
corrected_main.to_csv(out/'main_results.csv',index=False)
pd.DataFrame(corrections).to_csv(out/'auprc_tie_correction.csv',index=False)
report_path=out/'report.md'
report_path.write_text(report_path.read_text().split('## Primary results')[0]+'## Primary results\n\nAUPRC uses standard tied-score groups (uniform correction to the inherited legacy row-order kernel).\n\n'+corrected_main[['category','trait','representation','AUROC','ci_lo','ci_hi','AUPRC','matched_pair_concordance']].to_csv(index=False))
ex['postprocessing']={'script':'scripts/ewas_v3/finalize_results.py','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'AP_tie_correction':'Standard AP grouping equal scores; applied uniformly to every evaluated endpoint; no model refits, no AUROC/CI/pair/panel changes','n_AP_values_changed':sum(abs(c['delta'])>1e-12 for c in corrections)}
ex['output_sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*') if p.is_file() and p.name!='execution_manifest.json'}
# Finalize the stdout log checksum only after the execution process has closed it.
ex['output_sha256']['run.log']=hashlib.sha256((out/'run.log').read_bytes()).hexdigest()
ex['log_checksum_finalized_after_process_completion']=True
(out/'execution_manifest.json').write_text(json.dumps(ex,indent=2)+'\n')
for name,digest in ex['output_sha256'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest,name
allrows=pd.read_csv(out/'all_results.csv');main=pd.read_csv(out/'main_results.csv');cs=pd.read_csv(out/'main_paired_contrasts.csv')
assert len(main)==39 and len(cs)==39 and len(allrows)==156
assert (main.status=='complete').all() and (main.n_nonconverged_fits==0).all()
assert set(main.task)=={t['task'] for t in m['main_tasks']}
assert not main.duplicated(['task','representation']).any()
checks=[]
for t in m['main_tasks']:
 for cutoff,s in t['sets'].items():
  for design,matching in s['matching'].items():
   pairs=pd.read_csv(ROOT/matching['path']);reference=None
   for arm in ['regulatory_histone_dnase_v1','cpgpt_large_locus','deepcpg_dna_locus']:
    p=out/cutoff/design/'arms'/arm/t['task'];e=json.loads((p/'endpoint.json').read_text())
    assert e['n_matched']==len(pairs)==matching['n_matched_positives']
    if e['status']=='complete':
     assert hashlib.sha256((p/'oof.parquet').read_bytes()).hexdigest()==e['oof_sha256']
     df=pd.read_parquet(p/'oof.parquet');n=len(pairs)
     assert np.array_equal(df.cpg_idx.to_numpy(),np.r_[pairs.positive_cpg_idx,pairs.background_cpg_idx])
     assert np.array_equal(df.fold.to_numpy(),np.tile(pairs.outer_fold,2))
     assert np.array_equal(df.chrom.to_numpy(),np.tile(pairs.chromosome,2))
     key=df.drop(columns='oof_score')
     if reference is not None:assert reference.equals(key)
     reference=key
     scores=df.oof_score.to_numpy();assert np.isfinite(scores).all()
     assert np.mean(scores[:n]>scores[n:])==e['matched_pair_concordance']
    else:assert cutoff=='sensitivity' and set(pairs.outer_fold.unique())!=set(range(5))
    checks.append({'task':t['task'],'threshold':cutoff,'matching_design':design,'representation':arm,'status':e['status'],'n_matched':len(pairs),'n_nonconverged_fits':e.get('n_nonconverged_fits',0)})
audit={'n_primary_rows':39,'n_primary_contrasts':39,'n_total_endpoint_rows':156,'n_complete':int((allrows.status=='complete').sum()),'n_not_evaluable':int((allrows.status=='not_evaluable').sum()),'not_evaluable_tasks':sorted(allrows.loc[allrows.status=='not_evaluable','task'].unique().tolist()),'n_nonconverged_fits':int(allrows.n_nonconverged_fits.sum()),'checks':['All 13 frozen source analyses retained','Identical frozen locus pairs and chromosome folds across all arms at every design/threshold','OOF rows aligned exactly to frozen pair files','Finite OOF scores for every evaluated endpoint','Strict matched-pair concordance reproduced from OOF scores','All primary endpoints and paired contrasts complete','Non-evaluable sensitivity endpoints retain five-fold protocol; no task substitutions'],'execution_manifest_sha256':hashlib.sha256((out/'execution_manifest.json').read_bytes()).hexdigest()}
(reg/'completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
pd.DataFrame(checks).to_csv(reg/'endpoint_completion_audit.csv',index=False)
source_groups=pd.DataFrame([{'task':t['task'],'category':t['category'],'PMID':t['PMID'],'zenodo_record':t['zenodo_record'],'source_analysis_id':t['source_analysis_id']} for t in m['main_tasks']])
source_groups['n_tasks_same_record']=source_groups.groupby('zenodo_record')['task'].transform('size')
source_groups['n_tasks_same_PMID']=source_groups.groupby('PMID')['task'].transform('size')
source_groups.to_csv(reg/'shared_source_groups.csv',index=False)
for name in ['main_results.csv','main_paired_contrasts.csv','all_results.csv','all_paired_contrasts.csv','execution_manifest.json','environment.json','auprc_tie_correction.csv']:
 shutil.copyfile(out/name,reg/name)
fig=ROOT/'docs/figures';fig.mkdir(parents=True,exist_ok=True)
for ext in ['pdf','png']:shutil.copyfile(out/f'main_auroc.{ext}',fig/f'ewas_v3_main_auroc.{ext}')
lines=['# EWAS-v3: approved panel results','',f'Execution complete: **39 main endpoint rows**, **39 paired AUROC contrasts**; primary p < 1e-4, EWAS Catalog-associated CpGs. Frozen panel: 13 tasks, Cancer 5, immune/inflammatory 4, cardiometabolic 4. Source and task selection were fixed before evaluation.','', 'All three stores passed SHA256/dimension checks and full 408,399-locus coverage. All main matched pairs are shared across representations. Scalers are fitted only to each inner/outer training partition; nested L2 logistic probe, five chromosome folds, seed 17. CI: 1,000 paired chromosome bootstrap draws. Concordance is strict positive > background (ties zero). AUPRC is descriptive because of the matched 50% prevalence.','', '## Main absolute AUROC (95% chromosome-bootstrap CI)','', '| Category | Trait / endpoint | Regulatory 256D | CpGPT 512D | DeepCpG 128D |','|---|---|---:|---:|---:|']
for t in m['main_tasks']:
 subset=main[main.task==t['task']].set_index('representation')
 vals=[f"{subset.loc[a,'AUROC']:.3f} [{subset.loc[a,'ci_lo']:.3f}, {subset.loc[a,'ci_hi']:.3f}]" for a in ['regulatory_histone_dnase_v1','cpgpt_large_locus','deepcpg_dna_locus']]
 lines.append('| '+' | '.join([t['category'],t['trait']]+vals)+' |')
lines+=['','![Absolute AUROC by biological category](figures/ewas_v3_main_auroc.png)','', 'Figure markers † indicate >50% shared smaller positive set. The five flagged cardiometabolic pairs remain selected and cannot be interpreted as independent replications. Shared-source registry records repeated PMIDs and archival deposits separately: a bulk archive can contain independent studies. Selected endpoints with the same PMID may share participants; positive-set overlap alone does not establish cohort independence. No numerical participant overlap is inferred. No category AUROC averages or category win counts are calculated.','', '## Sensitivities and limitations','', f'Both the frozen genomic-context sensitivity and p < 1e-7 sensitivity were executed for the same sources/representations. Across four design/threshold combinations, **{audit["n_complete"]} endpoint rows are evaluated** and **{audit["n_not_evaluable"]} rows are not_evaluable**. The latter correspond to melanoma, colorectal cancer and psoriasis at p < 1e-7, across both matching designs and all three arms: fewer than five occupied chromosome folds. No replacement CV or tasks were introduced. Eight tasks have <100 strict-threshold positives and retain insufficient_support labels.','', f'Non-converged fits: **{audit["n_nonconverged_fits"]}**. AUPRC uses standard grouped ties; the inherited row-order AP defect is uniformly corrected from saved OOF scores, without fitting new models or changing AUROC/contrasts. Correction diagnostics are provided. All 39 main paired contrasts are available; bootstrap p-values are descriptive and unadjusted.','', 'Provenance: official EWAS Catalog Zenodo fallback; full current Catalog coverage remains incomplete, and actual Catalog StudyIDs are unavailable. Source analysis IDs, filenames, Zenodo records, PMIDs and hashes remain in the frozen manifest. The manuscript should describe these tasks as prediction of Catalog-associated CpGs, with tissue/sample-size differences recorded, rather than causal validation.','', '## Artifacts','', '- Tidy main table: `data/registry/ewas_v3_results/main_results.csv`.', '- Paired contrasts: `data/registry/ewas_v3_results/main_paired_contrasts.csv`.', '- All sensitivity endpoint/contrast rows: `all_results.csv` and `all_paired_contrasts.csv` in the same directory.', '- Completion audit, environment and execution/store/code provenance are versioned in the same directory.', '- Publication figures: `docs/figures/ewas_v3_main_auroc.pdf` and `.png`.', '- Per-fold checkpoints and OOF scores remain in `outputs/ewas_v3/ewas-v3-main-panel-freeze-v1/`.', '- Panel freeze commit: `640fdb5`; approval/runner commit: `44cf265`.','']
(ROOT/'docs/EWAS_V3_RESULTS.md').write_text('\n'.join(lines))
files=list(reg.glob('*'))+[ROOT/'docs/EWAS_V3_RESULTS.md']+[fig/f'ewas_v3_main_auroc.{ext}' for ext in ['pdf','png']]
(reg/'artifact_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.name!='artifact_sha256.json'},indent=2)+'\n')
print(json.dumps(audit,indent=2))
print(main[['task','representation','AUROC','ci_lo','ci_hi']].to_string(index=False))
