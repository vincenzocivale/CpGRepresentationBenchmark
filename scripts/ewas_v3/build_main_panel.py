#!/usr/bin/env python3
"""Freeze the prespecified EWAS panel using support/provenance only; never loads embeddings."""
import sys, json, hashlib, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from cpg_repr_benchmark.ewas_v2 import design as d
from cpg_repr_benchmark.data.coordinates import encode_many

# Allowed endpoint labels, not preselected source analyses.
PANEL = [
 ('Cancer','melanoma',['Cutaneous melanoma versus nevi']),
 ('Cancer','oral_pharyngeal_scc',['Oral or pharyngeal squamous cell carcinoma']),
 ('Cancer','breast_cancer',['Incident Breast Cancer','Prevalent Breast Cancer (Self-report)']),
 ('Cancer','colorectal_cancer',['Incident Colorectal Cancer','Prevalent Colorectal Cancer (Self-report)']),
 ('Cancer','lung_cancer',['Incident Lung Cancer','Prevalent Lung Cancer (Self-report)']),
 ('Immune/inflammatory','ibd_family',["Crohn's disease",'Inflammatory bowel disease','Ulcerative colitis','Incident Inflammatory Bowel Disease']),
 ('Immune/inflammatory','rheumatoid_arthritis',['Rheumatoid arthritis','Incident Rheumatoid Arthritis','Prevalent Rheumatoid Arthritis (Self-report)']),
 ('Immune/inflammatory','psoriasis',['Psoriasis']),
 ('Immune/inflammatory','graves_disease',["Graves' disease"]),
 ('Cardiometabolic','adiposity_bmi',['Body mass index']),
 ('Cardiometabolic','glycemic_t2d',['Incident Type 2 Diabetes','Prevalent Type 2 Diabetes (Self-report)']),
 ('Cardiometabolic','lipid_hdl',['HDL cholesterol']),
 ('Cardiometabolic','cardiovascular_ihd',['Incident Ischemic Heart Disease','Prevalent Ischemic Heart Disease (Self-report)']),
]
SUPPLEMENT = {'glucose':['Glucose'],'total_cholesterol':['Total cholesterol','Serum total cholesterol','Cholesterol'], 'body_fat':['Body fat percentage'], 'ckd':['Incident Chronic Kidney Disease','Prevalent Chronic Kidney Disease (Estimated)'], 'stroke':['Incident Stroke','Prevalent Stroke (Self-report)'], 'whr':['Waist-hip ratio']}

def clean(v):
 if isinstance(v, dict): return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v, (list,tuple)): return [clean(x) for x in v]
 if isinstance(v, np.generic): return v.item()
 if isinstance(v,float) and np.isnan(v): return None
 return v

def main():
 if subprocess.run(['git','rev-parse','--verify','refs/tags/ewas-v3-main-panel-freeze-v1'],cwd=ROOT,capture_output=True).returncode == 0:
  raise SystemExit('Panel already frozen. A panel change requires a new prospective version and freeze tag.')
 out=ROOT/'data/registry/ewas_v3'; out.mkdir(parents=True,exist_ok=True)
 sets=ROOT/'data/derived/ewas_v3'; sets.mkdir(parents=True,exist_ok=True)
 u=d.load_universe(ROOT); assert len(u)==408399
 canonical=np.asarray(encode_many(u.chrom,u.pos),dtype=np.int64)
 probes=pd.read_parquet(ROOT/'data/processed/ComputAgeBench/illumina_probe_grch38.parquet')
 probes['canonical']=encode_many(probes['chr'],probes.pos)
 mapper=probes.set_index('probe_id').canonical
 candidates=pd.read_csv(ROOT/'data/registry/ewas_catalog_expansion/all_candidate_studies.csv')
 excluded=np.zeros(len(u),bool)
 for f in (ROOT/'data/derived/ewas_catalog_expansion/positive_sets').glob('*.npy'):
  excluded |= np.isin(canonical,np.load(f,allow_pickle=False))
 for ids in d.load_sets(ROOT).values(): excluded |= np.isin(canonical,ids)
 np.save(sets/'background_exclusion_mask.npy',excluded)
 chrom=u.chrom.to_numpy(); pt=u.probe_type.to_numpy()
 X=u[list(d.S1_COLUMNS)].to_numpy(float); known=pt>0
 Z=(X-X[known].mean(0))/X[known].std(0,ddof=0)
 downloads=json.loads((ROOT/'data/registry/ewas_catalog_expansion/zenodo_download_manifest.json').read_text())
 def positive(r,p):
  f=ROOT/'data/raw/ewas_catalog_expansion/zenodo_prepared/analysis_cache'/f'{r.source_analysis_id}.tsv.gz'
  x=pd.read_csv(f,sep='\t',header=None,names=['cpg','p','source_analysis_id'])
  x=x[x.p<p]; raw=x.cpg.nunique()
  ids=x.cpg.map(mapper).dropna().to_numpy(np.int64)
  rows=np.flatnonzero(np.isin(canonical,ids))
  return raw,rows,f
 audit=[]; selected=[]; supplementary=[]
 for category,task,labels in PANEL+[('Supplementary',k,v) for k,v in SUPPLEMENT.items()]:
  eligible=candidates[candidates.trait.isin(labels)].copy(); assert len(eligible),task
  ranked=[]
  for _,r in eligible.iterrows():
   raw,rows,cache=positive(r,1e-4)
   bg=d.technical_background(rows,chrom,pt,excluded)
   rec={'task':task,'category':category,'source_analysis_id':r.source_analysis_id,'trait':r.trait,'n_positive_raw':raw,'n_positive_universe':len(rows),'n_matchable':bg['n_matched_positives'],'complete_matching':bg['n_matched_positives']==len(rows),'N_study':float(r.n_subjects_numeric) if pd.notna(r.n_subjects_numeric) else None,'support_tier':'main_eligible' if len(rows)>=200 else 'exploratory' if len(rows)>=100 else 'insufficient_support'}
   key=(-int(len(rows)>=200),-len(rows),-int(rec['complete_matching']),-(rec['N_study'] or -1),r.source_analysis_id)
   ranked.append((key,r,rec,rows,cache)); audit.append(rec)
  _,r,rec,rows,cache=min(ranked,key=lambda t:t[0]); rec['selected']=True
  if category=='Supplementary': supplementary.append(rec);continue
  assert rec['support_tier']=='main_eligible',task
  container=str(r.data_container)
  source=next(v for v in downloads if '/records/'+str(int(r.zenodo_record_id))+'/' in v['url'] and Path(container).name in v['url'])
  rec.update(PMID=str(r.pmid),Catalog_StudyID=None,tissue=r.tissue,platform=r.methylation_array,zenodo_record=str(int(r.zenodo_record_id)),source_url=r.source_url,original_filename=r.member_name if pd.notna(r.member_name) else Path(container).name,container=container,source_sha256=source['sha256'],official_md5=source['expected_md5'],metadata_provenance=r.metadata_provenance,n_provenance=r.n_provenance,cache_sha256=d.sha256_file(cache),sets={})
  for label,p in [('primary',1e-4),('sensitivity',1e-7)]:
   raw,rr,_=positive(r,p)
   npy=sets/f'{task}_{label}.npy';np.save(npy,canonical[rr])
   details={'p_strictly_less_than':p,'n_positive_raw':raw,'n_positive_universe':len(rr),'support_tier':'main_eligible' if len(rr)>=200 else 'exploratory' if len(rr)>=100 else 'insufficient_support','positive_set':str(npy.relative_to(ROOT)),'positive_sha256':d.sha256_file(npy),'matching':{}}
   for name,matched in [('main',d.technical_background(rr,chrom,pt,excluded)),('genomic_context_s1',d.context_matched_background(rr,chrom,pt,u.context.to_numpy(),Z,excluded))]:
    pairs=pd.DataFrame([(int(u.cpg_idx.iloc[a]),int(u.cpg_idx.iloc[b[0]]),chrom[a],int(d.chrom_to_fold()[chrom[a]])) for a,b in matched['pairs']],columns=['positive_cpg_idx','background_cpg_idx','chromosome','outer_fold'])
    path=out/f'{task}_{label}_{name}_pairs.csv.gz';pairs.to_csv(path,index=False,compression={'method':'gzip','mtime':0})
    details['matching'][name]={k:v for k,v in matched.items() if k not in ['pairs','dist']}
    details['matching'][name].update(path=str(path.relative_to(ROOT)),sha256=d.sha256_file(path))
   rec['sets'][label]=details
  selected.append(rec)
  print(task,rec['n_positive_universe'],flush=True)
 chosen={r['source_analysis_id'] for r in selected+supplementary}
 for r in audit:r['selected']=r['source_analysis_id'] in chosen
 pd.DataFrame(audit).to_csv(out/'candidate_selection_audit.csv',index=False)
 pd.DataFrame([{k:v for k,v in r.items() if k!='sets'} for r in selected]).to_csv(out/'main_panel_summary.csv',index=False)
 pd.DataFrame(supplementary).to_csv(out/'supplementary_panel.csv',index=False)
 flags={}
 for label in ['primary','sensitivity']:
  ids={r['task']:set(np.load(ROOT/r['sets'][label]['positive_set']).tolist()) for r in selected}
  overlaps=[]; matrix=pd.DataFrame(index=list(ids),columns=list(ids),dtype=float)
  for a,A in ids.items():
   for b,B in ids.items():matrix.loc[a,b]=len(A&B)/min(len(A),len(B)) if min(len(A),len(B)) else np.nan
  for i,a in enumerate(ids):
   for b in list(ids)[i+1:]:
    A,B=ids[a],ids[b];v=matrix.loc[a,b];overlaps.append(dict(trait_a=a,trait_b=b,n_shared=len(A&B),n_smaller=min(len(A),len(B)),containment=v,flag_above_50_percent=bool(v>.5)))
  matrix.to_csv(out/f'{label}_redundancy_matrix.csv')
  frame=pd.DataFrame(overlaps);frame.to_csv(out/f'{label}_overlap_pairs.csv',index=False)
  frame[frame.flag_above_50_percent].to_csv(out/f'{label}_overlap_flags.csv',index=False);flags[label]=int(frame.flag_above_50_percent.sum())
 pd.DataFrame([{'task':r['task'],'threshold':label,'matching_design':name,'n_positive_universe':s['n_positive_universe'],**m} for r in selected for label,s in r['sets'].items() for name,m in s['matching'].items()]).to_csv(out/'matching_feasibility.csv',index=False)
 pd.DataFrame(d.draw_table(d.CHROMS),columns=d.CHROMS).to_csv(out/'chromosome_bootstrap_multiplicities.csv',index=False)
 spec=yaml.safe_load((ROOT/'configs/ewas_v2/ewas_v2_spec_v1.yaml').read_text())
 spec['probe']['scaler']='StandardScaler fitted only on the training rows of each inner or outer fold; never on validation/test rows'
 manifest={'schema_version':1,'status':'FROZEN_AWAITING_USER_APPROVAL_NO_REPRESENTATIONS_RUN','freeze_tag':'ewas-v3-main-panel-freeze-v1','source':'Official EWAS Catalog Zenodo fallback','complete_current_catalog_coverage':False,'catalog_study_ids_available':False,'selection_uses_representation_results':False,'selection_rule':['main_eligible','descending intersected positive count','complete matching','descending N_study','ascending source_analysis_id'],'support_tiers':{'main_eligible':'>=200 intersected positives','exploratory':'100–199','insufficient_support':'<100'},'task_counts':{'Cancer':5,'Immune/inflammatory':4,'Cardiometabolic':4,'total':13},'terms':{'primary':'EWAS Catalog-associated CpGs; p < 1e-4','sensitivity':'high-confidence EWAS-associated CpGs; p < 1e-7'},'universe_size':408399,'matching':{'main':'Exact chromosome x Infinium type; 1:1 without replacement','genomic_context_s1':'EWAS-v2 exact chromosome x Infinium type x island class; nearest neighbor standardized GC, CpG density hg38, stored log10 TSS distance; Euclidean caliper 0.25; ddof=0 over probe-known universe','seed':17,'pair_cap':None,'background_exclusion':'Union of all 351 inventory primary p<1e-4 sets and the 16 frozen EWAS-v2 sets; fixed at both p thresholds','mask_sha256':d.sha256_file(sets/'background_exclusion_mask.npy'),'same_pairs_for_all_representations':True},'evaluation_contract_from_ewas_v2':{k:spec[k] for k in ['folds','probe','zero_rows','uncertainty']},'evaluation_addenda':{'matched_pair_concordance':'P(OOF score positive > OOF score matched background); strict ties count zero','category_average_auroc':False,'category_win_counts':False,'AUROC':'pooled out-of-fold primary','AUPRC':'secondary descriptive; artificial 50% prevalence','bootstrap':'paired chromosome blocks; 1000 draws; seed 17; percentile 95% CI; identical draws all arms and contrasts','contrasts':['Regulatory - CpGPT','Regulatory - DeepCpG','CpGPT - DeepCpG'],'sensitivity_panel':'Same source analyses; no reselection/drop based on strict-threshold support'},'representations':['regulatory_histone_dnase_v1','cpgpt_large_locus native 512D','deepcpg_dna_locus native 128D'],'main_tasks':selected,'prespecified_supplementary_tasks':supplementary,'inventory_only_families':['environmental/behavioral','psychiatric'],'other_cardiometabolic_inventory_traits':'Inventory only; any supplementary panel extension requires a new prospective freeze before evaluation','overlap_flags':flags,'provenance_sha256':{p:d.sha256_file(ROOT/p) for p in ['outputs/encode_atlas_v1/loci.parquet','data/derived/bioval_v2/covariates/universe_covariates.parquet','data/processed/ComputAgeBench/illumina_probe_grch38.parquet','data/registry/ewas_catalog_expansion/all_candidate_studies.csv','data/registry/ewas_catalog_expansion/zenodo_analysis_metadata.csv','data/registry/ewas_catalog_expansion/zenodo_download_manifest.json','data/registry/ewas_catalog_expansion/bulk_retry_manifest.json','scripts/ewas_v3/check_freeze.py','configs/ewas_v2/ewas_v2_spec_v1.yaml','src/cpg_repr_benchmark/ewas_v2/design.py','scripts/ewas_v3/build_main_panel.py']},'artifact_sha256':{str(p.relative_to(ROOT)):d.sha256_file(p) for p in sorted(out.glob('*')) if p.is_file()},'future_result_table_columns':['category','trait','source_analysis_id','PMID','tissue','N_study','n_positive_universe','n_matched','representation','AUROC','ci_lo','ci_hi','AUPRC','matched_pair_concordance'],'execution_gate':'Manifest and artifact hashes must match a committed freeze tag AND explicit user approval must be recorded before any representation execution.'}
 (ROOT/'configs/ewas_v3/main_trait_panel.yaml').write_text(yaml.safe_dump(clean(manifest),sort_keys=False,allow_unicode=True))
 print(json.dumps({'task_counts':manifest['task_counts'],'overlap_flags':flags}),flush=True)
if __name__=='__main__':main()
