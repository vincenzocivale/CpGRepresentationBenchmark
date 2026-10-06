#!/usr/bin/env python3
"""Execute the approved prospective panel. Read frozen pairs; never select tasks by performance."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, importlib.util, json, os, subprocess, sys, time, warnings
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import yaml
from cpg_repr_benchmark.bioval_v2_launch.embedding_eval import ArmSpec,EmbeddingStore
from cpg_repr_benchmark.bioval_v2_launch.block_bootstrap import BlockBootstrap,RankPlan,block_index,weighted_auroc,weighted_mean,percentile_ci,contrast
from cpg_repr_benchmark.biological_validation_v2.metrics import auroc_binary,average_precision
TAG='ewas-v3-main-panel-freeze-v1'
OUT=ROOT/'outputs/ewas_v3'/TAG
ARMS=('regulatory_histone_dnase_v1','cpgpt_large_locus','deepcpg_dna_locus')
CONTRASTS=(('Regulatory - CpGPT',ARMS[0],ARMS[1]),('Regulatory - DeepCpG',ARMS[0],ARMS[2]),('CpGPT - DeepCpG',ARMS[1],ARMS[2]))
CACHE={}
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  while b:=f.read(1<<22):h.update(b)
 return h.hexdigest()
def write_json(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def endpoint_dir(task,threshold,design,arm):return OUT/threshold/design/'arms'/arm/task

def preflight():
 subprocess.run([sys.executable,str(ROOT/'scripts/ewas_v3/check_freeze.py')],cwd=ROOT,check=True)
 m=yaml.safe_load((ROOT/'configs/ewas_v3/main_trait_panel.yaml').read_text())
 v2=yaml.safe_load((ROOT/'configs/ewas_v2/ewas_v2_spec_v1.yaml').read_text())
 universe=pd.read_parquet(ROOT/'outputs/encode_atlas_v1/loci.parquet',columns=['cpg_idx']).cpg_idx.to_numpy()
 specs={a:dict(v2['arms'][a]) for a in ARMS}
 identities={}
 for a,s in specs.items():
  store=EmbeddingStore(ArmSpec(a,a,s['store'],s['sha256'],s['dim'],s['dtype']),ROOT)
  store.rows_for(universe)
  identities[a]=store.identity()
  print(f'PREFLIGHT {a}: SHA256 verified; {store.dim}D; all 408399 loci covered',flush=True)
 code_files=['scripts/ewas_v3/run_experiments.py','scripts/ewas_v3/check_freeze.py','src/cpg_repr_benchmark/bioval_v2_launch/embedding_eval.py','src/cpg_repr_benchmark/bioval_v2_launch/block_bootstrap.py','src/cpg_repr_benchmark/biological_validation_v2/metrics.py']
 for p in code_files:
  # Record exact implementation bytes, including inherited user worktree changes.
  if p=='scripts/ewas_v3/run_experiments.py' and subprocess.check_output(['git','show','HEAD:'+p],cwd=ROOT)!=(ROOT/p).read_bytes():raise RuntimeError('Commit runner before model execution')
 draws=pd.read_csv(ROOT/'data/registry/ewas_v3/chromosome_bootstrap_multiplicities.csv').to_numpy(np.int64)
 bb=BlockBootstrap(np.array(sorted(f'chr{i}' for i in range(1,23))),1000,17)
 if not np.array_equal(draws,bb.draws):raise RuntimeError('Frozen bootstrap draw indices mismatch')
 run={'freeze_tag':TAG,'freeze_commit':subprocess.check_output(['git','rev-parse',TAG],cwd=ROOT,text=True).strip(),'implementation_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'manifest_sha256':sha(ROOT/'configs/ewas_v3/main_trait_panel.yaml'),'approval':json.loads((ROOT/'configs/ewas_v3/user_approval.json').read_text()),'code_sha256':{p:sha(ROOT/p) for p in code_files},'stores':identities,'bootstrap_fingerprint':bb.fingerprint(),'bootstrap_artifact_interpretation':'Frozen CSV stores draw indices (despite historical multiplicities filename); reconstruct block multiplicities using sorted chromosome names, per frozen contract.','workers':2,'threads_per_worker':4,'design_changes':False,'status':'RUNNING','started_utc':pd.Timestamp.now(tz='UTC').isoformat()}
 previous=OUT/'execution_manifest.json'
 if previous.exists():
  old=json.loads(previous.read_text())
  if any(old.get(k)!=run[k] for k in ['manifest_sha256','code_sha256','stores','bootstrap_fingerprint']):raise RuntimeError('Cannot resume endpoints under changed inputs/implementation')
  run['started_utc']=old['started_utc']
 write_json(previous,run)
 return m,specs

def load_embedding(arm,spec):
 if arm not in CACHE:
  import h5py
  store=EmbeddingStore(ArmSpec(arm,arm,spec['store'],spec['sha256'],spec['dim'],spec['dtype']),ROOT,verify_sha=False)
  with h5py.File(store.path,'r') as f:E=np.asarray(f[store.emb_key][:],dtype=np.float32)
  if not np.isfinite(E).all():raise RuntimeError('Non-finite store entries')
  CACHE[arm]=(store,E)
 return CACHE[arm]

def run_endpoint(job):
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import LogisticRegression
 from sklearn.exceptions import ConvergenceWarning
 from threadpoolctl import threadpool_limits
 task,threshold,design,arm,pairfile,spec,contract=job
 path=endpoint_dir(task,threshold,design,arm);path.mkdir(parents=True,exist_ok=True)
 digest=sha(ROOT/pairfile); done=path/'endpoint.json'
 if done.exists():
  e=json.loads(done.read_text())
  if e['pairs_sha256']!=digest:raise RuntimeError('Existing endpoint uses different pairs')
  if e['status']=='complete' and sha(path/'oof.parquet')!=e['oof_sha256']:raise RuntimeError('OOF checkpoint corrupted')
  return dict(task=task,threshold=threshold,design=design,arm=arm,status='resumed_'+e['status'])
 pairs=pd.read_csv(ROOT/pairfile)
 n=len(pairs);base=dict(task=task,threshold=threshold,design=design,arm=arm,n_matched=n,pairs_sha256=digest)
 if set(pairs.outer_fold.unique())!=set(range(5)):
  base.update(status='not_evaluable',reason='Frozen five-fold chromosome CV not feasible: one or more folds contain no matched pairs',represented_outer_folds=sorted(map(int,pairs.outer_fold.unique())))
  write_json(done,base);return base
 store,E=load_embedding(arm,spec)
 ids=np.r_[pairs.positive_cpg_idx,pairs.background_cpg_idx].astype(np.int64)
 X=E[store.rows_for(ids)]
 y=np.r_[np.ones(n,int),np.zeros(n,int)];fold=np.tile(pairs.outer_fold.to_numpy(int),2)
 chrom=np.tile(pairs.chromosome.to_numpy(str),2)
 Cs=contract['C_grid'];max_iter=contract['max_iter']
 oof=np.full(2*n,np.nan);chosen=[];stats=[];t0=time.time()
 with threadpool_limits(limits=4):
  for f in range(5):
   cp=path/f'outer_fold_{f}.npz'
   te=np.flatnonzero(fold==f);tr=np.flatnonzero(fold!=f)
   if cp.exists():
    q=np.load(cp,allow_pickle=False)
    if str(q['pairs_sha256'])!=digest or not np.array_equal(q['test_rows'],te):raise RuntimeError('Invalid outer fold checkpoint')
    oof[te]=q['scores'];chosen.append(float(q['C']));stats.append(dict(n_iter_max=int(q['n_iter_max']),n_nonconverged_fits=int(q['n_nonconverged_fits'])));continue
   nc=0;nit=0;inner_results=[]
   def fit_predict(train_X,train_y,test_X,C):
    nonlocal nc,nit
    model=LogisticRegression(penalty='l2',solver='lbfgs',C=C,max_iter=max_iter,class_weight=None,random_state=17)
    with warnings.catch_warnings(record=True) as caught:
     warnings.simplefilter('always',ConvergenceWarning);model.fit(train_X,train_y)
    nc+=int(any(issubclass(w.category,ConvergenceWarning) for w in caught));nit=max(nit,int(model.n_iter_.max()))
    return model.decision_function(test_X)
   # StandardScaler fitted separately to EACH inner training partition.
   inner=[]
   for g in range(5):
    if g==f:continue
    it=np.flatnonzero((fold!=f)&(fold!=g));iv=np.flatnonzero(fold==g)
    scaler=StandardScaler();xa=scaler.fit_transform(X[it].astype(np.float64));xv=scaler.transform(X[iv].astype(np.float64))
    inner.append((xa,y[it],xv,y[iv]))
   for C in Cs:
    aucs=[auroc_binary(yv,fit_predict(xa,ya,xv,C)) for xa,ya,xv,yv in inner]
    inner_results.append(float(np.mean(aucs)))
    write_json(path/'progress.json',dict(outer_fold=f,completed_inner_C=C,elapsed_seconds=round(time.time()-t0,1)))
   del inner
   best=min(C for C,s in zip(Cs,inner_results) if s==max(inner_results));chosen.append(best)
   scaler=StandardScaler();xa=scaler.fit_transform(X[tr].astype(np.float64));xt=scaler.transform(X[te].astype(np.float64))
   pred=fit_predict(xa,y[tr],xt,best);oof[te]=pred;stats.append(dict(n_iter_max=nit,n_nonconverged_fits=nc))
   np.savez(cp,test_rows=te,scores=pred,C=best,n_iter_max=nit,n_nonconverged_fits=nc,pairs_sha256=digest)
   write_json(path/'progress.json',dict(outer_fold_completed=f,elapsed_seconds=round(time.time()-t0,1)))
 if not np.isfinite(oof).all():raise RuntimeError('Non-finite OOF predictions')
 frame=pd.DataFrame(dict(pair_id=np.tile(np.arange(n),2),cpg_idx=ids,y=y,fold=fold,chrom=chrom,oof_score=oof))
 frame.to_parquet(path/'oof.parquet',index=False)
 base.update(status='complete',C_per_outer_fold=chosen,n_iter_max=max(s['n_iter_max'] for s in stats),n_nonconverged_fits=sum(s['n_nonconverged_fits'] for s in stats),AUROC=auroc_binary(y,oof),AUPRC=average_precision(y,oof),matched_pair_concordance=float(np.mean(oof[:n]>oof[n:])),zero_rows_positive=int((X[:n]==0).all(1).sum()),zero_rows_background=int((X[n:]==0).all(1).sum()),oof_sha256=sha(path/'oof.parquet'),seconds=round(time.time()-t0,1))
 write_json(done,base)
 return {k:base[k] for k in ['task','threshold','design','arm','status','seconds','n_nonconverged_fits']}

def report(m):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 bb=BlockBootstrap(np.array(sorted(f'chr{i}' for i in range(1,23))),1000,17)
 rows=[];contrasts=[]
 for threshold in ['primary','sensitivity']:
  for design in ['main','genomic_context_s1']:
   for task in m['main_tasks']:
    name=task['task'];scores={};rep={};eps={};ref=None
    for arm in ARMS:
     path=endpoint_dir(name,threshold,design,arm);e=json.loads((path/'endpoint.json').read_text());eps[arm]=e
     s=task['sets'][threshold]
     r=dict(category=task['category'],trait=task['trait'],task=name,source_analysis_id=task['source_analysis_id'],PMID=task['PMID'],tissue=task['tissue'],N_study=task['N_study'],n_positive_universe=s['n_positive_universe'],n_matched=e['n_matched'],representation=arm,threshold=threshold,matching_design=design,support_tier=s['support_tier'],status=e['status'],AUROC=np.nan,ci_lo=np.nan,ci_hi=np.nan,AUPRC=np.nan,matched_pair_concordance=np.nan,n_nonconverged_fits=e.get('n_nonconverged_fits',0),reason=e.get('reason',''))
     if e['status']=='complete':
      df=pd.read_parquet(path/'oof.parquet');key=df.drop(columns=['oof_score'])
      if ref is not None and not ref.equals(key):raise RuntimeError('Shared-pairs violation across representations')
      ref=key;n=len(df)//2;v=df.oof_score.to_numpy();scores[arm]=v
      blk=block_index(df.chrom.to_numpy(),bb.names);pairblk=blk[:n];y=df.y.to_numpy().astype(bool)
      plan=RankPlan(v);rep[arm]=bb.replicates(lambda w:weighted_auroc(plan,y,w),blk)
      ci=percentile_ci(rep[arm]);con=(v[:n]>v[n:]).astype(float)
      cci=percentile_ci(bb.replicates(lambda w:weighted_mean(con,w),pairblk))
      r.update(AUROC=e['AUROC'],ci_lo=ci[0],ci_hi=ci[1],AUPRC=e['AUPRC'],matched_pair_concordance=e['matched_pair_concordance'],concordance_ci_lo=cci[0],concordance_ci_hi=cci[1])
     rows.append(r)
    for label,a,b in CONTRASTS:
     r=dict(category=task['category'],trait=task['trait'],task=name,threshold=threshold,matching_design=design,contrast=label,metric='AUROC',status='not_evaluable')
     if a in rep and b in rep:
      if eps[a]['n_nonconverged_fits'] or eps[b]['n_nonconverged_fits']:raise RuntimeError('Non-converged fits: paired contrasts refused; requires prospective uniform remedy')
      r.update(contrast(eps[a]['AUROC'],rep[a],eps[b]['AUROC'],rep[b]),status='complete')
     contrasts.append(r)
 allrows=pd.DataFrame(rows);allcontrasts=pd.DataFrame(contrasts)
 allrows.to_csv(OUT/'all_results.csv',index=False);allcontrasts.to_csv(OUT/'all_paired_contrasts.csv',index=False)
 main=allrows[(allrows.threshold=='primary')&(allrows.matching_design=='main')]
 main.to_csv(OUT/'main_results.csv',index=False)
 allcontrasts[(allcontrasts.threshold=='primary')&(allcontrasts.matching_design=='main')].to_csv(OUT/'main_paired_contrasts.csv',index=False)
 flags=pd.read_csv(ROOT/'data/registry/ewas_v3/primary_overlap_flags.csv');correlated=set(flags.trait_a)|set(flags.trait_b)
 fig,axes=plt.subplots(3,1,figsize=(11,10),gridspec_kw={'height_ratios':[5,4,4]},layout='constrained')
 colors=['#0072B2','#D55E00','#009E73'];display=['Regulatory','CpGPT 512D','DeepCpG 128D']
 for ax,category in zip(axes,['Cancer','Immune/inflammatory','Cardiometabolic']):
  tasks=[t for t in m['main_tasks'] if t['category']==category]
  for j,(arm,color,label) in enumerate(zip(ARMS,colors,display)):
   s=main[(main.category==category)&(main.representation==arm)].set_index('task').loc[[t['task'] for t in tasks]]
   yy=np.arange(len(tasks))+(j-1)*.18
   ax.errorbar(s.AUROC,yy,xerr=np.vstack([s.AUROC-s.ci_lo,s.ci_hi-s.AUROC]),fmt='o',capsize=2,color=color,label=label,markersize=4)
  ax.set_yticks(np.arange(len(tasks)),[t['trait']+(' †' if t['task'] in correlated else '') for t in tasks]);ax.invert_yaxis();ax.set_xlim(.3,1);ax.axvline(.5,color='gray',ls='--',lw=.8);ax.set_title(category,loc='left');ax.grid(axis='x',alpha=.2)
 axes[0].legend(loc='lower right',ncol=3);axes[-1].set_xlabel('Pooled out-of-fold AUROC (95% paired chromosome-bootstrap CI)')
 fig.suptitle('EWAS-associated CpGs: p < 1e-4\n† Positive sets overlap >50% with another selected task',fontsize=12)
 fig.savefig(OUT/'main_auroc.pdf');fig.savefig(OUT/'main_auroc.png',dpi=200);plt.close(fig)
 brief=['# EWAS-v3 results','', 'Panel unchanged from approved freeze. Primary p < 1e-4; sensitivity p < 1e-7. Three representations, identical frozen matching pairs, nested chromosome CV, scalers fitted inside every training fold.','', f'Main results: {len(main)} rows; {int((main.status=="complete").sum())} evaluated. No category averages or win counts.','', 'Main figure annotates correlated traits (†); pairs above 50% overlap cannot be interpreted as independent replications. AUPRC is descriptive because matching imposes 50% prevalence. Concordance counts strict positive > background, ties zero.','', 'Insufficient-support sensitivity tasks remain present; endpoints lacking all five frozen folds are explicitly not_evaluable. No task or source substitution. Source: official EWAS Catalog Zenodo fallback, incomplete coverage of the full current Catalog; actual Catalog StudyIDs unavailable.','', '## Primary results','', main[['category','trait','representation','AUROC','ci_lo','ci_hi','AUPRC','matched_pair_concordance']].to_csv(index=False), '']
 (OUT/'report.md').write_text('\n'.join(brief))
 em=json.loads((OUT/'execution_manifest.json').read_text());em.update(status='COMPLETE',completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),n_endpoint_rows=len(allrows),n_primary_rows=len(main),n_not_evaluable=int((allrows.status!='complete').sum()),output_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.glob('*') if p.is_file() and p.name!='execution_manifest.json'})
 write_json(OUT/'execution_manifest.json',em)
 print(f'COMPLETE: {len(main)} primary rows; {len(allrows)} total rows; {em["n_not_evaluable"]} not evaluable',flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
 m,specs=preflight()
 if not args.report_only:
  jobs=[(t['task'],threshold,design,arm,t['sets'][threshold]['matching'][design]['path'],specs[arm],m['evaluation_contract_from_ewas_v2']['probe']) for threshold in ['primary','sensitivity'] for design in ['main','genomic_context_s1'] for t in m['main_tasks'] for arm in ARMS]
  import multiprocessing
  with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
   futures={pool.submit(run_endpoint,j):j for j in jobs}
   for f in concurrent.futures.as_completed(futures):
    r=f.result();print(json.dumps(r),flush=True)
 report(m)
if __name__=='__main__':main()
